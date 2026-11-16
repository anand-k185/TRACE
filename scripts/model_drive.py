#!/usr/bin/env python3
"""
TRACE - let the trained model drive the car in Gazebo, from the camera only.

Every camera frame: image -> ViT top-down model -> 6 waypoints (meters, car frame) ->
pure pursuit steering -> /drive. No map, no odometry, no lidar is used for driving.
The only fixed inputs are the camera's own calibration (camera_info + its mount from TF).

Also publishes, for watching it:
  /trace/overlay  (sensor_msgs/Image)  camera image with the predicted path drawn on it
  /trace/path     (nav_msgs/Path)      predicted waypoints in the car frame (RViz)

Needs trace_bev.py (same folder) and the checkpoint trained in Colab.

  python3 model_drive.py --ckpt trace_vit_bev.pt
  python3 model_drive.py --ckpt trace_vit_bev.pt --speed 0.6 --duration 300
"""
import argparse, math, os, sys, time
import numpy as np
from PIL import Image as PILImage, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

IPM_BUFFERS = ("gq", "vq", "gf", "vf")      # camera-geometry tables: rebuilt from the live camera


class Driver:
    """Model + controller, no ROS (so it can be tested on its own)."""

    def __init__(self, meta, ckpt, speed=0.8, lookahead=1.4, wheelbase=0.33, max_steer=0.40,
                 backbone="vit_small_patch16_224.augreg_in21k_ft_in1k", q_hw=(20, 32), threads=4,
                 steer_gain=1.0, smooth=0.0, curve_slow=0.45):
        import torch
        import trace_bev as tb
        self.torch, self.tb = torch, tb
        torch.set_num_threads(threads)
        self.meta = meta
        self.model = tb.ViTBEV(meta, n_wp=6, pretrained=False, backbone=backbone, q_hw=q_hw)
        sd = torch.load(ckpt, map_location="cpu")
        sd = {k: v for k, v in sd.items() if k not in IPM_BUFFERS}
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        assert set(missing) <= set(IPM_BUFFERS) and not unexpected, (missing, unexpected)
        self.model.eval()
        self.speed, self.ld, self.L, self.ms = speed, lookahead, wheelbase, max_steer
        self.gain, self.smooth, self.curve_slow = steer_gain, smooth, curve_slow
        self.prev_steer = 0.0
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def predict(self, rgb):
        """rgb: HxWx3 uint8 camera image -> waypoints (6, 2) meters [x fwd, y left], confidence."""
        torch, tb = self.torch, self.tb
        im = np.asarray(PILImage.fromarray(rgb).resize((tb.IMG, tb.IMG), PILImage.BILINEAR), dtype=np.float32) / 255.0
        x = torch.from_numpy(((im - self.mean) / self.std).transpose(2, 0, 1).copy())[None]
        with torch.no_grad():
            logits = self.model(x).float()
            wp = tb.decode(logits)[0].numpy()
            conf = float(logits.flatten(2).softmax(-1).amax(-1).mean())
        return wp, conf

    def control(self, wp):
        """Pure pursuit toward the point `lookahead` meters along the predicted path."""
        path = np.vstack([[0.0, 0.0], wp])
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
        cum = np.concatenate([[0], np.cumsum(seg)])
        d = min(self.ld, cum[-1])
        tx, ty = np.interp(d, cum, path[:, 0]), np.interp(d, cum, path[:, 1])
        steer = self.gain * math.atan2(2 * self.L * ty, tx * tx + ty * ty)      # gain < 1 softens oversteer
        steer = self.smooth * self.prev_steer + (1 - self.smooth) * steer        # smoothing between frames
        steer = float(np.clip(steer, -self.ms, self.ms))
        self.prev_steer = steer
        speed = self.speed * (1.0 - self.curve_slow * abs(steer) / self.ms)    # slow down more in bends
        return steer, speed

    def overlay(self, rgb, wp, steer, conf, ms):
        im = PILImage.fromarray(rgb).convert("RGB")
        W, H = im.size
        uv, vis = self.tb.car_to_image224(wp, self.meta)
        uv = uv * [W / self.tb.IMG, H / self.tb.IMG]
        d = ImageDraw.Draw(im)
        pts = [tuple(p) for p, v in zip(uv, vis) if v]
        if len(pts) > 1:
            d.line(pts, fill=(255, 40, 40), width=3)
        for p in pts:
            d.ellipse([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4], outline=(255, 255, 0), width=2)
        n_off = int((~vis).sum())
        d.text((6, 6), f"steer {math.degrees(steer):+.0f} deg  conf {conf:.2f}  {ms:.0f} ms", fill=(255, 255, 0))
        if n_off:
            d.text((6, 20), f"{n_off} waypoint(s) off-screen (on the bend)", fill=(255, 200, 0))
        return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="trace_vit_bev.pt from Colab")
    ap.add_argument("--speed", type=float, default=0.8)
    ap.add_argument("--lookahead", type=float, default=1.4, help="pure-pursuit lookahead along the predicted path (m)")
    ap.add_argument("--duration", type=float, default=0, help="stop after N sim-seconds (0 = until Ctrl-C)")
    ap.add_argument("--steer-gain", type=float, default=1.0, help="<1 softens steering (counters oversteer)")
    ap.add_argument("--smooth", type=float, default=0.0, help="0-0.9: blend with previous steering (damps jerks)")
    ap.add_argument("--curve-slow", type=float, default=0.45, help="0-0.8: how much to slow down when steering hard")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads for the model")
    ap.add_argument("--image-topic", default="/camera/color/image_raw")
    ap.add_argument("--info-topic", default="/camera/color/camera_info")
    ap.add_argument("--base-frame", default="ego_racecar/base_link")
    ap.add_argument("--cam-frame", default="ego_racecar/camera_optical_link")
    ap.add_argument("--backbone", default="vit_small_patch16_224.augreg_in21k_ft_in1k")
    ap.add_argument("--q-hw", default="20,32")
    a = ap.parse_args()

    import rclpy
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import Image, CameraInfo
    from nav_msgs.msg import Path
    from geometry_msgs.msg import PoseStamped
    from ackermann_msgs.msg import AckermannDriveStamped
    import tf2_ros

    def quat_to_mat(x, y, z, w):
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

    class ModelDrive(Node):
        def __init__(self):
            super().__init__("trace_model_drive", parameter_overrides=[
                Parameter("use_sim_time", Parameter.Type.BOOL, True)])
            self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)
            self.ov_pub = self.create_publisher(Image, "/trace/overlay", 2)
            self.path_pub = self.create_publisher(Path, "/trace/path", 2)
            self.tf_buf = tf2_ros.Buffer()
            self.tf_l = tf2_ros.TransformListener(self.tf_buf, self)
            self.create_subscription(CameraInfo, a.info_topic, self.on_info, qos_profile_sensor_data)
            self.create_subscription(Image, a.image_topic, self.on_image, qos_profile_sensor_data)
            self.info, self.driver, self.latest = None, None, None
            self.cmd = (0.0, 0.0)
            self.last_pred_wall = 0.0
            self.t0 = None
            self.n, self.ms_sum, self.last_log = 0, 0.0, time.time()
            self.create_timer(0.05, self.tick)          # 20 Hz: run model on newest frame + send /drive
            self.get_logger().info("waiting for camera_info + TF ...")

        def on_info(self, m):
            self.info = m

        def on_image(self, m):
            self.latest = m

        def build(self):
            try:
                tf = self.tf_buf.lookup_transform(a.base_frame, a.cam_frame, Time())
            except Exception:
                return False
            t, q = tf.transform.translation, tf.transform.rotation
            T = np.eye(4)
            T[:3, :3] = quat_to_mat(q.x, q.y, q.z, q.w)
            T[:3, 3] = [t.x, t.y, t.z]
            meta = {"K": np.array(self.info.k, dtype=float).reshape(3, 3).tolist(), "T_base_cam": T.tolist(),
                    "width": self.info.width, "height": self.info.height}
            t0 = time.time()
            self.driver = Driver(meta, a.ckpt, speed=a.speed, lookahead=a.lookahead, threads=a.threads,
                                 backbone=a.backbone, q_hw=tuple(int(v) for v in a.q_hw.split(",")),
                                 steer_gain=a.steer_gain, smooth=a.smooth, curve_slow=a.curve_slow)
            self.get_logger().info(f"model loaded in {time.time() - t0:.1f}s "
                                   f"(camera {self.info.width}x{self.info.height}). driving.")
            return True

        def tick(self):
            if self.driver is None:
                if self.info is None or not self.build():
                    return
            now = self.get_clock().now().nanoseconds * 1e-9
            self.t0 = now if self.t0 is None else self.t0
            if a.duration and now - self.t0 > a.duration:
                self.get_logger().info("duration reached, stopping")
                self.send(0.0, 0.0)
                raise SystemExit
            m, self.latest = self.latest, None
            if m is not None:
                arr = np.frombuffer(m.data, dtype=np.uint8).reshape(m.height, m.step)[:, : m.width * 3]
                arr = arr.reshape(m.height, m.width, 3)
                if m.encoding.lower().startswith("bgr"):
                    arr = arr[:, :, ::-1]
                arr = np.ascontiguousarray(arr)
                t1 = time.time()
                wp, conf = self.driver.predict(arr)
                ms = (time.time() - t1) * 1000
                self.cmd = self.driver.control(wp)
                self.last_pred_wall = time.time()
                self.publish_viz(m, arr, wp, conf, ms)
                self.n += 1
                self.ms_sum += ms
                if time.time() - self.last_log > 5:
                    self.get_logger().info(f"{self.n} frames | model {self.ms_sum / self.n:.0f} ms/frame | "
                                           f"steer {math.degrees(self.cmd[0]):+.0f} deg | conf {conf:.2f}")
                    self.last_log = time.time()
            if time.time() - self.last_pred_wall > 1.5:      # no fresh prediction -> stop safely
                self.send(0.0, 0.0)
            else:
                self.send(*self.cmd)

        def publish_viz(self, m, arr, wp, conf, ms):
            ov = self.driver.overlay(arr, wp, self.cmd[0], conf, ms)
            img = Image()
            img.header = m.header
            img.height, img.width = ov.shape[:2]
            img.encoding, img.step = "rgb8", ov.shape[1] * 3
            img.data = ov.tobytes()
            self.ov_pub.publish(img)
            path = Path()
            path.header.stamp = m.header.stamp
            path.header.frame_id = a.base_frame
            for x, y in wp:
                ps = PoseStamped()
                ps.header = path.header
                ps.pose.position.x, ps.pose.position.y = float(x), float(y)
                ps.pose.orientation.w = 1.0
                path.poses.append(ps)
            self.path_pub.publish(path)

        def send(self, steer, speed):
            msg = AckermannDriveStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.drive.steering_angle = float(steer)
            msg.drive.speed = float(speed)
            self.drive_pub.publish(msg)

    rclpy.init()
    node = ModelDrive()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        try:
            for _ in range(5):
                node.send(0.0, 0.0)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
