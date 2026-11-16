#!/usr/bin/env python3
"""
TRACE - data recorder for f1tenth_gym_gazebo (Gazebo Classic 11 + ROS 2).

Run it while the sim is up and you drive with teleop_keyboard. It saves:
  - every RGB frame from /camera/color/image_raw  (jpg)
  - the car's ground-truth pose at that instant from /ego_racecar/odom
    (in this sim odom IS ground truth: dynamics_plant integrates the car and
     writes that pose into Gazebo, and odom == map == Gazebo world frame)
  - the last /drive command (steer, speed)
  - once per episode: camera intrinsics K (from camera_info) and the fixed
    base_link -> camera_optical_link transform (from TF)

Each run = one episode folder. Ctrl-C to stop and close the episode.
Labels are made afterwards with gz_make_labels.py.

  python3 gz_collect.py --map lab_00 --out data/
"""
import argparse, json, math, os, time
from collections import deque

import numpy as np
from PIL import Image as PILImage

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool
import tf2_ros


def quat_to_mat(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def stamp_s(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class Recorder(Node):
    def __init__(self, a):
        super().__init__("trace_recorder", parameter_overrides=[
            Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        self.a = a
        self.ep_dir = os.path.join(a.out, a.map, time.strftime("ep_%Y%m%d_%H%M%S"))
        os.makedirs(os.path.join(self.ep_dir, "rgb"), exist_ok=True)
        self.log = open(os.path.join(self.ep_dir, "frames.jsonl"), "w")

        self.odom = deque(maxlen=500)          # (t, x, y, z, yaw, speed)
        self.drive = {"steer": 0.0, "speed": 0.0}
        self.noise_now = False
        self.noise_since_save = False   # any autopilot drift since the last saved frame
        self.K = None
        self.T_base_cam = None
        self.meta_written = False
        self.i = 0
        self.last_save_t = -1e9

        self.tf_buf = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf, self)

        self.create_subscription(Odometry, a.odom_topic, self.on_odom,
                                 QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.create_subscription(CameraInfo, a.info_topic, self.on_info, qos_profile_sensor_data)
        self.create_subscription(Image, a.image_topic, self.on_image, qos_profile_sensor_data)
        self.create_subscription(AckermannDriveStamped, a.drive_topic, self.on_drive, 10)
        self.create_subscription(Bool, "/trace/noise", self.on_noise, 10)  # from autodrive.py (optional)
        self.get_logger().info(f"recording -> {self.ep_dir}  (Ctrl-C to stop)")

    # ---------- callbacks ----------
    def on_odom(self, m):
        q = m.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        v = m.twist.twist.linear
        p = m.pose.pose.position
        self.odom.append((stamp_s(m.header.stamp), p.x, p.y, p.z, yaw, math.hypot(v.x, v.y)))

    def on_noise(self, m):
        self.noise_now = m.data
        self.noise_since_save |= m.data

    def on_drive(self, m):
        self.drive = {"steer": m.drive.steering_angle, "speed": m.drive.speed}

    def on_info(self, m):
        if self.K is None:
            self.K = np.array(m.k, dtype=float).reshape(3, 3)
            self.W, self.H = m.width, m.height

    def try_tf(self):
        try:
            tf = self.tf_buf.lookup_transform(self.a.base_frame, self.a.cam_frame, Time())
        except Exception:
            return False
        t, q = tf.transform.translation, tf.transform.rotation
        T = np.eye(4)
        T[:3, :3] = quat_to_mat(q.x, q.y, q.z, q.w)
        T[:3, 3] = [t.x, t.y, t.z]
        self.T_base_cam = T  # camera_optical pose expressed in base_link
        return True

    def on_image(self, m):
        if self.K is None or (self.T_base_cam is None and not self.try_tf()):
            return  # wait for camera_info + TF
        if not self.meta_written:
            meta = {"map": self.a.map, "hz": self.a.hz, "width": self.W, "height": self.H,
                    "K": self.K.tolist(), "T_base_cam": self.T_base_cam.tolist(),
                    "base_frame": self.a.base_frame, "cam_frame": self.a.cam_frame}
            json.dump(meta, open(os.path.join(self.ep_dir, "meta.json"), "w"), indent=2)
            self.meta_written = True

        t = stamp_s(m.header.stamp)
        if t - self.last_save_t < 1.0 / self.a.hz - 1e-3:   # throttle to --hz
            return
        if not self.odom:
            return
        # nearest odom sample (plant publishes ~100 Hz, so this is within ~10 ms)
        o = min(self.odom, key=lambda r: abs(r[0] - t))
        if abs(o[0] - t) > 0.05:
            return
        self.last_save_t = t

        arr = np.frombuffer(m.data, dtype=np.uint8).reshape(m.height, m.step)[:, : m.width * 3]
        arr = arr.reshape(m.height, m.width, 3)
        if m.encoding.lower().startswith("bgr"):
            arr = arr[:, :, ::-1]
        PILImage.fromarray(arr).save(os.path.join(self.ep_dir, "rgb", f"{self.i:06d}.jpg"), quality=92)

        rec = {"i": self.i, "t": t,
               "ego": {"x": o[1], "y": o[2], "z": o[3], "yaw": o[4]},
               "speed": o[5], "drive": dict(self.drive),
               "noise": bool(self.noise_since_save or self.noise_now)}
        self.noise_since_save = False
        self.log.write(json.dumps(rec) + "\n")
        self.i += 1
        if self.i % 150 == 0:
            self.log.flush()
            self.get_logger().info(f"{self.i} frames")

    def close(self):
        self.log.close()
        self.get_logger().info(f"saved {self.i} frames -> {self.ep_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--map", default="lab_00", help="just used to name the output folder")
    p.add_argument("--out", default="data")
    p.add_argument("--hz", type=float, default=10.0, help="save rate (<= camera rate)")
    p.add_argument("--image-topic", default="/camera/color/image_raw")
    p.add_argument("--info-topic", default="/camera/color/camera_info")
    p.add_argument("--odom-topic", default="/ego_racecar/odom")
    p.add_argument("--drive-topic", default="/drive")
    p.add_argument("--base-frame", default="ego_racecar/base_link")
    p.add_argument("--cam-frame", default="ego_racecar/camera_optical_link")
    a = p.parse_args()

    rclpy.init()
    node = Recorder(a)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
