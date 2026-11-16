#!/usr/bin/env python3
"""
TRACE - autopilot for f1tenth_gym_gazebo, so you don't have to drive laps by hand.

How it works:
  1. Loads the track's occupancy map (maps/<map>.yaml, shipped with the repo; the
     map frame == Gazebo world == odom frame).
  2. From the car's first odom pose, traces the corridor's centerline once
     (walks along the ridge of the distance-to-wall field until it closes the loop),
     in whatever direction the car is facing.
  3. Follows that loop with pure pursuit using /ego_racecar/odom (ground truth here).
  4. For recovery data, now and then adds a steering offset for 0.4-1.0 s and
     publishes /trace/noise (Bool) while it does. gz_collect.py records the flag,
     gz_make_labels.py drops frames whose future path contains the drift.

  python3 autodrive.py --map lab_00
  python3 autodrive.py --map lab_00 --speed 0.8 --laps 3

To drive the track the other way, launch the sim with yaw:=<spawn yaw + 3.1416>.
Ctrl-C stops the car.
"""
import argparse, math, os, random
import numpy as np

# ---------------- map + centerline (pure numpy/scipy, no ROS) ----------------

def load_map(yaml_path):
    import yaml
    from PIL import Image
    m = yaml.safe_load(open(yaml_path))
    img = os.path.join(os.path.dirname(yaml_path), m["image"])
    g = np.array(Image.open(img).convert("L")).astype(float) / 255.0
    occ_p = g if m.get("negate", 0) else 1.0 - g
    occ = occ_p > m.get("occupied_thresh", 0.65)
    return occ, float(m["resolution"]), m["origin"]


class Field:
    """Distance-to-nearest-wall in meters, sampled bilinearly in world coords."""
    def __init__(self, occ, res, origin):
        from scipy import ndimage
        self.d = ndimage.distance_transform_edt(~occ) * res
        self.res, self.ox, self.oy, self.H = res, origin[0], origin[1], occ.shape[0]

    def __call__(self, x, y):
        c = (np.asarray(x) - self.ox) / self.res
        r = (self.H - 1) - (np.asarray(y) - self.oy) / self.res
        c = np.clip(c, 0, self.d.shape[1] - 1.001)
        r = np.clip(r, 0, self.d.shape[0] - 1.001)
        c0, r0 = np.floor(c).astype(int), np.floor(r).astype(int)
        fc, fr = c - c0, r - r0
        d = self.d
        return ((1 - fr) * ((1 - fc) * d[r0, c0] + fc * d[r0, c0 + 1])
                + fr * ((1 - fc) * d[r0 + 1, c0] + fc * d[r0 + 1, c0 + 1]))


def trace_centerline(field, x0, y0, yaw0, step=0.05, look=0.35, max_len=400.0):
    """Walk along the corridor's middle (ridge of the distance field) until the loop closes."""
    hs = np.linspace(-0.6, 0.6, 31)
    x, y, h = x0, y0, yaw0
    pts, travelled = [(x, y)], 0.0
    for _ in range(int(max_len / step)):
        cand = h + hs
        score = field(x + look * np.cos(cand), y + look * np.sin(cand)) - 0.03 * np.abs(hs)
        h = cand[int(np.argmax(score))]
        x, y = x + step * math.cos(h), y + step * math.sin(h)
        pts.append((x, y))
        travelled += step
        if travelled > 3.0 and math.hypot(x - x0, y - y0) < 2 * step + 0.05:
            break
    else:
        raise RuntimeError("centerline did not close into a loop")
    P = np.array(pts[:-1])
    k = 9                                                # circular smoothing
    pad = np.vstack([P[-k:], P, P[:k]])
    P = np.stack([np.convolve(pad[:, i], np.ones(2 * k + 1) / (2 * k + 1), "valid") for i in range(2)], 1)
    return P


def weave_path(center, field, amp=0.25, seed=0, car_half=0.11, margin=0.14):
    """A path that wanders left/right of the centerline (two slow sine waves), so the recorded
    drive covers off-center positions and angled headings. The offset is capped so the car body
    always keeps `margin` meters from the ducts."""
    rng = np.random.default_rng(seed)
    n = len(center)
    seg = np.linalg.norm(np.diff(np.vstack([center, center[:1]]), axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)[:-1]])
    L = seg.sum()
    k1, k2 = max(1, round(L / rng.uniform(5.0, 7.0))), max(2, round(L / rng.uniform(2.6, 3.4)))
    off = 0.7 * np.sin(2 * np.pi * k1 * s / L + rng.uniform(0, 6.28)) + 0.3 * np.sin(2 * np.pi * k2 * s / L + rng.uniform(0, 6.28))
    half = field(center[:, 0], center[:, 1])                       # corridor half-width here
    cap = np.clip(half - car_half - margin, 0.0, amp)
    off = off * cap
    tan = np.roll(center, -1, 0) - np.roll(center, 1, 0)
    tan /= np.linalg.norm(tan, axis=1, keepdims=True).clip(1e-9)
    normal = np.stack([-tan[:, 1], tan[:, 0]], 1)                  # points left
    return center + normal * off[:, None]


class PurePursuit:
    def __init__(self, path, wheelbase=0.33, lookahead=0.55, max_steer=0.40, speed=1.0):
        self.P, self.L, self.ld, self.ms, self.v = path, wheelbase, lookahead, max_steer, speed
        self.idx = None
        seg = np.linalg.norm(np.diff(np.vstack([path, path[:1]]), axis=0), axis=1)
        self.ds = float(seg.mean())
        self.lap_len = float(seg.sum())
        self.progress = 0.0

    def __call__(self, x, y, yaw):
        n = len(self.P)
        if self.idx is None:
            self.idx = int(np.argmin(np.linalg.norm(self.P - [x, y], axis=1)))
        win = (self.idx + np.arange(0, 40)) % n               # search forward only
        j = win[int(np.argmin(np.linalg.norm(self.P[win] - [x, y], axis=1)))]
        self.progress += ((j - self.idx) % n) * self.ds
        self.idx = j
        t = self.P[(j + int(self.ld / self.ds)) % n]
        dx, dy = t[0] - x, t[1] - y
        lx = math.cos(yaw) * dx + math.sin(yaw) * dy
        ly = -math.sin(yaw) * dx + math.cos(yaw) * dy
        steer = math.atan2(2 * self.L * ly, lx * lx + ly * ly)
        steer = float(np.clip(steer, -self.ms, self.ms))
        speed = self.v * (1.0 - 0.45 * abs(steer) / self.ms)
        return steer, speed


# ---------------- ROS node ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True, help="e.g. lab_00 (must match the launched map)")
    ap.add_argument("--map-yaml", default=None, help="override path to <map>.yaml")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--laps", type=float, default=0, help="stop after N laps (0 = until Ctrl-C)")
    ap.add_argument("--noise-prob", type=float, default=0.003, help="chance per control tick (20 Hz) to start a drift (~1 per 17 s)")
    ap.add_argument("--noise-steer", type=float, default=0.18, help="max steering offset during a drift [rad]")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weave", type=float, default=0.0,
                    help="wander up to this many meters either side of the centerline (try 0.25). "
                         "Label these runs with gz_make_labels.py --labels centerline")
    a = ap.parse_args()

    import rclpy
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from nav_msgs.msg import Odometry
    from ackermann_msgs.msg import AckermannDriveStamped
    from std_msgs.msg import Bool

    if a.map_yaml is None:
        from ament_index_python.packages import get_package_share_directory
        a.map_yaml = os.path.join(get_package_share_directory("f1tenth_gym_gazebo"), "maps", a.map + ".yaml")
    occ, res, origin = load_map(a.map_yaml)
    field = Field(occ, res, origin)
    random.seed(a.seed)

    class AutoDrive(Node):
        def __init__(self):
            super().__init__("trace_autodrive", parameter_overrides=[
                Parameter("use_sim_time", Parameter.Type.BOOL, True)])
            self.pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)
            self.noise_pub = self.create_publisher(Bool, "/trace/noise", 10)
            self.create_subscription(Odometry, "/ego_racecar/odom", self.on_odom,
                                     QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
            self.pose = None
            self.pp = None
            self.noise_left, self.noise_val = 0, 0.0
            self.create_timer(0.05, self.tick)               # 20 Hz control
            self.get_logger().info(f"waiting for odom ... (map {a.map_yaml})")

        def on_odom(self, m):
            q = m.pose.pose.orientation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            self.pose = (m.pose.pose.position.x, m.pose.pose.position.y, yaw)

        def tick(self):
            if self.pose is None:
                return
            if self.pp is None:
                path = trace_centerline(field, *self.pose)
                if a.weave > 0:
                    path = weave_path(path, field, amp=a.weave, seed=a.seed)
                self.pp = PurePursuit(path, speed=a.speed)
                self.get_logger().info(f"centerline traced: {self.pp.lap_len:.1f} m loop. driving.")
            steer, speed = self.pp(*self.pose)

            if self.noise_left == 0 and random.random() < a.noise_prob:
                self.noise_left = random.randint(8, 20)                # 0.4-1.0 s
                self.noise_val = random.choice([-1, 1]) * random.uniform(0.5, 1.0) * a.noise_steer
            noisy = self.noise_left > 0
            if noisy:
                steer = float(np.clip(steer + self.noise_val, -0.40, 0.40))
                self.noise_left -= 1
            self.noise_pub.publish(Bool(data=noisy))

            if a.laps and self.pp.progress >= a.laps * self.pp.lap_len:
                self.get_logger().info(f"{a.laps} laps done")
                self.stop()
                raise SystemExit
            self.send(steer, speed)

        def send(self, steer, speed):
            msg = AckermannDriveStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.drive.steering_angle = float(steer)
            msg.drive.speed = float(speed)
            self.pub.publish(msg)

        def stop(self):
            for _ in range(5):
                self.send(0.0, 0.0)

    rclpy.init()
    node = AutoDrive()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
