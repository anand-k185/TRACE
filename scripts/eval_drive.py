#!/usr/bin/env python3
"""
TRACE - score a test drive recorded with gz_collect.py while model_drive.py was driving.

The map is used ONLY here, for grading (the model never saw it):
  - distance driven, laps completed
  - duct contacts (car footprint touching a wall) and closest approach
  - distance from the map centerline (mean / 95th percentile / max)
  - time spent stuck
and a plot of the driven path over the track, colored by distance from the centerline.

  python3 eval_drive.py --episode data_test/lab_15/ep_XXXX --map lab_15
"""
import argparse, json, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from autodrive import load_map, Field, trace_centerline

L_CAR, W_CAR, REAR = 0.37, 0.22, 0.02          # footprint used by the sim's collision guard


def footprint_clearance(field, x, y, yaw):
    xs, ys = np.meshgrid(np.linspace(-REAR, L_CAR - REAR, 9), np.linspace(-W_CAR / 2, W_CAR / 2, 5))
    c, s = math.cos(yaw), math.sin(yaw)
    px = x + xs * c - ys * s
    py = y + xs * s + ys * c
    return float(field(px.ravel(), py.ravel()).min())


def dist_to_loop(P, C):
    """distance from each point in P to the closed polyline C, and nearest index."""
    out, idx = np.empty(len(P)), np.empty(len(P), dtype=int)
    for k in range(0, len(P), 256):
        d = np.linalg.norm(P[k:k + 256, None, :] - C[None], axis=-1)
        idx[k:k + 256] = d.argmin(1)
        out[k:k + 256] = d.min(1)
    return out, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True, help="episode folder written by gz_collect.py")
    ap.add_argument("--map", required=True, help="e.g. lab_15")
    ap.add_argument("--map-yaml", default=None)
    ap.add_argument("--out", default=None, help="where to save the plot (default: inside the episode folder)")
    a = ap.parse_args()
    if a.map_yaml is None:
        try:
            from ament_index_python.packages import get_package_share_directory
            a.map_yaml = os.path.join(get_package_share_directory("f1tenth_gym_gazebo"), "maps", a.map + ".yaml")
        except Exception:
            a.map_yaml = os.path.expanduser(f"~/sim_ws/src/f1tenth_gym_gazebo/maps/{a.map}.yaml")

    fr = [json.loads(l) for l in open(os.path.join(a.episode, "frames.jsonl")) if l.strip()]
    if len(fr) < 5:
        sys.exit("episode too short")
    occ, res, origin = load_map(a.map_yaml)
    field = Field(occ, res, origin)
    P = np.array([[f["ego"]["x"], f["ego"]["y"]] for f in fr])
    yaw = np.array([f["ego"]["yaw"] for f in fr])
    t = np.array([f["t"] for f in fr])
    spd = np.array([f["speed"] for f in fr])
    C = trace_centerline(field, P[0, 0], P[0, 1], yaw[0])

    dist = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
    dev, idx = dist_to_loop(P, C)
    n = len(C)
    step = np.diff(idx)
    step = (step + n // 2) % n - n // 2                     # unwrap around the loop
    laps = float(step.sum()) / n
    clear = np.array([footprint_clearance(field, x, y, w) for (x, y), w in zip(P, yaw)])
    touching = clear < 0.02
    contacts = int(np.sum(touching[1:] & ~touching[:-1]) + touching[0])
    dt = np.diff(t, prepend=t[0])
    stuck_s = float(dt[(spd < 0.05) & (t > t[0] + 3)].sum())

    rep = {
        "duration_s": round(float(t[-1] - t[0]), 1),
        "distance_m": round(dist, 1),
        "laps": round(laps, 2),
        "duct_contacts": contacts,
        "closest_approach_m": round(float(clear.min()), 3),
        "centerline_dev_mean_m": round(float(dev.mean()), 3),
        "centerline_dev_p95_m": round(float(np.percentile(dev, 95)), 3),
        "centerline_dev_max_m": round(float(dev.max()), 3),
        "stuck_s": round(stuck_s, 1),
    }
    print(json.dumps(rep, indent=2))
    verdict = ("PASS: completed {:.1f} lap(s) with no duct contact".format(laps) if contacts == 0 and laps >= 1
               else "needs work: see contacts / laps above")
    print(verdict)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rr, cc = np.nonzero(occ)
    wx, wy = origin[0] + cc * res, origin[1] + (occ.shape[0] - 1 - rr) * res
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(wx, wy, s=0.3, c="0.6")
    ax.plot(C[:, 0], C[:, 1], "-", color="deepskyblue", lw=1.5, label="map centerline (grading only)")
    sc = ax.scatter(P[:, 0], P[:, 1], c=dev, s=4, cmap="RdYlGn_r", vmin=0, vmax=0.35, label="driven by model")
    if touching.any():
        ax.scatter(P[touching, 0], P[touching, 1], s=40, facecolors="none", edgecolors="r", label="duct contact")
    ax.plot(P[0, 0], P[0, 1], "k^", ms=10, label="start")
    plt.colorbar(sc, ax=ax, shrink=0.7, label="distance from centerline (m)")
    pad = 1.0
    ax.set_xlim(P[:, 0].min() - pad, P[:, 0].max() + pad); ax.set_ylim(P[:, 1].min() - pad, P[:, 1].max() + pad)
    ax.set_aspect("equal"); ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"{a.map}: {rep['laps']} laps, {contacts} contacts, mean dev {rep['centerline_dev_mean_m']} m")
    out = a.out or os.path.join(a.episode, "drive_eval.png")
    plt.savefig(out, dpi=110, bbox_inches="tight")
    print("plot ->", out)


if __name__ == "__main__":
    main()
