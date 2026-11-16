#!/usr/bin/env python3
"""
TRACE - trajectory labels for episodes recorded with gz_collect.py.

Waypoints are spaced by DISTANCE along the path the car actually drove
(default 1.0, 1.5, ... 3.5 m ahead), not by time. The F1TENTH car drives
slowly (teleop max ~1 m/s), so time-spaced points would bunch up under the
camera, and this sim's camera can't see the ground closer than ~0.72 m.
Distance spacing also means stopped frames don't produce collapsed labels.

Output per sample: pixel coords (uv) for the heatmap model, the same points
in the car frame (ego_xy, meters, x forward / y left), and a visibility mask.

  python3 gz_make_labels.py --data data/lab_00 --viz 30

--labels centerline (recommended, needed for --weave recordings):
  the waypoints are points on the MAP CENTERLINE ahead of the car, instead of the path the car
  happened to drive. Every frame gets a label that leads back to the middle of the corridor, even
  when the car is off-center or at an angle, so the model learns to recover. Nothing is dropped
  for drift or end-of-episode. The map is used only here, to make labels.

  python3 gz_make_labels.py --data data/lab_00 --labels centerline --start 1.0 --spacing 0.3 --viz 20
"""
import argparse, glob, json, os, random
from collections import Counter
import numpy as np


def load_jsonl(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def make_sample(ep_dir, fr, wx, wy, n_wp, T_cam_base, K, W, H, off_center):
    # world -> base_link (2D pose; ground is world z = 0)
    e = fr["ego"]
    c, s = np.cos(e["yaw"]), np.sin(e["yaw"])
    dx, dy = wx - e["x"], wy - e["y"]
    bx = c * dx + s * dy
    by = -s * dx + c * dy
    pb = np.stack([bx, by, np.full(n_wp, -e.get("z", 0.0)), np.ones(n_wp)])
    pc = (T_cam_base @ pb)[:3]                      # optical frame: x right, y down, z fwd
    uvw = K @ pc
    uv = (uvw[:2] / np.clip(uvw[2], 1e-6, None)).T
    valid = (pc[2] > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    smp = {
        "img": os.path.join(ep_dir, "rgb", f"{fr['i']:06d}.jpg"),
        "uv": uv.round(2).tolist(),
        "valid": valid.tolist(),
        "ego_xy": np.stack([bx, by], 1).round(3).tolist(),
        "speed": fr["speed"],
        "command": fr.get("command", "NONE"),
    }
    if off_center is not None:
        smp["off_center"] = round(off_center, 3)
    return smp


def find_map_yaml(name):
    try:
        from ament_index_python.packages import get_package_share_directory
        p = os.path.join(get_package_share_directory("f1tenth_gym_gazebo"), "maps", name + ".yaml")
        if os.path.exists(p):
            return p
    except Exception:
        pass
    return os.path.expanduser(f"~/sim_ws/src/f1tenth_gym_gazebo/maps/{name}.yaml")


def centerline_targets(frames, map_yaml, targets):
    """For every frame: points on the map centerline `targets` meters ahead of the car (world xy)."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from autodrive import load_map, Field, trace_centerline
    e0 = frames[0]["ego"]
    C = trace_centerline(Field(*load_map(map_yaml)), e0["x"], e0["y"], e0["yaw"])
    n = len(C)
    seg = np.linalg.norm(np.diff(np.vstack([C, C[:1]]), axis=0), axis=1)
    L = seg.sum()
    cum = np.concatenate([[0.0], np.cumsum(seg)])                  # length n+1, cum[n] = L
    Cx, Cy = np.append(C[:, 0], C[0, 0]), np.append(C[:, 1], C[0, 1])
    out, j = [], None
    for f in frames:
        p = np.array([f["ego"]["x"], f["ego"]["y"]])
        if j is None:
            j = int(np.argmin(np.linalg.norm(C - p, axis=1)))
        else:                                                      # track along the loop (no jumps to a
            win = (j + np.arange(-10, 80)) % n                     # neighbouring corridor)
            j = int(win[np.argmin(np.linalg.norm(C[win] - p, axis=1))])
        want = (cum[j] + targets) % L
        out.append((np.interp(want, cum, Cx), np.interp(want, cum, Cy), float(np.linalg.norm(C[j] - p))))
    return out


def label_episode(ep_dir, n_wp, start, spacing, min_speed, labels="driven", map_yaml=None):
    meta = json.load(open(os.path.join(ep_dir, "meta.json")))
    frames = load_jsonl(os.path.join(ep_dir, "frames.jsonl"))
    if len(frames) < 2:
        return [], 0, meta
    K = np.asarray(meta["K"])
    T_cam_base = np.linalg.inv(np.asarray(meta["T_base_cam"]))   # base_link -> optical
    W, H = meta["width"], meta["height"]

    P = np.array([[f["ego"]["x"], f["ego"]["y"]] for f in frames])
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    cum = cum + np.arange(len(cum)) * 1e-9            # strictly increasing for np.interp
    targets = start + spacing * np.arange(n_wp)

    cl = centerline_targets(frames, map_yaml, targets) if labels == "centerline" else None
    out, dropped = [], 0
    for t, fr in enumerate(frames):
        if fr["speed"] < min_speed:
            dropped += 1
            continue
        if cl is not None:
            wx, wy, off = cl[t]
            if off > 0.6:                                # lost track of the centerline (shouldn't happen)
                dropped += 1
                continue
            out.append(make_sample(ep_dir, fr, wx, wy, n_wp, T_cam_base, K, W, H, off))
            continue
        want = cum[t] + targets
        if want[-1] > cum[-1]:                       # not enough path left in episode
            dropped += 1
            continue
        end = int(np.searchsorted(cum, want[-1]))   # last frame the label path uses
        if any(f.get("noise", False) for f in frames[t + 1 : end + 1]):
            dropped += 1                             # future contains autopilot drift, not good driving
            continue
        wx = np.interp(want, cum, P[:, 0])
        wy = np.interp(want, cum, P[:, 1])

        out.append(make_sample(ep_dir, fr, wx, wy, n_wp, T_cam_base, K, W, H, None))
    return out, dropped, meta


def draw(samples, out_dir, n):
    from PIL import Image, ImageDraw
    os.makedirs(out_dir, exist_ok=True)
    for s in random.sample(samples, min(n, len(samples))):
        im = Image.open(s["img"]).convert("RGB")
        d = ImageDraw.Draw(im)
        pts = [tuple(p) for p, v in zip(s["uv"], s["valid"]) if v]
        if len(pts) > 1:
            d.line(pts, fill=(0, 255, 0), width=3)
        for p in pts:
            d.ellipse([p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5], outline=(255, 0, 0), width=2)
        d.text((8, 8), f"{s['speed']:.2f} m/s", fill=(255, 255, 0))
        name = s["img"].replace(os.sep, "_").replace(".jpg", ".png")[-90:]
        im.save(os.path.join(out_dir, name))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="folder containing ep_* dirs")
    p.add_argument("--n-wp", type=int, default=6)
    p.add_argument("--start", type=float, default=1.0, help="meters ahead of first waypoint")
    p.add_argument("--spacing", type=float, default=0.5, help="meters between waypoints")
    p.add_argument("--min-speed", type=float, default=0.15, help="drop frames slower than this (m/s)")
    p.add_argument("--viz", type=int, default=0)
    p.add_argument("--labels", choices=["driven", "centerline"], default="driven",
                   help="driven = path the car drove; centerline = map centerline ahead (use for --weave runs)")
    p.add_argument("--map-yaml", default=None, help="map file for --labels centerline (default: found from the folder name)")
    a = p.parse_args()
    if a.labels == "centerline" and a.map_yaml is None:
        a.map_yaml = find_map_yaml(os.path.basename(os.path.normpath(a.data)))
        if not os.path.exists(a.map_yaml):
            raise SystemExit(f"map file not found: {a.map_yaml}  (pass --map-yaml)")

    samples, drop = [], 0
    for ep in sorted(glob.glob(os.path.join(a.data, "ep_*"))):
        if not (os.path.exists(os.path.join(ep, "meta.json"))
                and os.path.exists(os.path.join(ep, "frames.jsonl"))):
            print(f"{ep}: skipped (empty episode: recorder never got camera data)")
            continue
        s, d, _ = label_episode(ep, a.n_wp, a.start, a.spacing, a.min_speed, a.labels, a.map_yaml)
        samples += s
        drop += d
        print(f"{ep}: {len(s)} samples, {d} dropped")

    out = os.path.join(a.data, "labels.jsonl")
    with open(out, "w") as fh:
        for s in samples:
            fh.write(json.dumps(s) + "\n")
    print(f"\n{len(samples)} samples -> {out}  ({drop} dropped: slow, end of episode, or drift ahead)")
    if samples:
        print("frac all-waypoints-visible:", round(float(np.mean([all(s["valid"]) for s in samples])), 3))
        print("per-waypoint visible:", np.mean([s["valid"] for s in samples], 0).round(2).tolist())
        if "off_center" in samples[0]:
            oc = np.array([s["off_center"] for s in samples])
            print(f"car distance from centerline: mean {oc.mean():.2f} m, 95th pct {np.percentile(oc, 95):.2f} m")
    if a.viz and samples:
        draw(samples, os.path.join(a.data, "viz"), a.viz)
        print(f"drew {a.viz} frames -> {os.path.join(a.data, 'viz')}")


if __name__ == "__main__":
    main()
