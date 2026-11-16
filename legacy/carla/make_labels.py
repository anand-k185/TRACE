"""
TRACE - build trajectory labels from collected episodes.

For each frame t, take the ego car's FUTURE positions (t + k*step), and express them:
  - in the ego frame at time t (meters; x forward, y right)  -> useful for control later
  - as pixels in the image at time t                          -> what the heatmap model predicts

Frames are dropped if: not enough future left, a noise burst overlaps the future window
(those futures show drift, not good driving), or a crash happens in the window.
Frames where the car is noisy NOW are kept: they show the recovery.

Heatmaps are made on the fly in the training Dataset from these pixel coords
(so you can change sigma / resolution without relabeling).

Usage:
  python make_labels.py --data data/Town04 --n-wp 8 --dt 0.5
  python make_labels.py --data data/Town04 --viz 30     # also draw labels on 30 random frames
"""
import argparse, glob, json, os, random
import numpy as np


def load_jsonl(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh]


def project(world_pts, world2cam, K):
    """world_pts: (N,3). Returns pixels (N,2) and depth (N,)."""
    pts = np.c_[world_pts, np.ones(len(world_pts))] @ np.asarray(world2cam).T  # CARLA cam: x fwd, y right, z up
    cam = np.stack([pts[:, 1], -pts[:, 2], pts[:, 0]], 1)  # -> standard: x right, y down, z forward
    depth = cam[:, 2]
    uv = (np.asarray(K) @ cam.T).T
    uv = uv[:, :2] / np.clip(uv[:, 2:3], 1e-6, None)
    return uv, depth


def label_episode(ep_dir, n_wp, dt):
    meta = json.load(open(os.path.join(ep_dir, "meta.json")))
    frames = load_jsonl(os.path.join(ep_dir, "frames.jsonl"))
    step = int(round(dt * meta["hz"]))
    W, H, K = meta["width"], meta["height"], meta["K"]
    horizon = n_wp * step
    out, dropped = [], 0

    for t, fr in enumerate(frames):
        if t + horizon >= len(frames):
            break
        window = frames[t + 1 : t + horizon + 1]
        if any(f["noise"] or f["crashed"] for f in window):
            dropped += 1
            continue

        fut = [frames[t + k * step]["ego"] for k in range(1, n_wp + 1)]
        world_pts = np.array([[p["x"], p["y"], p["z"]] for p in fut])  # vehicle origin ~ ground level

        # ego frame (meters)
        ego_pts = np.c_[world_pts, np.ones(n_wp)] @ np.asarray(fr["ego_world2veh"]).T
        ego_xy = ego_pts[:, :2]

        # image pixels
        uv, depth = project(world_pts, fr["cam_world2cam"], K)
        in_img = (depth > 0.5) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)

        out.append({
            "img": os.path.join(ep_dir, "rgb", f"{fr['i']:06d}.jpg"),
            "uv": uv.round(2).tolist(),
            "valid": in_img.tolist(),
            "ego_xy": ego_xy.round(3).tolist(),
            "speed": fr["speed"],
            "command": fr["command"],
            "noisy_now": fr["noise"],
        })
    return out, dropped, meta


def draw(samples, meta, out_dir, n):
    from PIL import Image, ImageDraw
    os.makedirs(out_dir, exist_ok=True)
    for s in random.sample(samples, min(n, len(samples))):
        im = Image.open(s["img"]).convert("RGB")
        d = ImageDraw.Draw(im)
        pts = [tuple(p) for p, v in zip(s["uv"], s["valid"]) if v]
        if len(pts) > 1:
            d.line(pts, fill=(0, 255, 0), width=3)
        for k, (p, v) in enumerate(zip(s["uv"], s["valid"])):
            if v:
                r = 6
                d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], outline=(255, 0, 0), width=2)
        d.text((10, 10), f"{s['command']}  {s['speed']*3.6:.0f} km/h", fill=(255, 255, 0))
        name = s["img"].replace(os.sep, "_").replace(".jpg", ".png")
        im.save(os.path.join(out_dir, name[-80:]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="folder containing ep_XXXX dirs")
    p.add_argument("--n-wp", type=int, default=8)
    p.add_argument("--dt", type=float, default=0.5, help="seconds between waypoints")
    p.add_argument("--viz", type=int, default=0)
    a = p.parse_args()

    all_samples, total_drop = [], 0
    meta = None
    for ep_dir in sorted(glob.glob(os.path.join(a.data, "ep_*"))):
        s, d, meta = label_episode(ep_dir, a.n_wp, a.dt)
        all_samples += s
        total_drop += d
        print(f"{ep_dir}: {len(s)} samples, {d} dropped (noise/crash in window)")

    out_path = os.path.join(a.data, "labels.jsonl")
    with open(out_path, "w") as fh:
        for s in all_samples:
            fh.write(json.dumps(s) + "\n")

    from collections import Counter
    print(f"\n{len(all_samples)} samples -> {out_path}  ({total_drop} dropped)")
    print("commands:", dict(Counter(s["command"] for s in all_samples)))
    print("frac all-waypoints-visible:", np.mean([all(s["valid"]) for s in all_samples]).round(3))

    if a.viz and all_samples:
        draw(all_samples, meta, os.path.join(a.data, "viz"), a.viz)
        print(f"drew {a.viz} frames -> {os.path.join(a.data, 'viz')}")


if __name__ == "__main__":
    main()
