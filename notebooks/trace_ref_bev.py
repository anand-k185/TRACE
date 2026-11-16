# TRACE v2 (top-down model) - compare predictions with the actual path and the hard-coded map centerline.
# Needs: data/<map>/ep_*/{meta.json,frames.jsonl} (from gz_collect.py) and the repo's maps/<map>.yaml.
import json, math, os
import numpy as np
import torch
import torch.nn.functional as F

import trace_bev as tb

# ------------------------------------------------------------ map + centerline (same as autodrive.py)
def load_map(yaml_path):
    import yaml
    from PIL import Image
    m = yaml.safe_load(open(yaml_path))
    g = np.array(Image.open(os.path.join(os.path.dirname(yaml_path), m["image"])).convert("L")).astype(float) / 255.0
    occ_p = g if m.get("negate", 0) else 1.0 - g
    return occ_p > m.get("occupied_thresh", 0.65), float(m["resolution"]), m["origin"]


class Field:
    def __init__(self, occ, res, origin):
        from scipy import ndimage
        self.occ = occ
        self.d = ndimage.distance_transform_edt(~occ) * res
        self.res, self.ox, self.oy, self.H = res, origin[0], origin[1], occ.shape[0]

    def __call__(self, x, y):
        c = np.clip((np.asarray(x) - self.ox) / self.res, 0, self.d.shape[1] - 1.001)
        r = np.clip((self.H - 1) - (np.asarray(y) - self.oy) / self.res, 0, self.d.shape[0] - 1.001)
        c0, r0 = np.floor(c).astype(int), np.floor(r).astype(int)
        fc, fr = c - c0, r - r0
        d = self.d
        return ((1 - fr) * ((1 - fc) * d[r0, c0] + fc * d[r0, c0 + 1])
                + fr * ((1 - fc) * d[r0 + 1, c0] + fc * d[r0 + 1, c0 + 1]))

    def walls_near(self, x, y, radius=5.0, max_pts=4000):
        """World coords of occupied cells within radius (for the bird's-eye plot)."""
        rr, cc = np.nonzero(self.occ)
        wx = self.ox + cc * self.res
        wy = self.oy + (self.H - 1 - rr) * self.res
        m = (np.abs(wx - x) < radius) & (np.abs(wy - y) < radius)
        wx, wy = wx[m], wy[m]
        if len(wx) > max_pts:
            k = np.random.default_rng(0).choice(len(wx), max_pts, replace=False)
            wx, wy = wx[k], wy[k]
        return wx, wy


def trace_centerline(field, x0, y0, yaw0, step=0.05, look=0.35, max_len=400.0):
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
    k = 9
    pad = np.vstack([P[-k:], P, P[:k]])
    return np.stack([np.convolve(pad[:, i], np.ones(2 * k + 1) / (2 * k + 1), "valid") for i in range(2)], 1)


# ------------------------------------------------------------ per-episode context
class Episodes:
    """Looks up, for any labels.jsonl sample: the car pose at that frame, camera meta,
    and the map centerline (traced once per episode in the direction it was driven)."""

    def __init__(self, maps_dir):
        self.maps_dir = maps_dir
        self.meta, self.frames, self.center, self.fields = {}, {}, {}, {}

    def _ep(self, s):
        ep_dir = os.path.dirname(os.path.dirname(s["img"]))
        if ep_dir not in self.meta:
            self.meta[ep_dir] = json.load(open(os.path.join(ep_dir, "meta.json")))
            fr = [json.loads(l) for l in open(os.path.join(ep_dir, "frames.jsonl")) if l.strip()]
            self.frames[ep_dir] = {f["i"]: f for f in fr}
            mp = s["map"]
            if mp not in self.fields:
                self.fields[mp] = Field(*load_map(os.path.join(self.maps_dir, mp + ".yaml")))
            e0 = fr[0]["ego"]
            self.center[ep_dir] = trace_centerline(self.fields[mp], e0["x"], e0["y"], e0["yaw"])
        return ep_dir

    def context(self, s):
        ep = self._ep(s)
        i = int(os.path.splitext(os.path.basename(s["img"]))[0])
        return {"meta": self.meta[ep], "pose": self.frames[ep][i]["ego"],
                "center": self.center[ep], "field": self.fields[s["map"]]}


# ------------------------------------------------------------ geometry
def world_to_car(pts_xy, pose):
    c, s = math.cos(pose["yaw"]), math.sin(pose["yaw"])
    d = np.asarray(pts_xy) - [pose["x"], pose["y"]]
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], 1)     # x fwd, y left


def car_to_pixels(car_xy, pose, meta):
    """Ground points in car frame -> native-resolution pixels (+ visibility)."""
    K = np.asarray(meta["K"])
    T_cam_base = np.linalg.inv(np.asarray(meta["T_base_cam"]))
    n = len(car_xy)
    pb = np.stack([car_xy[:, 0], car_xy[:, 1], np.full(n, -pose.get("z", 0.0)), np.ones(n)])
    pc = (T_cam_base @ pb)[:3]
    uvw = K @ pc
    uv = (uvw[:2] / np.clip(uvw[2], 1e-6, None)).T
    W, H = meta["width"], meta["height"]
    vis = (pc[2] > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    return uv, vis


def pixels_to_car(uv_native, pose, meta):
    """Native pixels -> where that ray hits the floor, in car frame (meters). NaN above the horizon."""
    K_inv = np.linalg.inv(np.asarray(meta["K"]))
    T = np.asarray(meta["T_base_cam"])
    R, c = T[:3, :3], T[:3, 3]
    uv1 = np.c_[uv_native, np.ones(len(uv_native))]
    d = (R @ (K_inv @ uv1.T)).T                                   # ray directions in base frame
    z_ground = -pose.get("z", 0.0)
    t = (z_ground - c[2]) / np.where(np.abs(d[:, 2]) < 1e-9, -1e-9, d[:, 2])
    p = c[None, :] + t[:, None] * d
    p[(t <= 0) | (d[:, 2] >= 0)] = np.nan
    p[np.linalg.norm(p[:, :2], axis=1) > 10.0] = np.nan          # near-horizon pixels -> unreliable range
    return p[:, :2]


def centerline_ahead(ctx, start=0.3, end=4.0):
    """Centerline segment from `start` to `end` meters ahead of the car (along the loop), car frame."""
    P = ctx["center"]
    pose = ctx["pose"]
    j = int(np.argmin(np.linalg.norm(P - [pose["x"], pose["y"]], axis=1)))
    seg = np.linalg.norm(np.diff(np.vstack([P, P[:1]]), axis=0), axis=1)
    order = (j + np.arange(len(P))) % len(P)
    arc = np.concatenate([[0], np.cumsum(seg[order][:-1])])
    keep = (arc >= start) & (arc <= end)
    return world_to_car(P[order][keep], pose)


def dist_to_polyline(pts, line):
    """Distance from each point to the nearest segment of a polyline."""
    out = np.full(len(pts), np.nan)
    if len(line) < 2:
        return out
    a, b = line[:-1], line[1:]
    ab = b - a
    L2 = (ab ** 2).sum(1).clip(1e-12)
    for k, p in enumerate(pts):
        if np.any(np.isnan(p)):
            continue
        t = (((p - a) * ab).sum(1) / L2).clip(0, 1)
        out[k] = np.sqrt((((a + t[:, None] * ab) - p) ** 2).sum(1)).min()
    return out



# ------------------------------------------------------------ model helpers (top-down model)
@torch.no_grad()
def predict(model, ds, idx, device, bs=64):
    """-> pred waypoints in meters (n, K, 2) and per-waypoint BEV probability maps (n, K, BH, BW)."""
    model.eval()
    P, H = [], []
    for b in range(0, len(idx), bs):
        x = torch.stack([ds[i][0] for i in idx[b:b + bs]]).to(device)
        logits = model(x).float()
        P.append(tb.decode(logits).cpu())
        H.append(logits.flatten(2).softmax(-1).view_as(logits).cpu())
    return torch.cat(P).numpy(), torch.cat(H).numpy()


def metrics_m(model, ds, samples, eps, device, curve_thresh=0.8):
    pred, _ = predict(model, ds, list(range(len(samples))), device)
    gt = np.array([s["ego_xy"] for s in samples])
    e_gt = np.linalg.norm(pred - gt, axis=-1)
    d_pc, d_gc = [], []
    for k, s in enumerate(samples):
        cl = centerline_ahead(eps.context(s), 0.0, 5.0)
        d_pc.append(dist_to_polyline(pred[k], cl))
        d_gc.append(dist_to_polyline(gt[k], cl))
    d_pc, d_gc = np.array(d_pc), np.array(d_gc)
    curve = np.abs(gt[:, -1, 1]) > curve_thresh
    return {
        "per_wp": {
            "pred_vs_actual_m (mean)": np.nanmean(e_gt, 0),
            "pred_vs_actual_m (median)": np.nanmedian(e_gt, 0),
            "pred_off_centerline_m (mean)": np.nanmean(d_pc, 0),
            "actual_off_centerline_m (mean)": np.nanmean(d_gc, 0),
        },
        "curves_vs_straights": {
            "curve frames": [int(curve.sum()), np.nanmean(e_gt[curve]), np.nanmean(d_pc[curve])],
            "straight frames": [int((~curve).sum()), np.nanmean(e_gt[~curve]), np.nanmean(d_pc[~curve])],
        },
        "worst_frames": list(np.argsort(-e_gt.mean(1))[:6]),
        "curve_frames": list(np.nonzero(curve)[0]),
    }


def show_panels(model, ds, samples, eps, device, picks, save=None):
    """Per frame: (1) top-down heatmaps + actual path | (2) prediction vs hard-coded centerline + walls
    | (3) camera view with all three."""
    import matplotlib.pyplot as plt
    pred, hms = predict(model, ds, picks, device)
    ext = [tb.Y_RANGE[1], tb.Y_RANGE[0], tb.X_RANGE[0], tb.X_RANGE[1]]
    fig, axes = plt.subplots(len(picks), 3, figsize=(16, 4.8 * len(picks)))
    axes = np.atleast_2d(axes)

    def setup_bev(ax):
        ax.set_xlim(tb.Y_RANGE[1], tb.Y_RANGE[0]); ax.set_ylim(-0.3, tb.X_RANGE[1]); ax.set_aspect("equal")
        ax.set_xlabel("<- left   lateral (m)   right ->"); ax.set_ylabel("forward (m)")
        ax.plot([0], [0], marker=(3, 0, 0), color="k", ms=12)

    for r, i in enumerate(picks):
        s, ctx = samples[i], eps.context(samples[i])
        gt = np.asarray(s["ego_xy"])
        cl = centerline_ahead(ctx, 0.0, 4.5)
        wx, wy = ctx["field"].walls_near(ctx["pose"]["x"], ctx["pose"]["y"], radius=5.0)
        walls = world_to_car(np.stack([wx, wy], 1), ctx["pose"]) if len(wx) else np.zeros((0, 2))
        e_act = np.linalg.norm(pred[r] - gt, axis=1).mean()
        e_cl = np.nanmean(dist_to_polyline(pred[r], cl))

        # (1) the heatmaps the transformer produced, with the actual path (reference)
        ax = axes[r, 0]
        h = hms[r].max(0)
        ax.imshow(h / h.max(), cmap="inferno", extent=ext, origin="upper")
        ax.plot(gt[:, 1], gt[:, 0], "o", mfc="none", mec="lime", ms=9, mew=2, label="actual waypoints")
        ax.plot(pred[r][:, 1], pred[r][:, 0], "x", color="cyan", ms=8, mew=2, label="decoded peaks")
        setup_bev(ax); ax.legend(loc="lower left", fontsize=7)
        ax.set_title(f"top-down heatmaps vs actual path   err {e_act:.2f} m")

        # (2) path taken from the heatmaps vs the hard-coded centerline (+ walls)
        ax = axes[r, 1]
        if len(walls):
            ax.scatter(walls[:, 1], walls[:, 0], s=1, c="0.6", label="ducts (map)")
        ax.plot(cl[:, 1], cl[:, 0], "-", color="deepskyblue", lw=3, label="map centerline (hard-coded)")
        ax.plot(gt[:, 1], gt[:, 0], "o-", color="lime", lw=2, ms=4, label="actual path")
        ax.plot(pred[r][:, 1], pred[r][:, 0], "o-", color="red", lw=2, ms=4, label="model path")
        setup_bev(ax); ax.grid(alpha=0.3); ax.legend(loc="lower left", fontsize=7)
        ax.set_title(f"model path vs centerline   off-centerline {e_cl:.2f} m")

        # (3) camera view
        ax = axes[r, 2]
        ax.imshow(ds.imgs[i])
        for pts, col, lab, sty in [(cl, "deepskyblue", "centerline", "-"), (gt, "lime", "actual", "o-"),
                                   (pred[r], "red", "model", "o-")]:
            uv, vis = tb.car_to_image224(pts, ctx["meta"])
            ax.plot(uv[vis, 0], uv[vis, 1], sty, color=col, lw=2.5 if sty == "-" else 2, ms=4, label=lab)
        ax.set_xlim(0, tb.IMG); ax.set_ylim(tb.IMG, 0); ax.axis("off"); ax.legend(loc="upper left", fontsize=7)
        ax.set_title("camera view (only the part inside the image)")
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=100)
    plt.show()


def show_waypoint_heatmaps(model, ds, i, device):
    import matplotlib.pyplot as plt
    _, hms = predict(model, ds, [i], device)
    gt = ds.xy[i]
    ext = [tb.Y_RANGE[1], tb.Y_RANGE[0], tb.X_RANGE[0], tb.X_RANGE[1]]
    K = hms.shape[1]
    fig, axes = plt.subplots(1, K, figsize=(3.2 * K, 3.0))
    for k, ax in enumerate(axes):
        ax.imshow(hms[0, k] / hms[0, k].max(), cmap="inferno", extent=ext, origin="upper")
        ax.plot(gt[k, 1], gt[k, 0], "o", mfc="none", mec="lime", ms=10, mew=2)
        ax.set_xlim(tb.Y_RANGE[1], tb.Y_RANGE[0]); ax.set_ylim(-0.3, tb.X_RANGE[1])
        ax.set_title(f"waypoint {k + 1}  ({1.0 + 0.3 * k:.2f} m)"); ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.show()
