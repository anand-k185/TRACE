# TRACE - compare model output against the ground-truth path and the hard-coded map centerline.
# Needs: data/<map>/ep_*/{meta.json,frames.jsonl} (from gz_collect.py) and the repo's maps/<map>.yaml.
import json, math, os
import numpy as np
import torch
import torch.nn.functional as F

import trace_core as tc

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


# ------------------------------------------------------------ model helpers
@torch.no_grad()
def predict(model, ds, idx, device, bs=64):
    """Returns pred uv in IMG (224) coords and softmaxed heatmaps (n, K, HM, HM)."""
    model.eval()
    uvs, hms = [], []
    for b in range(0, len(idx), bs):
        x = torch.stack([ds[i][0] for i in idx[b:b + bs]]).to(device)
        logits = model(x).float()
        uvs.append(tc.from_hm(tc.soft_argmax(logits)).cpu())
        B, K, H, W = logits.shape
        hms.append(logits.flatten(2).softmax(-1).view(B, K, H, W).cpu())
    return torch.cat(uvs).numpy(), torch.cat(hms).numpy()


def to_native(uv224, meta):
    return uv224 * [meta["width"] / tc.IMG, meta["height"] / tc.IMG]


def to_224(uv_native, meta):
    return np.asarray(uv_native) * [tc.IMG / meta["width"], tc.IMG / meta["height"]]


# ------------------------------------------------------------ metrics over a whole split
def metrics_m(model, ds, samples, eps, device):
    """Ground-plane errors in meters: prediction vs actual path, and both vs the map centerline."""
    idx = list(range(len(samples)))
    pred224, _ = predict(model, ds, idx, device)
    e_gt, d_pred_c, d_gt_c = [], [], []
    for k, s in enumerate(samples):
        ctx = eps.context(s)
        pred_car = pixels_to_car(to_native(pred224[k], ctx["meta"]), ctx["pose"], ctx["meta"])
        gt_car = np.asarray(s["ego_xy"])
        cl = centerline_ahead(ctx, 0.0, 5.0)
        e_gt.append(np.linalg.norm(pred_car - gt_car, axis=1))
        d_pred_c.append(dist_to_polyline(pred_car, cl))
        d_gt_c.append(dist_to_polyline(gt_car, cl))
    e_gt, d_pred_c, d_gt_c = map(np.array, (e_gt, d_pred_c, d_gt_c))
    return {
        "pred_vs_actual_m (mean)": np.nanmean(e_gt, 0),
        "pred_vs_actual_m (median)": np.nanmedian(e_gt, 0),
        "pred_off_centerline_m (mean)": np.nanmean(d_pred_c, 0),
        "pred_off_centerline_m (median)": np.nanmedian(d_pred_c, 0),
        "actual_off_centerline_m (mean)": np.nanmean(d_gt_c, 0),
        "worst_frames": list(np.argsort(-np.nan_to_num(e_gt.mean(1), nan=99))[:6]),
        "n": len(samples),
    }


# ------------------------------------------------------------ visualisation
def show_panels(model, ds, samples, eps, device, picks, save=None):
    """One row per sample: heatmap overlay | paths in image | bird's-eye (meters)."""
    import matplotlib.pyplot as plt
    pred224, hms = predict(model, ds, picks, device)
    fig, axes = plt.subplots(len(picks), 3, figsize=(15, 4.6 * len(picks)))
    axes = np.atleast_2d(axes)
    for r, i in enumerate(picks):
        s, ctx = samples[i], eps.context(samples[i])
        meta, pose = ctx["meta"], ctx["pose"]
        img = ds.imgs[i]
        gt224 = np.asarray(ds.uv[i]); gv = np.asarray(ds.valid[i])
        cl_car = centerline_ahead(ctx)
        cl_uv, cl_vis = car_to_pixels(cl_car, pose, meta)
        cl224 = to_224(cl_uv, meta)
        pred_car = pixels_to_car(to_native(pred224[r], meta), pose, meta)
        gt_car = np.asarray(s["ego_xy"])

        # (1) heatmaps: max over the 6 waypoints, upsampled onto the image
        ax = axes[r, 0]
        hm = torch.from_numpy(hms[r].max(0))[None, None]
        hm = F.interpolate(hm, size=(tc.IMG, tc.IMG), mode="bilinear", align_corners=False)[0, 0].numpy()
        ax.imshow(img)
        ax.imshow(hm / hm.max(), cmap="inferno", alpha=0.55)
        ax.plot(gt224[gv, 0], gt224[gv, 1], "o", mfc="none", mec="lime", ms=8, mew=2, label="actual waypoints")
        ax.set_title("model heatmaps (all 6 waypoints)"); ax.axis("off"); ax.legend(loc="upper left", fontsize=8)

        # (2) paths in the image
        ax = axes[r, 1]
        ax.imshow(img)
        ax.plot(cl224[cl_vis, 0], cl224[cl_vis, 1], "-", color="deepskyblue", lw=3, alpha=0.9, label="map centerline (hard-coded)")
        ax.plot(gt224[gv, 0], gt224[gv, 1], "o-", color="lime", lw=2, ms=4, label="actual path driven")
        ax.plot(pred224[r][:, 0], pred224[r][:, 1], "o-", color="red", lw=2, ms=4, label="model prediction")
        ax.set_xlim(0, tc.IMG); ax.set_ylim(tc.IMG, 0); ax.axis("off")
        ax.set_title("paths in camera view"); ax.legend(loc="upper left", fontsize=8)

        # (3) bird's-eye in meters (car at origin, forward = up, left = left)
        ax = axes[r, 2]
        wx, wy = ctx["field"].walls_near(pose["x"], pose["y"])
        w_car = world_to_car(np.stack([wx, wy], 1), pose) if len(wx) else np.zeros((0, 2))
        ax.scatter(-w_car[:, 1], w_car[:, 0], s=1, c="0.6", label="ducts (map)")
        ax.plot(-cl_car[:, 1], cl_car[:, 0], "-", color="deepskyblue", lw=3, label="map centerline")
        ax.plot(-gt_car[:, 1], gt_car[:, 0], "o-", color="lime", lw=2, ms=4, label="actual path")
        ax.plot(-pred_car[:, 1], pred_car[:, 0], "o-", color="red", lw=2, ms=4, label="prediction")
        ax.plot([0], [0], marker=(3, 0, 0), color="k", ms=12)
        e = np.nanmean(np.linalg.norm(pred_car - gt_car, axis=1))
        dc = np.nanmean(dist_to_polyline(pred_car, cl_car))
        ax.set_title(f"top-down (m): err vs actual {e:.2f} m, off centerline {dc:.2f} m")
        ax.set_xlim(-3, 3); ax.set_ylim(-0.5, 4.5); ax.set_aspect("equal")
        ax.set_xlabel("← left   (m)   right →"); ax.set_ylabel("forward (m)")
        ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=7)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=100)
    plt.show()


def show_waypoint_heatmaps(model, ds, i, device):
    """The 6 individual heatmaps for one frame."""
    import matplotlib.pyplot as plt
    _, hms = predict(model, ds, [i], device)
    gt = np.asarray(ds.uv[i]); gv = np.asarray(ds.valid[i])
    fig, axes = plt.subplots(1, hms.shape[1], figsize=(3 * hms.shape[1], 3.2))
    for k, ax in enumerate(axes):
        h = F.interpolate(torch.from_numpy(hms[0, k])[None, None], size=(tc.IMG, tc.IMG),
                          mode="bilinear", align_corners=False)[0, 0].numpy()
        ax.imshow(ds.imgs[i]); ax.imshow(h / h.max(), cmap="inferno", alpha=0.6)
        if gv[k]:
            ax.plot(gt[k, 0], gt[k, 1], "o", mfc="none", mec="lime", ms=10, mew=2)
        ax.set_title(f"waypoint {k + 1}"); ax.axis("off")
    plt.tight_layout(); plt.show()
