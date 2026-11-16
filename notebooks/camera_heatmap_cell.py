# Camera-view heatmaps: project the model's top-down heatmaps into the camera image.
# Run after the "Load model + test map" cell of TRACE_inspect_bev.ipynb (uses model, ds, samples, device, OUT, MAP).
import numpy as np, torch, torch.nn.functional as F, matplotlib.pyplot as plt

META = tb.episode_meta(samples[0])

def camera_lookup(meta, size=tb.IMG):
    """For each pixel of the 224x224 model image: where its line of sight hits the floor, as
    grid_sample coords into the top-down heatmap, plus a mask (False = above horizon / outside grid)."""
    K_inv = np.linalg.inv(np.asarray(meta["K"], float))
    T = np.asarray(meta["T_base_cam"], float); R, c = T[:3, :3], T[:3, 3]
    W, H = meta["width"], meta["height"]
    U, V = np.meshgrid((np.arange(size) + 0.5) * W / size, (np.arange(size) + 0.5) * H / size)
    d = (R @ (K_inv @ np.stack([U.ravel(), V.ravel(), np.ones(U.size)]))).T     # ray directions, car frame
    down = d[:, 2] < -1e-6
    t = np.where(down, -c[2] / np.where(down, d[:, 2], -1.0), 0.0)               # hit the floor (z = 0)
    x, y = c[0] + t * d[:, 0], c[1] + t * d[:, 1]
    col = (tb.Y_RANGE[1] - y) / tb.CELL                                          # 0..BW across the grid
    row = (tb.X_RANGE[1] - x) / tb.CELL                                          # 0..BH down the grid
    ok = down & (col >= 0) & (col < tb.BW) & (row >= 0) & (row < tb.BH)
    g = np.stack([col / tb.BW * 2 - 1, row / tb.BH * 2 - 1], -1)
    g[~ok] = -2.0
    return (torch.tensor(g.reshape(size, size, 2), dtype=torch.float32)[None],
            torch.tensor(ok.reshape(size, size)))

GRID, ON_FLOOR = camera_lookup(META)

@torch.no_grad()
def camera_heatmaps(idx):
    """-> per-waypoint heatmaps in camera view (n, K, 224, 224), top-down probs, predicted waypoints."""
    pred, prob = tr.predict(model, ds, idx, device)
    p = torch.from_numpy(prob)
    cam = F.grid_sample(p, GRID.expand(len(idx), -1, -1, -1), align_corners=False, padding_mode="zeros")
    cam = cam * ON_FLOOR
    return cam.numpy(), prob, pred

# ------------------------------------------------ data: how much of each waypoint's heatmap the camera can see
@torch.no_grad()
def visible_mass_table():
    vis_bev = model.vf[0, 0].cpu().numpy()                      # top-down cells inside the camera view
    _, prob = tr.predict(model, ds, list(range(len(samples))), device)
    frac = (prob * vis_bev).sum((-1, -2))                       # (N, K) share of probability the camera sees
    gt = np.array([s["ego_xy"] for s in samples])
    curve = np.abs(gt[:, -1, 1]) > 0.8
    cols = {"all frames": frac.mean(0), "straight frames": frac[~curve].mean(0), "curve frames": frac[curve].mean(0)}
    return pd.DataFrame({k: (v.astype(float) * 100).round(1) for k, v in cols.items()},
                        index=[f"wp{k+1} ({1.0+0.3*k:.2f} m)" for k in range(frac.shape[1])])

print(f"{MAP}: % of each waypoint's heatmap that lands inside the camera image")
display(visible_mass_table())
print(f"(top-down grid cells the camera can see at all: {model.vf.mean().item():.0%})")

# ------------------------------------------------ pictures
def show_camera_heatmaps(picks, save=None):
    cam, prob, pred = camera_heatmaps(picks)
    ext = [tb.Y_RANGE[1], tb.Y_RANGE[0], tb.X_RANGE[0], tb.X_RANGE[1]]
    vis_bev = model.vf[0, 0].cpu().numpy()
    fig, axes = plt.subplots(len(picks), 2, figsize=(11, 4.6 * len(picks)), gridspec_kw={"width_ratios": [1, 1.2]})
    axes = np.atleast_2d(axes)
    for r, i in enumerate(picks):
        gt = ds.xy[i]
        seen = (prob[r] * vis_bev).sum((-1, -2))                # per-waypoint visible share
        # camera view
        ax = axes[r, 0]
        h = cam[r].max(0)
        ax.imshow(ds.imgs[i])
        ax.imshow(np.ma.masked_where(h < 0.02 * max(h.max(), 1e-9), h / max(h.max(), 1e-9)),
                  cmap="inferno", alpha=0.65, vmin=0, vmax=1)
        for pts, colr, lab in [(gt, "lime", "actual"), (pred[r], "cyan", "predicted")]:
            uv, v = tb.car_to_image224(pts, META)
            ax.plot(uv[v, 0], uv[v, 1], "o", mfc="none", mec=colr, ms=8, mew=2, label=lab)
        ax.set_xlim(0, tb.IMG); ax.set_ylim(tb.IMG, 0); ax.axis("off"); ax.legend(loc="upper left", fontsize=8)
        ax.set_title(f"camera view of the heatmaps\nvisible share per waypoint: " +
                     " ".join(f"{100*s:.0f}%" for s in seen), fontsize=9)
        # top-down, with the camera's field of view outlined
        ax = axes[r, 1]
        hb = prob[r].max(0)
        ax.imshow(hb / hb.max(), cmap="inferno", extent=ext, origin="upper")
        ax.contour(np.flipud(vis_bev), levels=[0.5], colors="w", linewidths=1, linestyles="--",
                   extent=[tb.Y_RANGE[1], tb.Y_RANGE[0], tb.X_RANGE[0], tb.X_RANGE[1]])
        ax.plot(gt[:, 1], gt[:, 0], "o", mfc="none", mec="lime", ms=8, mew=2)
        ax.plot(pred[r][:, 1], pred[r][:, 0], "x", color="cyan", ms=7, mew=2)
        ax.plot([0], [0], marker=(3, 0, 0), color="w", ms=12)
        ax.set_xlim(tb.Y_RANGE[1], tb.Y_RANGE[0]); ax.set_ylim(-0.2, tb.X_RANGE[1]); ax.set_aspect("equal")
        ax.set_xlabel("<- left   lateral (m)   right ->"); ax.set_ylabel("forward (m)")
        ax.set_title("same heatmaps top-down (dashed = what the camera sees)", fontsize=9)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=100)
    plt.show()

def show_waypoints_in_camera(i):
    """The 6 individual waypoint heatmaps, each projected into the camera view."""
    cam, prob, _ = camera_heatmaps([i])
    vis_bev = model.vf[0, 0].cpu().numpy()
    gt = ds.xy[i]
    uv, v = tb.car_to_image224(gt, META)
    fig, axes = plt.subplots(1, cam.shape[1], figsize=(3.1 * cam.shape[1], 3.3))
    for k, ax in enumerate(axes):
        h = cam[0, k]
        ax.imshow(ds.imgs[i])
        if h.max() > 0:
            ax.imshow(np.ma.masked_where(h < 0.02 * h.max(), h / h.max()), cmap="inferno", alpha=0.7, vmin=0, vmax=1)
        if v[k]:
            ax.plot(uv[k, 0], uv[k, 1], "o", mfc="none", mec="lime", ms=10, mew=2)
        share = (prob[0, k] * vis_bev).sum()
        ax.set_title(f"wp{k+1} ({1.0+0.3*k:.2f} m): {100*share:.0f}% visible", fontsize=9)
        ax.set_xlim(0, tb.IMG); ax.set_ylim(tb.IMG, 0); ax.axis("off")
    plt.tight_layout(); plt.show()

gt_all = np.array([s["ego_xy"] for s in samples]); curv = np.abs(gt_all[:, -1, 1])
picks = [int(np.argmin(curv))] + [int(i) for i in np.argsort(-curv)[[0, 25, 60]]]   # 1 straight + 3 bends
show_camera_heatmaps(picks, save=f"{OUT}/camera_heatmaps_{MAP}.png")
show_waypoints_in_camera(picks[1])
