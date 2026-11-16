# TRACE v2 - camera image -> TOP-DOWN (bird's-eye) waypoint heatmaps.
#
# Why: in v1 the heatmaps lived in image pixels, so on tight bends the far waypoints fell outside
# the picture, got masked out of the loss, and were never learned. Here every waypoint lives on a
# floor grid around the car (meters), so it is supervised on every frame, curves included.
#
# Encoder: pretrained ViT (14x14 patch tokens). Decoder: floor queries initialised from the image
# via the known camera geometry (IPM), refined by a transformer decoder that cross-attends to all
# image tokens, then a conv head outputs one heatmap per waypoint. Prediction = soft-argmax, meters.
import json, math, os, random, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

IMG = 224
# floor grid in the car frame: x forward, y left (meters)
X_RANGE = (0.0, 4.0)
Y_RANGE = (-3.2, 3.2)
CELL = 0.08
BH = int(round((X_RANGE[1] - X_RANGE[0]) / CELL))     # 50 rows   (row 0 = far, last row = at the car)
BW = int(round((Y_RANGE[1] - Y_RANGE[0]) / CELL))     # 80 cols   (col 0 = left, last col = right)


def xy_to_cell(xy):
    """car-frame meters (..., 2) -> continuous (col, row) cell coords."""
    col = (Y_RANGE[1] - xy[..., 1]) / CELL - 0.5
    row = (X_RANGE[1] - xy[..., 0]) / CELL - 0.5
    return torch.stack([col, row], -1) if torch.is_tensor(xy) else np.stack([col, row], -1)


def cell_to_xy(cr):
    x = X_RANGE[1] - (cr[..., 1] + 0.5) * CELL
    y = Y_RANGE[1] - (cr[..., 0] + 0.5) * CELL
    return torch.stack([x, y], -1) if torch.is_tensor(cr) else np.stack([x, y], -1)


def in_grid(xy):
    return ((xy[..., 0] > X_RANGE[0]) & (xy[..., 0] < X_RANGE[1])
            & (xy[..., 1] > Y_RANGE[0]) & (xy[..., 1] < Y_RANGE[1]))


# ------------------------------------------------------------------ data
def load_split(root, maps):
    out = []
    for m in maps:
        with open(os.path.join(root, m, "labels.jsonl")) as fh:
            for l in fh:
                s = json.loads(l)
                if not os.path.isabs(s["img"]):
                    s["img"] = os.path.join(os.path.dirname(root), s["img"])
                s["map"] = m
                out.append(s)
    return out


def split_train_val(samples, val_frac=0.15):
    """Val = last 15% of every episode (time-contiguous -> no near-duplicate leakage)."""
    by_ep = {}
    for s in samples:
        by_ep.setdefault((s["map"], os.path.dirname(os.path.dirname(s["img"]))), []).append(s)
    tr, va = [], []
    for ss in by_ep.values():
        ss.sort(key=lambda s: s["img"])
        n = int(len(ss) * (1 - val_frac))
        tr += ss[:n]
        va += ss[n:]
    return tr, va


def curvature(s):
    """How much the path bends: lateral offset (m) of the farthest waypoint."""
    return abs(s["ego_xy"][-1][1])


def curve_weights(samples, boost=3.0, full_at=1.5):
    """Sampling weight 1 on straights up to 1+boost on sharp bends."""
    return [1.0 + boost * min(curvature(s) / full_at, 1.0) for s in samples]


class TrajDataset(torch.utils.data.Dataset):
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __init__(self, samples, train=False):
        self.s, self.train = samples, train
        imgs = []
        for i, s in enumerate(samples):
            imgs.append(np.asarray(Image.open(s["img"]).convert("RGB").resize((IMG, IMG), Image.BILINEAR)))
            if (i + 1) % 2000 == 0:
                print(f"  loaded {i + 1}/{len(samples)}")
        self.imgs = np.stack(imgs)
        self.xy = np.array([s["ego_xy"] for s in samples], dtype=np.float32)        # (N, K, 2) meters

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        x = torch.from_numpy(self.imgs[i]).permute(2, 0, 1).float() / 255.0
        xy = torch.from_numpy(self.xy[i].copy())
        if self.train:
            if random.random() < 0.5:                 # mirror the image -> mirror the path (y -> -y)
                x = x.flip(-1)
                xy[:, 1] = -xy[:, 1]
            b = 1 + (random.random() - 0.5) * 0.5
            c = 1 + (random.random() - 0.5) * 0.5
            m = x.mean()
            x = (((x - m) * c + m) * b).clamp(0, 1)
        x = (x - self.MEAN) / self.STD
        return x, xy, in_grid(xy)


def make_loaders(ds_tr, ds_va, ds_te=None, bs=64, workers=2, curve_boost=3.0):
    w = torch.tensor(curve_weights(ds_tr.s, boost=curve_boost), dtype=torch.double)
    sampler = torch.utils.data.WeightedRandomSampler(w, num_samples=len(w), replacement=True)
    mk = lambda d, **kw: torch.utils.data.DataLoader(d, batch_size=bs, num_workers=workers, pin_memory=True, **kw)
    out = [mk(ds_tr, sampler=sampler, drop_last=True), mk(ds_va, shuffle=False)]
    if ds_te is not None:
        out.append(mk(ds_te, shuffle=False))
    return out


# ------------------------------------------------------------------ model
def ipm_grid(meta, rows, cols):
    """For a rows x cols floor grid (same extent as the BEV), where each cell's center lands in the
    camera image: normalized grid_sample coords in [-1, 1] and a visible mask. Uses the fixed camera
    intrinsics/mount from meta.json, assuming a flat floor."""
    K = np.asarray(meta["K"], dtype=np.float64)
    T_cam_base = np.linalg.inv(np.asarray(meta["T_base_cam"], dtype=np.float64))
    W, H = meta["width"], meta["height"]
    xs = X_RANGE[1] - (np.arange(rows) + 0.5) * (X_RANGE[1] - X_RANGE[0]) / rows
    ys = Y_RANGE[1] - (np.arange(cols) + 0.5) * (Y_RANGE[1] - Y_RANGE[0]) / cols
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size), np.ones(X.size)])
    pc = (T_cam_base @ P)[:3]
    uvw = K @ pc
    u, v = uvw[0] / np.clip(uvw[2], 1e-6, None), uvw[1] / np.clip(uvw[2], 1e-6, None)
    vis = (pc[2] > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    g = np.stack([u / W * 2 - 1, v / H * 2 - 1], -1).reshape(rows, cols, 2)
    g[~vis.reshape(rows, cols)] = 0.0
    return torch.tensor(g, dtype=torch.float32), torch.tensor(vis.reshape(rows, cols), dtype=torch.float32)


class ViTBEV(nn.Module):
    """ViT encoder -> top-down waypoint heatmaps.
    1. Image patch tokens are projected to d channels (a 14x14 feature map).
    2. Geometry prior (inverse perspective mapping): every floor cell starts with the image feature
       found where the camera sees that cell; cells the camera can't see get a learned "unseen" vector.
    3. A transformer decoder (self-attention between floor queries + cross-attention to all image
       tokens) adds global context, which is how off-screen parts of a bend get filled in.
    4. A conv head combines the decoded queries with full-resolution IPM features and outputs one
       heatmap per waypoint."""

    def __init__(self, meta, n_wp=6, backbone="vit_small_patch16_224.augreg_in21k_ft_in1k", pretrained=True,
                 d=256, q_hw=(20, 32), layers=3, heads=8):
        super().__init__()
        import timm
        self.enc = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.n_prefix = getattr(self.enc, "num_prefix_tokens", 1)
        self.grid_tok = IMG // self.enc.patch_embed.patch_size[0]
        self.mem_proj = nn.Sequential(nn.Linear(self.enc.num_features, d), nn.LayerNorm(d))
        self.q_hw = q_hw
        gq, vq = ipm_grid(meta, *q_hw)
        gf, vf = ipm_grid(meta, BH, BW)
        self.register_buffer("gq", gq[None]); self.register_buffer("vq", vq[None, None])
        self.register_buffer("gf", gf[None]); self.register_buffer("vf", vf[None, None])
        self.q_pos = nn.Parameter(torch.randn(1, d, *q_hw) * 0.02)
        self.unseen = nn.Parameter(torch.zeros(1, d, 1, 1))
        layer = nn.TransformerDecoderLayer(d, heads, dim_feedforward=2 * d, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(layer, layers)

        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.GELU())
        self.q_head = block(d, 96)
        self.ipm_head = block(d + 1, 64)
        self.head = nn.Sequential(block(96 + 64, 96), block(96, 64), nn.Conv2d(64, n_wp, 1))

    def forward(self, x):
        B = x.shape[0]
        tok = self.mem_proj(self.enc.forward_features(x)[:, self.n_prefix:])       # (B, N, d)
        fmap = tok.transpose(1, 2).reshape(B, -1, self.grid_tok, self.grid_tok)    # (B, d, 14, 14)
        # floor queries initialised from the image (IPM) or "unseen"
        q = F.grid_sample(fmap, self.gq.expand(B, -1, -1, -1), align_corners=False)
        q = q * self.vq + self.unseen * (1 - self.vq) + self.q_pos                  # (B, d, qh, qw)
        q = self.dec(q.flatten(2).transpose(1, 2), tok)                             # global context
        q = q.transpose(1, 2).reshape(B, -1, *self.q_hw)
        q = F.interpolate(self.q_head(q), size=(BH, BW), mode="bilinear", align_corners=False)
        # full-resolution IPM features for precise placement where the camera can see
        f = F.grid_sample(fmap, self.gf.expand(B, -1, -1, -1), align_corners=False) * self.vf
        f = self.ipm_head(torch.cat([f, self.vf.expand(B, -1, -1, -1)], 1))
        return self.head(torch.cat([q, f], 1))                                       # (B, n_wp, BH, BW)


# ------------------------------------------------------------------ loss / decoding
def _grid(device):
    rows, cols = torch.meshgrid(torch.arange(BH, device=device).float(),
                                torch.arange(BW, device=device).float(), indexing="ij")
    return rows, cols


def soft_argmax_xy(logits):
    """BEV logits -> expected waypoint position in meters (B, K, 2)."""
    B, K, H, W = logits.shape
    p = logits.flatten(2).softmax(-1).view(B, K, H, W)
    rows, cols = _grid(logits.device)
    cr = torch.stack([(p * cols).sum((-1, -2)), (p * rows).sum((-1, -2))], -1)
    return cell_to_xy(cr)


def peak_xy(logits, win=3):
    """Decode at the STRONGEST peak: argmax cell, then soft-argmax only within +-win cells of it.
    Robust when a heatmap has two blobs (soft-argmax over the whole map would average them)."""
    B, K, H, W = logits.shape
    flat = logits.flatten(2)
    idx = flat.argmax(-1)
    pr, pc = (idx // W).float(), (idx % W).float()
    rows, cols = _grid(logits.device)
    near = ((rows - pr[..., None, None]).abs() <= win) & ((cols - pc[..., None, None]).abs() <= win)
    p = flat.masked_fill(~near.flatten(2), float("-inf")).softmax(-1).view(B, K, H, W)
    cr = torch.stack([(p * cols).sum((-1, -2)), (p * rows).sum((-1, -2))], -1)
    return cell_to_xy(cr)


def chained_xy(logits, spacing=0.45, slack=0.35, win=3):
    """Decode waypoints in order, each only from the part of its heatmap within reach of the previous
    one (waypoints are `spacing` m apart along the path; the first is searched near the car).
    Stops a far waypoint from jumping to a spurious blob on the other side of the corridor."""
    B, K, H, W = logits.shape
    rows, cols = _grid(logits.device)
    cell_xy = cell_to_xy(torch.stack([cols, rows], -1))                   # (H, W, 2) meters
    out = []
    prev = torch.zeros(B, 2, device=logits.device)
    reach = 1.2 + slack                                                    # first waypoint ~1.2 m ahead
    for k in range(K):
        d = (cell_xy[None] - prev[:, None, None, :]).norm(dim=-1)        # (B, H, W)
        ok = d <= reach
        lk = logits[:, k].masked_fill(~ok, float("-inf"))
        lk = torch.where(ok.flatten(1).any(-1)[:, None, None], lk, logits[:, k])
        xy = peak_xy(lk[:, None], win=win)[:, 0]
        out.append(xy)
        prev = xy
        reach = spacing + slack
    return torch.stack(out, 1)


def decode(logits):
    """What the model 'predicts' at inference time: path-consistent peak decoding."""
    return chained_xy(logits)


def loss_fn(logits, xy, valid, sigma=1.5, w_m=1.0):
    """CE between softmax(heatmap) and a Gaussian at the target cell + L1 in meters."""
    cr = xy_to_cell(xy)
    rows, cols = _grid(xy.device)
    g = torch.exp(-((cols - cr[..., 0, None, None]) ** 2 + (rows - cr[..., 1, None, None]) ** 2) / (2 * sigma ** 2))
    g = g / g.flatten(2).sum(-1).clamp_min(1e-8)[..., None, None]
    logp = logits.flatten(2).log_softmax(-1).view_as(logits)
    ce = -(g * logp).sum((-1, -2))
    l1 = (soft_argmax_xy(logits) - xy).abs().sum(-1)
    m = valid.float()
    return ((ce + w_m * l1) * m).sum() / m.sum().clamp_min(1)


# ------------------------------------------------------------------ eval
@torch.no_grad()
def evaluate(model, loader, device, curve_thresh=0.8):
    """Waypoint error in meters, overall and split into straight vs curve frames."""
    model.eval()
    errs, curv = [], []
    for x, xy, valid in loader:
        x, xy = x.to(device), xy.to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(x)
        pred = decode(logits.float())
        errs.append((pred - xy).norm(dim=-1).cpu())
        curv.append(xy[:, -1, 1].abs().cpu())
    e, c = torch.cat(errs), torch.cat(curv)
    cm = c > curve_thresh
    out = {"ADE_m": e.mean().item(), "FDE_m": e[:, -1].mean().item(),
           "per_wp_m": [round(float(v), 3) for v in e.mean(0)],
           "curve_ADE_m": e[cm].mean().item() if cm.any() else float("nan"),
           "straight_ADE_m": e[~cm].mean().item() if (~cm).any() else float("nan"),
           "curve_frac": cm.float().mean().item()}
    return out


def train(model, tr_loader, va_loader, device, epochs=20, lr_head=5e-4, lr_enc=3e-5,
          freeze_epochs=2, ckpt="trace_vit_bev.pt"):
    enc_p = list(model.enc.parameters())
    head_p = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
    opt = torch.optim.AdamW([{"params": enc_p, "lr": lr_enc}, {"params": head_p, "lr": lr_head}], weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr_enc, lr_head],
                                                total_steps=epochs * len(tr_loader), pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    best, hist = 1e9, []
    for ep in range(epochs):
        model.train()
        frozen = ep < freeze_epochs
        for p in enc_p:
            p.requires_grad_(not frozen)
        t0, tot, nb = time.time(), 0.0, 0
        for x, xy, valid in tr_loader:
            x, xy, valid = x.to(device), xy.to(device), valid.to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x)
            loss = loss_fn(logits.float(), xy, valid)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
            nb += 1
        m = evaluate(model, va_loader, device)
        hist.append({"epoch": ep + 1, "train_loss": tot / nb, **m})
        flag = ""
        if m["ADE_m"] < best:
            best = m["ADE_m"]
            torch.save(model.state_dict(), ckpt)
            flag = "  (saved)"
        print(f"epoch {ep + 1:2d}/{epochs}  loss {tot / nb:.3f}  val ADE {m['ADE_m']:.3f} m  "
              f"(curves {m['curve_ADE_m']:.3f} / straights {m['straight_ADE_m']:.3f})  "
              f"{'[enc frozen] ' if frozen else ''}{time.time() - t0:.0f}s{flag}")
    return hist


# ------------------------------------------------------------------ camera projection (for drawing BEV points on the image)
def car_to_image224(xy, meta):
    """car-frame floor points (N,2) -> pixel coords in the 224x224 model image (+ visibility)."""
    K = np.asarray(meta["K"])
    T_cam_base = np.linalg.inv(np.asarray(meta["T_base_cam"]))
    xy = np.asarray(xy, dtype=float)
    n = len(xy)
    pc = (T_cam_base @ np.stack([xy[:, 0], xy[:, 1], np.zeros(n), np.ones(n)]))[:3]
    uvw = K @ pc
    uv = (uvw[:2] / np.clip(uvw[2], 1e-6, None)).T
    W, H = meta["width"], meta["height"]
    vis = (pc[2] > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    return uv * [IMG / W, IMG / H], vis


def episode_meta(sample):
    return json.load(open(os.path.join(os.path.dirname(os.path.dirname(sample["img"])), "meta.json")))


# ------------------------------------------------------------------ quick viz (training notebook)
@torch.no_grad()
def show_predictions(model, ds, device, picks, save=None):
    import matplotlib.pyplot as plt
    model.eval()
    meta = episode_meta(ds.s[0])
    x = torch.stack([ds[i][0] for i in picks]).to(device)
    logits = model(x).float()
    pred = decode(logits).cpu().numpy()
    prob = logits.flatten(2).softmax(-1).view_as(logits).amax(1).cpu().numpy()
    fig, axes = plt.subplots(len(picks), 2, figsize=(10, 4.2 * len(picks)))
    axes = np.atleast_2d(axes)
    ext = [Y_RANGE[1], Y_RANGE[0], X_RANGE[0], X_RANGE[1]]        # left on the left, forward up
    for r, i in enumerate(picks):
        gt = ds.xy[i]
        ax = axes[r, 0]
        ax.imshow(ds.imgs[i])
        for pts, col, lab in [(gt, "lime", "actual"), (pred[r], "red", "predicted")]:
            uv, vis = car_to_image224(pts, meta)
            ax.plot(uv[vis, 0], uv[vis, 1], "o-", color=col, lw=2, ms=4, label=lab)
        ax.set_xlim(0, IMG); ax.set_ylim(IMG, 0); ax.axis("off"); ax.legend(loc="upper left", fontsize=8)
        ax.set_title("camera view (points inside the image only)")
        ax = axes[r, 1]
        ax.imshow(prob[r] / prob[r].max(), cmap="inferno", extent=ext, origin="upper")
        ax.plot(gt[:, 1], gt[:, 0], "o-", color="lime", lw=2, ms=4)
        ax.plot(pred[r][:, 1], pred[r][:, 0], "o-", color="red", lw=2, ms=4)
        ax.plot([0], [0], marker=(3, 0, 0), color="w", ms=12)
        ax.set_xlim(Y_RANGE[1], Y_RANGE[0]); ax.set_ylim(-0.2, X_RANGE[1])
        ax.set_xlabel("← left   lateral (m)   right →"); ax.set_ylabel("forward (m)")
        err = np.linalg.norm(pred[r] - gt, axis=1).mean()
        ax.set_title(f"top-down heatmaps   err {err:.2f} m")
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=100)
    plt.show()
