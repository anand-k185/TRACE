# TRACE - ViT + heatmap trajectory model: dataset, model, loss, train/eval, viz.
# Used by the Colab notebook (each section is pasted into a cell there).
import glob, json, math, os, random, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

IMG = 224          # model input (square)
HM = 56            # heatmap size (IMG / 4)

# ------------------------------------------------------------------ data
def load_split(root, maps):
    """All samples from data/<map>/labels.jsonl for the given maps."""
    out = []
    for m in maps:
        p = os.path.join(root, m, "labels.jsonl")
        with open(p) as fh:
            for l in fh:
                s = json.loads(l)
                s["img"] = os.path.join(os.path.dirname(root), s["img"]) if not os.path.isabs(s["img"]) else s["img"]
                s["map"] = m
                out.append(s)
    return out


def episode_of(s):
    return os.path.basename(os.path.dirname(os.path.dirname(s["img"])))


def split_train_val(samples, val_frac=0.15):
    """Val = the last 15% of every episode (time-contiguous, so no near-duplicate leak)."""
    by_ep = {}
    for s in samples:
        by_ep.setdefault((s["map"], episode_of(s)), []).append(s)
    tr, va = [], []
    for k, ss in by_ep.items():
        ss.sort(key=lambda s: s["img"])
        n = int(len(ss) * (1 - val_frac))
        tr += ss[:n]
        va += ss[n:]
    return tr, va


class TrajDataset(torch.utils.data.Dataset):
    """Images pre-loaded into RAM at IMG x IMG (uint8) for speed; targets in IMG pixel coords."""
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __init__(self, samples, train=False):
        self.s, self.train = samples, train
        self.imgs, self.uv, self.valid = [], [], []
        for i, s in enumerate(samples):
            im = Image.open(s["img"]).convert("RGB")
            W, H = im.size
            self.imgs.append(np.asarray(im.resize((IMG, IMG), Image.BILINEAR)))
            uv = np.array(s["uv"], dtype=np.float32)
            uv[:, 0] *= IMG / W
            uv[:, 1] *= IMG / H
            self.uv.append(uv)
            self.valid.append(np.array(s["valid"], dtype=bool))
            if (i + 1) % 2000 == 0:
                print(f"  loaded {i + 1}/{len(samples)}")
        self.imgs = np.stack(self.imgs)

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        x = torch.from_numpy(self.imgs[i]).permute(2, 0, 1).float() / 255.0
        uv = torch.from_numpy(self.uv[i].copy())
        valid = torch.from_numpy(self.valid[i].copy())
        if self.train:
            if random.random() < 0.5:                         # mirror image + path
                x = x.flip(-1)
                uv[:, 0] = IMG - 1 - uv[:, 0]
            b = 1 + (random.random() - 0.5) * 0.5             # brightness / contrast jitter
            c = 1 + (random.random() - 0.5) * 0.5
            m = x.mean()
            x = ((x - m) * c + m) * b
            x = x.clamp(0, 1)
        x = (x - self.MEAN) / self.STD
        return x, uv, valid


# ------------------------------------------------------------------ model
class ViTHeatmap(nn.Module):
    """Pretrained ViT encoder -> patch tokens as a 14x14 feature map -> upsampling conv
    decoder -> one 56x56 heatmap per waypoint."""

    def __init__(self, n_wp=6, backbone="vit_small_patch16_224.augreg_in21k_ft_in1k", pretrained=True):
        super().__init__()
        import timm
        self.enc = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        d = self.enc.num_features
        self.n_prefix = getattr(self.enc, "num_prefix_tokens", 1)
        self.grid = IMG // self.enc.patch_embed.patch_size[0]

        def up(cin, cout):
            return nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                 nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.GELU())
        self.dec = nn.Sequential(
            nn.Conv2d(d, 256, 1), nn.BatchNorm2d(256), nn.GELU(),
            up(256, 128),                                        # 14 -> 28
            up(128, 64),                                         # 28 -> 56
            nn.Conv2d(64, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, n_wp, 1))

    def forward(self, x):
        t = self.enc.forward_features(x)[:, self.n_prefix:]      # (B, N, d) patch tokens
        B, N, d = t.shape
        f = t.transpose(1, 2).reshape(B, d, self.grid, self.grid)
        hm = self.dec(f)
        if hm.shape[-1] != HM:
            hm = F.interpolate(hm, size=(HM, HM), mode="bilinear", align_corners=False)
        return hm                                                 # (B, n_wp, HM, HM) logits


# ------------------------------------------------------------------ loss / decoding
_ys, _xs = torch.meshgrid(torch.arange(HM).float(), torch.arange(HM).float(), indexing="ij")


def to_hm(uv):
    """IMG pixel coords -> heatmap cell coords (cell centers)."""
    return (uv + 0.5) * (HM / IMG) - 0.5


def from_hm(c):
    return (c + 0.5) * (IMG / HM) - 0.5


def soft_argmax(logits):
    B, K, H, W = logits.shape
    p = logits.flatten(2).softmax(-1).view(B, K, H, W)
    xs, ys = _xs.to(logits.device), _ys.to(logits.device)
    return torch.stack([(p * xs).sum((-1, -2)), (p * ys).sum((-1, -2))], -1)   # heatmap coords


def loss_fn(logits, uv, valid, sigma=1.5, w_coord=0.1):
    """Cross-entropy between softmax(heatmap) and a Gaussian at the target, plus L1 on soft-argmax.
    Waypoints outside the image are ignored."""
    c = to_hm(uv)                                                   # (B, K, 2)
    xs, ys = _xs.to(uv.device), _ys.to(uv.device)
    g = torch.exp(-((xs - c[..., 0, None, None]) ** 2 + (ys - c[..., 1, None, None]) ** 2) / (2 * sigma ** 2))
    g = g / g.flatten(2).sum(-1).clamp_min(1e-8)[..., None, None]
    logp = logits.flatten(2).log_softmax(-1).view_as(logits)
    ce = -(g * logp).sum((-1, -2))                                  # (B, K)
    l1 = (soft_argmax(logits) - c).abs().sum(-1)                    # (B, K)
    m = valid.float()
    n = m.sum().clamp_min(1)
    return ((ce + w_coord * l1) * m).sum() / n


# ------------------------------------------------------------------ train / eval
def evaluate(model, loader, device, native_wh=(320, 240)):
    """Mean pixel error per waypoint (in the ORIGINAL camera resolution), visible points only."""
    model.eval()
    sx, sy = native_wh[0] / IMG, native_wh[1] / IMG
    err_sum, cnt = None, None
    with torch.no_grad():
        for x, uv, valid in loader:
            x, uv, valid = x.to(device), uv.to(device), valid.to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x)
            pred = from_hm(soft_argmax(logits.float()))
            d = pred - uv
            e = torch.sqrt((d[..., 0] * sx) ** 2 + (d[..., 1] * sy) ** 2) * valid
            err_sum = e.sum(0) if err_sum is None else err_sum + e.sum(0)
            cnt = valid.sum(0) if cnt is None else cnt + valid.sum(0)
    per_wp = (err_sum / cnt.clamp_min(1)).cpu().numpy()
    return {"ADE_px": float(per_wp.mean()), "FDE_px": float(per_wp[-1]), "per_wp_px": per_wp.round(1).tolist()}


def train(model, tr_loader, va_loader, device, epochs=15, lr_head=1e-3, lr_enc=3e-5,
          freeze_epochs=2, ckpt="trace_vit_heatmap.pt", native_wh=(320, 240)):
    enc_p = list(model.enc.parameters())
    head_p = list(model.dec.parameters())
    opt = torch.optim.AdamW([{"params": enc_p, "lr": lr_enc}, {"params": head_p, "lr": lr_head}], weight_decay=0.05)
    steps = epochs * len(tr_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr_enc, lr_head], total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    best = 1e9
    hist = []
    for ep in range(epochs):
        model.train()
        frozen = ep < freeze_epochs                                  # warm up the decoder first
        for p in enc_p:
            p.requires_grad_(not frozen)
        t0, tot, nb = time.time(), 0.0, 0
        for x, uv, valid in tr_loader:
            x, uv, valid = x.to(device), uv.to(device), valid.to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x)
            loss = loss_fn(logits.float(), uv, valid)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
            nb += 1
        m = evaluate(model, va_loader, device, native_wh)
        hist.append({"epoch": ep + 1, "train_loss": tot / nb, **m})
        flag = ""
        if m["ADE_px"] < best:
            best = m["ADE_px"]
            torch.save(model.state_dict(), ckpt)
            flag = "  (saved)"
        print(f"epoch {ep + 1:2d}/{epochs}  loss {tot / nb:.3f}  val ADE {m['ADE_px']:.1f}px  "
              f"FDE {m['FDE_px']:.1f}px  {'[enc frozen] ' if frozen else ''}{time.time() - t0:.0f}s{flag}")
    return hist


# ------------------------------------------------------------------ viz
def show_predictions(model, ds, device, n=12, cols=4, seed=0, save=None):
    import matplotlib.pyplot as plt
    model.eval()
    rng = random.Random(seed)
    idx = rng.sample(range(len(ds)), min(n, len(ds)))
    rows = math.ceil(len(idx) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, i in zip(axes, idx):
        x, uv, valid = ds[i]
        with torch.no_grad():
            pred = from_hm(soft_argmax(model(x[None].to(device)).float()))[0].cpu()
        ax.imshow(ds.imgs[i])
        v = valid.numpy()
        ax.plot(uv[v, 0], uv[v, 1], "o-", color="lime", lw=2, ms=4, label="ground truth")
        ax.plot(pred[:, 0], pred[:, 1], "o-", color="red", lw=2, ms=4, label="predicted")
        ax.set_xlim(0, IMG); ax.set_ylim(IMG, 0); ax.axis("off")
    for ax in axes[len(idx):]:
        ax.axis("off")
    axes[0].legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=110)
    plt.show()
