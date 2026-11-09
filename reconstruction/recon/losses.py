"""Loss registry and metrics. A loss maps (out, batch, **kw) -> (total, dict of parts)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

LOSSES = {}


def register(name):
    def deco(fn):
        LOSSES[name] = fn
        return fn
    return deco


def soft_dice(p, y, w):
    """per-sample weighted soft Dice loss on the highlight class, mean over the batch"""
    inter = (w * p * y).sum(-1)
    den = (w * p).sum(-1) + (w * y).sum(-1)
    return (1 - (2 * inter + 1) / (den + 1)).mean()


def strand_weight(b, w_occluded):
    vis_any = b["vis"].any(-1).float()
    return w_occluded + (1 - w_occluded) * vis_any


@register("bce_dice")
def bce_dice(out, b, w_occluded=0.3, w_dice=1.0, w_color=2.0, w_scalp=1.0, w_seg=0.5, w_strand=1.0,
             color_target="param", **_):
    """color_target: 'param' = the renderer's input RGB; 'obs' = the median observed color of the class's
    pixels (appearance; masked where the class is not seen)"""
    y = b["labels"]
    w = strand_weight(b, w_occluded)
    lg = out["strand_logit"].float()
    bce = (F.binary_cross_entropy_with_logits(lg, y, reduction="none") * w).sum() / w.sum()
    dice = soft_dice(torch.sigmoid(lg), y, w)
    if color_target == "obs":
        okc = b["colors_obs_ok"][..., None]
        col = ((out["colors"].float() - b["colors_obs"]).abs() * okc).sum() / (okc.sum() * 3).clamp(min=1)
    else:
        col = (out["colors"].float() - b["colors"]).abs().mean()
    parts = dict(bce=bce, dice=dice, color=col)
    total = w_strand * (bce + w_dice * dice) + w_color * col
    if "seg2d" in out:
        sg = out["seg2d"].float()
        t = F.interpolate(b["lab2d"][:, None].float(), size=sg.shape[-2:], mode="nearest")[:, 0].long()
        # hair pixels count more (they are few): class weights none 0.2, body 0.2, base 1, highlight 2
        cw = torch.tensor([0.2, 0.2, 1.0, 2.0], device=sg.device)
        parts["seg"] = F.cross_entropy(sg, t, weight=cw)
        total = total + w_seg * parts["seg"]
    if "scalp_logit" in out:
        t = b["scalp32"]
        ok = torch.isfinite(t)
        sl = out["scalp_logit"].float()
        if sl.shape[-1] != t.shape[-1]:
            sl = F.interpolate(sl[:, None], size=t.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        tt = torch.nan_to_num(t, nan=0.0)
        sb = F.binary_cross_entropy_with_logits(sl[ok], tt[ok])
        sd = soft_dice(torch.sigmoid(sl).flatten(1), tt.flatten(1), ok.float().flatten(1))
        parts.update(scalp_bce=sb, scalp_dice=sd)
        total = total + w_scalp * (sb + w_dice * sd)
    return total, parts


# ---------------------------------------------------------------- metrics (numpy, per sample)

def _lab(rgb):
    c = np.clip(np.asarray(rgb, np.float64), 0, 1)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = lin @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    xyz = xyz / np.array([0.9505, 1.0, 1.089])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def iou(pred, gt):
    u = (pred | gt).sum()
    return float((pred & gt).sum() / u) if u else float("nan")


def sample_metrics(prob, labels, vis_any, colors_pred, colors_gt, scalp_pred=None, scalp_gt=None):
    """prob / labels / vis_any: (S,) for one sample. Metrics:
    iou_all / iou_vis   highlight IoU over all strands / strands with a visible point (threshold 0.5)
    acc_vis             accuracy on visible strands
    de_base / de_hl     CIELAB ΔE76 of the predicted vs true render input colors
    iou_scalp           highlight IoU of the 32x32 root-mean mask (cells with roots in both)"""
    p = prob >= 0.5
    y = labels >= 0.5
    m = dict(iou_all=iou(p, y), iou_vis=iou(p[vis_any], y[vis_any]),
             acc_vis=float((p[vis_any] == y[vis_any]).mean()) if vis_any.any() else float("nan"))
    d = np.linalg.norm(_lab(colors_pred) - _lab(colors_gt), axis=-1)
    m.update(de_base=float(d[0]), de_hl=float(d[1]))
    if scalp_pred is not None and scalp_gt is not None:
        ok = np.isfinite(scalp_pred) & np.isfinite(scalp_gt)
        m["iou_scalp"] = iou(scalp_pred[ok] >= 0.5, scalp_gt[ok] >= 0.5)
    return m
