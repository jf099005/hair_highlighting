"""DiffLocks HDiT hidden state as a scalp-space image feature (difflocks env):

    ~/anaconda3/envs/difflocks/bin/python tools/hdit_features.py --out data/hdit [--synth_box data/synth_crop_box.json] \
        --list <json list of [key, image_path]>   |   <img> ...

One denoiser forward at sigma SIGMA from fixed noise, conditioned on the image's DINOv2 tokens (as DiffLocks does);
a forward hook keeps the mid level (32 x 32 tokens, global attention, width 3072) = image-conditioned features laid out
on the scalp texture. A fixed random orthogonal projection (seed 0) cuts 3072 -> 256 channels. Saved as
<out>/<key>.npy, float16 (32, 32, 256), texture layout: row = (1 - mesh v), col = mesh u (checked: DiffLocks
root_uv01 (row, col) vs nearest-scalp-vertex mesh UV, corr 0.999).
Real photos: the crop box comes from <difflocks_dir>/crop.json when the key starts with "real__".
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

NCKU = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "NCKU_3D_hair_reconstruction"))
SIGMA = 20.0
DIM = 256


def crop770(frame, box):
    x1, y1, size = box
    H, W = frame.shape[:2]
    canvas = np.zeros((size, size, 3), np.uint8)
    sx0, sy0, sx1, sy1 = max(x1, 0), max(y1, 0), min(x1 + size, W), min(y1 + size, H)
    canvas[sy0 - y1:sy1 - y1, sx0 - x1:sx1 - x1] = frame[sy0:sy1, sx0:sx1]
    return cv2.resize(canvas, (770, 770), interpolation=cv2.INTER_AREA if size > 770 else cv2.INTER_CUBIC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*")
    ap.add_argument("--list", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--synth_box", default=None)
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    jobs = json.load(open(a.list)) if a.list else [[os.path.basename(os.path.dirname(p)), p, None] for p in a.images]
    jobs = [(j[0], os.path.abspath(j[1]), j[2] if len(j) > 2 else None) for j in jobs]
    synth = json.load(open(a.synth_box))["box"] if a.synth_box else None
    os.chdir(NCKU)
    sys.path.insert(0, NCKU)
    import k_diffusion as K
    import torchvision.transforms as T
    torch.autograd.set_grad_enabled(False)
    config = K.config.load_config("./configs/config_scalp_texture_conditional.json")
    inner = K.config.make_model(config).cuda().eval()
    model = K.config.make_denoiser_wrapper(config)(inner)
    ck = torch.load("./checkpoints/difflocks_diffusion/scalp_v9_40k_06730000.pth", map_location="cpu")
    model.inner_model.load_state_dict(ck["model_ema"])
    del ck
    model = model.cuda().eval()
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg").cuda().eval()
    prep = T.Compose([T.Resize(770, interpolation=T.InterpolationMode.BICUBIC),
                      T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    g = torch.Generator().manual_seed(0)
    proj = torch.linalg.qr(torch.randn(3072, DIM, generator=g))[0].cuda()
    noise = torch.randn(1, 65, 256, 256, generator=g).cuda()
    feats = {}
    inner.mid_level.register_forward_hook(lambda m, i, o: feats.__setitem__("mid", o))
    for key, path, box in jobs:
        dst = os.path.join(out, key + ".npy")
        if os.path.exists(dst):
            continue
        frame = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        if box:
            b = box
        elif synth is not None:
            b = synth
        else:
            raise ValueError(f"no crop box for {path}")
        crop = crop770(frame, b)
        rgb = torch.tensor(crop).cuda().permute(2, 0, 1)[None].float() / 255.0
        o = dino.forward_features(prep(rgb))
        pt = o["x_norm_patchtokens"]
        s = int(pt.shape[1] ** 0.5)
        extra = {"latents_dict": {"dinov2": {"cls_token": o["x_norm_clstoken"],
                                             "final_latent": pt.reshape(1, s, s, -1).permute(0, 3, 1, 2).contiguous()}}}
        sigma = torch.full((1,), SIGMA, device="cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            model(noise * SIGMA, sigma, **extra)
        f = feats["mid"].float()                      # (1, 32, 32, 3072)
        f = (f.reshape(-1, f.shape[-1]) @ proj).reshape(f.shape[1], f.shape[2], DIM)
        f = f / (f.std() + 1e-6)
        np.save(dst, f.cpu().numpy().astype(np.float16))
        print(key, tuple(feats["mid"].shape), flush=True)


if __name__ == "__main__":
    main()
