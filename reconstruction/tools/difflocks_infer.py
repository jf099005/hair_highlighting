"""Run DiffLocks (unmodified, imported from ../../NCKU_3D_hair_reconstruction) on images and save the predicted hair.

Must run in the `difflocks` conda env (NATTEN / flash attention) from anywhere:
    ~/anaconda3/envs/difflocks/bin/python tools/difflocks_infer.py --out data/difflocks <img> [<img> ...]
    ... --synth_box data/synth_crop_box.json     # synthetic renders: use the fixed crop box instead of landmarks

Per image -> <out>/<stem>/: strands.npz (positions (S, 256, 3) as DiffLocks returns them, root_uv (S, 2)),
crop.json (box x1, y1, size in source pixels; landmarks), rgb_crop.png (770 crop fed to DINOv2), seconds.
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

NCKU = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "NCKU_3D_hair_reconstruction"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--synth_box", default=None)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out_root = os.path.abspath(a.out)
    images = [os.path.abspath(p) for p in a.images]
    synth = json.load(open(a.synth_box))["box"] if a.synth_box else None
    os.chdir(NCKU)
    sys.path.insert(0, NCKU)
    sys.path.insert(0, os.path.join(NCKU, "inference"))
    import img2hair as ih
    from utils.strand_util import sample_strands_from_scalp_with_density
    from utils.diffusion_utils import sample_images_cfg
    from data_loader.mesh_utils import tbn_space_to_world

    m = ih.DiffLocksInference("./checkpoints/strand_vae/strand_codec.pt",
                              "./configs/config_scalp_texture_conditional.json",
                              "./checkpoints/difflocks_diffusion/scalp_v9_40k_06730000.pth",
                              None, cfg_val=1.0, nr_iters_denoise=a.iters)
    for img in images:
        stem = os.path.splitext(os.path.basename(img))[0]
        if os.path.basename(os.path.dirname(img)) not in ("", "gt_examples") and stem == "front":
            stem = os.path.basename(os.path.dirname(img))
        od = os.path.join(out_root, stem)
        if os.path.exists(os.path.join(od, "strands.npz")):
            continue
        os.makedirs(od, exist_ok=True)
        t0 = time.time()
        frame = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        if synth is not None:
            x1, y1, size = synth
            lm = None
        else:
            lm_px, lm = m.mediapipe_img.run(frame)
            if lm is None:
                print("no face", img)
                continue
            h, w = frame.shape[:2]
            P = [(int(l[0] * w), int(l[1] * h)) for l in lm]
            chin, fore, lc, rc = P[152], P[10], P[234], P[454]
            size = int(max(rc[0] - lc[0], chin[1] - fore[1]) * 2.8)
            x1 = int((lc[0] + rc[0]) // 2 - size // 2)
            y1 = int(int(fore[1] * 0.4 + chin[1] * 0.6) - size // 2)
        # same crop as ih.crop_face (zero padding, resize to 770)
        H, W = frame.shape[:2]
        canvas = np.zeros((size, size, 3), np.uint8)
        sx0, sy0, sx1, sy1 = max(x1, 0), max(y1, 0), min(x1 + size, W), min(y1 + size, H)
        canvas[sy0 - y1:sy1 - y1, sx0 - x1:sx1 - x1] = frame[sy0:sy1, sx0:sx1]
        crop = cv2.resize(canvas, (770, 770), interpolation=cv2.INTER_AREA if size > 770 else cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(od, "rgb_crop.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

        torch.manual_seed(a.seed)
        rgb = torch.tensor(crop).cuda().permute(2, 0, 1).unsqueeze(0).float() / 255.0
        x = m.dinov2_latents_preprocessor(rgb)
        o = m.dinov2_latents_model.forward_features(x)
        pt = o["x_norm_patchtokens"]
        b, n, c = pt.shape
        s = int(n ** 0.5)
        extra = {"latents_dict": {"dinov2": {"cls_token": o["x_norm_clstoken"],
                                             "final_latent": pt.reshape(b, s, s, c).permute(0, 3, 1, 2).contiguous()}}}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            tex = sample_images_cfg(1, cfg_val=m.cfg_val, cfg_interval=[0.0, 5.0], model_ema=m.model_ema,
                                    model_config=m.model_config, nr_iters=m.nr_iters_denoise, extra_args=extra)
        tex = tex.float()
        dens = (tex[:, -1:] * (0.5 / m.model_config["sigma_data"]) + 0.5).clamp(0, 1)
        dens[dens < 0.02] = 0.0
        pts, _, ruv = sample_strands_from_scalp_with_density(
            tex[:, :-1], dens, m.strand_codec, normalization_dict=m.normalization_dict,
            scalp_mesh_data=m.scalp_mesh_data, tbn_space_to_world_func=tbn_space_to_world,
            nr_chunks=m.nr_chunks_decode_strands, upsample_multiplier=3)
        np.savez(os.path.join(od, "strands.npz"), positions=pts.cpu().numpy().astype(np.float32),
                 root_uv=ruv.cpu().numpy().astype(np.float32))
        np.savez_compressed(os.path.join(od, "scalp_texture.npz"), scalp_texture=tex.cpu().numpy().astype(np.float16))
        json.dump(dict(source=img, box=[int(x1), int(y1), int(size)], synthetic=synth is not None,
                       landmarks=None if lm is None else [list(map(float, l)) for l in lm],
                       seconds=round(time.time() - t0, 1), n_strands=int(pts.shape[0])),
                  open(os.path.join(od, "crop.json"), "w"))
        print(stem, pts.shape, round(time.time() - t0, 1), "s", flush=True)


if __name__ == "__main__":
    main()
