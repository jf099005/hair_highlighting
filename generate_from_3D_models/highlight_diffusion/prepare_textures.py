"""Stage 0: encode dataset hairstyles into DiffLocks scalp textures (the HDiT's output space).

For every hairstyle this writes data/textures/<name>/
    texture.pt   (65, 256, 256) float16: 64 strand-VAE latents (push-pull inpainted, as
                 create_scalp_textures.py) + density (scaled as train_scalp_diffusion.py)
    strands.npz  per strand of full_strands.npz: texel (row, col), root position, TBN frame (as the decoder
                 needs it), raw-uv scalp-grid cell, plus mirror flag
    meta.json    VAE round-trip error (mean / p95 vertex distance, m) on a subset

Texture = what DiffLocks' diffusion model is trained to produce, so it stands in for "HDiT inference output"
during training (running the 100-step HDiT on every training image would cost ~20 s each and adds the
image->hair error on top). At inference, the HDiT output texture is fed in the same way (see pipeline.py).

    python -m highlighting.generate_from_3D_models.highlight_diffusion.prepare_textures --names a,b | --from_cases
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
import torch

from . import common as C


def push_pull_inpaint(tex, mask):
    """tex (B, C, H, W), mask (B, 1, H, W) {0,1}: fill empty texels from a coarse-to-fine pyramid
    (same idea as DiffLocks' push_pull_inpaint CUDA extension)."""
    texs, masks = [tex * mask], [mask]
    t, m = tex * mask, mask
    while t.shape[-1] > 1:
        ts = torch.nn.functional.avg_pool2d(t, 2)
        ms = torch.nn.functional.avg_pool2d(m, 2)
        t = torch.where(ms > 0, ts / ms.clamp_min(1e-8), torch.zeros_like(ts))
        m = (ms > 0).float()
        texs.append(t)
        masks.append(m)
    out = texs[-1]
    for t, m in zip(reversed(texs[:-1]), reversed(masks[:-1])):
        up = torch.nn.functional.interpolate(out, size=t.shape[-2:], mode="bilinear", align_corners=False)
        out = torch.where(m > 0, t, up)
    return out


class Codec:
    """strand VAE (encoder + decoder) + normalization, on cuda"""

    def __init__(self):
        from models.strand_codec import StrandCodec
        from data_loader.dataloader import DiffLocksDataset, DEFAULT_BODY_DATA_DIR
        self.codec = StrandCodec(do_vae=False, decode_type="dir", scale_init=30.0, nr_verts_per_strand=256,
                                 nr_values_to_decode=255, dim_per_value_decoded=3).cuda()
        self.codec.load_state_dict(torch.load(C.CKPT_STRAND_VAE, map_location="cuda", weights_only=True))
        self.codec.eval()
        for p in self.codec.parameters():
            p.requires_grad_(False)
        self.norm = DiffLocksDataset.get_normalization_data()
        self.scalp_mesh_data = None
        self._body = DEFAULT_BODY_DATA_DIR

    def scalp(self):
        if self.scalp_mesh_data is None:
            from data_loader.dataloader import DiffLocksDataset
            _, self.scalp_mesh_data = DiffLocksDataset.compute_scalp_data(os.path.join(self._body, "scalp.ply"))
        return self.scalp_mesh_data

    @torch.no_grad()
    def encode(self, positions, tbn, root_normal, chunk=20000):
        from data_loader.mesh_utils import world_to_tbn_space
        from models.strand_codec import normalize_gt_data
        from utils.strand_util import compute_dirs
        zs = []
        for s in range(0, positions.shape[0], chunk):
            pos = torch.from_numpy(positions[s:s + chunk]).cuda()[None]
            t = torch.from_numpy(tbn[s:s + chunk]).cuda()[None]
            rn = torch.from_numpy(root_normal[s:s + chunk]).cuda()[None]
            local, _ = world_to_tbn_space(t, pos, rn)
            local = local.reshape(-1, 256, 3)
            d = dict(strand_positions=local, strand_directions=compute_dirs(local, append_last_dir=False))
            zs.append(self.codec.encoder(normalize_gt_data(d, self.norm))["z"])
        return torch.cat(zs)

    def decode_local(self, z):
        """(N, 64) latents -> (N, 256, 3) strand points in the root's TBN space (root at 0); differentiable"""
        return self.codec.decoder(z, None, self.norm)["strand_positions"]


def frame_for_decode(tbn):
    """(S, 3, 3) TBN of the dataset (columns T, B, N) -> the rotation tbn_space_to_world applies
    (columns T, N, B with T negated), so world = root + R @ local."""
    R = tbn[:, :, [0, 2, 1]].copy()
    R[:, :, 0] = -R[:, :, 0]
    return R


def _np(a):
    return np.ascontiguousarray(a.detach().cpu().numpy() if torch.is_tensor(a) else a, dtype=np.float32)


def process(name, codec, out_root, check_n=4000):
    from data_loader.dataloader import StrandsData
    out = os.path.join(out_root, name)
    if os.path.isfile(os.path.join(out, "meta.json")):
        return "skip"
    t0 = time.time()
    hdir = C.hairstyle_dir(name)
    meta = C.load_json(os.path.join(hdir, "metadata.json"))
    npz = np.load(os.path.join(hdir, "full_strands.npz"))
    raw_uv = np.asarray(npz["root_uv"], np.float32)
    sd = StrandsData(npz, meta, codec.scalp(), None, compute_tbn=True)   # applies the mirror fix to root_uv
    S = sd.positions.shape[0]
    mirror = bool(meta.get("mirror_hair", False))
    tex_uv = C.raw_uv_to_texture_uv(raw_uv, mirror)
    tr, tc = C.root_texels(tex_uv)
    tbn = _np(sd.tbn)
    z = codec.encode(sd.positions, tbn, _np(sd.root_normal))

    res = C.TEX_RES
    tex = torch.zeros(1, 64, res, res, device="cuda")
    occ = torch.zeros(1, 1, res, res, device="cuda")
    ti, ci = torch.from_numpy(tr).cuda(), torch.from_numpy(tc).cuda()
    tex[0, :, ti, ci] = z.T
    occ[0, 0, ti, ci] = 1.0
    tex = push_pull_inpaint(tex, occ)
    import cv2
    dens = cv2.imread(os.path.join(hdir, "density.png"), cv2.IMREAD_UNCHANGED)
    dens = dens[..., 2] if dens.ndim == 3 else dens           # cv2 is BGR; DiffLocks reads channel 0 = R
    dens = torch.from_numpy(dens.astype(np.float32) / 255.0)[None, None].cuda()
    dens = (dens - 0.5) / (0.5 / C.SIGMA_DATA)
    dens = torch.nn.functional.interpolate(dens, size=(res, res), mode="nearest")
    full = torch.cat([tex, dens], 1)[0].half().cpu()

    g = C.scalp_grid()
    gr, gc = g.root_rowcol(raw_uv)
    R = frame_for_decode(tbn)
    os.makedirs(out, exist_ok=True)
    torch.save(full, os.path.join(out, "texture.pt"))
    np.savez_compressed(os.path.join(out, "strands.npz"), texel=np.stack([tr, tc], 1).astype(np.int16),
                        root=sd.positions[:, 0].astype(np.float32), frame=R.astype(np.float32),
                        grid_rc=np.stack([gr, gc], 1).astype(np.int16), raw_uv=raw_uv, mirror=mirror)

    # VAE + texture round trip on a subset: decode the texel latent at the strand's own root
    rng = np.random.default_rng(0)
    sel = rng.choice(S, min(check_n, S), replace=False)
    with torch.no_grad():
        zt = full[:64].float().cuda()[:, ti[sel], ci[sel]].T
        loc = codec.decode_local(zt)
        Rt = torch.from_numpy(R[sel]).cuda()
        world = torch.from_numpy(sd.positions[sel, :1]).cuda() + torch.einsum("sij,spj->spi", Rt, loc)
        err = (world - torch.from_numpy(sd.positions[sel]).cuda()).norm(dim=-1).mean(1).cpu().numpy()
        loc2 = codec.decode_local(z[sel])            # own latent (no texel sharing)
        world2 = torch.from_numpy(sd.positions[sel, :1]).cuda() + torch.einsum("sij,spj->spi", Rt, loc2)
        err2 = (world2 - torch.from_numpy(sd.positions[sel]).cuda()).norm(dim=-1).mean(1).cpu().numpy()
    length = float(np.median(np.linalg.norm(np.diff(sd.positions[sel], axis=1), axis=-1).sum(1)))
    C.save_json(os.path.join(out, "meta.json"), dict(
        name=name, n_strands=int(S), mirror=mirror, median_length=length,
        texels_used=int(occ.sum().item()), recon_err_texture_mean=float(err.mean()),
        recon_err_texture_p95=float(np.percentile(err, 95)), recon_err_vae_mean=float(err2.mean()),
        seconds=round(time.time() - t0, 1)))
    return f"{S} strands, err tex {err.mean()*1000:.2f} mm vae {err2.mean()*1000:.2f} mm, {time.time()-t0:.1f}s"


def case_hairstyles():
    """hairstyles of the gen_dataset hl_1000 candidates and the contour_algo_test hl_n50_s0 benchmark"""
    names = set()
    for pat in ("gen_dataset/hl_1000/candidates/cases/*/case.json", "contour_algo_test/data/hl_n50_s0/cases/*/case.json"):
        for f in glob.glob(os.path.join(C.GEN3D_DIR, pat)):
            names.add(C.load_json(f)["hairstyle"]["name"])
    return sorted(names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default=None, help="comma separated hairstyle names")
    ap.add_argument("--from_cases", action="store_true", help="every hairstyle used by hl_1000 / hl_n50_s0")
    ap.add_argument("--extra_random", type=int, default=0, help="also N random other hairstyles (seed 0)")
    ap.add_argument("--out", default=os.path.join(C.DATA_DIR, "textures"))
    args = ap.parse_args()
    names = []
    if args.names:
        names += args.names.split(",")
    if args.from_cases:
        names += case_hairstyles()
    if args.extra_random:
        allh = sorted(os.listdir(os.path.join(C.DATASET_PATH, "generated_hairstyles")))
        rest = [h for h in allh if h not in set(names)
                and os.path.isfile(os.path.join(C.hairstyle_dir(h), "full_strands.npz"))]
        names += list(np.random.default_rng(0).choice(rest, min(args.extra_random, len(rest)), replace=False))
    names = list(dict.fromkeys(names))
    print(f"{len(names)} hairstyles -> {args.out}", flush=True)
    codec = Codec()
    for i, n in enumerate(names):
        try:
            print(f"[{i+1}/{len(names)}] {n}: {process(n, codec, args.out)}", flush=True)
        except Exception as e:  # keep going; a broken hairstyle is just left out
            print(f"[{i+1}/{len(names)}] {n}: FAILED {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
