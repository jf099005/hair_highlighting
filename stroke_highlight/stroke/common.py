"""Paths and sample rows. Samples are reconstruction's indexed samples (recon.data_index) and their color variants;
this project only adds renders / masks / caches of its own under stroke_highlight/data/ (keyed like recon's render
dirs: the sample id with "/" -> "__").
"""
from __future__ import annotations

import os

from recon import data_index as di

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                         # stroke_highlight/
DATA = os.path.join(ROOT, "data")
MONO = os.path.join(DATA, "renders_mono")            # single-color renders (same geometry / seed as the dyed one)
MASKS = os.path.join(DATA, "masks")                  # precise masks (render space)
CACHE = os.path.join(DATA, "cache")                  # crop-space training cache
RUNS = os.path.join(DATA, "runs")

VARIANTS = (None, "nat")                             # None = recon's random-RGB renders, nat = melanin renders
# "dl" (optional, rows(variants=(..., "dl"))): recon's DiffLocks hair of a source sample (orig or nat), per-strand
# oracle labels; the image is the source's mono render, the hair does not line up with it (as at inference)


def key(row):
    """renders / masks / caches of this project are per SOURCE sample: a "dl" row (recon's DiffLocks-reconstructed
    hair of a rendered sample) shares its source sample's mono render, masks and stroke cache"""
    return row.get("src_id", row["id"]).replace("/", "__")


def rows(split=None, variants=VARIANTS, styles=None):
    out = []
    for v in variants:
        out += di.load_variant(v, split=split, styles=styles) if v else di.load_index(split=split, styles=styles)
    return out


def dyed_dir(row):
    """reconstruction's (highlighted) render of this sample: front.png + colors.json"""
    return di.render_dir(row)


def mono_dir(row):
    return os.path.join(MONO, key(row))


def mask_path(row):
    return os.path.join(MASKS, key(row) + ".npz")


def cache_path(row, ext="npz"):
    return os.path.join(CACHE, key(row) + "." + ext)
