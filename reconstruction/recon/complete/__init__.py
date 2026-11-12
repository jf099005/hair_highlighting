"""Occluded-region completion in scalp space with generative models (separate from the partition; it consumes the
partition's per-strand predictions of the visible strands and fills in the hidden ones).

    build   scalp-space samples: GT highlight share per 64x64 scalp cell (full hair), hair coverage, and a strand
            subset with root cell, visibility (z-buffer from the fixed camera, no render needed) and label.
            Masks come from every generated dataset (not only the rendered ones), test hairstyles excluded from train.
    models  U-Net (deterministic completion) and the same U-Net as a conditional DDPM (with optional RePaint-style
            clamping of the observed cells while sampling)
    run     train / evaluate (hidden-strand and all-strand IoU vs the partition alone) / figures
"""
