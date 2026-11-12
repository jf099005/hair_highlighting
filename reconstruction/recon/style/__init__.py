"""Highlight style prediction (dense / money piece / random), to pick the post-processing automatically. Separate
from the partition: it reads the partition's outputs (per-strand probabilities -> scalp map) and the image.
    features   scalp-map features from partition probabilities + frozen DINOv2 image features
    run        train / evaluate (logistic regression), real photos, and the automatic choice
"""
