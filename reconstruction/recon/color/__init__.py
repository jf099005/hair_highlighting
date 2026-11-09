"""Color reconstruction (base / highlight color), separate from the region partition.

Dependency direction: this package uses the partition's OUTPUT (a front label image: 0 none, 1 body, 2 base,
3 highlight, in the crop space) and may call the partition code to produce it; nothing in recon/*.py imports this
package. Modules:
    regions     label images: GT (prep cache), predicted by a partition run (synthetic / real), mask augmentation
    intrinsic   albedo / shading decomposition backends (none, retinex; 'ordinal' = Careaga & Aksoy, optional install)
    features    robust per-region color statistics
    estimators  per-region color estimators (render curve, kNN, statistics regression, mask-pooling CNN)
    run         train / evaluate / real-photo inference and renders
"""
