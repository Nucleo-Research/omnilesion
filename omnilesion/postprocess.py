"""Post-processing of the network's foreground probability map, shared by offline prediction and the container.

    1. binarise at ``binarise`` (0.20);
    2. keep only connected components whose 99th-percentile probability is at least ``prune`` (0.98);
    3. empty rule: if what survives totals at most ``empty_total_ml`` millilitres (0.5) in at most
       ``empty_max_components`` components (2), return an empty mask.

Both thresholds were fixed before any evaluation and are not tuned per dataset. Components use scipy's default
6-connectivity, as in the scored submissions.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

BINARISE = 0.20
PRUNE = 0.98
EMPTY_TOTAL_ML = 0.5
EMPTY_MAX_COMPONENTS = 2


def postprocess(probabilities: np.ndarray, voxel_ml: float, binarise: float = BINARISE, prune: float = PRUNE,
                empty_total_ml: float = EMPTY_TOTAL_ML, empty_max_components: int = EMPTY_MAX_COMPONENTS,
                ) -> tuple[np.ndarray, bool]:
    """Return the uint8 lesion mask and whether the empty rule fired."""
    labelled, count = ndimage.label(probabilities > binarise)
    output = np.zeros(probabilities.shape, dtype=np.uint8)
    if not count:
        return output, False
    index = np.arange(1, count + 1)
    if prune > 0:
        p99 = ndimage.labeled_comprehension(probabilities, labelled, index,
                                            lambda values: np.percentile(values, 99), np.float64, 0.0)
        index = index[p99 >= prune]
    if not len(index):
        return output, False
    if empty_total_ml > 0 and len(index) <= empty_max_components:
        total_ml = float(np.isin(labelled, index).sum()) * voxel_ml
        if total_ml <= empty_total_ml:
            return output, True
    output[np.isin(labelled, index)] = 1
    return output, False
