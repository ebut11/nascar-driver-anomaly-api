"""Custom transformer used by the driver-anomaly pipeline.

Imported by build_pipeline.py (to fit + dump the bundle) and by serve.py /
modal_serve.py (to unpickle it) -- must be import-identical in all three
places, which is why it lives in its own module.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Raw stat columns (from the Advanced Short Track Data CSV) this transformer
# expects, in order.
RAW_FEATURES = ["asp", "arp", "afp", "succ_pct", "pgae_per_100", "gr_lr", "ss", "cpoms"]

# Engineered columns this transformer appends.
ENGINEERED_FEATURES = ["surge_index", "consistency_gap"]

ALL_FEATURES = RAW_FEATURES + ENGINEERED_FEATURES


class FormIndexEngineer(BaseEstimator, TransformerMixin):
    """Engineers two robust, field-relative form features for each driver.

    - surge_index: robust z-score (median/IQR) of a blend of positions-gained
      (pgae_per_100) and net rating (gr_lr) -- how much a driver over-performs
      raw speed relative to the fitted field.
    - consistency_gap: robust z-score of (arp - afp) -- how much a driver's
      finish beats their average running position, i.e. closes well.

    The median/IQR used for the z-scores are learned in `fit` from the
    training field, so a fresh, unfitted instance of this class gives
    different (wrong) output than the one loaded from the dumped bundle.
    """

    def __init__(self, surge_weight: float = 0.5):
        # __init__ only assigns its argument -- no computation here.
        self.surge_weight = surge_weight

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        pgae_per_100 = X[:, RAW_FEATURES.index("pgae_per_100")]
        gr_lr = X[:, RAW_FEATURES.index("gr_lr")]
        arp = X[:, RAW_FEATURES.index("arp")]
        afp = X[:, RAW_FEATURES.index("afp")]

        surge_raw = self.surge_weight * pgae_per_100 + (1 - self.surge_weight) * gr_lr * 10
        consistency_raw = arp - afp

        self.surge_median_ = float(np.median(surge_raw))
        self.surge_iqr_ = float(np.subtract(*np.percentile(surge_raw, [75, 25])) or 1.0)
        self.consistency_median_ = float(np.median(consistency_raw))
        self.consistency_iqr_ = float(np.subtract(*np.percentile(consistency_raw, [75, 25])) or 1.0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        pgae_per_100 = X[:, RAW_FEATURES.index("pgae_per_100")]
        gr_lr = X[:, RAW_FEATURES.index("gr_lr")]
        arp = X[:, RAW_FEATURES.index("arp")]
        afp = X[:, RAW_FEATURES.index("afp")]

        surge_raw = self.surge_weight * pgae_per_100 + (1 - self.surge_weight) * gr_lr * 10
        consistency_raw = arp - afp

        surge_index = (surge_raw - self.surge_median_) / self.surge_iqr_
        consistency_gap = (consistency_raw - self.consistency_median_) / self.consistency_iqr_

        return np.column_stack([X, surge_index, consistency_gap])
