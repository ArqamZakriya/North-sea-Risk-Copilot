"""Model-agnostic explanation of a single prediction.

Each feature is replaced in turn by its training median, and the change in the predicted
probability of the chosen class is measured. A positive value means the feature's current
value raises that probability compared with a typical value. It works for any classifier
with predict_proba, so it does not depend on which model wins.
"""

import numpy as np
import pandas as pd


def feature_medians(train, features):
    """Training median of each feature."""
    return train[features].median()


def top_drivers(model, scaler, features, medians, row, cls, k=8):
    """Return the k features whose median swap changes the probability of class `cls` the most."""
    base = row[features].astype(float).to_numpy()
    mat = np.tile(base, (len(features) + 1, 1))
    for i, f in enumerate(features):
        mat[i + 1, i] = medians[f]
    p = model.predict_proba(scaler.transform(pd.DataFrame(mat, columns=features)))[:, cls]
    drive = pd.Series(p[0] - p[1:], index=features)
    return drive.reindex(drive.abs().sort_values(ascending=False).index).head(k)
