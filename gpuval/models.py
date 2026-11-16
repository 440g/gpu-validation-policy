"""Two model families on the interpretability axis. Each is a hurdle model:
p = P(fail | x) and m = E[queue + run minutes | fail, x], so the policy can price a validation."""
from itertools import product

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from . import config as C
from .clustering import KMeansLabel
from .features import CATEGORICAL, NUMERIC

GRIDS = {
    "glm": [dict(C=c, alpha=a) for c, a in product((0.01, 0.1, 1.0), (0.01, 0.1, 1.0))],
    "hgb": [dict(max_depth=d, min_samples_leaf=s) for d, s in product((3, 6, None), (20, 200, 2000))],
}


def _prep(family, k):
    """Column order is numeric, job type, then (if k) the k-means cluster of the numeric features."""
    if family == "glm":
        numeric = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())
        job_type = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        numeric = "passthrough"
        job_type = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
    parts = [("num", numeric, NUMERIC), ("cat", job_type, CATEGORICAL)]
    if k:
        parts.append(("km", KMeansLabel(k, onehot=(family == "glm")), NUMERIC))
    return ColumnTransformer(parts)


def make(family, params, k=None):
    """k: number of k-means clusters added as features (R4); None means no clustering."""
    if family == "glm":
        clf = LogisticRegression(C=params["C"], max_iter=2000)
        reg = PoissonRegressor(alpha=params["alpha"], max_iter=1000)
    else:
        categorical = list(range(len(NUMERIC), len(NUMERIC) + 1 + bool(k)))
        common = dict(learning_rate=0.1, max_iter=200, early_stopping=False, random_state=C.SEED,
                      categorical_features=categorical, **params)
        clf = HistGradientBoostingClassifier(**common)
        reg = HistGradientBoostingRegressor(loss="poisson", **common)
    return Hurdle(make_pipeline(_prep(family, k), clf), make_pipeline(_prep(family, k), reg))


class Hurdle:
    """clf predicts p; reg is fitted on failed jobs only and predicts m."""

    def __init__(self, clf, reg):
        self.clf, self.reg = clf, reg

    def fit(self, df):
        X, y = df[NUMERIC + CATEGORICAL], df["y"].to_numpy()
        self.clf.fit(X, y)
        self.reg.fit(X[y == 1], df["fb"].to_numpy()[y == 1])
        return self

    def predict(self, df):
        X = df[NUMERIC + CATEGORICAL]
        return self.clf.predict_proba(X)[:, 1], self.reg.predict(X)
