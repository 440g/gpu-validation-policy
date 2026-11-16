"""R4: k-means over the submission-time feature space, fitted on each fold's training jobs only."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C


class KMeansLabel(BaseEstimator, TransformerMixin):
    """Cluster membership as one-hot columns (for the GLM) or a single label column (for HGB)."""

    def __init__(self, k, onehot):
        self.k, self.onehot = k, onehot

    def fit(self, X, y=None):
        self.prep_ = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(X)
        self.km_ = KMeans(self.k, n_init=4, random_state=C.SEED).fit(self.prep_.transform(X))
        return self

    def predict(self, X):
        return self.km_.predict(self.prep_.transform(X))

    def transform(self, X):
        lab = self.predict(X)
        return np.eye(self.k)[lab] if self.onehot else lab[:, None].astype(float)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"cluster_{i}" for i in range(self.k)] if self.onehot else ["cluster"])


def profile(df, labels, saved, saved_by_oracle):
    """Per cluster: size, failure rate, and its share of the saving the policy left behind."""
    y = df["y"].to_numpy()
    jobs = pd.DataFrame({"cluster": labels, "y": y, "fb_failed": df["fb"].where(y == 1).to_numpy(),
                         "missed": saved_by_oracle - saved})
    out = jobs.groupby("cluster").agg(n=("y", "size"), fail_rate=("y", "mean"),
                                      mean_fb_failed=("fb_failed", "mean"), missed=("missed", "sum"))
    out["share_of_jobs"] = out["n"] / out["n"].sum()
    missed = out.pop("missed")
    out["share_of_missed_saving"] = missed / missed.sum()
    return out
