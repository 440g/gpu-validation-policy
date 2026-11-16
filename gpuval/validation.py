"""Expanding-window rolling-origin validation inside the training period only."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config as C
from . import models, policy

CANDIDATE = ["family", "config", "k"]


def folds(train):
    """Yield (fold, fit, val). Each val block is VAL_BLOCK_DAYS whole days; fit jobs ended before it."""
    day = train["day"].to_numpy()
    first_days = [day[-1] + 1 - C.VAL_BLOCK_DAYS * (C.N_FOLDS - k) for k in range(C.N_FOLDS)]
    starts = list(np.searchsorted(day, first_days, side="left")) + [len(train)]
    for k in range(C.N_FOLDS):
        origin = train["submit_time"].iloc[starts[k]]
        fit = train.iloc[:starts[k]]
        yield k, fit[fit["end_time"] <= origin], train.iloc[starts[k]:starts[k + 1]]


def score(val, p, m, rng):
    """Savings per 1,000 jobs of every ranking rule and of random, plus the secondary metrics."""
    day, y = val["day"].to_numpy(), val["y"].to_numpy()
    gain = policy.gain(y, val["fb"].to_numpy())
    out = {rule: policy.per_1000(policy.apply_rule(rule, day, p, m, rng) * gain) for rule in policy.RULES}
    out["random"] = policy.per_1000(policy.random_probability(day) * gain)
    out["precision_p_rank"] = y[policy.select(day, p, rng)].mean()
    out["auc"] = roc_auc_score(y, p)
    return out


def grid():
    """Every (family, config index, k) of the two hyperparameter grids, without clustering."""
    return [(family, i, None) for family, configs in models.GRIDS.items() for i in range(len(configs))]


def run(train, candidates, log=print):
    """One row per candidate x fold. candidates: (family, config index, k clusters or None)."""
    rng = np.random.default_rng(C.SEED)
    rows = []
    for fold, fit, val in folds(train):
        log(f"fold {fold}: fit {len(fit):,} val {len(val):,}")
        for family, i, k in candidates:
            p, m = models.make(family, models.GRIDS[family][i], k).fit(fit).predict(val)
            rows.append(dict(family=family, config=i, k=k or 0, fold=fold, n_val=len(val),
                             **score(val, p, m, rng)))
    return pd.DataFrame(rows)


def pooled(cv):
    """Savings per 1,000 validation jobs for every candidate x rule, folds weighted by their job counts."""
    long = cv.melt(id_vars=CANDIDATE + ["fold", "n_val"], value_vars=list(policy.RULES),
                   var_name="rule", value_name="savings")
    long["minutes"] = long["savings"] * long["n_val"]
    out = long.groupby(CANDIDATE + ["rule"], as_index=False)[["minutes", "n_val"]].sum()
    out["savings"] = out.pop("minutes") / out.pop("n_val")
    return out


def choose(cv):
    """Per family, the candidate x rule with the highest pooled validation savings."""
    p = pooled(cv)
    return p.loc[p.groupby("family")["savings"].idxmax()].reset_index(drop=True)


def adopt_clusters(cv, cv_clusters, family, config, rule):
    """R4 rule: the k with the largest pooled gain over no clustering that also wins in at least
    R4_MIN_FOLD_WINS folds; None if no k qualifies."""
    base = cv[(cv["family"] == family) & (cv["config"] == config)].set_index("fold")
    best_k, best_gain = None, 0.0
    for k, with_k in cv_clusters[cv_clusters["family"] == family].groupby("k"):
        with_k = with_k.set_index("fold")
        diff = with_k[rule] - base[rule]
        pooled_gain = (diff * with_k["n_val"]).sum() / with_k["n_val"].sum()
        if pooled_gain > best_gain and (diff > 0).sum() >= C.R4_MIN_FOLD_WINS:
            best_k, best_gain = int(k), pooled_gain
    return best_k
