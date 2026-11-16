"""Test-period evaluation: paired day-block bootstrap, slices, cost sensitivity."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config as C
from . import policy

BASELINES = ("random", "pass_all")
EXPLORATORY = "exploratory"


def policies(test, scores, rules, rng, d=C.D_BASE, r=C.R_BASE):
    """Return (selected, saved), one column per policy and one row per test job.
    scores: {model: (p, m)}, rules: {model: rule}. Random is a selection probability."""
    day = test["day"].to_numpy()
    gain = policy.gain(test["y"].to_numpy(), test["fb"].to_numpy(), d, r)
    selected = {model: policy.apply_rule(rules[model], day, p, m, rng, d, r)
                for model, (p, m) in scores.items()}
    selected["random"] = policy.random_probability(day)
    selected["pass_all"] = np.zeros(len(test))
    selected["oracle"] = policy.select(day, gain, rng, positive_only=True)
    selected = pd.DataFrame(selected, index=test.index)
    return selected, selected.mul(gain, axis=0)


class DayBootstrap:
    """Resample whole test days with replacement; every policy is scored on the same draws (paired)."""

    def __init__(self, day):
        days, self.day_index = np.unique(day, return_inverse=True)
        draws = np.random.default_rng(C.SEED).integers(0, len(days), size=(C.N_BOOT, len(days)))
        self.weights = np.zeros((C.N_BOOT, len(days)))  # times each day appears in each resample
        np.add.at(self.weights, (np.repeat(np.arange(C.N_BOOT), len(days)), draws.ravel()), 1)

    def per_1000(self, saved, mask=None):
        """Point estimate and bootstrap draws of savings per 1,000 jobs, optionally within a job mask."""
        mask = np.ones(len(saved), bool) if mask is None else mask
        n_days = self.weights.shape[1]
        daily_saved = np.zeros((n_days, saved.shape[1]))
        np.add.at(daily_saved, self.day_index, np.where(mask[:, None], saved.to_numpy(), 0.0))
        daily_jobs = np.bincount(self.day_index, weights=mask.astype(float), minlength=n_days)
        with np.errstate(invalid="ignore", divide="ignore"):
            point = 1000 * daily_saved.sum(0) / daily_jobs.sum()
            draws = 1000 * (self.weights @ daily_saved) / (self.weights @ daily_jobs)[:, None]
        return pd.Series(point, saved.columns), pd.DataFrame(draws, columns=saved.columns)


def ci95(draws):
    return np.nanpercentile(draws, [2.5, 97.5])


def interval(point, draws, pairs=()):
    """Estimate and 95% percentile interval for each column and each paired (a, b) difference."""
    rows = [(c, point[c], *ci95(draws[c])) for c in point.index]
    rows += [(f"{a} - {b}", point[a] - point[b], *ci95(draws[a] - draws[b])) for a, b in pairs]
    return pd.DataFrame(rows, columns=["quantity", "estimate", "lo", "hi"])


def comparisons(models):
    """Each model against each baseline, then the models against each other."""
    return [(m, b) for m in models for b in BASELINES] + list(zip(models[:-1], models[1:]))


def secondary(test, selected, scores, rules):
    """Reported only: ROC-AUC of p and precision among the validated jobs."""
    y = test["y"].to_numpy()
    return pd.DataFrame([dict(model=m, rule=rules[m], auc=roc_auc_score(y, scores[m][0]),
                              validated_share=selected[m].mean(),
                              precision_at_budget=y[selected[m].to_numpy()].mean())
                         for m in scores])


def slice_frame(jobs, train_users):
    """Pre-specified slice variables, plus test week, which is marked exploratory."""
    week = (jobs["day"] - jobs["day"].min()) // 7 + 1
    return pd.DataFrame({
        "gpu_count": pd.cut(jobs["gpu_num"], C.GPU_BINS, labels=C.GPU_LABELS).astype(str),
        "job_type": jobs["type"].astype(str),
        "user": np.where(jobs["user"].isin(train_users), "seen", "unseen"),
        f"test_week ({EXPLORATORY})": "week " + week.astype(str),
    }, index=jobs.index)


def slices(test, selected, saved, slice_df, models, boot):
    """Per slice level: n, failures, savings per 1,000 slice jobs, gain over random, oracle share."""
    y = test["y"].to_numpy()
    rows = []
    for variable in slice_df.columns:
        for level in sorted(slice_df[variable].unique()):
            mask = (slice_df[variable] == level).to_numpy()
            point, draws = boot.per_1000(saved, mask)
            row = dict(variable=variable, level=level, n=mask.sum(), n_failed=y[mask].sum(),
                       fail_rate=y[mask].mean(), oracle=point["oracle"], random=point["random"])
            for m in models:
                validated = mask & selected[m].to_numpy()
                gain_lo, gain_hi = ci95(draws[m] - draws["random"])
                share_lo, share_hi = ci95(draws[m] / draws["oracle"])
                row.update({
                    m: point[m],
                    f"{m}_validated_share": validated.sum() / mask.sum(),
                    f"{m}_precision": y[validated].mean() if validated.any() else np.nan,
                    f"{m}_gain_vs_random": point[m] - point["random"],
                    f"{m}_gain_lo": gain_lo, f"{m}_gain_hi": gain_hi,
                    f"{m}_oracle_share": point[m] / point["oracle"],
                    f"{m}_share_lo": share_lo, f"{m}_share_hi": share_hi,
                })
            rows.append(row)
    return pd.DataFrame(rows)


def worst_slices(slice_table, models):
    """Per model, the pre-specified slice capturing the smallest share of the achievable (oracle) saving."""
    eligible = slice_table[~slice_table["variable"].str.contains(EXPLORATORY)
                           & (slice_table["n_failed"] >= C.MIN_SLICE_FAILED)]
    rows = []
    for m in models:
        worst = eligible.loc[eligible[f"{m}_oracle_share"].idxmin()]
        rows.append(dict(model=m, variable=worst["variable"], level=worst["level"], n=int(worst["n"]),
                         n_failed=int(worst["n_failed"]), savings=worst[m],
                         oracle_share=worst[f"{m}_oracle_share"],
                         ci95=[worst[f"{m}_share_lo"], worst[f"{m}_share_hi"]]))
    return pd.DataFrame(rows)


def delay_profile(train, test, train_users):
    """R5 mechanism: how the cost of a missed failure shifted between periods, per pre-specified slice."""
    rows = []
    for period, jobs in (("train", train), ("test", test)):
        failed = jobs[jobs["y"] == 1]
        slice_df = slice_frame(failed, train_users)
        for variable in [c for c in slice_df.columns if EXPLORATORY not in c]:
            for level, group in failed.groupby(slice_df[variable]):
                fb = np.sort(group["fb"].to_numpy())[::-1]
                by_user = group.groupby("user")["fb"].sum()
                rows.append(dict(variable=variable, level=level, period=period, n_failed=len(group),
                                 mean_delay=fb.mean(), median_delay=np.median(fb),
                                 top1pct_share=fb[:max(1, len(fb) // 100)].sum() / fb.sum(),
                                 queue_share=group["queue"].sum() / 60 / fb.sum(),
                                 top_user_share=by_user.max() / by_user.sum()))
    return pd.DataFrame(rows)


def sensitivity(test, scores, rules, rng, boot):
    """Savings with interval for every (d, r) in the locked grid, plus the break-even failure probability."""
    models = list(scores)
    rows = []
    for d in C.D_GRID:
        for r in C.R_GRID:
            _, saved = policies(test, scores, rules, rng, d, r)
            table = interval(*boot.per_1000(saved), comparisons(models)).set_index("quantity")
            row = dict(d=d, r=r)
            for quantity in models + ["random", "oracle"] + [f"{a} - {b}" for a, b in comparisons(models)]:
                row[quantity], row[f"{quantity}_lo"], row[f"{quantity}_hi"] = table.loc[quantity]
            for model, (p, m) in scores.items():
                break_even = policy.break_even_probability(m, d, r)
                row[f"{model}_median_break_even_p"] = np.median(np.minimum(break_even, 1.0))
                row[f"{model}_share_above_break_even"] = (p > break_even).mean()
            rows.append(row)
    return pd.DataFrame(rows)


def d_curve(test, scores, rules, rng, r=C.R_BASE, grid=(0.25, 0.5, 1, 2, 3, 5, 8, 12, 20, 30)):
    """Savings as validation cost d grows beyond the locked grid (exploratory, for R6)."""
    rows = []
    for d in grid:
        selected, saved = policies(test, scores, rules, rng, d, r)
        rows.append(dict(d=d, **{k: policy.per_1000(saved[k].to_numpy()) for k in list(scores) + ["random"]},
                         **{f"{k}_validated_share": selected[k].mean() for k in scores}))
    return pd.DataFrame(rows)
