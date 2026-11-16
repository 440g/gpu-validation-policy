"""The decision: which of each day's jobs to validate, and what that is worth in feedback minutes."""
import numpy as np

from . import config as C

RULES = ("p_fill", "ev_fill", "ev_positive")


def _jobs_that_day(day):
    _, inverse, counts = np.unique(day, return_inverse=True, return_counts=True)
    return counts[inverse]


def daily_cap(day):
    """At most floor(BUDGET * n_day) validations on each day."""
    return np.floor(C.BUDGET * _jobs_that_day(day)).astype(int)


def random_probability(day):
    """Random selection under the same cap, as an exact per-job probability (no seed noise)."""
    return daily_cap(day) / _jobs_that_day(day)


def select(day, score, rng, positive_only=False):
    """Validate the top-scoring jobs of each day up to the cap; ties are broken at random."""
    order = np.lexsort((rng.random(len(day)), -score, day))
    day_sorted = day[order]
    rank = np.empty(len(day), int)
    rank[order] = np.arange(len(day)) - np.searchsorted(day_sorted, day_sorted, side="left")
    chosen = rank < daily_cap(day)
    if positive_only:
        chosen &= score > 0
    return chosen


def apply_rule(rule, day, p, m, rng, d=C.D_BASE, r=C.R_BASE):
    """p_fill ranks by failure probability; the ev rules rank by expected net benefit r*p*m - d."""
    if rule == "p_fill":
        return select(day, p, rng)
    return select(day, r * p * m - d, rng, positive_only=(rule == "ev_positive"))


def gain(y, fb, d=C.D_BASE, r=C.R_BASE):
    """Net feedback minutes saved by validating each job: r * (queue + run) if it fails, minus d."""
    return r * y * fb - d


def per_1000(saving):
    return 1000.0 * saving.sum() / len(saving)


def break_even_probability(m, d=C.D_BASE, r=C.R_BASE):
    """Failure probability above which validating a job with expected failure delay m pays off."""
    return d / (r * np.maximum(m, 1e-9))
