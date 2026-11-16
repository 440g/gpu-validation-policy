"""Submission-time features. A job's history only uses jobs that had *ended* before it was submitted."""
import numpy as np
import pandas as pd

from . import config as C

RECENT = 20
BURST_WINDOW_S = 600

NUMERIC = [
    "log_gpu", "log_cpu", "log_nodes", "hour_sin", "hour_cos", "weekend",
    "log_user_n", "user_fail_rate", "user_recent_fail_rate", "user_type_fail_rate",
    "user_last_failed", "log_hours_since_fail", "user_log_fb_fail", "user_log_dur",
    "log_user_burst", "log_pending_jobs", "log_running_gpus",
]
CATEGORICAL = ["type"]  # raw trace column


def _history(end, submit, y, logfb, logdur):
    """As-of statistics over a group's jobs with end < submit."""
    order = np.argsort(end, kind="stable")
    e, ys = end[order], y[order]
    pos = np.searchsorted(e, submit, side="left")

    def cum(v):
        return np.concatenate([[0.0], np.cumsum(v)])

    cy, cfb, cdur = cum(ys), cum(logfb[order] * ys), cum(logdur[order])
    n, f = pos.astype(float), cy[pos]
    lo = np.maximum(pos - RECENT, 0)
    fail_end = np.concatenate([[np.nan], e[ys == 1]])
    last_fail = fail_end[np.searchsorted(fail_end[1:], submit, side="left")]
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "n": n,
            "fail_rate": (f + 1) / (n + 2),
            "recent_fail_rate": (cy[pos] - cy[lo] + 1) / (pos - lo + 2),
            "last_failed": np.where(pos > 0, ys[np.maximum(pos - 1, 0)], np.nan),
            "hours_since_fail": (submit - last_fail) / 3600,
            "log_fb_fail": np.where(f > 0, cfb[pos] / np.maximum(f, 1), np.nan),
            "log_dur": np.where(n > 0, cdur[pos] / np.maximum(n, 1), np.nan),
        }


def _per_group(df, keys, fn):
    out = {}
    for idx in df.groupby(keys, sort=False).indices.values():
        for k, v in fn(idx).items():
            out.setdefault(k, np.full(len(df), np.nan))[idx] = v
    return out


def _cluster_load(df):
    """Jobs waiting and GPUs busy at each submission, reconstructed from the trace timestamps."""
    submit, gpus = df["submit_time"].to_numpy(), df["gpu_num"].to_numpy(float)

    def gpus_before(event):
        """Total GPUs of the jobs whose start or end event precedes each submission."""
        order = np.argsort(df[event].to_numpy())
        total = np.concatenate([[0], np.cumsum(gpus[order])])
        return total[np.searchsorted(df[event].to_numpy()[order], submit, side="left")]

    started = np.searchsorted(np.sort(df["start_time"].to_numpy()), submit, side="left")
    pending = np.arange(len(df)) - started
    running = gpus_before("start_time") - gpus_before("end_time")
    return np.maximum(pending, 0), np.maximum(running, 0)


def build(df):
    """df must be sorted by submission (data.load_jobs). Returns the NUMERIC columns aligned with df."""
    end, sub = df["end_time"].to_numpy(), df["submit_time"].to_numpy()
    y = df["y"].to_numpy(float)
    logfb = np.log1p(df["fb"].to_numpy())
    logdur = np.log1p(df["duration"].to_numpy() / 60)

    def hist(idx):
        return _history(end[idx], sub[idx], y[idx], logfb[idx], logdur[idx])

    u = _per_group(df, "user", hist)
    ut = _per_group(df, ["user", "type"], hist)

    def burst(idx):
        t = sub[idx]
        return {"burst": np.arange(len(t)) - np.searchsorted(t, t - BURST_WINDOW_S, side="left")}

    b = _per_group(df, "user", burst)["burst"]
    pending, running = _cluster_load(df)
    hour = (sub / 3600 + C.UTC_OFFSET_H) % 24
    weekday = (df["day"].to_numpy() + 3) % 7  # day 0 (1970-01-01) was a Thursday, so Monday = 0

    return pd.DataFrame({
        "log_gpu": np.log2(df["gpu_num"].to_numpy()),
        "log_cpu": np.log2(df["cpu_num"].clip(lower=1).to_numpy()),
        "log_nodes": np.log2(df["node_num"].to_numpy(float)),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "weekend": (weekday >= 5).astype(float),
        "log_user_n": np.log1p(u["n"]),
        "user_fail_rate": u["fail_rate"],
        "user_recent_fail_rate": u["recent_fail_rate"],
        "user_type_fail_rate": ut["fail_rate"],
        "user_last_failed": u["last_failed"],
        "log_hours_since_fail": np.log1p(u["hours_since_fail"]),
        "user_log_fb_fail": u["log_fb_fail"],
        "user_log_dur": u["log_dur"],
        "log_user_burst": np.log1p(b),
        "log_pending_jobs": np.log1p(pending),
        "log_running_gpus": np.log1p(running),
    }, index=df.index)
