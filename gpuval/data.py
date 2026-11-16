import pandas as pd

from . import config as C

TIME_COLS = ("submit_time", "start_time", "end_time")
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def _epoch_seconds(s):
    return (pd.to_datetime(s, utc=True) - EPOCH).dt.total_seconds().to_numpy()


def load_jobs(path=C.DATA_PATH):
    """GPU jobs ordered by submission, with label y, feedback delay fb (min) and local day."""
    df = pd.read_csv(path)
    df = df[df["gpu_num"] > 0].copy()
    for c in TIME_COLS:
        df[c] = _epoch_seconds(df[c])
    df = df.sort_values(["submit_time", "job_id"], kind="stable").reset_index(drop=True)
    df["y"] = (df["state"] == "FAILED").astype(int)
    df["fb"] = (df["queue"] + df["duration"]) / 60.0
    df["day"] = ((df["submit_time"] + C.UTC_OFFSET_H * 3600) // 86400).astype(int)
    return df


def split(df):
    """Return (train, test, cutoff). Test is the final TEST_FRAC of jobs by submission order;
    training jobs still unfinished at the cutoff are dropped because their labels were unknown then."""
    cut = int(len(df) * (1 - C.TEST_FRAC))
    cutoff = df["submit_time"].iloc[cut]
    train, test = df.iloc[:cut], df.iloc[cut:]
    train = train[train["end_time"] <= cutoff]
    return train.reset_index(drop=True), test.reset_index(drop=True), cutoff
