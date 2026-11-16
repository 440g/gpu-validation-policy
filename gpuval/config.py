"""Every number fixed by the proposal or chosen before looking at the test period lives here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "trace_seren.csv"
RESULTS = ROOT / "results"

DATA_URL = ("https://huggingface.co/datasets/Qinghao/AcmeTrace/resolve/"
            "a484f7763a7812fc72eb49b8ba865b12fe7638e1/data/job_trace/trace_seren.csv")
DATA_SHA256 = "747854c73ce8b3263b764aa9ff8d866d89903705d31fcd0e0520c33dd14af9a1"

SEED = 0
UTC_OFFSET_H = 8          # Seren timestamps are UTC+8; "each day" is the local calendar day

# Locked in the proposal
TEST_FRAC = 0.30
BUDGET = 0.10
D_BASE, R_BASE = 0.5, 0.5
D_GRID = (0.25, 0.5, 2.0)
R_GRID = (0.25, 0.5, 1.0)

N_FOLDS = 4
VAL_BLOCK_DAYS = 14      # length of each rolling-origin validation block
N_BOOT = 2000

GPU_BINS = (0, 1, 8, float("inf"))
GPU_LABELS = ("1", "2-8", ">8")
MIN_SLICE_FAILED = 100    # a pre-specified slice needs this many test failures to be named the worst

# R4: clusters enter a model only if pooled validation savings rise
# and they win in at least this many of the N_FOLDS folds
K_GRID = (4, 8, 16)
R4_MIN_FOLD_WINS = 3
R4_REFERENCE_K = 8        # k used for the test ablation of an unclustered model and for the miss profile
