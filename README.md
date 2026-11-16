# gpu-validation-policy

The Seren platform team can run a dry-run validation (`d` minutes) on at most 10% of each day's GPU jobs.
A caught failure saves `r · (queue + run time)`; every validation costs `d`.
This repo learns which jobs to validate, using only what is known at submission.

## Reproduce

```bash
pip install -r requirements.txt
python scripts/fetch_data.py   # downloads trace_seren.csv (pinned HF commit) and checks its sha256
python run.py                  # ~4 min; writes results/ (tables, figures, headline.json)
```

Seed 0 is fixed in `gpuval/config.py`. Two runs from a clean directory produce identical `results/headline.json`.
The numbers below come from the pinned versions in `requirements.txt` (Python 3.9.6, scikit-learn 1.6.1). The GLM results are identical under scikit-learn 1.9.1. Gradient boosting is not: there it selects a different rule and k, and scores 1,476 instead of 1,455. The conclusions do not change.

## Headline (test period, d = 0.5, r = 0.5)

Net feedback-time saving per 1,000 test jobs. Intervals are 95% paired day-block bootstrap intervals over the 45 test days.

| Policy | min / 1,000 jobs | 95% CI |
|---|---:|---|
| GLM (logistic + Poisson), expected-value ranking | **1,825** | 1,346 – 2,380 |
| Gradient boosting, expected-value ranking | 1,455 | 1,036 – 1,967 |
| Random under the same cap (baseline) | 296 | 171 – 489 |
| Pass-all (baseline) | 0 | — |
| Oracle ceiling | 3,367 | 2,106 – 5,296 |

- GLM − Random is 1,529 (1,055 – 2,065).
- GLM − Gradient boosting is 370 (172 – 624).
- Worst pre-specified slice: 1-GPU jobs (n = 99,360, 23,429 failures). There the GLM captures 12% (2% – 55%) of the achievable saving.

## Layout

| Path | Role |
|---|---|
| `gpuval/config.py` | Every locked or pre-chosen number: split, budget, d, r, grids, slices, R4 rule |
| `gpuval/data.py` | Load, label (`state == FAILED`), chronological 70/30 split; drop train jobs unfinished at cutoff |
| `gpuval/features.py` | Submission-time features. User history uses only jobs that had ended before submission |
| `gpuval/policy.py` | Daily cap, ranking rules, net-saving metric, Random/Pass-all baselines (R1, R2) |
| `gpuval/models.py` | Two hurdle models: P(fail) × E[delay \| fail]. GLM vs gradient boosting (R3) |
| `gpuval/validation.py` | Expanding-window rolling-origin validation inside the training period (4 × 14-day blocks) |
| `gpuval/clustering.py` | k-means cluster features and the per-cluster profile of missed savings (R4) |
| `gpuval/evaluation.py` | Paired day-block bootstrap, slices, delay-shift diagnosis, cost sensitivity (R5, R6) |
| `gpuval/figures.py` | Report figures |
| `run.py` | Runs everything in order. Choices are written to the log before the test set is scored |
| `archive/` | Pre-proposal feasibility scripts, memos and outputs (Oct–Nov); not used by `run.py` |

## Decisions made on validation only (before scoring the test period)

- **Ranking rule.** The policy ranks jobs by expected net benefit `r·p̂·m̂ − d`, not by failure probability `p̂`. At the chosen configurations on validation, ranking by `p̂` alone saved 82–85 min/1,000; ranking by expected benefit saved 410–415. Most failures end within 30 s (median delay 0.48 min), so catching them is worth less than `d`.
- **Hyperparameters.** 9 configurations per family, selected by pooled validation saving.
- **R4 clustering.** k-means clusters enter a model only if they raise pooled saving and win in at least 3 of the 4 folds. Under this rule, gradient boosting adopted k = 4 and the GLM adopted none. The test ablation in `results/test_ablation.csv` shows that neither change is distinguishable from zero.

## Data

Seren job trace from AcmeTrace (Shanghai AI Lab), CC BY 4.0, 818,327 jobs, of which 450,590 request a GPU.
