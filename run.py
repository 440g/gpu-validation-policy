"""Reproduce every number in the report:  python run.py   (writes results/)."""
import json
import platform
import sys
import time

import numpy as np
import pandas as pd
import sklearn

from gpuval import clustering, data, evaluation, features, figures, models, validation
from gpuval import config as C

LABELS = {"glm": "GLM (logistic + Poisson)", "hgb": "Gradient boosting"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def prepare():
    if not C.DATA_PATH.exists():
        sys.exit(f"{C.DATA_PATH} not found. Run: python scripts/fetch_data.py")
    jobs = data.load_jobs()
    train, test, cutoff = data.split(jobs.join(features.build(jobs)))
    log(f"train {len(train):,} jobs, test {len(test):,} jobs, cutoff {pd.to_datetime(cutoff, unit='s')} UTC")
    return train, test, cutoff


def tune(train):
    """Everything chosen on the training period: per family the config, ranking rule and k (R3, R4)."""
    log("rolling-origin validation over both grids")
    cv = validation.run(train, validation.grid(), log)
    best = validation.choose(cv)
    log("R4: k-means features at each family's chosen config")
    with_clusters = [(b.family, b.config, k) for b in best.itertuples() for k in C.K_GRID]
    cv_clusters = validation.run(train, with_clusters, log)
    choices = {b.family: dict(config=int(b.config), params=models.GRIDS[b.family][b.config], rule=b.rule,
                              k=validation.adopt_clusters(cv, cv_clusters, b.family, b.config, b.rule),
                              val_savings=float(b.savings))
               for b in best.itertuples()}
    return cv, cv_clusters, choices


def fit(family, choice, train, k):
    return models.make(family, models.GRIDS[family][choice["config"]], k).fit(train)


def ablate_clusters(train, test, choices, saved, rng, boot):
    """R4 on test: each chosen model against the same model with clustering toggled."""
    tables = []
    for family, choice in choices.items():
        other_k = None if choice["k"] else C.R4_REFERENCE_K
        other = {family: fit(family, choice, train, other_k).predict(test)}
        _, other_saved = evaluation.policies(test, other, {family: choice["rule"]}, rng)
        pair = pd.DataFrame({"chosen": saved[family], "alternative": other_saved[family]})
        table = evaluation.interval(*boot.per_1000(pair), [("chosen", "alternative")])
        tables.append(table.assign(model=family, chosen_k=choice["k"] or 0, alternative_k=other_k or 0))
    return pd.concat(tables, ignore_index=True)


def coefficients(glm):
    rows = []
    for part, pipeline in (("logistic: log-odds of failure", glm.clf), ("poisson: log mean delay", glm.reg)):
        names = pipeline[0].get_feature_names_out()
        rows += [dict(part=part, feature=n, coef=c) for n, c in zip(names, np.ravel(pipeline[-1].coef_))]
    return pd.DataFrame(rows)


def evaluate(train, test, choices):
    """Score the locked choices on the test period. Each returned table becomes results/test_<name>.csv."""
    rng = np.random.default_rng(C.SEED)
    fitted = {family: fit(family, choice, train, choice["k"]) for family, choice in choices.items()}
    scores = {family: model.predict(test) for family, model in fitted.items()}
    rules = {family: choice["rule"] for family, choice in choices.items()}
    families, train_users = list(choices), set(train["user"])
    boot = evaluation.DayBootstrap(test["day"].to_numpy())

    log("headline")
    selected, saved = evaluation.policies(test, scores, rules, rng)
    out = {"table": evaluation.interval(*boot.per_1000(saved), evaluation.comparisons(families)),
           "secondary": evaluation.secondary(test, selected, scores, rules)}

    log("slices")
    slice_df = evaluation.slice_frame(test, train_users)
    out["slices"] = evaluation.slices(test, selected, saved, slice_df, families, boot)
    out["delay_profile"] = evaluation.delay_profile(train, test, train_users)

    log("R4 on test: ablation and cluster profile")
    out["ablation"] = ablate_clusters(train, test, choices, saved, rng, boot)
    kmeans = clustering.KMeansLabel(C.R4_REFERENCE_K, onehot=False).fit(train[features.NUMERIC])
    clusters = kmeans.predict(test[features.NUMERIC])
    for family in families:
        out[f"cluster_profile_{family}"] = clustering.profile(
            test, clusters, saved[family].to_numpy(), saved["oracle"].to_numpy()).reset_index()

    log("cost sensitivity")
    out["sensitivity"] = evaluation.sensitivity(test, scores, rules, rng, boot)
    out["d_curve"] = evaluation.d_curve(test, scores, rules, rng)
    out["coefficients"] = coefficients(fitted["glm"])
    return out


def headline(tables, choices):
    table = tables["table"].set_index("quantity")
    out = {}
    for family in choices:
        versus = table.loc[f"{family} - random"]
        out[LABELS[family]] = {"savings_per_1000": table.loc[family, "estimate"],
                               "ci95": [table.loc[family, "lo"], table.loc[family, "hi"]],
                               "minus_random": versus["estimate"],
                               "minus_random_ci95": [versus["lo"], versus["hi"]]}
    out["Random (same cap)"] = {"savings_per_1000": table.loc["random", "estimate"]}
    out["Pass-all"] = {"savings_per_1000": 0.0}
    out["Oracle (ceiling)"] = {"savings_per_1000": table.loc["oracle", "estimate"]}
    out["worst_slice"] = evaluation.worst_slices(tables["slices"], list(choices)).to_dict("records")
    return out


def save(cv, cv_clusters, choices, tables, train, test, cutoff):
    out = C.RESULTS
    out.mkdir(parents=True, exist_ok=True)
    cv.to_csv(out / "validation_grid.csv", index=False)
    validation.pooled(cv).to_csv(out / "validation_pooled.csv", index=False)
    cv_clusters.to_csv(out / "validation_clusters.csv", index=False)
    for name, table in tables.items():
        table.to_csv(out / f"test_{name}.csv", index=False)
    summary = headline(tables, choices)
    info = dict(python=platform.python_version(), sklearn=sklearn.__version__, pandas=pd.__version__,
                numpy=np.__version__, seed=C.SEED, d=C.D_BASE, r=C.R_BASE, budget=C.BUDGET,
                cutoff_utc=str(pd.to_datetime(cutoff, unit="s")), n_train=len(train), n_test=len(test),
                choices=choices, headline=summary)
    (out / "headline.json").write_text(json.dumps(info, indent=2, default=float))
    figures.draw(out)
    log(f"wrote {out}")
    print(json.dumps(summary, indent=2, default=lambda v: round(float(v), 1)))


def main():
    train, test, cutoff = prepare()
    cv, cv_clusters, choices = tune(train)
    log(f"locked before test: {choices}")
    tables = evaluate(train, test, choices)
    save(cv, cv_clusters, choices, tables, train, test, cutoff)


if __name__ == "__main__":
    main()
