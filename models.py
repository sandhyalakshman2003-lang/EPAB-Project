"""
models.py
=========
UDISE+ state/UT panel 2023-24 to 2025-26.

Paths are resolved relative to the folder that contains this file, so the
same code works locally and on Streamlit Community Cloud:

    working dir : the folder containing models.py
    CSV         : <that folder>/udise_panel_2023_to_2026_analysis_ready.csv
    outputs     : <that folder>/Output
                  (falls back to /tmp on read-only hosts)

Fits, for each dropout outcome (primary / upper primary / secondary):
  * Multiple Linear Regression (statsmodels, cluster-robust SEs by state)
  * Random Forest (sklearn, GroupKFold by state to avoid year leakage)

Statistical caveats:
  * n is small: up to 36 states x 3 years = 108 rows; primary and secondary
    dropout are missing for 2023-24 (NEP-structure booklet), so those models
    use ~72 rows.
  * Data are state-level (ecological) - results describe states, not students.
  * Three years is not enough to identify causal effects. Treat everything
    here as descriptive / exploratory.
  * Cluster-robust SEs correct for within-state autocorrelation across years,
    but with only 2-3 rows per cluster they are still noisy.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
#  PATHS - resolved relative to this file
# --------------------------------------------------------------------------- #

WORK_DIR = Path(__file__).resolve().parent
DATA_PATH = WORK_DIR / "udise_panel_2023_to_2026_analysis_ready.csv"

OUTPUT_DIR = WORK_DIR / "Output"
try:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
except (PermissionError, OSError):
    # Read-only host (e.g. Streamlit Cloud): write to a scratch folder.
    OUTPUT_DIR = Path("/tmp") / "udise_outputs"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"\n\n*** CSV NOT FOUND ***\n"
        f"Looked for: {DATA_PATH}\n"
        f"Files in {WORK_DIR}:\n"
        + "\n".join("    " + p.name for p in WORK_DIR.iterdir())
    )

DATA_PATH = str(DATA_PATH)
OUTPUT_DIR = str(OUTPUT_DIR)

print(f"DATA_PATH  = {DATA_PATH}")
print(f"OUTPUT_DIR = {OUTPUT_DIR}")


# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #

TARGETS: dict[str, str] = {
    "primary":       "dropout_primary_total",
    "upper_primary": "dropout_upper_primary_total",
    "secondary":     "dropout_secondary_total",
}

INFRA_PREDICTORS: list[str] = [
    "pct_func_electricity",
    "pct_func_drinking_water",
    "pct_func_girls_toilet",
    "pct_func_boys_toilet",
    "pct_func_toilet_any",
    "pct_handwash",
    "pct_library",
    "pct_medical_checkup",
    "pct_computer",
    "pct_internet",
    "pct_func_cwsn_toilet",
    "pct_ramp",
    "pct_kitchen_garden",
    "pct_integrated_science_lab",
    "pct_solar_panel",
    "pct_smart_classroom",
    "pct_digital_library",
]

OTHER_PREDICTORS: list[str] = [
    "ptr_primary",
    "ptr_upper_primary",
    "ptr_secondary",
    "pct_single_teacher_schools",
    "pct_female_teachers",
    "pct_govt_schools",
    "pct_enrol_sc",
    "pct_enrol_st",
    "pct_enrol_obc",
    "pct_enrol_muslim",
    "avg_enrolment_per_school",
]

VIF_THRESHOLD: float = 10.0
MAX_PREDICTORS: int = 8
N_SPLITS: int = 5
RANDOM_STATE: int = 42
RF_N_ESTIMATORS: int = 500
RF_MIN_SAMPLES_LEAF: int = 2


# --------------------------------------------------------------------------- #
#  Data loading and preparation
# --------------------------------------------------------------------------- #

def load_panel(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the analysis-ready panel and drop the national ('India') row, which
    is not a state observation and would inflate the R^2 of every model.
    """
    df = pd.read_csv(path, low_memory=False)
    df = df[df["geo_level"] != "National"].copy()
    df["year"] = df["year"].astype(str)
    return df.reset_index(drop=True)


def prepare_modeling_data(df, target, predictors):
    """
    Return (X, y, groups, meta) for a given target.

    * Columns not present in `df` are silently dropped from `predictors`.
    * Rows with any NaN in target or any retained predictor are dropped.
    * `meta` records how many rows were dropped and why, so the caller can
      print an honest accounting rather than silently discarding half the data.
    """
    available = [p for p in predictors if p in df.columns]
    missing_cols = [p for p in predictors if p not in df.columns]

    if not available:
        raise ValueError(f"No usable predictors for target {target!r}.")

    needed = ["state_ut", "year", target] + available
    sub = df[needed].copy()
    n_start = len(sub)

    missing_counts = {c: int(sub[c].isna().sum()) for c in [target] + available}

    sub = sub.dropna(subset=[target] + available)
    n_end = len(sub)

    meta = {
        "target": target,
        "predictors_used": available,
        "predictors_missing_from_csv": missing_cols,
        "n_start": n_start,
        "n_after_dropna": n_end,
        "n_dropped": n_start - n_end,
        "missing_per_column": missing_counts,
        "years_covered": sorted(sub["year"].unique().tolist()),
        "n_states": sub["state_ut"].nunique(),
    }

    X = sub[available].astype(float).reset_index(drop=True)
    y = sub[target].astype(float).reset_index(drop=True)
    groups = sub["state_ut"].reset_index(drop=True)

    return X, y, groups, meta


# --------------------------------------------------------------------------- #
#  VIF-based multicollinearity pruning
# --------------------------------------------------------------------------- #

def _compute_vif(X: pd.DataFrame) -> pd.Series:
    """
    Compute Variance Inflation Factors. Rows must be complete (no NaN).
    Columns with zero variance are dropped first (their VIF is undefined).
    Returns a Series indexed by column name.
    """
    X = X.loc[:, X.nunique() > 1]
    if X.shape[1] < 2:
        return pd.Series(dtype=float)

    Xc = sm.add_constant(X, has_constant="add")
    vifs = []
    for i in range(Xc.shape[1]):
        try:
            vifs.append(variance_inflation_factor(Xc.values, i))
        except Exception:
            vifs.append(np.nan)
    return pd.Series(vifs, index=Xc.columns).drop("const", errors="ignore")


def prune_by_vif(
    X: pd.DataFrame,
    threshold: float = VIF_THRESHOLD,
    max_predictors: int = MAX_PREDICTORS,
    verbose: bool = True,
) -> tuple[list[str], pd.DataFrame]:
    """
    Iteratively drop the predictor with the largest VIF until either:
      * all remaining VIFs <= threshold, AND
      * len(predictors) <= max_predictors.
    Returns (kept_columns, history_df).
    """
    cols = list(X.columns)
    history: list[dict] = []

    while len(cols) > 1:
        vif = _compute_vif(X[cols])
        if vif.empty:
            break
        vif = vif.sort_values(ascending=False)
        for name, val in vif.items():
            history.append({"variable": name, "vif": float(val)})
        worst_name = vif.index[0]
        worst_val = float(vif.iloc[0])

        if worst_val > threshold or len(cols) > max_predictors:
            if verbose:
                print(f"    drop {worst_name:<32s} VIF={worst_val:6.2f} "
                      f"(n_predictors={len(cols)})")
            cols.remove(worst_name)
        else:
            break

    return cols, pd.DataFrame(history)


# --------------------------------------------------------------------------- #
#  Multiple Linear Regression
# --------------------------------------------------------------------------- #

def fit_mlr(X, y, groups, cluster: bool = True):
    """
    Fit OLS on standardized X with (by default) cluster-robust SEs by state.

    Why standardize: coefficients become comparable across predictors on very
    different scales (percentages vs ratios), and the intercept has a clean
    interpretation (mean dropout at average predictor values). Standardizing
    does not change the OLS fit itself - R^2, p-values, and the sign/significance
    of coefficients are invariant to linear rescaling.
    """
    scaler = StandardScaler()
    Xs = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    Xs = sm.add_constant(Xs, has_constant="add")

    model = sm.OLS(y, Xs)
    if cluster:
        # Cluster SEs account for the fact that the same state appears in
        # multiple years: residuals within a state are correlated.
        res = model.fit(cov_type="cluster", cov_kwds={"groups": groups.values})
    else:
        res = model.fit()

    res._scaler = scaler
    res._feature_names = list(X.columns)
    return res


def mlr_oof_predictions(X, y, groups, n_splits: int = N_SPLITS) -> np.ndarray:
    """
    Out-of-fold predictions using GroupKFold by state.

    We split by state (not by row) so that no state appears in both train and
    test - otherwise the model would memorise a state and its held-out rows
    would be too easy.
    """
    n_groups = groups.nunique()
    k = min(n_splits, n_groups)
    gkf = GroupKFold(n_splits=k)
    preds = np.full(len(y), np.nan)

    for tr, te in gkf.split(X, y, groups):
        scaler = StandardScaler().fit(X.iloc[tr])
        Xtr = pd.DataFrame(scaler.transform(X.iloc[tr]), columns=X.columns,
                           index=X.index[tr])
        Xte = pd.DataFrame(scaler.transform(X.iloc[te]), columns=X.columns,
                           index=X.index[te])
        Xtr = sm.add_constant(Xtr, has_constant="add")
        Xte = sm.add_constant(Xte, has_constant="add")
        Xte = Xte.reindex(columns=Xtr.columns, fill_value=0.0)
        res = sm.OLS(y.iloc[tr], Xtr).fit()
        preds[te] = res.predict(Xte).values
    return preds


def mlr_summary_table(res) -> pd.DataFrame:
    """Readable coefficient table with p-values and 95% CIs."""
    ci = res.conf_int()
    tbl = pd.DataFrame({
        "variable": res.params.index,
        "coef_std": res.params.values,
        "std_err": res.bse.values,
        "t": res.tvalues.values,
        "p_value": res.pvalues.values,
        "ci_lower": ci[0].values,
        "ci_upper": ci[1].values,
    })
    tbl["signif"] = np.where(tbl["p_value"] < 0.01, "***",
                     np.where(tbl["p_value"] < 0.05, "**",
                      np.where(tbl["p_value"] < 0.10, "*", "")))
    return tbl


# --------------------------------------------------------------------------- #
#  Random Forest
# --------------------------------------------------------------------------- #

def fit_rf(X, y, groups, n_splits: int = N_SPLITS):
    """
    Fit a RandomForestRegressor.

    Evaluation uses GroupKFold by state so that a state's rows never appear in
    both train and test. We also compute permutation importance per test fold
    (fitted model -> permute one feature in the test set -> drop in R^2),
    averaging across folds. Built-in `feature_importances_` are reported from
    the model fit on the full data.
    """
    n_groups = groups.nunique()
    k = min(n_splits, n_groups)
    gkf = GroupKFold(n_splits=k)
    oof_pred = np.full(len(y), np.nan)

    perm_means = []
    for tr, te in gkf.split(X, y, groups):
        rf = RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        rf.fit(X.iloc[tr], y.iloc[tr])
        oof_pred[te] = rf.predict(X.iloc[te])

        pi = permutation_importance(
            rf, X.iloc[te], y.iloc[te],
            n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1,
        )
        perm_means.append(pi.importances_mean)

    # Full-data fit for built-in importances / deployment
    rf_full = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_full.fit(X, y)

    perm_mean = np.mean(perm_means, axis=0)
    perm_std = np.std(perm_means, axis=0)

    return {
        "model": rf_full,
        "oof_pred": oof_pred,
        "feature_names": list(X.columns),
        "builtin_importance": rf_full.feature_importances_,
        "perm_importance_mean": perm_mean,
        "perm_importance_std": perm_std,
    }


def rf_importance_table(rf_res: dict) -> pd.DataFrame:
    return pd.DataFrame({
        "variable": rf_res["feature_names"],
        "builtin_importance": rf_res["builtin_importance"],
        "perm_importance_mean": rf_res["perm_importance_mean"],
        "perm_importance_std": rf_res["perm_importance_std"],
    }).sort_values("perm_importance_mean", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
#  Metrics + plotting helpers
# --------------------------------------------------------------------------- #

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = ~np.isnan(y_pred)
    if mask.sum() < 3:
        return {"n": int(mask.sum()), "r2": np.nan,
                "rmse": np.nan, "mae": np.nan}
    return {
        "n": int(mask.sum()),
        "r2": float(r2_score(y_true[mask], y_pred[mask])),
        "rmse": float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))),
        "mae": float(mean_absolute_error(y_true[mask], y_pred[mask])),
    }


def plot_pred_vs_actual(y_true: np.ndarray, y_pred: np.ndarray,
                        title: str, path: str) -> None:
    mask = ~np.isnan(y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true[mask], y_pred[mask], alpha=0.7, edgecolor="k")
    lo = float(min(y_true[mask].min(), y_pred[mask].min()))
    hi = float(max(y_true[mask].max(), y_pred[mask].max()))
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax.set_xlabel("Actual dropout (%)")
    ax.set_ylabel("Predicted dropout (%)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_rf_importance(rf_res: dict, title: str, path: str) -> None:
    names = rf_res["feature_names"]
    means = rf_res["perm_importance_mean"]
    stds = rf_res["perm_importance_std"]
    order = np.argsort(means)
    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(names) + 1)))
    ax.barh([names[i] for i in order], [means[i] for i in order],
            xerr=[stds[i] for i in order], color="steelblue", alpha=0.85)
    ax.set_xlabel("Permutation importance (mean drop in R2, CV folds)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Orchestration
# --------------------------------------------------------------------------- #

def run_analysis(df: pd.DataFrame, stage: str, target: str) -> dict:
    """
    Run the full MLR + RF workflow for one target. Returns a dict with model
    results, tables, and metrics.
    """
    print(f"\n=== {stage.upper()} (target: {target}) ===")

    candidate_predictors = INFRA_PREDICTORS + OTHER_PREDICTORS
    X, y, groups, meta = prepare_modeling_data(df, target, candidate_predictors)

    print(f"  rows after dropna: {meta['n_after_dropna']} "
          f"(dropped {meta['n_dropped']} of {meta['n_start']})")
    print(f"  states: {meta['n_states']}, years: {meta['years_covered']}")

    # ---- VIF pruning among the candidate set -----------------------------
    print(f"  VIF pruning (threshold = {VIF_THRESHOLD}, "
          f"max_predictors = {MAX_PREDICTORS}):")
    kept_cols, vif_hist = prune_by_vif(X)
    if not kept_cols:
        print("  ! No predictors survived VIF pruning.")
        return {"stage": stage, "meta": meta, "error": "no predictors survived"}

    X_kept = X[kept_cols].copy()
    print(f"  kept {len(kept_cols)} predictors: {kept_cols}")

    # ---- MLR -------------------------------------------------------------
    mlr_res = fit_mlr(X_kept, y, groups, cluster=True)
    mlr_tbl = mlr_summary_table(mlr_res)
    print("\n  MLR (cluster-robust SEs by state):")
    print(mlr_tbl.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"  R2 (in-sample) = {mlr_res.rsquared:.4f}, "
          f"adj R2 = {mlr_res.rsquared_adj:.4f}, "
          f"n = {int(mlr_res.nobs)}")

    mlr_oof = mlr_oof_predictions(X_kept, y, groups)
    mlr_metrics = regression_metrics(y.values, mlr_oof)
    print(f"  MLR OOF: R2={mlr_metrics['r2']:.3f}, "
          f"RMSE={mlr_metrics['rmse']:.3f}, MAE={mlr_metrics['mae']:.3f}")

    # ---- RF --------------------------------------------------------------
    rf_res = fit_rf(X_kept, y, groups)
    rf_tbl = rf_importance_table(rf_res)
    print("\n  RF feature importance (permutation, mean over CV folds):")
    print(rf_tbl.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    rf_metrics = regression_metrics(y.values, rf_res["oof_pred"])
    print(f"  RF OOF: R2={rf_metrics['r2']:.3f}, "
          f"RMSE={rf_metrics['rmse']:.3f}, MAE={rf_metrics['mae']:.3f}")

    # ---- save artefacts --------------------------------------------------
    mlr_tbl.to_csv(os.path.join(OUTPUT_DIR, f"mlr_coefficients_{stage}.csv"),
                   index=False)
    rf_tbl.to_csv(os.path.join(OUTPUT_DIR, f"rf_importances_{stage}.csv"),
                  index=False)
    vif_hist.to_csv(os.path.join(OUTPUT_DIR, f"vif_history_{stage}.csv"),
                    index=False)
    plot_pred_vs_actual(
        y.values, mlr_oof,
        f"MLR - {stage} dropout (out-of-fold)",
        os.path.join(OUTPUT_DIR, f"mlr_pred_vs_actual_{stage}.png"),
    )
    plot_pred_vs_actual(
        y.values, rf_res["oof_pred"],
        f"Random Forest - {stage} dropout (out-of-fold)",
        os.path.join(OUTPUT_DIR, f"rf_pred_vs_actual_{stage}.png"),
    )
    plot_rf_importance(
        rf_res, f"RF permutation importance - {stage}",
        os.path.join(OUTPUT_DIR, f"rf_importance_{stage}.png"),
    )

    return {
        "stage": stage,
        "meta": meta,
        "kept_cols": kept_cols,
        "mlr": mlr_res,
        "mlr_table": mlr_tbl,
        "mlr_metrics": mlr_metrics,
        "mlr_oof": mlr_oof,
        "rf": rf_res,
        "rf_table": rf_tbl,
        "rf_metrics": rf_metrics,
        "y_true": y.values,
    }


def main() -> None:
    print(f"Loading panel from {DATA_PATH} ...")
    df = load_panel(DATA_PATH)
    print(f"  shape after dropping India: {df.shape}")
    print(f"  years: {sorted(df['year'].unique())}")
    print(f"  states/UTs: {df['state_ut'].nunique()}")

    all_results = {}
    for stage, target in TARGETS.items():
        all_results[stage] = run_analysis(df, stage, target)

    print("\n=============== SUMMARY ===============")
    for stage, r in all_results.items():
        if "error" in r:
            print(f"{stage:<14s}  ERROR: {r['error']}")
            continue
        print(f"{stage:<14s}  "
              f"n={r['meta']['n_after_dropna']:>3d}  "
              f"k={len(r['kept_cols']):>2d}  "
              f"MLR OOF R2={r['mlr_metrics']['r2']:.3f}  "
              f"RF  OOF R2={r['rf_metrics']['r2']:.3f}")

    print(f"\nArtefacts written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()