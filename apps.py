"""
app.py
======
Streamlit dashboard for the UDISE+ dropout panel.

Run from a terminal:
    cd C:\\Users\\DELL\\Downloads\\files
    streamlit run app.py
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from models import (
    DATA_PATH,
    OUTPUT_DIR,
    TARGETS,
    INFRA_PREDICTORS,
    OTHER_PREDICTORS,
    load_panel,
    prepare_modeling_data,
    prune_by_vif,
    fit_mlr,
    mlr_summary_table,
    mlr_oof_predictions,
    fit_rf,
    rf_importance_table,
    regression_metrics,
)

st.set_page_config(page_title="UDISE+ Dropout Dashboard", layout="wide")


# --------------------------------------------------------------------------- #
#  Cached loaders / fitters
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False)
def cached_panel(path: str = DATA_PATH) -> pd.DataFrame:
    return load_panel(path)


@st.cache_data(show_spinner=False)
def cached_analysis(stage: str, candidate_predictors: tuple) -> dict:
    df = cached_panel()
    target = TARGETS[stage]
    X, y, groups, meta = prepare_modeling_data(df, target, list(candidate_predictors))

    kept_cols, _ = prune_by_vif(X, verbose=False)
    if not kept_cols:
        return {"error": "no predictors survived VIF pruning", "meta": meta}

    X_kept = X[kept_cols]
    mlr_res = fit_mlr(X_kept, y, groups, cluster=True)
    mlr_tbl = mlr_summary_table(mlr_res)
    mlr_oof = mlr_oof_predictions(X_kept, y, groups)
    mlr_metrics = regression_metrics(y.values, mlr_oof)

    rf_res = fit_rf(X_kept, y, groups)
    rf_tbl = rf_importance_table(rf_res)
    rf_metrics = regression_metrics(y.values, rf_res["oof_pred"])

    return {
        "meta": meta,
        "kept_cols": kept_cols,
        "mlr_table": mlr_tbl,
        "mlr_metrics": mlr_metrics,
        "rf_table": rf_tbl,
        "rf_metrics": rf_metrics,
        "mlr_r2": float(mlr_res.rsquared),
        "mlr_adj_r2": float(mlr_res.rsquared_adj),
        "n_obs": int(mlr_res.nobs),
    }


# --------------------------------------------------------------------------- #
#  Sidebar
# --------------------------------------------------------------------------- #

panel = cached_panel()

st.sidebar.title("UDISE+ dropout dashboard")
st.sidebar.caption(
    "Descriptive / exploratory only. State-level (ecological) data, small n, "
    "not causal."
)

stage = st.sidebar.selectbox(
    "Dropout stage",
    options=list(TARGETS.keys()),
    format_func=lambda s: s.replace("_", " ").title(),
)

years = sorted(panel["year"].unique().tolist())
year = st.sidebar.selectbox("Year", years, index=len(years) - 1)

state_options = sorted(panel["state_ut"].unique().tolist())
default_state = "Karnataka" if "Karnataka" in state_options else state_options[0]
state = st.sidebar.selectbox("State / UT (trend view)", state_options,
                             index=state_options.index(default_state))

scatter_var = st.sidebar.selectbox(
    "Infrastructure variable (scatter)",
    options=[c for c in INFRA_PREDICTORS if c in panel.columns],
    format_func=lambda c: c.replace("pct_", "").replace("_", " ").title(),
)

candidate_predictors = tuple(INFRA_PREDICTORS + OTHER_PREDICTORS)

with st.spinner("Fitting models (cached after first run) ..."):
    analysis = cached_analysis(stage, candidate_predictors)


# --------------------------------------------------------------------------- #
#  Header
# --------------------------------------------------------------------------- #

target_col = TARGETS[stage]
st.title(f"{stage.replace('_', ' ').title()} dropout - UDISE+ panel 2023-24 to 2025-26")

if "error" in analysis:
    st.error(f"Model could not be fitted: {analysis['error']}")
    st.stop()

meta = analysis["meta"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows used", meta["n_after_dropna"])
c2.metric("States / UTs", meta["n_states"])
c3.metric("Predictors (post-VIF)", len(analysis["kept_cols"]))
c4.metric("Years covered", ", ".join(meta["years_covered"]))

st.caption(
    "2023-24 is missing for this stage in the NEP-structure booklet, so it is "
    "excluded automatically. VIF-pruned predictor set: "
    + ", ".join(analysis["kept_cols"]) + "."
)

st.divider()


# --------------------------------------------------------------------------- #
#  Row 1: by-state bar + scatter
# --------------------------------------------------------------------------- #

col_bar, col_scatter = st.columns(2)

with col_bar:
    st.subheader(f"Dropout by state - {year}")
    snap = panel[panel["year"] == year][["state_ut", target_col]].dropna()
    snap = snap.sort_values(target_col, ascending=True)
    fig = px.bar(
        snap, x=target_col, y="state_ut", orientation="h",
        labels={target_col: f"{stage} dropout (%)", "state_ut": ""},
        height=650,
    )
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "No shapefile is bundled, so a ranked bar chart is used instead of a "
        "choropleth. To add a map, place a state-level GeoJSON next to app.py "
        "and swap in px.choropleth."
    )

with col_scatter:
    st.subheader(f"{scatter_var} vs {stage} dropout")
    scat = panel[["state_ut", "year", scatter_var, target_col]].dropna()
    if scat.empty:
        st.info("No overlapping non-null rows for that pair.")
    else:
        fig = px.scatter(
            scat, x=scatter_var, y=target_col, color="year",
            hover_name="state_ut", trendline="ols",
            labels={scatter_var: scatter_var,
                    target_col: f"{stage} dropout (%)"},
        )
        fig.update_traces(marker=dict(size=9, line=dict(width=0.5, color="white")))
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------- #
#  Row 2: performance + tables
# --------------------------------------------------------------------------- #

st.divider()
st.subheader("Model performance (out-of-fold, GroupKFold by state)")
p1, p2, p3, p4 = st.columns(4)
p1.metric("MLR OOF R2",  f"{analysis['mlr_metrics']['r2']:.3f}")
p2.metric("MLR RMSE",    f"{analysis['mlr_metrics']['rmse']:.3f}")
p3.metric("RF OOF R2",   f"{analysis['rf_metrics']['r2']:.3f}")
p4.metric("RF RMSE",     f"{analysis['rf_metrics']['rmse']:.3f}")

st.caption(
    "GroupKFold by state means no state appears in both train and test. "
    "RMSE / MAE are in percentage points of dropout. In-sample MLR R2 for "
    f"reference is {analysis['mlr_r2']:.3f}."
)

st.divider()
col_mlr, col_rf = st.columns(2)

with col_mlr:
    st.subheader("MLR coefficients (standardized X, cluster-robust SEs)")
    show = analysis["mlr_table"][
        ["variable", "coef_std", "std_err", "t", "p_value",
         "ci_lower", "ci_upper", "signif"]
    ].copy()
    st.dataframe(show, use_container_width=True, hide_index=True)
    st.caption(
        "Coefficients are per 1-SD change in the predictor. Cluster SEs are "
        "grouped by state. *** p<0.01, ** p<0.05, * p<0.10."
    )

with col_rf:
    st.subheader("Random Forest feature importances")
    st.dataframe(
        analysis["rf_table"][
            ["variable", "builtin_importance",
             "perm_importance_mean", "perm_importance_std"]
        ],
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "Built-in importance = mean decrease in impurity. Permutation "
        "importance is computed on the held-out fold of each GroupKFold "
        "split, averaged across folds."
    )


# --------------------------------------------------------------------------- #
#  Row 3: trend for a single state
# --------------------------------------------------------------------------- #

st.divider()
st.subheader(f"Year-over-year trend - {state}")
trend = (
    panel[panel["state_ut"] == state][["year", target_col]]
    .dropna().sort_values("year")
)
if trend.empty:
    st.info(f"No {stage} dropout data for {state} in this panel.")
else:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trend["year"], y=trend[target_col],
        mode="lines+markers", name=f"{state} - {stage}",
    ))
    nat = panel[panel["geo_level"] == "National"][["year", target_col]].dropna()
    if not nat.empty:
        fig.add_trace(go.Scatter(
            x=nat["year"], y=nat[target_col],
            mode="lines+markers", name="India (national)",
            line=dict(dash="dash", color="grey"),
        ))
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title=f"{stage} dropout (%)",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------- #
#  Footer
# --------------------------------------------------------------------------- #

st.divider()
with st.expander("How to read this dashboard"):
    st.markdown(
        """
        * **Data.** 36 states/UTs x 3 years from the UDISE+ reports. 2023-24
          was published under the NEP structure, so primary (I-V) and secondary
          (IX-X) dropout are blank for that year; those models use only
          2024-25 and 2025-26 (~72 rows).
        * **Targets.** `_total` dropout rates. `_boys` / `_girls` and the
          `_nep` columns exist in the CSV but are not used here.
        * **Predictors.** Infrastructure shares (Section 7 / 9) plus a small
          set of teacher / school-structure covariates. VIF pruning drops
          collinear variables before fitting.
        * **Why standardize.** Predictors are on different scales. Standardizing
          makes coefficients comparable without changing the OLS fit.
        * **Why cluster SEs.** Each state appears up to three times; cluster-
          robust SEs by state correct for within-state correlation.
        * **Why GroupKFold.** Splitting by state means the model is never
          tested on a state it saw in training.
        * **Caveats.** Small n, three years, state-level data. Read
          associations, not causal effects.
        """
    )

st.caption(f"Data file: {os.path.basename(DATA_PATH)}")