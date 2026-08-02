"""Interactive dashboard for the market-signal-dashboard project.

Reads the artifacts produced by the offline pipeline (src/model.py, src/backtest.py —
run manually or by the scheduled GitHub Action) and adds one live piece: a
same-day model score computed from freshly fetched price data.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_ingestion import fetch_prices
from src.features import FEATURE_COLUMNS, build_feature_dataset
from src.theme import (
    CHART_BUY_HOLD,
    CHART_LINE,
    CHART_SIGNAL_MARKER,
    CHART_STRATEGY,
    CUSTOM_CSS,
    style_figure,
)

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
BACKTEST_METRICS_PATH = ROOT / "reports" / "backtest_metrics.json"
BACKTEST_PREDICTIONS_PATH = ROOT / "reports" / "backtest_predictions.csv"

st.set_page_config(page_title="Market Signal Dashboard", page_icon=":material/monitoring:", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


@st.cache_data
def load_backtest_predictions() -> pd.DataFrame:
    return pd.read_csv(BACKTEST_PREDICTIONS_PATH, parse_dates=["date"])


@st.cache_data(ttl=3600)
def fetch_live_features(ticker: str) -> pd.DataFrame:
    prices = fetch_prices([ticker], period="6mo")
    return build_feature_dataset(prices)


def missing_artifacts_notice() -> bool:
    missing = [p.name for p in [MODEL_PATH, METRICS_PATH, BACKTEST_METRICS_PATH, BACKTEST_PREDICTIONS_PATH] if not p.exists()]
    if missing:
        st.error(
            "Missing pipeline artifacts: "
            + ", ".join(missing)
            + ". Run `python -m src.model` and `python -m src.backtest` first "
            "(or wait for the scheduled GitHub Action to generate them)."
        )
        return True
    return False


def stat_card(label: str, value: str, sub: str = "") -> str:
    return f"""
    <div class="stat-card">
        <div class="stat-card-label">{label}</div>
        <div class="stat-card-value">{value}</div>
        <div class="stat-card-sub">{sub}</div>
    </div>
    """


def get_live_signal(ticker: str) -> tuple[float | None, str | None, str | None]:
    """Returns (proba_up, as_of_date_str, error_message)."""
    try:
        live_feats = fetch_live_features(ticker)
        latest_row = live_feats.iloc[[-1]]
        model = load_model()
        proba_up = float(model.predict_proba(latest_row[FEATURE_COLUMNS + ["ticker"]])[0, 1])
        as_of = str(latest_row["date"].iloc[0].date())
        return proba_up, as_of, None
    except Exception as exc:  # live fetch can fail (rate limit, holiday, etc.) — don't crash the dashboard
        return None, None, str(exc)


def main() -> None:
    st.title(":material/candlestick_chart: Market Signal Dashboard")
    st.caption(
        "Live data ingestion → next-day-direction model → backtest, all in one pipeline. "
        "Educational portfolio project — not investment advice."
    )

    if missing_artifacts_notice():
        st.stop()

    metrics = load_json(METRICS_PATH)
    backtest_metrics = load_json(BACKTEST_METRICS_PATH)
    predictions = load_backtest_predictions()
    tickers = metrics["tickers"]

    with st.sidebar:
        st.header(":material/tune: Controls")
        ticker = st.selectbox("Ticker", tickers)
        threshold = st.slider("Signal threshold (P(up day) to go long)", 0.3, 0.7, 0.5, 0.01)
        st.markdown("---")
        st.caption(f"**Model:** {metrics['best_model']}")
        st.caption(f"**Last trained:** {metrics['trained_at_utc'][:19]} UTC")
        st.caption(f"**Training rows:** {metrics['n_rows']}")

    proba_up, as_of, live_error = get_live_signal(ticker)
    ticker_metrics = backtest_metrics["by_ticker"].get(ticker, {})
    strat_sharpe = ticker_metrics.get("strategy", {}).get("sharpe", 0.0)
    bh_sharpe = ticker_metrics.get("buy_hold", {}).get("sharpe", 0.0)

    # --- Top KPI row -----------------------------------------------------
    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        if proba_up is not None:
            badge_class, badge_icon, badge_text = (
                ("long", ":material/trending_up:", "LONG") if proba_up > threshold else ("flat", ":material/trending_flat:", "FLAT")
            )
            value_html = f"{proba_up:.1%}"
            sub_html = f'<span class="signal-badge {badge_class}">{badge_text}</span> as of {as_of}'
        else:
            value_html, sub_html = "—", f"live fetch failed: {live_error}"
        st.markdown(stat_card("P(next close up)", value_html, sub_html), unsafe_allow_html=True)
    with kpi_cols[1]:
        st.markdown(stat_card("Model", metrics["best_model"].replace("_", " ").title(), "walk-forward selected"), unsafe_allow_html=True)
    with kpi_cols[2]:
        st.markdown(stat_card("Strategy Sharpe", f"{strat_sharpe:.2f}", f"vs buy-hold {bh_sharpe:.2f}"), unsafe_allow_html=True)
    with kpi_cols[3]:
        st.markdown(stat_card("Mean ROC AUC", f"{metrics['comparison'][metrics['best_model']]['mean_roc_auc']:.3f}", "walk-forward mean"), unsafe_allow_html=True)

    st.write("")
    ticker_predictions = predictions[predictions["ticker"] == ticker].sort_values("date")
    scored = ticker_predictions.dropna(subset=["next_day_return"]).copy()
    scored["signal"] = (scored["proba_up"] > threshold).astype(int)
    scored["strategy_return"] = scored["signal"] * scored["next_day_return"]
    scored["strategy_curve"] = (1 + scored["strategy_return"]).cumprod()
    scored["buy_hold_curve"] = (1 + scored["next_day_return"]).cumprod()

    tab_overview, tab_backtest, tab_model = st.tabs(
        [":material/show_chart: Overview", ":material/query_stats: Backtest", ":material/insights: Model Insights"]
    )

    with tab_overview:
        st.subheader(f"{ticker} — price & out-of-sample signal")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ticker_predictions["date"], y=ticker_predictions["close"], name="Close", line=dict(color=CHART_LINE, width=2)))
        longs = ticker_predictions[ticker_predictions["proba_up"] > threshold]
        fig.add_trace(
            go.Scatter(
                x=longs["date"], y=longs["close"], mode="markers", name="Long signal",
                marker=dict(color=CHART_SIGNAL_MARKER, size=7, symbol="triangle-up"),
            )
        )
        st.plotly_chart(style_figure(fig, height=420), use_container_width=True)

    with tab_backtest:
        st.subheader("Strategy vs buy-and-hold (out-of-sample only)")
        perf_col, chart_col = st.columns([1, 2])
        with perf_col:
            strat_m, bh_m = ticker_metrics.get("strategy", {}), ticker_metrics.get("buy_hold", {})
            st.dataframe(
                pd.DataFrame({"Strategy": strat_m, "Buy & Hold": bh_m}).T[["total_return", "sharpe", "max_drawdown"]],
                use_container_width=True,
            )
        with chart_col:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=scored["date"], y=scored["strategy_curve"], name="Strategy", line=dict(color=CHART_STRATEGY, width=2)))
            fig2.add_trace(go.Scatter(x=scored["date"], y=scored["buy_hold_curve"], name="Buy & Hold", line=dict(color=CHART_BUY_HOLD, width=2, dash="dot")))
            st.plotly_chart(style_figure(fig2, height=340), use_container_width=True)

    with tab_model:
        st.subheader("Model comparison (walk-forward validation, mean across folds)")
        comparison_rows = {
            name: {
                "mean_accuracy": vals["mean_accuracy"],
                "mean_roc_auc": vals["mean_roc_auc"],
                "mean_precision": vals["mean_precision"],
                "mean_recall": vals["mean_recall"],
            }
            for name, vals in metrics["comparison"].items()
        }
        st.dataframe(pd.DataFrame(comparison_rows).T, use_container_width=True)

        st.subheader("Feature importance")
        importance = pd.Series(metrics["feature_importance"]).sort_values()
        fig3 = go.Figure(go.Bar(x=importance.values, y=importance.index, orientation="h", marker=dict(color=CHART_LINE)))
        st.plotly_chart(style_figure(fig3, height=350), use_container_width=True)

    st.info(
        "**Read the numbers honestly**: next-day direction from daily technical indicators is close to "
        "an efficient-markets coin flip (ROC AUC near 0.5 above). That's expected, not a bug — the value "
        "of this project is the pipeline (live ingestion → walk-forward validated training → backtest → "
        "dashboard), not a claim of alpha."
    )


if __name__ == "__main__":
    main()
