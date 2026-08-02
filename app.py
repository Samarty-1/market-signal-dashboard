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

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
BACKTEST_METRICS_PATH = ROOT / "reports" / "backtest_metrics.json"
BACKTEST_PREDICTIONS_PATH = ROOT / "reports" / "backtest_predictions.csv"

st.set_page_config(page_title="Market Signal Dashboard", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


@st.cache_data
def load_backtest_predictions() -> pd.DataFrame:
    df = pd.read_csv(BACKTEST_PREDICTIONS_PATH, parse_dates=["date"])
    return df


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


def main() -> None:
    st.title("📈 Market Signal Dashboard")
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
        st.header("Controls")
        ticker = st.selectbox("Ticker", tickers)
        threshold = st.slider("Signal threshold (P(up day) to go long)", 0.3, 0.7, 0.5, 0.01)
        st.markdown("---")
        st.caption(f"Model: **{metrics['best_model']}**")
        st.caption(f"Last trained: {metrics['trained_at_utc'][:19]} UTC")
        st.caption(f"Training rows: {metrics['n_rows']}")

    live_col, signal_col = st.columns([3, 1])

    with signal_col:
        st.subheader("Live signal")
        try:
            live_feats = fetch_live_features(ticker)
            latest_row = live_feats.iloc[[-1]]
            model = load_model()
            proba_up = model.predict_proba(latest_row[FEATURE_COLUMNS + ["ticker"]])[0, 1]
            as_of = latest_row["date"].iloc[0].date()
            st.metric("P(next close > today's close)", f"{proba_up:.1%}")
            st.write("🟢 Long signal" if proba_up > threshold else "⚪ Flat / no signal")
            st.caption(f"As of close on {as_of} (fetched live just now)")
        except Exception as exc:  # live fetch can fail (rate limit, holiday, etc.) — don't crash the dashboard
            st.warning(f"Live fetch failed, showing historical data only: {exc}")

    ticker_predictions = predictions[predictions["ticker"] == ticker].sort_values("date")

    with live_col:
        st.subheader(f"{ticker} — price & out-of-sample signal")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ticker_predictions["date"], y=ticker_predictions["close"], name="Close", line=dict(color="#4C78A8")))
        longs = ticker_predictions[ticker_predictions["proba_up"] > threshold]
        fig.add_trace(
            go.Scatter(
                x=longs["date"], y=longs["close"], mode="markers", name="Long signal",
                marker=dict(color="#54A24B", size=6, symbol="triangle-up"),
            )
        )
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Backtest: strategy vs buy-and-hold (out-of-sample only)")

    scored = ticker_predictions.dropna(subset=["next_day_return"]).copy()
    scored["signal"] = (scored["proba_up"] > threshold).astype(int)
    scored["strategy_return"] = scored["signal"] * scored["next_day_return"]
    scored["strategy_curve"] = (1 + scored["strategy_return"]).cumprod()
    scored["buy_hold_curve"] = (1 + scored["next_day_return"]).cumprod()

    perf_col, chart_col = st.columns([1, 2])
    with perf_col:
        ticker_metrics = backtest_metrics["by_ticker"].get(ticker, {})
        strat_m, bh_m = ticker_metrics.get("strategy", {}), ticker_metrics.get("buy_hold", {})
        st.table(
            pd.DataFrame(
                {"Strategy": strat_m, "Buy & Hold": bh_m}
            ).T[["total_return", "sharpe", "max_drawdown"]]
        )
    with chart_col:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=scored["date"], y=scored["strategy_curve"], name="Strategy"))
        fig2.add_trace(go.Scatter(x=scored["date"], y=scored["buy_hold_curve"], name="Buy & Hold"))
        fig2.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
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
    st.table(pd.DataFrame(comparison_rows).T)

    st.subheader("Feature importance")
    importance = pd.Series(metrics["feature_importance"]).sort_values()
    fig3 = go.Figure(go.Bar(x=importance.values, y=importance.index, orientation="h"))
    fig3.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig3, use_container_width=True)

    st.info(
        "**Read the numbers honestly**: next-day direction from daily technical indicators is close to "
        "an efficient-markets coin flip (ROC AUC near 0.5 above). That's expected, not a bug — the value "
        "of this project is the pipeline (live ingestion → walk-forward validated training → backtest → "
        "dashboard), not a claim of alpha."
    )


if __name__ == "__main__":
    main()
