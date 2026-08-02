"""Design tokens for the dashboard — dark fintech theme.

Colors and type pairing come from a design-system pass (dark mode primary,
Fira Code/Fira Sans, green=bullish/red=bearish per standard financial-chart
convention). Centralized here so app.py and any future page share one source
of truth instead of hardcoded hex values scattered through UI code.
"""

from __future__ import annotations

import plotly.graph_objects as go

BACKGROUND = "#0B1120"
SURFACE = "#111827"
SURFACE_ALT = "#1A2333"
BORDER = "#243049"
FOREGROUND = "#F8FAFC"
MUTED = "#94A3B8"

ACCENT = "#22C55E"       # bullish / positive / primary CTA
BEARISH = "#EF4444"      # bearish / negative
NEUTRAL = "#64748B"      # flat / no-signal

CHART_LINE = "#38BDF8"           # price line
CHART_SIGNAL_MARKER = "#22C55E"  # long-signal markers
CHART_STRATEGY = "#22C55E"
CHART_BUY_HOLD = "#94A3B8"

FONT_FAMILY = "'Fira Sans', -apple-system, sans-serif"
MONO_FONT_FAMILY = "'Fira Code', 'SFMono-Regular', monospace"

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600;700&family=Fira+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: {FONT_FAMILY};
}}

/* Fade-in for the whole page body on load/rerun — subtle, not decorative-only:
   it signals "fresh data just rendered" after a ticker/threshold change. */
.main .block-container {{
    animation: fadeIn 350ms ease-out;
}}
@keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(6px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}
@media (prefers-reduced-motion: reduce) {{
    .main .block-container {{ animation: none; }}
}}

/* Stat card grid used for the top KPI row */
.stat-card {{
    background: linear-gradient(160deg, {SURFACE} 0%, {SURFACE_ALT} 100%);
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 16px 18px;
    transition: border-color 200ms ease, transform 200ms ease;
    height: 100%;
}}
.stat-card:hover {{
    border-color: {ACCENT}66;
    transform: translateY(-2px);
}}
.stat-card-label {{
    font-family: {MONO_FONT_FAMILY};
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: {MUTED};
    margin-bottom: 6px;
}}
.stat-card-value {{
    font-family: {MONO_FONT_FAMILY};
    font-size: 1.6rem;
    font-weight: 600;
    color: {FOREGROUND};
    line-height: 1.2;
}}
.stat-card-sub {{
    font-size: 0.78rem;
    color: {MUTED};
    margin-top: 4px;
}}

/* Signal badge */
.signal-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 999px;
    font-family: {MONO_FONT_FAMILY};
    font-size: 0.85rem;
    font-weight: 600;
}}
.signal-badge.long {{
    background: {ACCENT}22;
    color: {ACCENT};
    border: 1px solid {ACCENT}55;
}}
.signal-badge.flat {{
    background: {NEUTRAL}22;
    color: {MUTED};
    border: 1px solid {NEUTRAL}55;
}}

hr {{ border-color: {BORDER}; }}
</style>
"""


def style_figure(fig: go.Figure, height: int = 380) -> go.Figure:
    """Apply the dark theme to a Plotly figure so it matches the page background."""
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_FAMILY, color=FOREGROUND, size=13),
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
        hovermode="x unified",
    )
    return fig
