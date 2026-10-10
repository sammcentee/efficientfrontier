"""Shared visual styles for the app, charts, and reports."""

import plotly.graph_objects as go


FONT = '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
COLORS = {"Minimum volatility": "#3977b8", "Maximum Sharpe": "#91683b", "Equal weight": "#737b85",
          "Low": "#3977b8", "Medium": "#175bc0", "Extreme": "#825ca6"}
BENCHMARK_COLORS = {"S&P 500 (SPY)": "#555c65", "Nasdaq-100 (QQQ)": "#947245"}

APP_CSS = """<style>
.stApp, button, input, textarea, select {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.block-container {max-width: 1160px; padding-top: 2rem; padding-bottom: 3rem;}
h1, h2, h3 {letter-spacing: -.035em; color: #1d1d1f;}
h1 {font-weight: 650 !important; line-height: 1.12 !important;}
h2 {font-weight: 600 !important;}
[data-testid="stMetric"] {background: #fff; border: 1px solid #e6e7eb;
  border-radius: 16px; padding: 18px 20px;}
[data-testid="stMetricValue"] {font-size: 1.7rem; letter-spacing: -.035em;}
[data-testid="stMetricLabel"] {color: #62666e;}
[data-testid="stSidebar"] {border-right: 1px solid #e6e7eb;}
[data-testid="stExpander"] {background: #fff; border-radius: 14px;}
[data-testid="stPlotlyChart"] {background: #fff; border-radius: 16px; overflow: hidden;}
[data-testid="stCaptionContainer"] {color: #62666e; opacity: 1;}
[data-testid="stButton"] button, [data-testid="stFormSubmitButton"] button,
[data-testid="stDownloadButton"] button {border-radius: 10px; min-height: 42px;}
[data-baseweb="tab-list"] {gap: 1.5rem;}
[data-baseweb="tab"] {padding: .75rem 0;}
@media (max-width: 600px) {
  .block-container {padding: 1.2rem 1rem 2rem;}
  h1 {font-size: 2.25rem !important;}
  [data-testid="stMetric"] {padding: 14px 16px;}
  [data-baseweb="tab-list"] {gap: 1.1rem;}
}
</style>"""


def style_chart(figure: go.Figure, title: str, *, height: int = 460,
                hovermode: str = "closest") -> go.Figure:
    """Apply a light theme without changes to chart data."""
    figure.update_layout(
        title={"text": title, "font": {"size": 17}, "x": .04, "xanchor": "left"},
        template="plotly_white", paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font={"family": FONT, "color": "#35383f", "size": 12},
        colorway=list(COLORS.values()),
        margin={"l": 55, "r": 18, "t": 55, "b": 105},
        legend={"orientation": "h", "y": -.22, "x": 0, "xanchor": "left",
                "font": {"size": 11}, "maxheight": 85},
        height=height, hovermode=hovermode,
        hoverlabel={"bgcolor": "#ffffff", "font": {"color": "#1d1d1f"}},
    )
    figure.update_xaxes(gridcolor="#eceef1", zerolinecolor="#d8dce2", automargin=True)
    figure.update_yaxes(gridcolor="#eceef1", zerolinecolor="#d8dce2", automargin=True)
    return figure
