"""Shared visual styles for the app, charts, and reports."""

import plotly.graph_objects as go


FONT = '-apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI Variable Text", "Segoe UI", Inter, Roboto, "Helvetica Neue", Arial, sans-serif'
INK, INK_2, INK_3, INK_4 = "#1d1d1f", "#424245", "#6e6e73", "#86868b"
GRID, REST, ACCENT, FILL, CANVAS = "#e8e8ed", "#c7c7cc", "#0071e3", "#f2f2f4", "#f5f5f7"
COLORS = {"Low": "#5e9eea", "Medium": "#0071e3", "Extreme": "#0a3f8f",
          "Minimum volatility": "#5e9eea", "Equal weight": "#0071e3", "Maximum Sharpe": "#0a3f8f"}
BENCHMARK_COLORS = {"S&P 500 (SPY)": "#8e8e93", "Nasdaq-100 (QQQ)": "#545458"}
PLOTLY_CONFIG = {"displayModeBar": False, "displaylogo": False, "scrollZoom": False,
                 "doubleClick": False, "responsive": True}


def display_label(name) -> str:
    """Show the engine profile "Extreme" as "Highest". Data keys do not change."""
    text = str(name)
    return "Highest" if text == "Extreme" else text.replace(" · Extreme", " · Highest")


def style_chart(figure: go.Figure, title: str | None = None, *, height: int = 420,
                hovermode: str = "closest", story: bool = False) -> go.Figure:
    """Apply the Portfolio Lab theme. Chart data does not change."""
    figure.update_layout(
        template="none", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title=({"text": title, "x": 0, "xref": "paper", "xanchor": "left", "y": 1, "yref": "container",
                "yanchor": "top", "pad": {"t": 6}, "font": {"family": FONT, "size": 15, "color": INK}}
               if title and not story else {"text": ""}),
        font={"family": FONT, "size": 12, "color": INK_3},
        colorway=[ACCENT, "#545458", "#8e8e93"],
        margin={"l": 8, "r": 64 if story else 16, "t": 48 if story else (44 if title else 16), "b": 8 if story else 88},
        legend=({"orientation": "h", "x": 0, "xanchor": "left", "y": 1.02, "yanchor": "bottom"} if story else
                {"orientation": "h", "x": 0, "xanchor": "left", "y": -0.16, "yanchor": "top", "maxheight": 80})
               | {"font": {"size": 12, "color": INK_2}, "title": {"text": ""}, "bgcolor": "rgba(0,0,0,0)"},
        hovermode=hovermode, hoverdistance=24, spikedistance=-1,
        hoverlabel={"bgcolor": INK, "bordercolor": INK, "align": "left",
                    "font": {"family": FONT, "size": 13, "color": "#ffffff"}},
        height=height, dragmode=False, barcornerradius=4, bargap=0.35,
    )
    axis = {"showline": False, "zeroline": False, "ticks": "", "automargin": True, "fixedrange": True,
            "tickfont": {"size": 12, "color": INK_3}, "title": {"font": {"size": 12, "color": INK_3}}}
    figure.update_xaxes(**axis, showgrid=False, showspikes=hovermode == "x unified", spikemode="across",
                        spikesnap="cursor", spikethickness=1, spikecolor=INK_4, spikedash="solid")
    figure.update_yaxes(**axis, showgrid=True, gridcolor=GRID, gridwidth=1, nticks=6)
    return figure


TOKENS_CSS = """
:root{--pl-canvas:#f5f5f7;--pl-tile:#fff;--pl-ink:#1d1d1f;--pl-ink-2:#424245;--pl-ink-3:#6e6e73;--pl-ink-4:#86868b;
--pl-hair:#d2d2d7;--pl-hair-soft:#e8e8ed;--pl-fill:#f2f2f4;--pl-track:#e3e3e8;--pl-accent:#0071e3;--pl-accent-wash:rgba(0,113,227,.10);
--pl-rest:#c7c7cc;--pl-spy:#8e8e93;--pl-qqq:#545458;--pl-risk-low:#5e9eea;--pl-risk-med:#0071e3;--pl-risk-high:#0a3f8f;
--pl-error:#c4001a;--pl-error-wash:#fff0f0;--pl-error-ink:#6e000e;
--pl-font:-apple-system,BlinkMacSystemFont,"SF Pro Text","Segoe UI Variable Text","Segoe UI",Inter,Roboto,"Helvetica Neue",Arial,sans-serif;
--pl-display:-apple-system,BlinkMacSystemFont,"SF Pro Display","Segoe UI Variable Display","Segoe UI",Inter,Roboto,"Helvetica Neue",Arial,sans-serif;
--pl-ease:cubic-bezier(.25,.1,.25,1);--pl-ease-out:cubic-bezier(.2,.8,.2,1);
--pl-r-tile:28px;--pl-pad-tile:40px;--pl-gap:96px;--pl-focus:0 0 0 4px rgba(0,113,227,.35)}
@media (max-width:640px){:root{--pl-r-tile:22px;--pl-pad-tile:22px;--pl-gap:64px}}
"""

APP_ONLY_CSS = """
/* 1 Chrome */
[data-testid="stHeader"]{display:none}
[data-testid="stHeaderActionElements"]{display:none}
.modebar-container{display:none!important}
[data-testid="stMain"]{overflow-x:hidden}
[data-testid="stMainBlockContainer"]{max-width:1072px;padding:0 24px 64px}
.stApp{font-family:var(--pl-font);letter-spacing:-.01em;-webkit-font-smoothing:antialiased}
:focus-visible{outline:none;box-shadow:var(--pl-focus)!important;border-radius:8px}

/* 2 Stale content and work */
[data-stale="true"]{opacity:.55!important;transition:opacity .25s var(--pl-ease) .4s}
.stApp:has(.st-key-work_progress) [data-stale="true"],
.stApp:has(.st-key-study_work) [data-stale="true"]{opacity:.35!important;pointer-events:none}

/* 3 Nav bar */
.st-key-localnav,[data-testid="stLayoutWrapper"]:has(>.st-key-localnav){position:sticky;top:0;z-index:990}
.st-key-localnav{min-height:52px;align-items:center}
.st-key-localnav::before{content:"";position:absolute;inset:0;z-index:-1;background:rgba(245,245,247,.8);
 -webkit-backdrop-filter:saturate(180%) blur(20px);backdrop-filter:saturate(180%) blur(20px);
 box-shadow:0 0 0 100vmax rgba(245,245,247,.8);clip-path:inset(0 -100vmax)}
.st-key-localnav::after{content:"";position:absolute;left:0;right:0;bottom:0;height:1px;z-index:-1;background:rgba(0,0,0,.08);
 box-shadow:0 0 0 100vmax rgba(0,0,0,.08);clip-path:inset(0 -100vmax)}
.pl-wordmark{font:600 19px/1 var(--pl-display);letter-spacing:-.02em;white-space:nowrap;color:var(--pl-ink)}
.st-key-view [data-testid="stButtonGroup"] [role="radiogroup"]{background:none;padding:0;gap:20px}
.st-key-view [data-testid="stButtonGroup"] button[data-variant="segmented_control"]{background:none!important;border:0!important;box-shadow:none!important;
 min-height:32px;padding:0 2px;color:var(--pl-ink-3)}
.st-key-view [data-testid="stButtonGroup"] button[data-variant="segmented_control"] p{font-size:13px;font-weight:400}
.st-key-view [data-testid="stButtonGroup"] button[aria-checked="true"] p{color:var(--pl-ink);font-weight:600}
.st-key-export button{min-height:32px;padding:0 14px}
.st-key-export button p{font-size:13px}
.st-key-keys{display:none!important}

/* 4 Hero */
.st-key-hero{text-align:center;padding:72px 0 48px}
.pl-context{font-size:15px;color:var(--pl-ink-3)}
.st-key-edit_setup button{border:1px solid var(--pl-hair)!important;background:#fff;min-height:30px;padding:2px 12px}
.st-key-risk_profile{max-width:396px;margin:0 auto}

/* 5 Segmented controls */
[data-testid="stButtonGroup"] [role="radiogroup"]{background:var(--pl-track);border-radius:9px;padding:2px;gap:0}
[data-testid="stButtonGroup"] button[data-variant="segmented_control"]{border:0!important;border-radius:7px!important;
 background:transparent;min-height:30px;color:var(--pl-ink);box-shadow:none;transition:background .16s var(--pl-ease),box-shadow .16s var(--pl-ease)}
[data-testid="stButtonGroup"] button[data-variant="segmented_control"] p{font-size:13px;font-weight:500}
[data-testid="stButtonGroup"] button[aria-checked="true"]{background:#fff!important;box-shadow:0 3px 8px rgba(0,0,0,.12),0 3px 1px rgba(0,0,0,.04)!important}
[data-testid="stButtonGroup"] button[aria-checked="true"] p{font-weight:600}
.st-key-risk_profile [role="radiogroup"]{border-radius:12px;padding:3px}
.st-key-risk_profile button[data-variant="segmented_control"]{min-height:44px;border-radius:9px!important;flex:1}
.st-key-risk_profile button[data-variant="segmented_control"] p{font-size:17px}

/* 6 Tiles */
[class*="st-key-tile_"]{background:var(--pl-tile)!important;border:0!important;border-radius:var(--pl-r-tile)!important;
 padding:var(--pl-pad-tile)!important}
[class*="st-key-tile_"] [data-testid="stElementToolbar"]{display:none}
@supports (animation-timeline:view()){@media (prefers-reduced-motion:no-preference){
 [class*="st-key-tile_"]{animation:pl-rise linear both;animation-timeline:view();animation-range:entry 0% cover 22%}}}
@keyframes pl-rise{from{opacity:0;transform:translateY(28px)}}

/* 7 Expanders as disclosure rows (1.65: details > summary > span > [icon span, label div]) */
[data-testid="stExpander"] details{border:0;border-top:1px solid var(--pl-hair-soft);border-radius:0;background:transparent}
[data-testid="stExpander"] summary{padding:18px 0}
[data-testid="stExpander"] summary>span{flex-direction:row-reverse;justify-content:space-between;width:100%}
[data-testid="stExpander"] summary p{font-size:17px;font-weight:500}
[data-testid="stExpander"] summary .stMarkdownColoredText{margin-left:8px;font-size:15px;font-weight:400;color:var(--pl-ink-3)!important}
[data-testid="stExpander"] summary:hover{background:transparent;color:var(--pl-ink)}
[data-testid="stExpander"] summary span:has(>[data-testid="stIconMaterial"]){display:grid;place-items:center;flex:none;width:28px;height:28px;
 border-radius:50%;background:var(--pl-fill)}
[data-testid="stExpanderDetails"]{padding:0 0 16px;color:var(--pl-ink-2)}

/* 8 Buttons, inputs, alerts, tables, captions */
button[data-testid="stBaseButton-primary"]{background:var(--pl-ink);border-color:var(--pl-ink)}
button[data-testid="stBaseButton-primary"]:hover{background:#000;border-color:#000}
button[data-testid="stBaseButton-tertiary"]{color:var(--pl-ink)}
.st-key-build button{min-height:48px;padding:0 28px}
.st-key-build button p{font-size:17px;font-weight:500}
[data-testid="stAlertContainer"]{border-radius:16px}
[data-testid="stDataFrame"]{border-radius:12px;overflow:hidden}
[data-testid="stCaptionContainer"]{color:var(--pl-ink-3);opacity:1}
.st-key-mustread [data-testid="stCaptionContainer"]{font-size:17px;line-height:1.47;color:var(--pl-ink)}
@media (hover:none){[data-testid="stElementToolbar"],[data-has-shortcut="true"] kbd{display:none}}

/* 9 Drawer and popovers */
[data-testid="stDialog"] [role="dialog"]{background:var(--pl-tile);border-radius:24px 0 0 24px;box-shadow:0 30px 80px rgba(0,0,0,.2),0 0 0 1px rgba(0,0,0,.04)}
[data-testid="stPopoverBody"]{border:0;border-radius:18px;width:min(380px,calc(100vw - 24px));max-height:min(78vh,720px);
 overflow-y:auto;box-shadow:0 30px 80px rgba(0,0,0,.18),0 0 0 1px rgba(0,0,0,.05)}
[data-testid="stDialog"] [role="dialog"]:focus-visible{box-shadow:0 30px 80px rgba(0,0,0,.2),0 0 0 1px rgba(0,0,0,.04)!important}
.st-key-market{width:100%}
.st-key-market [role="radiogroup"]{gap:0;border-radius:14px;box-shadow:inset 0 0 0 1px var(--pl-hair-soft)}
.st-key-market [role="radiogroup"]>div{align-self:stretch;margin:0;padding:12px 16px;border-top:1px solid var(--pl-hair-soft)}
.st-key-market [role="radiogroup"]>div:first-child{border-top:0}

/* 10 Work card */
.st-key-hero [data-testid="stProgress"]{max-width:360px;margin:6px auto 0}
.st-key-cancel_build,.st-key-cancel_study{display:flex;justify-content:center}

/* 11 Anchors */
[id^="pl-"]{scroll-margin-top:72px}
.pl-section[id^="pl-"]{scroll-margin-top:calc(72px - var(--pl-gap))}

/* 12 Reduced motion */
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}

/* 13 Mobile */
@media (max-width:640px){
 [data-testid="stMainBlockContainer"]{padding:0 16px 48px}
 .st-key-hero{padding:36px 0 32px}
 .pl-wordmark{font-size:17px}
 .st-key-localnav{min-height:48px}
 .st-key-view [data-testid="stButtonGroup"] [role="radiogroup"]{gap:14px}
 .st-key-export [data-testid="stMarkdownContainer"]{display:none}
 .st-key-export button{width:36px;min-width:36px;height:36px;min-height:36px;padding:0;justify-content:center}
 .st-key-export button kbd{display:none}
 .st-key-risk_profile{max-width:none}
 .st-key-risk_profile button[data-variant="segmented_control"] p{font-size:16px}
 [data-testid="stDialog"] [role="dialog"]{border-radius:24px 24px 0 0}
}
"""

STORY_CSS = """
/* Context, headlines and leads */
.pl-context{display:flex;flex-wrap:wrap;justify-content:center;align-items:center;gap:4px 10px;margin:0;
 font-size:15px;line-height:20px;color:var(--pl-ink-3)}
.pl-sep{color:var(--pl-hair)}
.pl-badge{display:inline-block;padding:5px 9px;border-radius:980px;background:var(--pl-fill);color:var(--pl-ink-2);
 font:600 12px/1 var(--pl-font);letter-spacing:.02em;white-space:nowrap}
.pl-display{margin:0 auto;max-width:900px;font:650 56px/60px var(--pl-display);letter-spacing:-.028em;color:var(--pl-ink);text-wrap:balance}
.pl-working{color:var(--pl-ink-2)}
.pl-nb{white-space:nowrap}
.pl-note{margin:0 auto;max-width:560px;font-size:15px;line-height:20px;color:var(--pl-ink-3)}
.pl-section{padding-top:var(--pl-gap)}
.pl-eyebrow{margin:0 0 10px;font:600 17px/22px var(--pl-font);letter-spacing:-.01em;color:var(--pl-ink-3)}
.pl-answer{width:fit-content;margin:0 0 16px;max-width:860px;font:650 48px/52px var(--pl-display);letter-spacing:-.024em;color:var(--pl-ink)}
.pl-page{margin:0 0 12px;font:650 40px/44px var(--pl-display);letter-spacing:-.022em;color:var(--pl-ink)}
.pl-lead{margin:0;max-width:760px;font:400 21px/31px var(--pl-font);letter-spacing:-.012em;color:var(--pl-ink-2)}
.pl-title{margin:0;font:600 21px/26px var(--pl-display);letter-spacing:-.015em;color:var(--pl-ink)}
.pl-head{display:flex;flex-wrap:wrap;align-items:baseline;justify-content:space-between;gap:8px 16px;margin:0 0 12px}
.pl-head small,.pl-small{font-size:13px;line-height:18px;letter-spacing:-.005em;color:var(--pl-ink-3)}
.pl-small{margin:12px 0 0}
.pl-sub{margin:0;font-size:15px;line-height:20px;color:var(--pl-ink-3)}

/* Stats strip */
.pl-stats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));margin:0 0 32px}
.pl-stat{min-width:0}
.pl-stats>.pl-stat{padding:0 24px;border-left:1px solid var(--pl-hair-soft)}
.pl-stats>.pl-stat:first-child{padding-left:0;border-left:0}
.pl-num{display:block;font:600 40px/44px var(--pl-display);letter-spacing:-.02em;color:var(--pl-ink);font-variant-numeric:tabular-nums}
.pl-lbl{display:block;margin-top:6px;font:600 15px/20px var(--pl-font);letter-spacing:-.005em;color:var(--pl-ink)}
.pl-stat small{display:block;margin-top:2px;font-size:13px;line-height:18px;color:var(--pl-ink-3)}

/* Holdings list */
ul.pl-list{list-style:none;margin:0;padding:0}
.pl-row{display:grid;grid-template-columns:minmax(160px,320px) 1fr 72px;grid-template-areas:"n b p";align-items:center;gap:16px;
 padding:11px 0;border-top:1px solid var(--pl-hair-soft)}
.pl-row:first-child{border-top:0}
.pl-name{grid-area:n;display:flex;flex-direction:column;min-width:0}
.pl-name b{overflow:hidden;font-size:17px;line-height:20px;font-weight:500;color:var(--pl-ink);white-space:nowrap;text-overflow:ellipsis}
.pl-name small{font-size:13px;line-height:16px;color:var(--pl-ink-3)}
.pl-bar{grid-area:b;height:8px;overflow:hidden;border-radius:999px;background:var(--pl-fill)}
.pl-bar i{display:block;height:100%;border-radius:999px;background:var(--pl-accent);transform-origin:left center;
 animation:pl-grow .52s var(--pl-ease-out) both;animation-delay:calc(var(--i,0) * 28ms)}
.is-rest .pl-bar i{background:var(--pl-rest)}
.pl-pct{grid-area:p;text-align:right;font:500 17px/20px var(--pl-font);color:var(--pl-ink);font-variant-numeric:tabular-nums}
@keyframes pl-grow{from{transform:scaleX(0)}}

/* Concentration callout and must-read note */
.pl-callout{display:flex;gap:14px;align-items:flex-start;margin:0 0 24px;padding:16px 20px;border-radius:16px;background:var(--pl-fill)}
.pl-glyph{flex:none;display:grid;place-items:center;width:24px;height:24px;border-radius:50%;background:var(--pl-ink);color:#fff;
 font:700 14px/1 var(--pl-font)}
.pl-callout p{margin:0;font-size:15px;line-height:20px;color:var(--pl-ink-2)}
.pl-callout p b{display:block;margin-bottom:2px;font-size:17px;line-height:22px;font-weight:600;color:var(--pl-ink)}
.pl-mustread{display:flex;gap:10px;align-items:flex-start;max-width:820px;margin:16px 0 0;font-size:15px;line-height:21px;color:var(--pl-ink-2)}
.pl-mustread svg{flex:none;width:18px;height:18px;margin-top:2px;color:var(--pl-ink-3)}
.pl-mustread b{font-weight:600;color:var(--pl-ink)}

/* Levels ledger */
.pl-ledger{width:100%;border-collapse:collapse;font-size:15px;line-height:20px;font-variant-numeric:tabular-nums}
.pl-ledger th{padding:0 12px 10px;background:none;border:0;text-align:right;white-space:nowrap;
 font:500 13px/18px var(--pl-font);color:var(--pl-ink-3)}
.pl-ledger td{padding:12px;border:0;border-top:1px solid var(--pl-hair-soft);text-align:right;color:var(--pl-ink)}
.pl-ledger th:first-child,.pl-ledger td:first-child{padding-left:12px;text-align:left;white-space:nowrap}
.pl-ledger tr.on td{background:var(--pl-canvas);font-weight:600}
.pl-ledger tr.on td:first-child{border-radius:10px 0 0 10px}
.pl-ledger tr.on td:last-child{border-radius:0 10px 10px 0}
.pl-ldot{display:inline-block;width:10px;height:10px;margin-right:10px;border-radius:50%}

/* Result tile */
.pl-kicker{display:block;margin:0 0 18px;font:500 13px/18px var(--pl-font);letter-spacing:.01em;color:var(--pl-ink-3)}
.pl-bignums{display:grid;grid-auto-flow:column;grid-auto-columns:minmax(0,1fr)}
.pl-bignum{min-width:0;padding:0 28px;border-left:1px solid var(--pl-hair-soft)}
.pl-bignum:first-child{padding-left:0;border-left:0}
.pl-key{display:inline-flex;align-items:center;gap:8px;font:600 15px/20px var(--pl-font);color:var(--pl-ink)}
.pl-key i{display:inline-block;width:14px;height:3px;border-radius:2px;background:var(--pl-accent)}
.pl-key i.rule{height:4px}
.pl-key i.spy{background:var(--pl-spy)}
.pl-key i.qqq{background:var(--pl-qqq)}
.pl-bignum .pl-num{margin:10px 0 4px;font-size:48px;line-height:53px;letter-spacing:-.022em}
.pl-bignum small{display:block;font-size:15px;line-height:20px;color:var(--pl-ink-3);font-variant-numeric:tabular-nums}
.pl-verdict{max-width:820px;margin:32px 0 0;font:600 24px/31px var(--pl-display);letter-spacing:-.016em;color:var(--pl-ink)}
.pl-verdict span{font-weight:500;color:var(--pl-ink-3)}
.pl-clarity{max-width:820px;margin:8px 0 0;font-size:17px;line-height:25px;color:var(--pl-ink-2)}

/* Interval chart */
.pl-forest{margin:8px 0 0}
.pl-forest .f-sides,.pl-forest .f-axis{display:grid;grid-template-columns:132px 1fr;gap:18px}
.pl-forest .f-sides{margin-bottom:6px;font-size:12px;line-height:16px;color:var(--pl-ink-3)}
.pl-forest .f-sides .mid{position:relative;height:16px}
.pl-forest .f-sides .mid span{position:absolute;top:0;white-space:nowrap}
.pl-forest .f-row{display:grid;grid-template-columns:132px 1fr;grid-template-areas:"l p" "v p";align-items:center;gap:2px 18px;padding:14px 0}
.pl-forest .f-label{grid-area:l;align-self:end;font:600 15px/20px var(--pl-font);color:var(--pl-ink)}
.pl-forest .f-value{grid-area:v;align-self:start;font-size:14px;line-height:20px;color:var(--pl-ink-2);font-variant-numeric:tabular-nums}
.pl-forest .f-value b{font-weight:600;color:var(--pl-ink)}
.pl-forest .f-value small{display:block;font-size:13px;line-height:18px;color:var(--pl-ink-3)}
.pl-forest .f-plot{grid-area:p;position:relative;height:28px}
.pl-forest .f-zero{position:absolute;top:-16px;bottom:-16px;width:1px;background:var(--pl-ink-4)}
.pl-forest .f-ci{position:absolute;top:13px;height:2px;border-radius:2px;background:var(--pl-ink-3)}
.pl-forest .f-ci::before,.pl-forest .f-ci::after{content:"";position:absolute;top:-5px;width:2px;height:12px;border-radius:1px;background:var(--pl-ink-3)}
.pl-forest .f-ci::before{left:0}
.pl-forest .f-ci::after{right:0}
.pl-forest .f-dot{position:absolute;top:7px;width:14px;height:14px;margin-left:-7px;border-radius:50%;background:var(--pl-ink);box-shadow:0 0 0 2px #fff}
.pl-forest .f-axis{margin-top:4px}
.pl-forest .f-axis .ticks{position:relative;height:34px;border-top:1px solid var(--pl-hair-soft)}
.pl-forest .f-axis .ticks span{position:absolute;top:8px;transform:translateX(-50%);font-size:12px;line-height:16px;color:var(--pl-ink-3);
 font-variant-numeric:tabular-nums}

/* Stretch table */
.pl-wtable{width:100%;border-collapse:collapse;font-size:15px;line-height:20px;font-variant-numeric:tabular-nums}
.pl-wtable th{padding:0 8px 10px 0;background:none;border:0;text-align:left;vertical-align:bottom;font:500 12px/16px var(--pl-font);color:var(--pl-ink-3)}
.pl-wtable td{padding:12px 8px 12px 0;border:0;border-top:1px solid var(--pl-hair-soft);text-align:left;white-space:nowrap;color:var(--pl-ink)}
.pl-wtable td small{display:block;font-size:13px;line-height:18px;color:var(--pl-ink-3)}
.pl-wtable th:first-child,.pl-wtable td:first-child{font-weight:600;white-space:nowrap}
.w-dot{display:inline-block;width:10px;height:10px;margin-right:8px;border-radius:50%;box-shadow:inset 0 0 0 1.5px var(--pl-ink-3)}
.w-dot.on{background:var(--pl-ink);box-shadow:none}

/* Evidence extras */
.pl-settle{margin:0}
.pl-settle p{max-width:760px;margin:8px 0 0;font-size:17px;line-height:25px;color:var(--pl-ink-2)}
.pl-words{font-size:13px;line-height:18px;color:var(--pl-ink-3);white-space:nowrap}

/* Fine print and footer */
.pl-fine{display:grid;grid-template-columns:1fr 1fr;gap:10px 48px;margin-top:var(--pl-gap);padding-top:28px;border-top:1px solid var(--pl-hair)}
.pl-fine h3{grid-column:1/-1;margin:0 0 6px;font:600 15px/20px var(--pl-font);color:var(--pl-ink)}
.pl-fine p{max-width:none;margin:0;font-size:13px;line-height:18px;color:var(--pl-ink-3)}
.pl-fine p b{font-weight:600;color:var(--pl-ink-2)}
.pl-foot{display:flex;flex-wrap:wrap;justify-content:space-between;gap:6px 18px;margin:0;font-size:12px;line-height:16px;color:var(--pl-ink-3)}
.pl-keys kbd{padding:3px 6px;border-radius:6px;background:#fff;box-shadow:inset 0 0 0 1px var(--pl-hair);font:500 11px/1 var(--pl-font);color:var(--pl-ink-3)}
@media (hover:none){.pl-keys{display:none}}

/* Work card */
ol.pl-steps{max-width:440px;margin:0 auto;padding:0;list-style:none;text-align:left}
.pl-steps li{position:relative;display:grid;grid-template-columns:20px 1fr;gap:12px;padding:0 0 18px;font-size:17px;line-height:20px;color:var(--pl-ink-3)}
.pl-steps li:last-child{padding-bottom:0}
.pl-steps li:not(:last-child)::after{content:"";position:absolute;top:26px;bottom:4px;left:9.5px;width:1px;background:var(--pl-hair-soft)}
.pl-steps i{box-sizing:border-box;display:grid;place-items:center;width:20px;height:20px;border-radius:50%;
 box-shadow:inset 0 0 0 1.5px var(--pl-rest);font:700 14px/1 var(--pl-font);font-style:normal}
.pl-steps .done,.pl-steps .now{color:var(--pl-ink)}
.pl-steps .now{font-weight:500}
.pl-steps .done i{box-shadow:none;color:var(--pl-ink)}
.pl-steps .now i{box-shadow:none;border:2px solid var(--pl-track);border-top-color:var(--pl-ink);animation:pl-spin 1s linear infinite}
.pl-steps small{display:block;margin-top:2px;font-size:13px;line-height:18px;font-weight:400;color:var(--pl-ink-3)}
@keyframes pl-spin{to{transform:rotate(360deg)}}
.pl-skel{height:14px;border-radius:7px;background:linear-gradient(90deg,var(--pl-fill) 0%,#fafafb 50%,var(--pl-fill) 100%);
 background-size:200% 100%;animation:pl-shimmer 1.4s linear infinite}
.pl-skel-row{display:grid;grid-template-columns:minmax(160px,320px) 1fr 72px;gap:16px;align-items:center;padding:20px 0;
 border-top:1px solid var(--pl-hair-soft)}
.pl-skel-row:first-child{border-top:0}
@keyframes pl-shimmer{to{background-position:-200% 0}}

/* Welcome preview */
ol.pl-preview{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:24px;margin:48px 0 0;padding:24px 0 0;
 border-top:1px solid var(--pl-hair-soft);list-style:none;counter-reset:pl;text-align:left}
.pl-preview li{counter-increment:pl;font-size:15px;line-height:20px;color:var(--pl-ink)}
.pl-preview li::before{content:counter(pl);display:block;margin-bottom:6px;font:600 13px/18px var(--pl-font);color:var(--pl-ink-3)}
.pl-preview b{display:block;font-weight:600}
.pl-preview span{display:block;margin-top:4px;font-size:13px;line-height:18px;color:var(--pl-ink-3)}

/* Utilities (last, so they win over the component margins) */
.pl-center{margin-left:auto;margin-right:auto;text-align:center;justify-content:center}

@media (prefers-reduced-motion:reduce){.pl-bar i,.pl-steps .now i,.pl-skel{animation:none}}

@media (max-width:640px){
 .pl-context{font-size:14px}
 .pl-display{font-size:34px;line-height:37px}
 .pl-display br{display:none}
 .pl-answer{margin-bottom:12px;font-size:30px;line-height:34px}
 .pl-page{font-size:30px;line-height:34px}
 .pl-eyebrow{margin-bottom:8px;font-size:15px;line-height:20px}
 .pl-lead{font-size:17px;line-height:25px}
 .pl-title{font-size:19px;line-height:24px}
 .pl-num{font-size:26px;line-height:29px}
 .pl-stats{margin-bottom:24px}
 .pl-stats>.pl-stat{padding:0 10px}
 .pl-stats .pl-lbl{font-size:13px;line-height:18px}
 .pl-stats small{display:none}
 .pl-row{grid-template-columns:minmax(0,1fr) auto;grid-template-areas:"n p" "b b";gap:6px 12px;padding:12px 0}
 .pl-name b{font-size:16px}
 .pl-bar{height:6px}
 .pl-pct{font-size:16px}
 .hide-m{display:none}
 .pl-ledger{font-size:14px}
 .pl-ledger th{white-space:normal;font-size:12px;line-height:16px}
 .pl-ledger th,.pl-ledger td{padding:8px}
 .pl-ledger th:first-child,.pl-ledger td:first-child{padding-left:8px}
 .pl-bignum{padding:0 8px}
 .pl-bignum .pl-num{margin:8px 0 2px;font-size:24px;line-height:28px}
 .pl-bignum small{font-size:12px;line-height:16px}
 .pl-bignum small span{display:block}
 .pl-bignum small .pl-sep{display:none}
 .pl-key{gap:6px;font-size:12px;line-height:16px}
 .pl-key i{width:10px}
 .pl-verdict{margin-top:24px;font-size:19px;line-height:25px}
 .pl-mustread{font-size:14px;line-height:20px}
 .pl-forest .f-row{grid-template-columns:1fr auto;grid-template-areas:"l v" "p p";gap:8px 12px;padding:14px 0 18px}
 .pl-forest .f-label,.pl-forest .f-value{align-self:start}
 .pl-forest .f-value{text-align:right}
 .pl-forest .f-zero{top:-4px;bottom:-4px}
 .pl-forest .f-axis,.pl-forest .f-sides{grid-template-columns:1fr}
 .pl-forest .f-axis>div:not(.ticks),.pl-forest .f-sides>div:not(.mid){display:none}
 .pl-wtable{font-size:13px}
 .pl-wtable th{font-size:11px}
 .pl-wtable td{white-space:normal}
 .pl-fine{grid-template-columns:1fr}
 .pl-skel-row{grid-template-columns:minmax(0,1fr) 50px}
 .pl-skel-row .pl-skel:nth-child(2){display:none}
 ol.pl-preview{grid-template-columns:1fr}
}
"""

REPORT_ONLY_CSS = """
:root{color-scheme:light}
*{box-sizing:border-box}
body{margin:0;background:var(--pl-canvas);color:var(--pl-ink);font:400 16px/1.55 var(--pl-font);letter-spacing:-.01em;
 -webkit-font-smoothing:antialiased}
main{max-width:1072px;margin:auto;padding:48px 24px}
h1{margin:6px 0 20px;font:650 40px/44px var(--pl-display);letter-spacing:-.022em}
h2{margin:40px 0 12px;font:600 26px/31px var(--pl-display);letter-spacing:-.02em}
h3{margin:28px 0 10px;font:600 19px/24px var(--pl-display);letter-spacing:-.015em}
p{max-width:860px}
b,strong{font-weight:600}
.muted{color:var(--pl-ink-3)}
.eyebrow{margin:0;font-size:13px;line-height:18px;color:var(--pl-ink-3)}
.source,aside{margin:16px 0;padding:16px 20px;border-radius:16px;background:var(--pl-fill)}
aside h2,aside h3{margin-top:0}
.tile,main>section:not(#summary){margin:16px 0;padding:32px;border-radius:20px;background:var(--pl-tile)}
.tile>:first-child,main>section>:first-child,main>section>section>:first-child{margin-top:0}
.tile>:last-child,main>section>:last-child{margin-bottom:0}
.tile>.pl-sub{margin:6px 0 16px}
.tile>.pl-stat{margin:16px 0 8px}
.chart{margin:24px 0;overflow:hidden}
.scroll{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}
th,td{padding:10px 12px;border-bottom:1px solid var(--pl-hair-soft);text-align:right}
thead th{background:var(--pl-canvas);font-weight:500;color:var(--pl-ink-3)}
tbody th{font-weight:500}
th:first-child,td:first-child{text-align:left}
.metadata th{width:30%;text-align:left}
.metadata td{text-align:left;overflow-wrap:anywhere}
footer{margin-top:32px;font-size:13px;color:var(--pl-ink-3)}
pre{white-space:pre-wrap}
summary{padding:12px 0;font-weight:500;cursor:pointer}
button{padding:10px 18px;border:0;border-radius:980px;background:var(--pl-ink);color:#fff;font:500 15px/1 var(--pl-font);cursor:pointer}
hr{margin:64px 0 12px;border:0;border-top:1px solid var(--pl-hair)}
#summary{margin:32px 0 0}
#summary .pl-context{justify-content:flex-start}
#summary>h2{margin:12px 0 0;font:650 40px/44px var(--pl-display);letter-spacing:-.022em}
#summary .pl-eyebrow{margin-top:64px}
#summary .pl-answer{margin-bottom:12px;font-size:32px;line-height:37px}
#summary .pl-lead{margin-bottom:16px;font-size:19px;line-height:28px}
.pl-read{display:grid;grid-template-columns:1fr 1fr;gap:10px 40px}
.pl-read h3{grid-column:1/-1;margin:0 0 4px;font-size:15px;line-height:20px}
.pl-read p{margin:0;font-size:14px;line-height:20px;color:var(--pl-ink-2)}
.pl-cols{display:grid;grid-template-columns:7fr 5fr;gap:16px}
.pl-cols>.tile{margin:0}
@media print{
@page{size:A4 landscape;margin:12mm}
:root{color-scheme:light}
body{background:white;color:var(--pl-ink);font-size:10pt}
main{max-width:none;padding:0}
h1{font-size:25pt}
h2{break-after:avoid}
.tile,main>section:not(#summary){border:1px solid var(--pl-hair-soft)}
.source,aside{background:var(--pl-canvas)}
.muted,footer{color:var(--pl-ink-2)}
.chart,#summary .tile,.pl-cols{break-inside:avoid;background:white}
.scroll{overflow:visible}
table{font-size:8pt}
th,td{padding:5px}
tr{break-inside:avoid}
thead{display:table-header-group}
button,.modebar{display:none!important}
details{display:none}
.metadata{font-size:8pt}
}
@media screen and (max-width:640px){
main{padding:24px 16px}
h1{font-size:30px;line-height:34px}
h2{font-size:21px;line-height:26px}
.tile,main>section:not(#summary){padding:20px}
#summary>h2{font-size:30px;line-height:34px}
#summary .pl-answer{font-size:26px;line-height:30px}
.pl-read,.pl-cols{grid-template-columns:1fr}
.source,aside{padding:12px}
th,td{padding:8px}
}
"""

APP_CSS = TOKENS_CSS + APP_ONLY_CSS + STORY_CSS
REPORT_CSS = TOKENS_CSS + REPORT_ONLY_CSS + STORY_CSS
