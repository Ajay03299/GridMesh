"""Session-local dashboard palettes; never change research plotting defaults."""
from io import BytesIO

import altair as alt
import matplotlib.pyplot as plt
import streamlit as st


def palette(dark):
    if dark:
        return dict(background="#0b1220", surface="#152033", ink="#e5edf7",
                    muted="#a8b6ca", grid="#334155", blue="#60a5fa",
                    orange="#fb923c", fault="#fb7185",
                    healthy=["#e2e8f0", "#a8b6ca", "#8395ad", "#cbd5e1"])
    return dict(background="#f5f7fa", surface="#ffffff", ink="#172b40",
                muted="#526579", grid="#dce3eb", blue="#174a72",
                orange="#c95115", fault="#bd2444",
                healthy=["#334155", "#64748b", "#8995a5", "#475569"])


def apply_theme(colors, dark):
    """Theme visible controls and overlays without changing global server config."""
    scheme = "dark" if dark else "light"
    st.html(f"""<style>
    :root {{ color-scheme: {scheme}; }}
    .stMainBlockContainer {{ max-width: 1480px; padding-top: 2.2rem;
        padding-bottom: 3rem; }}
    [data-testid="stSidebar"] {{ border-right: 1px solid {colors['grid']}; }}
    h1 {{ font-size: 2.3rem !important; letter-spacing: -0.045em;
        font-weight: 750 !important; }}
    h2, h3 {{ letter-spacing: -0.025em; }}
    [data-testid="stMetricLabel"] {{ font-size: .85rem; font-weight: 600;
        min-height: 38px; align-items: flex-start; }}
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] div,
    [data-testid="stMetricLabel"] p {{ white-space: normal !important;
        overflow: visible !important; text-overflow: clip !important; }}
    [data-testid="stMetricLabel"] p {{ min-height: 2.4em; line-height: 1.2; }}
    .stApp, [data-testid="stHeader"], [data-testid="stMain"] {{
        background: {colors['background']}; color: {colors['ink']}; }}
    [data-testid="stSidebar"], [data-testid="stBottom"] {{
        background: {colors['surface']}; color: {colors['ink']}; }}
    h1, h2, h3, h4, label, [data-testid="stMarkdownContainer"],
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"],
    [data-testid="stWidgetLabel"], [data-testid="stExpander"] summary {{
        color: {colors['ink']}; }}
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
    [data-testid="stCaption"], [data-testid="stCaption"] p {{
        color: {colors['muted']}; opacity: 1; }}
    [data-testid="stMetricValue"] {{ font-size: 2rem; font-weight: 700;
        letter-spacing: -.035em; }}
    [data-testid="stMetric"] {{ background: {colors['surface']};
        border: 1px solid {colors['grid']}; border-radius: 10px;
        padding: 18px 20px; min-height: 104px; }}
    [data-testid="stMetric"] [data-testid="stMetricValue"] {{ color: {colors['blue']}; }}
    [data-testid="stExpander"] {{ background: {colors['surface']};
        border-color: {colors['grid']}; }}
    [data-testid="stExpander"] summary {{ background: {colors['surface']} !important; }}
    [data-baseweb="select"] div, [data-baseweb="popover"],
    [data-baseweb="menu"], [role="listbox"], [role="option"],
    [data-testid="stBaseButton-headerNoPadding"], [data-testid="stBaseButton-secondary"] {{
        background-color: {colors['surface']} !important; color: {colors['ink']} !important; }}
    [data-baseweb="select"] svg {{ fill: {colors['ink']}; }}
    [data-baseweb="select"] input {{ background: {colors['surface']} !important;
        color: {colors['ink']} !important; }}
    [data-testid="stSelectbox"] [data-baseweb="select"] > div {{
        background-color: {colors['surface']} !important;
        color: {colors['ink']} !important; border-color: {colors['grid']} !important; }}
    [role="group"]:has(input[role="combobox"]),
    [role="group"]:has(input[role="combobox"]) input,
    [role="group"]:has(input[role="combobox"]) button {{
        background-color: {colors['surface']} !important;
        color: {colors['ink']} !important; border-color: {colors['grid']} !important; }}
    [data-testid="stTabs"] button {{ color: {colors['muted']}; }}
    [data-testid="stTabs"] button[aria-selected="true"] {{ color: {colors['blue']}; }}
    [data-testid="stAlert"] {{ background: {colors['surface']};
        border: 1px solid {colors['grid']}; }}
    [data-testid="stAlert"] p {{ color: {colors['ink']}; }}
    .gm-kicker {{ color: {colors['blue']}; font-size: .78rem; font-weight: 700;
        letter-spacing: .12em; text-transform: uppercase; margin-bottom: .4rem; }}
    .gm-header {{ border-bottom: 1px solid {colors['grid']}; padding-bottom: 1rem;
        margin-bottom: 1.1rem; }}
    .gm-header h1 {{ margin: 0; padding: 0; color: {colors['ink']}; }}
    .gm-header p {{ color: {colors['muted']}; font-size: .95rem;
        margin: .5rem 0 0; line-height: 1.5; }}
    .gm-snapshot-title {{ font-size: 1.05rem; font-weight: 700;
        color: {colors['ink']}; margin-bottom: .7rem; }}
    .gm-meta {{ display: flex; flex-wrap: wrap; gap: 1.6rem;
        padding: .8rem 0; color: {colors['muted']}; font-size: .9rem; }}
    .gm-meta strong {{ color: {colors['ink']}; font-weight: 650; }}
    .gm-limits {{ padding-top: .75rem; border-top: 1px solid {colors['grid']};
        color: {colors['muted']}; font-size: .9rem; line-height: 1.5; }}
    .gm-limits strong {{ color: {colors['ink']}; }}
    .st-key-advice_snapshot {{ background: {colors['surface']};
        border: 1px solid {colors['grid']}; border-radius: 12px; padding: 20px; }}
    .gm-review {{ border-left: 3px solid {colors['blue']}; padding: .7rem 1rem;
        background: {colors['surface']}; color: {colors['ink']}; line-height: 1.55; }}
    @media (max-width: 640px) {{
        .stMainBlockContainer {{ padding: 1.2rem; }}
        .gm-header h1 {{ font-size: 1.75rem !important; }}
        .gm-meta {{ gap: .6rem; }}
    }}
    </style>""")


def render_figure(figure, colors):
    """Apply colors to this figure only, including heatmaps and shared axes."""
    figure.set_facecolor(colors["surface"])
    for text in figure.texts:
        text.set_color(colors["ink"])
    for ax in figure.axes:
        ax.set_facecolor(colors["surface"])
        ax.tick_params(colors=colors["muted"])
        for text in (ax.title, ax.xaxis.label, ax.yaxis.label):
            text.set_color(colors["ink"])
        for location in ("left", "right"):
            ax.set_title(ax.get_title(loc=location), loc=location, color=colors["ink"])
        for spine in ax.spines.values():
            spine.set_edgecolor(colors["grid"])
        for line in ax.get_xgridlines() + ax.get_ygridlines():
            line.set_color(colors["grid"])
        legend = ax.get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_color(colors["ink"])
    output = BytesIO()
    figure.savefig(output, format="png", bbox_inches="tight", dpi=200,
                   facecolor=colors["surface"])
    st.image(output.getvalue(), width="stretch")
    plt.close(figure)


def styled_table(frame, colors):
    return frame.style.set_properties(**{
        "background-color": colors["surface"], "color": colors["ink"]})


def evidence_chart(frame, x, y, colors, series=None, kind="line"):
    """Explicit Vega colors also follow the in-app light toggle."""
    chart = alt.Chart(frame).encode(x=x, y=y)
    if series:
        chart = chart.encode(color=alt.Color(series, scale=alt.Scale(
            range=[colors["blue"], colors["orange"], "#34b588", "#c084fc"])))
    chart = chart.mark_bar(color=colors["blue"]) if kind == "bar" else chart.mark_line(point=True)
    chart = chart.configure(background=colors["surface"]).configure_axis(
        labelColor=colors["muted"], titleColor=colors["ink"],
        gridColor=colors["grid"], domainColor=colors["grid"]).configure_legend(
        labelColor=colors["ink"], titleColor=colors["ink"]).configure_view(stroke=None)
    st.altair_chart(chart, width="stretch", theme=None)
