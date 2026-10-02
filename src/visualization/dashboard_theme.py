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
    return dict(background="#f7fafc", surface="#ffffff", ink="#172b40",
                muted="#526579", grid="#d5dfe9", blue="#2563eb",
                orange="#c95115", fault="#bd2444",
                healthy=["#334155", "#64748b", "#8995a5", "#475569"])


def apply_theme(colors, dark):
    """Theme visible controls and overlays without changing global server config."""
    scheme = "dark" if dark else "light"
    st.html(f"""<style>
    :root {{ color-scheme: {scheme}; }}
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
    [data-testid="stMetricValue"] {{ font-size: 1.8rem; }}
    [data-testid="stMetric"] {{ background: {colors['surface']};
        border: 1px solid {colors['grid']}; border-radius: 12px; padding: 14px; }}
    [data-testid="stExpander"] {{ background: {colors['surface']};
        border-color: {colors['grid']}; }}
    [data-testid="stExpander"] summary {{ background: {colors['surface']} !important; }}
    [data-baseweb="select"] div, [data-baseweb="popover"],
    [data-baseweb="menu"], [role="listbox"], [role="option"],
    [data-testid="stBaseButton-headerNoPadding"], [data-testid="stBaseButton-secondary"] {{
        background-color: {colors['surface']} !important; color: {colors['ink']} !important; }}
    [data-baseweb="select"] svg {{ fill: {colors['ink']}; }}
    [data-testid="stTabs"] button {{ color: {colors['muted']}; }}
    [data-testid="stTabs"] button[aria-selected="true"] {{ color: {colors['blue']}; }}
    [data-testid="stAlert"] {{ background: {colors['surface']};
        border: 1px solid {colors['grid']}; }}
    [data-testid="stAlert"] p {{ color: {colors['ink']}; }}
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
