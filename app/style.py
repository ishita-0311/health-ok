"""Shared visual theme for the Health-OK Streamlit app."""
import matplotlib.pyplot as plt
import streamlit as st

PLOT_BG = "#161C2C"
PLOT_FG = "#E7EBF3"
PLOT_ACCENT = "#2E6BE6"
PLOT_ACCENT2 = "#14B8A6"


def apply_mpl_theme():
    plt.rcParams.update({
        "figure.facecolor": PLOT_BG,
        "axes.facecolor": PLOT_BG,
        "savefig.facecolor": PLOT_BG,
        "axes.edgecolor": "#33405C",
        "axes.labelcolor": PLOT_FG,
        "text.color": PLOT_FG,
        "xtick.color": PLOT_FG,
        "ytick.color": PLOT_FG,
        "grid.color": "#2A3350",
        "axes.grid": True,
        "grid.alpha": 0.35,
        "font.family": "sans-serif",
    })

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ---------- Layout polish ---------- */
.block-container {
    padding-top: 2.2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] .stMarkdown p {
    font-weight: 500;
}

/* ---------- Headings ---------- */
h1 {
    font-weight: 800 !important;
    letter-spacing: -0.02em;
}
h2, h3 {
    font-weight: 700 !important;
    letter-spacing: -0.01em;
}

/* ---------- Hero banner ---------- */
.hok-hero {
    background: linear-gradient(120deg, #1C3FAA 0%, #2E6BE6 45%, #14B8A6 100%);
    border-radius: 20px;
    padding: 2.6rem 2.4rem;
    margin-bottom: 1.6rem;
    box-shadow: 0 20px 45px -20px rgba(46, 107, 230, 0.55);
}
.hok-hero h1 {
    color: #FFFFFF !important;
    font-size: 2.5rem !important;
    margin-bottom: 0.5rem !important;
}
.hok-hero p {
    color: rgba(255,255,255,0.92);
    font-size: 1.05rem;
    max-width: 760px;
    margin: 0;
}

/* ---------- Stat pills row ---------- */
.hok-stats { display: flex; gap: 0.9rem; margin: 1.4rem 0 0.4rem 0; flex-wrap: wrap; }
.hok-stat {
    background: rgba(255,255,255,0.14);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 14px;
    padding: 0.55rem 1.1rem;
    color: #fff;
    font-size: 0.9rem;
    font-weight: 600;
    backdrop-filter: blur(6px);
}

/* ---------- Disease / info cards ---------- */
.hok-card {
    border-radius: 18px;
    padding: 1.5rem;
    margin-bottom: 1.3rem;
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.08);
    transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
}
.hok-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 18px 34px -18px rgba(0,0,0,0.55);
    border-color: rgba(46, 107, 230, 0.45);
}
.hok-card-icon {
    border-radius: 14px;
    overflow: hidden;
    aspect-ratio: 4 / 3;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 0.9rem;
}
.hok-card-icon svg { width: 100%; height: 100%; display: block; }
.hok-card h3 {
    margin: 0 0 0.45rem 0 !important;
    font-size: 1.25rem !important;
}
.hok-card p.hok-desc {
    color: rgba(231,235,243,0.82);
    font-size: 0.92rem;
    line-height: 1.5;
    margin-bottom: 0;
}
.hok-nav-icon {
    width: 2.6rem;
    height: 2.6rem;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #2E6BE6, #14B8A6);
    color: #fff;
    font-weight: 800;
    font-size: 1.1rem;
    margin-bottom: 0.9rem;
}

/* ---------- Eyebrow label ---------- */
.hok-eyebrow {
    display: inline-block;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 0.72rem;
    font-weight: 700;
    color: rgba(255,255,255,0.75);
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.22);
    border-radius: 999px;
    padding: 0.25rem 0.75rem;
    margin-bottom: 0.9rem;
}

/* ---------- Badges / pills ---------- */
.hok-badge {
    display: inline-block;
    padding: 0.28rem 0.7rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.01em;
}
.hok-badge-low   { background: rgba(34,197,94,0.16);  color: #4ADE80; border: 1px solid rgba(34,197,94,0.35); }
.hok-badge-med   { background: rgba(245,158,11,0.16); color: #FBBF24; border: 1px solid rgba(245,158,11,0.35); }
.hok-badge-high  { background: rgba(239,68,68,0.16);  color: #F87171; border: 1px solid rgba(239,68,68,0.35); }

/* ---------- Result card (Prediction page) ---------- */
.hok-result {
    border-radius: 18px;
    padding: 1.8rem 2rem;
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.08);
    margin-top: 0.6rem;
}
.hok-result .hok-pct {
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 0.2rem 0 0.4rem 0;
}

/* ---------- Section divider spacing ---------- */
hr { margin: 1.6rem 0 !important; opacity: 0.15; }

/* ---------- Buttons ---------- */
.stButton>button {
    border-radius: 10px;
    font-weight: 600;
    padding: 0.55rem 1.4rem;
    transition: transform 0.15s ease;
}
.stButton>button:hover { transform: translateY(-1px); }

/* ---------- Alerts ---------- */
div[data-testid="stAlert"] {
    border-radius: 12px;
}

/* ---------- Expanders ---------- */
details[class*="st-"] summary, .streamlit-expanderHeader {
    border-radius: 10px !important;
}
</style>
"""


def inject():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, stats: list[str] | None = None, eyebrow: str | None = None):
    stats_html = ""
    if stats:
        pills = "".join(f'<div class="hok-stat">{s}</div>' for s in stats)
        stats_html = f'<div class="hok-stats">{pills}</div>'
    eyebrow_html = f'<div class="hok-eyebrow">{eyebrow}</div>' if eyebrow else ""
    st.markdown(
        f"""
        <div class="hok-hero">
            {eyebrow_html}
            <h1>{title}</h1>
            <p>{subtitle}</p>
            {stats_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_badge(risk_label: str) -> str:
    cls = {"Low Risk": "hok-badge-low", "Medium Risk": "hok-badge-med", "High Risk": "hok-badge-high"}[risk_label]
    return f'<span class="hok-badge {cls}">{risk_label}</span>'
