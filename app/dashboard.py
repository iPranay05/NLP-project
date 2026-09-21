"""
LeadLens — Interactive Streamlit Dashboard
Multilingual customer inquiry analysis for small businesses.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import json
import time
import textwrap
from datetime import datetime
from pathlib import Path

try:
    from fpdf import FPDF
    _PDF_AVAILABLE = True
except ImportError:
    _PDF_AVAILABLE = False

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from nlp.pipeline import analyze, analyze_batch
from config.settings import INTENTS, PRIORITY_CONFIG, LANGUAGES

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="LeadLens — NLP Lead Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── Root theme ── */
:root {
    --bg-primary:   #0D0F1A;
    --bg-card:      #13162A;
    --bg-card2:     #1A1D33;
    --accent-blue:  #6C63FF;
    --accent-teal:  #00D4AA;
    --accent-pink:  #FF6B9D;
    --accent-orange:#FF8C42;
    --text-primary: #F0F2FF;
    --text-muted:   #8892B0;
    --border:       rgba(108,99,255,0.20);
    --hot:          #FF4757;
    --warm:         #FFA502;
    --cold:         #5352ED;
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

.stApp { background: var(--bg-primary); }

/* ── Hide Streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #1a1d33 0%, #0d1b2a 50%, #0f0d1a 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(108,99,255,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -30%;
    left: 20%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(0,212,170,0.10) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6C63FF 0%, #00D4AA 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.1;
}
.hero-subtitle {
    color: var(--text-muted);
    font-size: 1.05rem;
    margin-top: 0.5rem;
    font-weight: 400;
}
.hero-badge {
    display: inline-block;
    background: rgba(108,99,255,0.15);
    border: 1px solid rgba(108,99,255,0.4);
    color: #9D97FF;
    padding: 0.25rem 0.75rem;
    border-radius: 50px;
    font-size: 0.78rem;
    font-weight: 500;
    margin-right: 0.5rem;
    margin-top: 1rem;
}

/* ── KPI metric cards ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease;
}
.kpi-card:hover { transform: translateY(-2px); }
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 16px 16px 0 0;
}
.kpi-card.blue::before  { background: linear-gradient(90deg, #6C63FF, #9D97FF); }
.kpi-card.teal::before  { background: linear-gradient(90deg, #00D4AA, #00FFD1); }
.kpi-card.hot::before   { background: linear-gradient(90deg, #FF4757, #FF6B9D); }
.kpi-card.warm::before  { background: linear-gradient(90deg, #FFA502, #FFD32A); }
.kpi-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.5rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1;
}
.kpi-label { color: var(--text-muted); font-size: 0.85rem; margin-top: 0.4rem; }
.kpi-icon  { font-size: 2rem; float: right; opacity: 0.6; }

/* ── Analyze input area ── */
.analyze-section {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 1.5rem;
}
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.3rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 1rem;
}

/* ── Result card ── */
.result-card {
    background: var(--bg-card2);
    border-radius: 16px;
    padding: 1.75rem;
    margin-top: 1.5rem;
    border: 1px solid var(--border);
    animation: fadeIn 0.4s ease;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Priority badges ── */
.badge {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 50px;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}
.badge-hot  { background: rgba(255,71,87,0.15);  border: 1px solid #FF4757; color: #FF4757; }
.badge-warm { background: rgba(255,165,2,0.15);  border: 1px solid #FFA502; color: #FFA502; }
.badge-cold { background: rgba(83,82,237,0.15);  border: 1px solid #5352ED; color: #9D9BFF; }
.badge-intent {
    background: rgba(108,99,255,0.12);
    border: 1px solid rgba(108,99,255,0.4);
    color: #9D97FF;
    padding: 0.25rem 0.75rem;
    border-radius: 50px;
    font-size: 0.82rem;
    font-weight: 500;
}
.badge-lang {
    background: rgba(0,212,170,0.12);
    border: 1px solid rgba(0,212,170,0.35);
    color: #00D4AA;
    padding: 0.25rem 0.75rem;
    border-radius: 50px;
    font-size: 0.82rem;
    font-weight: 500;
}

/* ── Entity pills ── */
.entity-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    margin-top: 0.75rem;
}
.entity-pill {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 8px;
    padding: 0.4rem 0.85rem;
    font-size: 0.83rem;
}
.entity-pill .entity-type {
    color: var(--text-muted);
    font-size: 0.72rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    display: block;
}
.entity-pill .entity-value { color: var(--text-primary); font-weight: 500; }

/* ── Score gauge container ── */
.gauge-container {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.25rem;
}

/* ── Lead table ── */
.lead-table-wrapper {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 1.5rem;
}

/* ── Action box ── */
.action-box {
    background: rgba(108,99,255,0.08);
    border: 1px solid rgba(108,99,255,0.25);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    color: #B8B4FF;
    font-size: 0.9rem;
    margin-top: 1rem;
    line-height: 1.6;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label { color: var(--text-muted); }

/* ── Score progress bar ── */
.score-bar-outer {
    background: rgba(255,255,255,0.06);
    border-radius: 50px;
    height: 8px;
    width: 100%;
    margin-top: 0.3rem;
}
.score-bar-inner {
    height: 8px;
    border-radius: 50px;
    transition: width 0.8s ease;
}

/* ── Divider ── */
.fancy-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 1.5rem 0;
}

/* ── Upload section ── */
.upload-section {
    border: 2px dashed rgba(108,99,255,0.3);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    background: rgba(108,99,255,0.04);
    transition: border-color 0.3s;
}
.upload-section:hover { border-color: rgba(108,99,255,0.6); }

/* Streamlit widget label overrides */
.stTextArea label { color: var(--text-muted) !important; font-size: 0.9rem !important; }
.stButton button {
    background: linear-gradient(135deg, #6C63FF 0%, #5A52E0 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.8rem !important;
    font-size: 0.95rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(108,99,255,0.3) !important;
}
.stButton button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(108,99,255,0.45) !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  DATA PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════════
LEADS_FILE = Path(__file__).parent.parent / "data" / "leads_store.json"


def load_leads() -> list:
    if LEADS_FILE.exists():
        try:
            with open(LEADS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_lead(result: dict):
    leads = load_leads()
    leads.insert(0, result)          # newest first
    with open(LEADS_FILE, "w", encoding="utf-8") as f:
        json.dump(leads, f, ensure_ascii=False, indent=2)


def generate_pdf(leads: list) -> bytes:
    """Generate a clean PDF report for the given list of leads."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── Title ──────────────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(108, 99, 255)
    pdf.cell(0, 12, "LeadLens - Lead Intelligence Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(136, 146, 176)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  Total leads: {len(leads)}", ln=True, align="C")
    pdf.ln(4)

    # ── Summary Stats ──────────────────────────────────────────────────────────
    hot   = sum(1 for l in leads if l.get("priority") == "Hot")
    warm  = sum(1 for l in leads if l.get("priority") == "Warm")
    cold  = sum(1 for l in leads if l.get("priority") == "Cold")
    avg_s = round(sum(l.get("score", 0) for l in leads) / len(leads), 1) if leads else 0

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(240, 242, 255)
    pdf.set_fill_color(13, 15, 26)
    pdf.cell(0, 8, "Summary", ln=True, fill=True)

    pdf.set_font("Helvetica", "", 10)
    for label, val, color in [
        ("Hot Leads",  hot,  (255, 71, 87)),
        ("Warm Leads", warm, (255, 165, 2)),
        ("Cold Leads", cold, (83, 82, 237)),
        ("Avg Score",  avg_s,(108, 99, 255)),
    ]:
        pdf.set_text_color(*color)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(30, 7, str(val), ln=False)
        pdf.set_text_color(136, 146, 176)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7, label, ln=True)
    pdf.ln(4)

    # ── Table Header ───────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(108, 99, 255)
    pdf.set_text_color(255, 255, 255)
    col_w = [14, 30, 68, 20, 20, 28]
    headers = ["Score", "Priority", "Inquiry", "Intent", "Language", "Action"]
    for h, w in zip(headers, col_w):
        pdf.cell(w, 7, h, border=1, fill=True, align="C")
    pdf.ln()

    # ── Table Rows ─────────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "", 7)
    priority_colors = {"Hot": (255, 71, 87), "Warm": (255, 165, 2), "Cold": (83, 82, 237)}
    intent_map = {"bulk_order": "Bulk Order", "price_query": "Price Query",
                  "complaint": "Complaint", "spam": "Spam"}

    for i, lead in enumerate(leads):
        fill = i % 2 == 0
        bg   = (245, 245, 255) if fill else (255, 255, 255)
        pdf.set_fill_color(*bg)

        priority = lead.get("priority", "Cold")
        score    = lead.get("score", 0)
        text     = lead.get("original_text", "")[:55] + ("…" if len(lead.get("original_text","")) > 55 else "")
        intent   = intent_map.get(lead.get("intent",""), lead.get("intent",""))
        lang     = lead.get("language","").upper()
        action   = lead.get("recommended_action","")[:30] + ("…" if len(lead.get("recommended_action","")) > 30 else "")

        p_color = priority_colors.get(priority, (83, 82, 237))

        pdf.set_text_color(*p_color)
        pdf.set_font("Helvetica", "B", 7)
        pdf.cell(col_w[0], 6, str(score), border=1, fill=True, align="C")
        pdf.cell(col_w[1], 6, priority, border=1, fill=True, align="C")

        pdf.set_text_color(30, 30, 30)
        pdf.set_font("Helvetica", "", 7)
        pdf.cell(col_w[2], 6, text, border=1, fill=True)
        pdf.cell(col_w[3], 6, intent, border=1, fill=True, align="C")
        pdf.cell(col_w[4], 6, lang, border=1, fill=True, align="C")
        pdf.cell(col_w[5], 6, action, border=1, fill=True)
        pdf.ln()

    # ── Footer ─────────────────────────────────────────────────────────────────
    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(136, 146, 176)
    pdf.cell(0, 6, "Generated by LeadLens — Multilingual NLP Lead Intelligence System", align="C")

    return bytes(pdf.output())



def leads_to_df(leads: list) -> pd.DataFrame:
    """Flatten nested result dicts to a display-ready DataFrame."""
    rows = []
    for r in leads:
        ent = r.get("entities", {})
        rows.append({
            "ID":        r.get("id", ""),
            "Timestamp": r.get("timestamp", ""),
            "Inquiry":   r.get("original_text", "")[:80] + ("…" if len(r.get("original_text","")) > 80 else ""),
            "Language":  LANGUAGES.get(r.get("language","unknown"), {}).get("name","?"),
            "Intent":    INTENTS.get(r.get("intent",""), {}).get("label", r.get("intent","")),
            "Conf %":    f"{r.get('intent_confidence',0)*100:.0f}%",
            "Product":   ent.get("product") or "—",
            "Quantity":  ent.get("quantity",{}).get("raw","—") if ent.get("quantity") else "—",
            "Budget":    ent.get("budget",{}).get("display","—") if ent.get("budget") else "—",
            "Location":  ent.get("location") or "—",
            "Score":     r.get("score", 0),
            "Priority":  r.get("priority", "—"),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "leads" not in st.session_state:
    st.session_state.leads = load_leads()


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem 0 1.5rem 0;">
        <div style="font-family:'Space Grotesk',sans-serif;font-size:1.5rem;font-weight:700;
                    background:linear-gradient(135deg,#6C63FF,#00D4AA);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    background-clip:text;">
            🔍 LeadLens
        </div>
        <div style="color:#8892B0;font-size:0.78rem;margin-top:0.2rem;">
            NLP Lead Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.selectbox(
        "Navigate",
        ["🏠 Dashboard", "🔬 Analyze Inquiry", "📋 Lead Pipeline", "📊 Analytics", "📤 Bulk Upload"],
        key="nav_page",
    )

    st.markdown("---")
    st.markdown("<div style='color:#8892B0;font-size:0.8rem;font-weight:600;letter-spacing:0.05em;'>QUICK DEMO</div>", unsafe_allow_html=True)

    demo_samples = {
        "🇬🇧 Bulk Order (EN)": "I need 500 kg of cotton fabric wholesale, best price for Mumbai delivery",
        "🇮🇳 Price Query (HI)": "हमें हर महीने 2 टन चावल चाहिए, दिल्ली डिलीवरी, बजट ₹80000",
        "🟠 Bulk Order (MR)": "आम्हाला दर महिन्याला 500 किलो साखर लागते मुंबईसाठी",
        "🔀 Complaint (HG)": "Yaar delivery bahut slow hai 2 hafte ho gaye maal nahi aaya",
        "🚫 Spam": "Click here WIN iPhone Limited time offer FREE gift!",
        "🔀 Price Query (HG)": "Bhai denim fabric ka rate kya hai per meter, budget 5k hai",
    }

    def load_demo(txt):
        st.session_state["demo_text"] = txt
        st.session_state["nav_page"] = "🔬 Analyze Inquiry"

    for label, txt in demo_samples.items():
        st.button(label, key=f"demo_{label}", on_click=load_demo, args=(txt,))

    st.markdown("---")
    st.markdown(
        "<div style='color:#8892B0;font-size:0.75rem;text-align:center;margin-top:1rem;'>"
        "Supports EN · HI · MR · Hinglish<br>"
        "v1.0 · Built with ❤️ in Pune"
        "</div>",
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER WIDGETS
# ══════════════════════════════════════════════════════════════════════════════
def priority_badge(priority: str) -> str:
    cls = {"Hot": "badge-hot", "Warm": "badge-warm", "Cold": "badge-cold"}.get(priority, "badge-cold")
    cfg = PRIORITY_CONFIG.get(priority, {})
    return f'<span class="badge {cls}">{cfg.get("emoji","●")} {priority}</span>'


def intent_badge(intent: str) -> str:
    cfg = INTENTS.get(intent, {})
    return f'<span class="badge-intent">{cfg.get("emoji","●")} {cfg.get("label", intent)}</span>'


def lang_badge(lang: str) -> str:
    cfg = LANGUAGES.get(lang, {})
    return f'<span class="badge-lang">{cfg.get("emoji","🌐")} {cfg.get("name", lang)}</span>'


def score_color(score: int) -> str:
    if score >= 70: return "#FF4757"
    if score >= 40: return "#FFA502"
    return "#5352ED"


def draw_gauge(score: int) -> go.Figure:
    color = score_color(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        number={"font": {"color": color, "size": 48, "family": "Space Grotesk"}, "suffix": ""},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "#8892B0",
                "tickfont": {"color": "#8892B0", "size": 10},
            },
            "bar":            {"color": color, "thickness": 0.25},
            "bgcolor":        "rgba(0,0,0,0)",
            "borderwidth":    0,
            "steps": [
                {"range": [0, 40],  "color": "rgba(83,82,237,0.12)"},
                {"range": [40, 70], "color": "rgba(255,165,2,0.12)"},
                {"range": [70, 100],"color": "rgba(255,71,87,0.12)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.8,
                "value": score,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
        margin=dict(t=20, b=10, l=20, r=20),
        height=200,
        font={"color": "#F0F2FF"},
    )
    return fig


def render_result_card(r: dict):
    """Render the full analysis result card."""
    ent   = r.get("entities", {})
    score = r.get("score", 0)
    bd    = r.get("score_breakdown", {})

    # ── Header row ────────────────────────────────────────────────────────────
    col_l, col_r = st.columns([3, 1])
    with col_l:
        st.markdown(
            f"{priority_badge(r['priority'])} &nbsp; "
            f"{intent_badge(r['intent'])} &nbsp; "
            f"{lang_badge(r['language'])}",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='color:#8892B0;font-size:0.82rem;margin-top:0.5rem;'>"
            f"Intent confidence: <b style='color:#F0F2FF'>{r['intent_confidence']*100:.0f}%</b> &nbsp;|&nbsp; "
            f"Lead ID: <code style='background:rgba(255,255,255,0.05);padding:0.1rem 0.4rem;border-radius:4px;color:#9D97FF'>{r['id']}</code>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col_r:
        st.plotly_chart(draw_gauge(score), use_container_width=True)
        # Priority label under gauge
        cfg = PRIORITY_CONFIG.get(r["priority"], {})
        st.markdown(
            f"<div style='text-align:center;margin-top:-1rem;font-size:0.78rem;color:{cfg.get('color','#888')};font-weight:600;'>"
            f"{cfg.get('sla','')}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

    # ── Score breakdown bars ──────────────────────────────────────────────────
    st.markdown("<div style='color:#8892B0;font-size:0.8rem;font-weight:600;letter-spacing:0.05em;margin-bottom:0.5rem;'>SCORE BREAKDOWN</div>", unsafe_allow_html=True)
    breakdown_cols = st.columns(4)
    breakdown_items = [
        ("Intent",      bd.get("intent_score",0),   40, "#6C63FF"),
        ("Entities",    bd.get("entity_score",0),    30, "#00D4AA"),
        ("Language",    bd.get("language_score",0),  15, "#FF6B9D"),
        ("Specificity", bd.get("specificity",0),     15, "#FF8C42"),
    ]
    for col, (label, val, max_val, color) in zip(breakdown_cols, breakdown_items):
        with col:
            pct = (val / max_val * 100) if max_val else 0
            st.markdown(
                f"<div style='font-size:0.78rem;color:#8892B0;'>{label}</div>"
                f"<div style='font-size:1.3rem;font-weight:700;color:{color};font-family:Space Grotesk;'>{val:.0f}<span style='font-size:0.7rem;color:#8892B0;'>/{max_val}</span></div>"
                f"<div class='score-bar-outer'><div class='score-bar-inner' style='width:{pct:.0f}%;background:{color};'></div></div>",
                unsafe_allow_html=True
            )

    st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

    # ── Entities ──────────────────────────────────────────────────────────────
    st.markdown("<div style='color:#8892B0;font-size:0.8rem;font-weight:600;letter-spacing:0.05em;margin-bottom:0.5rem;'>EXTRACTED ENTITIES</div>", unsafe_allow_html=True)
    entity_pills = ""
    entity_map = {
        "product":  ("📦 Product",  ent.get("product")),
        "quantity": ("⚖️ Quantity",  ent.get("quantity",{}).get("raw") if ent.get("quantity") else None),
        "budget":   ("💰 Budget",   ent.get("budget",{}).get("display") if ent.get("budget") else None),
        "location": ("📍 Location",  ent.get("location")),
    }
    any_entity = False
    for key, (type_label, value) in entity_map.items():
        if value:
            any_entity = True
            entity_pills += (
                f"<div class='entity-pill'>"
                f"<span class='entity-type'>{type_label}</span>"
                f"<span class='entity-value'>{value}</span>"
                f"</div>"
            )
    if not any_entity:
        entity_pills = "<div style='color:#8892B0;font-size:0.85rem;'>No specific entities extracted</div>"

    st.markdown(f"<div class='entity-grid'>{entity_pills}</div>", unsafe_allow_html=True)

    st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

    # ── Recommended Action ────────────────────────────────────────────────────
    st.markdown("<div style='color:#8892B0;font-size:0.8rem;font-weight:600;letter-spacing:0.05em;margin-bottom:0.4rem;'>RECOMMENDED ACTION</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='action-box'>🎯 {r['recommended_action']}</div>", unsafe_allow_html=True)

    # ── Original text ─────────────────────────────────────────────────────────
    with st.expander("📄 Original Inquiry Text"):
        st.markdown(
            f"<div style='background:rgba(255,255,255,0.03);border-radius:8px;padding:1rem;"
            f"color:#B0B8D1;font-size:0.9rem;line-height:1.7;border-left:3px solid #6C63FF;'>"
            f"{r['original_text']}</div>",
            unsafe_allow_html=True,
        )

    # ── All intent scores ─────────────────────────────────────────────────────
    with st.expander("📊 All Intent Probabilities"):
        scores = r.get("intent_scores", {})
        for ikey, iconf in sorted(scores.items(), key=lambda x: -x[1]):
            cfg = INTENTS.get(ikey, {})
            bar_pct = int(iconf * 100)
            st.markdown(
                f"<div style='margin-bottom:0.5rem;'>"
                f"<div style='display:flex;justify-content:space-between;font-size:0.83rem;'>"
                f"<span>{cfg.get('emoji','●')} {cfg.get('label',ikey)}</span>"
                f"<span style='color:#8892B0;'>{iconf*100:.1f}%</span></div>"
                f"<div class='score-bar-outer'><div class='score-bar-inner' "
                f"style='width:{bar_pct}%;background:{cfg.get('color','#6C63FF')};'></div></div>"
                f"</div>",
                unsafe_allow_html=True
            )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def page_dashboard():
    # Hero
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🔍 LeadLens</div>
        <div class="hero-subtitle">
            Multilingual NLP Lead Intelligence · English · Hindi · Marathi · Hinglish
        </div>
        <div style="margin-top:1rem;">
            <span class="hero-badge">🧠 NLP-Powered</span>
            <span class="hero-badge">🌐 4 Languages</span>
            <span class="hero-badge">⚡ Real-time</span>
            <span class="hero-badge">📊 Smart Scoring</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    leads = st.session_state.leads

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    total      = len(leads)
    hot        = sum(1 for l in leads if l.get("priority") == "Hot")
    warm       = sum(1 for l in leads if l.get("priority") == "Warm")
    avg_score  = round(sum(l.get("score",0) for l in leads) / max(1, total), 1)

    c1, c2, c3, c4 = st.columns(4)
    kpi_data = [
        (c1, "Total Leads", total,     "📬", "blue"),
        (c2, "Avg Score",   avg_score, "📈", "teal"),
        (c3, "Hot Leads 🔴", hot,      "🔥", "hot"),
        (c4, "Warm Leads 🟡",warm,     "☀️", "warm"),
    ]
    for col, label, val, icon, cls in kpi_data:
        with col:
            st.markdown(
                f"<div class='kpi-card {cls}'>"
                f"<div class='kpi-icon'>{icon}</div>"
                f"<div class='kpi-value'>{val}</div>"
                f"<div class='kpi-label'>{label}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

    if not leads:
        st.markdown("""
        <div style="text-align:center;padding:4rem;color:#8892B0;">
            <div style="font-size:3rem;margin-bottom:1rem;">📭</div>
            <div style="font-size:1.1rem;font-weight:500;">No leads analyzed yet</div>
            <div style="font-size:0.85rem;margin-top:0.5rem;">Use <b>Analyze Inquiry</b> or <b>Bulk Upload</b> to get started</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Charts row ────────────────────────────────────────────────────────────
    col_a, col_b, col_c = st.columns([1.2, 1.2, 1])

    # Intent Distribution Pie
    with col_a:
        st.markdown("<div class='gauge-container'>", unsafe_allow_html=True)
        intent_counts = {}
        for l in leads:
            ikey = l.get("intent","unknown")
            intent_counts[ikey] = intent_counts.get(ikey, 0) + 1
        labels = [INTENTS.get(k,{}).get("label", k) for k in intent_counts]
        colors = [INTENTS.get(k,{}).get("color","#888") for k in intent_counts]
        fig_pie = go.Figure(go.Pie(
            labels=labels, values=list(intent_counts.values()),
            hole=0.55,
            marker=dict(colors=colors, line=dict(color="#0D0F1A", width=2)),
            textfont=dict(color="#F0F2FF", size=12),
            hovertemplate="<b>%{label}</b><br>%{value} leads (%{percent})<extra></extra>",
        ))
        fig_pie.update_layout(
            title=dict(text="Intent Distribution", font=dict(color="#F0F2FF", size=14), x=0.05),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(font=dict(color="#8892B0", size=11), bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=40, b=10, l=10, r=10), height=260,
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Priority Distribution Bar
    with col_b:
        st.markdown("<div class='gauge-container'>", unsafe_allow_html=True)
        priority_counts = {"Hot": 0, "Warm": 0, "Cold": 0}
        for l in leads:
            p = l.get("priority", "Cold")
            if p in priority_counts:
                priority_counts[p] += 1
        fig_bar = go.Figure(go.Bar(
            x=list(priority_counts.keys()),
            y=list(priority_counts.values()),
            marker_color=["#FF4757", "#FFA502", "#5352ED"],
            text=list(priority_counts.values()),
            textposition="outside",
            textfont=dict(color="#F0F2FF"),
            hovertemplate="<b>%{x}</b><br>%{y} leads<extra></extra>",
        ))
        fig_bar.update_layout(
            title=dict(text="Lead Priority", font=dict(color="#F0F2FF", size=14), x=0.05),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(tickfont=dict(color="#8892B0"), gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(tickfont=dict(color="#8892B0"), gridcolor="rgba(255,255,255,0.04)", showgrid=True),
            margin=dict(t=40, b=20, l=20, r=20), height=260,
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Language Breakdown
    with col_c:
        st.markdown("<div class='gauge-container'>", unsafe_allow_html=True)
        lang_counts = {}
        for l in leads:
            lang = LANGUAGES.get(l.get("language","unknown"), {}).get("name","Unknown")
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        lang_colors = ["#6C63FF", "#00D4AA", "#FF6B9D", "#FF8C42", "#888"]
        fig_lang = go.Figure(go.Pie(
            labels=list(lang_counts.keys()), values=list(lang_counts.values()),
            hole=0.55,
            marker=dict(colors=lang_colors[:len(lang_counts)], line=dict(color="#0D0F1A", width=2)),
            textfont=dict(color="#F0F2FF", size=11),
        ))
        fig_lang.update_layout(
            title=dict(text="Language Split", font=dict(color="#F0F2FF", size=14), x=0.05),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(font=dict(color="#8892B0", size=10), bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=40, b=10, l=10, r=10), height=260,
        )
        st.plotly_chart(fig_lang, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Recent Hot Leads ──────────────────────────────────────────────────────
    hot_leads = [l for l in leads if l.get("priority") == "Hot"][:5]
    if hot_leads:
        st.markdown("---")
        st.markdown("<div class='section-title'>🔴 Hot Leads — Respond Now</div>", unsafe_allow_html=True)
        for lead in hot_leads:
            ent = lead.get("entities", {})
            cfg = INTENTS.get(lead.get("intent",""), {})
            st.markdown(
                f"<div style='background:rgba(255,71,87,0.05);border:1px solid rgba(255,71,87,0.2);"
                f"border-radius:12px;padding:1rem 1.25rem;margin-bottom:0.75rem;'>"
                f"<div style='display:flex;justify-content:space-between;align-items:flex-start;'>"
                f"<div>"
                f"<div style='font-size:0.9rem;color:#F0F2FF;margin-bottom:0.3rem;'>{lead.get('original_text','')[:100]}...</div>"
                f"<div style='font-size:0.78rem;color:#8892B0;'>"
                f"{cfg.get('emoji','●')} {cfg.get('label','')} &nbsp;·&nbsp; "
                f"📦 {ent.get('product','—')} &nbsp;·&nbsp; "
                f"⚖️ {ent.get('quantity',{}).get('raw','—') if ent.get('quantity') else '—'} &nbsp;·&nbsp; "
                f"💰 {ent.get('budget',{}).get('display','—') if ent.get('budget') else '—'}"
                f"</div></div>"
                f"<div style='text-align:right;min-width:80px;'>"
                f"<div style='font-family:Space Grotesk;font-size:2rem;font-weight:700;color:#FF4757;'>{lead.get('score',0)}</div>"
                f"<div style='font-size:0.72rem;color:#8892B0;'>{lead.get('timestamp','')[:10]}</div>"
                f"</div></div></div>",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: ANALYZE INQUIRY
# ══════════════════════════════════════════════════════════════════════════════
def page_analyze():
    st.markdown("<div class='section-title'>🔬 Analyze Customer Inquiry</div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='color:#8892B0;font-size:0.9rem;margin-bottom:1.5rem;'>"
        "Paste any customer message in English, Hindi, Marathi, or Hinglish. "
        "LeadLens will classify intent, extract entities, and score the lead instantly."
        "</div>",
        unsafe_allow_html=True,
    )

    # Pre-fill from demo sidebar
    prefill = st.session_state.pop("demo_text", "")

    text_input = st.text_area(
        "Customer Inquiry",
        value=prefill,
        height=140,
        placeholder="Type or paste a customer message here…\nExample: Mujhe 500 kg cotton chahiye, rate batao, Mumbai delivery",
        key="inquiry_text",
    )

    col_btn, col_clear, _ = st.columns([1, 1, 5])
    with col_btn:
        analyze_clicked = st.button("⚡ Analyze", use_container_width=True)
    with col_clear:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.last_result = None
            st.rerun()

    if analyze_clicked and text_input.strip():
        with st.spinner("Analyzing inquiry…"):
            result = analyze(text_input)
            st.session_state.last_result = result
            save_lead(result)
            st.session_state.leads = load_leads()

    # ── Show result ───────────────────────────────────────────────────────────
    if st.session_state.last_result:
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        render_result_card(st.session_state.last_result)
        st.markdown("</div>", unsafe_allow_html=True)
    elif not analyze_clicked:
        # Placeholder hints
        st.markdown("""
        <div style="margin-top:2rem;padding:2rem;border:1px dashed rgba(108,99,255,0.25);
                    border-radius:16px;text-align:center;color:#8892B0;">
            <div style="font-size:2rem;margin-bottom:0.75rem;">💬</div>
            <div style="font-size:0.95rem;">Enter a customer message above and click <b style="color:#9D97FF;">⚡ Analyze</b></div>
            <div style="font-size:0.82rem;margin-top:0.4rem;">Or try a quick demo from the sidebar →</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: LEAD PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def page_pipeline():
    st.markdown("<div class='section-title'>📋 Lead Pipeline</div>", unsafe_allow_html=True)
    leads = st.session_state.leads

    if not leads:
        st.info("No leads yet. Analyze some inquiries first.")
        return

    # Filters
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        priority_filter = st.selectbox("Priority", ["All", "Hot", "Warm", "Cold"])
    with fc2:
        intent_opts = ["All"] + [INTENTS[k]["label"] for k in INTENTS]
        intent_filter = st.selectbox("Intent", intent_opts)
    with fc3:
        lang_opts = ["All"] + [v["name"] for v in LANGUAGES.values() if v["name"] != "Unknown"]
        lang_filter = st.selectbox("Language", lang_opts)
    with fc4:
        min_score = st.slider("Min Score", 0, 100, 0)

    # Filter leads
    filtered = leads
    if priority_filter != "All":
        filtered = [l for l in filtered if l.get("priority") == priority_filter]
    if intent_filter != "All":
        rev_map = {v["label"]: k for k, v in INTENTS.items()}
        filtered = [l for l in filtered if l.get("intent") == rev_map.get(intent_filter)]
    if lang_filter != "All":
        rev_lang = {v["name"]: k for k, v in LANGUAGES.items()}
        filtered = [l for l in filtered if l.get("language") == rev_lang.get(lang_filter)]
    filtered = [l for l in filtered if l.get("score", 0) >= min_score]

    st.markdown(f"<div style='color:#8892B0;font-size:0.85rem;margin-bottom:1rem;'>Showing <b style='color:#F0F2FF;'>{len(filtered)}</b> of {len(leads)} leads</div>", unsafe_allow_html=True)

    if not filtered:
        st.warning("No leads match the selected filters.")
        return

    df = leads_to_df(filtered)

    # Color-code priority column
    def color_priority(val):
        colors = {"Hot": "#FF4757", "Warm": "#FFA502", "Cold": "#5352ED"}
        c = colors.get(val, "#888")
        return f"color: {c}; font-weight: 600;"

    styled = df.style.applymap(color_priority, subset=["Priority"])
    st.dataframe(
        styled,
        use_container_width=True,
        height=420,
        column_config={
            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d"),
            "Inquiry": st.column_config.TextColumn("Inquiry", width="large"),
        }
    )

    # ── Export Buttons ────────────────────────────────────────────────────────
    ex_col1, ex_col2 = st.columns([1, 1])
    with ex_col1:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Export as CSV",
            data=csv,
            file_name=f"leadlens_export_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with ex_col2:
        if _PDF_AVAILABLE:
            pdf_bytes = generate_pdf(filtered)
            st.download_button(
                "📄 Export as PDF Report",
                data=pdf_bytes,
                file_name=f"leadlens_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.warning("Install fpdf2 for PDF export: `pip install fpdf2`")

    # Clear leads
    if st.button("🗑️ Clear All Leads", type="secondary"):
        with open(LEADS_FILE, "w") as f:
            json.dump([], f)
        st.session_state.leads = []
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
def page_analytics():
    st.markdown("<div class='section-title'>📊 Analytics & Insights</div>", unsafe_allow_html=True)
    leads = st.session_state.leads

    if len(leads) < 2:
        st.info("Analyze at least 2 leads to see analytics.")
        return

    scores = [l.get("score", 0) for l in leads]
    c1, c2 = st.columns(2)

    # Score Distribution Histogram
    with c1:
        fig_hist = px.histogram(
            x=scores, nbins=15,
            labels={"x": "Lead Score", "count": "# Leads"},
            color_discrete_sequence=["#6C63FF"],
            title="Score Distribution",
        )
        fig_hist.update_traces(marker_line_color="#0D0F1A", marker_line_width=1)
        fig_hist.add_vline(x=70, line_dash="dash", line_color="#FF4757", annotation_text="Hot",
                           annotation_font_color="#FF4757")
        fig_hist.add_vline(x=40, line_dash="dash", line_color="#FFA502", annotation_text="Warm",
                           annotation_font_color="#FFA502")
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8892B0"), height=280,
            title_font=dict(color="#F0F2FF"), margin=dict(t=40, b=20, l=20, r=20),
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # Entity Completeness Radar
    with c2:
        entity_counts = {"Product": 0, "Quantity": 0, "Budget": 0, "Location": 0}
        for l in leads:
            ent = l.get("entities", {})
            if ent.get("product"):  entity_counts["Product"]  += 1
            if ent.get("quantity"): entity_counts["Quantity"] += 1
            if ent.get("budget"):   entity_counts["Budget"]   += 1
            if ent.get("location"): entity_counts["Location"] += 1

        fig_radar = go.Figure(go.Scatterpolar(
            r=list(entity_counts.values()),
            theta=list(entity_counts.keys()),
            fill="toself",
            fillcolor="rgba(108,99,255,0.15)",
            line=dict(color="#6C63FF", width=2),
            marker=dict(size=6, color="#6C63FF"),
        ))
        fig_radar.update_layout(
            title=dict(text="Entity Coverage", font=dict(color="#F0F2FF", size=14)),
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, gridcolor="rgba(255,255,255,0.08)", tickfont=dict(color="#8892B0")),
                angularaxis=dict(tickfont=dict(color="#8892B0"), gridcolor="rgba(255,255,255,0.08)"),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8892B0"), height=280,
            margin=dict(t=40, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Intent × Priority heatmap
    st.markdown("---")
    heatmap_data = {}
    for l in leads:
        row = INTENTS.get(l.get("intent",""), {}).get("label", "Unknown")
        col = l.get("priority", "Cold")
        heatmap_data.setdefault(row, {"Hot": 0, "Warm": 0, "Cold": 0})
        if col in heatmap_data[row]:
            heatmap_data[row][col] += 1

    if heatmap_data:
        hm_df = pd.DataFrame(heatmap_data).T.fillna(0)
        for c in ["Hot", "Warm", "Cold"]:
            if c not in hm_df.columns:
                hm_df[c] = 0
        hm_df = hm_df[["Hot", "Warm", "Cold"]]

        fig_hm = px.imshow(
            hm_df, text_auto=True, aspect="auto",
            color_continuous_scale=[[0,"#13162A"],[0.5,"#6C63FF"],[1,"#FF4757"]],
            title="Intent × Priority Heatmap",
            labels=dict(x="Priority", y="Intent", color="Count"),
        )
        fig_hm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8892B0"), height=300,
            title_font=dict(color="#F0F2FF"),
            margin=dict(t=40, b=20, l=20, r=20),
            coloraxis_colorbar=dict(tickfont=dict(color="#8892B0")),
        )
        st.plotly_chart(fig_hm, use_container_width=True)

    # Score trend (time series)
    if len(leads) >= 3:
        trend_df = pd.DataFrame([
            {"time": l.get("timestamp",""), "score": l.get("score",0), "priority": l.get("priority","Cold")}
            for l in reversed(leads)
        ])
        trend_df["time"] = pd.to_datetime(trend_df["time"], errors="coerce")
        trend_df = trend_df.dropna(subset=["time"])

        if not trend_df.empty:
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=trend_df["time"], y=trend_df["score"],
                mode="lines+markers",
                line=dict(color="#6C63FF", width=2),
                marker=dict(
                    size=8,
                    color=[{"Hot": "#FF4757", "Warm": "#FFA502", "Cold": "#5352ED"}.get(p,"#888") for p in trend_df["priority"]],
                    line=dict(color="#0D0F1A", width=1),
                ),
                name="Lead Score",
                hovertemplate="<b>%{x}</b><br>Score: %{y}<extra></extra>",
            ))
            fig_trend.add_hrect(y0=70, y1=100, fillcolor="#FF4757", opacity=0.05, line_width=0, annotation_text="Hot Zone")
            fig_trend.add_hrect(y0=40, y1=70, fillcolor="#FFA502", opacity=0.05, line_width=0, annotation_text="Warm Zone")
            fig_trend.update_layout(
                title=dict(text="Lead Score Over Time", font=dict(color="#F0F2FF", size=14), x=0.02),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#8892B0"), height=300,
                xaxis=dict(gridcolor="rgba(255,255,255,0.04)", tickfont=dict(color="#8892B0")),
                yaxis=dict(gridcolor="rgba(255,255,255,0.04)", tickfont=dict(color="#8892B0"), range=[0,105]),
                margin=dict(t=40, b=20, l=20, r=20),
                showlegend=False,
            )
            st.plotly_chart(fig_trend, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: BULK UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
def page_bulk():
    st.markdown("<div class='section-title'>📤 Bulk Inquiry Upload</div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='color:#8892B0;font-size:0.9rem;margin-bottom:1.5rem;'>"
        "Upload a CSV file with a <b style='color:#9D97FF;'>text</b> column containing customer inquiries. "
        "LeadLens will analyze all messages and add them to your pipeline."
        "</div>",
        unsafe_allow_html=True,
    )

    # Sample CSV option
    c1, c2 = st.columns([2, 1])
    with c2:
        sample_path = Path(__file__).parent.parent / "data" / "sample_inquiries.csv"
        if sample_path.exists():
            with open(sample_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Sample CSV",
                    data=f.read(),
                    file_name="sample_inquiries.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    with c1:
        uploaded_file = st.file_uploader(
            "Upload CSV",
            type=["csv"],
            help="CSV must contain a 'text' column with one inquiry per row",
        )

    if uploaded_file:
        try:
            df_upload = pd.read_csv(uploaded_file)
            if "text" not in df_upload.columns:
                st.error("❌ CSV must have a column named **text**.")
                return

            texts = df_upload["text"].dropna().tolist()
            st.markdown(
                f"<div style='color:#00D4AA;margin:1rem 0;font-size:0.9rem;'>"
                f"✅ Loaded <b>{len(texts)}</b> inquiries. Click Analyze to process.</div>",
                unsafe_allow_html=True,
            )

            if st.button(f"⚡ Analyze All {len(texts)} Inquiries", use_container_width=False):
                progress = st.progress(0, text="Analyzing…")
                results = []
                for i, text in enumerate(texts):
                    r = analyze(text)
                    results.append(r)
                    progress.progress((i + 1) / len(texts), text=f"Processing {i+1}/{len(texts)}…")
                    time.sleep(0.001)

                # Save all at once to prevent file lock/Errno 22 issues
                existing_leads = load_leads()
                # Insert newest first
                all_leads = results[::-1] + existing_leads
                with open(LEADS_FILE, "w", encoding="utf-8") as f:
                    json.dump(all_leads, f, indent=2, ensure_ascii=False)

                st.session_state.leads = load_leads()
                progress.empty()
                st.success(f"✅ Analyzed {len(results)} inquiries successfully!")

                # Quick summary
                hot_n  = sum(1 for r in results if r["priority"] == "Hot")
                warm_n = sum(1 for r in results if r["priority"] == "Warm")
                cold_n = sum(1 for r in results if r["priority"] == "Cold")
                avg_s  = round(sum(r["score"] for r in results) / len(results), 1)

                mc1, mc2, mc3, mc4 = st.columns(4)
                for c, label, val, color in [
                    (mc1, "Hot Leads",   hot_n,  "#FF4757"),
                    (mc2, "Warm Leads",  warm_n, "#FFA502"),
                    (mc3, "Cold Leads",  cold_n, "#5352ED"),
                    (mc4, "Avg Score",   avg_s,  "#6C63FF"),
                ]:
                    with c:
                        st.markdown(
                            f"<div style='background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);"
                            f"border-radius:12px;padding:1rem;text-align:center;'>"
                            f"<div style='font-family:Space Grotesk;font-size:2rem;font-weight:700;color:{color};'>{val}</div>"
                            f"<div style='color:#8892B0;font-size:0.8rem;'>{label}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                st.markdown("→ Go to **Lead Pipeline** to view all results.", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error processing file: {e}")

    # Manual multi-line input
    st.markdown("---")
    st.markdown("<div class='section-title' style='font-size:1rem;'>Or paste multiple inquiries</div>", unsafe_allow_html=True)
    multi_text = st.text_area(
        "One inquiry per line",
        height=180,
        placeholder="Inquiry 1\nInquiry 2\nInquiry 3…",
        key="multi_inquiry",
    )
    if st.button("⚡ Analyze Pasted Inquiries") and multi_text.strip():
        lines = [l.strip() for l in multi_text.strip().splitlines() if l.strip()]
        results = []
        for text in lines:
            r = analyze(text)
            results.append(r)
            
        existing_leads = load_leads()
        all_leads = results[::-1] + existing_leads
        with open(LEADS_FILE, "w", encoding="utf-8") as f:
            json.dump(all_leads, f, indent=2, ensure_ascii=False)
        st.session_state.leads = load_leads()
        st.success(f"✅ Analyzed {len(results)} inquiries!")
        df_results = leads_to_df(results)
        st.dataframe(df_results, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTER
# ══════════════════════════════════════════════════════════════════════════════
current_page = st.session_state.get("nav_page", "🏠 Dashboard")

if current_page == "🏠 Dashboard":
    page_dashboard()
elif current_page == "🔬 Analyze Inquiry":
    page_analyze()
elif current_page == "📋 Lead Pipeline":
    page_pipeline()
elif current_page == "📊 Analytics":
    page_analytics()
elif current_page == "📤 Bulk Upload":
    page_bulk()
