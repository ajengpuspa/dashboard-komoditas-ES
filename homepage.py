import streamlit as st

from utils import (
    init_session_state,
    inject_custom_css,
    render_sidebar,
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="KomoditasAI Dashboard",
    page_icon="💰",
    layout="wide",
)

# ============================================================
# INITIALIZATION
# ============================================================

init_session_state()
inject_custom_css()
render_sidebar()

# ============================================================
# BREADCRUMB
# ============================================================

st.markdown(
    '<div style="font-size:12px;color:#8A8D98;margin-bottom:12px;">app / homepage</div>',
    unsafe_allow_html=True,
)

# ============================================================
# HERO
# ============================================================

st.html("""
<div class="homepage-hero">

    <div class="homepage-dots"></div>
    <div class="homepage-circle"></div>

    <div class="hero-content">

        <div class="hero-logo">
            Rp
        </div>

        <div class="hero-kicker">
            KomoditasAI • Stacking Ensemble Dashboard
        </div>

        <div class="hero-title">
            Integrasi Model Kecerdasan Buatan dan Stokastik
            Melalui Stacking Ensemble Learning untuk
            Prediksi Harga dan Risiko Komoditas Pangan
            di Indonesia
        </div>

        <div class="hero-accent"></div>

        <div class="hero-authors">

            <div class="hero-author-icon">
                👥
            </div>

            <div>
                Disusun oleh:
                <b>Mohammad Idhom, Trimono, Ajeng Puspa, Shafira Amanda</b>
            </div>

        </div>

    </div>

</div>
""")