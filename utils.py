import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ======================================================================
# SESSION STATE
# ======================================================================

# Nilai default seluruh session_state yang dipakai lintas halaman.
SESSION_DEFAULTS = {
    "df": None,
    "original_df": None,
    "dataset_name": None,
    "date_column": None,
    "commodity_column": None,
    "analysis_range": None,
    "model_result": None,
    "model_data": None,
    "model_params": None,
}


def init_session_state():
    """Pastikan seluruh key session_state sudah terdaftar sebelum dipakai halaman mana pun."""
    for key, default in SESSION_DEFAULTS.items():
        st.session_state.setdefault(key, default)


# ======================================================================
# STYLING (CSS GLOBAL)
# ======================================================================

_CUSTOM_CSS = """
<style>

/* ==========================================================
   GLOBAL
========================================================== */

html{
    font-size:14px;
}

body,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"]{
    background:#FFFFFF;
    color:#31333F;
    font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
}

/* ==========================================================
   STREAMLIT HEADER
========================================================== */

header[data-testid="stHeader"]{
    background:#FFFFFF !important;
    border-bottom:1px solid #E6E6E9;
}

[data-testid="stToolbar"]{
    background:#FFFFFF !important;
}

/* ==========================================================
   MAIN CONTAINER
========================================================== */

.block-container{
    max-width:1400px;
    padding:1rem 2rem 1.5rem;
}

/* ==========================================================
   TYPOGRAPHY
========================================================== */

h1{
    font-size:2.6rem !important;
    font-weight:800 !important;
    color:#31333F;
    margin-bottom:.5rem;
}

h2{
    font-size:2rem !important;
    font-weight:700 !important;
    color:#31333F;
    margin-top:1.4rem;
    margin-bottom:.6rem;
}

h3{
    font-size:1.45rem !important;
    font-weight:700 !important;
    color:#31333F;
}

h4{
    font-size:1.15rem !important;
    font-weight:600 !important;
    color:#31333F;
}

p,
span,
label,
li{
    font-size:13px;
    line-height:1.7;
    color:#31333F;
}

/* ==========================================================
   HERO TITLE (Homepage)
========================================================== */

/* ==========================================================
   HOMEPAGE HERO
========================================================== */

.homepage-hero{
    position:relative;
    overflow:hidden;
    min-height:690px;
    background:
        radial-gradient(circle at 85% 20%, rgba(255,107,107,.07), transparent 28%),
        linear-gradient(135deg,#FFFFFF 0%,#FFFDFD 55%,#FFF7F5 100%);
    border:1px solid #E8E8EC;
    border-radius:28px;
    padding:70px 70px 45px;
    box-shadow:0 8px 30px rgba(49,51,63,.04);
}

/* Decorative dots */

.homepage-hero::before{
    content:"";
    position:absolute;
    top:55px;
    left:45px;
    width:110px;
    height:110px;
    opacity:.45;
    background-image:radial-gradient(#FF8A7A 1.7px, transparent 1.7px);
    background-size:22px 22px;
}

/* Right decorative circle */

.homepage-hero::after{
    content:"";
    position:absolute;
    width:330px;
    height:330px;
    right:-120px;
    top:210px;
    border-radius:50%;
    background:radial-gradient(
        circle,
        rgba(255,115,105,.10) 0%,
        rgba(255,115,105,.04) 45%,
        transparent 70%
    );
}

/* ==========================================================
   HERO LOGO
========================================================== */

.hero-logo{
    position:relative;
    z-index:2;
    width:118px;
    height:118px;
    margin:5px auto 35px;
    border-radius:30px;
    background:linear-gradient(135deg,#FF4B4B 0%,#FF9A8B 100%);
    display:flex;
    justify-content:center;
    align-items:center;
    color:white;
    font-size:52px;
    font-weight:800;
    box-shadow:
        0 18px 35px rgba(255,75,75,.18),
        inset 0 1px 0 rgba(255,255,255,.35);
}

/* ==========================================================
   HERO CONTENT
========================================================== */

.hero-content{
    position:relative;
    z-index:3;
    text-align:center;
    max-width:1050px;
    margin:0 auto;
}

.hero-kicker{
    display:inline-block;
    padding:7px 15px;
    margin-bottom:20px;
    border-radius:30px;
    background:#FFF0EE;
    color:#E84C4C;
    font-size:12px;
    font-weight:700;
    letter-spacing:.8px;
    text-transform:uppercase;
}

.hero-title{
    font-size:42px;
    font-weight:800;
    line-height:1.22;
    letter-spacing:-1.2px;
    color:#292B38;
    margin:0 auto;
    max-width:980px;
}

.hero-accent{
    width:90px;
    height:5px;
    border-radius:10px;
    background:linear-gradient(90deg,#FF4B4B,#FFB199);
    margin:28px auto 24px;
}

.hero-authors{
    display:flex;
    align-items:center;
    justify-content:center;
    gap:12px;
    font-size:19px;
    font-weight:600;
    color:#4E5060;
}

.hero-author-icon{
    width:38px;
    height:38px;
    border-radius:50%;
    background:linear-gradient(135deg,#FF5A52,#FF9E8F);
    color:white;
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:23px;
}

/* ==========================================================
   DECORATIVE BOTTOM WAVE
========================================================== */

.hero-wave{
    position:absolute;
    left:-5%;
    bottom:-5px;
    width:110%;
    height:150px;
    z-index:1;
    opacity:.9;
}

.hero-wave svg{
    width:100%;
    height:100%;
}

/* ==========================================================
   SMALL SCREEN
========================================================== */

@media(max-width:900px){

    .homepage-hero{
        padding:50px 30px 35px;
        min-height:620px;
    }

    .hero-title{
        font-size:31px;
    }

    .hero-logo{
        width:95px;
        height:95px;
        font-size:42px;
        border-radius:24px;
    }

    .hero-authors{
        font-size:13px;
    }
}
/* ==========================================================
   SIDEBAR
========================================================== */

section[data-testid="stSidebar"]{
    width:245px !important;
    background:#F6F7FA;
    border-right:1px solid #E7E8EC;
}

section[data-testid="stSidebar"] > div{
    background:#F6F7FA;
}

section[data-testid="stSidebarContent"]{
    padding:18px 16px;
}

/* Hide default Streamlit multipage navigation */
[data-testid="stSidebarNav"]{
    display:none !important;
}

[data-testid="stSidebarNavItems"]{
    display:none !important;
}


/* ==========================================================
   SIDEBAR BRAND
========================================================== */

.sidebar-brand{
    display:flex;
    align-items:center;
    gap:11px;
    margin-bottom:28px;
}

.sidebar-brand-text{
    line-height:1.25;
}

.sidebar-brand-title{
    font-size:15px;
    font-weight:750;
    color:#31333F;
}

.sidebar-brand-subtitle{
    font-size:11px;
    color:#737687;
    margin-top:3px;
}


/* ==========================================================
   SIDEBAR TITLE
========================================================== */

.sidebar-title{
    font-size:15px;
    font-weight:700;
    color:#31333F;
    margin:0 0 10px 2px;
}


/* ==========================================================
   PAGE LINK
========================================================== */

div[data-testid="stPageLink"]{
    margin-bottom:4px;
}

div[data-testid="stPageLink"] a{
    display:flex;
    align-items:center;
    min-height:40px;
    padding:8px 12px;
    border-radius:11px;
    font-size:13px;
    font-weight:500;
    color:#424552;
    transition:
        background .18s ease,
        color .18s ease;
}

div[data-testid="stPageLink"] a:hover{
    background:#EAECF3;
    color:#31333F;
}

div[data-testid="stPageLink"][aria-current="page"] a{
    background:#E9ECF4;
    color:#31333F;
    font-weight:650;
}

div[data-testid="stPageLink"] a::before{
    content:"•";
    color:#9BA0AD;
    margin-right:9px;
    font-size:11px;
}

div[data-testid="stPageLink"][aria-current="page"] a::before{
    color:#6E7382;
}


/* ==========================================================
   SIDEBAR DIVIDER
========================================================== */

.sidebar-divider{
    border-top:1px solid #E2E4E9;
    margin:20px 0 14px;
}


/* ==========================================================
   SIDEBAR FOOTER
========================================================== */

section[data-testid="stSidebar"] .stCaption{
    font-size:11px !important;
    color:#858997 !important;
}


/* ==========================================================
   PAGE HEADER
========================================================== */

.page-breadcrumb{
    font-size:11px;
    color:#9295A0;
    margin-bottom:5px;
}

.page-title{
    font-size:32px;
    font-weight:800;
    letter-spacing:-.5px;
    color:#31333F;
    margin:0;
}

.page-caption{
    font-size:13px;
    color:#747784;
    margin-top:5px;
}


/* ==========================================================
   CARD
========================================================== */

.card{
    background:#FFFFFF;
    border:1px solid #E5E6EA;
    border-radius:16px;
    padding:22px;
    box-shadow:0 3px 12px rgba(49,51,63,.025);
}

.stMarkdown .card h3{
    font-size:18px !important;
    font-weight:700 !important;
    margin:0 0 10px !important;
    color:#31333F !important;
}

.stMarkdown .card p{
    font-size:13px !important;
    line-height:1.7 !important;
    color:#626574 !important;
}


/* ==========================================================
   STREAMLIT CONTAINER
========================================================== */

div[data-testid="stVerticalBlockBorderWrapper"]{
    border:1px solid #E5E6EA;
    border-radius:16px;
    background:#FFFFFF;
}


/* ==========================================================
   INFO / WARNING / ERROR / SUCCESS
========================================================== */

div[data-testid="stInfo"],
div[data-testid="stWarning"],
div[data-testid="stError"],
div[data-testid="stSuccess"]{
    border-radius:12px;
    border:1px solid #E5E6EA;
    padding:.75rem 1rem;
    font-size:13px;
}


/* ==========================================================
   METRIC
========================================================== */

[data-testid="stMetric"]{
    background:#FFFFFF;
    border:1px solid #E5E6EA;
    border-radius:14px;
    padding:16px;
}

[data-testid="stMetricLabel"]{
    font-size:12px !important;
    color:#777B88 !important;
}

[data-testid="stMetricValue"]{
    font-size:25px !important;
    font-weight:750 !important;
    color:#31333F !important;
}

[data-testid="stMetricDelta"]{
    font-size:12px !important;
}

[data-testid="stMetricDelta"] svg{
    display:none;
}


/* ==========================================================
   BUTTON
========================================================== */

.stButton button{
    border-radius:10px;
    border:1px solid #E0E2E7;
    font-size:13px;
    font-weight:600;
    padding:.48rem 1rem;
    transition:.18s ease;
}

.stButton button:hover{
    border-color:#FF8A7A;
}


/* ==========================================================
   INPUT
========================================================== */

.stTextInput label,
.stSelectbox label,
.stNumberInput label,
.stDateInput label,
.stRadio label,
.stCheckbox label{
    font-size:13px;
    font-weight:600;
    color:#454855;
}

.stTextInput input,
.stNumberInput input{
    font-size:13px;
    border-radius:9px;
}

.stSelectbox div[data-baseweb="select"]{
    font-size:13px;
}


/* ==========================================================
   DATAFRAME
========================================================== */

[data-testid="stDataFrame"]{
    font-size:13px;
}


/* ==========================================================
   TABS
========================================================== */

button[data-baseweb="tab"]{
    font-size:13px;
    padding:9px 16px;
}


/* ==========================================================
   SCROLLBAR
========================================================== */

::-webkit-scrollbar{
    width:6px;
    height:6px;
}

::-webkit-scrollbar-track{
    background:transparent;
}

::-webkit-scrollbar-thumb{
    background:#D9DBE2;
    border-radius:10px;
}
</style>
"""


# path halaman & label navigasi (urutan sesuai konsep tampilan awal)
NAV_ITEMS = [
    ("homepage.py", "Homepage"),
    ("pages/input_data.py", "Input Dataset"),
    ("pages/analisis_desk.py", "Analisis Deskriptif"),
    ("pages/input_params.py", "Input Parameter"),
    ("pages/output.py", "Output"),
]


def inject_custom_css():
    """Suntikkan CSS global (card, sidebar, page link, info box, container)."""
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

def render_sidebar():
    """Render sidebar KomoditasAI."""

    with st.sidebar:

        # ======================================================
        # BRAND
        # ======================================================

        st.html("""
        <div style="
            display:flex;
            align-items:center;
            gap:10px;
            margin-bottom:28px;
        ">

            <div style="
                width:48px;
                height:48px;
                flex-shrink:0;
                border-radius:13px;
                background:linear-gradient(
                    135deg,
                    #FF4B4B,
                    #FFB199
                );
                display:flex;
                justify-content:center;
                align-items:center;
                color:white;
                font-weight:800;
                font-size:20px;
                box-shadow:
                    0 7px 16px rgba(255,75,75,.16);
            ">
                Rp
            </div>

            <div style="
                line-height:1.25;
            ">

                <div style="
                    font-size:15px;
                    font-weight:750;
                    color:#31333F;
                ">
                    KomoditasAI
                </div>

                <div style="
                    font-size:11px;
                    color:#737687;
                    margin-top:3px;
                ">
                    Stacking Ensemble Dashboard
                </div>

            </div>

        </div>
        """)

        # ======================================================
        # NAVIGATION
        # ======================================================

        st.markdown(
            '<div class="sidebar-title">Navigasi</div>',
            unsafe_allow_html=True,
        )

        for path, label in NAV_ITEMS:
            st.page_link(path, label=label)

        # ======================================================
        # FOOTER
        # ======================================================

        st.markdown(
            '<div class="sidebar-divider"></div>',
            unsafe_allow_html=True,
        )

        st.caption("© 2026 • KomoditasAI Dashboard")

def page_header(breadcrumb: str, title: str, caption: str = ""):
    """Header konsisten untuk tiap halaman: breadcrumb, judul, dan sub-judul."""
    st.markdown(f"`{breadcrumb}`")
    st.title(title)
    if caption:
        st.caption(caption)


def setup_page(page_title: str, page_icon: str, breadcrumb: str, title: str, caption: str = ""):
    """
    Satu pemanggilan untuk seluruh boilerplate awal sebuah halaman:
    page_config -> session_state -> CSS -> sidebar -> header.
    Dipanggil paling atas, tepat setelah import.
    """
    st.set_page_config(page_title=page_title, page_icon=page_icon, layout="wide")
    init_session_state()
    inject_custom_css()
    render_sidebar()
    page_header(breadcrumb, title, caption)


# ======================================================================
# GUARD / VALIDASI ALUR HALAMAN
# ======================================================================

def require_dataset():
    """
    Pastikan dataset & pemetaan kolom (tanggal, harga) sudah tersedia.
    Menghentikan halaman dengan pesan yang konsisten jika belum siap.
    """
    if st.session_state.get("df") is None or len(st.session_state.df) == 0:
        st.warning("Silakan unggah dan pilih dataset terlebih dahulu pada halaman **Input Dataset**.")
        st.stop()
    if st.session_state.get("date_column") is None:
        st.error("Kolom tanggal belum ditentukan pada halaman Input Dataset.")
        st.stop()
    if st.session_state.get("commodity_column") is None:
        st.error("Kolom harga belum ditentukan pada halaman Input Dataset.")
        st.stop()

    return (
        st.session_state.df.copy(),
        st.session_state.date_column,
        st.session_state.commodity_column,
    )


def require_trained_model():
    """Pastikan proses training (Input Parameter) sudah pernah dijalankan."""
    if st.session_state.get("model_result") is None:
        st.warning("Silakan jalankan proses training terlebih dahulu pada halaman **Input Parameter**.")
        st.stop()

    return (
        st.session_state.model_result,
        st.session_state.model_data,
        st.session_state.model_params,
    )


# ======================================================================
# PEMBERSIHAN DATA HARGA KOMODITAS
# ======================================================================

def clean_commodity_series(df: pd.DataFrame, commodity_column: str) -> pd.Series:
    """
    Bersihkan kolom harga komoditas dari format Rupiah (mis. "Rp12.345,67")
    menjadi nilai numerik (float), dan ubah placeholder ("-", "nan", dst) menjadi NaN.
    """
    cleaned = (
        df[commodity_column]
        .astype(str)
        .str.replace("Rp", "", regex=False)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip()
        .replace(["-", "", "nan", "None"], np.nan)
    )
    return pd.to_numeric(cleaned, errors="coerce")


# ======================================================================
# FORMATTING ANGKA (GAYA INDONESIA)
# ======================================================================

def format_id(value, decimal: int = 0) -> str:
    """Format angka dengan pemisah ribuan '.' dan desimal ',' (gaya Indonesia)."""
    return f"{value:,.{decimal}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def format_rupiah(value) -> str:
    """Format angka menjadi teks Rupiah, mis. 12345.6 -> 'Rp12.345,6'."""
    text = format_id(value, decimal=2).rstrip("0").rstrip(",")
    return f"Rp{text}"


# ======================================================================
# METRIK EVALUASI MODEL (dipakai di Input Parameter & Output)
# ======================================================================

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape_safe(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > 1e-12
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.abs(y_true) + np.abs(y_pred)
    mask = denominator > 1e-12
    if not mask.any():
        return np.nan
    return float(np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denominator[mask]) * 100)


def mase(y_true, y_pred, insample) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    insample = np.asarray(insample, dtype=float)
    scale = np.mean(np.abs(np.diff(insample)))
    if scale <= 1e-12:
        return np.nan
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def evaluate_prediction(y_true, y_pred, insample, model_name: str) -> dict:
    """Ringkasan metrik evaluasi (RMSE, MAE, MAPE, sMAPE, MASE, R2, Bias) untuk satu model."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return {
        "Model": model_name,
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE (%)": mape_safe(y_true, y_pred),
        "sMAPE (%)": smape(y_true, y_pred),
        "MASE": mase(y_true, y_pred, insample),
        "R2": float(r2_score(y_true, y_pred)),
        "Bias": float(np.mean(y_pred - y_true)),
    }