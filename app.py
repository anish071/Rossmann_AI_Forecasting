import os
import io
from huggingface_hub import hf_hub_download
import glob
import json
from datetime import date

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Rossmann AI Forecast",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# PATHS  (train.csv is OPTIONAL - unlocks real KPIs, map, trends)
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SALES_MODEL_PATH = os.path.join(MODELS_DIR, "rossmann_rf_01-10-2026-14-31-51.pkl")
CUSTOMER_MODEL_PATH = os.path.join(MODELS_DIR, "rossmann_customer_rf_01-10-2026-15-40-57.pkl")
STORE_FILE = os.path.join(BASE_DIR, "store.csv")
TRAIN_FILE = os.path.join(BASE_DIR, "train.csv")

# Your real metrics (from your notebook)
MODEL_METRICS = {"Sales RF MAE": 809.63, "Customer RF MAE": 56.45}
# Placeholder comparison numbers (from the mockup) -> replace with YOUR RMSE values
DEFAULT_EUR_INR = 108.4  # 1 EUR in INR (2 Oct 2026). Editable from the sidebar.

_orig_plotly_chart = st.plotly_chart


def _plotly_chart(fig, **kw):
    kw.pop("use_container_width", None)
    try:
        return _orig_plotly_chart(fig, width="stretch", **kw)
    except TypeError:  # older Streamlit
        return _orig_plotly_chart(fig, use_container_width=True, **kw)


st.plotly_chart = _plotly_chart

# =========================================================
# CSS  (glass + 3D + animated background)
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp {
  background:
    radial-gradient(circle at 10% 10%, rgba(0,255,255,.13), transparent 28%),
    radial-gradient(circle at 90% 15%, rgba(140,80,255,.18), transparent 30%),
    radial-gradient(circle at 50% 100%, rgba(0,120,255,.12), transparent 35%),
    #050816;
  color: white;
}
.stApp::before { content:""; position:fixed; width:500px; height:500px; border-radius:50%;
  background: radial-gradient(circle, rgba(0,255,255,.10), transparent 65%);
  top:-180px; left:-120px; animation: float1 8s ease-in-out infinite; pointer-events:none; }
.stApp::after { content:""; position:fixed; width:450px; height:450px; border-radius:50%;
  background: radial-gradient(circle, rgba(130,70,255,.12), transparent 65%);
  bottom:-180px; right:-120px; animation: float2 10s ease-in-out infinite; pointer-events:none; }
@keyframes float1 { 0%,100%{transform:translate(0,0)} 50%{transform:translate(100px,70px)} }
@keyframes float2 { 0%,100%{transform:translate(0,0)} 50%{transform:translate(-80px,-60px)} }

.hero { padding: 10px 4px 14px 4px; }
.hero-title { font-size: 38px; font-weight: 800; letter-spacing:-1px;
  background: linear-gradient(90deg,#00ffff,#8b5cf6,#38bdf8); background-size:200% auto;
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; animation: shimmer 6s linear infinite; }
@keyframes shimmer { to { background-position: 200% center; } }
.hero-subtitle { color:#94a3b8; font-size:15px; margin-top:4px; }

.glass { background: linear-gradient(145deg, rgba(255,255,255,.09), rgba(255,255,255,.025));
  border:1px solid rgba(255,255,255,.10); border-radius:20px; padding:20px;
  box-shadow: 0 20px 50px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.06);
  backdrop-filter: blur(18px); }
.sec-title { font-size:18px; font-weight:700; margin: 8px 0 6px 0; }

/* 3D KPI cards (static HTML version) */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 20px;
    width: 100%;
    margin: 10px 0 25px 0;
}

@media (max-width: 1100px) {
    .kpi-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

@media (max-width: 700px) {
    .kpi-grid {
        grid-template-columns: 1fr;
    }
}
.kpi-card { position:relative; padding:22px; min-height:140px; border-radius:22px; overflow:hidden;
  background: radial-gradient(circle at 20% 10%, rgba(0,220,255,.16), transparent 45%),
              linear-gradient(145deg, rgba(255,255,255,.14), rgba(255,255,255,.035));
  border:1px solid rgba(120,200,255,.25);
  box-shadow: 0 18px 45px rgba(0,0,0,.45), inset 0 1px 0 rgba(255,255,255,.12);
  transform: perspective(900px) translateZ(0);
  transition: transform .35s ease, box-shadow .35s ease, border-color .35s ease; }
.kpi-card:hover { transform: perspective(900px) rotateX(4deg) rotateY(-4deg) translateY(-10px) scale(1.025);
  border-color: rgba(0,220,255,.65); box-shadow: 0 30px 70px rgba(0,0,0,.6), 0 0 35px rgba(0,220,255,.18); }
.kpi-card::after { content:""; position:absolute; top:0; left:-120%; width:70%; height:100%;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,.12), transparent);
  transform: skewX(-20deg); animation: kpiShine 5s infinite; pointer-events:none; }
@keyframes kpiShine { 0%{left:-120%} 45%{left:140%} 100%{left:140%} }
.kpi-icon { font-size:28px; margin-bottom:8px; }
.kpi-label { color:#94a3b8; font-size:13px; font-weight:500; }
.kpi-value { color:#f8fafc; font-size:27px; font-weight:800; margin-top:6px; }
.kpi-sub { color:#22c55e; font-size:12px; margin-top:6px; }

/* model cards */
.model-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:20px; margin-top:14px; }
.model-card { padding:24px; min-height:150px; border-radius:22px;
  background: radial-gradient(circle at 20% 10%, rgba(139,92,246,.18), transparent 45%),
              linear-gradient(145deg, rgba(255,255,255,.12), rgba(255,255,255,.035));
  border:1px solid rgba(139,92,246,.25); box-shadow: 0 18px 45px rgba(0,0,0,.45);
  transition: all .35s ease; }
.model-card:hover { transform: perspective(900px) rotateX(3deg) rotateY(-3deg) translateY(-8px) scale(1.02);
  border-color: rgba(139,92,246,.65); box-shadow: 0 30px 70px rgba(0,0,0,.6), 0 0 35px rgba(139,92,246,.18); }
.model-icon { font-size:32px; margin-bottom:10px; }
.model-name { font-size:18px; font-weight:700; }
.model-status { margin-top:8px; color:#4ade80; font-weight:600; }
.model-metric { margin-top:8px; color:#94a3b8; font-size:13px; }

.tbl-wrap { max-height:460px; overflow:auto; border-radius:16px; border:1px solid rgba(255,255,255,.10);
  background: linear-gradient(145deg, rgba(255,255,255,.07), rgba(255,255,255,.02)); box-shadow:0 18px 45px rgba(0,0,0,.35); }
.tbl { width:100%; border-collapse:collapse; font-size:13px; color:#e2e8f0; }
.tbl th { position:sticky; top:0; background:#0f172a; color:#94a3b8; text-align:left; padding:11px 12px;
  font-weight:600; letter-spacing:.3px; border-bottom:1px solid rgba(255,255,255,.12); white-space:nowrap; }
.tbl td { padding:9px 12px; border-bottom:1px solid rgba(255,255,255,.05); white-space:nowrap; }
.tbl tbody tr:hover { background: rgba(0,220,255,.08); }
.tbl td.num { text-align:right; font-variant-numeric: tabular-nums; font-weight:600; }
.pill { padding:2px 10px; border-radius:999px; font-size:11px; font-weight:700; }
.pill.green { background:rgba(34,197,94,.18); color:#4ade80; }
.pill.red { background:rgba(239,68,68,.18); color:#f87171; }
.pill.violet { background:rgba(139,92,246,.24); color:#c4b5fd; }
.pill.gray { background:rgba(148,163,184,.15); color:#94a3b8; }
[data-testid="stWidgetLabel"] p, [data-testid="stCaptionContainer"] { color:#cbd5e1; }
/* sidebar + 3D cube logo */
section[data-testid="stSidebar"] { background: linear-gradient(180deg, rgba(8,15,35,.97), rgba(4,8,20,.98));
  border-right:1px solid rgba(255,255,255,.08); }
.cube-wrap { perspective:600px; height:90px; display:flex; justify-content:center; align-items:center; }
.cube { width:44px; height:44px; position:relative; transform-style:preserve-3d; animation: spin 9s linear infinite; }
.cube div { position:absolute; width:44px; height:44px; border:1px solid rgba(0,255,255,.6);
  background: linear-gradient(135deg, rgba(0,255,255,.25), rgba(139,92,246,.35)); }
.cube .f{transform:translateZ(22px)} .cube .b{transform:rotateY(180deg) translateZ(22px)}
.cube .r{transform:rotateY(90deg) translateZ(22px)} .cube .l{transform:rotateY(-90deg) translateZ(22px)}
.cube .t{transform:rotateX(90deg) translateZ(22px)} .cube .d{transform:rotateX(-90deg) translateZ(22px)}
@keyframes spin { from{transform:rotateX(-20deg) rotateY(0)} to{transform:rotateX(-20deg) rotateY(360deg)} }

.stButton > button, .stDownloadButton > button { width:100%; border-radius:12px; border:1px solid rgba(0,255,255,.25);
  background: linear-gradient(90deg,#0891b2,#6366f1); color:white; font-weight:700; transition:.25s; }
.stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-2px); box-shadow: 0 10px 30px rgba(0,200,255,.25); color:white; }
[data-testid="stFileUploader"] { background: rgba(255,255,255,.035); border-radius:16px; padding:8px; border:1px dashed rgba(0,255,255,.25); }
#MainMenu {visibility:hidden;} footer {visibility:hidden;} header {background:transparent !important;}
@media (max-width: 900px) { .model-grid { grid-template-columns: 1fr; } }
</style>
""", unsafe_allow_html=True)


# =========================================================
# LOADERS
# =========================================================
def _latest(pattern):
    files = sorted(glob.glob(os.path.join(MODELS_DIR, pattern)))
    return files[-1] if files else None


@st.cache_resource
def load_models():
    sales_model_path = hf_hub_download(
        repo_id="anish071-ai/rossmann-models",
        filename="rossmann_lgbm_sales_compact.pkl"
    )

    customer_model_path = hf_hub_download(
        repo_id="anish071-ai/rossmann-models",
        filename="rossmann_lgbm_customers_compact.pkl"
    )

    sales_model = joblib.load(sales_model_path)
    customer_model = joblib.load(customer_model_path)

    return sales_model, customer_model

@st.cache_data
def load_store_data():
    return pd.read_csv(STORE_FILE)


@st.cache_data
def load_train():
    if not os.path.exists(TRAIN_FILE):
        return None
    df = pd.read_csv(
        TRAIN_FILE,
        usecols=["Store", "DayOfWeek", "Date", "Sales", "Customers", "Open", "Promo"],
        parse_dates=["Date"],
    )
    return df


try:
    sales_model, customer_model = load_models()
    store_data = load_store_data()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"Model loading error: {e}")
    try:
        store_data = load_store_data()
    except Exception:
        store_data = pd.DataFrame()

train_data = load_train()


# =========================================================
# FEATURE ENGINEERING + PREDICTION  (same pipeline as your models)
# =========================================================
def prepare_features(df, store_df):
    data = df.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.merge(store_df, on="Store", how="left")

    data["Year"] = data["Date"].dt.year
    data["Month"] = data["Date"].dt.month
    data["Day"] = data["Date"].dt.day
    data["WeekOfYear"] = data["Date"].dt.isocalendar().week.astype(int)
    data["IsWeekend"] = data["DayOfWeek"].isin([6, 7]).astype(int)
    data["MonthPeriod"] = data["Date"].dt.to_period("M").astype(str)
    data["IsHoliday"] = (data["StateHoliday"].astype(str) != "0").astype(int)
    data["HolidayBefore"] = 0
    data["HolidayAfter"] = 0

    comp_start = pd.to_datetime(
        dict(year=data["CompetitionOpenSinceYear"], month=data["CompetitionOpenSinceMonth"], day=1),
        errors="coerce",
    )
    data["CompetitionAgeDays"] = (data["Date"] - comp_start).dt.days.fillna(0).clip(lower=0)
    data["Promo2Active"] = data["Promo2"].fillna(0)
    data["DaysSinceCompetitionOpen"] = data["CompetitionAgeDays"]
    return data


NUMERIC = [
    "Store", "DayOfWeek", "Open", "Promo", "SchoolHoliday", "CompetitionDistance",
    "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear", "Promo2", "Promo2SinceWeek",
    "Promo2SinceYear", "Year", "Month", "Day", "WeekOfYear", "IsWeekend", "IsHoliday",
    "HolidayBefore", "HolidayAfter", "CompetitionAgeDays", "Promo2Active", "DaysSinceCompetitionOpen",
]
CATEGORICAL = ["StateHoliday", "StoreType", "Assortment", "PromoInterval", "MonthPeriod"]


def _predict_raw(input_df, store_df=None):
    store_df = store_data if store_df is None else store_df
    result = input_df.copy()
    if isinstance(sales_model, dict):  # improved bundles from train.py
        from train import predict_bundle
        result["PredictedSales"] = predict_bundle(sales_model, input_df, store_df)
        result["PredictedCustomers"] = predict_bundle(customer_model, input_df, store_df)
        closed = result["Open"] == 0
        result.loc[closed, ["PredictedSales", "PredictedCustomers"]] = 0
        return result
    feats = prepare_features(input_df, store_df)
    X = feats[NUMERIC + CATEGORICAL].copy()
    result["PredictedSales"] = np.maximum(sales_model.predict(X), 0)
    result["PredictedCustomers"] = np.maximum(customer_model.predict(X), 0)
    closed = result["Open"] == 0
    result.loc[closed, ["PredictedSales", "PredictedCustomers"]] = 0
    return result



def make_prediction(input_df, store_df=None):
    """Models predict in EUR (Rossmann data). Convert sales to INR with the sidebar rate."""
    out = _predict_raw(input_df, store_df)
    out["PredictedSales"] = out["PredictedSales"] * RATE
    return out


@st.cache_data(show_spinner=False)
def predict_cached(df, rate):
    return make_prediction(df)


# =========================================================
# INDIAN RUPEE FORMAT + STYLED TABLE
# =========================================================
def ind(v, decimals=0):
    """Indian digit grouping: 23,92,76,880"""
    neg = v < 0
    s = f"{abs(v):.{decimals}f}"
    whole, _, frac = s.partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        whole = ",".join(parts + [tail])
    return ("-" if neg else "") + whole + ("." + frac if frac else "")


def fmt_inr(v, decimals=0):
    return "\u20b9 " + ind(v, decimals)


def short_inr(v):
    if v >= 1e7:
        return f"\u20b9 {v / 1e7:.2f} Cr"
    if v >= 1e5:
        return f"\u20b9 {v / 1e5:.2f} L"
    return fmt_inr(v)


DAY_NAMES = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}


def pill(txt, kind):
    return f'<span class="pill {kind}">{txt}</span>'


def forecast_table(fc, max_rows=200):
    cols = ["Id", "Store", "Date", "Day", "Status", "Promo", "State Hol.", "School Hol.", "Pred. Sales", "Pred. Customers"]
    head = "".join(f"<th>{c}</th>" for c in cols)
    body = []
    for i, r in enumerate(fc.head(max_rows).to_dict("records")):
        sh = str(r.get("StateHoliday", "0"))
        cells = [
            str(r.get("Id", i + 1)), str(r["Store"]), pd.Timestamp(r["Date"]).strftime("%d %b %Y"),
            DAY_NAMES.get(int(r["DayOfWeek"]), ""),
            pill("Open", "green") if r["Open"] == 1 else pill("Closed", "red"),
            pill("Promo", "violet") if r["Promo"] == 1 else pill("-", "gray"),
            "-" if sh == "0" else pill(sh.upper(), "violet"),
            pill("Yes", "violet") if int(r["SchoolHoliday"]) == 1 else "-",
            fmt_inr(r["PredictedSales"]), ind(r["PredictedCustomers"]),
        ]
        tds = "".join(f'<td class="num">{c}</td>' if k >= 8 else f"<td>{c}</td>" for k, c in enumerate(cells))
        body.append(f"<tr>{tds}</tr>")
    rows_html = "".join(body)
    st.markdown(f'<div class="tbl-wrap"><table class="tbl"><thead><tr>{head}</tr></thead><tbody>{rows_html}</tbody></table></div>',
                unsafe_allow_html=True)
    if len(fc) > max_rows:
        st.caption(f"Showing first {max_rows:,} of {len(fc):,} rows - download the CSV for everything.")


# =========================================================
# UI HELPERS
# =========================================================
def hero(title, subtitle):
    st.markdown(f'<div class="hero"><div class="hero-title">{title}</div>'
                f'<div class="hero-subtitle">{subtitle}</div></div>', unsafe_allow_html=True)


def card(icon, label, value, sub=""):
    return (f'<div class="kpi-card"><div class="kpi-icon">{icon}</div><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>')


def style_fig(fig, h=380):
    fig.update_layout(template="plotly_dark", height=h, margin=dict(l=10, r=10, t=40, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      hovermode="x unified", legend=dict(orientation="h", y=1.12))
    return fig


KPI_HTML = """<!DOCTYPE html><html><head><style>
*{box-sizing:border-box;font-family:Inter,'Segoe UI',sans-serif}
body{margin:0;padding:6px 4px;background:transparent;overflow:hidden}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;padding:0 6px;perspective:1000px}
.card{position:relative;display:flex;align-items:center;gap:14px;padding:16px;border-radius:20px;height:112px;min-width:0;color:#fff;
 background:linear-gradient(145deg,rgba(255,255,255,.14),rgba(255,255,255,.03));border:1px solid rgba(255,255,255,.14);
 box-shadow:0 18px 40px rgba(0,0,0,.45),inset 0 1px 0 rgba(255,255,255,.15);transform-style:preserve-3d;
 transition:transform .15s ease-out;animation:rise .8s backwards}
@keyframes rise{from{opacity:0;transform:translateY(40px) rotateX(-30deg)}}
.icon{width:56px;height:56px;border-radius:16px;display:grid;place-items:center;font-size:28px;transform:translateZ(22px);
 box-shadow:0 10px 20px rgba(0,0,0,.4);animation:bob 3s ease-in-out infinite}
@keyframes bob{50%{transform:translateZ(22px) translateY(-4px)}}
.label{font-size:12px;color:#94a3b8;transform:translateZ(15px)}
.value{font-size:clamp(18px,2vw,25px);font-weight:800;white-space:nowrap;transform:translateZ(15px)}
.delta{font-size:12px;color:#4ade80;transform:translateZ(15px)}
@media(max-width:700px){.grid{grid-template-columns:repeat(2,1fr)}}
</style></head><body><div class="grid" id="g"></div><script>
const items=__DATA__;
const g=document.getElementById('g');
function fmt(it,v){
 if(it.prefix.indexOf('₹')>=0){ if(v>=1e7)return it.prefix+(v/1e7).toFixed(2)+' Cr'; if(v>=1e5)return it.prefix+(v/1e5).toFixed(2)+' L'; }
 return it.prefix+v.toLocaleString('en-IN',{minimumFractionDigits:it.decimals,maximumFractionDigits:it.decimals})+it.suffix;
}
items.forEach((it,i)=>{
 const c=document.createElement('div');c.className='card';c.style.animationDelay=(i*0.12)+'s';
 c.innerHTML=`<div class="icon" style="background:${it.color}">${it.icon}</div><div><div class="label">${it.label}</div><div class="value"><span class="n">0</span></div><div class="delta">${it.delta||''}</div></div>`;
 g.appendChild(c);
 const n=c.querySelector('.n');const t0=performance.now()+i*120;
 function f(t){const p=Math.min(Math.max((t-t0)/1400,0),1);const e=1-Math.pow(1-p,3);
  n.textContent=fmt(it,it.value*e);
  if(p<1)requestAnimationFrame(f);}
 requestAnimationFrame(f);
 c.addEventListener('mousemove',e=>{const r=c.getBoundingClientRect();const x=(e.clientX-r.left)/r.width-.5,y=(e.clientY-r.top)/r.height-.5;
  c.style.transform=`rotateY(${x*12}deg) rotateX(${-y*12}deg) translateZ(10px)`;});
 c.addEventListener('mouseleave',()=>{c.style.transform='';});
});
</script></body></html>"""


def kpi_row(items):
    """Animated count-up + mouse-tilt 3D KPI cards."""
    html = KPI_HTML.replace("__DATA__", json.dumps(items))
    try:
        components.html(html, height=150)
    except Exception:
        st.iframe(html, height=150)


def dashboard_kpis():
    if train_data is not None:
        op = train_data[train_data["Open"] == 1]
        items = [
            dict(icon="🏬", label="Total Stores", value=int(train_data["Store"].nunique()), prefix="", suffix="", decimals=0,
                 color="linear-gradient(135deg,#2563eb,#38bdf8)", delta="Rossmann network"),
            dict(icon="💶", label="Avg. Daily Sales", value=float(op["Sales"].mean()), prefix="₹ ", suffix="", decimals=0,
                 color="linear-gradient(135deg,#16a34a,#4ade80)", delta="open days only"),
            dict(icon="👥", label="Avg. Customers", value=float(op["Customers"].mean()), prefix="", suffix="", decimals=0,
                 color="linear-gradient(135deg,#7c3aed,#60a5fa)", delta="per store / day"),
            dict(icon="🎯", label="Promo Days", value=float(train_data["Promo"].mean() * 100), prefix="", suffix="%", decimals=1,
                 color="linear-gradient(135deg,#db2777,#c084fc)", delta="share of all days"),
        ]
    else:
        s = store_data
        items = [
            dict(icon="🏬", label="Total Stores", value=int(len(s)), prefix="", suffix="", decimals=0,
                 color="linear-gradient(135deg,#2563eb,#38bdf8)", delta="from store.csv"),
            dict(icon="📏", label="Avg Competition Dist.", value=float(s["CompetitionDistance"].mean()), prefix="", suffix=" m", decimals=0,
                 color="linear-gradient(135deg,#16a34a,#4ade80)", delta="nearest competitor"),
            dict(icon="🎁", label="Stores on Promo2", value=float(s["Promo2"].mean() * 100), prefix="", suffix="%", decimals=1,
                 color="linear-gradient(135deg,#7c3aed,#60a5fa)", delta="continuing promotion"),
            dict(icon="📦", label="Extended Assortment", value=float((s["Assortment"] == "c").mean() * 100), prefix="", suffix="%", decimals=1,
                 color="linear-gradient(135deg,#db2777,#c084fc)", delta="assortment c"),
        ]
    kpi_row(items)


# =========================================================
# FORECAST PANEL (form + results)
# =========================================================
def build_input(sid, start, days, promo_mode, state_hol, school_hol, sunday_closed):
    dates = pd.date_range(start, periods=days)
    dow = dates.dayofweek + 1  # 1=Mon ... 7=Sun
    if promo_mode.startswith("Yes"):
        promo = np.ones(days, dtype=int)
    elif promo_mode.startswith("No"):
        promo = np.zeros(days, dtype=int)
    else:  # alternate weeks Mon-Fri
        promo = ((dow <= 5) & (((dates - dates[0]).days // 7) % 2 == 0)).astype(int)
    sh = np.array(["0"] * days, dtype=object)
    sh[0] = state_hol
    opened = np.ones(days, dtype=int)
    if sunday_closed:
        opened[dow == 7] = 0
    if state_hol != "0":
        opened[0] = 0
    return pd.DataFrame({
        "Id": range(1, days + 1), "Store": sid, "DayOfWeek": dow, "Date": dates,
        "Open": opened, "Promo": promo, "StateHoliday": sh, "SchoolHoliday": int(school_hol),
    })


def forecast_figure(fc, sid):
    fig = go.Figure()
    if train_data is not None:
        act = train_data[(train_data["Store"] == sid) & train_data["Date"].between(fc["Date"].min(), fc["Date"].max())]
        if not act.empty:
            fig.add_trace(go.Scatter(x=act["Date"], y=act["Sales"], name="Actual Sales", mode="lines+markers",
                                     line=dict(color="#38bdf8", width=3)))
    fig.add_trace(go.Bar(x=fc["Date"], y=fc["PredictedCustomers"], name="Predicted Customers", yaxis="y2",
                         marker=dict(color="rgba(217,70,239,.65)")))
    fig.add_trace(go.Scatter(x=fc["Date"], y=fc["PredictedSales"], name="Predicted Sales", mode="lines+markers",
                             line=dict(color="#8b5cf6", width=4, dash="dot", shape="spline"),
                             fill="tozeroy", fillcolor="rgba(139,92,246,.10)"))
    style_fig(fig, 400)
    fig.update_layout(
        yaxis=dict(title="Sales (₹)"),
        yaxis2=dict(title="Customers", overlaying="y", side="right", showgrid=False),
        transition=dict(duration=600),
    )
    return fig


def run_forecast(p):
    sid = p["sid"]
    if p.get("use_csv") and uploaded_file is not None:
        uploaded_file.seek(0)
        raw = pd.read_csv(uploaded_file, dtype={"StateHoliday": str})
        req = ["Store", "DayOfWeek", "Date", "Open", "Promo", "StateHoliday", "SchoolHoliday"]
        miss = [c for c in req if c not in raw.columns]
        if miss:
            raise ValueError(f"Uploaded CSV is missing columns: {miss}")
        inp = raw[raw["Store"] == sid].copy()
        if inp.empty:
            raise ValueError(f"Store {sid} not found in uploaded CSV")
        inp["Open"] = inp["Open"].fillna(1).astype(int)
        inp["Date"] = pd.to_datetime(inp["Date"])
        inp = inp.sort_values("Date")
        if "Id" not in inp.columns:
            inp.insert(0, "Id", range(1, len(inp) + 1))
    else:
        inp = build_input(sid, p["start"], p["days"], p["promo"], p["state"], p["school"], p["sunday"])
    sdf = store_data.copy()
    if p["override"]:
        m = sdf["Store"] == sid
        sdf.loc[m, "StoreType"] = p["stype"]
        sdf.loc[m, "Assortment"] = p["assort"]
        sdf.loc[m, "CompetitionDistance"] = p["cdist"]
        sdf.loc[m, "CompetitionOpenSinceYear"] = p["cyear"]
        sdf.loc[m, "CompetitionOpenSinceMonth"] = p["cmonth"]
        sdf.loc[m, "Promo2"] = 1 if p["promo2"] == "Yes" else 0
    fc = make_prediction(inp, sdf)
    fc["Date"] = pd.to_datetime(fc["Date"])
    fig = forecast_figure(fc, sid)
    try:
        png = fig.to_image(format="png", scale=2)  # needs: pip install kaleido
    except Exception:
        png = None
    st.session_state["fc"] = dict(df=fc, sid=sid, fig=fig, png=png)


def forecast_panel():
    left, right = st.columns([1, 1.7], gap="large")
    stores = store_data["Store"].tolist()

    with left:
        st.markdown('<div class="sec-title">🧾 Enter Store Details</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        sid = c1.number_input("Store ID", 1, 1115, 101, key="f_store")
        start = c2.date_input("Start Date", date(2015, 8, 1), key="f_start")
        days = st.selectbox("Number of Days", [7, 14, 28, 42], index=3,
                            format_func=lambda d: f"{d} ({d // 7} weeks)", key="f_days")
        c1, c2 = st.columns(2)
        promo = c1.selectbox("Promo", ["Yes (all days)", "No", "Alternate weeks (Mon–Fri)"], key="f_promo")
        state = c2.selectbox("State Holiday (start day)", ["0", "a", "b", "c"], key="f_state")
        c1, c2 = st.columns(2)
        school = c1.selectbox("School Holiday", ["No (0)", "Yes (1)"], key="f_school").startswith("Yes")
        sunday = c2.checkbox("Closed on Sundays", True, key="f_sun")

        row = store_data[store_data["Store"] == sid]
        r = row.iloc[0] if not row.empty else None
        override = st.checkbox("Override store attributes", False, key="f_over")
        stype = assort = promo2 = None
        cdist, cyear, cmonth = 0, 2010, 1
        if override and r is not None:
            c1, c2 = st.columns(2)
            stype = c1.selectbox("Store Type", list("abcd"), index="abcd".index(r["StoreType"]), key="f_st")
            assort = c2.selectbox("Assortment", list("abc"), index="abc".index(r["Assortment"]), key="f_as")
            cdist = c1.number_input("Competition Distance (m)", 0, 100000,
                                    int(0 if pd.isna(r["CompetitionDistance"]) else r["CompetitionDistance"]), key="f_cd")
            promo2 = c2.selectbox("Promo2", ["No", "Yes"], index=int(r["Promo2"]), key="f_p2")
            cyear = c1.number_input("Comp. Open Since (Year)", 1900, 2030,
                                    int(2010 if pd.isna(r["CompetitionOpenSinceYear"]) else r["CompetitionOpenSinceYear"]), key="f_cy")
            cmonth = c2.number_input("Comp. Open Since (Month)", 1, 12,
                                     int(1 if pd.isna(r["CompetitionOpenSinceMonth"]) else r["CompetitionOpenSinceMonth"]), key="f_cm")
        elif r is not None:
            st.caption(f"Using store.csv → Type **{r['StoreType']}**, Assortment **{r['Assortment']}**, "
                       f"Promo2 **{int(r['Promo2'])}**")

        use_csv = False
        if uploaded_file is not None:
            use_csv = st.checkbox("Use uploaded CSV rows for this store", True, key="f_csv")
        else:
            st.caption("💡 Upload test.csv from the sidebar to forecast from your own file.")
        st.caption("Tip: models were trained on 2013–2015, so dates near Aug 2015 give the most reliable results.")
        go_btn = st.button("📈  Predict Sales", key="f_go")

        params = dict(
        sid=int(sid), start=start, days=int(days), promo=promo,
        state=state, school=school, sunday=sunday, override=override,
        stype=stype, assort=assort, cdist=cdist,
        cyear=cyear, cmonth=cmonth, promo2=promo2,
        use_csv=use_csv
    )

    if st.session_state.get("last_sid") != sid:
        st.session_state.pop("fc", None)
        st.session_state["last_sid"] = sid

    up_name = uploaded_file.name if uploaded_file is not None else None
    new_upload = st.session_state.get("last_up") != up_name
    st.session_state["last_up"] = up_name

    if models_loaded and (go_btn or new_upload or "fc" not in st.session_state):
        if sid not in stores:
            st.warning("Store ID not found in store.csv")
        else:
            with st.spinner("🤖 AI is generating forecast..."):
                try:
                    run_forecast(params)
                except Exception as e:
                    st.error(f"Prediction error: {e}")

    with right:
        st.markdown('<div class="sec-title">🔮 Forecast Results</div>', unsafe_allow_html=True)
        if "fc" not in st.session_state:
            st.info("Click **Predict Sales** to generate a forecast.")
            return
        res = st.session_state["fc"]
        fc = res["df"]
        c1, c2, c3 = st.columns(3)
        c1.markdown(card("🛒", "Predicted Total Sales", short_inr(fc['PredictedSales'].sum()), f"● Store {res['sid']}"),
                    unsafe_allow_html=True)
        c2.markdown(card("👥", "Predicted Total Customers", f"{fc['PredictedCustomers'].sum():,.0f}", "● Forecast period"),
                    unsafe_allow_html=True)
        c3.markdown(card("📅", "Open / Promo Days", f"{int(fc['Open'].sum())} / {int(fc['Promo'].sum())} Promo", "● of " + str(len(fc)) + " days"),
                    unsafe_allow_html=True)
        st.plotly_chart(res["fig"], use_container_width=True)

        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Download Predictions (CSV)", fc.to_csv(index=False).encode("utf-8"),
                           file_name=f"rossmann_store_{res['sid']}_forecast.csv", mime="text/csv")
        if res["png"]:
            d2.download_button("🖼️ Download Plot (PNG)", res["png"], file_name=f"store_{res['sid']}_plot.png", mime="image/png")
        else:
            d2.download_button("🖼️ Download Plot (HTML)", res["fig"].to_html().encode("utf-8"),
                               file_name=f"store_{res['sid']}_plot.html", mime="text/html",
                               help="Install kaleido (pip install kaleido) for PNG export")

    if "fc" in st.session_state:
        st.markdown('<div class="sec-title">📋 Prediction Results Table</div>', unsafe_allow_html=True)
        forecast_table(st.session_state["fc"]["df"], 100)


# =========================================================
# EXTRA VISUALS
# =========================================================
CITIES = [("Berlin", 52.52, 13.40), ("Hamburg", 53.55, 9.99), ("Munich", 48.14, 11.58), ("Cologne", 50.94, 6.96),
          ("Frankfurt", 50.11, 8.68), ("Stuttgart", 48.78, 9.18), ("Dusseldorf", 51.23, 6.78),
          ("Dresden", 51.05, 13.74), ("Leipzig", 51.34, 12.37), ("Hannover", 52.37, 9.73)]


def viz_data():
    """Data for the analytics charts: train.csv if present, else predictions from the uploaded CSV."""
    if train_data is not None:
        return train_data, "Source: historical data (train.csv)"
    if uploaded_file is None or not models_loaded:
        return None, ""
    try:
        uploaded_file.seek(0)
        raw = pd.read_csv(uploaded_file, dtype={"StateHoliday": str})
        req = ["Store", "DayOfWeek", "Date", "Open", "Promo", "StateHoliday", "SchoolHoliday"]
        if any(c not in raw.columns for c in req):
            return None, ""
        raw["Open"] = raw["Open"].fillna(1).astype(int)
        raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
        raw = raw.dropna(subset=["Date"])
        raw["StateHoliday"] = raw["StateHoliday"].fillna("0").astype(str).replace({"0.0": "0"})
        raw["SchoolHoliday"] = raw["SchoolHoliday"].fillna(0).astype(int)
        raw["Promo"] = raw["Promo"].fillna(0).astype(int)
        raw = raw[raw["Store"].isin(set(store_data["Store"]))]
        out = predict_cached(raw, float(RATE)).rename(columns={"PredictedSales": "Sales", "PredictedCustomers": "Customers"})
        return out, "Source: AI predictions for your uploaded CSV"
    except Exception:
        return None, ""


def store_map():
    st.markdown('<div class="sec-title">🗺️ Store Sales Map</div>', unsafe_allow_html=True)
    vd, src = viz_data()
    tot = vd.groupby("Store")["Sales"].sum().reset_index()
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(CITIES), len(tot))
    tot["lat"] = np.array([CITIES[i][1] for i in idx]) + rng.normal(0, 0.35, len(tot))
    tot["lon"] = np.array([CITIES[i][2] for i in idx]) + rng.normal(0, 0.45, len(tot))
    MapTrace = getattr(go, "Scattermap", None) or go.Scattermapbox
    map_key = "map" if hasattr(go, "Scattermap") else "mapbox"
    fig = go.Figure(MapTrace(
        lat=tot["lat"], lon=tot["lon"], mode="markers",
        marker=dict(size=8, color=tot["Sales"], colorscale="Turbo", showscale=True,
                    colorbar=dict(title="Total Sales")),
        text=[f"Store {s}<br>{fmt_inr(v)}" for s, v in zip(tot["Store"], tot["Sales"])], hoverinfo="text"))
    fig.update_layout(**{map_key: dict(style="carto-darkmatter", center=dict(lat=51.2, lon=10.3), zoom=4.8)},
                      height=380, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("store.csv has no coordinates, so locations are simulated around major German cities. " + src)


def promo_vs_sales():
    st.markdown('<div class="sec-title">🎯 Promo vs Sales Analysis</div>', unsafe_allow_html=True)
    vd, _ = viz_data()
    m = vd[vd["Open"] == 1].merge(store_data[["Store", "StoreType"]], on="Store")
    g = m.groupby(["StoreType", "Promo"])["Sales"].mean().unstack()
    fig = go.Figure()
    if 0 in g.columns:
        fig.add_bar(x=g.index, y=g[0], name="No Promo", marker_color="#3b82f6")
    if 1 in g.columns:
        fig.add_bar(x=g.index, y=g[1], name="Promo", marker_color="#a855f7")
    style_fig(fig, 380).update_layout(barmode="group", xaxis_title="Store Type", yaxis_title="Avg Daily Sales (₹)")
    st.plotly_chart(fig, use_container_width=True)


def trend_chart():
    st.markdown('<div class="sec-title">📈 Sales Trend & Seasonality</div>', unsafe_allow_html=True)
    mode = st.radio("Granularity", ["Daily", "Weekly", "Monthly"], horizontal=True, label_visibility="collapsed")
    vd, _ = viz_data()
    daily = vd[vd["Open"] == 1].groupby("Date")["Sales"].mean()
    rule = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}[mode]
    s = daily.resample(rule).mean()
    fig = go.Figure(go.Scatter(x=s.index, y=s.values, mode="lines", line=dict(color="#8b5cf6", width=3, shape="spline"),
                               fill="tozeroy", fillcolor="rgba(139,92,246,.18)", name="Avg Sales"))
    style_fig(fig, 380).update_layout(yaxis_title="Sales (₹)")
    st.plotly_chart(fig, use_container_width=True)


def need_train():
    st.info("📂 Upload test.csv from the sidebar (or put train.csv next to app.py) to unlock this chart.")


def model_performance_chart():
    st.markdown(
        '<div class="sec-title">🤖 Actual Model Performance</div>',
        unsafe_allow_html=True
    )

    names = [
        "Sales Random Forest",
        "Customer Random Forest",
        "Feature LSTM"
    ]

    rmse_values = [
        1256.89,
        86.79,
        762.53
    ]

    fig = go.Figure(
        go.Bar(
            x=names,
            y=rmse_values,
            text=[f"{v:,.2f}" for v in rmse_values],
            textposition="outside",
            marker_color=["#2563eb", "#8b5cf6", "#ec4899"],
            hovertemplate="<b>%{x}</b><br>RMSE: %{y:,.2f}<extra></extra>"
        )
    )

    style_fig(fig, 380).update_layout(
        yaxis_title="RMSE",
        xaxis_title="Model",
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)
def model_cards():
    st.markdown("""
<div class="model-grid">

<div class="model-card">
<div class="model-icon">🌲</div>
<div class="model-name">Sales Random Forest</div>
<div class="model-status">● ACTIVE</div>
<div class="model-metric">
MAE ≈ 809.63
<br>
<span style="font-size:13px;color:#94a3b8;">
RMSE ≈ 1,256.89
</span>
</div>
</div>

<div class="model-card">
<div class="model-icon">👥</div>
<div class="model-name">Customer Random Forest</div>
<div class="model-status">● ACTIVE</div>
<div class="model-metric">
MAE ≈ 56.45 customers
<br>
<span style="font-size:13px;color:#94a3b8;">
RMSE ≈ 86.79 customers
</span>
</div>
</div>

<div class="model-card">
<div class="model-icon">🧠</div>
<div class="model-name">Feature LSTM</div>
<div class="model-status">● ACTIVE</div>
<div class="model-metric">
MAE ≈ 453.62
<br>
<span style="font-size:13px;color:#94a3b8;">
RMSE ≈ 762.53
</span>
<br>
<span style="font-size:11px;color:#64748b;">
Store 1 validation
</span>
</div>
</div>

</div>
""", unsafe_allow_html=True)
# =========================================================
# PAGES
# =========================================================
def page_dashboard():
    hero("Sales Forecasting Across Multiple Retail Stores",
         "Predict store sales and customer count up to 6 weeks in advance using ML & LSTM")
    dashboard_kpis()
    forecast_panel()
    st.markdown("---")
    c1, c2, c3 = st.columns([1.2, 1, 1.2])
    has_viz = viz_data()[0] is not None
    for col, fn in ((c1, store_map), (c2, promo_vs_sales), (c3, trend_chart)):
        with col:
            if has_viz:
                fn()
            else:
                need_train()
    st.markdown("---")
    model_performance_chart()
    model_cards()


def page_single():
    hero("Single Store Forecast", "Configure one store and forecast sales & customers")
    forecast_panel()


def page_batch():
    hero("Batch Forecast (CSV)", "Upload Rossmann test.csv to forecast many stores at once")
    up = uploaded_file  # uploaded from the sidebar
    if up is None:
        st.markdown('<div class="glass"><div style="font-size:35px;">\U0001F4CA</div><h3>Welcome to the Forecast Center</h3>'
                    '<p style="color:#94a3b8;">\U0001F448 Upload your CSV from the sidebar. Required columns: Store, DayOfWeek, '
                    'Date, Open, Promo, StateHoliday, SchoolHoliday</p></div>', unsafe_allow_html=True)
        return
    up.seek(0)
    try:
        df = pd.read_csv(up, dtype={"StateHoliday": str})
    except Exception as e:
        st.error(f"CSV reading error: {e}")
        return
    req = ["Store", "DayOfWeek", "Date", "Open", "Promo", "StateHoliday", "SchoolHoliday"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        st.error(f"Missing columns: {miss}")
        return

    # ---------------- data check / cleaning ----------------
    n_rows = len(df)
    open_nan = int(df["Open"].isna().sum())
    df["Open"] = df["Open"].fillna(1).astype(int)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    bad_dates = int(df["Date"].isna().sum())
    if bad_dates:
        st.error(f"{bad_dates} rows have an invalid Date. Fix the CSV and re-upload.")
        return
    df["StateHoliday"] = df["StateHoliday"].fillna("0").astype(str).replace({"0.0": "0"})
    df["SchoolHoliday"] = df["SchoolHoliday"].fillna(0).astype(int)
    df["Promo"] = df["Promo"].fillna(0).astype(int)
    known = set(store_data["Store"])
    unknown = sorted(set(df["Store"]) - known)
    if unknown:
        df = df[df["Store"].isin(known)].copy()
    dups = int(df.duplicated(["Store", "Date"]).sum())
    dow_bad = int(((df["Date"].dt.dayofweek + 1) != df["DayOfWeek"]).sum())
    sun_open = int(((df["DayOfWeek"] == 7) & (df["Open"] == 1)).sum())
    has_sales = "Sales" in df.columns

    with st.expander("\u2705 Data check - test.csv", expanded=True):
        checks = [
            (True, f"{n_rows:,} rows - {df['Store'].nunique():,} stores - {df['Date'].min():%d %b %Y} to {df['Date'].max():%d %b %Y}"),
            (open_nan == 0, f"Open column: {open_nan} missing values" + (" (filled with 1 = open, same as Kaggle practice)" if open_nan else "")),
            (not unknown, f"Stores not in store.csv: {len(unknown)}" + (f" (rows dropped: {unknown[:8]})" if unknown else "")),
            (dups == 0, f"Duplicate Store+Date rows: {dups}"),
            (dow_bad == 0, f"DayOfWeek not matching Date: {dow_bad} rows"),
            (True, f"Open on Sundays: {sun_open:,} rows"),
        ]
        if has_sales:
            checks.append((True, "CSV contains a Sales column (looks like train data, not test data) - predictions still work"))
        for ok, text in checks:
            st.markdown(("\u2705 " if ok else "\u26a0\ufe0f ") + text)
    if df.empty:
        st.error("No valid rows left after cleaning.")
        return

    c1, c2 = st.columns([1, 2])
    scope = c1.radio("Scope", ["All stores", "One store"], horizontal=True)
    if scope == "One store":
        sid = c2.selectbox("Store", sorted(df["Store"].unique()))
        df = df[df["Store"] == sid]
    if not models_loaded:
        return

    with st.spinner(f"\U0001F916 Predicting {len(df):,} rows..."):
        out = predict_cached(df, float(RATE)).copy()
    out["Date"] = pd.to_datetime(out["Date"])
    opened = out[out["Open"] == 1]

    kpi_row([
        dict(
            icon="💰",
            label="Total Predicted Sales",
            value=float(out["PredictedSales"].sum()),
            prefix="₹ ",
            suffix="",
            decimals=0,
            color="linear-gradient(135deg,#16a34a,#4ade80)",
            delta=f"{ind(out['PredictedSales'].sum())} total"
        ),

        dict(
            icon="👥",
            label="Total Customers",
            value=float(out["PredictedCustomers"].sum()),
            prefix="",
            suffix="",
            decimals=0,
            color="linear-gradient(135deg,#7c3aed,#60a5fa)",
            delta="all selected rows"
        ),

        dict(
            icon="🏬",
            label="Stores",
            value=int(out["Store"].nunique()),
            prefix="",
            suffix="",
            decimals=0,
            color="linear-gradient(135deg,#2563eb,#38bdf8)",
            delta=f"{len(out):,} rows"
        ),

        dict(
            icon="📈",
            label="Avg Sales / Open Day",
            value=float(opened["PredictedSales"].mean()) if len(opened) else 0.0,
            prefix="₹ ",
            suffix="",
            decimals=0,
            color="linear-gradient(135deg,#db2777,#c084fc)",
            delta=f"Store {int(out['Store'].iloc[0])}" if len(out) else "No data"
        ),
    ])
    daily = out.groupby("Date")[["PredictedSales", "PredictedCustomers"]].sum().reset_index()
    st.markdown('<div class="sec-title">\U0001F4C8 Daily Forecast</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_bar(x=daily["Date"], y=daily["PredictedCustomers"], name="Customers", yaxis="y2", marker_color="rgba(217,70,239,.55)")
    fig.add_scatter(x=daily["Date"], y=daily["PredictedSales"], name="Sales (\u20b9)", line=dict(color="#22d3ee", width=3, shape="spline"),
                    fill="tozeroy", fillcolor="rgba(34,211,238,.10)")
    style_fig(fig, 380).update_layout(yaxis_title="Sales (\u20b9)", yaxis2=dict(title="Customers", overlaying="y", side="right", showgrid=False))
    st.plotly_chart(fig, use_container_width=True)

    if out["Store"].nunique() > 1:
        st.markdown('<div class="sec-title">\U0001F3C6 Top 10 Stores by Predicted Sales</div>', unsafe_allow_html=True)
        top = out.groupby("Store")["PredictedSales"].sum().nlargest(10).iloc[::-1]
        figt = go.Figure(go.Bar(y=[f"Store {i}" for i in top.index], x=top.values, orientation="h",
                                text=[short_inr(v) for v in top.values], textposition="outside",
                                marker=dict(color=top.values, colorscale="Viridis")))
        style_fig(figt, 380).update_layout(hovermode="closest", xaxis_title="Predicted Sales (\u20b9)")
        st.plotly_chart(figt, use_container_width=True)

    st.markdown('<div class="sec-title">\U0001F4CB Prediction Results</div>', unsafe_allow_html=True)
    f1, f2 = st.columns([1, 1])
    store_opts = ["All stores"] + [int(x) for x in sorted(out["Store"].unique())[:2000]]
    pick = f1.selectbox("Filter by store", store_opts, key="tbl_store")
    rows = f2.selectbox("Rows to show", [50, 100, 200, 500], index=1, key="tbl_rows")
    view = out if pick == "All stores" else out[out["Store"] == pick]
    forecast_table(view, rows)

    dl = out.rename(columns={"PredictedSales": "PredictedSales_INR"})
    dl["Date"] = dl["Date"].dt.strftime("%Y-%m-%d")
    st.download_button("\u2b07\ufe0f Download All Predictions (CSV)", dl.to_csv(index=False).encode("utf-8"),
                       file_name="rossmann_batch_forecast_INR.csv", mime="text/csv")


def page_viz():
    hero("Visualizations", "Interactive analytics across the Rossmann network")
    if viz_data()[0] is None:
        need_train()
        return
    store_map()
    c1, c2 = st.columns(2)
    with c1:
        promo_vs_sales()
    with c2:
        trend_chart()
    st.markdown('<div class="sec-title">📆 Sales by Day of Week</div>', unsafe_allow_html=True)
    vd, _ = viz_data()
    g = vd[vd["Open"] == 1].groupby("DayOfWeek")["Sales"].mean()
    fig = go.Figure(go.Bar(x=[DAY_NAMES[int(i)] for i in g.index], y=g.values,
                           marker=dict(color=g.values, colorscale="Viridis")))
    st.plotly_chart(style_fig(fig, 320), use_container_width=True)


def page_insights():
    hero("Store Insights", "Explore store attributes and history")

    sid = st.number_input(
        "Store ID",
        1,
        1115,
        1,
        key="ins_store"
    )

    row = store_data[store_data["Store"] == sid]

    if not row.empty:
        r = row.iloc[0]

        kpi_row([
            dict(
                icon="🏷️",
                label="Store Type",
                value=0,
                prefix=str(r["StoreType"]).upper(),
                suffix="",
                decimals=0,
                color="linear-gradient(135deg,#2563eb,#38bdf8)",
                delta="from store.csv"
            ),

            dict(
                icon="📦",
                label="Assortment",
                value=0,
                prefix=str(r["Assortment"]).upper(),
                suffix="",
                decimals=0,
                color="linear-gradient(135deg,#16a34a,#4ade80)",
                delta="a=basic b=extra c=extended"
            ),

            dict(
                icon="📏",
                label="Competition (m)",
                value=float(
                    0
                    if pd.isna(r["CompetitionDistance"])
                    else r["CompetitionDistance"]
                ),
                prefix="",
                suffix="",
                decimals=0,
                color="linear-gradient(135deg,#7c3aed,#60a5fa)",
                delta="nearest competitor"
            ),

            dict(
                icon="🎁",
                label="Promo2",
                value=0,
                prefix="Yes" if r["Promo2"] == 1 else "No",
                suffix="",
                decimals=0,
                color="linear-gradient(135deg,#db2777,#c084fc)",
                delta="continuing promo"
            ),
        ])

        if train_data is not None:
            h = (
                train_data[
                    (train_data["Store"] == sid)
                    & (train_data["Open"] == 1)
                ]
                .set_index("Date")["Sales"]
                .resample("W")
                .mean()
            )

            fig = go.Figure(
                go.Scatter(
                    x=h.index,
                    y=h.values,
                    line=dict(color="#22d3ee", width=3),
                    fill="tozeroy",
                    fillcolor="rgba(34,211,238,.12)"
                )
            )

            st.markdown(
                '<div class="sec-title">📈 Weekly Sales History</div>',
                unsafe_allow_html=True
            )

            st.plotly_chart(
                style_fig(fig, 340),
                use_container_width=True
            )

    c1, c2 = st.columns(2)

    with c1:
        vc = store_data["StoreType"].value_counts().sort_index()

        st.plotly_chart(
            style_fig(
                go.Figure(
                    go.Bar(
                        x=vc.index,
                        y=vc.values,
                        marker_color="#8b5cf6"
                    )
                ),
                300
            ).update_layout(title="Stores by Type"),
            use_container_width=True
        )

    with c2:
        st.plotly_chart(
            style_fig(
                go.Figure(
                    go.Histogram(
                        x=store_data["CompetitionDistance"].dropna(),
                        nbinsx=40,
                        marker_color="#22d3ee"
                    )
                ),
                300
            ).update_layout(title="Competition Distance"),
            use_container_width=True
        )
def page_models():
    hero("Model Performance", "Compare forecasting models")
    model_cards()
    model_performance_chart()


def page_about():
    hero(
        "About Rossmann AI",
        "End-to-end machine learning system for store sales & customer forecasting"
    )

    st.markdown("""
<div class="glass">

<h2>🚀 Rossmann AI Forecasting System</h2>

<p style="color:#94a3b8;font-size:16px;line-height:1.7;">
An end-to-end forecasting dashboard that predicts daily store sales and
customer counts using machine learning, feature engineering and
time-series forecasting techniques.
</p>

</div>
""", unsafe_allow_html=True)

    st.markdown("### 📊 Project Overview")

    st.markdown("""
<div class="kpi-grid">

<div class="kpi-card">
<div class="kpi-icon">🏪</div>
<div class="kpi-label">Stores</div>
<div class="kpi-value">1,115</div>
<div class="kpi-sub">Rossmann stores</div>
</div>

<div class="kpi-card">
<div class="kpi-icon">📅</div>
<div class="kpi-label">Training Data</div>
<div class="kpi-value">1M+</div>
<div class="kpi-sub">Historical records</div>
</div>

<div class="kpi-card">
<div class="kpi-icon">🔮</div>
<div class="kpi-label">Forecast Horizon</div>
<div class="kpi-value">6 Weeks</div>
<div class="kpi-sub">Future predictions</div>
</div>

<div class="kpi-card">
<div class="kpi-icon">🤖</div>
<div class="kpi-label">ML Models</div>
<div class="kpi-value">3</div>
<div class="kpi-sub">RF + RF + LSTM</div>
</div>

</div>
""", unsafe_allow_html=True)

    st.markdown("### 🧠 Models Used")

    st.markdown("""
<div class="model-grid">

<div class="model-card">
<div class="model-icon">🌲</div>
<div class="model-name">Sales Random Forest</div>
<div class="model-status">● ACTIVE</div>
<div class="model-metric">
Daily Sales Forecasting
<br>
<span style="font-size:13px;color:#94a3b8;">
MAE ≈ 809.63
</span>
</div>
</div>

<div class="model-card">
<div class="model-icon">👥</div>
<div class="model-name">Customer Random Forest</div>
<div class="model-status">● ACTIVE</div>
<div class="model-metric">
Customer Forecasting
<br>
<span style="font-size:13px;color:#94a3b8;">
MAE ≈ 56.45
</span>
</div>
</div>

<div class="model-card">
<div class="model-icon">🧠</div>
<div class="model-name">Feature LSTM</div>
<div class="model-status">● ACTIVE</div>
<div class="model-metric">
Time-Series Forecasting
<br>
<span style="font-size:13px;color:#94a3b8;">
MAE ≈ 453.62
</span>
<br>
<span style="font-size:11px;color:#64748b;">
Store 1 validation
</span>
</div>
</div>

</div>
""", unsafe_allow_html=True)

    st.markdown("### ⚙️ Key Features")

    f1, f2 = st.columns(2)

    with f1:
        st.markdown("""
<div class="glass" style="min-height:260px;">

<h3>📈 Forecasting Engine</h3>

<p style="color:#94a3b8;line-height:1.7;">
Generate machine-learning based forecasts for individual stores
or complete CSV datasets.
</p>

<div style="margin-top:18px;line-height:2.1;color:#cbd5e1;">
<div>✓ Single-store forecasting</div>
<div>✓ Batch CSV forecasting</div>
<div>✓ Sales prediction</div>
<div>✓ Customer prediction</div>
<div>✓ 6-week forecast horizon</div>
</div>

</div>
""", unsafe_allow_html=True)

    with f2:
        st.markdown("""
<div class="glass" style="min-height:260px;">

<h3>📊 Analytics & Insights</h3>

<p style="color:#94a3b8;line-height:1.7;">
Explore interactive analytics to understand sales trends,
promotions and store-level performance.
</p>

<div style="margin-top:18px;line-height:2.1;color:#cbd5e1;">
<div>✓ Sales trend analysis</div>
<div>✓ Promo vs Sales analysis</div>
<div>✓ Store-level insights</div>
<div>✓ Interactive visualizations</div>
<div>✓ Forecast CSV download</div>
</div>

</div>
""", unsafe_allow_html=True)

    st.markdown("### 🛠️ Technology Stack")

    tech = [
        ("🐍", "Python", "Core programming"),
        ("🐼", "Pandas", "Data processing"),
        ("🔢", "NumPy", "Numerical computing"),
        ("🌲", "Scikit-learn", "Machine learning"),
        ("🧠", "TensorFlow", "LSTM forecasting"),
        ("📊", "Plotly", "Interactive charts"),
        ("🎈", "Streamlit", "Web dashboard"),
    ]

    cols = st.columns(4)

    for i, (icon, name, desc) in enumerate(tech):
        with cols[i % 4]:
            st.markdown(f"""
<div class="glass" style="min-height:145px;margin-bottom:18px;text-align:center;">

<div style="font-size:36px;margin-bottom:8px;">
{icon}
</div>

<div style="font-size:18px;font-weight:700;color:#f8fafc;">
{name}
</div>

<div style="font-size:13px;color:#94a3b8;margin-top:6px;">
{desc}
</div>

</div>
""", unsafe_allow_html=True)
    st.markdown("### 🔄 ML Forecasting Pipeline")

    st.markdown("""
<div class="glass" style="padding:28px;">

<div style="
display:flex;
align-items:center;
justify-content:space-between;
gap:12px;
flex-wrap:wrap;
text-align:center;
">

<div>
<div style="font-size:32px;">📂</div>
<div style="font-weight:700;color:#f8fafc;">CSV Data</div>
<div style="font-size:12px;color:#94a3b8;">Train / Test</div>
</div>

<div style="font-size:25px;color:#8b5cf6;">→</div>

<div>
<div style="font-size:32px;">🧹</div>
<div style="font-weight:700;color:#f8fafc;">Data Cleaning</div>
<div style="font-size:12px;color:#94a3b8;">Prepare data</div>
</div>

<div style="font-size:25px;color:#8b5cf6;">→</div>

<div>
<div style="font-size:32px;">⚙️</div>
<div style="font-weight:700;color:#f8fafc;">Feature Engineering</div>
<div style="font-size:12px;color:#94a3b8;">Create ML features</div>
</div>

<div style="font-size:25px;color:#8b5cf6;">→</div>

<div>
<div style="font-size:32px;">🤖</div>
<div style="font-weight:700;color:#f8fafc;">ML Models</div>
<div style="font-size:12px;color:#94a3b8;">RF + LSTM</div>
</div>

<div style="font-size:25px;color:#8b5cf6;">→</div>

<div>
<div style="font-size:32px;">🔮</div>
<div style="font-weight:700;color:#f8fafc;">Predictions</div>
<div style="font-size:12px;color:#94a3b8;">Sales + Customers</div>
</div>

<div style="font-size:25px;color:#8b5cf6;">→</div>

<div>
<div style="font-size:32px;">📊</div>
<div style="font-weight:700;color:#f8fafc;">Dashboard</div>
<div style="font-size:12px;color:#94a3b8;">Interactive insights</div>
</div>

</div>

</div>
""", unsafe_allow_html=True)
# =========================================================
# SIDEBAR + ROUTER
# =========================================================
PAGES = {
    "🏠 Dashboard": page_dashboard,
    "🔮 Single Store Forecast": page_single,
    "📄 Batch Forecast (CSV)": page_batch,
    "📊 Visualizations": page_viz,
    "🏪 Store Insights": page_insights,
    "🤖 Model Performance": page_models,
    "ℹ️ About Project": page_about,
}

with st.sidebar:
    st.markdown("""
<div class="cube-wrap"><div class="cube"><div class="f"></div><div class="b"></div><div class="r"></div>
<div class="l"></div><div class="t"></div><div class="d"></div></div></div>
<div style="text-align:center;padding-bottom:14px;">
<div style="font-size:21px;font-weight:800;">ROSSMANN AI</div>
<div style="color:#64748b;font-size:12px;">SALES FORECAST ENGINE</div></div>""", unsafe_allow_html=True)
    page = st.radio("Go to", list(PAGES), label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### 📁 Upload Forecast Data")
    uploaded_file = st.file_uploader("Upload test.csv", type=["csv"], key="upload_csv")
    if uploaded_file is not None:
        st.caption(f"✅ {uploaded_file.name}")
    st.markdown("---")
    st.markdown("### \U0001F4B1 Currency")
    RATE = st.number_input("1 EUR = \u20b9", 50.0, 200.0, DEFAULT_EUR_INR, 0.1,
                           help="Rossmann data is in euros; all sales are shown in Indian rupees using this rate.")
    st.markdown("---")
    if models_loaded:
        st.success("🟢 AI Models Online")
        st.caption("Sales Random Forest")
        st.caption("Customer Random Forest")
    else:
        st.error("🔴 Models Offline")

if train_data is not None:
    train_data = train_data.assign(Sales=train_data["Sales"] * RATE)  # EUR -> INR for display

if not models_loaded and page in ("🏠 Dashboard", "🔮 Single Store Forecast", "📄 Batch Forecast (CSV)"):
    st.warning("Models failed to load — check the paths in the PATHS section.")
    st.stop()

PAGES[page]()