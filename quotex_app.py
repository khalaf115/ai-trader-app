import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta

# إعدادات الصفحة (واجهة داكنة تشبه كوكتكس)
st.set_page_config(page_title="Quotex AI Pro v4", page_icon="🔥", layout="wide")

# إخفاء القائمة الافتراضية وإضافة خلفية داكنة
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: white;
    }
    .css-18e3th9 {
        padding-top: 0rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-weight: bold;
        font-size: 20px;
    }
    .call-btn {
        background-color: #00b894 !important;
        color: white !important;
    }
    .put-btn {
        background-color: #e17055 !important;
        color: white !important;
    }
    .big-number {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
    }
    .payout-box {
        background-color: #2d3436;
        padding: 15px;
        border-radius: 15px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ========== الإعدادات الأساسية ==========
PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF", "NZD/USD", "BTC/USD"]
TIMEFRAMES = {"1 Min": "1min", "5 Min": "5min", "15 Min": "15min", "30 Min": "30min", "1 H": "1h"}
TIMEFRAMES_SHORT = {"M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min", "H1": "1h"}

# ========== جلب البيانات ==========
@st.cache_data(ttl=30)
def fetch_data(pair, interval, mode="Demo", api_key=""):
    try:
        if mode == "Live" and api_key:
            url = "https://api.twelvedata.com/time_series"
            r = requests.get(url, params={"symbol": pair, "interval": interval, "outputsize": 500, "apikey": api_key}, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if "values" in data:
                    df = pd.DataFrame(data["values"])
                    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
                    for c in ["open", "high", "low", "close"]:
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                    return df.sort_values("datetime").set_index("datetime").dropna()
    except: pass
    
    rng = np.random.default_rng(seed=sum(map(ord, pair)))
    n = 300
    idx = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=n, freq=interval)
    returns = rng.normal(0.0001, 0.0005, n) + np.sin(np.linspace(0, 12, n)) * 0.0002
    close = 1.085 + np.cumsum(returns)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + rng.uniform(0.0001, 0.0005, n)
    low = np.minimum(open_, close) - rng.uniform(0.0001, 0.0005, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)

# ========== المؤشرات الفنية ==========
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    down = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - (100/(1+rs))
def macd(s):
    m = ema(s, 12) - ema(s, 26)
    sig = ema(m, 9)
    return m, sig, m - sig
def atr(df, n=14):
    pc = df.close.shift(1)
    tr = pd.concat([(df.high - df.low), (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

# ========== المحرك الرئيسي ==========
def analyze_quotex(df):
    close = df['close']; high = df['high']; low = df['low']
    price = close.iloc[-1]
    e20 = ema(close, 20); e50 = ema(close, 50); e200 = ema(close, 200)
    rsi_val = rsi(close).iloc[-1]
    macd_l, macd_s, hist = macd(close)
    atr_val = atr(df).iloc[-1]
    
    score = 50.0; reasons = []
    if price > e200.iloc[-1]: score += 12; reasons.append("✅ فوق EMA200 (اتجاه صاعد)")
    else: score -= 12; reasons.append("❌ تحت EMA200 (اتجاه هابط)")
    if e20.iloc[-1] > e50.iloc[-1]: score += 10; reasons.append("✅ EMA20 > EMA50 (تقاطع صاعد)")
    else: score -= 10; reasons.append("❌ EMA20 < EMA50 (تقاطع هابط)")
    if 55 < rsi_val < 75: score += 10; reasons.append(f"✅ RSI {rsi_val:.1f} (منطقة شراء)")
    elif 25 < rsi_val < 45: score -= 10; reasons.append(f"❌ RSI {rsi_val:.1f} (منطقة بيع)")
    elif rsi_val >= 75: score -= 5; reasons.append(f"⚠️ RSI {rsi_val:.1f} (تشبع شرائي)")
    elif rsi_val <= 25: score += 5; reasons.append(f"⚠️ RSI {rsi_val:.1f} (تشبع بيعي)")
    if hist.iloc[-1] > 0: score += 10; reasons.append("✅ MACD موجب (زخم صاعد)")
    else: score -= 10; reasons.append("❌ MACD سالب (زخم هابط)")
    
    atr_pct = (atr_val / price) * 100
    if atr_pct > 0.15: expiry = "5 دقائق"
    elif atr_pct > 0.08: expiry = "2 دقائق"
    else: expiry = "1 دقيقة"
    reasons.append(f"📊 التقلب {atr_pct:.2f}% → وقت الانتهاء: {expiry}")
    
    score = np.clip(score, 0, 100)
    if score >= 65:
        signal = "CALL ⬆️"; direction = "صعود"; confidence = int(50 + (score - 50) * 1.2)
    elif score <= 35:
        signal = "PUT ⬇️"; direction = "هبوط"; confidence = int(50 + (50 - score) * 1.2)
    else:
        signal = "انتظار ⏸️"; direction = "محايد"; confidence = int(30 + (score / 100) * 30)
    confidence = np.clip(confidence, 30, 95)
    
    return {
        "signal": signal, "confidence": confidence, "score": int(score),
        "price": price, "rsi": rsi_val, "macd": hist.iloc[-1], "atr_pct": atr_pct,
        "expiry": expiry, "direction": direction, "reasons": reasons,
        "ema20": e20.iloc[-1], "ema50": e50.iloc[-1], "ema200": e200.iloc[-1],
        "support": low.tail(40).min(), "resistance": high.tail(40).max()
    }

# ========== واجهة المستخدم ==========
st.title("🔥 Quotex AI Pro v4")
st.caption("تحليل الخيارات الثنائية (CALL/PUT) مع تحديد زمن الانتهاء والعائد المتوقع")

with st.sidebar:
    st.header("⚙️ الإعدادات")
    pair = st.selectbox("الزوج", PAIRS)
    tf = st.selectbox("الفريم الزمني", list(TIMEFRAMES.keys()))
    mode = st.radio("مصدر البيانات", ["Demo", "Live"])
    api_key = st.text_input("API Key (اختياري)", type="password") if mode == "Live" else ""
    st.divider()
    st.subheader("💰 إدارة المخاطر")
    balance = st.number_input("رصيد الحساب ($)", value=10000.0, step=100.0)
    risk_percent = st.slider("نسبة المخاطرة %", 0.5, 5.0, 1.0, 0.5)
    payout_percent = st.slider("نسبة العائد المتوقعة % (Payout)", 70, 95, 79, 1)

df = fetch_data(pair, TIMEFRAMES[tf], mode, api_key)
if df.empty:
    st.error("تعذر جلب البيانات، استخدم Demo")
    df = fetch_data(pair, TIMEFRAMES[tf], "Demo", "")

analysis = analyze_quotex(df)
price = analysis['price']; resistance = analysis['resistance']; support = analysis['support']

col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
with col1:
    st.markdown(f"<div class='big-number'>{price:.5f}</div>", unsafe_allow_html=True)
    st.caption(f"{pair} | {tf}")
with col2: st.metric("الدعم", f"{support:.5f}")
with col3: st.metric("المقاومة", f"{resistance:.5f}")
with col4: st.metric("التقلب (ATR%)", f"{analysis['atr_pct']:.2f}%")

st.markdown("---")
c1, c2, c3 = st.columns([1, 1.5, 1])
with c1:
    st.markdown(f"**الإشارة:**")
    if "CALL" in analysis['signal']: st.success(f"### {analysis['signal']}")
    elif "PUT" in analysis['signal']: st.error(f"### {analysis['signal']}")
    else: st.warning(f"### {analysis['signal']}")
    st.metric("الثقة", f"{analysis['confidence']}%")
    st.metric("الدرجة (Score)", f"{analysis['score']}/100")
with c2:
    st.markdown("<div class='payout-box'>", unsafe_allow_html=True)
    st.markdown("### 💰 العائد المتوقع")
    risk_amount = balance * (risk_percent / 100)
    payout = risk_amount * (payout_percent / 100)
    col_a, col_b = st.columns(2)
    with col_a: st.metric("المخاطرة", f"${risk_amount:.2f}")
    with col_b: st.metric("العائد (Payout)", f"${payout:.2f}", delta=f"{payout_percent}%")
    st.markdown("</div>", unsafe_allow_html=True)
with c3:
    st.markdown(f"**⏱️ وقت الانتهاء المقترح:**")
    st.info(f"### {analysis['expiry']}")
    st.markdown(f"**الاتجاه:** {analysis['direction']}")
    st.progress(analysis['confidence'] / 100)

st.markdown("---")
st.subheader("🔭 تحليل الأطر الزمنية المتعددة (Multi-Timeframe)")
mt_data = {}
for tf_short in ["M5", "M15", "H1"]:
    try:
        df_tf = fetch_data(pair, TIMEFRAMES_SHORT[tf_short], mode, api_key)
        if len(df_tf) > 30:
            res = analyze_quotex(df_tf)
            mt_data[tf_short] = {"الإشارة": res['signal'], "الثقة": res['confidence'], "RSI": round(res['rsi'], 1), "الاتجاه": res['direction']}
        else: mt_data[tf_short] = {"الإشارة": "بيانات غير كافية", "الثقة": 0}
    except: mt_data[tf_short] = {"الإشارة": "خطأ", "الثقة": 0}
st.dataframe(pd.DataFrame(mt_data).T, use_container_width=True)

st.markdown("---")
st.subheader("🧠 أسباب قرار التحليل")
for r in analysis['reasons']: st.write(f"- {r}")

st.markdown("---")
st.subheader("📈 الرسم البياني مع المتوسطات المتحركة")
chart_df = df[['close']].tail(200).copy()
chart_df['EMA20'] = ema(df.close, 20)
chart_df['EMA50'] = ema(df.close, 50)
chart_df['EMA200'] = ema(df.close, 200)
st.line_chart(chart_df, height=300)

st.info("⚠️ **تنويه:** هذه الأداة للتحليل والتعليم فقط. لا ننفذ صفقات آلياً على كوكتكس. استخدمها كمساعدة وليس بديلاً عن خبرتك الشخصية.")
