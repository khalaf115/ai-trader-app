import time
import requests
import pandas as pd
import numpy as np
import streamlit as st
import re
from datetime import datetime, timedelta
import pytz

st.set_page_config(page_title="AI Trader Pro v3", page_icon="🧠", layout="wide")

PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF", "NZD/USD"]
INTERVALS = {"1 min": "1min", "5 min": "5min", "15 min": "15min", "30 min": "30min", "1 hour": "1h"}
TF_LABELS = list(INTERVALS.keys())
TF_MAP = INTERVALS

@st.cache_data(ttl=60)
def demo_data(n=300, seed=7, tf="1min"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=n, freq=tf)
    returns = rng.normal(0.00002, 0.00045, n)
    close = 1.085 + np.cumsum(returns)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + rng.uniform(0.00003, 0.00020, n)
    low = np.minimum(open_, close) - rng.uniform(0.00003, 0.00020, n)
    volume = rng.integers(100, 1200, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)

@st.cache_data(ttl=60)
def fetch_twelvedata(symbol, interval, apikey):
    url = "https://api.twelvedata.com/time_series"
    r = requests.get(url, params={"symbol": symbol.replace("/", "/"), "interval": interval, "outputsize": 300, "apikey": apikey, "format": "JSON"}, timeout=15)
    r.raise_for_status()
    data = r.json()
    if "values" not in data: raise RuntimeError(data.get("message", "No market data returned"))
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("datetime").set_index("datetime").dropna()
    return df

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
def support_resistance(df, window=40):
    hi = df.high.tail(window).max()
    lo = df.low.tail(window).min()
    return float(hi), float(lo)
def candle_patterns(df):
    if len(df) < 2: return {}
    x = df.iloc[-1]; p = df.iloc[-2]
    body = abs(x.close - x.open); rng = max(x.high - x.low, 1e-12)
    upper = x.high - max(x.open, x.close); lower = min(x.open, x.close) - x.low
    bullish_engulf = (p.close < p.open and x.close > x.open and x.close >= p.open and x.open <= p.close)
    bearish_engulf = (p.close > p.open and x.close < x.open and x.open >= p.close and x.close <= p.open)
    hammer = (lower > body * 2 and upper < body * 0.8 and body / rng < 0.45)
    shooting = (upper > body * 2 and lower < body * 0.8 and body / rng < 0.45)
    return {"Bullish Engulfing": bullish_engulf, "Bearish Engulfing": bearish_engulf, "Hammer": hammer, "Shooting Star": shooting}

class EconomicNewsFilter:
    def __init__(self, api_key=None, manual_events=""):
        self.api_key = api_key; self.manual_events = manual_events; self.events = []
    def fetch_events(self):
        if self.api_key:
            try:
                url = "https://eodhistoricaldata.com/api/economic-events"
                params = {"api_token": self.api_key, "from": datetime.now().strftime("%Y-%m-%d"), "to": (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d"), "fmt": "json"}
                r = requests.get(url, params=params, timeout=10)
                if r.status_code == 200:
                    data = r.json()
                    self.events = []
                    for item in data:
                        dt = datetime.fromisoformat(item['date'].replace('Z', '+00:00'))
                        self.events.append({"time": dt, "name": item.get('name', ''), "impact": item.get('impact', 'low').lower()})
                    return self.events
            except: pass
        if self.manual_events.strip():
            lines = self.manual_events.strip().split('\n')
            for line in lines:
                m = re.match(r'(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2})\s+(High|Medium|Low)\s*[:|]\s*(.+)', line, re.IGNORECASE)
                if m:
                    date_str, time_str, impact, name = m.groups()
                    dt_str = f"{date_str} {time_str}:00+00:00"
                    try:
                        dt = datetime.fromisoformat(dt_str)
                        self.events.append({"time": dt, "name": name.strip(), "impact": impact.lower()})
                    except: pass
        return self.events
    def has_high_impact_soon(self, current_time, blackout_minutes=120):
        for ev in self.events:
            if ev['impact'] == 'high':
                delta = (ev['time'] - current_time).total_seconds() / 60
                if 0 <= delta <= blackout_minutes:
                    return True
        return False
    def get_upcoming_events(self, current_time, lookahead_hours=12):
        upcoming = []
        for ev in self.events:
            delta = (ev['time'] - current_time).total_seconds() / 3600
            if 0 <= delta <= lookahead_hours:
                upcoming.append(ev)
        return upcoming

class AIEngine:
    def __init__(self, df, pair, tf_label):
        self.df = df; self.pair = pair; self.tf_label = tf_label
    def analyze(self):
        df = self.df; c = df['close']
        e20, e50, e200 = ema(c, 20), ema(c, 50), ema(c, 200)
        rsi_val = rsi(c).iloc[-1]; macd_line, signal_line, hist = macd(c)
        atr_val = atr(df).iloc[-1]; res, sup = support_resistance(df)
        patterns = candle_patterns(df); price = float(c.iloc[-1])
        score = 50.0; factors = {}; reasons = []
        if price > e20.iloc[-1]: score += 8; factors['ema20'] = 1; reasons.append("Price above EMA20")
        else: score -= 8; factors['ema20'] = -1; reasons.append("Price below EMA20")
        if e20.iloc[-1] > e50.iloc[-1]: score += 10; factors['ema_cross'] = 1; reasons.append("EMA20 > EMA50")
        else: score -= 10; factors['ema_cross'] = -1; reasons.append("EMA20 < EMA50")
        if price > e200.iloc[-1]: score += 8; factors['trend'] = 1; reasons.append("Above long-term EMA")
        else: score -= 8; factors['trend'] = -1; reasons.append("Below long-term EMA")
        if 55 < rsi_val < 72: score += 8; factors['rsi'] = 1; reasons.append("RSI bullish (55-72)")
        elif 28 < rsi_val < 45: score -= 8; factors['rsi'] = -1; reasons.append("RSI bearish (28-45)")
        else: factors['rsi'] = 0
        if hist.iloc[-1] > 0: score += 10; factors['macd'] = 1; reasons.append("MACD histogram positive")
        else: score -= 10; factors['macd'] = -1; reasons.append("MACD histogram negative")
        factors['atr'] = atr_val
        dist_to_res = (res - price) / (res - sup + 1e-12); dist_to_sup = (price - sup) / (res - sup + 1e-12)
        if dist_to_res < 0.05: reasons.append("Near resistance"); factors['s_r'] = -1; score -= 6
        elif dist_to_sup < 0.05: reasons.append("Near support"); factors['s_r'] = 1; score += 6
        else: factors['s_r'] = 0
        if patterns.get('Bullish Engulfing') or patterns.get('Hammer'): score += 8; factors['candle'] = 1; reasons.append("Bullish candle pattern")
        elif patterns.get('Bearish Engulfing') or patterns.get('Shooting Star'): score -= 8; factors['candle'] = -1; reasons.append("Bearish candle pattern")
        else: factors['candle'] = 0
        factors['mtf'] = 0 # simplified
        score = float(np.clip(score, 0, 100))
        if score >= 65: signal = "CALL"; direction = "Bullish"
        elif score <= 35: signal = "PUT"; direction = "Bearish"
        else: signal = "NO TRADE"; direction = "Neutral"
        if signal != "NO TRADE":
            confidence = int(np.clip(round(abs(score - 50) * 1.6 + 50), 50, 95))
        else:
            confidence = int(np.clip(round(100 - abs(score - 50) * 2), 30, 70))
        return {"signal": signal, "confidence": confidence, "score": score, "direction": direction, "price": price, "rsi": rsi_val, "ema20": e20.iloc[-1], "ema50": e50.iloc[-1], "ema200": e200.iloc[-1], "macd_hist": hist.iloc[-1], "atr": atr_val, "resistance": res, "support": sup, "patterns": patterns, "factors": factors, "reasons": reasons}

class SignalFilter:
    def __init__(self, min_confidence=65, max_atr_ratio=0.02, min_atr_ratio=0.001, news_filter=None, blackout_minutes=120):
        self.min_confidence = min_confidence; self.max_atr_ratio = max_atr_ratio; self.min_atr_ratio = min_atr_ratio; self.news_filter = news_filter; self.blackout_minutes = blackout_minutes
    def apply(self, analysis, df):
        signal = analysis['signal']
        if signal == "NO TRADE": return "NO TRADE", "Base signal neutral"
        if analysis['confidence'] < self.min_confidence: return "NO TRADE", f"Confidence {analysis['confidence']} < {self.min_confidence}"
        price = analysis['price']; atr_val = analysis['atr']; atr_ratio = atr_val / price if price != 0 else 0
        if atr_ratio > self.max_atr_ratio: return "NO TRADE", f"Volatility too high (ATR/price={atr_ratio:.4f})"
        if atr_ratio < self.min_atr_ratio: return "NO TRADE", f"Volatility too low (ATR/price={atr_ratio:.4f})"
        if self.news_filter:
            current_time = df.index[-1]
            if self.news_filter.has_high_impact_soon(current_time, self.blackout_minutes):
                return "NO TRADE", "High-impact news event within blackout window"
        return signal, "Passed all filters"

class SignalHistory:
    def __init__(self):
        if 'signal_history' not in st.session_state: st.session_state.signal_history = []
        self.history = st.session_state.signal_history
    def add_signal(self, signal_dict, df, horizon_candles):
        last_time = df.index[-1]
        for s in self.history:
            if s['timestamp'] == last_time: return
        self.history.append({'timestamp': last_time, 'signal': signal_dict['signal'], 'confidence': signal_dict['confidence'], 'entry': signal_dict['price'], 'exit': None, 'win': None, 'horizon': horizon_candles, 'status': 'open'})
    def update_outcomes(self, df, horizon_candles):
        if len(df) < horizon_candles + 1: return
        for s in self.history:
            if s['status'] == 'open' and s.get('horizon') == horizon_candles:
                entry_time = s['timestamp']
                if entry_time in df.index:
                    entry_pos = df.index.get_loc(entry_time)
                    if len(df) - entry_pos > horizon_candles:
                        exit_time = df.index[entry_pos + horizon_candles]
                        exit_price = df.loc[exit_time, 'close']
                        s['exit'] = exit_price
                        s['win'] = (exit_price > s['entry']) if s['signal'] == 'CALL' else (exit_price < s['entry'])
                        s['status'] = 'closed'
    def get_stats(self):
        total = len(self.history); closed = [s for s in self.history if s['status'] == 'closed']; wins = sum(1 for s in closed if s['win'])
        return {'total': total, 'closed': len(closed), 'wins': wins, 'win_rate': wins / len(closed) * 100 if closed else 0}
    def clear(self): st.session_state.signal_history = []

def main():
    st.title("🧠 AI Trader Pro v3")
    st.caption("Advanced AI Engine with filtering, history, and news awareness.")
    with st.sidebar:
        st.header("⚙️ Settings")
        pair = st.selectbox("Currency Pair", PAIRS)
        tf_label = st.selectbox("Timeframe", TF_LABELS)
        data_mode = st.radio("Data Source", ["Demo", "Twelve Data Live"])
        api_key = ""
        if data_mode == "Twelve Data Live": api_key = st.text_input("Twelve Data API Key", type="password")
        st.divider()
        st.subheader("🔎 Signal Filtering")
        min_conf = st.slider("Min Confidence", 50, 90, 65, 5)
        max_atr = st.number_input("Max ATR/Price", 0.001, 0.05, 0.025, 0.001, format="%.3f")
        min_atr = st.number_input("Min ATR/Price", 0.0001, 0.01, 0.0005, 0.0001, format="%.4f")
        st.divider()
        st.subheader("🗞️ News Filter")
        news_mode = st.radio("News Source", ["Manual", "Disabled"])
        manual_news = ""
        if news_mode == "Manual": manual_news = st.text_area("Paste events (one per line):\nFormat: YYYY-MM-DD HH:MM High : Event name", height=150)
        blackout_min = st.slider("Blackout minutes before high-impact event", 30, 240, 120, 15)
        st.divider()
        st.subheader("📈 Signal History")
        horizon = st.slider("Evaluation horizon (candles)", 1, 20, 5)
        if st.button("Clear History"):
            SignalHistory().clear(); st.rerun()
    try:
        if data_mode == "Twelve Data Live" and api_key:
            df = fetch_twelvedata(pair, TF_MAP[tf_label], api_key)
        else:
            df = demo_data(seed=sum(map(ord, pair)) + len(tf_label), tf=TF_MAP[tf_label])
    except Exception as e:
        st.error(f"Data error: {e}"); df = demo_data()
    if df.empty: st.warning("No data available."); return
    news_filter = None
    if news_mode != "Disabled":
        news_filter = EconomicNewsFilter(manual_events=manual_news)
        news_filter.fetch_events()
    engine = AIEngine(df, pair, tf_label); analysis = engine.analyze()
    filt = SignalFilter(min_confidence=min_conf, max_atr_ratio=max_atr, min_atr_ratio=min_atr, news_filter=news_filter, blackout_minutes=blackout_min)
    filtered_signal, filter_reason = filt.apply(analysis, df)
    analysis['filtered_signal'] = filtered_signal; analysis['filter_reason'] = filter_reason
    hist = SignalHistory()
    if filtered_signal != "NO TRADE" and analysis['signal'] != "NO TRADE":
        hist.add_signal(analysis, df, horizon)
    hist.update_outcomes(df, horizon)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Signal", filtered_signal); col2.metric("Confidence", f"{analysis['confidence']}/100"); col3.metric("RSI", f"{analysis['rsi']:.1f}"); col4.metric("Support", f"{analysis['support']:.5f}"); col5.metric("Resistance", f"{analysis['resistance']:.5f}")
    if filtered_signal != analysis['signal']: st.warning(f"⚠️ Signal filtered: {filter_reason}")
    left, right = st.columns([2.2, 1])
    with left:
        st.subheader(f"{pair} — {tf_label}")
        chart = df[['close']].copy()
        chart['EMA20'] = ema(df.close, 20); chart['EMA50'] = ema(df.close, 50); chart['EMA200'] = ema(df.close, 200)
        st.line_chart(chart.tail(180), height=400)
        res = analysis['resistance']; sup = analysis['support']
        st.caption(f"Resistance: {res:.5f} | Support: {sup:.5f}")
    with right:
        st.subheader("📊 Analysis")
        if filtered_signal == "CALL": st.success("⬆️ CALL (Bullish)")
        elif filtered_signal == "PUT": st.error("⬇️ PUT (Bearish)")
        else: st.warning("⏸️ NO TRADE")
        st.progress(analysis['confidence'] / 100)
        st.write(f"**Direction:** {analysis['direction']}"); st.write(f"**Score:** {analysis['score']:.1f}"); st.write(f"**ATR/Price:** {analysis['atr']/analysis['price']:.4f}")
        st.write("**Factors:**")
        for f, val in analysis['factors'].items():
            if f != 'atr': st.write(f"  {f}: {'Bullish' if val>0 else 'Bearish' if val<0 else 'Neutral'}")
        st.write("**Reasons:**")
        for r in analysis['reasons']: st.write(f"• {r}")
    if news_filter and news_filter.events:
        st.divider(); st.subheader("🗞️ Upcoming News")
        current_time = df.index[-1]; upcoming = news_filter.get_upcoming_events(current_time, lookahead_hours=6)
        if upcoming:
            for ev in upcoming:
                impact_color = "🔴" if ev['impact'] == 'high' else "🟡" if ev['impact'] == 'medium' else "🟢"
                st.write(f"{impact_color} {ev['time'].strftime('%Y-%m-%d %H:%M')} - {ev['name']} ({ev['impact']})")
        else: st.caption("No upcoming events in next 6 hours.")
    st.divider(); st.subheader("📈 Signal History & Performance")
    stats = hist.get_stats()
    cola, colb, colc = st.columns(3)
    cola.metric("Total Signals", stats['total']); colb.metric("Closed", stats['closed']); colc.metric("Win Rate", f"{stats['win_rate']:.1f}%" if stats['closed'] > 0 else "N/A")
    if stats['closed'] > 0:
        closed_trades = [s for s in hist.history if s['status'] == 'closed']
        df_hist = pd.DataFrame(closed_trades)
        df_hist['timestamp'] = pd.to_datetime(df_hist['timestamp'])
        df_hist = df_hist.sort_values('timestamp')
        df_hist['win_int'] = df_hist['win'].astype(int)
        df_hist['rolling_win'] = df_hist['win_int'].rolling(20, min_periods=1).mean() * 100
        st.line_chart(df_hist.set_index('timestamp')[['rolling_win']], height=200)
    if hist.history: st.dataframe(pd.DataFrame(hist.history).tail(20), use_container_width=True)
    st.divider(); st.subheader("🔭 Multi-Timeframe Quick View")
    mt_data = {}
    for tf in ["M5", "M15", "H1"]:
        try:
            if data_mode == "Twelve Data Live" and api_key: df_tf = fetch_twelvedata(pair, TF_MAP[tf], api_key)
            else: df_tf = demo_data(seed=sum(map(ord, pair)) + len(tf), tf=TF_MAP[tf])
            if len(df_tf) > 50:
                eng = AIEngine(df_tf, pair, tf); res = eng.analyze()
                mt_data[tf] = {"Signal": res['signal'], "Confidence": res['confidence'], "Score": round(res['score'], 1), "RSI": round(res['rsi'], 1), "Direction": res['direction']}
        except: mt_data[tf] = {"Signal": "Error", "Confidence": 0}
    st.dataframe(pd.DataFrame(mt_data).T, use_container_width=True)
    st.info("⚠️ **Disclaimer:** This tool is for educational purposes only. Past performance does not guarantee future results. Always test thoroughly before real trading.")

if __name__ == "__main__":
    main()
