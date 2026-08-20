import os
import time
import math
import re
import requests
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import pytz

st.set_page_config(page_title="AI Trader Pro v3", page_icon="🧠", layout="wide")

# ---------- CONSTANTS ----------
PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF", "NZD/USD"]
INTERVALS = {"1 min": "1min", "5 min": "5min", "15 min": "15min", "30 min": "30min", "1 hour": "1h"}
TF_LABELS = list(INTERVALS.keys())
TF_MAP = INTERVALS

# ---------- DATA HELPERS ----------
@st.cache_data(ttl=60)
def demo_data(n=300, seed=7, tf="1min"):
    """Synthetic data for demo."""
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
    r = requests.get(url, params={
        "symbol": symbol.replace("/", "/"),
        "interval": interval,
        "outputsize": 300,
        "apikey": apikey,
        "format": "JSON"
    }, timeout=15)
    r.raise_for_status()
    data = r.json()
    if "values" not in data:
        raise RuntimeError(data.get("message", "No market data returned"))
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("datetime").set_index("datetime").dropna()
    return df

# ---------- TECHNICAL INDICATORS ----------
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
    """Detect last candle patterns."""
    if len(df) < 2:
        return {}
    x = df.iloc[-1]
    p = df.iloc[-2]
    body = abs(x.close - x.open)
    rng = max(x.high - x.low, 1e-12)
    upper = x.high - max(x.open, x.close)
    lower = min(x.open, x.close) - x.low
    bullish_engulf = (p.close < p.open and x.close > x.open and x.close >= p.open and x.open <= p.close)
    bearish_engulf = (p.close > p.open and x.close < x.open and x.open >= p.close and x.close <= p.open)
    hammer = (lower > body * 2 and upper < body * 0.8 and body / rng < 0.45)
    shooting = (upper > body * 2 and lower < body * 0.8 and body / rng < 0.45)
    return {
        "Bullish Engulfing": bullish_engulf,
        "Bearish Engulfing": bearish_engulf,
        "Hammer": hammer,
        "Shooting Star": shooting
    }

# ---------- ECONOMIC NEWS FILTER ----------
class EconomicNewsFilter:
    def __init__(self, api_key=None, manual_events=""):
        self.api_key = api_key
        self.manual_events = manual_events
        self.events = []

    def fetch_events(self):
        """Fetch events from API or parse manual input."""
        if self.api_key:
            # Try EOD Historical Data API (free)
            try:
                url = "https://eodhistoricaldata.com/api/economic-events"
                params = {
                    "api_token": self.api_key,
                    "from": datetime.now().strftime("%Y-%m-%d"),
                    "to": (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d"),
                    "fmt": "json"
                }
                r = requests.get(url, params=params, timeout=10)
                if r.status_code == 200:
                    data = r.json()
                    # Parse events: each has 'date', 'name', 'impact' (high/medium/low)
                    self.events = []
                    for item in data:
                        dt = datetime.fromisoformat(item['date'].replace('Z', '+00:00'))
                        impact = item.get('impact', 'low').lower()
                        self.events.append({
                            "time": dt,
                            "name": item.get('name', ''),
                            "impact": impact
                        })
                    return self.events
            except Exception as e:
                st.warning(f"Could not fetch news via API: {e}. Using manual input if provided.")
        # Manual parsing
        if self.manual_events.strip():
            lines = self.manual_events.strip().split('\n')
            for line in lines:
                # Expected format: "2025-03-15 14:30 High : FOMC Statement"
                # Try to parse with regex
                m = re.match(r'(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2})\s+(High|Medium|Low)\s*[:|]\s*(.+)', line, re.IGNORECASE)
                if m:
                    date_str, time_str, impact, name = m.groups()
                    dt_str = f"{date_str} {time_str}:00+00:00"
                    try:
                        dt = datetime.fromisoformat(dt_str)
                        self.events.append({
                            "time": dt,
                            "name": name.strip(),
                            "impact": impact.lower()
                        })
                    except:
                        pass
        return self.events

    def has_high_impact_soon(self, current_time, blackout_minutes=120):
        """Check if any high-impact event occurs within blackout minutes."""
        for ev in self.events:
            if ev['impact'] == 'high':
                delta = (ev['time'] - current_time).total_seconds() / 60
                if 0 <= delta <= blackout_minutes:
                    return True
        return False

    def get_upcoming_events(self, current_time, lookahead_hours=12):
        """Return list of events within lookahead."""
        upcoming = []
        for ev in self.events:
            delta = (ev['time'] - current_time).total_seconds() / 3600
            if 0 <= delta <= lookahead_hours:
                upcoming.append(ev)
        return upcoming

# ---------- AI ENGINE ----------
class AIEngine:
    def __init__(self, df, pair, tf_label):
        self.df = df
        self.pair = pair
        self.tf_label = tf_label
        self.raw = None

    def analyze(self):
        df = self.df
        c = df['close']
        # Indicators
        e20 = ema(c, 20)
        e50 = ema(c, 50)
        e200 = ema(c, 200)
        rsi_val = rsi(c).iloc[-1]
        macd_line, signal_line, hist = macd(c)
        atr_val = atr(df).iloc[-1]
        res, sup = support_resistance(df)
        patterns = candle_patterns(df)
        price = float(c.iloc[-1])

        # Factor scoring
        score = 50.0
        factors = {}
        reasons = []

        # 1. Price vs EMAs
        if price > e20.iloc[-1]:
            score += 8
            factors['ema20'] = 1
            reasons.append("Price above EMA20")
        else:
            score -= 8
            factors['ema20'] = -1
            reasons.append("Price below EMA20")

        if e20.iloc[-1] > e50.iloc[-1]:
            score += 10
            factors['ema_cross'] = 1
            reasons.append("EMA20 > EMA50")
        else:
            score -= 10
            factors['ema_cross'] = -1
            reasons.append("EMA20 < EMA50")

        if price > e200.iloc[-1]:
            score += 8
            factors['trend'] = 1
            reasons.append("Above long-term EMA")
        else:
            score -= 8
            factors['trend'] = -1
            reasons.append("Below long-term EMA")

        # 2. RSI
        if 55 < rsi_val < 72:
            score += 8
            factors['rsi'] = 1
            reasons.append("RSI bullish (55-72)")
        elif 28 < rsi_val < 45:
            score -= 8
            factors['rsi'] = -1
            reasons.append("RSI bearish (28-45)")
        else:
            factors['rsi'] = 0

        # 3. MACD
        if hist.iloc[-1] > 0:
            score += 10
            factors['macd'] = 1
            reasons.append("MACD histogram positive")
        else:
            score -= 10
            factors['macd'] = -1
            reasons.append("MACD histogram negative")

        # 4. Volatility (ATR) - if too high or low, reduce confidence
        avg_atr = atr_val
        factors['atr'] = avg_atr

        # 5. Support/Resistance proximity
        dist_to_res = (res - price) / (res - sup + 1e-12)
        dist_to_sup = (price - sup) / (res - sup + 1e-12)
        if dist_to_res < 0.05:
            reasons.append("Near resistance")
            factors['s_r'] = -1
            score -= 6
        elif dist_to_sup < 0.05:
            reasons.append("Near support")
            factors['s_r'] = 1
            score += 6
        else:
            factors['s_r'] = 0

        # 6. Candlestick patterns
        if patterns.get('Bullish Engulfing') or patterns.get('Hammer'):
            score += 8
            factors['candle'] = 1
            reasons.append("Bullish candle pattern")
        elif patterns.get('Bearish Engulfing') or patterns.get('Shooting Star'):
            score -= 8
            factors['candle'] = -1
            reasons.append("Bearish candle pattern")
        else:
            factors['candle'] = 0

        # 7. Multi-timeframe consensus (if we have MT data)
        mt_signal = self._multi_tf_consensus()
        if mt_signal == 1:
            score += 12
            reasons.append("MTF bullish consensus")
            factors['mtf'] = 1
        elif mt_signal == -1:
            score -= 12
            reasons.append("MTF bearish consensus")
            factors['mtf'] = -1
        else:
            factors['mtf'] = 0

        # Clamp score
        score = float(np.clip(score, 0, 100))

        # Determine signal
        if score >= 65:
            signal = "CALL"
            direction = "Bullish"
        elif score <= 35:
            signal = "PUT"
            direction = "Bearish"
        else:
            signal = "NO TRADE"
            direction = "Neutral"

        # Confidence (conservative)
        if signal != "NO TRADE":
            confidence = int(round(abs(score - 50) * 1.6 + 50))
            confidence = int(np.clip(confidence, 50, 95))
        else:
            confidence = int(round(100 - abs(score - 50) * 2))
            confidence = int(np.clip(confidence, 30, 70))

        self.raw = {
            "signal": signal,
            "confidence": confidence,
            "score": score,
            "direction": direction,
            "price": price,
            "rsi": rsi_val,
            "ema20": e20.iloc[-1],
            "ema50": e50.iloc[-1],
            "ema200": e200.iloc[-1],
            "macd_hist": hist.iloc[-1],
            "atr": avg_atr,
            "resistance": res,
            "support": sup,
            "patterns": patterns,
            "factors": factors,
            "reasons": reasons,
            "mtf_signal": mt_signal
        }
        return self.raw

    def _multi_tf_consensus(self):
        """Check higher timeframes for agreement (simplified)."""
        # For demo, we'll just fetch data for M5, M15, H1 and run quick analysis
        # But we can use the existing multi_tf function from v2
        # To avoid duplicate code, we'll call a helper.
        # We'll implement a lightweight version: compare current price to EMAs on higher TFs.
        # This is a placeholder for demonstration.
        # In full version, we'd fetch data for each TF.
        # For now, we'll return 0 (neutral) if we can't get data.
        try:
            # Use the same demo/live functions for other TFs
            tf_labels = ["M5", "M15", "H1"]
            signals = []
            for tf in tf_labels:
                # We'll use a separate function to fetch data for that TF
                # For simplicity, we'll use the existing demo_data with different seed.
                # In live mode, we'd fetch from API.
                # We'll implement a quick fetch via Twelve Data if live, else demo.
                df_tf = get_data_for_tf(self.pair, tf, self.df.index[-1])  # dummy
                if df_tf is not None and len(df_tf) > 50:
                    c = df_tf['close']
                    e20 = ema(c, 20)
                    e200 = ema(c, 200)
                    if c.iloc[-1] > e20.iloc[-1] and c.iloc[-1] > e200.iloc[-1]:
                        signals.append(1)
                    elif c.iloc[-1] < e20.iloc[-1] and c.iloc[-1] < e200.iloc[-1]:
                        signals.append(-1)
                    else:
                        signals.append(0)
            if len(signals) >= 2:
                avg = np.mean(signals)
                if avg > 0.5:
                    return 1
                elif avg < -0.5:
                    return -1
        except:
            pass
        return 0

# Helper to get data for different TF (used by MTF)
def get_data_for_tf(pair, tf_label, end_time):
    """Fetch data for a specific timeframe. Uses session state for mode."""
    # We'll rely on the main app's mode and api_key.
    # For simplicity, we'll return None if not live, else fetch.
    # Since we cannot easily pass mode, we'll implement a cache in session.
    # In practice, we'll call the main fetch function.
    # But to avoid complexity, we'll use demo data for MTF in demo mode.
    if st.session_state.get('data_mode') == 'Live' and st.session_state.get('api_key'):
        try:
            return fetch_twelvedata(pair, TF_MAP[tf_label], st.session_state['api_key'])
        except:
            return None
    else:
        # Demo MTF
        return demo_data(seed=sum(map(ord, pair)) + len(tf_label), tf=TF_MAP[tf_label])

# ---------- SIGNAL FILTER ----------
class SignalFilter:
    def __init__(self, min_confidence=65, max_atr_ratio=0.02, min_atr_ratio=0.001,
                 news_filter=None, blackout_minutes=120, proximity_threshold=0.05):
        self.min_confidence = min_confidence
        self.max_atr_ratio = max_atr_ratio
        self.min_atr_ratio = min_atr_ratio
        self.news_filter = news_filter
        self.blackout_minutes = blackout_minutes
        self.proximity_threshold = proximity_threshold

    def apply(self, analysis, df):
        """Return (filtered_signal, reason)."""
        signal = analysis['signal']
        if signal == "NO TRADE":
            return "NO TRADE", "Base signal neutral"

        # 1. Confidence
        if analysis['confidence'] < self.min_confidence:
            return "NO TRADE", f"Confidence {analysis['confidence']} < {self.min_confidence}"

        # 2. Volatility (ATR relative to price)
        price = analysis['price']
        atr = analysis['atr']
        atr_ratio = atr / price if price != 0 else 0
        if atr_ratio > self.max_atr_ratio:
            return "NO TRADE", f"Volatility too high (ATR/price={atr_ratio:.4f})"
        if atr_ratio < self.min_atr_ratio:
            return "NO TRADE", f"Volatility too low (ATR/price={atr_ratio:.4f})"

        # 3. Support/Resistance proximity already factored, but we can add extra filter
        # We already have proximity in analysis factors, but we can check again
        # We'll just ensure we are not too close to S/R if we're trading against it.
        # Actually, the AI engine already includes S/R in scoring, so we might skip.

        # 4. News filter
        if self.news_filter:
            current_time = df.index[-1]  # last candle time
            if self.news_filter.has_high_impact_soon(current_time, self.blackout_minutes):
                return "NO TRADE", "High-impact news event within blackout window"

        # All passed
        return signal, "Passed all filters"

# ---------- SIGNAL HISTORY ----------
class SignalHistory:
    def __init__(self):
        if 'signal_history' not in st.session_state:
            st.session_state.signal_history = []
        self.history = st.session_state.signal_history

    def add_signal(self, signal_dict, df, horizon_candles):
        """Add a new signal to history with entry price and timestamp."""
        # Avoid duplicates (same timestamp)
        last_time = df.index[-1]
        # Check if we already have a signal at this time
        for s in self.history:
            if s['timestamp'] == last_time:
                return  # duplicate
        entry = signal_dict['price']
        direction = signal_dict['signal']  # CALL or PUT
        confidence = signal_dict['confidence']
        self.history.append({
            'timestamp': last_time,
            'signal': direction,
            'confidence': confidence,
            'entry': entry,
            'exit': None,
            'win': None,
            'horizon': horizon_candles,
            'status': 'open'
        })

    def update_outcomes(self, df, horizon_candles):
        """Check open trades and close them after horizon."""
        if len(df) < horizon_candles + 1:
            return
        # Use last candle as current time
        current_idx = df.index[-1]
        for s in self.history:
            if s['status'] == 'open' and s.get('horizon') == horizon_candles:
                # Find the index of entry
                entry_time = s['timestamp']
                if entry_time in df.index:
                    entry_pos = df.index.get_loc(entry_time)
                    if len(df) - entry_pos > horizon_candles:
                        exit_time = df.index[entry_pos + horizon_candles]
                        exit_price = df.loc[exit_time, 'close']
                        s['exit'] = exit_price
                        if s['signal'] == 'CALL':
                            s['win'] = exit_price > s['entry']
                        else:
                            s['win'] = exit_price < s['entry']
                        s['status'] = 'closed'

    def get_stats(self):
        total = len(self.history)
        closed = [s for s in self.history if s['status'] == 'closed']
        wins = sum(1 for s in closed if s['win'])
        win_rate = wins / len(closed) * 100 if closed else 0
        return {
            'total': total,
            'closed': len(closed),
            'wins': wins,
            'win_rate': win_rate
        }

    def clear(self):
        st.session_state.signal_history = []

# ---------- MAIN APP ----------
def main():
    st.title("🧠 AI Trader Pro v3")
    st.caption("Advanced AI Engine with filtering, history, and news awareness.")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        pair = st.selectbox("Currency Pair", PAIRS)
        tf_label = st.selectbox("Timeframe", TF_LABELS)
        data_mode = st.radio("Data Source", ["Demo", "Twelve Data Live"])
        api_key = ""
        if data_mode == "Twelve Data Live":
            api_key = st.text_input("Twelve Data API Key", type="password")
            if not api_key:
                st.warning("Enter API key for live data, else Demo will be used.")
        st.divider()

        st.subheader("🔎 Signal Filtering")
        min_conf = st.slider("Min Confidence", 50, 90, 65, 5)
        max_atr = st.number_input("Max ATR/Price", 0.001, 0.05, 0.025, 0.001, format="%.3f")
        min_atr = st.number_input("Min ATR/Price", 0.0001, 0.01, 0.0005, 0.0001, format="%.4f")
        st.divider()

        st.subheader("🗞️ News Filter")
        news_mode = st.radio("News Source", ["Manual", "EOD API", "Disabled"])
        manual_news = ""
        eod_key = ""
        if news_mode == "Manual":
            manual_news = st.text_area(
                "Paste events (one per line):\nFormat: YYYY-MM-DD HH:MM High : Event name",
                height=150,
                help="Example: 2025-03-20 14:30 High : FOMC Press Conference"
            )
        elif news_mode == "EOD API":
            eod_key = st.text_input("EOD Historical Data API Key", type="password")
        blackout_min = st.slider("Blackout minutes before high-impact event", 30, 240, 120, 15)
        st.divider()

        st.subheader("📈 Signal History")
        horizon = st.slider("Evaluation horizon (candles)", 1, 20, 5)
        if st.button("Clear History"):
            SignalHistory().clear()
            st.rerun()

    # Store settings in session for MTF helper
    st.session_state['data_mode'] = data_mode
    st.session_state['api_key'] = api_key

    # Fetch data
    try:
        if data_mode == "Twelve Data Live" and api_key:
            df = fetch_twelvedata(pair, TF_MAP[tf_label], api_key)
            st.success("Live data loaded")
        else:
            df = demo_data(seed=sum(map(ord, pair)) + len(tf_label), tf=TF_MAP[tf_label])
            st.info("Demo data (synthetic)")
    except Exception as e:
        st.error(f"Data error: {e}")
        df = demo_data()

    if df.empty:
        st.warning("No data available.")
        return

    # Initialize News Filter
    news_filter = None
    if news_mode != "Disabled":
        if news_mode == "Manual":
            news_filter = EconomicNewsFilter(manual_events=manual_news)
        else:
            news_filter = EconomicNewsFilter(api_key=eod_key)
        news_filter.fetch_events()

    # Run AI Engine
    engine = AIEngine(df, pair, tf_label)
    analysis = engine.analyze()

    # Apply Signal Filter
    filt = SignalFilter(
        min_confidence=min_conf,
        max_atr_ratio=max_atr,
        min_atr_ratio=min_atr,
        news_filter=news_filter,
        blackout_minutes=blackout_min
    )
    filtered_signal, filter_reason = filt.apply(analysis, df)

    # Update analysis with filtered signal
    analysis['filtered_signal'] = filtered_signal
    analysis['filter_reason'] = filter_reason

    # Signal History
    hist = SignalHistory()
    if filtered_signal != "NO TRADE" and analysis['signal'] != "NO TRADE":
        hist.add_signal(analysis, df, horizon)

    # Update outcomes for all trades
    hist.update_outcomes(df, horizon)

    # Display
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Signal", filtered_signal)
    col2.metric("Confidence", f"{analysis['confidence']}/100")
    col3.metric("RSI", f"{analysis['rsi']:.1f}")
    col4.metric("Support", f"{analysis['support']:.5f}")
    col5.metric("Resistance", f"{analysis['resistance']:.5f}")

    if filtered_signal != analysis['signal']:
        st.warning(f"⚠️ Signal filtered: {filter_reason}")

    left, right = st.columns([2.2, 1])
    with left:
        st.subheader(f"{pair} — {tf_label}")
        chart = df[['close']].copy()
        chart['EMA20'] = ema(df.close, 20)
        chart['EMA50'] = ema(df.close, 50)
        chart['EMA200'] = ema(df.close, 200)
        st.line_chart(chart.tail(180), height=400)

        # Show support/resistance lines if we have them
        # Not directly in line chart, but can annotate with text
        res = analysis['resistance']
        sup = analysis['support']
        st.caption(f"Resistance: {res:.5f} | Support: {sup:.5f}")

    with right:
        st.subheader("📊 Analysis")
        if filtered_signal == "CALL":
            st.success("⬆️ CALL (Bullish)")
        elif filtered_signal == "PUT":
            st.error("⬇️ PUT (Bearish)")
        else:
            st.warning("⏸️ NO TRADE")
        st.progress(analysis['confidence'] / 100)
        st.write(f"**Direction:** {analysis['direction']}")
        st.write(f"**Score:** {analysis['score']:.1f}")
        st.write(f"**ATR/Price:** {analysis['atr']/analysis['price']:.4f}")
        st.write("**Factors:**")
        for f, val in analysis['factors'].items():
            if f != 'atr':
                st.write(f"  {f}: {'Bullish' if val>0 else 'Bearish' if val<0 else 'Neutral'}")
        st.write("**Reasons:**")
        for r in analysis['reasons']:
            st.write(f"• {r}")

    # News upcoming events display
    if news_filter and news_filter.events:
        st.divider()
        st.subheader("🗞️ Upcoming News")
        current_time = df.index[-1]
        upcoming = news_filter.get_upcoming_events(current_time, lookahead_hours=6)
        if upcoming:
            for ev in upcoming:
                impact_color = "🔴" if ev['impact'] == 'high' else "🟡" if ev['impact'] == 'medium' else "🟢"
                st.write(f"{impact_color} {ev['time'].strftime('%Y-%m-%d %H:%M')} - {ev['name']} ({ev['impact']})")
        else:
            st.caption("No upcoming events in next 6 hours.")

    # Signal History Stats
    st.divider()
    st.subheader("📈 Signal History & Performance")
    stats = hist.get_stats()
    cola, colb, colc = st.columns(3)
    cola.metric("Total Signals", stats['total'])
    colb.metric("Closed", stats['closed'])
    colc.metric("Win Rate", f"{stats['win_rate']:.1f}%" if stats['closed'] > 0 else "N/A")

    # Rolling win rate chart
    if stats['closed'] > 0:
        closed_trades = [s for s in hist.history if s['status'] == 'closed']
        df_hist = pd.DataFrame(closed_trades)
        df_hist['timestamp'] = pd.to_datetime(df_hist['timestamp'])
        df_hist = df_hist.sort_values('timestamp')
        df_hist['win_int'] = df_hist['win'].astype(int)
        # Rolling win rate (last 20 trades)
        df_hist['rolling_win'] = df_hist['win_int'].rolling(20, min_periods=1).mean() * 100
        st.line_chart(df_hist.set_index('timestamp')[['rolling_win']], height=200)

    # Display history table
    if hist.history:
        st.dataframe(pd.DataFrame(hist.history).tail(20), use_container_width=True)

    # Multi-timeframe (optional, but we can show from v2)
    st.divider()
    st.subheader("🔭 Multi-Timeframe Quick View")
    mt_data = {}
    for tf in ["M5", "M15", "H1"]:
        try:
            if data_mode == "Twelve Data Live" and api_key:
                df_tf = fetch_twelvedata(pair, TF_MAP[tf], api_key)
            else:
                df_tf = demo_data(seed=sum(map(ord, pair)) + len(tf), tf=TF_MAP[tf])
            if len(df_tf) > 50:
                eng = AIEngine(df_tf, pair, tf)
                res = eng.analyze()
                mt_data[tf] = {
                    "Signal": res['signal'],
                    "Confidence": res['confidence'],
                    "Score": round(res['score'], 1),
                    "RSI": round(res['rsi'], 1),
                    "Direction": res['direction']
                }
        except:
            mt_data[tf] = {"Signal": "Error", "Confidence": 0}
    st.dataframe(pd.DataFrame(mt_data).T, use_container_width=True)

    st.info("⚠️ **Disclaimer:** This tool is for educational purposes only. Past performance does not guarantee future results. Always test thoroughly before real trading.")

if __name__ == "__main__":
    main()