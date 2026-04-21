import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from nifty50 import NIFTY50
from data_loader import load_data
from indicators import add_indicators
from model import train_model, predict

st.set_page_config(
    page_title="NIFTY 50 AI Stock App",
    page_icon="📈",
    layout="wide"
)

st.title("📈 NIFTY 50 AI Stock App")
st.caption("Powered by yfinance + scikit-learn | For educational purposes only")

with st.sidebar:
    st.header("⚙️ Settings")
    stock = st.selectbox("Select Stock", list(NIFTY50.keys()))
    period = st.selectbox("Period", ["6mo", "1y", "2y", "5y"], index=2)
    show_bollinger = st.checkbox("Show Bollinger Bands", value=True)
    show_rsi = st.checkbox("Show RSI", value=True)
    show_macd = st.checkbox("Show MACD", value=True)

symbol = NIFTY50[stock]

@st.cache_data(ttl=3600)
def get_data(symbol, period):
    df = load_data(symbol, period=period)
    if df is None:
        return None
    return add_indicators(df)

with st.spinner(f"Loading data for {stock}..."):
    df = get_data(symbol, period)

if df is None or df.empty:
    st.error(f"❌ Could not load data for **{stock}**. Try another stock or period.")
    st.stop()

feature_cols = ['MA10', 'MA50', 'RSI', 'MACD',
                'Bollinger_Upper', 'Bollinger_Lower']
feature_cols = [c for c in feature_cols if c in df.columns]

X = df[feature_cols]
y = df['Close']

if X.isnull().values.any():
    df = df.dropna(subset=feature_cols)
    X = df[feature_cols]
    y = df['Close']

if len(X) < 50:
    st.error("❌ Not enough data. Try a longer period.")
    st.stop()

model = train_model(X.values, y)
pred = predict(model, X.values)

last_close  = y.values[-1]
last_pred   = pred[-1]
change      = last_pred - last_close
change_pct  = (change / last_close) * 100

st.subheader(f"📊 {stock} ({symbol})")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Last Close",  f"₹{last_close:.2f}")
c2.metric("Predicted",   f"₹{last_pred:.2f}")
c3.metric("Change",      f"₹{change:.2f}",    f"{change_pct:+.2f}%")
c4.metric("Data Points", f"{len(df)}")

if change > 0:
    st.success(f"### 🟢 BUY Signal — Predicted upside of ₹{change:.2f} ({change_pct:+.2f}%)")
else:
    st.error(f"### 🔴 SELL Signal — Predicted downside of ₹{abs(change):.2f} ({change_pct:+.2f}%)")

st.divider()

st.subheader("📉 Price Chart")

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df.index, y,    label="Actual Close",  color="#00BFFF", linewidth=1.5)
ax.plot(df.index, pred, label="Predicted",     color="#FFA500",
        linewidth=1.5, linestyle="--")
ax.plot(df.index, df['MA10'], label="MA10",    color="#90EE90",
        linewidth=1, alpha=0.7)
ax.plot(df.index, df['MA50'], label="MA50",    color="#FFB6C1",
        linewidth=1, alpha=0.7)

if show_bollinger and 'Bollinger_Upper' in df.columns:
    ax.fill_between(df.index,
                    df['Bollinger_Upper'],
                    df['Bollinger_Lower'],
                    alpha=0.1, color="cyan", label="Bollinger Band")

ax.set_facecolor("#0e1117")
fig.patch.set_facecolor("#0e1117")
ax.tick_params(colors="white")
ax.xaxis.label.set_color("white")
ax.yaxis.label.set_color("white")
ax.title.set_color("white")
ax.set_title(f"{stock} — Actual vs Predicted Close Price")
ax.set_ylabel("Price (₹)")
ax.legend(facecolor="#1a1a2e", labelcolor="white")
ax.grid(alpha=0.2)

st.pyplot(fig)

if show_rsi and 'RSI' in df.columns:
    st.subheader("📊 RSI (Relative Strength Index)")
    fig2, ax2 = plt.subplots(figsize=(14, 2.5))
    ax2.plot(df.index, df['RSI'], color="#FF6347", linewidth=1.2)
    ax2.axhline(70, color="red",   linestyle="--", alpha=0.6, label="Overbought (70)")
    ax2.axhline(30, color="green", linestyle="--", alpha=0.6, label="Oversold (30)")
    ax2.fill_between(df.index, df['RSI'], 70,
                     where=(df['RSI'] >= 70), alpha=0.2, color="red")
    ax2.fill_between(df.index, df['RSI'], 30,
                     where=(df['RSI'] <= 30), alpha=0.2, color="green")
    ax2.set_facecolor("#0e1117")
    fig2.patch.set_facecolor("#0e1117")
    ax2.tick_params(colors="white")
    ax2.set_ylabel("RSI", color="white")
    ax2.legend(facecolor="#1a1a2e", labelcolor="white")
    ax2.set_ylim(0, 100)
    ax2.grid(alpha=0.2)
    st.pyplot(fig2)

    rsi_now = df['RSI'].values[-1]
    if rsi_now >= 70:
        st.warning(f"⚠️ RSI is **{rsi_now:.1f}** — Stock may be **overbought**")
    elif rsi_now <= 30:
        st.info(f"💡 RSI is **{rsi_now:.1f}** — Stock may be **oversold**")
    else:
        st.success(f"✅ RSI is **{rsi_now:.1f}** — Neutral zone")

if show_macd and 'MACD' in df.columns and 'MACD_Signal' in df.columns:
    st.subheader("📊 MACD")
    fig3, ax3 = plt.subplots(figsize=(14, 2.5))
    ax3.plot(df.index, df['MACD'],        label="MACD",   color="#00BFFF", linewidth=1.2)
    ax3.plot(df.index, df['MACD_Signal'], label="Signal", color="#FFA500", linewidth=1.2)
    ax3.bar(df.index,
            df['MACD'] - df['MACD_Signal'],
            label="Histogram",
            color=["green" if v >= 0 else "red"
                   for v in (df['MACD'] - df['MACD_Signal'])],
            alpha=0.4, width=1)
    ax3.set_facecolor("#0e1117")
    fig3.patch.set_facecolor("#0e1117")
    ax3.tick_params(colors="white")
    ax3.set_ylabel("MACD", color="white")
    ax3.legend(facecolor="#1a1a2e", labelcolor="white")
    ax3.grid(alpha=0.2)
    st.pyplot(fig3)

with st.expander("📋 View Raw Data"):
    st.dataframe(
        df[['Close', 'MA10', 'MA50', 'RSI',
            'MACD', 'Bollinger_Upper', 'Bollinger_Lower']
          ].tail(30).sort_index(ascending=False),
        use_container_width=True
    )