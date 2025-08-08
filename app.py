import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

from risk_range import fetch_prices, compute_indicators, build_risk_range, make_table


st.set_page_config(page_title="Multi-Stock Risk Range", layout="wide")

st.title("📊 Multi-Stock Risk Range Comparison — Price • Volume • Volatility")

with st.sidebar:
    st.header("Inputs")
    tickers = st.text_input("Tickers (comma-separated)", value="AAPL,MSFT").upper().replace(" ", "").split(",")
    years = st.slider("Lookback (years)", 1, 15, 2)
    conf = st.select_slider("Confidence (Z)", options=[0.67, 1.0, 1.28, 1.65, 1.96, 2.33], value=1.65)
    st.markdown("---")
    st.subheader("Volatility Settings")
    hl = st.slider("EWMA half-life (days)", 3, 60, 10)
    atr_win = st.slider("ATR window (days)", 5, 30, 14)
    vol_win = st.slider("Volume z-score window", 10, 60, 20)
    vov_win = st.slider("Vol-of-Vol window", 10, 60, 20)
    st.markdown("---")
    st.subheader("Blend & Tilt")
    w_ewm = st.slider("Weight: EWMA vol", 0.0, 1.0, 0.5)
    w_gk  = st.slider("Weight: Garman–Klass vol", 0.0, 1.0, 0.3)
    w_atr = st.slider("Weight: ATR% vol", 0.0, 1.0, 0.2)
    vol_adj = st.slider("Adj by Volume Z (α)", 0.0, 0.50, 0.15)
    vov_adj = st.slider("Adj by Vol-of-Vol Z (β)", 0.0, 0.50, 0.10)
    tilt_gamma = st.slider("Trend Tilt by ROC (γ)", -0.5, 0.5, 0.10, step=0.01)
    st.markdown("---")
    show_table_rows = st.slider("Table rows", 10, 200, 60)

@st.cache_data(show_spinner=False)
def load_data(ticker, years):
    return fetch_prices(ticker, years=years)

if tickers:
    for ticker in tickers:
        st.markdown(f"## {ticker}")
        try:
            df = load_data(ticker, years)
            df = compute_indicators(df, hl=hl, atr_win=atr_win, vol_win=vol_win, vov_win=vov_win)
            df = build_risk_range(
                df,
                z=conf,
                w_ewm=w_ewm,
                w_gk=w_gk,
                w_atr=w_atr,
                vol_adj=vol_adj,
                vov_adj=vov_adj,
                tilt_gamma=tilt_gamma
            )
            latest = df.dropna().iloc[-1]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Close", f"{latest['Close']:.2f}")
            c2.metric("Risk Range", f"{latest['Lower']:.2f} – {latest['Upper']:.2f}")
            c3.metric("Width %", f"{100*latest['WidthPct']:.2f}%")
            c4.metric("Daily ROC", f"{100*latest['ROC_1d']:.2f}%")

            fig = plt.figure(figsize=(10,3))
            plt.plot(df.index, df["Close"], label="Close")
            plt.plot(df.index, df["Upper"], label="Upper")
            plt.plot(df.index, df["Lower"], label="Lower")
            plt.title(f"{ticker} — Price with Risk Range")
            plt.legend()
            st.pyplot(fig, clear_figure=True)

            st.dataframe(make_table(df).tail(show_table_rows), use_container_width=True)

            csv = make_table(df).to_csv(index=True).encode("utf-8")
            st.download_button("Download CSV", data=csv, file_name=f"{ticker}_risk_range.csv", mime="text/csv")

            st.caption("Method: Bands = center ± width. Width = Z * CombinedVol * (1 + α·VolZ) * (1 + β·VoV_Z). "
                       "Center = Close + γ · ROC(20d) · width.")
        except Exception as e:
            st.error(f"Error for {ticker}: {e}")
else:
    st.info("Enter at least one ticker.")
