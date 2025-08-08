import pandas as pd
import numpy as np
import yfinance as yf

def fetch_prices(ticker: str, years: int = 2) -> pd.DataFrame:
    """Download daily OHLCV from Yahoo Finance."""
    period = f"{years}y"
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise ValueError("No data returned for ticker.")
    df = df.dropna()
    return df

def _true_range(row, prev_close):
    hl = row["High"] - row["Low"]
    hc = abs(row["High"] - prev_close)
    lc = abs(row["Low"] - prev_close)
    return max(hl, hc, lc)

def compute_indicators(df: pd.DataFrame, hl=10, atr_win=14, vol_win=20, vov_win=20) -> pd.DataFrame:
    out = df.copy()
    out["Ret"] = out["Close"].pct_change()
    out["LogRet"] = np.log(out["Close"]).diff()

    # EWMA close-to-close volatility (daily)
    out["Vol_EWMA"] = out["LogRet"].ewm(halflife=hl, adjust=False).std()

    # ATR in price units, then as a percent of price
    prev_close = out["Close"].shift(1)
    tr = out.apply(lambda r: _true_range(r, prev_close.loc[r.name]), axis=1)
    out["ATR"] = tr.rolling(atr_win, min_periods=atr_win).mean()
    out["Vol_ATR"] = out["ATR"] / out["Close"]  # as % of price

    # Garman–Klass daily volatility estimator (variance -> stdev)
    gk_var = (0.5 * (np.log(out["High"] / out["Low"]))**2
              - (2*np.log(2) - 1) * (np.log(out["Close"] / out["Open"]))**2)
    out["Vol_GK"] = np.sqrt(np.maximum(gk_var, 0))

    # Volume z-score (how unusual is today vs. last N days)
    out["VolZ"] = (out["Volume"] - out["Volume"].rolling(vol_win).mean()) / out["Volume"].rolling(vol_win).std()

    # Volatility-of-volatility (stdev of EWMA vol), then z-score
    vov = out["Vol_EWMA"].rolling(vov_win).std()
    out["VoV_Z"] = (vov - vov.rolling(vov_win).mean()) / vov.rolling(vov_win).std()

    # Daily Rate of Change (ROC)
    out["ROC_1d"] = out["Close"].pct_change()
    out["ROC_20d"] = out["Close"].pct_change(20)

    return out

def build_risk_range(out: pd.DataFrame,
                     z=1.65,
                     w_ewm=0.5, w_gk=0.3, w_atr=0.2,
                     vol_adj=0.15, vov_adj=0.10,
                     tilt_gamma=0.10) -> pd.DataFrame:
    out = out.copy()

    # Normalize weights
    s = (w_ewm + w_gk + w_atr)
    if s == 0:
        w_ewm = 0.5; w_gk = 0.3; w_atr = 0.2
        s = 1.0
    w_ewm, w_gk, w_atr = w_ewm/s, w_gk/s, w_atr/s

    # Combined daily volatility (fraction)
    out["Vol_Combined"] = (w_ewm * out["Vol_EWMA"]
                           + w_gk  * out["Vol_GK"]
                           + w_atr * out["Vol_ATR"])

    # Adjust width by volume and vol-of-vol regimes (z-scores)
    vol_factor = (1 + vol_adj * out["VolZ"].fillna(0))
    vov_factor = (1 + vov_adj * out["VoV_Z"].fillna(0))

    # Width in price units
    out["WidthPct"] = (z * out["Vol_Combined"] * vol_factor * vov_factor).clip(lower=0)
    out["Width"] = out["WidthPct"] * out["Close"]

    # Tilt center by recent trend (20d ROC), scaled by gamma
    center_tilt = tilt_gamma * out["ROC_20d"].fillna(0) * out["Width"]
    out["Center"] = out["Close"] + center_tilt

    out["Upper"] = out["Center"] + out["Width"]
    out["Lower"] = out["Center"] - out["Width"]

    return out

def make_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["Open","High","Low","Close","Volume",
            "Upper","Lower","Width","WidthPct",
            "Vol_EWMA","Vol_GK","Vol_ATR","Vol_Combined",
            "VolZ","VoV_Z","ROC_1d","ROC_20d"]
    tbl = df[cols].dropna().copy()
    pretty = {
        "WidthPct":"Width_%",
        "Vol_EWMA":"Vol_EWMA_%",
        "Vol_GK":"Vol_GK_%",
        "Vol_ATR":"Vol_ATR_%",
        "Vol_Combined":"Vol_Combined_%",
        "ROC_1d":"ROC_1d_%",
        "ROC_20d":"ROC_20d_%",
    }
    for k,v in pretty.items():
        tbl[v] = 100*tbl[k]
        del tbl[k]
    return tbl
