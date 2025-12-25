import requests
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

# =========================
# Config
# =========================
FMP_BASE = "https://financialmodelingprep.com/stable"
FMP_API_KEY = "8DD30uW6GEg13Zh6aWv9vLGt4N6IQ91u"   # ok for local prototyping




LOOKBACK = 30     # past N trading days used as input window
HORIZON  = 1      # predict close H trading days ahead

# =========================
# API
# =========================
def fetch_eod_full(symbol: str) -> pd.DataFrame:
    url = f"{FMP_BASE}/historical-price-eod/full"
    params = {"symbol": symbol, "apikey": FMP_API_KEY}

    print(f"[API] Requesting EOD history: {url} | symbol={symbol}")
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()

    j = r.json()
    if isinstance(j, dict) and "historical" in j:
        rows = j["historical"]
    elif isinstance(j, list):
        rows = j
    else:
        print("[API] ❌ Unexpected response structure (showing first 400 chars):")
        print(str(j)[:400])
        raise ValueError("Unexpected API response structure")

    if not rows:
        raise ValueError("No EOD rows returned (symbol wrong? plan limitation? throttled?)")

    df = pd.DataFrame(rows)

    if "date" not in df.columns:
        raise ValueError("EOD data missing 'date' column")

    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values("date").set_index("date")

    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["close"])
    df.index.name = "date"

    print(f"[API] ✅ Received {len(df)} daily rows ({df.index.min().date()} → {df.index.max().date()})")
    return df
def fetch_current_price(symbol: str) -> float | None:
    url = f"{FMP_BASE}/quote"
    params = {"symbol": symbol, "apikey": FMP_API_KEY}

    print(f"[API] Requesting current price: {url} | symbol={symbol}")
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        j = r.json()
        if isinstance(j, list) and len(j) > 0 and "price" in j[0]:
            return float(j[0]["price"])
        print("[API] ❌ Unexpected quote response:", str(j)[:200])
        return None
    except requests.RequestException as e:
        print("[API] ❌ Current price request failed")
        print(e)
        return None
# =========================
# Feature engineering
# =========================
def add_basic_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret_1d"] = out["close"].pct_change(1)
    out["hl_range"] = (out["high"] - out["low"]) / out["close"]
    out["oc_range"] = (out["close"] - out["open"]) / out["open"]

    if "volume" in out.columns:
        out["vol_chg"] = out["volume"].pct_change(1)
    else:
        out["vol_chg"] = np.nan
    return out

def make_supervised_from_series(
    df: pd.DataFrame,
    target_col: str,
    lookback: int,
    horizon: int,
    feature_base_cols: list[str],
):
    tmp = df.copy()
    tmp["y"] = tmp[target_col].shift(-horizon)

    lag_frames = []
    feature_cols = []

    for col in feature_base_cols:
        for k in range(lookback):
            name = f"{col}_lag{k}"
            lag_frames.append(tmp[col].shift(k).rename(name))
            feature_cols.append(name)

    # concatenate all lag columns at once
    tmp = pd.concat([tmp] + lag_frames, axis=1)

    data = tmp.dropna(subset=feature_cols + ["y"]).copy()
    X = data[feature_cols].to_numpy()
    y = data["y"].to_numpy()
    return data, X, y, feature_cols

# =========================
# Model persistence with "feature contract"
# =========================
def create_mlp():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=500,
            warm_start=True,
            random_state=0,
        )),
    ])

def load_or_create_model(feature_cols: list[str], symbol: str):
    MODEL_PATH = Path(f"{symbol}_stock_mlp_Today.joblib")
    META_PATH  = Path(f"{symbol}_stock_mlp_Today_Meta.joblib")
    if MODEL_PATH.exists() and META_PATH.exists():
        meta = joblib.load(META_PATH)
        old_feats = meta.get("feature_cols")
        old_symbol = meta.get("symbol")

        if old_feats != feature_cols or old_symbol != symbol:
            print("[MODEL] ⚠️ Contract changed (features or symbol). Creating NEW model.")
            # back up old
            MODEL_PATH.rename(MODEL_PATH.with_suffix(".joblib.bak"))
            META_PATH.rename(META_PATH.with_suffix(".joblib.bak"))
            model = create_mlp()
        else:
            print("[MODEL] Loading existing model...")
            model = joblib.load(MODEL_PATH)
    else:
        print("[MODEL] Creating new model...")
        model = create_mlp()

    joblib.dump({"feature_cols": feature_cols, "symbol": symbol}, META_PATH)
    return model

def save_model(model, path):
    joblib.dump(model, path)
    print(f"[MODEL] ✅ Saved model to {path}")

# =========================
# Predict "3 trading days from today"
# =========================
def predict_from_today(feat: pd.DataFrame, model, feature_base_cols: list[str],
                       target_col: str, horizon: int, lookback: int, symbol: str):
    """
    Build a feature-only row for the LATEST available trading day in `feat`
    (no need for future y), then predict close at +horizon trading steps.
    """
    # Build the same lag feature names as training
    feature_cols = []
    tmp = feat.copy()

    lag_frames = []
    feature_cols = []

    for col in feature_base_cols:

        for k in range(lookback):
            name = f"{col}_lag{k}"
            lag_frames.append(tmp[col].shift(k).rename(name))
            feature_cols.append(name)

    # concatenate all lag columns at once
    tmp = pd.concat([tmp] + lag_frames, axis=1)

    # Latest day with all lag features available (this *will* be the real "today" row)
    today_row = tmp.dropna(subset=feature_cols).iloc[-1]
    anchor_date = today_row.name
    anchor_close = float(today_row[target_col])

    X_today = today_row[feature_cols].to_numpy().reshape(1, -1)
    y_pred = float(model.predict(X_today)[0])

    # True target date = +horizon trading steps in the FULL feat index
    idx = feat.index
    anchor_pos = idx.get_loc(anchor_date)
    target_pos = anchor_pos + horizon
    target_date = idx[target_pos] if target_pos < len(idx) else None

    pred_delta = y_pred - anchor_close
    pred_pct = (pred_delta / anchor_close) * 100.0 if anchor_close != 0 else float("nan")

    # Print last two weeks of closes ending at anchor (true today)
    hist_slice = feat.loc[:anchor_date].tail(lookback)
    
    print("\n[LAST 2 WEEKS LOOKBACK]")
    for d, row in hist_slice.iterrows():
        print(f"{d.date()} : {float(row[target_col]):8.2f}")

    print("\n[PREDICTION]")
    print(f"Symbol                    : {symbol}")
    print(f"Anchor date (today)       : {anchor_date.date()}")
    print(f"Anchor {target_col}       : {anchor_close:.2f}")

    current_price = fetch_current_price(symbol)
    print(f"Current price (quote)     : {current_price:.2f}" if current_price is not None
          else "Current price (quote)     : (unavailable)")

    if target_date is not None:
        print(f"Target date (estimated)   : {target_date.date()} (+{horizon} trading steps)")
    else:
        print(f"Target date (estimated)   : +{horizon} trading steps (beyond available EOD data)")

    print(f"Predicted {target_col}    : {y_pred:.2f}")
    print(f"Predicted Δ               : {pred_delta:+.2f}")
    print(f"Predicted %Δ              : {pred_pct:+.2f}%")
    target_date = anchor_date + pd.Timedelta(days=HORIZON)
    print(f"Target date (estimated normal days)   : {target_date.date()} (+{HORIZON} days)")
    

    return pred_pct


# =========================
# Main
# =========================
def predict(SYMBOL):
    symbol = SYMBOL
    
    raw = fetch_eod_full(symbol)
    feat = add_basic_signals(raw)

    feature_base_cols = [
        "close",
        "volume",
        "ret_1d",
        "hl_range",
        "oc_range",
        "vol_chg",
    ]
    feature_base_cols = [c for c in feature_base_cols if c in feat.columns]

    target_col = "close"

    data, X, y, feature_cols = make_supervised_from_series(
        feat,
        target_col=target_col,
        lookback=LOOKBACK,
        horizon=HORIZON,
        feature_base_cols=feature_base_cols
    )

    print(f"\n[DATA]")
    print(f"Daily rows: {len(feat)} | Supervised samples: {len(X)} | Inputs: {X.shape[1]}")
    print(f"Signals   : {len(feature_base_cols)} -> {feature_base_cols}")

    model = load_or_create_model(feature_cols, symbol)

    model.fit(X, y)

    mlp = model.named_steps["mlp"]
    print("\n[TRAINING STATE]")
    print(f"  Iterations this run: {len(mlp.loss_curve_)}")
    print(f"  Final loss         : {mlp.loss_curve_[-1]:.6f}")
    path = MODEL_PATH = Path(f"{symbol}_stock_mlp_Today.joblib")
    save_model(model, path)

    # Predict 3 trading days from "today" (latest data)
    pred = predict_from_today(feat, model, feature_base_cols, target_col, HORIZON, LOOKBACK, symbol)

    return pred

if __name__ == "__main__":
    predict("NVDA")
