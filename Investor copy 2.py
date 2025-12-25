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

MODEL_PATH = Path("ecosystem_mlp.joblib")
META_PATH  = Path("ecosystem_mlp_meta.joblib")  # stores feature list + config so we can detect mismatch

LOOKBACK = 14   # past N trading days (lags)
HORIZON  = 3    # predict WMT return H trading days ahead

# Instruments in the ecosystem
TARGET_TICKER = "WMT"
ECOSYSTEM = {
    "adm": "ADM",
    "gis": "GIS",
    "wmt": "WMT",
    "wheat": "ZW=F",   # wheat futures on FMP (symbol is commonly ZW=F)
}

# =========================
# API
# =========================
def fetch_eod_full(symbol: str) -> pd.DataFrame:
    """
    Fetch full EOD history (daily candles) for a symbol.
    Returns a dataframe indexed by date ascending with columns:
      open, high, low, close, volume  (as available)
    """
    url = f"{FMP_BASE}/historical-price-eod/full"
    params = {"symbol": symbol, "apikey": FMP_API_KEY}

    print(f"[API] Requesting EOD history: {url} | symbol={symbol}")
    try:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
    except requests.RequestException as e:
        print("[API] ❌ Request failed")
        print(f"[API] Error: {e}")
        raise

    j = r.json()

    # FMP often returns either {"symbol":..., "historical":[...]} or a direct list depending on endpoint/version.
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

    # force numeric for known fields if present
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["close"])  # must have close
    df.index.name = "date"

    print(f"[API] ✅ Received {len(df)} daily rows ({df.index.min().date()} → {df.index.max().date()})")
    return df

# =========================
# Feature engineering
# =========================
def add_basic_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a few simple derived signals from OHLCV.
    """
    out = df.copy()

    out["ret_1d"] = out["close"].pct_change(1)
    out["hl_range"] = (out["high"] - out["low"]) / out["close"]
    out["oc_range"] = (out["close"] - out["open"]) / out["open"]

    if "volume" in out.columns:
        out["vol_chg"] = out["volume"].pct_change(1)
    else:
        out["vol_chg"] = np.nan

    return out

def prefix_cols(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [f"{prefix}_{c}" for c in df.columns]
    return df

def build_big_ecosystem_frame() -> pd.DataFrame:
    """
    Build a wide dataframe on WMT trading days (master index),
    containing prefixed signals from ADM, GIS, WMT, and wheat.
    """
    # Fetch raw
    raw = {}
    for key, sym in ECOSYSTEM.items():
        raw[key] = fetch_eod_full(sym)[["open", "high", "low", "close", "volume"]]

    # Add signals
    feat = {k: add_basic_signals(v) for k, v in raw.items()}

    # Use WMT index as the master calendar
    master = feat["wmt"].sort_index()
    idx = master.index

    # Reindex all to WMT days (forward fill only from past)
    aligned = {}
    for k, df in feat.items():
        df = df.sort_index().reindex(idx).ffill()
        aligned[k] = df

    # Prefix columns so we can concatenate safely
    wide_parts = []
    for k, df in aligned.items():
        wide_parts.append(prefix_cols(df, k))

    big = pd.concat(wide_parts, axis=1)

    # ---- TARGET: WMT future log return over HORIZON trading days ----
    # y(t) = log(WMT_close[t+H] / WMT_close[t])
    big["y"] = np.log(big["wmt_close"].shift(-HORIZON) / big["wmt_close"])

    # Drop rows that can't form target
    big = big.dropna(subset=["y"])

    return big

def make_supervised_lags(
    df: pd.DataFrame,
    lookback: int,
    feature_base_cols: list[str],
):
    """
    Build X/y with lag features.
    Here y is already aligned at t (future return computed in build_big_ecosystem_frame),
    so we do NOT shift y again.
    """
    tmp = df.copy()

    feature_cols = []
    for col in feature_base_cols:
        for k in range(lookback):
            name = f"{col}_lag{k}"
            tmp[name] = tmp[col].shift(k)
            feature_cols.append(name)

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
            hidden_layer_sizes=(128, 64),  # slightly larger for multi-series inputs
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=500,
            warm_start=True,
            random_state=0,
        )),
    ])

def load_or_create_model(feature_cols: list[str], model_id: str):
    """
    Loads model if available AND feature_cols match AND model_id matches.
    Otherwise creates a new model.
    """
    if MODEL_PATH.exists() and META_PATH.exists():
        meta = joblib.load(META_PATH)
        old_feats = meta.get("feature_cols")
        old_id = meta.get("model_id")

        if old_feats != feature_cols:
            print("[MODEL] ⚠️ Feature set changed since last run.")
            print(f"[MODEL] Old inputs: {len(old_feats)} | New inputs: {len(feature_cols)}")
            print("[MODEL] Creating NEW model (old model kept on disk).")
            MODEL_PATH.rename(MODEL_PATH.with_suffix(".joblib.bak"))
            META_PATH.rename(META_PATH.with_suffix(".joblib.bak"))
            model = create_mlp()
        elif old_id != model_id:
            print(f"[MODEL] ⚠️ Model id mismatch (old={old_id}, new={model_id}).")
            print("[MODEL] Creating NEW model.")
            MODEL_PATH.rename(MODEL_PATH.with_suffix(".joblib.bak"))
            META_PATH.rename(META_PATH.with_suffix(".joblib.bak"))
            model = create_mlp()
        else:
            print("[MODEL] Loading existing model...")
            model = joblib.load(MODEL_PATH)
    else:
        print("[MODEL] Creating new model...")
        model = create_mlp()

    joblib.dump({"feature_cols": feature_cols, "model_id": model_id}, META_PATH)
    return model

def save_model(model):
    joblib.dump(model, MODEL_PATH)
    print(f"[MODEL] ✅ Saved model to {MODEL_PATH}")

# =========================
# Evaluation: random date check (movement-space)
# =========================
def random_date_check(df_supervised: pd.DataFrame, model, feature_cols: list[str]):
    """
    Prints:
      - anchor date
      - horizon steps
      - actual movement (log return) and predicted movement (log return)
      - also shows implied % move for readability
    """
    rng = np.random.default_rng()
    i = int(rng.integers(0, len(df_supervised)))
    row = df_supervised.iloc[i]
    anchor_date = df_supervised.index[i]

    y_true = float(row["y"])  # actual future log return
    x = row[feature_cols].to_numpy().reshape(1, -1)
    y_pred = float(model.predict(x)[0])

    # For context: show last 3 days of WMT close
    prev_dates = df_supervised.index[max(0, i-3):i]
    print("\n[RECENT HISTORY: WMT close]")
    for d in prev_dates:
        v = float(df_supervised.loc[d, "wmt_close"])
        print(f"{d.date()} : {v:8.2f}")

    # Movement print (log-return + implied percent)
    true_pct = (np.exp(y_true) - 1.0) * 100.0
    pred_pct = (np.exp(y_pred) - 1.0) * 100.0

    direction_ok = np.sign(y_true) == np.sign(y_pred)

    print("\n[RANDOM DATE CHECK]")
    print(f"Anchor date (features end): {anchor_date.date()}")
    print(f"Target (horizon)          : +{HORIZON} trading steps")
    print(f"Actual movement (logret)  : {y_true:+.6f}  ({true_pct:+.3f}%)")
    print(f"Pred   movement (logret)  : {y_pred:+.6f}  ({pred_pct:+.3f}%)")
    print(f"Error (logret)            : {y_pred - y_true:+.6f}")
    print(f"Direction                 : {'✅ CORRECT' if direction_ok else '❌ WRONG'}")

# =========================
# Main
# =========================
def main():
    print("[BUILD] Ecosystem: ADM + GIS + WMT + Wheat -> predict WMT future return")
    big = build_big_ecosystem_frame()

    # Base features = everything except y
    # (Later: you can drop raw OHLC if you want and keep only derived signals)
    feature_base_cols = [c for c in big.columns if c != "y"]

    data, X, y, feature_cols = make_supervised_lags(
        big,
        lookback=LOOKBACK,
        feature_base_cols=feature_base_cols
    )

    print(f"\n[DATA]")
    print(f"Wide rows (after y): {len(big)} | Supervised samples: {len(X)} | Inputs: {X.shape[1]}")
    print(f"Base signals        : {len(feature_base_cols)} columns (before lagging)")
    print(f"Lagged inputs       : {len(feature_cols)} columns (after lagging)")
    print(f"Target              : y = log(WMT_close[t+{HORIZON}] / WMT_close[t])")

    # Make a model id so contract changes if you change the ecosystem makeup
    model_id = f"ecosystem:{','.join([ECOSYSTEM[k] for k in ECOSYSTEM.keys()])}|lookback={LOOKBACK}|h={HORIZON}"

    model = load_or_create_model(feature_cols, model_id)

    model.fit(X, y)

    mlp = model.named_steps["mlp"]
    print("\n[TRAINING CONFIG]")
    print(f"  Hidden layers     : {mlp.hidden_layer_sizes}")
    print(f"  Activation        : {mlp.activation}")
    print(f"  Solver            : {mlp.solver}")
    print(f"  Alpha (L2)        : {mlp.alpha}")
    print(f"  Learning rate     : {mlp.learning_rate_init}")
    print(f"  Max iter / run    : {mlp.max_iter}")
    print(f"  Warm start        : {mlp.warm_start}")

    print("\n[TRAINING STATE]")
    print(f"  Iterations this run: {len(mlp.loss_curve_)}")
    print(f"  Final loss         : {mlp.loss_curve_[-1]:.6f}")

    save_model(model)

    random_date_check(data, model, feature_cols)


if __name__ == "__main__":
    main()
