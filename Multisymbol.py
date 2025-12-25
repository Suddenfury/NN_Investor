import time
import pandas as pd

from Investortoday import predict   # ← your existing script

# =========================
# Symbols to scan
# =========================
SYMBOLS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    "TSLA",
    "KO",
    "PEP",
    "WMT",
    "SPY",
    
]

SLEEP_SECONDS = 1.2  # be nice to the API

# =========================
# Run
# =========================
def run_batch(symbols):
    results = []

    for sym in symbols:
        print("\n" + "=" * 60)
        print(f"Running prediction for {sym}")

        try:
            pred_pct = predict(sym)
            results.append({
                "symbol": sym,
                "predicted_pct": pred_pct
            })
            print(f"[RESULT] {sym}: {pred_pct:+.2f}%")

        except Exception as e:
            print(f"[ERROR] {sym}: {e}")
            results.append({
                "symbol": sym,
                "predicted_pct": float("nan")
            })

        time.sleep(SLEEP_SECONDS)

    return pd.DataFrame(results)

if __name__ == "__main__":
    df = run_batch(SYMBOLS)

    print("\n" + "=" * 60)
    print("[SUMMARY – Predicted % Move in +3 Trading Days]")
    print("=" * 60)

    df = df.sort_values("predicted_pct", ascending=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:+.2f}%"))
