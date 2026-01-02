"""Simple ETL: clean raw matches and compute result label."""
from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "matches_raw.csv"
PROCESSED_DIR = ROOT / "data" / "processed"


def compute_result(row):
    try:
        h = int(row['home_score'])
        a = int(row['away_score'])
    except Exception:
        return None
    if h > a:
        return 'home'
    if a > h:
        return 'away'
    return 'draw'


def main():
    if not RAW.exists():
        print(f"Raw file not found: {RAW}. Run src/ingest.py first.")
        return
    df = pd.read_csv(RAW)

    # Basic cleaning
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['year'] = df['date'].dt.year

    # normalize score columns
    for c in ['home_score', 'away_score']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    df['result'] = df.apply(compute_result, axis=1)
    df = df.dropna(subset=['result'])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / 'matches_processed.csv'
    df.to_csv(out, index=False)
    print(f"Processed data saved to {out}")


if __name__ == '__main__':
    main()
