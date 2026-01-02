"""Ingest script: copy source CSV to data/raw and produce simple EDA."""
from pathlib import Path
import pandas as pd
import os


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    src_csv = ROOT / "afcon_country_matches_2000-2025.csv"
    if not src_csv.exists():
        print(f"Fichier source introuvable: {src_csv}")
        return

    df = pd.read_csv(src_csv)
    out = RAW_DIR / "matches_raw.csv"
    df.to_csv(out, index=False)
    print(f"Saved raw to {out}")

    # Simple EDA
    eda = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "teams_home_top10": df['home_team'].value_counts().head(10).to_dict(),
        "teams_away_top10": df['away_team'].value_counts().head(10).to_dict(),
    }
    eda_df = pd.DataFrame([eda])
    eda_df.to_csv(REPORTS / "eda_summary.csv", index=False)
    print(f"EDA summary saved to {REPORTS / 'eda_summary.csv'}")


if __name__ == '__main__':
    main()
