"""Feature engineering: create model-ready CSV with one-hot teams and basic numeric features."""
from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "matches_processed.csv"
FEATURES_OUT = ROOT / "data" / "processed" / "features.csv"


def main():
    if not PROCESSED.exists():
        print(f"Processed file not found: {PROCESSED}. Run src/etl.py first.")
        return
    df = pd.read_csv(PROCESSED)

    # Simple numeric features
    df['goal_diff'] = df['home_score'] - df['away_score']
    df['home_goals'] = df['home_score']
    df['away_goals'] = df['away_score']

    # One-hot encode teams (sparse but OK for demo)
    teams_home = pd.get_dummies(df['home_team'], prefix='home')
    teams_away = pd.get_dummies(df['away_team'], prefix='away')

    X = pd.concat([df[['year', 'home_goals', 'away_goals', 'goal_diff']], teams_home, teams_away], axis=1)
    y = df['result']

    out = FEATURES_OUT
    X.assign(result=y).to_csv(out, index=False)
    print(f"Features saved to {out}")


if __name__ == '__main__':
    main()
