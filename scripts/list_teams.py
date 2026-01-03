from pathlib import Path
import pandas as pd

p = Path(__file__).resolve().parents[1] / 'data' / 'processed' / 'matches_processed.csv'
if not p.exists():
    print('FICHIER_INTRouvABLE:', p)
    raise SystemExit(1)

df = pd.read_csv(p)
teams = set()
if 'home_team' in df.columns:
    teams.update(df['home_team'].dropna().unique().tolist())
if 'away_team' in df.columns:
    teams.update(df['away_team'].dropna().unique().tolist())

print('--- Teams found in processed CSV ---')
for t in sorted(teams):
    print(t)
print('--- Count:', len(teams), '---')
