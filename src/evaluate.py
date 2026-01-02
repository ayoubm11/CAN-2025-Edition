"""Load model and produce evaluation metrics file."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import classification_report
import json


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "processed" / "features.csv"
MODEL_DIR = ROOT / "models"


def main():
    model_path = MODEL_DIR / 'can2025_model.joblib'
    cols_path = MODEL_DIR / 'feature_columns.json'
    if not model_path.exists() or not cols_path.exists():
        print("Model not found. Run src/model.py first.")
        return

    df = pd.read_csv(FEATURES)
    X = df.drop(columns=['result'])
    y = df['result']

    model = joblib.load(model_path)
    preds = model.predict(X)
    report = classification_report(y, preds, output_dict=True)

    out = MODEL_DIR / 'evaluation.json'
    with open(out, 'w', encoding='utf8') as f:
        json.dump(report, f, indent=2)
    print(f"Evaluation saved to {out}")


if __name__ == '__main__':
    main()
