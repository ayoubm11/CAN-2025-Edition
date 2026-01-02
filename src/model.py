"""Train a simple classifier and save the model and feature columns."""
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import json


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "processed" / "features.csv"
MODEL_DIR = ROOT / "models"


def main():
    if not FEATURES.exists():
        print(f"Features file not found: {FEATURES}. Run src/features.py first.")
        return
    df = pd.read_csv(FEATURES)
    if 'result' not in df.columns:
        print('No target column `result` in features file')
        return

    X = df.drop(columns=['result'])
    y = df['result']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(classification_report(y_test, preds))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / 'can2025_model.joblib'
    joblib.dump(model, model_path)

    cols_path = MODEL_DIR / 'feature_columns.json'
    with open(cols_path, 'w', encoding='utf8') as f:
        json.dump({'columns': X.columns.tolist()}, f)

    print(f"Model saved to {model_path}")


if __name__ == '__main__':
    main()
