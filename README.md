# Projet CAN 2025 — Analyse des Performances des Équipes

Projet local pour analyser les matchs de la CAN 2000-2025, construire un pipeline ETL, entraîner un modèle simple et fournir un dashboard interactif local (Streamlit).

Principes:
- Pas besoin d'un compte Azure — la solution s'exécute localement
- Outils: Python, Pandas, scikit-learn, Streamlit, Prefect (optionnel)

Fichiers créés:
- `src/ingest.py` — ingestion et EDA basique
- `src/etl.py` — nettoyage et transformation
- `src/features.py` — feature engineering
- `src/model.py` — entraînement & sauvegarde du modèle
- `src/evaluate.py` — évaluation et métriques
- `src/dashboard.py` — dashboard Streamlit
- `requirements.txt` — dépendances
- `.env.example` — variables d'environnement

Installation (Windows PowerShell):

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Exemples de commandes:

```powershell
# Ingest + EDA
py -3 src\ingest.py

# Run ETL
py -3 src\etl.py

# Feature engineering
py -3 src\features.py

# Train model
py -3 src\model.py

# Evaluate
py -3 src\evaluate.py

# Start dashboard
streamlit run src/dashboard.py
```

Notes:
- Les données source doivent être `afcon_country_matches_2000-2025.csv` à la racine du projet.
- Le dashboard est une alternative locale à Power BI; si vous préférez Power BI Desktop, vous pouvez charger les fichiers CSV produits dans `data/processed`.

Structure produite:

- `data/raw` — données brutes
- `data/processed` — données transformées
- `models` — modèles entraînés
- `reports` — rapports/EDA
