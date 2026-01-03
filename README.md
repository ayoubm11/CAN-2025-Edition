# Projet CAN 2025 — Analyse des Performances des Équipes Africaines

<div align="center">
  <img src="photo/head.png" alt="CAN 2025 Header" width="100%"/>
  
  ![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
  ![License](https://img.shields.io/badge/License-MIT-green.svg)
  ![Status](https://img.shields.io/badge/Status-Active-success.svg)
  ![Maintenance](https://img.shields.io/badge/Maintained-Yes-brightgreen.svg)
</div>

## 🎯 À Propos

Projet d'analyse approfondie des performances des équipes africaines de football lors de la **Coupe d'Afrique des Nations (CAN)** sur une période de **25 ans (2000-2025)**. Cette solution exploite les techniques modernes de **Data Science** et **Machine Learning** pour extraire des insights stratégiques et prédictifs.

### 🎓 SBI Student Challenge - Édition CAN 2025

Ce projet fait partie du **SBI Student Challenge**, une initiative visant à promouvoir l'analyse data-driven dans le sport africain.

### 🌟 Points Forts

- ✅ **Pipeline ETL Complet** — Ingestion, transformation et enrichissement automatisés
- ✅ **Machine Learning Prédictif** — Modèle de prédiction de performances
- ✅ **Dashboard Interactif** — Visualisations dynamiques avec Streamlit
- ✅ **Open Source** — Code documenté et réutilisable

## ✨ Fonctionnalités

### 📊 Analyse des Données
- Exploration approfondie de **3685 matchs**
- Analyse de **26 équipes** africaines
- Identification de **24 joueurs clés**
- Calcul du ratio moyen de buts : **2.19 buts/match**

### 🤖 Machine Learning
- Modèle prédictif de résultats de matchs
- Feature engineering avancé
- Validation croisée et optimisation
- Évaluation multi-métriques

### 📈 Visualisations
- Dashboard Streamlit interactif
- Graphiques temporels d'évolution
- Analyses comparatives d'équipes
- Radar charts de performances

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Raw Data      │────▶│   ETL Pipeline  │────▶│  Processed Data │
│   (CSV)         │     │   (Cleaning)    │     │   (Enriched)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Dashboard     │◀────│   ML Model      │◀────│   Features      │
│   (Streamlit)   │     │   (Training)    │     │   Engineering   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## 🚀 Installation

### Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de packages)
  
### Installation Windows (PowerShell)

```powershell
# Cloner le repository
git clone https://github.com/ayoubm11/CAN-2025-Edition.git
cd CAN-2025-Edition

# Créer l'environnement virtuel
py -3 -m venv .venv

# Activer l'environnement
.\.venv\Scripts\Activate.ps1

# Installer les dépendances
pip install -r requirements.txt
```

### Installation macOS/Linux

```bash
# Cloner le repository
git clone https://github.com/ayoubm11/CAN-2025-Edition.git
cd CAN-2025-Edition

# Créer l'environnement virtuel
python3 -m venv .venv

# Activer l'environnement
source .venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

## 💻 Utilisation

### Pipeline Complet

```powershell
# 1. Ingestion et EDA
py -3 src\ingest.py

# 2. ETL (nettoyage et transformation)
py -3 src\etl.py

# 3. Feature engineering
py -3 src\features.py

# 4. Entraînement du modèle
py -3 src\model.py

# 5. Évaluation
py -3 src\evaluate.py

# 6. Lancement du dashboard
streamlit run src/dashboard.py
```

### Dashboard Streamlit

Une fois le dashboard lancé, accédez à : `http://localhost:8501`

## 📊 Métriques Clés

| Métrique | Valeur |
|----------|--------|
| 🎮 **Matchs Analysés** | 3,685 |
| 🏴 **Équipes** | 26 |
| ⚽ **Joueurs Clés** | 24 |
| 🎯 **Ratio Moyen Buts** | 2.19/match |
| 📅 **Période** | 2000-2025 |

## 🎨 Dashboard

Le dashboard Streamlit offre :

- **📊 KPIs Principaux** — Métriques clés en temps réel
- **📈 Évolution Temporelle** — Graphiques d'évolution annuelle
- **🔍 Filtres Dynamiques** — Sélection d'équipes et périodes
- **🎯 Analyse Comparative** — Benchmarking d'équipes
- **📉 Distribution Statistique** — Box plots et distributions
- **🎭 Profils Radar** — Forces et faiblesses par équipe

## 📁 Structure du Projet

```
can2025-analysis/
│
├── data/
│   ├── raw/                    # Données brutes
│   ├── processed/              # Données transformées
│   └── features/               # Features engineerées
│
├── src/
│   ├── ingest.py              # Ingestion et EDA
│   ├── etl.py                 # Pipeline ETL
│   ├── features.py            # Feature engineering
│   ├── model.py               # Entraînement ML
│   ├── evaluate.py            # Évaluation
│   └── dashboard.py           # Interface Streamlit
│
├── models/                     # Modèles sauvegardés
├── reports/                    # Rapports générés
├── photo/                      # Images et visuels
│   └── head.png               # Header du projet
│
├── requirements.txt            # Dépendances Python
├── .env.example               # Configuration exemple
└── README.md                  # Documentation
```

## 🛠️ Technologies

### Langages et Frameworks
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

### Stack Technique

- **Data Manipulation** : Pandas, NumPy
- **Machine Learning** : Scikit-learn, XGBoost, LightGBM
- **Visualisation** : Matplotlib, Seaborn, Plotly
- **Dashboard** : Streamlit
- **Orchestration** : Prefect (optionnel)

## 🎯 Résultats

### Insights Principaux

- ✅ Identification des **facteurs de succès** des équipes performantes
- ✅ Analyse de l'**évolution stratégique** sur 25 ans
- ✅ **Patterns de victoire** clairement identifiés
- ✅ **Prédictions** basées sur l'historique et les tendances

### Performance du Modèle

Le modèle de Machine Learning développé permet de prédire avec une précision significative :

- Résultats de matchs (victoire/nul/défaite)
- Tendances de performance d'équipes
- Probabilités de progression dans la compétition

## 🗺️ Roadmap

### Court Terme (Q1-Q2 2026)
- [ ] Intégration données en temps réel
- [ ] Dashboard mobile responsive
- [ ] Système d'alertes automatiques

### Moyen Terme (2026-2027)
- [ ] Deep Learning (LSTM) pour prédictions
- [ ] Computer Vision sur vidéos de matchs
- [ ] API REST pour intégrations externes

### Long Terme (2027+)
- [ ] Plateforme collaborative cloud
- [ ] IA explicable pour décisions tactiques
- [ ] Jumeaux numériques d'équipes

## 🤝 Contribution

Les contributions sont les bienvenues ! Pour contribuer :

1. **Fork** le projet
2. Créez votre branche (`git checkout -b feature/AmazingFeature`)
3. Committez vos changements (`git commit -m 'Add AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrez une **Pull Request**

### Guidelines

- Code bien documenté
- Tests unitaires pour nouvelles fonctionnalités
- Respect des conventions Python (PEP 8)
- Commit messages clairs et descriptifs

## 📝 Licence

Ce projet est sous licence **MIT**. Voir le fichier `LICENSE` pour plus de détails.

## 🙏 Remerciements

- **CAF** pour les données de la Coupe d'Afrique des Nations
- **SBI** pour l'organisation du challenge


<div align="center">
  
  **⭐ Si ce projet vous a aidé, n'hésitez pas à lui donner une étoile !**
  
  Made with ❤️ for African Football
  
  ![Visitors](https://visitor-badge.laobi.icu/badge?page_id=votre-username.can2025-analysis)
  
</div>
