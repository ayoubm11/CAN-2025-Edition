"""Streamlit dashboard for local model predictions and simple reports."""
import streamlit as st
from pathlib import Path
import pandas as pd
import joblib
import json
import plotly.express as px


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "matches_processed.csv"
MODEL_DIR = ROOT / "models"


st.set_page_config(layout="wide")
st.title("CAN 2025 — Dashboard local (Streamlit)")


def compute_team_stats(df):
    # matches played and wins per team
    teams = {}
    for _, r in df.iterrows():
        h = r['home_team']
        a = r['away_team']
        res = r['result']
        teams.setdefault(h, {'played': 0, 'wins': 0})
        teams.setdefault(a, {'played': 0, 'wins': 0})
        teams[h]['played'] += 1
        teams[a]['played'] += 1
        if res == 'home':
            teams[h]['wins'] += 1
        elif res == 'away':
            teams[a]['wins'] += 1
    stats = pd.DataFrame([{'team': t, 'played': v['played'], 'wins': v['wins'], 'win_rate': v['wins']/v['played'] if v['played']>0 else 0} for t, v in teams.items()])
    return stats.sort_values('win_rate', ascending=False)


if not PROCESSED.exists():
    st.warning("Données traitées introuvables. Exécutez src/ingest.py puis src/etl.py et src/features.py.")
else:
    df = pd.read_csv(PROCESSED)

    # Top row: KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Matches", int(len(df)))
    with col2:
        st.metric("Seasons", int(df['year'].nunique()) if 'year' in df.columns else 0)
    with col3:
        st.metric("Teams", int(pd.concat([df['home_team'], df['away_team']]).nunique()))
    with col4:
        avg_goals = (df['home_score'].astype(float).fillna(0) + df['away_score'].astype(float).fillna(0)).mean()
        st.metric("Avg goals per match", f"{avg_goals:.2f}")

    st.markdown("---")

    # Layout: left controls, right visuals
    left, right = st.columns([1, 2])

    with left:
        st.subheader("Filtres")
        years = sorted(df['year'].dropna().unique().astype(int).tolist()) if 'year' in df.columns else []
        if years:
            y_min = int(min(years))
            y_max = int(max(years))
            yr_range = st.slider('Années', min_value=y_min, max_value=y_max, value=(y_min, y_max))
            df = df[(df['year'] >= yr_range[0]) & (df['year'] <= yr_range[1])]
        team_list = sorted(pd.concat([df['home_team'], df['away_team']]).unique())
        selected_team = st.selectbox('Team focus (optionnel)', options=['All'] + team_list)

    with right:
        st.subheader("Distribution des résultats")
        res_counts = df['result'].value_counts().reset_index()
        res_counts.columns = ['result', 'count']
        fig1 = px.bar(res_counts, x='result', y='count', color='result', title='Counts by result')
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Distribution de l'écart de buts")
        if 'home_score' in df.columns and 'away_score' in df.columns:
            goal_diff = df['home_score'].fillna(0) - df['away_score'].fillna(0)
            gd_df = goal_diff.to_frame(name='goal_diff')
            fig2 = px.histogram(gd_df, x='goal_diff', nbins=30, title='Goal difference (home - away)', color_discrete_sequence=['#17becf'])
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Matches per year")
        if 'year' in df.columns:
            yearly = df.groupby('year').size().reset_index(name='matches')
            fig3 = px.line(yearly, x='year', y='matches', markers=True, title='Matches per year')
            st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # Team-level insights
    st.subheader('Team performance')
    stats = compute_team_stats(df)
    if selected_team and selected_team != 'All':
        st.write(stats[stats['team'] == selected_team])
        # head-to-head simple: show matches where selected team participated
        hh = df[(df['home_team'] == selected_team) | (df['away_team'] == selected_team)].copy()
        st.write(hh[['date','home_team','away_team','home_score','away_score','result']].sort_values('date', ascending=False).head(50))
    else:
        st.write('Top 10 teams by win rate (min 5 matches)')
        top = stats[stats['played']>=5].head(10).reset_index(drop=True)
        st.dataframe(top)
        fig4 = px.bar(top, x='win_rate', y='team', orientation='h', title='Top teams by win rate', color='win_rate', color_continuous_scale='Purples')
        fig4.update_yaxes(autorange='reversed')
        st.plotly_chart(fig4, use_container_width=True)

    # Prediction widget (kept)
    st.markdown('---')
    st.subheader('Prédiction d\'un match')
    st.sidebar.header("Prédiction d'un match")
    home = st.sidebar.selectbox('Home team', options=sorted(df['home_team'].unique()))
    away = st.sidebar.selectbox('Away team', options=sorted(df['away_team'].unique()))
    year = st.sidebar.number_input('Year', min_value=2000, max_value=2030, value=2025)
    home_goals = st.sidebar.number_input('Home goals (example)', min_value=0, max_value=20, value=0)
    away_goals = st.sidebar.number_input('Away goals (example)', min_value=0, max_value=20, value=0)

    model_path = MODEL_DIR / 'can2025_model.joblib'
    cols_path = MODEL_DIR / 'feature_columns.json'

    if model_path.exists() and cols_path.exists():
        model = joblib.load(model_path)
        cols = json.loads(cols_path.read_text())['columns']

        if st.sidebar.button('Predict'):
            row = {c: 0 for c in cols}
            row['year'] = year
            row['home_goals'] = home_goals
            row['away_goals'] = away_goals
            row['goal_diff'] = home_goals - away_goals
            home_col = f'home_{home}'
            away_col = f'away_{away}'
            if home_col in row:
                row[home_col] = 1
            if away_col in row:
                row[away_col] = 1

            X = pd.DataFrame([row])
            pred = model.predict(X)[0]
            st.success(f'Prediction: {pred}')
    else:
        st.info('Modèle absent — exécutez src/model.py pour entraîner.')
