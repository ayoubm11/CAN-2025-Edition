"""Tableau de bord Streamlit (FR) — visualisations avancées pour la CAN 2025."""
import streamlit as st
from pathlib import Path
import pandas as pd
import joblib
import json
import plotly.express as px
import pycountry
import numpy as np
import unicodedata
import difflib


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "matches_processed.csv"
MODEL_DIR = ROOT / "models"

# Liste des 24 pays participants à la CAN 2025 (utiliser ces noms exactement)
PARTICIPANTS = [
    'Maroc', 'Burkina Faso', 'Cameroun', 'Algérie', 'République démocratique du Congo', 'Sénégal',
    'Égypte', 'Angola', 'Guinée équatoriale', "Côte d’Ivoire", 'Ouganda', 'Afrique du Sud',
    'Gabon', 'Tunisie', 'Nigeria', 'Zambie', 'Mali', 'Zimbabwe', 'Comores', 'Soudan',
    'Bénin', 'Tanzanie', 'Botswana', 'Mozambique'
]

# Normalisation et mapping des noms d'équipe pour gérer différences d'orthographe/accents
def normalize(name: str) -> str:
    if not isinstance(name, str):
        return ''
    s = name.replace("’", "'").replace('–','-')
    s = s.replace('`', "'")
    s = s.strip().lower()
    # remove diacritics
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    # remove punctuation except spaces
    s = ''.join(ch for ch in s if ch.isalnum() or ch.isspace())
    s = ' '.join(s.split())
    return s

# build normalized participant map
NORMALIZED_PARTICIPANTS = {normalize(p): p for p in PARTICIPANTS}

def map_name_to_participant(name: str):
    n = normalize(name)
    if not n:
        return None
    if n in NORMALIZED_PARTICIPANTS:
        return NORMALIZED_PARTICIPANTS[n]
    # Additional alias mapping to handle English names and common variants
    # Keys must be normalized forms
    ALIASES = {
        'morocco': 'Maroc',
        'ivory coast': "Côte d’Ivoire",
        'cote d ivoire': "Côte d’Ivoire",
        'cote divoire': "Côte d’Ivoire",
        'dr congo': 'République démocratique du Congo',
        'democratic republic of the congo': 'République démocratique du Congo',
        'democratic republic of congo': 'République démocratique du Congo',
        'equatorial guinea': 'Guinée équatoriale',
        'south africa': 'Afrique du Sud',
        'egypt': 'Égypte',
        'tunisia': 'Tunisie',
        'zambia': 'Zambie',
        'comoros': 'Comores',
        'sudan': 'Soudan',
        'benin': 'Bénin',
        'nigeria': 'Nigeria',
        'uganda': 'Ouganda',
        'angola': 'Angola',
        'gabon': 'Gabon',
        'botswana': 'Botswana',
        'mozambique': 'Mozambique',
        'zimbabwe': 'Zimbabwe',
        'burkina faso': 'Burkina Faso',
        'cameroon': 'Cameroun',
        'mali': 'Mali',
        'tanzania': 'Tanzanie'
    }
    # normalize alias keys once
    ALIASES_NORM = {normalize(k): v for k, v in ALIASES.items()}
    if n in ALIASES_NORM:
        return ALIASES_NORM[n]
    # fuzzy match against normalized participant keys
    keys = list(NORMALIZED_PARTICIPANTS.keys())
    match = difflib.get_close_matches(n, keys, n=1, cutoff=0.75)
    if match:
        return NORMALIZED_PARTICIPANTS[match[0]]
    return None


st.set_page_config(layout="wide", page_title="CAN 2025 — Dashboard")
st.title("CAN 2025 — Tableau de bord local")


def compute_team_stats(df):
    teams = {}
    for _, r in df.iterrows():
        h = r.get('home_team')
        a = r.get('away_team')
        res = r.get('result')
        if pd.isna(h) or pd.isna(a):
            continue
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


def country_to_flag(country_name: str) -> str:
    try:
        c = pycountry.countries.search_fuzzy(country_name)[0]
        code = c.alpha_2
    except Exception:
        return ''
    try:
        return ''.join(chr(ord(ch) + 127397) for ch in code.upper())
    except Exception:
        return ''


if not PROCESSED.exists():
    st.warning("Données traitées introuvables. Exécutez `src/ingest.py`, `src/etl.py` puis `src/features.py`.")
else:
    df = pd.read_csv(PROCESSED)
    # Basic normalization
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Filtrer seulement les matches entre les 24 pays participants (si présents dans les données)
    orig_df = df.copy()
    if 'home_team' in df.columns and 'away_team' in df.columns:
        # map raw names to canonical participant names where possible
        orig_df['home_mapped'] = orig_df['home_team'].apply(map_name_to_participant)
        orig_df['away_mapped'] = orig_df['away_team'].apply(map_name_to_participant)
        mask = orig_df['home_mapped'].notnull() & orig_df['away_mapped'].notnull()
        if mask.any():
            df = orig_df[mask].copy().reset_index(drop=True)
            # replace team names by canonical participant names for consistency
            df['home_team'] = df['home_mapped']
            df['away_team'] = df['away_mapped']
        else:
            st.warning("Aucun match trouvé entre les 24 pays participants dans les données — affichage des données d'origine.")
            df = orig_df

    # déterminer la liste des équipes autorisées (intersection des PARTICIPANTS et des équipes présentes dans les données)
    if 'home_team' in orig_df.columns and 'away_team' in orig_df.columns:
        all_teams_in_data = pd.concat([orig_df['home_team'], orig_df['away_team']]).dropna().unique().tolist()
        # map data teams to participants where possible
        mapped = [map_name_to_participant(t) or t for t in all_teams_in_data]
        allowed_teams = [t for t in PARTICIPANTS if t in mapped]
        if not allowed_teams:
            allowed_teams = PARTICIPANTS.copy()
    else:
        allowed_teams = PARTICIPANTS.copy()

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Matches", int(len(df)))
    with col2:
        st.metric("Saisons", int(df['year'].nunique()) if 'year' in df.columns else 0)
    with col3:
        st.metric("Équipes", int(pd.concat([df['home_team'], df['away_team']]).nunique()))
    with col4:
        avg_goals = (df['home_score'].fillna(0).astype(float) + df['away_score'].fillna(0).astype(float)).mean()
        st.metric("Moy. buts / match", f"{avg_goals:.2f}")

    st.markdown("---")

    # Layout: controls + main
    left, main = st.columns([1, 3])

    with left:
        st.subheader("Filtres")
        years = sorted(df['year'].dropna().unique().astype(int).tolist()) if 'year' in df.columns else []
        if years:
            yr_range = st.slider('Années', min_value=int(min(years)), max_value=int(max(years)), value=(int(min(years)), int(max(years))))
            df = df[(df['year'] >= yr_range[0]) & (df['year'] <= yr_range[1])]
        present_teams = pd.concat([df['home_team'], df['away_team']]).dropna().unique().tolist()
        # ne montrer que les équipes autorisées (participants)
        teams = sorted([t for t in present_teams if t in allowed_teams])
        if not teams:
            teams = sorted(allowed_teams)
        selected_team = st.selectbox('Équipe (optionnel)', options=['Toutes'] + teams)
        show_flags = st.checkbox('Afficher les drapeaux', value=True)
        chart_choice = st.selectbox('Graphique principal', options=['Résumé','Buts marqués/encaissés','Différence de buts','Performance Domicile vs Extérieur','Heatmap tournoi','Profil équipe (radar)'])

    # Main visuals
    with main:
        st.subheader('Résumé')

        # Répartition résultats (pie)
        if 'result' in df.columns:
            res_counts = df['result'].value_counts().reset_index()
            res_counts.columns = ['result', 'count']
            fig_res = px.pie(res_counts, names='result', values='count', title='Répartition des résultats')
            st.plotly_chart(fig_res, use_container_width=True)

        # Evolution par année
        if 'year' in df.columns:
            yearly = df.groupby('year').size().reset_index(name='matches')
            fig_year = px.line(yearly, x='year', y='matches', markers=True, title='Matches par année')
            st.plotly_chart(fig_year, use_container_width=True)

        # Boxplot des scores
        if 'home_score' in df.columns and 'away_score' in df.columns:
            box_df = pd.DataFrame({'domicile': df['home_score'].fillna(0), 'extérieur': df['away_score'].fillna(0)})
            fig_box = px.box(box_df.melt(var_name='type', value_name='buts'), x='type', y='buts', title='Distribution des buts (domicile vs extérieur)')
            st.plotly_chart(fig_box, use_container_width=True)

        st.markdown('---')

        # Team level
        st.subheader('Performance des équipes')
        stats = compute_team_stats(df)

        if selected_team != 'Toutes':
            team = selected_team
            flag = country_to_flag(team) if show_flags else ''
            st.markdown(f"### {flag} {team}")
            hh = df[(df['home_team'] == team) | (df['away_team'] == team)].copy()
            if not hh.empty:
                played = len(hh)
                scored = hh.apply(lambda r: r['home_score'] if r['home_team']==team else r['away_score'], axis=1).sum()
                conceded = hh.apply(lambda r: r['away_score'] if r['home_team']==team else r['home_score'], axis=1).sum()
                wins = hh.apply(lambda r: (r['result']=='home' and r['home_team']==team) or (r['result']=='away' and r['away_team']==team), axis=1).sum()
                st.write({'Matchs': int(played), 'Buts marqués': int(scored), 'Buts encaissés': int(conceded), 'Victoires': int(wins), 'Taux de victoire': f"{wins/played:.2f}" if played>0 else '0'})

                # Radar
                st.subheader('Profil équipe (radar)')
                metrics = {'Buts marqués': scored, 'Buts encaissés': conceded, 'Matchs': played, 'Victoires': wins, 'Taux victoire': wins/played if played>0 else 0}
                radar_df = pd.DataFrame({'metric': list(metrics.keys()), 'value': list(metrics.values())})
                fig_radar = px.line_polar(radar_df, r='value', theta='metric', line_close=True, title=f'Profil {team}')
                st.plotly_chart(fig_radar, use_container_width=True)

                # Recent matches
                cols_show = [c for c in ['date','home_team','away_team','home_score','away_score','result'] if c in hh.columns]
                st.write(hh[cols_show].sort_values('date', ascending=False).head(30))
            else:
                st.info('Aucun match trouvé pour cette équipe dans la période sélectionnée.')
        else:
            top = stats[stats['played']>=5].head(10).reset_index(drop=True)
            if show_flags:
                top['flag'] = top['team'].apply(country_to_flag)
                top['équipe'] = top['flag'] + ' ' + top['team']
                display_cols = ['équipe','played','wins','win_rate']
            else:
                top['équipe'] = top['team']
                display_cols = ['équipe','played','wins','win_rate']
            st.dataframe(top[display_cols].rename(columns={'played':'Matchs','wins':'Victoires','win_rate':'Taux victoire'}))
            fig_top = px.bar(top, x='win_rate', y='team', orientation='h', title='Top équipes par taux de victoire', color='win_rate', color_continuous_scale='Purples')
            fig_top.update_yaxes(autorange='reversed')
            st.plotly_chart(fig_top, use_container_width=True)

        # Additional selected chart
        if chart_choice == 'Buts marqués/encaissés':
            st.subheader('Buts marqués vs encaissés par équipe')
            g1 = df.groupby('home_team')['home_score'].sum().rename('home_scored') if 'home_score' in df.columns else pd.Series()
            g2 = df.groupby('away_team')['away_score'].sum().rename('away_scored') if 'away_score' in df.columns else pd.Series()
            scored = pd.concat([g1, g2], axis=1).fillna(0)
            scored['scored'] = scored.sum(axis=1)
            c1 = df.groupby('home_team')['away_score'].sum().rename('home_conceded')
            c2 = df.groupby('away_team')['home_score'].sum().rename('away_conceded')
            conceded = pd.concat([c1, c2], axis=1).fillna(0)
            conceded['conceded'] = conceded.sum(axis=1)
            agg = pd.DataFrame({'team': scored.index, 'scored': scored['scored'].values, 'conceded': conceded.loc[scored.index,'conceded'].values})
            fig_sc = px.bar(agg.melt(id_vars='team', value_vars=['scored','conceded']), x='team', y='value', color='variable', barmode='group', title='Buts marqués vs encaissés')
            st.plotly_chart(fig_sc, use_container_width=True)

        if chart_choice == 'Différence de buts':
            st.subheader('Différence de buts par match')
            if 'home_score' in df.columns and 'away_score' in df.columns:
                df['goal_diff'] = df['home_score'] - df['away_score']
                sample = df.sort_values('date', ascending=False).head(200)
                fig_gd = px.scatter(sample, x='date', y='goal_diff', color='result', title='Différence de buts par match')
                st.plotly_chart(fig_gd, use_container_width=True)

        if chart_choice == 'Performance Domicile vs Extérieur':
            st.subheader('Performance domicile vs extérieur')
            dom = df[df['result']=='home'].groupby('home_team').size().rename('home_wins')
            ext = df[df['result']=='away'].groupby('away_team').size().rename('away_wins')
            perf = pd.concat([dom, ext], axis=1).fillna(0)
            perf = perf.reset_index().rename(columns={'index':'team'})
            perf.columns = ['team','home_wins','away_wins']
            perf_m = perf.melt(id_vars='team', value_vars=['home_wins','away_wins'], var_name='location', value_name='wins')
            fig_perf = px.bar(perf_m, x='team', y='wins', color='location', barmode='group', title='Victoires domicile vs extérieur')
            st.plotly_chart(fig_perf, use_container_width=True)

        if chart_choice == 'Heatmap tournoi':
            st.subheader('Heatmap des buts par tournoi (si disponible)')
            if 'tournament' in df.columns:
                df['total_goals'] = df['home_score'].fillna(0) + df['away_score'].fillna(0)
                heat = df.groupby(['tournament','year'])['total_goals'].sum().reset_index()
                heat_pivot = heat.pivot(index='tournament', columns='year', values='total_goals').fillna(0)
                fig_hm = px.imshow(heat_pivot, labels=dict(x='Année', y='Tournoi', color='Buts'), aspect='auto', title='Buts par tournoi et par année')
                st.plotly_chart(fig_hm, use_container_width=True)
            else:
                st.info('Aucune colonne `tournament` disponible dans les données.')

    # Prediction (sidebar)
    st.sidebar.header("Prédiction d'un match")
    # Utiliser la liste `teams` (définie plus haut) pour éviter d'afficher toute la liste des pays
    home = st.sidebar.selectbox('Équipe domicile', options=teams)
    away = st.sidebar.selectbox('Équipe extérieur', options=teams)
    year = st.sidebar.number_input('Année', min_value=2000, max_value=2030, value=2025)
    home_goals = st.sidebar.number_input('Buts domicile (exemple)', min_value=0, max_value=20, value=0)
    away_goals = st.sidebar.number_input('Buts extérieur (exemple)', min_value=0, max_value=20, value=0)

    model_path = MODEL_DIR / 'can2025_model.joblib'
    cols_path = MODEL_DIR / 'feature_columns.json'

    if model_path.exists() and cols_path.exists():
        model = joblib.load(model_path)
        cols = json.loads(cols_path.read_text())['columns']
        if st.sidebar.button('Prédire'):
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
            st.sidebar.success(f'Prediction: {pred}')
    else:
        st.sidebar.info('Modèle absent — exécutez `src/model.py` pour entraîner.')
