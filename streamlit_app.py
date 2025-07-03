#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------
# Chargement des librairies
# -----------------------------
import streamlit as st

import pandas as pd
import numpy as np
from scipy.stats import randint

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, f1_score, classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, recall_score, precision_score)
from sklearn.utils.multiclass import unique_labels

from imblearn.metrics import classification_report_imbalanced
import xgboost as xgb

import shap

import pickle
import joblib
import cloudpickle

import os
import time
import requests
import gdown
import urllib.request
from io import StringIO
from datetime import datetime
from dateutil.relativedelta import relativedelta

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# -----------------------------
# Configs et chemins
# -----------------------------
st.set_page_config(page_title="Rain in Australia", layout="wide")


DATA_PATH = "data"
DATASET_PATH = "dataset"
MODELS_PATH = "models"
SCALER_PATH = "dico_scaler" #dico, scaler, imputer

MODEL_LIST = {
    "XGBoost Final": "final_xgb_model_pluie.joblib",
    "Stacking v1": "stacking_model_25features.pkl",
    "Stacking Amélioré": "stacking_ameliore_25features.pkl",
    "Stacking Simple": "stacking_simple_model.pkl",
    "Voting Classifier": "voting_model_25features.pkl"
}

MODEL_LIST_Non_temporel = {
    "Régression logistique": "LogReg_X_train_normal_model_and_threshold.joblib",
    "XGB Classifier": "XGBClassifier_X_train_model_and_threshold.joblib",
    "RNN": "RNN_ABO_X_scaled_normal_model_and_threshold.joblib"}

#Bypass lfs des fichiers lourds (hébergés sur Drive)
MODEL_DRIVE_IDS = {
    "Stacking v1": "1blJeynAHvbVDGM_55CGNLdE2OEvayN-u",
    "Stacking Amélioré": "1guY89tbCJy3s5_t3D30F60e7IHxC2saU"
}

# -----------------------------
#  CACHE
# -----------------------------

# With `ttl`, objects in cache are removed after 24 hours.
@st.cache_data(ttl=24*3600, max_entries=1)
def load_data():
    df = pd.read_csv(os.path.join(DATA_PATH, "weatherAUS.csv"))
    liste = list(df.columns) # servira pour créer les colonnes des nouvelles données.
    gps = pd.read_csv(os.path.join(SCALER_PATH, "localisations_gps.csv"))
    climat = pd.read_csv(os.path.join(SCALER_PATH, "climat_mapping.csv"))
    df = df.merge(gps, on="Location", how="left")
    df = df.merge(climat, on="Location", how="left")
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    return df, liste

@st.cache_data(ttl=24*3600, max_entries=2)
def load_dataset(name):#utile en cache? appelé une seule fois
    if name == "Ancien test set":
        X = pd.read_csv(os.path.join(DATASET_PATH,"X_test_reduit.csv"), index_col=0)
        y = pd.read_csv(os.path.join(DATASET_PATH,"y_test.csv"), index_col=0).squeeze()
    else:
        X = pd.read_csv(os.path.join(DATASET_PATH,"data_2024-25_reduit.csv"), index_col=0)
        y = pd.read_csv(os.path.join(DATASET_PATH,"target_2024-25.csv"), index_col=0).squeeze()
    y = y.astype(int)
    return X, y

@st.cache_data(ttl=24*3600, max_entries=2)
def load_old_dataset():#utile en cache? appelé une seule fois
  X = pd.read_csv(os.path.join(DATASET_PATH, "X_test_reduit.csv"), index_col=0)
  y = pd.read_csv(os.path.join(DATASET_PATH, "y_test.csv"), index_col=0).squeeze()
  y = y.astype(int)
  return X, y

# Cache pour les modèles entrainés
# Dossier de destination local
MODEL_CACHE_PATH = "models_from_drive"
os.makedirs(MODEL_CACHE_PATH, exist_ok=True)

@st.cache_resource(show_spinner="Téléchargement du modèle depuis Google Drive...")
def load_model_from_drive(model_name):
    file_id = MODEL_DRIVE_IDS[model_name]
    output_path = os.path.join(MODEL_CACHE_PATH, f"{model_name.replace(' ', '_')}.pkl")
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    with open(output_path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_model(name):
    if name in MODEL_DRIVE_IDS:
        try:
            return load_model_from_drive(name)
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement depuis Drive ({name}) : {e}")
            return None
    else:
        try:
            return joblib.load(os.path.join(MODELS_PATH, MODEL_LIST[name]))
        except FileNotFoundError:
            st.error(f"❌ Modèle introuvable : {MODEL_LIST[name]}")
            return None

@st.cache_resource
def load_model_non_temporel(name): #utile en cache? appelé une seule fois
    return joblib.load(os.path.join(MODELS_PATH, MODEL_LIST_Non_temporel[name]))

@st.cache_resource
def load_features():
  return joblib.load(os.path.join(MODELS_PATH, "xgb_25features_list.joblib"))

#Cache pour les imputers, scalers, dico
@st.cache_resource
def load_cloudpickle(imputer): #sert pour le preprocessing ABO : complétion NaN cloud et autres
    with open(os.path.join(SCALER_PATH, imputer), "rb") as f:
        return cloudpickle.load(f)


@st.cache_data
def load_pickle(dico): # dico stations pour données actuelles
    with open(os.path.join(SCALER_PATH, dico), "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_joblib(scaler): #scalers et knn_model de Florent
  return joblib.load(os.path.join(SCALER_PATH, scaler))

# -------------------------------
# FONCTIONS
# -------------------------------
# -------------------------------
# AFFICHAGE DES RÉSULTATS
# -------------------------------
def afficher_resultats(model, X, y, label, seuil=0.5):
    proba = model.predict_proba(X)[:, 1]
    y_pred = (proba >= seuil).astype(int)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, proba)
    report = classification_report(y, y_pred, output_dict=True)
    cm = confusion_matrix(y, y_pred)

    st.markdown(f"### Résultats sur : **{label}**")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### Rapport de classification")
        st.dataframe(pd.DataFrame(report).T.round(2))

        st.markdown("#### Matrice de confusion")
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues")
        fig_cm.update_layout(xaxis=dict(title="Prédit", tickvals=[0, 1], ticktext=["0", "1"]),
                         yaxis=dict(title="Réel", tickvals=[0, 1], ticktext=["0", "1"]))
        st.plotly_chart(fig_cm, use_container_width=True, key=f"plot_cm_{label}")

    with col2:
        st.metric("F1 Score", f"{f1:.2f}")
        st.metric("ROC AUC", f"{auc:.2f}")

        fpr, tpr, _ = roc_curve(y, proba)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC Curve"))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")))
        fig_roc.update_layout(title="Courbe ROC", xaxis_title="Faux positifs", yaxis_title="Vrais positifs")
        st.plotly_chart(fig_roc, use_container_width=True, key=f"plot_roc_{label}")
# -------------------------------
# AFFICHAGE DES RÉSULTATS
# -------------------------------
def affichage_resultas_donnees_actuelles (X_test, y, y_pred):
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, pos_label=1)
    recall = recall_score(y, y_pred, pos_label=1)
    precision = precision_score(y, y_pred, pos_label=1)

    st.success("Prédictions effectuées")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎯 Accuracy", f"{accuracy*100:.2f}%")
    col2.metric("🌧️ F1", f"{f1:.2f}")
    col3.metric("🌧️ Recall", f"{recall:.2f}")
    col4.metric("🌧️ Precision", f"{precision:.2f}")

    st.info("Pour plus de détail ↓")
    with st.expander(" Rapport complet"):
        labels = unique_labels(y, y_pred)
        report = classification_report_imbalanced(
            y, y_pred,
            labels=labels,
            target_names=["Pas de pluie", "Pluie"][:len(labels)],
            output_dict=True
        )
        report_df = pd.DataFrame(report).transpose().round(2)
        st.dataframe(report_df)

    cm = confusion_matrix(y, y_pred)
    st.subheader("Matrice de confusion")
    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues")
    fig_cm.update_layout(xaxis=dict(title="Prédit", tickvals=[0, 1], ticktext=["0", "1"]),
                         yaxis=dict(title="Réel", tickvals=[0, 1], ticktext=["0", "1"]))
    st.plotly_chart(fig_cm, use_container_width=True)

    # Zoom sur les mauvaises prédictions
    df_result = pd.DataFrame({
        "RainTomorrow": y,
        "Prediction": y_pred
    }, index=X_test.index)
    df_result["Correct"] = df_result["RainTomorrow"] != df_result["Prediction"]
    st.write("##### Données où le modèle a mal prédit")
    X_test_bad = X_test[df_result["Correct"]]
    st.dataframe(X_test_bad.head())


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 1. Charger les données
df, liste_colonne_df  = load_data()
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 2. Définir la structure
st.title("Projet de classification binaire sur la pluie en Australie") # sera répercuté sur toutes les pages du Streamlit
st.sidebar.title("Sommaire")
pages=["Analyse exploratoire", "Comparaison des modèles", "Datas actuelles", "Conclusion"]
page=st.sidebar.radio("Aller vers", pages)
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# -----------------------------
# Page 1 : EDA
# -----------------------------
if page == "Analyse exploratoire":
# -----------------------------
# En-tête
# -----------------------------
    st.title("Analyse exploratoire des données météo en Australie")
    st.markdown("""
    Bienvenue dans notre projet de data science réalisé dans le cadre de la formation **DataScientest**.

    **Objectif :** Prédire s’il pleuvra demain en Australie (RainTomorrow) à partir de données météo historiques.

    **Données :**
    - Source : Bureau of Meteorology (Australie) via Kaggle
    - Période : 2007 à 2017
    - Nombre de stations : {:d}
    - Nombre d'observations : {:,}
    - Nombre de colonnes : {:d}
    """.format(df['Location'].nunique(), len(df), df.shape[1]-3)) #-3 car a ajouté climat, latitude et longitude dès le chargement du df


    # -----------------------------
    # 1. Carte des stations
    # -----------------------------
    st.subheader("Carte des stations météo")
    df_map = df[['Location', 'Latitude', 'Longitude', 'Climate']].dropna().drop_duplicates(subset="Location")
    fig_map = px.scatter_mapbox(
        df_map,
        lat="Latitude",
        lon="Longitude",
        color="Climate",
        hover_name="Location",
        zoom=2.5,
        height=550
    )
    fig_map.update_layout(
        mapbox_style="open-street-map",
        mapbox_center={"lat": -25.0, "lon": 135.0}
    )
    st.plotly_chart(fig_map)

    # -----------------------------
    # 2. Distribution de la variable cible & pluviométrie moyenne
    # -----------------------------
    tab1, tab2 = st.tabs(["📊 Distribution de la variable cible", "🌧️ Pluviométrie moyenne"])

    # -----------------------------
    # 2.1 Distributions
    # -----------------------------
    with tab1 :
      col1, col2 = st.columns(2)

      with col1:
          # Distribution pluie/pas pluie
          rain_counts = df['RainTomorrow'].value_counts().rename(index={'No': 'Pas de pluie', 'Yes': 'Pluie'})
          fig_pie = px.pie(
              values=rain_counts.values,
              names=rain_counts.index,
              title="Répartition pluie/pas pluie",
              color_discrete_sequence=px.colors.qualitative.Pastel2
          )
          st.plotly_chart(fig_pie)

      with col2:
          # Distribution par climat
          df_plot = df[df['RainTomorrow'].notna()]
          rain_by_climate = df_plot.groupby(['Climate', 'RainTomorrow']).size().reset_index(name='count')
          fig_bar = px.bar(
              rain_by_climate,
              x="Climate",
              y="count",
              color="RainTomorrow",
              barmode="group",
              title="Distribution par climat"
          )
          st.plotly_chart(fig_bar)

    # -----------------------------
    # 2.2 Pluviométrie
    # -----------------------------
    with tab2 :

      col1, col2 = st.columns([2,3])

      # -----------------------------
      # 2.2.1 Pluviométrie mensuelle
      # -----------------------------
      with col1 :
        df_month = df.copy()
        df_month["Month"] = df_month["Date"].dt.month
        rain_by_month = df_month.groupby("Month")["Rainfall"].mean().reset_index()
        fig_month = px.bar(
            rain_by_month,
            x="Month",
            y="Rainfall",
            title=" Pluviométrie moyenne par mois",
            labels={"Rainfall": "mm de pluie"},
            color="Rainfall",
            color_continuous_scale="Blues")
        fig_month.update_xaxes(
            tickmode="linear",
            tick0=1,    # point de départ Janvier
            dtick=3,    # intervalle de 3 mois
            title_text="Mois")

        st.plotly_chart(fig_month)

      # -----------------------------
      # 2.2.2. Pluviométrie par station
      # -----------------------------
      with col2 :
        df_pluie = df.groupby("Location").agg({
            "Rainfall": "mean",
            "Latitude": "first",
            "Longitude": "first",
            "Climate": "first"
        }).reset_index()

        fig_pluie = px.scatter_mapbox(
            df_pluie,
            title = "Pluviométrie moyenne par station",
            lat="Latitude",
            lon="Longitude",
            size="Rainfall",
            color="Rainfall",
            hover_name="Location",
            hover_data=["Climate"],
            color_continuous_scale="Blues",
            zoom=2.5,
            height=600
        )
        fig_pluie.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig_pluie)

    # -----------------------------
    # 3. Analyse par station : Carte de visite : station météo
    # -----------------------------
    st.divider()

    st.subheader("Carte de visite d'une station météo")

    station = st.selectbox("Choisir une station :", sorted(df["Location"].dropna().unique()))
    df_station = df[df["Location"] == station].copy()

    col1, col2 = st.columns([1,1])

    col1.metric("Climat", df_station["Climate"].mode()[0])

    part_pluie = df_station['RainTomorrow'].value_counts(normalize=True)
    pourcentage_rain_yes = part_pluie.get("Yes", 0) * 100
    col2.metric("% Pluie demain", value=f"{pourcentage_rain_yes:.2f} %")


    tab1, tab2, tab3, tab4 = st.tabs(["🌍 Station", "🌡️ Températures", "💨 Vent", "🌧️ Pluviométrie"])
    # ----------------------------
    # Colonnes principales : Climat + Coordonnées + Mini-carte
    # ----------------------------
    with tab1:
      col1, col2 = st.columns([1,1])

      with col1:
          st.metric("Climat", df_station["Climate"].mode()[0])

          coord = f"{df_station['Latitude'].iloc[0]:.2f}, {df_station['Longitude'].iloc[0]:.2f}"
          st.metric("Coordonnées", coord)

      with col2:
          st.markdown("#### Mini-carte de localisation")

          df_map_station = df_station[["Location", "Latitude", "Longitude"]].drop_duplicates()

          fig_mini = px.scatter_mapbox(
              df_map_station,
              lat="Latitude",
              lon="Longitude",
              hover_name="Location",
              zoom=2,
              height=200
          )
          fig_mini.update_layout(
              mapbox_style="carto-positron",
              mapbox_center={"lat": -25.0, "lon": 133.0},
              margin={"r": 0, "t": 0, "l": 0, "b": 0}
          )
          # 🔧 Mise à jour de la taille du point
          fig_mini.update_traces(marker=dict(size=14))


          st.plotly_chart(fig_mini, use_container_width=True)

    # ----------------------------
    # Températures
    # ----------------------------
    with tab2:
      col1, col2, col3 = st.columns([3,1,1])

      # ----------------------------
      # Températures moyennes par saison
      # ----------------------------
      with col1 :
        df_station["Month"] = df_station["Date"].dt.month
        df_station["Saison"] = df_station["Month"].map({
            12: "Été", 1: "Été", 2: "Été",
            3: "Automne", 4: "Automne", 5: "Automne",
            6: "Hiver", 7: "Hiver", 8: "Hiver",
            9: "Printemps", 10: "Printemps", 11: "Printemps"
        })

        temp_saison = (
            df_station
            .groupby("Saison")[["MinTemp", "MaxTemp"]]
            .mean()
            .reindex(["Été", "Automne", "Hiver", "Printemps"])
        )

        st.markdown("🌡️ **Températures moyennes par saison**")
        st.dataframe(temp_saison.style.format("{:.1f} °C"))

      # ----------------------------
      # Températures extrêmes
      # ----------------------------
      min_temp = df_station["MinTemp"].min()
      min_date = df_station.loc[df_station["MinTemp"].idxmin(), "Date"]
      max_temp = df_station["MaxTemp"].max()
      max_date = df_station.loc[df_station["MaxTemp"].idxmax(), "Date"]

      col2.metric(label = "❄️ Record Temp. min", value = f"{min_temp:.1f} °C", delta = f"📅 {min_date.date()}", delta_color="off")
      col3.metric(label = "🔥 Record Temp. max", value = f"{max_temp:.1f} °C", delta = f"📅 {max_date.date()}", delta_color="off")

    # ----------------------------
    # Vent dominant
    # ----------------------------
    with tab3:
      vent_dominant_pm = df_station["WindDir3pm"].mode()[0]
      vent_dominant_am = df_station["WindDir9am"].mode()[0]

      #graph radar
      WIND_DIRS = ['E','ENE','NE','NNE','N','NNW','NW','WNW','W','WSW','SW','SSW','S','SSE','SE','ESE']
      PLOTS = [
          ("RainTomorrow", "WindDir9am",    "9 h"),
          ("RainTomorrow", "WindDir3pm",    "15 h"),
      ]
      def create_monthly_animation(df):
          df = df.assign(month=df.Date.dt.strftime("%m"))
          months = sorted(df.month.unique())
          fig = make_subplots(
              rows=1, cols=2, specs=[[{"type":"polar"}]*2],
              subplot_titles=[t for *_, t in PLOTS],
              horizontal_spacing=0.03
          )
          # Ajuster l'espacement vertical des titres de subplots
          for annotation in fig['layout']['annotations']:
              annotation['y'] += 0.07

          def traces_for(m):
              data = []
              sub = df[df.month == m]
              for i,(rain,wind,_) in enumerate(PLOTS):
                  cnt = (sub.groupby([wind,rain])
                          .size().unstack(fill_value=0)
                          .reindex(index=WIND_DIRS,columns=["Yes","No"],fill_value=0))
                  pct = cnt.div(cnt.sum()).mul(100)
                  # showlegend only for first col, once per modality
                  show = (i==0)
                  data += [
                      go.Scatterpolar(
                          r=pct["Yes"], theta=WIND_DIRS, fill="toself",
                          marker_color="green", name="Rain=Yes", showlegend=show
                      ),
                      go.Scatterpolar(
                          r=pct["No"],  theta=WIND_DIRS, fill="toself",
                          marker_color="rgba(0, 0, 255, 0.2)",  name="Rain=No",  showlegend=show
                      )#rgba ou a=alpha pour transparence
                  ]
              return data

          # initial traces
          for idx, tr in enumerate(traces_for(months[0])):
              fig.add_trace(tr, row=1, col=idx//2+1)

          # frames
          fig.frames = [go.Frame(data=traces_for(m), name=m) for m in months]

          # uniformiser axes
          for i in range(1,3):
              ax = fig.layout["polar" if i==1 else f"polar{i}"].radialaxis
              ax.range, ax.tickvals, ax.ticktext = [0,12.5], [0,6.25,12.5], ["0%","1/16","2/16"]

          # slider
          slider = {
              "active": 0, "y": -0.15, "x": 0.1, "len": 0.8, "pad": {"t": 50},
              "steps": [
                  {"label": m, "method": "animate",
                  "args": [[m], {"mode":"immediate","frame":{"duration":0}}]}
                  for m in months
              ]
          }
          # play/pause boutons
          buttons = {
              "type":"buttons","direction":"left","showactive":False,
              "x":0.1,"y":-0.25,"pad":{"t":10,"r":10},
              "buttons":[
                  {"label":"▶️ Play","method":"animate",
                  "args":[None,{"frame":{"duration":700,"redraw":True},"fromcurrent":True}]},
                  {"label":"⏸️ Pause","method":"animate",
                  "args":[[None],{"mode":"immediate","frame":{"duration":0}}]}
              ]
          }

          fig.update_layout(
              updatemenus=[buttons],
              sliders=[slider],
              title="Vent vs Pluie par mois (MM)",
              showlegend=True, height=450, width=1200, margin=dict(t=60,b=80)
          )
          return fig

      # — Streamlit —
      col1, col2 = st.columns(2)
      col1.metric("💨 Vent dominant à 9h", f"{vent_dominant_am}")
      col2.metric("💨 Vent dominant à 15h", f"{vent_dominant_pm}")

      st.title("Animation Direction du Vent & Pluie Demain par mois")
      if "df_station" in globals():
          df_station["Date"] = pd.to_datetime(df_station["Date"])
          chart = create_monthly_animation(df_station)
          st.plotly_chart(chart, use_container_width=True)
      else:
          st.warning("⚠️ Chargez d'abord `df_station`.")

    # ----------------------------
    # Pluviométrie : moyenne, mois max, jour max
    # ----------------------------
    with tab4:
      mean_rain = df_station["Rainfall"].mean()
      rain_by_month = df_station.groupby("Month")["Rainfall"].mean()
      max_month = rain_by_month.idxmax()
      month_names = {1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril", 5: "Mai", 6: "Juin",
                  7: "Juillet", 8: "Août", 9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre"}

      max_rain = df_station["Rainfall"].max()
      max_rain_date = df_station.loc[df_station["Rainfall"].idxmax(), "Date"]


      col1, col2, col3 = st.columns(3)
      col1.metric("☁️ Pluviométrie moyenne", f"{mean_rain:.1f} mm")
      col2.metric("📅 Mois le + pluvieux", month_names[max_month])
      col3.metric("📅 Jours le + pluvieux", f"{max_rain:.1f} mm", f"📅 {max_rain_date.date()}",delta_color="off")

      # ----------------------------
      # Rang pluviométrie moyenne
      # ----------------------------
      st.subheader("Précipitations mensuelles moyennes")
      pluvio_rank = df.groupby("Location")["Rainfall"].mean().sort_values(ascending=False).reset_index()
      rank = pluvio_rank[pluvio_rank["Location"] == station].index[0] + 1
      total = pluvio_rank.shape[0]
      st.success(f"La station **{station}** est **{rank}ᵉ** sur {total} pour la pluviométrie moyenne.")

      # ----------------------------
      # Évolution des précipitations mensuelles
      # ----------------------------
      df_station["YearMonth"] = df_station["Date"].dt.to_period("M").astype(str)
      df_trend = df_station.groupby("YearMonth")["Rainfall"].mean().reset_index()
      df_trend["Date"] = pd.to_datetime(df_trend["YearMonth"])

      fig_line = px.line(
          df_trend,
          x="Date",
          y="Rainfall",
          #title="Évolution des précipitations mensuelles",
          labels={"Rainfall": "Précipitations moyennes (mm)"}
      )
      st.plotly_chart(fig_line)




    # -----------------------------
    # 4. Carte animée : évolution des précipitations
    # -----------------------------
    st.divider()
    st.subheader(" Animation des précipitations par station")

    # Assure-toi que la colonne Date est bien en datetime
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')

    # Sélecteurs utilisateur
    years_available = sorted(df["Date"].dropna().dt.year.unique())
    selected_year = st.selectbox("Choisir une année :", years_available, index=years_available.index(2015))
    granularity = st.radio("Granularité temporelle :", ["Jour", "Semaine", "Mois"], horizontal=True)

    # Filtrer les données
    df_anim = df.dropna(subset=["Date", "Rainfall", "Latitude", "Longitude"]).copy()
    df_anim = df_anim[df_anim["Date"].dt.year == selected_year]

    # Choix de la période
    if granularity == "Jour":
        df_anim["Period"] = df_anim["Date"].dt.strftime("%Y-%m-%d")
    elif granularity == "Semaine":
        df_anim["Period"] = df_anim["Date"].dt.strftime("%Y-W%U")
    else:  # Mois
        df_anim["Period"] = df_anim["Date"].dt.strftime("%Y-%m")

    # Agrégation
    df_anim_grouped = df_anim.groupby(["Location", "Period"]).agg({
        "Rainfall": "mean",
        "Latitude": "first",
        "Longitude": "first"
    }).reset_index()

    # Nettoyage des valeurs aberrantes
    df_anim_grouped = df_anim_grouped[df_anim_grouped["Rainfall"] >= 0]

    # Définir des échelles fixes
    max_rainfall = df_anim_grouped["Rainfall"].max()
    min_rainfall = df_anim_grouped["Rainfall"].min()

    # Carte animée avec échelles fixes
    fig_anim = px.scatter_mapbox(
        df_anim_grouped,
        lat="Latitude",
        lon="Longitude",
        size="Rainfall",
        color="Rainfall",
        animation_frame="Period",
        animation_group="Location",
        hover_name="Location",
        color_continuous_scale="Viridis",
        size_max=25,
        zoom=2.7,
        height=650,
        range_color=[min_rainfall, max_rainfall],  # Échelle de couleur fixe
        title=f"Animation des précipitations ({granularity.lower()}s) – {selected_year}"
    )

    # Configuration supplémentaire pour l'animation
    fig_anim.update_layout(
        mapbox_style="carto-positron",
        mapbox_center={"lat": -26.5, "lon": 133.5},
        margin={"r":0, "t":40, "l":0, "b":0}
    )

    # Configuration de l'animation
    fig_anim.update_traces(
        marker=dict(sizeref=2.*max_rainfall/(25**2))  # Échelle de taille fixe
    )

    # Ajout des paramètres d'animation pour une transition plus fluide
    fig_anim.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1000
    fig_anim.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 500

    st.plotly_chart(fig_anim)


    # -----------------------------
    # 5. Corrélations, distribution des features
    # -----------------------------
    st.divider()

    st.subheader("Analyse des corrélations et distribution des variables")
    tab1, tab2, tab3 = st.tabs(["📈 Distribution des variables numériques","🔗 Analyse des corrélations", "❓Visualisation des manquants", ])

    # -----------------------------
    # 5.1. Distribution d'une variable
    # -----------------------------
    with tab1:
      numeric_cols = [col for col in df.select_dtypes(include=np.number).columns if df[col].nunique() > 10 and col != 'RISK_MM']
      selected_var = st.selectbox("Choisir une variable :", numeric_cols, index=numeric_cols.index("Rainfall"))
      fig_hist = px.histogram(df, x=selected_var, nbins=100, title=f"Distribution de {selected_var}")
      st.plotly_chart(fig_hist)

    # -----------------------------
    # 5.2. Analyse des corrélations
    # -----------------------------
    with tab2 :
      col1, col2 = st.columns([2,1])

      with col1:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr = df[numeric_cols].corr()

        # Matrice de corrélation
        fig_corr = px.imshow(
            corr,
            title="Matrice de corrélation",
            color_continuous_scale="RdBu_r",
            aspect="auto"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        # Ajustements pour réduire l'espace vide
        fig_corr.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),  # marge gauche, droite, top, bottom,
            autosize =True
            # width=500,    # fixe une largeur (optionnel)
            # height=400    # fixe une hauteur (optionnel)
        )

      with col2:
        # Top corrélations
        st.write("Top 10 des corrélations")
        corr_unstack = corr.unstack()
        corr_unstack = corr_unstack[corr_unstack != 1.0]
        top_corr = corr_unstack.sort_values(ascending=False)[:10]
        st.dataframe(top_corr)

    # -----------------------------
    # 5.2. Analyse des manquants
    # -----------------------------
    with tab3:

      # ---------Manquants par mois et années --------------------
      col1, col2 = st.columns(2)
      with col1:
        df_MM_AAAA = df.copy()
        df_MM_AAAA["Year"] = df_MM_AAAA["Date"].dt.year
        df_MM_AAAA["Month"] = df_MM_AAAA["Date"].dt.month
        df_MM_AAAA = df_MM_AAAA.groupby(["Month", "Year"]).size().unstack(fill_value=0)

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(df_MM_AAAA, cmap="viridis", cbar=True, linewidths=0.5, linecolor="white", ax=ax)
        ax.set_title("Présence des données sur chaque mois")
        ax.set_xlabel("Année")
        ax.set_ylabel("Mois")
        fig.tight_layout()

        st.pyplot(fig, use_container_width=True)

      with col2:
        st.markdown("""
    **Remarques :**
    - Les stations ne commencent pas en même temps (Candberra est seule en 2007)
    - A partir de 2009, toutes les stations émettent (sauf Katherine, Nhil et Ulhuru qui commencent au 1er Mars 2013)
    - Certains mois sont manquants pour toutes les stations""")

      # ---------% de Manquants par features --------------------

      #Dataframe avec Booleen pour NAN sur toutes les colonnes sauf location
      df_sans_location= df.drop("Location",axis = 1).iloc[:, :].isna()
      df_col_location =  df.iloc[:, 1:2]
      df_NAN_apres_3e_col= df_col_location .join(df_sans_location)

      df_NAN_Location = df_NAN_apres_3e_col.groupby("Location").mean()

      # Affichage graphe
      fig, ax = plt.subplots(figsize=(25, 8))
      sns.heatmap(df_NAN_Location.T, cmap="Blues", cbar=True, linewidths=0.5, linecolor="white", annot= True, fmt= ".1f")
      ax.set_title("Part de données manquantes dans chaque variable sur chaque stations")  #Mettre les lignes du groupeby en y, et en x les colonnes du groupby
      ax.set_xlabel("Station")
      ax.set_ylabel("Variable")
      st.pyplot(fig)



#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# -----------------------------
# Page 2 : comparaison des modèles
# -----------------------------
if page == "Comparaison des modèles":

    # -------------------------------
    # FONCTION SHAP
    # -------------------------------
    def afficher_shap(model, X, sample_size):
        st.markdown("## Interprétabilité avec SHAP")

        X_sample = X.sample(sample_size, random_state=42)
        st.info(f"Interprétabilité SHAP sur **{sample_size}** observations")

        with st.expander("🧪 Lancer le calcul SHAP et afficher les graphiques", expanded=False):
            if st.button("💥 Calculer SHAP", key="launch_shap"):  # Bouton pour réellement déclencher le calcul
                with st.spinner("Calcul des valeurs SHAP…"):
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_sample)

                # Bar plot
                st.markdown("#### Importance globale des variables (bar plot)")
                fig_bar, ax = plt.subplots(figsize=(8, 4))
                shap.plots.bar(
                    shap.Explanation(
                        values=shap_values,
                        base_values=explainer.expected_value,
                        data=X_sample,
                        feature_names=X_sample.columns.tolist()
                    ),
                    show=False,
                    ax=ax
                )
                st.pyplot(fig_bar)

                # Summary plot
                st.markdown("#### Répartition des impacts (summary plot)")
                fig_summary = plt.figure(figsize=(8, 4))
                shap.summary_plot(shap_values, X_sample, plot_type="dot", show=False)
                st.pyplot(fig_summary)

    # -------------------------------
    # AFFICHAGE PRINCIPAL
    # -------------------------------
    st.title(" Comparaison des performances modèles")

    model_choice = st.selectbox("Choisissez un modèle :", list(MODEL_LIST.keys()), key="select_model")
    default_threshold = 0.38 if model_choice == "XGBoost Final" else 0.5
    seuil = st.slider("🎯 Seuil de décision (classification)", 0.0, 1.0, step=0.01, value=default_threshold, key="seuil_slider")

    model = load_model(model_choice)
    features = load_features()
    X_old, y_old = load_old_dataset()

    st.divider()
    st.header(f" Performances du modèle **{model_choice}** sur les anciennes données")
    afficher_resultats(model, X_old[features], y_old, "Ancien test set", seuil)

    if model_choice == "XGBoost Final":
        st.divider()
        max_sample = min(500, len(X_old))
        sample_size = st.slider("🔍 Taille de l'échantillon pour SHAP", min_value=50, max_value=max_sample, value=200, step=50, key="slider_shap")
        afficher_shap(model, X_old[features], sample_size)


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# -----------------------------
# Page 3 : Données actuelles du BOM
# -----------------------------

# -----------------------------
# 0. Cache dans la page
if "period_choice" not in st.session_state:
    st.session_state.period_choice = None
if "stations_choice" not in st.session_state:
    st.session_state.stations_choice = None
if "logic_choice" not in st.session_state:
    st.session_state.logic_choice = None
if "model_choice" not in st.session_state:
    st.session_state.model_choice = None
if "preprocessed_data" not in st.session_state:
    st.session_state.preprocessed_data = None
def reset_all():
    for k in [
        "period_choice", "stations_choice",
        "logic_choice", "model_choice",
        "preprocessed_data"
    ]:
        st.session_state[k] = None
    st.experimental_rerun()
# -----------------------------

if page == pages[2] :
  st.subheader("📡 Classification en temps réel sur les données actuelles du BOM")
#-1. Collecte des données actuelles---------------------------------------------------------------------------------------------------------------------------------
  #1.1 Initialisation
  ##1.1.1 Liste des mois
  mois_courant = datetime.now().replace(day=1)
  mois_depart = mois_courant - relativedelta(months=1) # Mois de départ = mois précédent
  # Générer les 13 mois vers le passé (de -12 à 0 mois avant mois_depart)
  liste_mois_a_selectionner = [(mois_depart - relativedelta(months=i)).strftime("%Y%m") for i in reversed(range(13))]

  ##1.1.2 Dico stations
  dico_stations_BOM = load_pickle("dico_station.pkl")

  ##1.1.3 Noms des colonnes dans le df généré (futur X_test)
  nom_colonnes_df_principal = {"Minimum temperature (°C)" : "MinTemp",
                             "Maximum temperature (°C)": "MaxTemp",
                             "Rainfall (mm)" : "Rainfall",
                             "Evaporation (mm)" : "Evaporation",
                             "Sunshine (hours)" :"Sunshine",
                             "Direction of maximum wind gust ":"WindGustDir",
                             "Speed of maximum wind gust (km/h)" : "WindGustSpeed",
                             "Time of maximum wind gust" : "Time of maximum wind gust" ,  #nouvelle colonne
                             "9am Temperature (°C)": "Temp9am",
                             "9am relative humidity (%)" : "Humidity9am",
                             "9am cloud amount (oktas)":"Cloud9am",
                             "9am wind direction" :"WindDir9am",
                             "9am wind speed (km/h)": "WindSpeed9am",
                             "9am MSL pressure (hPa)":"Pressure9am",
                             "3pm Temperature (°C)":"Temp3pm",
                             "3pm relative humidity (%)":"Humidity3pm",
                             "3pm cloud amount (oktas)": "Cloud3pm",
                             "3pm wind direction" : "WindDir3pm",
                             "3pm wind speed (km/h)": "WindSpeed3pm",
                             "3pm MSL pressure (hPa)":"Pressure3pm"}

  # 1.2 Saisie utilisateur
  st.write("#### Sélection ")
  st.button("🔄 Reset tout", on_click=reset_all) #Pour sélectionner de nouvelles données
  col1, col2 = st.columns(2) # Afficher deux listes déroulantes de sélection multiples : nom des mois, nom des stations

  with col1 :
    liste_mois = st.multiselect("Sélectionnez un mois, ou des mois consécutifs", liste_mois_a_selectionner)
    if liste_mois != st.session_state.period_choice:
      st.session_state.period_choice = liste_mois
      st.session_state.preprocessed_data = None  # Invalide l’ancien preprocessing

  with col2 : # Multiselect avec affichage du nom de la station
    stations_selectionnees = st.multiselect(
        "Sélectionnez une ou plusieurs stations",
        options=list(dico_stations_BOM.keys()),
        format_func=lambda x: dico_stations_BOM[x][2])
    if stations_selectionnees != st.session_state.stations_choice:
      st.session_state.stations_choice = stations_selectionnees
      st.session_state.preprocessed_data = None  # Invalide l’ancien preprocessing
  st.write("💡 Pb_Goulburn est une nouvelle station")
  # Générer un dictionnaire filtré identique en format à l’original
  dico_stations_DWO = {
      k: dico_stations_BOM[k]
      for k in stations_selectionnees}

  # 1.2.2 Tant que l'utilisateur n'a pas fait de sélection complète (mois + location) ne pas aller plus loin
  if not liste_mois or not dico_stations_DWO:
      st.stop()

  # 2 Code récupérant les csv et les conslidant dans le Df df_conso_station à partir d'une liste d'url
  compteur = 0 #pour consolider les df de station dans un un seul df : df_conso_station
  df_conso_station=pd.DataFrame()

  for no_report in dico_stations_DWO :
      i=0
      compteur+= 1
      df_une_station=pd.DataFrame()

      for le_annee_mois in liste_mois :
          url_concatene = ("http://www.bom.gov.au/climate/dwo/"+str(le_annee_mois)+"/text/"+no_report+"."+le_annee_mois+".csv") # Exemple : http://www.bom.gov.au/climate/dwo/202412/text/IDCJDW2804.202412.csv
          i+= 1
          # Essayer de télécharger le fichier avec des en-têtes de type navigateur
          headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
          # Effectuer la requête
          response = requests.get(url_concatene, headers=headers)
          # Vérifier si la requête est réussie
          if response.status_code == 200:
              # Utiliser StringIO pour lire le texte CSV dans un DataFrame
              csv_data = StringIO(response.text)
              # L'entete n'est a=pas toujours sur la meme ligne : On lit le fichier ligne par ligne pour trouver le header qui commence par ,"Date",
              lines = response.text.splitlines()
              # Trouver la ligne où "Date" apparaît pour la première fois
              header_row = None
              for i, line in enumerate(lines):
                  if "Date" in line:
                      header_row = i
                      break
              df_recupere = pd.read_csv(csv_data, sep=",", skiprows=header_row, encoding="latin1")
              #Faire un df consolidé par station
              if i == 1 :
                  df_une_station = df_une_station
              else :
                  df_une_station = pd.concat([df_une_station, df_recupere], ignore_index=True)
          else:
              st.write("Erreur lors du chargement de l'URL pour {} de {} : {} - URL: {}".format(le_annee_mois, dico_stations_DWO[no_report][2], response.status_code,url_concatene))

    # 2.2 Mise en forme du csv collecté
      df_une_station = df_une_station.rename(nom_colonnes_df_principal, axis = 1) #Mettre les noms de colonnes du df principal
      df_une_station = df_une_station.drop(["Unnamed: 0","Time of maximum wind gust"],axis =1) #Suppression de colonnes

      #Ajout de colonnes
      #Insérer en 2e position (loc=1) le nom de la station (3e colonne du dico) associé à ce rapport dans le dictionnaire dico_stations_DWO
      df_une_station.insert(1,column="Location", value=dico_stations_DWO[no_report][2]) #

      df_une_station["RainToday"]=df_une_station["Rainfall"].apply(lambda x: "Yes" if x>1 else "No")
      #Met les valeurs de RainToday à l'indice précédent dans RainTomorrow
      df_une_station["RainTomorrow"] = np.roll(df_une_station["RainToday"].values, 1) #Ex RainToday 02/01/24 -> RainTomorrow 01/01/2024

      #Supprimer le dernier relevé du df car RainTomorrow y sera toujours inconnu (suite au np.roll c'est la valeur de RainToday a la 1e ligne du df, et il aurait fallu celle du lendemain de la dernier ligne du df).
      df_une_station.drop(df_une_station.index[-1], inplace=True) #Si la dernière journée de relevé est le 30/06/24, nous ne pourrons pas calculer RainTomorrow (futur y_true) pour ce jour, car il aurait besoin des RainToday du 01/07/24

      #Mettre les colonnes dans le même ordre que le df de départ du projet
      df_une_station = df_une_station[liste_colonne_df]


      #Faire un df consolidé (df_conso_station) des df unitaire par station (df_une_station)
      if compteur == 1 :
          df_conso_station = df_une_station
      else :
          df_conso_station = pd.concat([df_conso_station, df_une_station], ignore_index=True)


  #-2. Preprocessing de base---------------------------------------------------------------------------------------------------------------------------------

  ## 2.1 Modification de la vitesse "Calm" par 0km/h
  df_conso_station["WindSpeed9am"] = df_conso_station["WindSpeed9am"].apply(lambda x: 0 if x =="Calm" else x)
  df_conso_station["WindSpeed3pm"] = df_conso_station["WindSpeed3pm"].apply(lambda x: 0 if x =="Calm" else x)
  df_conso_station["WindGustSpeed"] = df_conso_station["WindGustSpeed"].apply(lambda x: 0 if x =="Calm" else x)


  ## 2.2 Suprresion 25% des NAN
  # === Calcul du ratio de NaN ===
  total_cells_per_location = df_conso_station.groupby("Location").size() * (df_conso_station.shape[1] - 1)  # -1 car on exclut 'Location'
  nan_counts_per_location = df_conso_station.drop(columns="Location").isna().groupby(df_conso_station["Location"]).sum().sum(axis=1)
  nan_ratio = nan_counts_per_location / total_cells_per_location
  # === Filtrage des stations valides ===
  valid_locations = nan_ratio[nan_ratio <= 0.25].index.to_list() #>>ex : {'BadgerysCreek', 'Albury'}
  df_conso_station = df_conso_station[df_conso_station["Location"].isin(valid_locations)]

  # === Messages Streamlit ===
  # noms des stations selectionné par l'utilisateur
  stations_selectionnees_noms = [dico_stations_BOM[code][2] for code in stations_selectionnees]  #>> ex : 0:"Penrith" 1:"AliceSprings"
  stations_supprimees = sorted(set(stations_selectionnees_noms) - set(valid_locations)) #>> Stations sélectionnées : Penrith, AliceSprings
  if len(valid_locations) == 0:
    st.error("Toutes les stations sélectionnées ont plus de 25% de données manquantes. Veuillez en choisir d'autres.")
    st.stop()
  elif len(stations_supprimees) > 0:
      st.warning(f"Les stations suivantes ont été exclues car elles contiennent plus de 25% de données manquantes : {', '.join(stations_supprimees)}")

  # === Fin du tron commun avec Florent ===
  df_X_y_test = df_conso_station.copy()

  ## 2.3 Ajout de la latitude et de la longitude
  dico_charge = load_pickle("dico_station_geo.pkl") #ecart avec localisations_gps.csv : ajout des coordonnées de Goulburn
  df_dico_station_geo = pd.DataFrame.from_dict(dico_charge, orient="index",columns=["Lat", "Lon"])
  df_dico_station_geo.columns = ["Latitude", "Longitude"]
  df_X_y_test = df_X_y_test.merge(right=df_dico_station_geo, left_on="Location", right_index=True, how="left")

  ## 2.4 Ajout du climat
  climat_mapping = pd.read_csv(os.path.join(SCALER_PATH, "climat_mapping.csv"))#/!\
  climat_mapping_series = climat_mapping.set_index("Location")["Climate"]
  df_X_y_test['Climate'] = df_X_y_test["Location"].map(climat_mapping_series) #pour chaque valeur de df.Location, on récupère la valeur correspondante dans climat_mapping

  ## 2.5 Date, Saison
  df_X_y_test["Date"]=pd.to_datetime(df_X_y_test["Date"], format = "%Y-%m-%d")
  df_X_y_test["Month"] = df_X_y_test['Date'].dt.month
  df_X_y_test["Year"] = df_X_y_test['Date'].dt.year
  df_X_y_test["Saison"] = df_X_y_test["Month"].apply( lambda x : "Eté" if x in [12, 1, 2] else "Automne" if x in [3, 4, 5] else "Hiver" if x in [6, 7, 8] else "Printemps")

  ## 2.6 Suppression des features
  df_X_y_test = df_X_y_test.drop(["Sunshine","Evaporation"], axis = 1)

  ## 2.7 Traitement de la variable cible : Suppression des NaN et Label Encoder
  df_X_y_test = df_X_y_test.dropna(subset=["RainTomorrow"], axis=0, how="any")
  df_X_y_test["RainTomorrow"] = df_X_y_test["RainTomorrow"].map({"Yes": 1, "No": 0})
  df_X_y_test["RainToday"] = df_X_y_test["RainToday"].map({"Yes": 1, "No": 0})

  # 2.8 Aperçu des données à prédire
  st.write("Données à prédire")
  st.dataframe(df_X_y_test.head(6))

  #-3 Choix du modèle--------------------------------------------------------------------------------------------------------------------------------------------------------------
  st.write("#### Choix d'une méthode de prévision")
  #Choix par radio bouton entre Logique d'entrainement temporelle ou non
  choix_preprocessing = st.selectbox("Choisir une logique d'entrainement",["---", "temporel", "non-temporel"])

  if choix_preprocessing != st.session_state.logic_choice:
    st.session_state.logic_choice = choix_preprocessing
    st.session_state.model_choice = None
    st.session_state.preprocessed_data = None

  if st.session_state.logic_choice == "---": # Bloquer l'exécution si aucun modèle n'est sélectionné
      st.stop()
  #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  if choix_preprocessing == "temporel" :
    # Choix du modèle des modèles
    choix_model_temporel = st.selectbox("Choisissez un modèle :",["--- Sélectionner ---"] + list(MODEL_LIST.keys()))

    if choix_model_temporel == "--- Sélectionner ---": # Bloquer l'exécution si aucun modèle n'est sélectionné
      st.stop()

    ### 3.1.A Preprocessing Florent------------------------------------------------------------------------------------------------------------------------------------------------
    do_preprocess = st.checkbox("Lancer le preprocessing, la prédiction et l'évaluation du modèle")
    if not do_preprocess:
        st.stop()


    # --- Chargement des scalers / modèles ---
    scalers = load_joblib("weather_scalers.joblib")
    top_features = load_features()
    scaler_knn = load_joblib("scaler_knn.joblib")
    knn_model = load_joblib("knn_model.joblib")

    # --- Fonctions ---
    def impute_wind_features(df):
      df = df.copy()
      df["WindSpeed9am"] = pd.to_numeric(df["WindSpeed9am"], errors="coerce")
      df["WindSpeed3pm"] = pd.to_numeric(df["WindSpeed3pm"], errors="coerce")
      df["WindGustSpeed"] = pd.to_numeric(df["WindGustSpeed"], errors="coerce")

      df.loc[df["WindSpeed9am"].isna(), "WindSpeed9am"] = 0
      df.loc[df["WindSpeed3pm"].isna(), "WindSpeed3pm"] = 0

      mask_9am = (df["WindSpeed9am"] > 0) & df["WindDir9am"].isna()
      df.loc[mask_9am, "WindDir9am"] = df["WindDir9am"].mode()[0] if not df["WindDir9am"].mode().empty else "NoWind"
      mask_3pm = (df["WindSpeed3pm"] > 0) & df["WindDir3pm"].isna()
      df.loc[mask_3pm, "WindDir3pm"] = df["WindDir3pm"].mode()[0] if not df["WindDir3pm"].mode().empty else "NoWind"

      df.loc[df["WindSpeed9am"] == 0, "WindDir9am"] = "NoWind"
      df.loc[df["WindSpeed3pm"] == 0, "WindDir3pm"] = "NoWind"

      df["WindGustSpeed"] = df["WindGustSpeed"].fillna(df[["WindSpeed9am", "WindSpeed3pm"]].max(axis=1))

      mask_gust = df["WindGustDir"].isna() & (df["WindGustSpeed"] > 0)
      df.loc[mask_gust, "WindGustDir"] = df.apply(
          lambda row: row["WindDir9am"] if row["WindSpeed9am"] >= row["WindSpeed3pm"] else row["WindDir3pm"], axis=1)

      df["WindGustDir"] = df["WindGustDir"].fillna("NoWind")
      df["WindSpeed9am"].fillna(df["WindSpeed9am"].median(), inplace=True)
      df["WindSpeed3pm"].fillna(df["WindSpeed3pm"].median(), inplace=True)
      return df

    def impute_cloud_with_knn(df, variables, scaler, knn):
        df_knn = df[variables].copy()
        df_knn_scaled = scaler.transform(df_knn)
        df_imputed = knn.transform(df_knn_scaled)
        df_imputed = pd.DataFrame(df_imputed, columns=variables, index=df.index)
        for col in ["Cloud9am", "Cloud3pm"]:
            df[col] = df[col].fillna(df_imputed[col].round().clip(0, 8))
        return df

    def impute_temp_hum_press(df):
        for col in ["Temp9am", "Temp3pm", "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm"]:
            df[col] = df[col].interpolate().fillna(df[col].median())
        return df

    def add_features(df):
        df = df.copy()
        df.sort_values(["Location", "Date"], inplace=True)
        df["TempRange"] = df["MaxTemp"] - df["MinTemp"]
        df["TempDelta"] = df["Temp3pm"] - df["Temp9am"]
        df["PressureDelta"] = df["Pressure3pm"] - df["Pressure9am"]
        df["WindVariation"] = df["WindSpeed3pm"] - df["WindSpeed9am"]
        df["WindGustRatio"] = df["WindGustSpeed"] / (df["WindSpeed3pm"] + 1)
        df["HumidityDiff"] = df["Humidity3pm"] - df["Humidity9am"]
        df["CloudChange"] = df["Cloud3pm"] - df["Cloud9am"]
        df["Rainfall_rolling3"] = df["Rainfall"].rolling(window=3, min_periods=1).sum()

        for col, lags in {
            'Rainfall': [1, 2, 3], 'Pressure3pm': [1, 2], 'Humidity3pm': [1, 3],
            'WindGustSpeed': [1], 'Cloud3pm': [1, 2]
        }.items():
            for lag in lags:
                df[f"{col}_lag{lag}"] = df[col].shift(lag)

        return df

    def transform_features(df, scalers):
        """
        Applique les transformations finales (scalings, encodages) sur un DataFrame prétraité.
        Si df est vide, lève une erreur explicite.
        """
        if df.empty:
            raise ValueError("🚫 Le DataFrame est vide, impossible d'appliquer les scalers.")

        df = df.copy()

        # Définition manuelle des colonnes à transformer
        standard = [
            "MinTemp", "MaxTemp", "Temp9am", "Temp3pm",
            "TempRange", "TempDelta", "Pressure9am", "Pressure3pm",
            "PressureDelta", "Pressure3pm_lag1", "Pressure3pm_lag2"
        ]

        yeo = ["WindVariation", "HumidityDiff", "CloudChange"]

        log = [
            "Rainfall", "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm",
            "WindGustRatio", "Humidity9am", "Humidity3pm",
            "Rainfall_lag1", "Rainfall_lag2", "Rainfall_lag3",
            "Rainfall_rolling3", "WindGustSpeed_lag1"
        ]

        # Transformations
        df[standard] = scalers["standard"].transform(df[standard])
        df[yeo] = scalers["yeo"].transform(df[yeo])
        df[log] = df[log].apply(np.log1p)

        # Encodage des directions de vent
        wind_directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                          'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW', 'NoWind']
        dir_map = {d: i for i, d in enumerate(wind_directions)}

        for col in ['WindGustDir', 'WindDir9am', 'WindDir3pm']:
            df[col] = df[col].map(dir_map)
            df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / 16)
            df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / 16)
            df.loc[df[col] == 16, [f"{col}_sin", f"{col}_cos"]] = 0

        df.drop(columns=["WindGustDir", "WindDir9am", "WindDir3pm"], inplace=True)

        # Climat et date
        df["Climate"] = scalers["label_climate"].transform(df["Climate"].astype(str))
        df["Saison"] = df["Date"].dt.month.map(lambda x: 0 if x in [12, 1, 2]
                                              else 1 if x in [3, 4, 5]
                                              else 2 if x in [6, 7, 8]
                                              else 3)
        df["Month_sin"] = np.sin(2 * np.pi * df["Date"].dt.month / 12)
        df["Month_cos"] = np.cos(2 * np.pi * df["Date"].dt.month / 12)

        # Localisation
        df[["Latitude", "Longitude"]] = scalers["latlong"].transform(df[["Latitude", "Longitude"]])

        return df

    # --- Preprocessing ---
    def preprocessing_Florent(df):
        df["Date"] = pd.to_datetime(df["Date"])
        gps_mapping = pd.read_csv(os.path.join(SCALER_PATH, "localisations_gps.csv"))
        df = df.merge(climat_mapping, on="Location", how="left").merge(gps_mapping, on="Location", how="left")

        df["RainTomorrow"] = df["RainTomorrow"].map({"Yes": 1, "No": 0})
        df["RainToday"] = df["RainToday"].map({"Yes": 1, "No": 0})
        df = df.drop(columns=["Sunshine", "Evaporation"])
        df.dropna(subset=["RainTomorrow", "RainToday"], inplace=True)
        df = impute_wind_features(df)
        df = impute_cloud_with_knn(df, [
            "Cloud9am", "Cloud3pm", "Humidity9am", "Humidity3pm", "Rainfall",
            "Temp9am", "Temp3pm", "Pressure9am", "Pressure3pm",
            "WindSpeed9am", "WindSpeed3pm", "Latitude", "Longitude"
        ], scaler_knn, knn_model)
        df = impute_temp_hum_press(df)
        df = add_features(df)
        df = df[df.notna().all(axis=1)]
        df = transform_features(df, scalers)

        missing_cols = [col for col in top_features if col not in df.columns]
        if missing_cols:
            st.error(f"Colonnes manquantes : {missing_cols}")
            st.stop()

        X = df[top_features].copy()
        y_true = df["RainTomorrow"]

        return X, y_true, df["Date"], df


    # --- Application du preprocessing ---
    X, y, _, _ = preprocessing_Florent(df_conso_station)


    ### 3.1.B Modelisation Florent-----------------------------------------------------------------------------------------------------------------------------------------------

    # Chargement des objets
    model = load_model(choix_model_temporel)

 #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  elif choix_preprocessing == "non-temporel" :
    # Choix du modèle des modèles
    choix_model_non_temporel = st.selectbox("Choisissez un modèle :",["--- Sélectionner ---"] + list(MODEL_LIST_Non_temporel.keys()))

    if choix_model_non_temporel == "--- Sélectionner ---": # Bloquer l'exécution si aucun modèle n'est sélectionné
      st.stop()

    ### 3.2.A Preprocessing Amelie------------------------------------------------------------------------------------------------------------------------------------------------
    do_preprocess_abo = st.checkbox("Lancer preprocessing, prédictions puis évaluation")
    if not do_preprocess_abo :
        st.stop()

    #### 3.2.A.1 Suppresion des features avec trop de manquants
    df_X_y_test = df_X_y_test.drop(["RainToday","Saison","Climate"], axis = 1)

    #### 3.2.A.2 Complétions autorisées
    df_X_y_test["Pressure3pm"]=df_X_y_test["Pressure3pm"].fillna(df_X_y_test["Pressure9am"])
    df_X_y_test["Pressure9am"]=df_X_y_test["Pressure9am"].fillna(df_X_y_test["Pressure3pm"])

    df_X_y_test["WarmerTemp"] = df_X_y_test[["Temp9am", "Temp3pm"]].max(axis=1)
    df_X_y_test["MaxTemp"]=df_X_y_test["MaxTemp"].fillna(df_X_y_test["WarmerTemp"].round(0)) #arrondi à l'entier comme la definition du BOM
    df_X_y_test = df_X_y_test.drop(["WarmerTemp"],axis=1)

    df_X_y_test["Temp3pm"]=df_X_y_test["Temp3pm"].fillna(df_X_y_test["MaxTemp"])


    #### 3.2.A.3 Encodage Statless
    #####-----Fonction Encodage Statless-----------------------------------------------------------------------------------------------------------------------------------------------
    def encode_month(df, month_col="Month"):
      """Encode le mois en sin et cos puis supprime la colonne originale."""
      df['month_sin'] = np.sin(2 * np.pi * (df[month_col] - 1) / 12)
      df['month_cos'] = np.cos(2 * np.pi * (df[month_col] - 1) / 12)
      df=df.drop(columns=[month_col],axis=1)
      return df

    def encode_wind_direction(df):
        # Encodage cyclique de la direction du vent (et du cas "pas de vent")
        # 1) Définir la liste des 16 directions cycliques (rose des vents)
        # ------------------------------------------------------------------------
        #    Ici, on ordonne explicitement les directions dans le sens horaire,
        #    en commençant par "N" à l’indice 0, puis "NNE", "NE", etc.
        directions = [
            "N",   "NNE", "NE",  "ENE",
            "E",   "ESE", "SE",  "SSE",
            "S",   "SSW", "SW",  "WSW",
            "W",   "WNW", "NW",  "NNW"
        ]

        # ------------------------------------------------------------------------
        # 2) Construire le mapping direction → angle (en radians)
        # ------------------------------------------------------------------------
        #    Chaque direction est associée à un angle = idx * (2π / 16),
        #    où idx est l’indice de la direction dans la liste ci-dessus.
        #    Ex. : "N" → 0 rad, "ENE" → 3 * (2π/16) = 3π/8, etc.
        angle_mapping = {
            dir_name: (idx * 2 * np.pi / 16)
            for idx, dir_name in enumerate(directions)
        }

        # ------------------------------------------------------------------------
        # 3) Parcourir chaque couple (colonne de direction, colonne de vitesse)
        #    - Pour WindDir9am et WindDir3pm, on gère le cas “pas de vent”.
        #    - Pour WindGustDir, la vitesse est toujours > 0 (pas de “pas de vent”).
        #    On crée pour chaque couple :
        #      • des colonnes sin/cos de l’angle (avec NaN si direction absente),
        #      • éventuellement un indicateur NoWind_<col_speed> pour WindDir9am/3pm.
        # ------------------------------------------------------------------------
        for (col_dir, col_speed) in [
            ("WindDir9am",  "WindSpeed9am"),
            ("WindDir3pm",  "WindSpeed3pm"),
            ("WindGustDir", "WindGustSpeed")
        ]:
            # ------------------------------------------------------------
            # Détection du cas “pas de vent” ET direction absente/blanche
            # ------------------------------------------------------------
            handle_no_wind = col_dir in ["WindDir9am", "WindDir3pm"]
            if handle_no_wind:
                # a) Détecter les lignes où la vitesse vaut exactement 0
                is_exact_zero = (df[col_speed] == 0)
                # b) Détecter si la direction est manquante : NaN ou chaîne vide
                mask_dir_missing = df[col_dir].isna() | (df[col_dir].astype(str).str.strip() == "")
                # c) Combinaison : “pas de vent” ET direction absente
                mask_no_wind = is_exact_zero & mask_dir_missing
                # d) Créer l’indicateur NoWind_<col_speed> (1 si vitesse == 0)
                #    On met 1 si vitesse = 0, même si direction présente ou non.
                df[f"NoWind_{col_speed}"] = is_exact_zero.astype(int)
            else:
                # Pour WindGust, pas de “pas de vent” → on n’utilise pas NoWind
                is_exact_zero = pd.Series(False, index=df.index)
                mask_no_wind = pd.Series(False, index=df.index)

            # ------------------------------------------------------------
            # Mapper la direction textuelle → angle (NaN si direction absente ou non reconnue)
            # ------------------------------------------------------------
            df[f"{col_dir}_angle"] = df[col_dir].map(angle_mapping)

            # ------------------------------------------------------------
            # Si “pas de vent” ET direction absente, forcer angle = 0 rad
            # ------------------------------------------------------------
            if handle_no_wind:
                df.loc[mask_no_wind, f"{col_dir}_angle"] = 0.0

            # ------------------------------------------------------------
            # Calculer sin(angle) et cos(angle)
            #   • Si angle est NaN (direction absente pour d’autres raisons), sin/cos restent NaN.
            #   • Si “pas de vent”, angle forcé à 0 → sin=0, cos=1.
            #   • Sinon, angle valide → sin(angle), cos(angle).
            # ------------------------------------------------------------
            sin_col = f"{col_dir}_sin"
            cos_col = f"{col_dir}_cos"
            df[sin_col] = np.nan
            df[cos_col] = np.nan

            # a) Cas “pas de vent” (force angle=0) → sin=0, cos=1
            if handle_no_wind:
                df.loc[mask_no_wind, sin_col] = 0.0
                df.loc[mask_no_wind, cos_col] = 1.0

            # b) Cas angle valide pour toutes les lignes
            mask_angle_valid = df[f"{col_dir}_angle"].notna()
            df.loc[mask_angle_valid, sin_col] = np.sin(df.loc[mask_angle_valid, f"{col_dir}_angle"])
            df.loc[mask_angle_valid, cos_col] = np.cos(df.loc[mask_angle_valid, f"{col_dir}_angle"])

            # ------------------------------------------------------------
            # Nettoyage final : supprimer les colonnes de direction textuelle et d’angle
            # ------------------------------------------------------------
            df.drop(columns=[col_dir, f"{col_dir}_angle"], inplace=True)

        # ------------------------------------------------------------------------
        # À l’issue de cette boucle :
        # → Pour WindDir9am et WindDir3pm :
        #     • Une colonne NoWind_<col_speed> (1 si vitesse == 0, 0 sinon).
        #     • Deux colonnes <col_dir>_sin et <col_dir>_cos :
        #         - Si “pas de vent” & direction absente → (0, 1).
        #         - Si vent présent & angle valide → (sin(angle), cos(angle)).
        #         - Si vent présent mais angle manquant → (NaN, NaN).
        #
        # → Pour WindGustDir :
        #     • Pas de colonne NoWind (jamais de “pas de vent”).
        #     • Deux colonnes WindGustDir_sin et WindGustDir_cos :
        #         - Si direction valide → (sin(angle), cos(angle)).
        #         - Sinon (colonne direction initiale absente/mal encodée) → (NaN, NaN).
        # ------------------------------------------------------------------------
        return df
    #####-----Fin Fonction----------------------------------------------------------------------------------------------------------------------------------------------------------

    ##### Application de l'encodage stateless
    df_X_y_test = encode_month(df_X_y_test)
    df_X_y_test = encode_wind_direction(df_X_y_test)

    #### 3.2.A.5 Split Feaures/variable cible
    X_test_temporel = df_X_y_test.drop(columns = ["RainTomorrow"])
    y_test_temporel = df_X_y_test["RainTomorrow"] #pourrait différer cela les suppresions de lignes en NaN (Florent : RainToday)

    #### 3.2.A.6 Complétion des NAN
    #### 3.2.A.6.A Complétion des NAN nuages
    transformer_cloud = load_cloudpickle("cloud_imputer.pkl")
    X_test_temporel = transformer_cloud.transform(X_test_temporel)

    #### 3.2.A.6.B Complétion des autres NAN
    transformer = load_cloudpickle("transformer_KNNImputerABO.pkl")
    X_test_temporel = transformer.transform(X_test_temporel)

    #### 3.2.A.7 Enrichissement des features
    def amplitude_thermique(X) :
        X["Amplitude_Temp"] = X['MaxTemp']- X['MinTemp']
        X = X.drop(["MaxTemp","MinTemp"],axis=1)
        return X

    X_test_temporel = amplitude_thermique(X_test_temporel)

    #### 3.2.A.8 Suppression de features
    X_test_temporel = X_test_temporel.drop(["Date","Location"],axis=1)

    #### 3.2.A.9 Scaling
    #####-----Fonction Scaling-----------------------------------------------------------------------------------------------------------------------------------------------
    def add_engineered_features(X: pd.DataFrame,
                                ref_year: int = 2007, #ref_year = 1e année du dataset d'entrainement est 0
                                lat0: float = -25.0,
                                lon0: float = 133.0) -> pd.DataFrame:
        X_fe = X.copy()
        # Deltas temporel
        X_fe['Year_delta']      = X_fe['Year']      - ref_year
        # Deltas géographiques
        # Transformer Latitude et Longitude en Latitude_delta et Longitude_delta : où le centre du dataset d'entrainement est 0, et correspond au centre de l'Australie.
        # A noter : un degré de longitude à l'équateur = 110km, et à 60°Sud = 60 km . Pour corriger, on peut faire un Haversine et encodage azimut.
        X_fe['Latitude_delta']  = X_fe['Latitude']  - lat0
        X_fe['Longitude_delta'] = X_fe['Longitude'] - lon0
        # Log-transform
        X_fe['Rainfall']    = np.log1p(X_fe['Rainfall'])
        X_fe = X_fe.drop(['Year','Latitude','Longitude'], axis=1)
        return X_fe


    # Fonction pour charger et appliquer les scalers sur n’importe quel X
    def load_and_apply_scalers(X_fe: pd.DataFrame,
                                 scaler_name: str = "scalers.joblib") -> pd.DataFrame:
          artefact = load_joblib(scaler_name)
          scalers, feats = artefact['scalers'], artefact['feature_lists']
          X_scaled = X_fe.copy()
          for key, cols in feats.items():
              X_scaled[cols] = scalers[key].transform(X_scaled[cols])
          return X_scaled
    ####-----Fin Fonction-----------------------------------------------------------------------------------------------------------------------------------------------

    # Ajout des features d'ingénierie
    X_test_fe= add_engineered_features(X_test_temporel, ref_year=2007, lat0=-25.0, lon0=133.0)
    # Chargement et application des scalers (en paramètre le df obtenu avant)
    X_test_temporel  = load_and_apply_scalers(X_test_fe,  scaler_name = "scalers.joblib")

    #Aperçu des features en fin de preprocessing
    st.write("Aperçu des features en fin de preprocessing")
    st.dataframe(X_test_temporel.head(3))

    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ### 3.2.B Modelisation Amelie------------------------------------------------------------------------------------------------------------------------------------------------
    do_predict = st.checkbox("Lancer prédiction")
    if not do_predict:
        st.stop()

    # Chargement des objets
    modele_non_temporel = load_model_non_temporel(choix_model_non_temporel)
    model = modele_non_temporel["model"]
    X = X_test_temporel
    y = y_test_temporel

    best_threshold = modele_non_temporel["threshold"] #seuil optimisé lors de l'entrainement : préco pour l'évaluation


  #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  else :
    st.stop()
  #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


  # ----------------------------
  #  4 Evaluation 'commun aux modes temporel et non-temporel)
  # ----------------------------
  # Seuil de décision
  # Initialisation du slider du seuil
  if choix_preprocessing == "temporel" and  choix_model_temporel == "XGBoost Final":
    default_threshold = 0.38
  elif choix_preprocessing == "non-temporel" :
    default_threshold = best_threshold
  else :
    default_threshold = 0.5 #revient à faire model.predict(X)

  seuil = st.slider("🎯 Seuil de décision (classification)", 0.0, 1.0, step=0.01, value=default_threshold, key="seuil_slider")
  proba = model.predict_proba(X)[:, 1]
  y_pred = (proba >= seuil).astype(int)

  affichage_resultas_donnees_actuelles(X, y, y_pred)


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# -----------------------------
# Page 4 : Conclusions
# -----------------------------

if page == pages[3] :
  st.header("Conclusion")
