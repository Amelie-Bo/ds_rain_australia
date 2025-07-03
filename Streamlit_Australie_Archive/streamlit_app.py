%%writefile streamlit_app.py
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Chargement des librairies
import streamlit as st

import pandas as pd
import numpy as np
from scipy.stats import randint

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, f1_score, classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, precision_recall_fscore_support, brier_score_loss)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.decomposition import PCA

from imblearn.metrics import classification_report_imbalanced
import xgboost as xgb

import streamlit_shap
from streamlit_shap import st_shap
import shap

import pickle
from joblib import load
import cloudpickle

import os
import time
import requests
from io import StringIO
from datetime import datetime
from dateutil.relativedelta import relativedelta
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 0. Cache #Comment cela fonctionne?
#Par exemple pour la fonction_pickle, elle sera utilisée plusieurs fois, mais comment mettre en cache chacune des fois ou le cache est utilisé?
@st.cache_data
def load_dataset(path):
    return pd.read_csv(path)

@st.cache_resource
def load_model(path):
    return load(path)

@st.cache_resource # A utiliser
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 1. Charger les données
df = load_dataset("data/weatherAUS.csv")
liste_colonne_df = df.columns #servira à ordonner les colonnes dans les données actuelles
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 2. Définir la structure
st.title("Projet de classification binaire sur la pluie en Australie") # sera répercuté sur toutes les pages du Streamlit
st.sidebar.title("Sommaire")
pages=["Le projet", "Exmploration", "DataVizualization", "Preprocessing", "Modélisation et Evaluation","Datas actuelles", "Conclusion"]
page=st.sidebar.radio("Aller vers", pages)
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 3. Sur la page de présentation du projet
if page == pages[0] :
  st.header("Intro")
  st.write("Ce projet traite... ") #\n n'apporte rien autant faire un nouveau st.write

  # test visuel
  st.title("st.titre") #Gras, 44px
  st.header("st.header") #36px semi gras, comme ##
  st.subheader("st.subheader") #28px semi gras, comme ###
  st.write("#### 4#..") #24px semi gras
  st.write("##### 5#..") #20px semi gras
  st.write("###### 6#..") #16px semi gras
  st.write("normal") #16px normal (comme un print dans une jupyer NB)
  st.markdown("Texte **gras**, *italique*, et un [lien](https://example.com)") #pour mettre en forme


  st.dataframe(df.head(10))
  st.write(df.shape) #equivalent de print
  st.dataframe(df.describe()) #st.dataframe pour appeler des méthodes pandas qui entraine un affichage de df

  if st.checkbox("Afficher les NA") : #quand on coche la case, on affiche la méthode ci-dessous
    st.dataframe(df.isna().sum())
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 4. Sur la page Exploration du jeu de données (présentation des variables, des manquants)
#Sur le fichier source? sur le nouveau jeu de données?
if page == pages[1] :

  st.header("Exploration")

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 5. Sur la page Datavizualisation
#Sur le fichier source? sur le nouveau jeu de données?
if page == pages[2] :
  st.header("DataVizualization")
  #Afficher un graphique de la variable cible "Pluie demain"
  fig = plt.figure()
  sns.countplot(x = 'RainTomorrow', data = df)
  st.pyplot(fig)

  # Impact de features sur la variable cible
  fig1 = plt.figure()
  sns.countplot(x = 'RainTomorrow', hue='RainToday', data = df, title = "lien Variable cible et RainToday")
  st.pyplot(fig1)

  fig2 = sns.catplot(x='Cloud3pm', y='RainTomorrow', data=df, kind='point')
  st.pyplot(fig2)

  fig3 = sns.lmplot(x='Temp3pm', y='RainTomorrow', hue="Cloud3pm", data=df)
  st.pyplot(fig3)

  # Analyse multivariée par matrice de corrélation
  fig, ax = plt.subplots()
  sns.heatmap(df.corr(), ax=ax)
  st.write(fig)

  #Plotly
  # fig4 = px.scatter(df, x=, y=, title="")
  st.plotly_chart(fig4) #ADD

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 6. Sur la page de présentation du Preprocessing
if page == pages[3] :
  st.header("Preprocessing")

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 7. Sur la page Modélisation et Evaluation
if page == pages[4] :
  st.header("Modélisation")# sur X_test preprocesse ou non?(mon preprocessing + modelisationprend qq minutes )
  st.header("Evaluation")

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 8. Sur la page utilisant les données actuellement sur le site du BOM
if page == pages[5] :
  st.header("Datas actuelles")
#-1. Collecte des données actuelles---------------------------------------------------------------------------------------------------------------------------------
  #1.1 Initialisation
  ##1.1.1 Liste des mois
  mois_courant = datetime.now().replace(day=1)
  mois_depart = mois_courant - relativedelta(months=1) # Mois de départ = mois précédent
  # Générer les 13 mois vers le passé (de -12 à 0 mois avant mois_depart)
  liste_mois_a_selectionner = [(mois_depart - relativedelta(months=i)).strftime("%Y%m") for i in reversed(range(13))]

  ##1.1.2 Dico stations
  with open("dico_scaler/dico_station.pkl", "rb") as fichier:
    dico_stations_BOM = pickle.load(fichier)

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
  st.subheader("Sélection ")

  liste_mois = st.multiselect("Sélectionnez un mois", liste_mois_a_selectionner)

  ## 1.2.1 Afficher le nom des stations dans la liste déroulante
  # Multiselect avec affichage du nom de la station
  stations_selectionnees = st.multiselect(
      "Sélectionnez une ou plusieurs stations",
      options=list(dico_stations_BOM.keys()),
      format_func=lambda x: dico_stations_BOM[x][2])
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
      df_une_station.drop(df_une_station.index[-1], inplace=True)

      #Mettre les colonnes dans le même ordre que le df de départ du projet
      df_une_station = df_une_station[liste_colonne_df]


      #Faire un df consolidé (df_conso_station) des df unitaire par station (df_une_station)
      if compteur == 1 :
          df_conso_station = df_une_station
      else :
          df_conso_station = pd.concat([df_conso_station, df_une_station], ignore_index=True)


  #-2. Preprocessing de base---------------------------------------------------------------------------------------------------------------------------------
  df_X_y_test = df_conso_station

  ## 2.1 Modification de la vitesse "Calm" par 0km/h
  df_X_y_test["WindSpeed9am"] = df_X_y_test["WindSpeed9am"].apply(lambda x: 0 if x =="Calm" else x)
  df_X_y_test["WindSpeed3pm"] = df_X_y_test["WindSpeed3pm"].apply(lambda x: 0 if x =="Calm" else x)
  df_X_y_test["WindGustSpeed"] = df_X_y_test["WindGustSpeed"].apply(lambda x: 0 if x =="Calm" else x)

  ## 2.2 Suprresion 25% des NAN
  # === Calcul du ratio de NaN ===
  total_cells_per_location = df_X_y_test.groupby("Location").size() * (df_X_y_test.shape[1] - 1)  # -1 car on exclut 'Location'
  nan_counts_per_location = df_X_y_test.drop(columns="Location").isna().groupby(df_X_y_test["Location"]).sum().sum(axis=1)
  nan_ratio = nan_counts_per_location / total_cells_per_location
  # === Filtrage des stations valides ===
  valid_locations = nan_ratio[nan_ratio <= 0.25].index.to_list() #>>ex : {'BadgerysCreek', 'Albury'}
  df_X_y_test = df_X_y_test[df_X_y_test["Location"].isin(valid_locations)]

  # === Messages Streamlit ===
  # noms des stations selectionné par l'utilisateur
  stations_selectionnees_noms = [dico_stations_BOM[code][2] for code in stations_selectionnees]  #>> ex : 0:"Penrith" 1:"AliceSprings"
  stations_supprimees = sorted(set(stations_selectionnees_noms) - set(valid_locations)) #>> Stations sélectionnées : Penrith, AliceSprings
  if len(valid_locations) == 0:
    st.error("Toutes les stations sélectionnées ont plus de 25% de données manquantes. Veuillez en choisir d'autres.")
    st.stop()
  elif len(stations_supprimees) > 0:
      st.warning(f"Les stations suivantes ont été exclues car elles contiennent plus de 25% de données manquantes : {', '.join(stations_supprimees)}")


  ## 2.3 Ajout de la latitude et de la longitude
  with open("dico_scaler/dico_station_geo.pkl", "rb") as fichier:
    dico_charge = pickle.load(fichier)
  df_dico_station_geo = pd.DataFrame.from_dict(dico_charge, orient="index",columns=["Lat", "Lon"])
  df_dico_station_geo.columns = ["Latitude", "Longitude"]
  df_X_y_test = df_X_y_test.merge(right=df_dico_station_geo, left_on="Location", right_index=True, how="left")

  ## 2.4 Date, Saison
  df_X_y_test["Date"]=pd.to_datetime(df_X_y_test["Date"], format = "%Y-%m-%d")
  df_X_y_test["Month"] = df_X_y_test['Date'].dt.month
  df_X_y_test["Year"] = df_X_y_test['Date'].dt.year
  df_X_y_test["Saison"] = df_X_y_test["Month"].apply( lambda x : "Eté" if x in [12, 1, 2] else "Automne" if x in [3, 4, 5] else "Hiver" if x in [6, 7, 8] else "Printemps")

  ## 2.5 Ajout du climat
  climat_mapping = pd.read_csv("dico_scaler/climat_mapping.csv", index_col="Location")
  climat_mapping_series = climat_mapping.squeeze()  # Convertir en Series pour faciliter le mapping
  df_X_y_test['Climat'] = df_X_y_test["Location"].map(climat_mapping_series) #pour chaque valeur de df.Location, on récupère la valeur correspondante dans climat_mapping

  ## 2.6 Suppression des features
  df_X_y_test = df_X_y_test.drop(["Sunshine","Evaporation"], axis = 1)

  ## 2.7 Traitement de la variable cible : Suppression des NaN et Label Encoder
  df_X_y_test = df_X_y_test.dropna(subset=["RainTomorrow"], axis=0, how="any")
  encoder=LabelEncoder()
  df_X_y_test["RainTomorrow"] = encoder.fit_transform(df_X_y_test["RainTomorrow"])  #N=0, Y=1

  ## 2.8 Choix du jour à prédire
  # /!\ ne pas pouvoir sélectionner la dernière valeur chronologique de Date, sinon nous ne pourrons pas montrer la valeur le lendemain. ce qui est le but de notre prédiction.
  # Menu déroulant pour sélectionner un jour
  st.subheader("Sélectionnez le jour à prédire")

  # liste de dates en excluant la date la plus récente
  dates_uniques = df_X_y_test["Date"].unique()
  date_plus_recente = dates_uniques.max()
  dates_a_afficher = [date for date in dates_uniques if date != date_plus_recente]

  date_selectionnee = st.selectbox("Sélectionnez une date", ["--- Sélectionner ---"] + list(dates_a_afficher))
  if date_selectionnee == "--- Sélectionner ---": # Bloquer l'exécution si aucun modèle n'est sélectionné
    st.stop()

  ## 2.9 Journée à prédire
  df_X_y_test = df_X_y_test[(df_X_y_test["Date"] == date_selectionnee)]
  st.write("Données à prédire")
  st.dataframe(df_X_y_test.head(6)) # df_X_y_test fait 2 lignes * nb de stations sélectionnées.

  #-3 Choix du modèle--------------------------------------------------------------------------------------------------------------------------------------------------------------
  st.header("Choix du preprocessing")
  #Choix par radio bouton entre Logique d'entrainement temporelle ou non
  choix_preprocessing = st.selectbox("Choix entre Logique d'entrainement temporelle ou non",["temporel", "non-temporel"])

  #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  if choix_preprocessing == "temporel" :
    # Liste des modèles
    liste_modele_temporel = {
    "modele_florent_1": "models/xgb_25features_model.joblib",
    "modele_florent_2": "models/xgb_25features_model.joblib",}

    # Sélection des modèles entrainés
    choix_model_temporel = st.selectbox("Choix du modèle",["--- Sélectionner ---", "modele_florent_1", "modele_florent_2"])

    if choix_model_temporel == "--- Sélectionner ---": # Bloquer l'exécution si aucun modèle n'est sélectionné
        st.stop()

    # Chargement du modèle
    modele_non_temporel = load_model(liste_modele_temporel[choix_model_temporel])

    ### 3.1.A Preprocessing Florent------------------------------------------------------------------------------------------------------------------------------------------------
    do_preprocess = st.checkbox("Lancer preprocessing")
    if not do_preprocess:
        st.stop()

    #Ton code pour préprocessé ce df de test

    ### 3.1.B Modelisation Florent------------------------------------------------------------------------------------------------------------------------------------------------
    do_predict = st.checkbox("Lancer prédiction")
    if not do_predict:
        st.stop()
    # Ton code pour prédire sur test


 #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  elif choix_preprocessing == "non-temporel" :
    # Liste des modèles
    liste_modele_non_temporel = {
    "Régression logistique": "models/LogReg_X_train_normal_model_and_threshold.joblib",
    "XGB Classifier": "models/XGBClassifier_X_train_model_and_threshold.joblib",
    "RNN": "models/RNN_ABO_X_scaled_normal_model_and_threshold.joblib"}
    # Affichage des modèles entrainés
    choix_model_non_temporel = st.selectbox("Choix du modèle",["--- Sélectionner ---", "Régression logistique", "XGB Classifier", "RNN"])

    if choix_model_non_temporel == "--- Sélectionner ---": # Bloquer l'exécution si aucun modèle n'est sélectionné
        st.stop()

    # Chargement du modèle
    modele_non_temporel = load_model(liste_modele_non_temporel[choix_model_non_temporel])


    ### 3.2.A Preprocessing Amelie------------------------------------------------------------------------------------------------------------------------------------------------
    do_preprocess = st.checkbox("Lancer preprocessing")
    if not do_preprocess:
        st.stop()

    #### 3.2.A.1 Suppresion des features avec trop de manquants
    df_X_y_test = df_X_y_test.drop(["RainToday","Saison","Climat"], axis = 1)

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
    with open("dico_scaler/cloud_imputer.pkl", "rb") as f:
        transformer_cloud = cloudpickle.load(f)
    X_test_temporel = transformer_cloud.transform(X_test_temporel)
    #### 3.2.A.6.B Complétion des autres NAN
    with open("dico_scaler/transformer_KNNImputerABO.pkl", "rb") as f:
        transformer = cloudpickle.load(f)
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
                                 import_path: str = "dico_scaler/scalers.joblib") -> pd.DataFrame:
          artefact = load(import_path)
          scalers, feats = artefact['scalers'], artefact['feature_lists']
          X_scaled = X_fe.copy()
          for key, cols in feats.items():
              X_scaled[cols] = scalers[key].transform(X_scaled[cols])
          return X_scaled
    ####-----Fin Fonction-----------------------------------------------------------------------------------------------------------------------------------------------

    # Ajout des features d'ingénierie
    X_test_fe= add_engineered_features(X_test_temporel, ref_year=2007, lat0=-25.0, lon0=133.0)
    # Chargement et application des scalers (en paramètre le df obtenu avant)
    X_test_temporel  = load_and_apply_scalers(X_test_fe,  import_path="dico_scaler/scalers.joblib")

    #Aperçu des features en fin de preprocessing
    st.write("Aperçu des features en fin de preprocessing")
    st.dataframe(X_test_temporel.head(3))

    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ### 3.2.B Modelisation Amelie------------------------------------------------------------------------------------------------------------------------------------------------
    do_predict = st.checkbox("Lancer prédiction")
    if not do_predict:
        st.stop()

    best_model     = modele_non_temporel["model"]
    best_threshold = modele_non_temporel["threshold"]

    # et pour prédire sur X_new :
    y_proba = best_model.predict_proba(X_test_temporel)[:,1]
    y_pred  = (y_proba >= best_threshold).astype(int)


  #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  else :
    st.stop()
  #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



  # 4 Evaluation------------------------------------------------------------------------------------------------------------------------------------------------------------------
  #-----Fonction Evalutaion----------------------------------------------------------------------------------------------------------------------------------------------------------
  def evaluation_streamlit(y_test, y_pred, y_proba, model_name, best_threshold=None):
      acc = accuracy_score(y_test, y_pred)
      f1 = f1_score(y_test, y_pred)
      f1_positive = f1_score(y_test, y_pred, pos_label=1)
      roc_auc = roc_auc_score(y_test, y_proba)

      st.subheader(f"Évaluation du modèle : {model_name}")
      if best_threshold is not None:
          st.write(f"**F1-score**: {f1:.3f} | **Accuracy**: {acc:.3f} | **Seuil**: {best_threshold:.2f}")
      else:
          st.write(f"**F1-score**: {f1:.3f} | **Accuracy**: {acc:.3f}")

      st.markdown("### Rapport de classification")
      st.text(classification_report(y_test, y_pred))

      st.markdown("### Rapport déséquilibre (Imbalanced)")
      st.text(classification_report_imbalanced(y_test, y_pred))

      ## Création de la figure
      fig, axes = plt.subplots(1, 2, figsize=(12, 5))

      # Matrice de confusion
      cm = confusion_matrix(y_test, y_pred)
      disp = ConfusionMatrixDisplay(confusion_matrix=cm)
      disp.plot(ax=axes[0], cmap="Blues", values_format="d", colorbar=False)
      axes[0].set_title("Matrice de Confusion")

      # Courbe ROC
      fpr, tpr, _ = roc_curve(y_test, y_proba)
      axes[1].plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", color="darkorange")
      axes[1].plot([0, 1], [0, 1], linestyle='--', color="gray")
      axes[1].set_xlabel("Taux de faux positifs")
      axes[1].set_ylabel("Taux de vrais positifs")
      axes[1].set_title("Courbe ROC")
      axes[1].legend()
      axes[1].grid(True)

      # Mise en page et affichage dans Streamlit
      fig.suptitle(f"Évaluation du modèle : {model_name}", fontsize=14)
      fig.text(
          0.5, 0.88,
          f"F1-score (classe positive) : {f1_positive:.3f}",
          ha='center',
          fontsize=12
      )
      plt.tight_layout(rect=[0, 0.03, 1, 0.95])
      st.pyplot(fig)

   #-----Fin Fonction----------------------------------------------------------------------------------------------------------------------------------------------------------
  st.header("Prédictions puis évaluation")

  evaluation_streamlit(y_test_temporel, y_pred, y_proba, choix_model_non_temporel)

  # 5 Interprétation--------------------------------------------------------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 9. Sur la page de présentation du Preprocessing
if page == pages[6] :
  st.header("Conclusion")
