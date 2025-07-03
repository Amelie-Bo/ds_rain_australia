import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("weatherAUS.csv")

st.title("Projet de classification binaire sur la pluie en Australie")
st.sidebar.title("Sommaire")
pages=["Exploration", "DataVizualization", "Modélisation"]
page=st.sidebar.radio("Aller vers", pages)

if page == pages[0] : #Sur la page Exploration
  st.write("### Introduction")
  st.dataframe(df.head(10))
  st.write(df.shape) #equivalent de print
  st.dataframe(df.describe()) #st.dataframe pour appeler des méthodes pandas qui entraine un affichage de df
  
  if st.checkbox("Afficher les NA") : #quand on coche la case, on affiche la méthode ci-dessous
    st.dataframe(df.isna().sum()) 


if page == pages[1] : 
  st.write("### DataVizualization")
  #Afficher un graphique de la variable cible "Plui demain"
  fig = plt.figure()
  sns.countplot(x = 'RainTomorrow', data = df)
  st.pyplot(fig)

  # Analyse descriptive des variables
  fig = plt.figure()
  sns.displot(x = 'Age', data = df)
  plt.title("Distribution de l'âge des passagers")
  st.pyplot(fig)

  # Impact des facteurs sur la variable cible "Survived" 
  #fera bien 3graphiques différents (ce ne sont pas des subplots, meme si il n'y a pas plt.figure() avant chaque sns)
  fig = plt.figure()
  sns.countplot(x = 'Survived', hue='Sex', data = df)
  st.pyplot(fig)
  
  fig = sns.catplot(x='Pclass', y='Survived', data=df, kind='point')
  st.pyplot(fig)
  
  fig = sns.lmplot(x='Age', y='Survived', hue="Pclass", data=df)
  st.pyplot(fig)


  # Analyse multivariée par matrice de corrélation
  fig, ax = plt.subplots()
  sns.heatmap(df.corr(), ax=ax)
  st.write(fig)


if page == pages[2] : 
  st.write("### Modélisation")# sur X_test

  # Supprimer les variables inutiles
  df = df.drop(['Evaporation', 'Sunshine'], axis=1) 

  # Créer un df variable cible et deux df variables explicatives : un catégorique et un numérique


  # Compléter les valeurs manquantes

  # Encoder les variables catégorielles (scaling)


  # Séparer les données en train et test


  # Standardiser les données sur les colonnes numériques issues de X_num


  from sklearn.ensemble import RandomForestClassifier
  from sklearn.svm import SVC
  from sklearn.linear_model import LogisticRegression
  from sklearn.metrics import confusion_matrix
  def prediction(classifier):
      if classifier == 'Random Forest':
          clf = RandomForestClassifier()
      elif classifier == 'SVC':
          clf = SVC()
      elif classifier == 'Logistic Regression':
          clf = LogisticRegression()
      clf.fit(X_train, y_train)
      return clf
  
  # métrique. ici accuracy car classe équilibrées
  def scores(clf, choice):
    if choice == 'Accuracy':
        return clf.score(X_test, y_test)
    elif choice == 'Confusion matrix':
        return confusion_matrix(y_test, clf.predict(X_test))
    
  # Choisir le classifier
  choix = ['Random Forest', 'SVC', 'Logistic Regression']
  option = st.selectbox('Choix du modèle', choix) #Menu déroulant pour choisir le modèle
  st.write('Le modèle choisi est :', option)

  # Choisir la métrique à afficher  (bouton radio: un seul choix possible)
  clf = prediction(option) # entraînement du modèle
  display = st.radio('Que souhaitez-vous montrer ?', ('Accuracy', 'Confusion matrix'))
  if display == 'Accuracy': #sur X_test
    st.write(scores(clf, display))
  elif display == 'Confusion matrix': #lance la prédiction sur X_test
    st.dataframe(scores(clf, display))
