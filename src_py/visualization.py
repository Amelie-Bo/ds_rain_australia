# visualization.py
#pas utiliser pour le moment
# Ce module contient des fonctions de visualisation
import pandas as pd
import plotly.express as px

def map_rainfall_by_station(df: pd.DataFrame,
                            localisations: dict) -> px.scatter_geo:
    """
    Prend en entrée :
      - df : DataFrame avec au moins une colonne "Location" et "Rainfall"
      - localisations : dict {station_name: (lat, lon)}
    Retourne une figure Plotly scatter_geo des précipitations moyennes.
    """

    # 1) Construction du DataFrame loc_df
    loc_df = pd.DataFrame([
        {"Location": station, "Latitude": lat, "Longitude": lon}
        for station, (lat, lon) in localisations.items()
    ])

    # 2) Merge avec df
    df2 = df.merge(loc_df, how="left", on="Location")

    # 3) Calcul de la pluie moyenne par station
    avg = (
      df2
      .groupby("Location", as_index=False)["Rainfall"]
      .mean()
      .rename(columns={"Rainfall": "AvgRain"})
      .merge(loc_df, on="Location", how="left")
    )

    # 4) Création de la figure
    fig = px.scatter_geo(
        avg,
        lat="Latitude",
        lon="Longitude",
        text="Location",
        size="AvgRain",
        color="AvgRain",
        title="Précipitations moyennes par stations",
        hover_name="Location",
        width=800,      # ou 1500
        height=800      # ou 1500
    )

    fig.update_geos(
        center={"lat": -25.0, "lon": 133.0},
        projection_scale=4.2,
        showcountries=True,
        showcoastlines=True,
        showframe=False
    )
    fig.update_layout(
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        coloraxis_colorbar=dict(
            title="Précipitations (mm)",
            x=0.9,
            y=0.8,
            len=0.3,
            thickness=15
        )
    )

    return fig
