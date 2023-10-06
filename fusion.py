import pandas as pd

# Charger les données des matchs à partir du premier fichier CSV
match_data = pd.read_csv("rugby dataset.csv")

# Charger les données météorologiques à partir du deuxième fichier CSV
weather_data = pd.read_csv("donneenettoyer.csv")

# Fusionner les deux ensembles de données en fonction de la colonne "date" (ou toute autre colonne commune)
merged_data = pd.merge(match_data, weather_data, on="date", how="inner")

# Enregistrer les données fusionnées dans un nouveau fichier CSV
merged_data.to_csv("donnees_fusionnees.csv", index=False)
