# import pandas as pd

# # Charger les données des matchs à partir du premier fichier CSV
# match_data = pd.read_csv("rugby dataset.csv")

# # Charger les données météorologiques à partir du deuxième fichier CSV
# weather_data = pd.read_csv("donneenettoyer.csv")

# # Fusionner les deux ensembles de données en fonction de la colonne "date" (ou toute autre colonne commune)
# merged_data = pd.merge(match_data, weather_data, on="date", how="inner")

# # Enregistrer les données fusionnées dans un nouveau fichier CSV
# merged_data.to_csv("donnees_fusionnees.csv", index=False)









import pandas as pd

# Charger le fichier CSV
data = pd.read_csv("rugby dataset.csv")

# Ajouter une colonne "gagnant" qui contient le nom de l'équipe gagnante pour chaque match
data['gagnant'] = data.apply(lambda row: row['home_team'] if row['home_score'] > row['away_score'] else row['away_team'], axis=1)

# Enregistrer le fichier avec la nouvelle colonne
data.to_csv("fichier_modifie.csv", index=False)
