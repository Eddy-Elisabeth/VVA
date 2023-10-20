# import pandas as pd

# # Charger les données des matchs à partir du premier fichier CSV
# match_data = pd.read_csv("rugby dataset.csv")

# # Charger les données météorologiques à partir du deuxième fichier CSV
# weather_data = pd.read_csv("donneenettoyer.csv")

# # Fusionner les deux ensembles de données en fonction de la colonne "date" (ou toute autre colonne commune)
# merged_data = pd.merge(match_data, weather_data, on="date", how="inner")

# # Enregistrer les données fusionnées dans un nouveau fichier CSV
# merged_data.to_csv("donnees_fusionnees.csv", index=False)









# import pandas as pd

# # Charger le fichier CSV
# data = pd.read_csv("rugby dataset.csv")

# # Ajouter une colonne "gagnant" qui contient le nom de l'équipe gagnante pour chaque match
# data['gagnant'] = data.apply(lambda row: row['home_team'] if row['home_score'] > row['away_score'] else row['away_team'], axis=1)

# # Enregistrer le fichier avec la nouvelle colonne
# data.to_csv("fichier_modifie.csv", index=False)






# import pandas as pd

# # Charger le fichier CSV
# data = pd.read_csv("donnees_fusionnees.csv")

# # Calculer le score moyen pour chaque match
# data['score_moyen'] = (data['home_score'] + data['away_score']) / 2

# # Ajouter une colonne pour indiquer le résultat du match (Victoire, Égalité, Défaite)
# def resultat_match(row):
#     if row['home_score'] > row['away_score']:
#         return 'Victoire ' + row['home_team']
#     elif row['home_score'] < row['away_score']:
#         return 'Victoire ' + row['away_team']
#     else:
#         return 'Égalité'

# data['resultat'] = data.apply(resultat_match, axis=1)

# # Enregistrer le fichier avec les nouvelles colonnes
# data.to_csv("bonhomme.csv", index=False)










import pandas as pd

# Charger le fichier CSV
data = pd.read_csv("bonhomme.csv")

# Calculer la moyenne des précipitations par ville
moyenne_precipitations_ville = data.groupby('stadium')['precipitation_mm'].mean()

# Ajouter la colonne "moyenne_precipitations_ville" au DataFrame d'origine
data = data.merge(moyenne_precipitations_ville, left_on='stadium', right_index=True, suffixes=('', '_moyenne'))

# # Calculer la moyenne des précipitations par date
# moyenne_precipitations_date = data.groupby('date')['precipitation_mm'].mean()

# # Ajouter la colonne "moyenne_precipitations_date" au DataFrame d'origine
# data = data.merge(moyenne_precipitations_date, left_on='date', right_index=True, suffixes=('', '_moyenne'))

# Enregistrer le fichier avec les nouvelles colonnes
data.to_csv("bonhomme.csv", index=False)
