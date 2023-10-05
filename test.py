# # affiche les scores et les équipes de facon plus réaliste 

# import pandas as pd

# # Charger les données à partir de votre base de données
# data = pd.read_csv("rugby dataset.csv")

# # Demander à l'utilisateur de saisir les noms des équipes
# home_team = input("Nom de l'équipe à domicile : ")
# away_team = input("Nom de l'équipe à l'extérieur : ")

# # Filtrer les données pour les matchs impliquant les équipes sélectionnées
# home_matches = data[data["home_team"] == home_team]
# away_matches = data[data["away_team"] == away_team]

# # Calculer la moyenne des scores pour chaque équipe
# home_score_predicted = home_matches["home_score"].mean()
# away_score_predicted = away_matches["away_score"].mean()

# # Déterminer l'équipe gagnante prédite en fonction des scores prédits
# winner = home_team if home_score_predicted > away_score_predicted else away_team if away_score_predicted > home_score_predicted else "Match nul"

# # Afficher les résultats
# print("Prédictions pour le nouveau match:")
# print(f"{home_team} {round(home_score_predicted)} - {away_team} {round(away_score_predicted)}")
# print(f"Équipe gagnante prédite : {winner}")






































































# # meilleur code, il est parfait ! 

# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import LabelEncoder

# # Charger les données des matchs à partir de votre base de données
# match_data = pd.read_csv("rugby dataset.csv")

# # Charger les données météorologiques à partir de l'autre fichier CSV
# weather_data = pd.read_csv("donneenettoyer.csv")

# # Demander à l'utilisateur de saisir les informations sur le nouveau match
# home_team = input("Nom de l'équipe à domicile : ")
# away_team = input("Nom de l'équipe à l'extérieur : ")

# # Rechercher les scores des matchs précédents impliquant les mêmes équipes
# previous_matches_home = match_data[(match_data["home_team"] == home_team) & (match_data["away_team"] == away_team)]
# previous_matches_away = match_data[(match_data["home_team"] == away_team) & (match_data["away_team"] == home_team)]

# # Calculer les moyennes des scores précédents
# if len(previous_matches_home) > 0:
#     home_score_mean = previous_matches_home["home_score"].mean()
#     away_score_mean = previous_matches_home["away_score"].mean()
# else:
#     home_score_mean = 0
#     away_score_mean = 0

# if len(previous_matches_away) > 0:
#     away_score_mean = previous_matches_away["home_score"].mean()
#     home_score_mean = previous_matches_away["away_score"].mean()

# # Demander les données météorologiques à l'utilisateur
# avg_temp_c = float(input("Température moyenne (°C) : "))
# precipitation_mm = float(input("Précipitation (mm) : "))
# # Vous pouvez ajouter d'autres données météorologiques ici

# # Encodage des équipes en valeurs numériques
# label_encoder = LabelEncoder()
# match_data["home_team_encoded"] = label_encoder.fit_transform(match_data["home_team"])
# match_data["away_team_encoded"] = label_encoder.transform(match_data["away_team"])
# home_team_encoded = label_encoder.transform([home_team])
# away_team_encoded = label_encoder.transform([away_team])

# # Fusionner les données de match avec les données météorologiques en fonction de la date
# merged_data = pd.merge(match_data, weather_data, left_on="date", right_on="date", how="inner")

# # Créer un modèle de régression linéaire
# model = LinearRegression()

# # Caractéristiques pour la régression
# features = ["home_team_encoded", "away_team_encoded", "home_score", "away_score", "avg_temp_c", "precipitation_mm"]
# # Ajoutez d'autres caractéristiques météorologiques au besoin

# # Entraîner le modèle sur toutes les données disponibles
# model.fit(merged_data[features], merged_data["home_score"])

# # Créer un DataFrame pour le nouveau match avec les caractéristiques météorologiques
# new_match = pd.DataFrame({
#     "home_team_encoded": home_team_encoded,
#     "away_team_encoded": away_team_encoded,
#     "home_score": home_score_mean,
#     "away_score": away_score_mean,
#     "avg_temp_c": avg_temp_c,
#     "precipitation_mm": precipitation_mm
#     # Ajoutez d'autres caractéristiques météorologiques au besoin
# })

# # Prédire les scores du match
# home_score_predicted = model.predict(new_match[features])
# away_score_predicted = model.predict(new_match[features])

# # Déterminer l'équipe gagnante prédite en fonction des scores prédits
# winner = home_team if home_score_predicted > away_score_predicted else away_team if away_score_predicted > home_score_predicted else "Match nul"

# # Afficher les résultats
# print("Prédictions pour le nouveau match:")
# print(f"{home_team} {round(home_score_predicted[0])} - {away_team} {round(away_score_predicted[0])}")
# print(f"Équipe gagnante prédite : {winner}")





























































# #Affiche faux résultats

# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import LabelEncoder
# import random

# # Charger les données des matchs à partir de votre base de données
# match_data = pd.read_csv("rugby dataset.csv")

# # Charger les données météorologiques à partir de l'autre fichier CSV
# weather_data = pd.read_csv("donneenettoyer.csv")

# # Demander à l'utilisateur de saisir les informations sur le nouveau match
# home_team = input("Nom de l'équipe à domicile : ")
# away_team = input("Nom de l'équipe à l'extérieur : ")

# # Rechercher les scores des matchs précédents impliquant les mêmes équipes
# previous_matches_home = match_data[(match_data["home_team"] == home_team) & (match_data["away_team"] == away_team)]
# previous_matches_away = match_data[(match_data["home_team"] == away_team) & (match_data["away_team"] == home_team)]

# # Calculer les moyennes des scores précédents
# if len(previous_matches_home) > 0:
#     home_score_mean = previous_matches_home["home_score"].mean()
#     away_score_mean = previous_matches_home["away_score"].mean()
# else:
#     home_score_mean = 0
#     away_score_mean = 0

# if len(previous_matches_away) > 0:
#     away_score_mean = previous_matches_away["home_score"].mean()
#     home_score_mean = previous_matches_away["away_score"].mean()

# # Demander les données météorologiques à l'utilisateur
# avg_temp_c = float(input("Température moyenne (°C) : "))
# precipitation_mm = float(input("Précipitation (mm) : "))
# # Vous pouvez ajouter d'autres données météorologiques ici

# # Encodage des équipes en valeurs numériques
# label_encoder = LabelEncoder()
# match_data["home_team_encoded"] = label_encoder.fit_transform(match_data["home_team"])
# match_data["away_team_encoded"] = label_encoder.transform(match_data["away_team"])
# home_team_encoded = label_encoder.transform([home_team])
# away_team_encoded = label_encoder.transform([away_team])

# # Fusionner les données de match avec les données météorologiques en fonction de la date
# merged_data = pd.merge(match_data, weather_data, left_on="date", right_on="date", how="inner")

# # Créer un modèle de régression linéaire
# model = LinearRegression()

# # Caractéristiques pour la régression
# features = ["home_team_encoded", "away_team_encoded", "home_score", "away_score", "avg_temp_c", "precipitation_mm"]
# # Ajoutez d'autres caractéristiques météorologiques au besoin

# # Entraîner le modèle sur toutes les données disponibles
# model.fit(merged_data[features], merged_data["home_score"])

# # Créer un DataFrame pour le nouveau match avec les caractéristiques météorologiques
# new_match = pd.DataFrame({
#     "home_team_encoded": home_team_encoded,
#     "away_team_encoded": away_team_encoded,
#     "home_score": home_score_mean,
#     "away_score": away_score_mean,
#     "avg_temp_c": avg_temp_c,
#     "precipitation_mm": precipitation_mm
#     # Ajoutez d'autres caractéristiques météorologiques au besoin
# })

# # Prédire les scores du match
# home_score_predicted = model.predict(new_match[features]) + random.uniform(-1, 1)  # Ajout d'un facteur aléatoire
# away_score_predicted = model.predict(new_match[features]) + random.uniform(-1, 1)  # Ajout d'un facteur aléatoire

# # Déterminer l'équipe gagnante prédite en fonction des scores prédits
# winner = home_team if home_score_predicted > away_score_predicted else away_team if away_score_predicted > home_score_predicted else "Match nul"

# # Afficher les résultats
# print("Prédictions pour le nouveau match:")
# print(f"{home_team} {round(home_score_predicted[0])} - {away_team} {round(away_score_predicted[0])}")
# print(f"Équipe gagnante prédite : {winner}")
































# # affiche le gagnant comme je veux

# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import LabelEncoder

# # Charger les données des matchs à partir de votre base de données
# match_data = pd.read_csv("rugby dataset.csv")

# # Charger les données météorologiques à partir de l'autre fichier CSV
# weather_data = pd.read_csv("donneenettoyer.csv")

# # Demander à l'utilisateur de saisir les informations sur le nouveau match
# home_team = input("Nom de l'équipe à domicile : ")
# away_team = input("Nom de l'équipe à l'extérieur : ")

# # Rechercher les scores des matchs précédents impliquant les mêmes équipes
# previous_matches_home = match_data[(match_data["home_team"] == home_team) & (match_data["away_team"] == away_team)]
# previous_matches_away = match_data[(match_data["home_team"] == away_team) & (match_data["away_team"] == home_team)]

# # Calculer les moyennes des scores précédents
# if len(previous_matches_home) > 0:
#     home_score_mean = previous_matches_home["home_score"].mean()
#     away_score_mean = previous_matches_home["away_score"].mean()
# else:
#     home_score_mean = 0
#     away_score_mean = 0

# if len(previous_matches_away) > 0:
#     away_score_mean = previous_matches_away["home_score"].mean()
#     home_score_mean = previous_matches_away["away_score"].mean()

# # Demander les données météorologiques à l'utilisateur
# avg_temp_c = float(input("Température moyenne (°C) : "))
# precipitation_mm = float(input("Précipitation (mm) : "))
# # Vous pouvez ajouter d'autres données météorologiques ici

# # Encodage des équipes en valeurs numériques
# label_encoder = LabelEncoder()
# match_data["home_team_encoded"] = label_encoder.fit_transform(match_data["home_team"])
# match_data["away_team_encoded"] = label_encoder.transform(match_data["away_team"])
# home_team_encoded = label_encoder.transform([home_team])
# away_team_encoded = label_encoder.transform([away_team])

# # Fusionner les données de match avec les données météorologiques en fonction de la date
# merged_data = pd.merge(match_data, weather_data, left_on="date", right_on="date", how="inner")

# # Créer un modèle de régression linéaire pour prédire les scores des équipes
# home_score_model = LinearRegression()
# away_score_model = LinearRegression()

# # Caractéristiques pour la régression
# features = ["home_team_encoded", "away_team_encoded", "avg_temp_c", "precipitation_mm"]
# # Ajoutez d'autres caractéristiques météorologiques au besoin

# # Entraîner les modèles sur toutes les données disponibles
# home_score_model.fit(merged_data[features], merged_data["home_score"])
# away_score_model.fit(merged_data[features], merged_data["away_score"])

# # Créer un DataFrame pour le nouveau match avec les caractéristiques météorologiques
# new_match = pd.DataFrame({
#     "home_team_encoded": home_team_encoded,
#     "away_team_encoded": away_team_encoded,
#     "avg_temp_c": avg_temp_c,
#     "precipitation_mm": precipitation_mm
#     # Ajoutez d'autres caractéristiques météorologiques au besoin
# })

# # Prédire les scores du match pour les deux équipes
# home_score_predicted = home_score_model.predict(new_match[features])
# away_score_predicted = away_score_model.predict(new_match[features])

# # Déterminer l'équipe gagnante prédite en fonction des scores prédits
# winner = home_team if home_score_predicted > away_score_predicted else away_team if away_score_predicted > home_score_predicted else "Match nul"

# # Afficher les résultats
# print("Prédictions pour le nouveau match:")
# print(f"{home_team} {round(home_score_predicted[0])} - {away_team} {round(away_score_predicted[0])}")
# print(f"Équipe gagnante prédite : {winner}")


























import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Charger les données des matchs à partir de votre base de données
match_data = pd.read_csv("rugby dataset.csv")

# Charger les données météorologiques à partir de l'autre fichier CSV
weather_data = pd.read_csv("donneenettoyer.csv")

# Demander à l'utilisateur de saisir les informations sur le nouveau match
home_team = input("Nom de l'équipe à domicile : ")
away_team = input("Nom de l'équipe à l'extérieur : ")

# Rechercher les scores des matchs précédents impliquant les mêmes équipes
previous_matches_home = match_data[(match_data["home_team"] == home_team) & (match_data["away_team"] == away_team)]
previous_matches_away = match_data[(match_data["home_team"] == away_team) & (match_data["away_team"] == home_team)]

# Calculer les moyennes des scores précédents
if len(previous_matches_home) > 0:
    home_score_mean = previous_matches_home["home_score"].mean()
    away_score_mean = previous_matches_home["away_score"].mean()
else:
    home_score_mean = 0
    away_score_mean = 0

if len(previous_matches_away) > 0:
    away_score_mean = previous_matches_away["home_score"].mean()
    home_score_mean = previous_matches_away["away_score"].mean()

# Demander les données météorologiques à l'utilisateur
avg_temp_c = float(input("Température moyenne (°C) : "))
precipitation_mm = float(input("Précipitation (mm) : "))
# Vous pouvez ajouter d'autres données météorologiques ici

# Encodage des équipes en valeurs numériques
label_encoder = LabelEncoder()
match_data["home_team_encoded"] = label_encoder.fit_transform(match_data["home_team"])
match_data["away_team_encoded"] = label_encoder.transform(match_data["away_team"])
home_team_encoded = label_encoder.transform([home_team])
away_team_encoded = label_encoder.transform([away_team])

# Fusionner les données de match avec les données météorologiques en fonction de la date
merged_data = pd.merge(match_data, weather_data, left_on="date", right_on="date", how="inner")

# Créer un modèle de régression linéaire pour prédire les scores des équipes
home_score_model = LinearRegression()
away_score_model = LinearRegression()

# Caractéristiques pour la régression
features = ["home_team_encoded", "away_team_encoded", "avg_temp_c", "precipitation_mm"]
# Ajoutez d'autres caractéristiques météorologiques au besoin

# Entraîner les modèles sur toutes les données disponibles
home_score_model.fit(merged_data[features], merged_data["home_score"])
away_score_model.fit(merged_data[features], merged_data["away_score"])

# Créer un DataFrame pour le nouveau match avec les caractéristiques météorologiques
new_match = pd.DataFrame({
    "home_team_encoded": home_team_encoded,
    "away_team_encoded": away_team_encoded,
    "avg_temp_c": avg_temp_c,
    "precipitation_mm": precipitation_mm
    # Ajoutez d'autres caractéristiques météorologiques au besoin
})

# Prédire les scores du match pour les deux équipes
home_score_predicted = home_score_model.predict(new_match[features])
away_score_predicted = away_score_model.predict(new_match[features])

# Déterminer l'équipe gagnante prédite en fonction des scores prédits
winner = home_team if home_score_predicted > away_score_predicted else away_team if away_score_predicted > home_score_predicted else "Match nul"

# Calculer la cote (pourcentage de chance) du vainqueur
if winner == home_team:
    win_percentage = (home_score_predicted[0] / (home_score_predicted[0] + away_score_predicted[0])) * 100
elif winner == away_team:
    win_percentage = (away_score_predicted[0] / (home_score_predicted[0] + away_score_predicted[0])) * 100
else:
    win_percentage = 0

# Afficher les résultats
print("Prédictions pour le nouveau match:")
print(f"{home_team} {round(home_score_predicted[0])} - {away_team} {round(away_score_predicted[0])}")
print(f"Équipe gagnante prédite : {winner}")
print(f"Pourcentage de chance du {winner} : {win_percentage:.2f}%")
