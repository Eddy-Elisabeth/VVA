# from flask import Flask, render_template, request
# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import LabelEncoder

# app = Flask(__name__)

# # Charger les données des matchs à partir de votre base de données
# match_data = pd.read_csv("rugby dataset.csv")

# # Charger les données météorologiques à partir de l'autre fichier CSV
# weather_data = pd.read_csv("donneenettoyer.csv")

# # Encodage des équipes en valeurs numériques
# label_encoder = LabelEncoder()
# match_data["home_team_encoded"] = label_encoder.fit_transform(match_data["home_team"])
# match_data["away_team_encoded"] = label_encoder.transform(match_data["away_team"])

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

# @app.route("/", methods=["GET", "POST"])
# def predict_score():
#     if request.method == "POST":
#         home_team = request.form["home_team"]
#         away_team = request.form["away_team"]
#         avg_temp_c = float(request.form["avg_temp_c"])
#         precipitation_mm = float(request.form["precipitation_mm"])

#         # Encodage des équipes saisies par l'utilisateur en valeurs numériques
#         home_team_encoded = label_encoder.transform([home_team])
#         away_team_encoded = label_encoder.transform([away_team])

#         # Créer un DataFrame pour le nouveau match avec les caractéristiques météorologiques
#         new_match = pd.DataFrame({
#             "home_team_encoded": home_team_encoded,
#             "away_team_encoded": away_team_encoded,
#             "avg_temp_c": avg_temp_c,
#             "precipitation_mm": precipitation_mm
#             # Ajoutez d'autres caractéristiques météorologiques au besoin
#         })

#         # Prédire les scores du match pour les deux équipes
#         home_score_predicted = home_score_model.predict(new_match[features])
#         away_score_predicted = away_score_model.predict(new_match[features])

#         # Déterminer l'équipe gagnante prédite en fonction des scores prédits
#         winner = home_team if home_score_predicted > away_score_predicted else away_team if away_score_predicted > home_score_predicted else "Match nul"

#         # Calculer la cote (pourcentage de chance) du vainqueur
#         if winner == home_team:
#             win_percentage = (home_score_predicted[0] / (home_score_predicted[0] + away_score_predicted[0])) * 100
#         elif winner == away_team:
#             win_percentage = (away_score_predicted[0] / (home_score_predicted[0] + away_score_predicted[0])) * 100
#         else:
#             win_percentage = 0

#         return render_template("result.html", home_team=home_team, away_team=away_team,
#                                home_score=round(home_score_predicted[0]), away_score=round(away_score_predicted[0]),
#                                winner=winner, win_percentage=win_percentage)
#     return render_template("index.html")

# if __name__ == "__main__":
#     app.run(debug=True)













# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import LabelEncoder
# from flask import Flask, render_template, request

# # Charger les données des matchs à partir de votre base de données
# match_data = pd.read_csv("rugby dataset.csv")

# # Charger les données météorologiques à partir de l'autre fichier CSV
# weather_data = pd.read_csv("donneenettoyer.csv")

# # Initialiser l'application Flask
# app = Flask(__name__)

# @app.route("/", methods=["GET", "POST"])
# def home():
#     # Obtenir la liste des équipes disponibles à partir de la base de données
#     teams = match_data["home_team"].unique()

#     if request.method == "POST":
#         # Récupérer les données du formulaire
#         home_team = request.form["home_team"]
#         away_team = request.form["away_team"]
#         avg_temp_c = float(request.form["avg_temp_c"])
#         precipitation_mm = float(request.form["precipitation_mm"])

#         # Rechercher les scores des matchs précédents impliquant les mêmes équipes
#         previous_matches_home = match_data[(match_data["home_team"] == home_team) & (match_data["away_team"] == away_team)]
#         previous_matches_away = match_data[(match_data["home_team"] == away_team) & (match_data["away_team"] == home_team)]

#         # ... (le reste du code pour la prédiction)

#     return render_template("index.html", teams=teams)

# if __name__ == "__main__":
#     app.run(debug=True)






















import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, request

# Charger les données des matchs à partir de votre base de données
match_data = pd.read_csv("rugby dataset.csv")

# Charger les données météorologiques à partir de l'autre fichier CSV
weather_data = pd.read_csv("donneenettoyer.csv")

# Fusionner les données de match avec les données météorologiques en fonction de la date
merged_data = pd.merge(match_data, weather_data, left_on="date", right_on="date", how="inner")

# Initialiser l'application Flask
app = Flask(__name__)

# Encodage des équipes en valeurs numériques
label_encoder = LabelEncoder()
merged_data["home_team_encoded"] = label_encoder.fit_transform(merged_data["home_team"])
merged_data["away_team_encoded"] = label_encoder.transform(merged_data["away_team"])

# Créer un modèle de régression linéaire pour prédire les scores des équipes
home_score_model = LinearRegression()
away_score_model = LinearRegression()

# Caractéristiques pour la régression
features = ["home_team_encoded", "away_team_encoded", "avg_temp_c", "precipitation_mm"]

# Entraîner les modèles sur toutes les données disponibles
home_score_model.fit(merged_data[features], merged_data["home_score"])
away_score_model.fit(merged_data[features], merged_data["away_score"])

@app.route("/", methods=["GET", "POST"])
def home():
    # Obtenir la liste des équipes disponibles à partir de la base de données
    teams = merged_data["home_team"].unique()

    winner = None
    win_percentage = None

    if request.method == "POST":
        # Récupérer les données du formulaire
        home_team = request.form["home_team"]
        away_team = request.form["away_team"]
        avg_temp_c = float(request.form["avg_temp_c"])
        precipitation_mm = float(request.form["precipitation_mm"])

        # Encodage des équipes sélectionnées en valeurs numériques
        home_team_encoded = label_encoder.transform([home_team])
        away_team_encoded = label_encoder.transform([away_team])

        # Créer un DataFrame pour le nouveau match avec les caractéristiques météorologiques
        new_match = pd.DataFrame({
            "home_team_encoded": home_team_encoded,
            "away_team_encoded": away_team_encoded,
            "avg_temp_c": avg_temp_c,
            "precipitation_mm": precipitation_mm
        })

        # Prédire les scores du match pour les deux équipes
        home_score_predicted = home_score_model.predict(new_match[features].values.reshape(1, -1))
        away_score_predicted = away_score_model.predict(new_match[features].values.reshape(1, -1))

        # Déterminer l'équipe gagnante prédite en fonction des scores prédits
        winner = home_team if home_score_predicted > away_score_predicted else away_team if away_score_predicted > home_score_predicted else "Match nul"

        # Calculer la cote (pourcentage de chance) du vainqueur
        if winner == home_team:
            win_percentage = (home_score_predicted[0] / (home_score_predicted[0] + away_score_predicted[0])) * 100
        elif winner == away_team:
            win_percentage = (away_score_predicted[0] / (home_score_predicted[0] + away_score_predicted[0])) * 100
        else:
            win_percentage = 0

    return render_template("index.html", teams=teams, winner=winner, win_percentage=win_percentage)

if __name__ == "__main__":
    app.run(debug=True)

