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




















# # code pour la prédiction un peu plus jolie quand meme 

# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import LabelEncoder
# from flask import Flask, render_template, request
# from flask_bootstrap import Bootstrap

# # Charger les données des matchs à partir de votre base de données
# match_data = pd.read_csv("rugby dataset.csv")

# # Charger les données météorologiques à partir de l'autre fichier CSV
# weather_data = pd.read_csv("donneenettoyer.csv")

# # Fusionner les données de match avec les données météorologiques en fonction de la date
# merged_data = pd.merge(match_data, weather_data, left_on="date", right_on="date", how="inner")

# # Initialiser l'application Flask
# app = Flask(__name__)
# Bootstrap(app)

# # Encodage des équipes en valeurs numériques
# label_encoder = LabelEncoder()
# merged_data["home_team_encoded"] = label_encoder.fit_transform(merged_data["home_team"])
# merged_data["away_team_encoded"] = label_encoder.transform(merged_data["away_team"])

# # Créer un modèle de régression linéaire pour prédire les scores des équipes
# home_score_model = LinearRegression()
# away_score_model = LinearRegression()

# # Caractéristiques pour la régression
# features = ["home_team_encoded", "away_team_encoded", "avg_temp_c", "precipitation_mm"]

# # Entraîner les modèles sur toutes les données disponibles
# home_score_model.fit(merged_data[features], merged_data["home_score"])
# away_score_model.fit(merged_data[features], merged_data["away_score"])

# @app.route("/", methods=["GET", "POST"])
# def home():
#     # Obtenir la liste des équipes disponibles à partir de la base de données
#     teams = merged_data["home_team"].unique()

#     winner = None
#     win_percentage = None

#     if request.method == "POST":
#         # Récupérer les données du formulaire
#         home_team = request.form["home_team"]
#         away_team = request.form["away_team"]
#         avg_temp_c = float(request.form["avg_temp_c"])
#         precipitation_mm = float(request.form["precipitation_mm"])

#         # Encodage des équipes sélectionnées en valeurs numériques
#         home_team_encoded = label_encoder.transform([home_team])
#         away_team_encoded = label_encoder.transform([away_team])

#         # Créer un DataFrame pour le nouveau match avec les caractéristiques météorologiques
#         new_match = pd.DataFrame({
#             "home_team_encoded": home_team_encoded,
#             "away_team_encoded": away_team_encoded,
#             "avg_temp_c": avg_temp_c,
#             "precipitation_mm": precipitation_mm
#         })

#         # Prédire les scores du match pour les deux équipes
#         home_score_predicted = home_score_model.predict(new_match[features].values.reshape(1, -1))
#         away_score_predicted = away_score_model.predict(new_match[features].values.reshape(1, -1))

#         # Déterminer l'équipe gagnante prédite en fonction des scores prédits
#         winner = home_team if home_score_predicted > away_score_predicted else away_team if away_score_predicted > home_score_predicted else "Match nul"

#         # Calculer la cote (pourcentage de chance) du vainqueur
#         if winner == home_team:
#             win_percentage = (home_score_predicted[0] / (home_score_predicted[0] + away_score_predicted[0])) * 100
#         elif winner == away_team:
#             win_percentage = (away_score_predicted[0] / (home_score_predicted[0] + away_score_predicted[0])) * 100
#         else:
#             win_percentage = 0

#     return render_template("index.html", teams=teams, winner=winner, win_percentage=win_percentage)

# if __name__ == "__main__":
#     app.run(debug=True)

























# code avec machin learning plus poussé 

# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
# from sklearn.pipeline import make_pipeline
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# import joblib  # Pour sauvegarder le modèle

# # Charger les données à partir du CSV
# data = pd.read_csv("donnees_fusionnees.csv")

# # Créer une instance de LabelEncoder
# label_encoder = LabelEncoder()

# # Appliquer l'encodage aux colonnes "home_team" et "away_team"
# data["home_team_encoded"] = label_encoder.fit_transform(data["home_team"])
# data["away_team_encoded"] = label_encoder.transform(data["away_team"])

# # Sélectionner les caractéristiques (features) et la cible (target)
# features = ["home_team_encoded", "away_team_encoded", "avg_temp_c", "precipitation_mm"]
# target = "home_score"

# X = data[features]
# y = data[target]

# # Diviser les données en ensembles d'entraînement et de test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Créer un modèle de régression polynomiale
# model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())

# # Entraîner le modèle sur les données d'entraînement
# model.fit(X_train, y_train)

# # Prédire les scores sur les données de test
# y_pred = model.predict(X_test)

# # Évaluer les performances du modèle
# mse = mean_squared_error(y_test, y_pred)
# print(f"Mean Squared Error: {mse}")

# # Sauvegarder le modèle pour une utilisation ultérieure
# joblib.dump(model, "rugby_model.pkl")









# import pandas as pd
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
# from flask import Flask, render_template, request
# from sklearn.pipeline import make_pipeline
# import joblib

# # Charger les données des matchs à partir de votre base de données
# match_data = pd.read_csv("donnees_fusionnees.csv")

# # Charger le modèle de régression polynomial sauvegardé
# model = joblib.load("rugby_model.pkl")

# # Initialiser l'application Flask
# app = Flask(__name__)

# @app.route("/", methods=["GET", "POST"])
# def home():
#     # Obtenir la liste des équipes uniques à partir de la base de données
#     home_teams = match_data["home_team"].unique()
#     away_teams = match_data["away_team"].unique()

#     # Combiner les équipes uniques de l'équipe à domicile et de l'équipe à l'extérieur
#     teams = list(set(home_teams) | set(away_teams))

#     winner = None
#     win_percentage = None
#     # home_team = None
#     # away_team = None
#     home_score_predicted = 0  # Initialisez ces variables à 0
#     away_score_predicted = 0

#     if request.method == "POST":
#         # Récupérer les données du formulaire
#         home_team = request.form["home_team"]
#         away_team = request.form["away_team"]
#         avg_temp_c = float(request.form["avg_temp_c"])
#         precipitation_mm = float(request.form["precipitation_mm"])

#         # Rechercher les scores des matchs précédents impliquant les mêmes équipes
#         previous_matches_home = match_data[(match_data["home_team"] == home_team) & (match_data["away_team"] == away_team)]
#         previous_matches_away = match_data[(match_data["home_team"] == away_team) & (match_data["away_team"] == home_team)]

#         # Calculer la moyenne des scores précédents
#         avg_home_score = previous_matches_home["home_score"].mean()
#         avg_away_score = previous_matches_away["away_score"].mean()

#         try:
#             # Préparer les caractéristiques pour la prédiction
#             home_team_encoded = teams.index(home_team)
#             away_team_encoded = teams.index(away_team)
#             features = [home_team_encoded, away_team_encoded, avg_temp_c, precipitation_mm]

#             # Prédire le score du match avec le modèle de régression polynomial
#             score_predicted = model.predict([features])

#             # Déterminer l'équipe gagnante prédite
#             winner = home_team if score_predicted > 0 else away_team if score_predicted < 0 else "Match nul"

#             # Calculer la cote (pourcentage de chance) du vainqueur
#             win_percentage = min(abs(score_predicted[0]) * 100, 100)

#             # Ajoutez ces lignes pour obtenir les scores de chaque équipe
#             home_score_predicted = avg_home_score + (score_predicted[0] / 2)
#             away_score_predicted = avg_away_score - (score_predicted[0] / 2)
#         except Exception as e:
#             # Gérer les erreurs ici (par exemple, imprimer l'erreur pour le débogage)
#             print(f"Erreur lors de la prédiction : {e}")
#             # Vous pouvez également attribuer des valeurs par défaut ou renvoyer un message d'erreur à l'utilisateur
#             winner = "Erreur de prédiction"
#             win_percentage = None

#     return render_template("index.html", teams=teams, winner=winner, win_percentage=win_percentage, home_score=home_score_predicted, away_score=away_score_predicted)


# if __name__ == "__main__":
#     app.run(debug=True)













































# import pandas as pd
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
# from flask import Flask, render_template, request
# import joblib

# # Charger les données des matchs à partir du fichier CSV (rugby dataset.csv)
# match_data = pd.read_csv("rugby dataset.csv")

# # Charger les données météo à partir du fichier CSV (meteo_data.csv)
# weather_data = pd.read_csv("donneenettoyer.csv")

# # Charger le modèle de régression polynomial sauvegardé
# model = joblib.load("rugby_model.pkl")

# # Initialiser l'application Flask
# app = Flask(__name__)

# @app.route("/", methods=["GET", "POST"])
# def home():
#     # Obtenir la liste des équipes uniques à partir de la base de données des matchs
#     home_teams = match_data["home_team"].unique()
#     away_teams = match_data["away_team"].unique()

#     winner = None
#     win_percentage = None
#     home_score_predicted = 0
#     away_score_predicted = 0

#     if request.method == "POST":
#         home_team = request.form["home_team"]
#         away_team = request.form["away_team"]

#         # Rechercher les scores des matchs précédents impliquant les mêmes équipes
#         previous_matches_home = match_data[(match_data["home_team"] == home_team) & (match_data["away_team"] == away_team)]
#         previous_matches_away = match_data[(match_data["home_team"] == away_team) & (match_data["away_team"] == home_team)]

#         avg_home_score = previous_matches_home["home_score"].mean()
#         avg_away_score = previous_matches_away["away_score"].mean()

#         # Utiliser les données météo historiques pour estimer l'impact de la météo sur les matchs précédents
#         # Vous devrez ajuster ce code en fonction de vos données météo spécifiques
#         avg_temp_c = weather_data["avg_temp_c"].mean()
#         precipitation_mm = weather_data["precipitation_mm"].mean()

#         try:
#             features = [avg_temp_c, precipitation_mm, avg_home_score, avg_away_score]

#             # Prédire le score du match avec le modèle de régression polynomial
#             score_predicted = model.predict([features])

#             total_score = avg_home_score - avg_away_score
#             percent_team1 = 50 + (score_predicted[0] / total_score) * 50
#             percent_team2 = 100 - percent_team1

#             if percent_team1 > percent_team2:
#                 winner = home_team
#             elif percent_team2 > percent_team1:
#                 winner = away_team
#             else:
#                 winner = "Match nul"

#             # Calculer les pourcentages d'équipe
#             if percent_team1 is not None:
#                 percent_team1 = round(percent_team1, 1)
#                 percent_team2 = 100 - percent_team1

#                 # Normaliser les pourcentages pour s'assurer qu'ils s'additionnent à 100%
#                 total_percentage = percent_team1 + percent_team2
#                 if total_percentage != 100:
#                     # Si la somme n'est pas de 100 %, normalisons
#                     percent_team1 = (percent_team1 / total_percentage) * 100
#                     percent_team2 = (percent_team2 / total_percentage) * 100

#                 # Mettre à jour les scores prédits
#                 home_score_predicted = avg_home_score + (total_score * percent_team1 / 100)
#                 away_score_predicted = avg_away_score - (total_score * percent_team2 / 100)
#         except Exception as e:
#             print(f"Erreur lors de la prédiction : {e}")
#             winner = "Erreur de prédiction"

#     return render_template("index.html", home_teams=home_teams, away_teams=away_teams, winner=winner, win_percentage=win_percentage, home_score=home_score_predicted, away_score=away_score_predicted)

# if __name__ == "__main__":
#     app.run(debug=True)


























import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from flask import Flask, render_template, request
import joblib

# Charger les données des matchs à partir du fichier CSV (rugby dataset.csv)
match_data = pd.read_csv("rugby dataset.csv")

# Charger les données météo à partir du fichier CSV (meteo_data.csv)
weather_data = pd.read_csv("donneenettoyer.csv")

# Charger le modèle de régression polynomial sauvegardé
model = joblib.load("rugby_model.pkl")

# Initialiser l'application Flask
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    # Obtenir la liste des équipes uniques à partir de la base de données des matchs
    home_teams = match_data["home_team"].unique()
    away_teams = match_data["away_team"].unique()

    winner = None
    win_percentage = None
    home_score_predicted = 0
    away_score_predicted = 0

    if request.method == "POST":
        home_team = request.form["home_team"]
        away_team = request.form["away_team"]

        # Rechercher les scores des matchs précédents impliquant les mêmes équipes
        previous_matches_home = match_data[(match_data["home_team"] == home_team) & (match_data["away_team"] == away_team)]
        previous_matches_away = match_data[(match_data["home_team"] == away_team) & (match_data["away_team"] == home_team)]

        avg_home_score = previous_matches_home["home_score"].mean()
        avg_away_score = previous_matches_away["away_score"].mean()

        # Utiliser les données météo historiques pour estimer l'impact de la météo sur les matchs précédents
        # Vous devrez ajuster ce code en fonction de vos données météo spécifiques
        avg_temp_c = float(request.form["avg_temp_c"])
        precipitation_mm = float(request.form["precipitation_mm"])

        try:
            features = [avg_temp_c, precipitation_mm, avg_home_score, avg_away_score]

            # Prédire le score du match avec le modèle de régression polynomial
            score_predicted = model.predict([features])

            total_score = avg_home_score + avg_away_score
            percent_team1 = 50 + (score_predicted[0] / total_score) * 50
            percent_team2 = 100 - percent_team1

            if percent_team1 > 100:
                percent_team1 = 100
                percent_team2 = 0
            elif percent_team2 > 100:
                percent_team2 = 100
                percent_team1 = 0

            if percent_team1 > percent_team2:
                winner = home_team
            elif percent_team2 > percent_team1:
                winner = away_team
            else:
                winner = "Match nul"

            if percent_team1 is not None:
                win_percentage = round(max(percent_team1, percent_team2), 1)

            home_score_predicted = avg_home_score + (total_score * percent_team1 / 100)
            away_score_predicted = avg_away_score + (total_score * percent_team2 / 100)
        except Exception as e:
            print(f"Erreur lors de la prédiction : {e}")
            winner = "Erreur de prédiction"

    return render_template("index.html", home_teams=home_teams, away_teams=away_teams, winner=winner, win_percentage=win_percentage, home_score=home_score_predicted, away_score=away_score_predicted)

if __name__ == "__main__":
    app.run(debug=True)
