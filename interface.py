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
        location = request.form["location"]  # Récupérer le lieu du match depuis le formulaire

        # Rechercher les scores des matchs précédents impliquant les mêmes équipes
        previous_matches_home = match_data[(match_data["home_team"] == home_team) & (match_data["away_team"] == away_team)]
        previous_matches_away = match_data[(match_data["home_team"] == away_team) & (match_data["away_team"] == home_team)]

        avg_home_score = previous_matches_home["home_score"].mean()
        avg_away_score = previous_matches_away["away_score"].mean()

        # Utiliser les données météo historiques pour estimer l'impact de la météo sur les matchs précédents
        # Vous devrez ajuster ce code en fonction de vos données météo spécifiques
        avg_temp_c = float(request.form["avg_temp_c"])
        precipitation_mm = float(request.form["precipitation_mm"])


        # Ajouter une pondération en fonction du lieu du match
        if location == "home":
            home_weight = 1.1
            away_weight = 0.9
        elif location == "away":
            home_weight = 0.9
            away_weight = 1.1
        else:
            home_weight = 1.0
            away_weight = 1.0


        try:
            features = [avg_temp_c, precipitation_mm, avg_home_score, avg_away_score]

            # Prédire le score du match avec le modèle de régression polynomial
            score_predicted = model.predict([features])

            total_score = avg_home_score + avg_away_score
            percent_team1 = 50 + (score_predicted[0] / total_score) * 50
            percent_team2 = 100 - percent_team1

            # Appliquer la pondération en fonction du lieu du match
            percent_team1 *= home_weight
            percent_team2 *= away_weight

            # Assurez-vous que les pourcentages restent dans la plage de 0 à 100
            percent_team1 = max(0, min(100, percent_team1))
            percent_team2 = max(0, min(100, percent_team2))

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





















































# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
# from bs4 import BeautifulSoup
# import requests
# from flask import Flask, render_template, request

# # Charger les données des matchs à partir du fichier CSV (rugby dataset.csv)
# match_data = pd.read_csv("rugby dataset.csv")

# # Charger les données météo à partir du fichier CSV (meteo_dataset.csv)
# meteo_data = pd.read_csv("donneenettoyer.csv")

# # Initialiser l'application Flask
# app = Flask(__name__)

# # Fonction pour récupérer les cotes de paris en ligne
# def get_betting_odds(home_team, away_team):
#     url = "https://www.betclic.fr/rugby-a-xv-s5/coupe-du-monde-2023-c34"
#     response = requests.get(url)
#     html = response.text
#     soup = BeautifulSoup(html, "html.parser")

#     # Trouver toutes les balises span avec la classe "oddValue"
#     odds_elements = soup.find_all("span", class_="oddValue")

#     # Extraire les cotes pour l'équipe à domicile (premier élément) et l'équipe à l'extérieur (troisième élément)
#     home_odds = float(odds_elements[0].text.replace(",", ".")) if odds_elements and len(odds_elements) > 0 else None
#     away_odds = float(odds_elements[2].text.replace(",", ".")) if odds_elements and len(odds_elements) > 2 else None

#     return home_odds, away_odds

# @app.route("/", methods=["GET", "POST"])
# def home():
#     home_teams = match_data["home_team"].unique()
#     away_teams = match_data["away_team"].unique()

#     winner = None
#     win_percentage = None
#     home_score = 0
#     away_score = 0  # Ajout de cette ligne

#     if request.method == "POST":
#         home_team = request.form["home_team"]
#         away_team = request.form["away_team"]

#         # Obtenir les cotes de paris en ligne
#         home_odds, away_odds = get_betting_odds(home_team, away_team)

#         # Extraire les caractéristiques du match
#         avg_temp_c = float(request.form["avg_temp_c"])
#         precipitation_mm = float(request.form["precipitation_mm"])
#         avg_home_score = match_data[(match_data["home_team"] == home_team) & (match_data["away_team"] == away_team)]["home_score"].mean()
#         avg_away_score = match_data[(match_data["home_team"] == away_team) & (match_data["away_team"] == home_team)]["away_score"].mean()

#         # Entraîner un modèle de régression logistique sur les données historiques
#         features = match_data[["home_score", "away_score"]]
#         # target = match_data["result"]  # 1 si l'équipe à domicile gagne, 0 sinon
#         target = (match_data["home_score"] > match_data["away_score"]).astype(int)

#         # Normaliser les données
#         scaler = StandardScaler()
#         features = scaler.fit_transform(features)

#         model = LogisticRegression()
#         model.fit(features, target)

#         input_features = [avg_home_score, avg_away_score]
#         input_features = scaler.transform([input_features])
#         prediction = model.predict_proba(input_features)

#         home_probability = prediction[0][1]
#         away_probability = prediction[0][0]

#         # Appliquer la pondération en fonction des cotes de paris
#         if home_odds and away_odds:
#             home_weight = 1 / home_odds
#             away_weight = 1 / away_odds
#             home_probability *= home_weight
#             away_probability *= away_weight

#         home_probability = max(0, min(1, home_probability))
#         away_probability = max(0, min(1, away_probability))

#         if home_probability > away_probability:
#             winner = home_team
#             win_percentage = home_probability
#             home_score = int((home_probability + 0.5) * 30)
#         else:
#             winner = away_team
#             win_percentage = away_probability
#             away_score = int((away_probability + 0.5) * 30)

#     return render_template("index.html", home_teams=home_teams, away_teams=away_teams,
#                            winner=winner, win_percentage=win_percentage, home_score=home_score, away_score=away_score)

# if __name__ == "__main__":
#     app.run(debug=True)














































# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
# from bs4 import BeautifulSoup
# import requests
# from flask import Flask, render_template, request

# # Charger les données des matchs à partir du fichier CSV (rugby dataset.csv)
# match_data = pd.read_csv("rugby dataset.csv")

# # Charger les données météo à partir du fichier CSV (donneenettoyer.csv)
# meteo_data = pd.read_csv("donneenettoyer.csv")

# match_data['result'] = 0  # Initialisez la colonne "result" avec des zéros (0 pour une victoire à l'extérieur).

# # Mettez à jour la colonne "result" en fonction des scores.
# match_data.loc[match_data['home_score'] > match_data['away_score'], 'result'] = 1  # Victoire à domicile (1).

# # Initialiser l'application Flask
# app = Flask(__name__)

# # Fonction pour récupérer les cotes de paris en ligne depuis un site web
# def get_betting_odds(home_team, away_team):
#     url = "https://www.betclic.fr/coupe-du-monde-2023-s5/coupe-du-monde-2023-c34"  # Remplacez par l'URL de votre site Web réel
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, "html.parser")

#     # Trouver tous les éléments avec la classe "oddValue" qui contiennent les cotes
#     odds_elements = soup.find_all("span", class_="oddValue")

#     # Extraire les cotes
#     home_odds = float(odds_elements[0].text.replace(",", "."))
#     away_odds = float(odds_elements[2].text.replace(",", "."))

#     return home_odds, away_odds

# @app.route("/", methods=["GET", "POST"])
# def home():
#     home_teams = match_data["home_team"].unique()
#     away_teams = match_data["away_team"].unique()

#     winner = None
#     win_percentage = None
#     home_score = 0
#     away_score = 0

#     if request.method == "POST":
#         home_team = request.form["home_team"]
#         away_team = request.form["away_team"]

#         # Obtenir les cotes de paris en ligne
#         home_odds, away_odds = get_betting_odds(home_team, away_team)

#         # Extraire les caractéristiques du match à partir des données météo
#         match_meteo = meteo_data[(meteo_data["city_name"] == request.form["location"])]

#         avg_temp_c = match_meteo["avg_temp_c"].mean()
#         precipitation_mm = match_meteo["precipitation_mm"].mean()

#         # Entraîner un modèle de régression logistique sur les données historiques
#         features = meteo_data[["avg_temp_c", "precipitation_mm"]]
#         features = match_data[["home_score", "away_score"]]
#         target = match_data["result"]  # 1 si l'équipe à domicile gagne, 0 sinon

#         # Normaliser les données
#         scaler = StandardScaler()
#         features = scaler.fit_transform(features)

#         model = LogisticRegression()
#         model.fit(features, target)

#         input_features = [avg_temp_c, precipitation_mm, home_score, away_score]
#         input_features = scaler.transform([input_features])
#         prediction = model.predict_proba(input_features)

#         home_probability = prediction[0][1]
#         away_probability = prediction[0][0]

#         # Appliquer la pondération en fonction des cotes de paris
#         if home_odds and away_odds:
#             home_weight = 1 / home_odds
#             away_weight = 1 / away_odds
#             home_probability *= home_weight
#             away_probability *= away_weight

#         home_probability = max(0, min(1, home_probability))
#         away_probability = max(0, min(1, away_probability))

#         if home_probability > away_probability:
#             winner = home_team
#             win_percentage = home_probability * 100
#             home_score = int((home_probability + 0.5) * 30)
#         else:
#             winner = away_team
#             win_percentage = away_probability * 100
#             away_score = int((away_probability + 0.5) * 30)

#     return render_template("index.html", home_teams=home_teams, away_teams=away_teams,
#                            winner=winner, win_percentage=win_percentage, home_score=home_score, away_score=away_score)

# if __name__ == "__main__":
#     app.run(debug=True)




































# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
# from bs4 import BeautifulSoup
# import requests
# from flask import Flask, render_template, request

# # Charger les données des matchs à partir du fichier CSV (rugby dataset.csv)
# match_data = pd.read_csv("rugby dataset.csv")

# # Charger les données météo à partir du fichier CSV (donneenettoyer.csv)
# meteo_data = pd.read_csv("donneenettoyer.csv")

# # Initialiser l'application Flask
# app = Flask(__name__)

# # Fonction pour récupérer les cotes de paris en ligne depuis un site web
# def get_betting_odds(home_team, away_team):
#     url = "https://www.betclic.fr/rugby-a-xv-s5/coupe-du-monde-2023-c34"
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, "html.parser")

#     # Trouver les éléments qui contiennent les cotes (assurez-vous que les sélecteurs CSS sont corrects)
#     home_odds_element = soup.find("span", class_="oddValue", text=True)
#     away_odds_element = home_odds_element.find_next("span", class_="oddValue")

#     # Extraire les cotes
#     home_odds = float(home_odds_element.get_text().replace(",", "."))
#     away_odds = float(away_odds_element.get_text().replace(",", "."))

#     return home_odds, away_odds

# @app.route("/", methods=["GET", "POST"])
# def home():
#     home_teams = match_data["home_team"].unique()
#     away_teams = match_data["away_team"].unique()

#     winner = None
#     win_percentage = None
#     home_score = 0
#     away_score = 0

#     if request.method == "POST":
#         home_team = request.form["home_team"]
#         away_team = request.form["away_team"]

#         # Obtenir les cotes de paris en ligne
#         home_odds, away_odds = get_betting_odds(home_team, away_team)

#         # Extraire les caractéristiques du match à partir des données météo
#         match_meteo = meteo_data[meteo_data["city_name"] == request.form["location"]]
#         avg_temp_c = match_meteo["avg_temp_c"].mean()
#         precipitation_mm = match_meteo["precipitation_mm"].mean()

#         # Entraîner un modèle de régression logistique sur les scores des équipes
#         features = match_data[["home_score", "away_score"]]
#         target = (features["home_score"] > features["away_score"]).astype(int)  # 1 si l'équipe à domicile gagne, 0 sinon

#         # Normaliser les données
#         scaler = StandardScaler()
#         features = scaler.fit_transform(features)

#         model = LogisticRegression()
#         model.fit(features, target)

#         input_features = [home_score, away_score]
#         input_features = scaler.transform([input_features])
#         prediction = model.predict_proba(input_features)

#         home_probability = prediction[0][1]
#         away_probability = prediction[0][0]

#         # Appliquer la pondération en fonction des cotes de paris
#         if home_odds and away_odds:
#             home_weight = 1 / home_odds
#             away_weight = 1 / away_odds
#             home_probability *= home_weight
#             away_probability *= away_weight

#         home_probability = max(0, min(1, home_probability))
#         away_probability = max(0, min(1, away_probability))

#         if home_probability > away_probability:
#             winner = home_team
#             win_percentage = home_probability * 100
#             home_score = int((home_probability + 0.5) * 30)
#         else:
#             winner = away_team
#             win_percentage = away_probability * 100
#             away_score = int((away_probability + 0.5) * 30)

#     return render_template("index.html", home_teams=home_teams, away_teams=away_teams,
#                            winner=winner, win_percentage=win_percentage, home_score=home_score, away_score=away_score)

# if __name__ == "__main__":
#     app.run(debug=True)

