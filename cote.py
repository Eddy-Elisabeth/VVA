import requests
from bs4 import BeautifulSoup

def get_betting_odds(home_team, away_team):
    url = "https://www.betclic.fr/rugby-a-xv-s5/coupe-du-monde-2023-c34"
    response = requests.get(url)
    html = response.text
    soup = BeautifulSoup(html, "html.parser")

    # Trouver toutes les balises span avec la classe "oddValue"
    odds_elements = soup.find_all("span", class_="oddValue")

    # Extraire les cotes pour chaque équipe
    home_odds = float(odds_elements[0].text.replace(",", ".")) if odds_elements and len(odds_elements) > 0 else None
    away_odds = float(odds_elements[2].text.replace(",", ".")) if odds_elements and len(odds_elements) > 2 else None

    return home_odds, away_odds
