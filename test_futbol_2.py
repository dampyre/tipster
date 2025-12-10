import sys
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

def main():
    if len(sys.argv) < 2:
        print("Uso: python test_all_leagues.py TU_API_KEY")
        sys.exit(1)

    api_key = sys.argv[1]
    tz = ZoneInfo("America/Santiago")
    manana = (datetime.now(tz).date() + timedelta(days=1)).strftime("%Y-%m-%d")

    headers = {"x-apisports-key": api_key}

    # Paso 1: obtener todas las ligas disponibles
    url_leagues = "https://v3.football.api-sports.io/leagues"
    try:
        r = requests.get(url_leagues, headers=headers, timeout=15)
        r.raise_for_status()
        leagues_data = r.json().get("response", [])
    except Exception as e:
        print("Error al consultar ligas:", e)
        sys.exit(1)

    print(f"Consultando partidos de fútbol para mañana ({manana}) en todas las ligas...\n")

    # Paso 2: recorrer todas las ligas y consultar fixtures
    url_fixtures = "https://v3.football.api-sports.io/fixtures"
    for league in leagues_data:
        lid = league.get("league", {}).get("id")
        name = league.get("league", {}).get("name")
        season = league.get("seasons", [{}])[-1].get("year")  # toma la última season disponible

        if not lid or not season:
            continue

        params = {
            "date": manana,
            "league": lid,
            "season": season,
            "timezone": "America/Santiago"
        }
        try:
            r = requests.get(url_fixtures, headers=headers, params=params, timeout=15)
            r.raise_for_status()
            data = r.json().get("response", [])
            if data:
                print(f"{name} (Liga ID {lid}, Season {season}): {len(data)} partidos")
                for f in data:
                    home = f.get("teams", {}).get("home", {}).get("name")
                    away = f.get("teams", {}).get("away", {}).get("name")
                    hora = f.get("fixture", {}).get("date","")[11:16]
                    print(f"  - {hora}: {home} vs {away}")
                print()
        except Exception as e:
            print(f"Error consultando {name}: {e}")

if __name__ == "__main__":
    main()
