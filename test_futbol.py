import sys
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# Diccionario de ligas comunes en API-Sports
LEAGUES = {
    39: "Premier League",
    140: "La Liga",
    135: "Serie A",
    78: "Bundesliga",
    2: "UEFA Champions League"
}

def main():
    if len(sys.argv) < 2:
        print("Uso: python test_futbol.py TU_API_KEY")
        sys.exit(1)

    api_key = sys.argv[1]
    tz = ZoneInfo("America/Santiago")
    hoy = (datetime.now(tz).date() + timedelta(days=1)).strftime("%Y-%m-%d") #datetime.now(tz).date().strftime("%Y-%m-%d")

    headers = {"x-apisports-key": api_key}
    url = "https://v3.football.api-sports.io/fixtures"

    print(f"Consultando fútbol para {hoy}...\n")

    for lid, name in LEAGUES.items():
        params = {
            "date": hoy,
            #"league": lid,
            "season": "2025-2026",
            "timezone": "America/Santiago"
        }
        r = requests.get(url, headers=headers, params=params, timeout=15)
       
        data = r.json().get("response", [])
        print(f"{name} (Liga ID {lid}): {len(data)} partidos")
        for f in data:
            home = f.get("teams", {}).get("home", {}).get("name")
            away = f.get("teams", {}).get("away", {}).get("name")
            hora = f.get("fixture", {}).get("date","")[11:16]
            print(f"  - {hora}: {home} vs {away}")
        print()

if __name__ == "__main__":
    main()
