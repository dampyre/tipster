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

def obtener_season_activa(api_key, lid):
    """Consulta la API de ligas y devuelve la última season activa para esa liga."""
    headers = {"x-apisports-key": api_key}
    url = "https://v3.football.api-sports.io/leagues"
    params = {"id": lid}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=15)
        r.raise_for_status()
        data = r.json().get("response", [])
        if not data:
            return None
        seasons = data[0].get("seasons", [])
        # Busca la última season marcada como actual
        for s in reversed(seasons):
            if s.get("current"):
                return s.get("year")
        return seasons[-1].get("year") if seasons else None
    except Exception:
        return None

def consultar_fixtures(api_key, fecha_iso, etiqueta):
    headers = {"x-apisports-key": api_key}
    url = "https://v3.football.api-sports.io/fixtures"

    print(f"\n📅 {etiqueta} ({fecha_iso})\n")
    for lid, name in LEAGUES.items():
        season = obtener_season_activa(api_key, lid)
        if not season:
            print(f"{name} (Liga ID {lid}): no se pudo determinar la season activa")
            continue
        params = {
            "date": fecha_iso,
            "league": lid,
            "season": season,
            "timezone": "America/Santiago"
        }
        try:
            r = requests.get(url, headers=headers, params=params, timeout=15)
            r.raise_for_status()
            data = r.json().get("response", [])
            print(f"{name} (Liga ID {lid}, Season {season}): {len(data)} partidos")
            for f in data:
                home = f.get("teams", {}).get("home", {}).get("name")
                away = f.get("teams", {}).get("away", {}).get("name")
                hora = f.get("fixture", {}).get("date","")[11:16]
                print(f"  - {hora}: {home} vs {away}")
        except Exception as e:
            print(f"Error consultando {name}: {e}")

def main():
    if len(sys.argv) < 2:
        print("Uso: python test_futbol_auto_season.py TU_API_KEY")
        sys.exit(1)

    api_key = sys.argv[1]
    tz = ZoneInfo("America/Santiago")
    hoy = datetime.now(tz).date().strftime("%Y-%m-%d")
    manana = (datetime.now(tz).date() + timedelta(days=1)).strftime("%Y-%m-%d")

    consultar_fixtures(api_key, hoy, "HOY")
    consultar_fixtures(api_key, manana, "MAÑANA")

if __name__ == "__main__":
    main()
