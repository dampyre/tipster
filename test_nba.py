import sys
import requests
from datetime import datetime
from zoneinfo import ZoneInfo

def main():
    if len(sys.argv) < 2:
        print("Uso: python test_nba.py TU_API_KEY")
        sys.exit(1)

    api_key = sys.argv[1]
    tz = ZoneInfo("America/Santiago")
    hoy = datetime.now(tz).date().strftime("%Y-%m-%d")
    print(hoy)

    url = "https://v1.basketball.api-sports.io/games"
    headers = {"x-apisports-key": api_key}
    params = {
        "date": hoy,
        "league": 12,          # NBA
        "season": "2025-2026"  # temporada activa
    }

    print(f"Consultando NBA para {hoy}...")
    r = requests.get(url, headers=headers, params=params, timeout=15)
    print("HTTP status:", r.status_code)

    data = r.json()
    if "response" in data:
        games = data["response"]
        if not games:
            print("⚠️ No hay partidos devueltos para esa fecha/season.")
        else:
            for g in games:
                home = g.get("teams", {}).get("home", {}).get("name")
                away = g.get("teams", {}).get("away", {}).get("name")
                hora = g.get("time") or g.get("date","")[11:16]
                print(f"- {hora}: {home} vs {away}")
    else:
        print("Respuesta inesperada:", data)

if __name__ == "__main__":
    main()
