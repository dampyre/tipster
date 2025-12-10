import requests
from datetime import date, timedelta

API_KEY = "cfa59f64b11462f06bf0642a1d6df479"
headers = {"x-apisports-key": API_KEY}
url = "https://v1.basketball.api-sports.io/games"

hoy = date.today()
for i in range(5):  # prueba hoy y los próximos 4 días
    fecha = (hoy + timedelta(days=i)).strftime("%Y-%m-%d")
    params = {"date": fecha, "league": 12, "season": "2025-2026"}
    r = requests.get(url, headers=headers, params=params, timeout=15)
    data = r.json().get("response", [])
    print(f"{fecha}: {len(data)} partidos")
    for g in data:
        home = g.get("teams", {}).get("home", {}).get("name")
        away = g.get("teams", {}).get("away", {}).get("name")
        hora = g.get("time") or g.get("date","")[11:16]
        print(f"  - {hora}: {home} vs {away}")
