


import requests

# Usa el sport_id real en la URL
sport_id = 11  # EPL

url = f"https://therundown-therundown-v1.p.rapidapi.com/sports/{sport_id}/dates"

querystring = {"format": "date", "offset": "300"}

headers = {
    "X-RapidAPI-Key": "msh51ba1b0985d7624p185617jsn181dbf175cc0",  # reemplaza con tu clave real
    "X-RapidAPI-Host": "therundown-therundown-v1.p.rapidapi.com"
}

response = requests.get(url, headers=headers, params=querystring)

try:
    data = response.json()
    print("✅ JSON recibido")
    print(data)
except Exception as e:
    print("❌ Error al parsear JSON:", e)
    print("Cuerpo devuelto:", response.text[:500])
