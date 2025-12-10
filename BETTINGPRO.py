# =============================================================================
# BETTING PRO — v17.2
# =============================================================================
# Cambios clave:
# - API‑Sports: seasons corregidas a "2025-2026" para fútbol y NBA.
# - The Odds API: se elimina el parámetro de fecha; se filtra por commence_time (ISO) para "hoy" y "mañana".
# - Edge flexible (0–20%), columna de CALIDAD basada en edge.
# - Fútbol multiliga configurable y pronósticos también para mañana (fútbol y tenis).
# - Mensajes claros cuando no hay partidos/mercados para la fecha.
# - Documentación y comentarios para mantenimiento.
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import os
import json

# -----------------------------------------------------------------------------
# Constantes
# -----------------------------------------------------------------------------
CONFIG_FILE = "betting_config.json"
HIST_FILE   = "betting_historial.csv"
APP_VERSION = "v17.1"

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
def load_config():
    """
    Carga configuración desde JSON o retorna defaults.
    Incluye:
    - Finanzas: bankroll, min_edge, max_kelly, auto_cashout.
    - Claves API: API‑Sports y The Odds API.
    - Ligas/temporadas: NBA, fútbol multiliga (IDs), tenis.
    """
    default = {
        "bankroll": 10000,
        "min_edge": 0.07,
        "max_kelly": 0.10,
        "auto_cashout": 0.30,
        "API_SPORTS_KEY": "",
        "FOOTBALL_API_KEY": "",
        "BASKETBALL_API_KEY": "",
        "TENNIS_API_KEY": "",
        "ODDS_API_KEY": "",
        # Ligas/Season (corregidas a formato "YYYY-YYYY")
        "BB_LEAGUE_ID": 12,              # NBA
        "BB_SEASON": "2025-2026",
        "FB_SEASON": "2025-2026",
        "FB_LEAGUES": [39, 140, 135, 78, 2],  # Premier, La Liga, Serie A, Bundesliga, UCL
        "TN_LEAGUE_ID": None,
        # Regiones y mercados The Odds API
        "THE_ODDS_REGIONS": "us",
        "THE_ODDS_MARKETS": "h2h,totals,btts",
        "THE_ODDS_FORMAT": "decimal",
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
            # Normaliza FB_LEAGUES a lista
            if isinstance(data.get("FB_LEAGUES"), int):
                data["FB_LEAGUES"] = [data["FB_LEAGUES"]]
            default.update(data)
        except Exception:
            pass
    return default

def save_config(cfg):
    """Guarda configuración en JSON."""
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)

# =============================================================================
# HISTORIAL
# =============================================================================
def init_historial():
    """
    Carga historial desde CSV o crea estructura vacía:
    fecha, dia, deporte, partido, hora, mercado, apuesta, cuota,
    probabilidad, edge, kelly, stake, status, roi_real, pick_id.
    """
    cols = [
        "fecha","dia","deporte","partido","hora","mercado","apuesta",
        "cuota","probabilidad","edge","kelly","stake","status","roi_real","pick_id"
    ]
    if os.path.exists(HIST_FILE):
        try:
            df = pd.read_csv(HIST_FILE)
            for c in cols:
                if c not in df.columns:
                    df[c] = np.nan
            return df[cols]
        except Exception:
            pass
    return pd.DataFrame(columns=cols)

def save_historial(df):
    """Exporta historial a CSV."""
    df.to_csv(HIST_FILE, index=False)

# =============================================================================
# UTILIDADES
# =============================================================================
def kelly_fraction(edge, cuota, max_kelly):
    """
    Kelly fracción (limitada):
    - edge = p_model - p_implied
    - p_implied = 1/cuota
    - fracción = edge/(cuota-1), recortada por max_kelly y no negativa.
    """
    if edge <= 0 or cuota <= 1.0:
        return 0.0
    f = edge / (cuota - 1)
    return max(0.0, min(f, max_kelly))

def edge_from_probs(p_model, cuota):
    """Edge = p_model - 1/cuota."""
    p_implied = 1/float(cuota)
    return p_model - p_implied

def as_percent(x):
    """0.071 -> '7.1%'."""
    return f"{x*100:.1f}%"

def edge_str_to_float(edge_str):
    """'7.1%' -> 0.071."""
    try:
        return float(str(edge_str).replace("%","").replace(",", "."))/100
    except Exception:
        return 0.0

def stake_str_to_float(stake_str):
    """'$1,250' -> 1250.0."""
    try:
        s = str(stake_str).replace("$","").replace(",","").strip()
        return float(s)
    except Exception:
        return 0.0

def fmt_money(x):
    """Formatea dinero: '$1,234'."""
    try:
        return f"${x:,.0f}"
    except Exception:
        return f"${x:.0f}"

def make_pick_id(fecha, partido, mercado, apuesta):
    """ID único para evitar duplicados."""
    return f"{fecha}|{partido}|{mercado}|{apuesta}"

def calidad_apuesta(edge):
    """
    Clasificación de calidad basada en edge:
    - edge < 0: Sin valor
    - 0 <= edge < 3%: Bajo valor
    - 3% <= edge < 7%: Valor moderado
    - edge >= 7%: Alto valor
    """
    if edge < 0:
        return "❌ Sin valor"
    elif edge < 0.03:
        return "⚠️ Bajo valor"
    elif edge < 0.07:
        return "✅ Valor moderado"
    else:
        return "💎 Alto valor"

# Fechas (Chile)
CL_TZ = ZoneInfo("America/Santiago")
def get_local_dates():
    """
    Devuelve las fechas locales (Chile) para hoy y mañana, en formato date().
    """
    hoy = datetime.now(CL_TZ).date()
    manana = hoy + timedelta(days=1)
    return hoy, manana

# =============================================================================
# APIS — API‑Sports
# =============================================================================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_basketball_games_apisports(cfg, fecha_iso: str):
    """NBA fixtures (API‑Sports) por fecha."""
    key = cfg.get("BASKETBALL_API_KEY") or cfg.get("API_SPORTS_KEY","")
    headers = {"x-apisports-key": key}
    url = "https://v1.basketball.api-sports.io/games"
    params = {"date": fecha_iso, "league": cfg["BB_LEAGUE_ID"], "season": cfg["BB_SEASON"]}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=15)
        r.raise_for_status()
        return r.json().get("response", [])
    except Exception:
        return []

@st.cache_data(ttl=300, show_spinner=False)
def fetch_basketball_odds_apisports(cfg, fecha_iso: str):
    """NBA odds (API‑Sports) por fecha."""
    key = cfg.get("BASKETBALL_API_KEY") or cfg.get("API_SPORTS_KEY","")
    headers = {"x-apisports-key": key}
    url = "https://v1.basketball.api-sports.io/odds"
    params = {"date": fecha_iso, "league": cfg["BB_LEAGUE_ID"]}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=15)
        r.raise_for_status()
        return r.json().get("response", [])
    except Exception:
        return []

@st.cache_data(ttl=300, show_spinner=False)
def fetch_football_fixtures_apisports(cfg, fecha_iso: str, league_id: int):
    """Fútbol fixtures (API‑Sports) por fecha y liga."""
    key = cfg.get("FOOTBALL_API_KEY") or cfg.get("API_SPORTS_KEY","")
    headers = {"x-apisports-key": key}
    url = "https://v3.football.api-sports.io/fixtures"
    params = {"date": fecha_iso, "league": league_id, "season": cfg["FB_SEASON"], "timezone": "America/Santiago"}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=15)
        r.raise_for_status()
        return r.json().get("response", [])
    except Exception:
        return []

@st.cache_data(ttl=300, show_spinner=False)
def fetch_football_odds_apisports(cfg, fecha_iso: str, league_id: int):
    """Fútbol odds (API‑Sports) por fecha y liga."""
    key = cfg.get("FOOTBALL_API_KEY") or cfg.get("API_SPORTS_KEY","")
    headers = {"x-apisports-key": key}
    url = "https://v3.football.api-sports.io/odds"
    params = {"date": fecha_iso, "league": league_id}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=15)
        r.raise_for_status()
        return r.json().get("response", [])
    except Exception:
        return []

@st.cache_data(ttl=300, show_spinner=False)
def fetch_tennis_odds_apisports(cfg, fecha_iso: str):
    """Tenis odds (API‑Sports) por fecha."""
    key = cfg.get("TENNIS_API_KEY") or cfg.get("API_SPORTS_KEY","")
    headers = {"x-apisports-key": key}
    url = "https://v1.tennis.api-sports.io/odds"
    params = {"date": fecha_iso}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=15)
        r.raise_for_status()
        return r.json().get("response", [])
    except Exception:
        return []

# =============================================================================
# APIS — The Odds API
# =============================================================================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_theodds_sports(cfg):
    """Lista deportes disponibles (The Odds API)."""
    key = cfg.get("ODDS_API_KEY","")
    url = "https://api.the-odds-api.com/v4/sports"
    try:
        r = requests.get(url, params={"apiKey": key}, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []

@st.cache_data(ttl=300, show_spinner=False)
def fetch_theodds_odds(cfg, sport_key: str):
    """
    Obtiene odds de The Odds API (próximos eventos).
    Nota: No acepta parámetro de fecha; se filtra manualmente por commence_time.
    """
    key = cfg.get("ODDS_API_KEY","")
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "apiKey": key,
        "regions": cfg.get("THE_ODDS_REGIONS","us"),
        "markets": cfg.get("THE_ODDS_MARKETS","h2h,totals,btts"),
        "oddsFormat": cfg.get("THE_ODDS_FORMAT","decimal"),
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []

def filter_by_date_iso(events, fecha_iso: str):
    """
    Filtra eventos de The Odds API por commence_time (YYYY-MM-DD).
    """
    out = []
    for ev in events:
        c = ev.get("commence_time")  # ISO 'YYYY-MM-DDTHH:MM:SSZ'
        if c and c.startswith(fecha_iso):
            out.append(ev)
    return out

# =============================================================================
# MODELOS (placeholders)
# =============================================================================
def model_nba_under(game):
    tl = game.get("total_line")
    if tl is None: return 0.54
    return 0.55 if tl >= 228 else 0.58 if tl >= 222 else 0.52

def model_nba_over(game):
    tl = game.get("total_line")
    if tl is None: return 0.54
    return 0.57 if tl <= 223 else 0.54

def model_nba_ml_away(game):
    away = (game.get("away_team") or "").lower()
    return 0.61 if "knicks" in away else 0.58

def model_soccer_home(match): return 0.58
def model_soccer_btts(match): return 0.56
def model_tennis_ml_p1(match):
    p1 = (match.get("player1") or "").lower()
    return 0.63 if any(x in p1 for x in ["djokovic","sinner","alcaraz"]) else 0.60

def compute_row(p_model, cuota, cfg):
    edge = edge_from_probs(p_model, cuota)
    kelly = kelly_fraction(edge, cuota, cfg["max_kelly"])
    stake = cfg["bankroll"] * max(kelly, 0)
    return edge, kelly, stake

# =============================================================================
# NORMALIZACIÓN Y COMBINACIÓN DE ODDS
# =============================================================================
def extract_basketball_entries(cfg, fecha_iso: str):
    """
    NBA: fixtures + odds (API‑Sports) -> entradas con:
    hora, home_team, away_team, total_line, odds_over/under, odds_ml_home/away.
    """
    games = fetch_basketball_games_apisports(cfg, fecha_iso)
    odds_resp = fetch_basketball_odds_apisports(cfg, fecha_iso)
    odds_map = {}
    for o in odds_resp:
        gid = o.get("game", {}).get("id")
        odds_map.setdefault(gid, []).append(o)

    out = []
    for g in games:
        gid = g.get("id")
        home = g.get("teams", {}).get("home", {}).get("name")
        away = g.get("teams", {}).get("away", {}).get("name")
        hora = g.get("time") or (g.get("date","")[11:16] if g.get("date") else "")
        entry = {"hora": hora, "home_team": home, "away_team": away}
        mkts = odds_map.get(gid, [])
        total_line = None; odds_over = None; odds_under = None; odds_ml_home=None; odds_ml_away=None
        for o in mkts:
            for bk in o.get("bookmakers", []):
                for bet in bk.get("bets", []):
                    name = (bet.get("name") or "").lower()
                    if name in ("total points","totals"):
                        for v in bet.get("values", []):
                            val = (v.get("value") or "").lower()
                            odd = v.get("odd")
                            if not odd: continue
                            if "over" in val:
                                odds_over = float(odd)
                                try: total_line = float(val.split()[-1])
                                except: pass
                            elif "under" in val:
                                odds_under = float(odd)
                                try: total_line = float(val.split()[-1])
                                except: pass
                    if name in ("moneyline","winner"):
                        for v in bet.get("values", []):
                            val = (v.get("value") or "").lower()
                            odd = v.get("odd")
                            if not odd: continue
                            if "home" in val or (home and home.lower() in val):
                                odds_ml_home = float(odd)
                            if "away" in val or (away and away.lower() in val):
                                odds_ml_away = float(odd)
        if total_line: entry["total_line"] = total_line
        if odds_over: entry["odds_over"] = odds_over
        if odds_under: entry["odds_under"] = odds_under
        if odds_ml_home: entry["odds_ml_home"] = odds_ml_home
        if odds_ml_away: entry["odds_ml_away"] = odds_ml_away
        out.append(entry)
    return out

def extract_football_entries_single_league(cfg, fecha_iso: str, league_id: int):
    """
    Fútbol (una liga): fixtures + odds (API‑Sports) -> lista de entradas con:
    hora, home_team, away_team, odds_home, odds_btts_yes.
    """
    fixtures = fetch_football_fixtures_apisports(cfg, fecha_iso, league_id)
    odds_resp = fetch_football_odds_apisports(cfg, fecha_iso, league_id)
    # map odds por fixture id
    odds_map = {}
    for o in odds_resp:
        fid = o.get("fixture", {}).get("id")
        odds_map.setdefault(fid, []).append(o)

    entries = []
    for f in fixtures:
        fid = f.get("fixture", {}).get("id")
        hora = (f.get("fixture", {}).get("date","")[11:16]) or ""
        home = f.get("teams", {}).get("home", {}).get("name")
        away = f.get("teams", {}).get("away", {}).get("name")
        entry = {"hora": hora, "home_team": home, "away_team": away}
        mkts = odds_map.get(fid, [])
        odds_home = None; odds_btts_yes = None
        for o in mkts:
            for bk in o.get("bookmakers", []):
                for bet in bk.get("bets", []):
                    name = (bet.get("name") or "").lower()
                    if name in ("match winner","winner","1x2","moneyline"):
                        for v in bet.get("values", []):
                            odd = v.get("odd")
                            val = (v.get("value") or "").lower()
                            if odd and (val in ("home","1") or (home and home.lower() in val)):
                                odds_home = float(odd)
                    if "both teams to score" in name or "btts" in name:
                        for v in bet.get("values", []):
                            odd = v.get("odd")
                            val = (v.get("value") or "").lower()
                            if odd and ("yes" in val or "sí" in val):
                                odds_btts_yes = float(odd)
        if odds_home: entry["odds_home"] = odds_home
        if odds_btts_yes: entry["odds_btts_yes"] = odds_btts_yes
        entries.append(entry)
    return entries

def extract_football_entries_multi(cfg, fecha_iso: str):
    """
    Fútbol multiliga:
    - Itera sobre cfg["FB_LEAGUES"] (lista de IDs).
    - Construye un dict: {nombre_liga: [entradas]}.
    """
    league_names = {
        39: "Premier League",
        140: "La Liga",
        135: "Serie A",
        78: "Bundesliga",
        2: "UEFA Champions League",
    }
    leagues = cfg.get("FB_LEAGUES", [39])
    out_by_league = {}
    for lid in leagues:
        entries = extract_football_entries_single_league(cfg, fecha_iso, lid)
        name = league_names.get(lid, f"Liga {lid}")
        out_by_league[name] = entries
    return out_by_league

def extract_tennis_entries(cfg, fecha_iso: str):
    """
    Tenis: entradas con hora, player1, player2, odds_p1 (API‑Sports).
    """
    odds_resp = fetch_tennis_odds_apisports(cfg, fecha_iso)
    out = []
    for o in odds_resp:
        match = o.get("match", {})
        player1 = match.get("home","")
        player2 = match.get("away","")
        hora = (o.get("time") or "")[:5]
        odds_p1 = None
        for bk in o.get("bookmakers", []):
            for bet in bk.get("bets", []):
                name = (bet.get("name") or "").lower()
                if name in ("moneyline","winner"):
                    for v in bet.get("values", []):
                        val = (v.get("value") or "").lower()
                        odd = v.get("odd")
                        if odd and (val in ("home","player1") or (player1 and player1.lower() in val)):
                            odds_p1 = float(odd)
        if player1 and player2 and odds_p1:
            out.append({
                "hora": hora or "00:00",
                "player1": player1,
                "player2": player2,
                "odds_p1": odds_p1
            })
    return out

# Suplementos con The Odds API (filtrados por fecha via commence_time)
def supplement_with_theodds_basketball(cfg, fecha_iso: str, entries):
    """
    Complementa cuotas NBA con The Odds API (h2h, totals), escogiendo el mejor precio.
    Filtra por commence_time que empieza con fecha_iso (YYYY-MM-DD).
    """
    the_sports = fetch_theodds_sports(cfg)
    nba_keys = [s.get("key") for s in the_sports if "basketball_nba" in s.get("key","")]
    sport_key = nba_keys[0] if nba_keys else "basketball_nba"
    the_odds_all = fetch_theodds_odds(cfg, sport_key)
    the_odds = filter_by_date_iso(the_odds_all, fecha_iso)

    def key_matchup(h, a): return f"{(h or '').strip()} vs {(a or '').strip()}".lower()
    odds_map = {}
    for ev in the_odds:
        home = ev.get("home_team")
        away = ev.get("away_team")
        km = key_matchup(home, away)
        odds_map.setdefault(km, []).append(ev)

    for e in entries:
        km = key_matchup(e.get("home_team"), e.get("away_team"))
        events = odds_map.get(km, [])
        best_ml_home = e.get("odds_ml_home"); best_ml_away = e.get("odds_ml_away")
        best_over = e.get("odds_over"); best_under = e.get("odds_under"); tl = e.get("total_line")
        for ev in events:
            for bk in ev.get("bookmakers", []):
                for mk in bk.get("markets", []):
                    mkt_key = mk.get("key")
                    if mkt_key == "h2h":
                        for outc in mk.get("outcomes", []):
                            name = outc.get("name","").lower()
                            price = outc.get("price")
                            if not price: continue
                            if "home" in name or (e.get("home_team","").lower() in name):
                                best_ml_home = max(best_ml_home or 0, float(price))
                            if "away" in name or (e.get("away_team","").lower() in name):
                                best_ml_away = max(best_ml_away or 0, float(price))
                    if mkt_key == "totals":
                        for outc in mk.get("outcomes", []):
                            desc = (outc.get("description") or "").lower()
                            price = outc.get("price")
                            point = outc.get("point")
                            if not price: continue
                            if tl is None and point is not None:
                                tl = point
                            if "over" in desc:
                                best_over = max(best_over or 0, float(price))
                            elif "under" in desc:
                                best_under = max(best_under or 0, float(price))
        if best_ml_home: e["odds_ml_home"] = best_ml_home
        if best_ml_away: e["odds_ml_away"] = best_ml_away
        if tl: e["total_line"] = tl
        if best_over: e["odds_over"] = best_over
        if best_under: e["odds_under"] = best_under
    return entries

def supplement_with_theodds_football_league(cfg, fecha_iso: str, league_name: str, entries):
    """
    Fútbol: complementa cuotas por liga con The Odds API (h2h, btts).
    Busca sport_key que contenga el nombre de la liga (o usa soccer_epl como fallback).
    Filtra eventos por commence_time (fecha_iso).
    """
    the_sports = fetch_theodds_sports(cfg)
    candidates = []
    for s in the_sports:
        key = s.get("key","")
        title = (s.get("title") or "").lower()
        if league_name.lower() in title:
            candidates.append(key)
        elif "premier" in league_name.lower() and "soccer_epl" in key:
            candidates.append(key)
    sport_key = candidates[0] if candidates else "soccer_epl"

    the_odds_all = fetch_theodds_odds(cfg, sport_key)
    the_odds = filter_by_date_iso(the_odds_all, fecha_iso)

    def key_matchup(h, a): return f"{(h or '').strip()} vs {(a or '').strip()}".lower()
    odds_map = {}
    for ev in the_odds:
        home = ev.get("home_team")
        away = ev.get("away_team")
        km = key_matchup(home, away)
        odds_map.setdefault(km, []).append(ev)

    for e in entries:
        km = key_matchup(e.get("home_team"), e.get("away_team"))
        events = odds_map.get(km, [])
        best_home = e.get("odds_home"); best_btts_yes = e.get("odds_btts_yes")
        for ev in events:
            for bk in ev.get("bookmakers", []):
                for mk in bk.get("markets", []):
                    key = mk.get("key")
                    if key == "h2h":
                        for outc in mk.get("outcomes", []):
                            name = outc.get("name","").lower()
                            price = outc.get("price")
                            if not price: continue
                            if "home" in name or (e.get("home_team","").lower() in name):
                                best_home = max(best_home or 0, float(price))
                    if key == "btts":
                        for outc in mk.get("outcomes", []):
                            name = outc.get("name","").lower()
                            price = outc.get("price")
                            if not price: continue
                            if "yes" in name:
                                best_btts_yes = max(best_btts_yes or 0, float(price))
        if best_home: e["odds_home"] = best_home
        if best_btts_yes: e["odds_btts_yes"] = best_btts_yes
    return entries

def find_value_bets_theodds(cfg, sport_key: str):
    """
    Value Bets: compara dos casas (The Odds API) en h2h home y calcula 'edge extra'.
    No filtra por fecha; se usa para explorar mercados.
    """
    data = fetch_theodds_odds(cfg, sport_key)
    rows = []
    for ev in data:
        match = f"{ev.get('home_team')} vs {ev.get('away_team')}"
        casas = {}
        for bk in ev.get("bookmakers", []):
            name = bk.get("title") or bk.get("key")
            for mk in bk.get("markets", []):
                if mk.get("key") == "h2h":
                    for outc in mk.get("outcomes", []):
                        if outc.get("name","").lower() in ("home", (ev.get("home_team","") or "").lower()):
                            if outc.get("price"):
                                casas[name] = float(outc["price"])
        if len(casas) >= 2:
            items = list(casas.items())
            casa_a, cuota_a = items[0]
            casa_b, cuota_b = items[1]
            mejor_cuota = max(cuota_a, cuota_b)
            promedio = (cuota_a + cuota_b)/2
            edge_extra = (mejor_cuota - promedio) / promedio
            rows.append([match, cuota_a, cuota_b, mejor_cuota, f"+{edge_extra*100:.1f}%"])
    if rows:
        return pd.DataFrame(rows, columns=["Partido","Cuota Casa A","Cuota Casa B","Mejor Cuota","Edge Extra"])
    return pd.DataFrame(columns=["Partido","Cuota Casa A","Cuota Casa B","Mejor Cuota","Edge Extra"])

# =============================================================================
# CONSTRUCCIÓN DE TABLAS PARA EL HOME
# =============================================================================
def build_home_tables(cfg):
    """
    Genera tablas HOY/MAÑANA:
    - NBA: Totales y ML visitante.
    - Fútbol multiliga: ML local y BTTS Sí.
    - Tenis: ML jugador 1.
    Incluye columna CALIDAD basada en edge.
    """
    hoy, manana = get_local_dates()
    hoy_str = hoy.strftime("%Y-%m-%d")
    manana_str = manana.strftime("%Y-%m-%d")

    # ---------- HOY ----------
    nba_games = extract_basketball_entries(cfg, hoy_str)
    nba_games = supplement_with_theodds_basketball(cfg, hoy_str, nba_games)
    filas_nba_hoy = []

    for g in nba_games:
        # Under
        if "odds_under" in g and "total_line" in g:
            p_model = model_nba_under(g)
            edge, kelly, stake = compute_row(p_model, g["odds_under"], cfg)
            filas_nba_hoy.append([
                g.get("hora",""),
                f"{g.get('home_team','')} vs {g.get('away_team','')}",
                "Total Puntos",
                f"Under {g.get('total_line')}",
                round(g.get("odds_under"), 2),
                as_percent(edge),
                fmt_money(stake),
                "✅ Defensas sólidas; ritmo bajo esperado",
                calidad_apuesta(edge),
            ])
        # Over
        if "odds_over" in g and "total_line" in g:
            p_model = model_nba_over(g)
            edge, kelly, stake = compute_row(p_model, g["odds_over"], cfg)
            filas_nba_hoy.append([
                g.get("hora",""),
                f"{g.get('home_team','')} vs {g.get('away_team','')}",
                "Total Puntos",
                f"Over {g.get('total_line')}",
                round(g.get("odds_over"), 2),
                as_percent(edge),
                fmt_money(stake),
                "✅ Ataques eficientes; ritmo arriba del promedio",
                calidad_apuesta(edge),
            ])
        # ML visitante
        if "odds_ml_away" in g:
            p_model = model_nba_ml_away(g)
            edge, kelly, stake = compute_row(p_model, g["odds_ml_away"], cfg)
            filas_nba_hoy.append([
                g.get("hora",""),
                f"{g.get('home_team','')} vs {g.get('away_team','')}",
                "Moneyline",
                f"{g.get('away_team','')} Gana",
                round(g.get("odds_ml_away"), 2),
                as_percent(edge),
                fmt_money(stake),
                "✅ Forma y racha del visitante",
                calidad_apuesta(edge),
            ])

    hoy_nba = pd.DataFrame(
        filas_nba_hoy,
        columns=["HORA","PARTIDO","MERCADO","APUESTA","CUOTA","EDGE","STAKE","RAZÓN","CALIDAD"]
    )

    # Fútbol multiliga HOY
    soccer_by_league = extract_football_entries_multi(cfg, hoy_str)
    for liga_name, entries in soccer_by_league.items():
        soccer_by_league[liga_name] = supplement_with_theodds_football_league(cfg, hoy_str, liga_name, entries)

    hoy_futbol = {}
    for liga_name, matches in soccer_by_league.items():
        liga_rows = []
        for m in matches:
            if "odds_home" in m:
                p_model = model_soccer_home(m)
                edge, kelly, stake = compute_row(p_model, m["odds_home"], cfg)
                liga_rows.append([
                    m.get("hora",""),
                    f"{m.get('home_team','')} vs {m.get('away_team','')}",
                    "Moneyline",
                    m.get("home_team",""),
                    round(m.get("odds_home"),2),
                    as_percent(edge),
                    fmt_money(stake),
                    "✅ Ventaja local y forma reciente",
                    calidad_apuesta(edge),
                ])
            if "odds_btts_yes" in m:
                p_model = model_soccer_btts(m)
                edge, kelly, stake = compute_row(p_model, m["odds_btts_yes"], cfg)
                liga_rows.append([
                    m.get("hora",""),
                    f"{m.get('home_team','')} vs {m.get('away_team','')}",
                    "Ambos Anotan",
                    "Sí",
                    round(m.get("odds_btts_yes"),2),
                    as_percent(edge),
                    fmt_money(stake),
                    "✅ Tendencia ofensiva de ambos equipos",
                    calidad_apuesta(edge),
                ])
        df_liga = pd.DataFrame(liga_rows, columns=["HORA","PARTIDO","MERCADO","APUESTA","CUOTA","EDGE","STAKE","RAZÓN","CALIDAD"])
        hoy_futbol[liga_name] = df_liga

    # Tenis HOY
    tenis_matches = extract_tennis_entries(cfg, hoy_str)
    filas_tenis_hoy = []
    for t in tenis_matches:
        p_model = model_tennis_ml_p1(t)
        edge, kelly, stake = compute_row(p_model, t["odds_p1"], cfg)
        filas_tenis_hoy.append([
            t.get("hora",""),
            f"{t.get('player1','')} vs {t.get('player2','')}",
            "Moneyline",
            t.get("player1",""),
            round(t.get("odds_p1"),2),
            as_percent(edge),
            fmt_money(stake),
            "✅ Superioridad técnica y servicio",
            calidad_apuesta(edge),
        ])

    hoy_tenis = pd.DataFrame(
        filas_tenis_hoy,
        columns=["HORA","PARTIDO","MERCADO","APUESTA","CUOTA","EDGE","STAKE","RAZÓN","CALIDAD"]
    )

    # ---------- MAÑANA ----------
    nba_manana = extract_basketball_entries(cfg, manana_str)
    nba_manana = supplement_with_theodds_basketball(cfg, manana_str, nba_manana)
    filas_nba_man = []
    for g in nba_manana:
        if "odds_ml_home" in g:
            p_model = 0.59
            edge, kelly, stake = compute_row(p_model, g["odds_ml_home"], cfg)
            filas_nba_man.append([
                g.get("hora",""),
                f"{g.get('home_team','')} vs {g.get('away_team','')}",
                "Moneyline",
                g.get("home_team",""),
                round(g.get("odds_ml_home"),2),
                as_percent(edge),
                fmt_money(stake),
                "✅ Ventaja local y emparejamiento favorable",
                calidad_apuesta(edge),
            ])
        if "odds_over" in g and "total_line" in g:
            p_model = 0.57
            edge, kelly, stake = compute_row(p_model, g["odds_over"], cfg)
            filas_nba_man.append([
                g.get("hora",""),
                f"{g.get('home_team','')} vs {g.get('away_team','')}",
                "Total Puntos",
                f"Over {g.get('total_line')}",
                round(g.get("odds_over"),2),
                as_percent(edge),
                fmt_money(stake),
                "✅ Proyección de ritmo alto",
                calidad_apuesta(edge),
            ])

    manana_nba = pd.DataFrame(
        filas_nba_man,
        columns=["HORA","PARTIDO","MERCADO","APUESTA","CUOTA","EDGE","STAKE","RAZÓN","CALIDAD"]
    )

    # Fútbol multiliga MAÑANA
    soccer_by_league_man = extract_football_entries_multi(cfg, manana_str)
    for liga_name, entries in soccer_by_league_man.items():
        soccer_by_league_man[liga_name] = supplement_with_theodds_football_league(cfg, manana_str, liga_name, entries)

    manana_futbol = {}
    for liga_name, matches in soccer_by_league_man.items():
        liga_rows = []
        for m in matches:
            if "odds_home" in m:
                p_model = model_soccer_home(m)
                edge, kelly, stake = compute_row(p_model, m["odds_home"], cfg)
                liga_rows.append([
                    m.get("hora",""),
                    f"{m.get('home_team','')} vs {m.get('away_team','')}",
                    "Moneyline",
                    m.get("home_team",""),
                    round(m.get("odds_home"),2),
                    as_percent(edge),
                    fmt_money(stake),
                    "✅ Ventaja local y forma reciente",
                    calidad_apuesta(edge),
                ])
            if "odds_btts_yes" in m:
                p_model = model_soccer_btts(m)
                edge, kelly, stake = compute_row(p_model, m["odds_btts_yes"], cfg)
                liga_rows.append([
                    m.get("hora",""),
                    f"{m.get('home_team','')} vs {m.get('away_team','')}",
                    "Ambos Anotan",
                    "Sí",
                    round(m.get("odds_btts_yes"),2),
                    as_percent(edge),
                    fmt_money(stake),
                    "✅ Tendencia ofensiva de ambos equipos",
                    calidad_apuesta(edge),
                ])
        df_liga = pd.DataFrame(liga_rows, columns=["HORA","PARTIDO","MERCADO","APUESTA","CUOTA","EDGE","STAKE","RAZÓN","CALIDAD"])
        manana_futbol[liga_name] = df_liga

    # Tenis MAÑANA
    tenis_matches_man = extract_tennis_entries(cfg, manana_str)
    filas_tenis_man = []
    for t in tenis_matches_man:
        p_model = model_tennis_ml_p1(t)
        edge, kelly, stake = compute_row(p_model, t["odds_p1"], cfg)
        filas_tenis_man.append([
            t.get("hora",""),
            f"{t.get('player1','')} vs {t.get('player2','')}",
            "Moneyline",
            t.get("player1",""),
            round(t.get("odds_p1"),2),
            as_percent(edge),
            fmt_money(stake),
            "✅ Superioridad técnica y servicio",
            calidad_apuesta(edge),
        ])

    manana_tenis = pd.DataFrame(
        filas_tenis_man,
        columns=["HORA","PARTIDO","MERCADO","APUESTA","CUOTA","EDGE","STAKE","RAZÓN","CALIDAD"]
    )

    return {
        "hoy_fecha": hoy,
        "manana_fecha": manana,
        "hoy": {
            "NBA": hoy_nba,
            "FÚTBOL": hoy_futbol,   # dict por liga
            "TENIS": hoy_tenis,
        },
        "manana": {
            "NBA": manana_nba,
            "FÚTBOL": manana_futbol,  # dict por liga
            "TENIS": manana_tenis,
        },
    }

# =============================================================================
# FILTRO VISUAL POR EDGE
# =============================================================================
def df_filtrado(df, min_edge):
    """
    Muestra solo picks con edge >= min_edge en el Home.
    Convierte 'EDGE' (string con %) a float temporal para filtrar.
    """
    if df.empty:
        return df
    df2 = df.copy()
    try:
        df2["EDGE_FLOAT"] = df2["EDGE"].astype(str).str.replace("%","").str.replace(",",".").astype(float)/100
        df2 = df2[df2["EDGE_FLOAT"] >= float(min_edge)]
    except Exception:
        pass
    return df2.drop(columns=[c for c in ["EDGE_FLOAT"] if c in df2.columns])

# =============================================================================
# STREAMLIT APP
# =============================================================================
st.set_page_config(layout="wide", page_icon="🏆", page_title="BETTING PRO")

# Estado (config + historial)
if "config" not in st.session_state:
    st.session_state.config = load_config()
if "historial" not in st.session_state:
    st.session_state.historial = init_historial()

cfg = st.session_state.config
hist = st.session_state.historial

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Configuración")
    st.caption(f"BETTING PRO — {APP_VERSION}")

    cfg["bankroll"] = st.number_input("Bankroll base", 1000, 200000, int(cfg["bankroll"]), step=500)
    # Edge flexible: permite 0–20%
    cfg["min_edge"]  = st.slider("Edge mínimo (%)", 0, 20, int(cfg["min_edge"]*100))/100
    cfg["max_kelly"] = st.slider("Kelly máximo (%)", 5, 25, int(cfg["max_kelly"]*100))/100
    cfg["auto_cashout"] = st.slider("ROI Cash Out (%)", 10, 50, int(cfg["auto_cashout"]*100))/100
    st.markdown("---")
    st.subheader("📅 Temporadas (season)")

    cfg["FB_SEASON"] = st.text_input(
    "Season Fútbol (ej. 2025-2026)",
    value=cfg.get("FB_SEASON", "2025-2026")
    )

    cfg["BB_SEASON"] = st.text_input(
    "Season NBA (ej. 2025-2026)",
    value=cfg.get("BB_SEASON", "2025-2026")
    )

    st.markdown("---")
    st.subheader("🔑 API Keys API‑Sports & The Odds API")
    cfg["API_SPORTS_KEY"]     = st.text_input("API‑Sports (fallback general)", value=cfg.get("API_SPORTS_KEY",""))
    cfg["FOOTBALL_API_KEY"]   = st.text_input("Football API key", value=cfg.get("FOOTBALL_API_KEY",""))
    cfg["BASKETBALL_API_KEY"] = st.text_input("Basketball API key", value=cfg.get("BASKETBALL_API_KEY",""))
    cfg["TENNIS_API_KEY"]     = st.text_input("Tennis API key", value=cfg.get("TENNIS_API_KEY",""))
    cfg["ODDS_API_KEY"]       = st.text_input("The Odds API key", value=cfg.get("ODDS_API_KEY",""))

    st.markdown("---")
    st.subheader("⚽ Ligas de fútbol (API‑Sports)")
    liga_str = st.text_area(
        "IDs de liga separados por coma",
        value=",".join(str(x) for x in cfg.get("FB_LEAGUES", [39]))
    )
    try:
        cfg["FB_LEAGUES"] = [int(x.strip()) for x in liga_str.split(",") if x.strip()]
    except Exception:
        st.warning("No se pudo parsear la lista de ligas. Manteniendo configuración previa.")

    st.markdown("---")
    st.subheader("ℹ️ ¿Qué es el Edge?")
    with st.expander("Explicación del edge y cómo usarlo"):
        st.markdown("""
        El **edge** es la ventaja matemática de tu modelo frente a la cuota de la casa:
        - Edge = Probabilidad modelo – Probabilidad implícita (1/cuota).
        - Si el edge es positivo, hay valor esperado en la apuesta.
        - Si es negativo, la cuota está 'inflada' y conviene evitarla.
        Usa el control **Edge mínimo (%)** para filtrar picks visibles en el Home.
        """)

    st.markdown("---")
    st.subheader("🧪 Test APIs API‑Sports")
    col_api1, col_api2, col_api3 = st.columns(3)
    if col_api1.button("Test Fútbol"):
        try:
            url = "https://v3.football.api-sports.io/status"
            headers = {"x-apisports-key": cfg["FOOTBALL_API_KEY"] or cfg["API_SPORTS_KEY"]}
            r = requests.get(url, headers=headers, timeout=10)
            st.write("HTTP:", r.status_code); st.json(r.json())
        except Exception as e:
            st.error(f"Error: {e}")
    if col_api2.button("Test Basket"):
        try:
            url = "https://v1.basketball.api-sports.io/status"
            headers = {"x-apisports-key": cfg["BASKETBALL_API_KEY"] or cfg["API_SPORTS_KEY"]}
            r = requests.get(url, headers=headers, timeout=10)
            st.write("HTTP:", r.status_code); st.json(r.json())
        except Exception as e:
            st.error(f"Error: {e}")
    if col_api3.button("Test Tenis"):
        try:
            url = "https://v1.tennis.api-sports.io/status"
            headers = {"x-apisports-key": cfg["TENNIS_API_KEY"] or cfg["API_SPORTS_KEY"]}
            r = requests.get(url, headers=headers, timeout=10)
            st.write("HTTP:", r.status_code); st.json(r.json())
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("---")
    st.subheader("🧪 Test The Odds API")
    if st.button("Test Odds API"):
        try:
            url = "https://api.the-odds-api.com/v4/sports"
            params = {"apiKey": cfg["ODDS_API_KEY"]}
            r = requests.get(url, params=params, timeout=10)
            st.write("HTTP:", r.status_code); st.json(r.json())
        except Exception as e:
            st.error(f"Error: {e}")

    if st.button("💾 Guardar configuración", use_container_width=True):
        save_config(cfg); st.success("Configuración guardada.")

    with st.expander("📚 Diccionario de mercados"):
        st.markdown(
            "- **Moneyline:** Ganador del partido.\n"
            "- **Totales:** Over/Under sobre puntos/goles.\n"
            "- **BTTS (Ambos anotan):** Si ambos equipos marcarán al menos un gol.\n"
            "- **ROI:** Retorno relativo de la apuesta.\n"
            "- **Edge:** p_model - 1/cuota.\n"
            "- **Kelly:** Fracción del bankroll sugerida para stake."
        )

# ---------------- Header ----------------
hoy, manana = get_local_dates()
st.markdown(
    f"<h2 style='text-align:center'>🏆 PRONÓSTICOS COMPLETOS - {hoy.strftime('%d/%m/%Y')} Y {manana.strftime('%d/%m/%Y')}</h2>",
    unsafe_allow_html=True
)

c1,c2,c3 = st.columns(3)
c1.metric("Bankroll base", fmt_money(cfg['bankroll']))
total_bets = len(hist)
ganadas = len(hist[hist["status"]=="GANADO"])
winrate = ganadas/total_bets if total_bets>0 else 0
c2.metric("Winrate histórico", f"{winrate:.1%}")
c3.metric("Total apuestas", str(total_bets))

# ---------------- Tabs ----------------
tab_home, tab_res, tab_rep, tab_val = st.tabs(
    ["🏠 Home (pronósticos)", "✅ Resultados", "📈 Reportes", "💎 Value Bets"]
)

data_home = build_home_tables(cfg)

# =============================================================================
# HOME
# =============================================================================
with tab_home:
    hoy = data_home["hoy_fecha"]
    manana = data_home["manana_fecha"]
    tab_hoy, tab_man = st.tabs(
        [f"📅 HOY {hoy.strftime('%d/%m/%Y')}", f"📅 MAÑANA {manana.strftime('%d/%m/%Y')}"]
    )

    # ---------- HOY ----------
    with tab_hoy:
        # NBA
        st.markdown("### 🏀 NBA - Juegos del día")
        df_nba = df_filtrado(data_home["hoy"]["NBA"], cfg["min_edge"])
        if df_nba.empty:
            st.info("ℹ️ No se encontraron partidos o mercados de NBA con edge suficiente para hoy.")
        else:
            st.dataframe(df_nba, hide_index=True, use_container_width=True)

        # Fútbol multiliga
        st.markdown("### ⚽ FÚTBOL - Múltiples ligas")
        for liga_name, df_liga in data_home["hoy"]["FÚTBOL"].items():
            st.markdown(f"#### {liga_name}")
            df_liga_f = df_filtrado(df_liga, cfg["min_edge"])
            if df_liga.empty:
                st.info(f"ℹ️ No se encontraron partidos ni mercados para {liga_name} hoy.")
            elif df_liga_f.empty:
                st.info(f"ℹ️ Hay partidos en {liga_name}, pero ningún pick supera el edge mínimo configurado.")
            else:
                st.dataframe(df_liga_f, hide_index=True, use_container_width=True)

        # Tenis
        st.markdown("### 🎾 TENIS - Principales eventos")
        df_tenis = df_filtrado(data_home["hoy"]["TENIS"], cfg["min_edge"])
        if df_tenis.empty:
            st.info("ℹ️ No se encontraron mercados de Tenis con edge suficiente para hoy.")
        else:
            st.dataframe(df_tenis, hide_index=True, use_container_width=True)

        # Enviar HOY al historial
        if st.button("➕ Enviar pronósticos HOY al historial"):
            nuevos = []
            # NBA
            for _, r in df_nba.iterrows():
                edge = edge_str_to_float(r["EDGE"])
                stake = stake_str_to_float(r["STAKE"])
                cuota = float(r["CUOTA"])
                p_model = edge + 1/cuota
                if edge < cfg["min_edge"]:
                    continue
                pick_id = make_pick_id(str(hoy), r["PARTIDO"], r["MERCADO"], r["APUESTA"])
                nuevos.append({
                    "fecha": str(hoy), "dia": "HOY", "deporte": "NBA",
                    "partido": r["PARTIDO"], "hora": r["HORA"], "mercado": r["MERCADO"], "apuesta": r["APUESTA"],
                    "cuota": cuota, "probabilidad": float(p_model), "edge": float(edge),
                    "kelly": float(kelly_fraction(edge, cuota, cfg["max_kelly"])),
                    "stake": stake if stake>0 else cfg["bankroll"]*kelly_fraction(edge, cuota, cfg["max_kelly"]),
                    "status": "PENDIENTE", "roi_real": np.nan, "pick_id": pick_id,
                })
            # Fútbol por liga
            for liga_name, df_liga in data_home["hoy"]["FÚTBOL"].items():
                df_liga_f = df_filtrado(df_liga, cfg["min_edge"])
                for _, r in df_liga_f.iterrows():
                    edge = edge_str_to_float(r["EDGE"])
                    stake = stake_str_to_float(r["STAKE"])
                    cuota = float(r["CUOTA"])
                    p_model = edge + 1/cuota
                    if edge < cfg["min_edge"]:
                        continue
                    pick_id = make_pick_id(str(hoy), r["PARTIDO"], r["MERCADO"], r["APUESTA"])
                    nuevos.append({
                        "fecha": str(hoy), "dia": "HOY", "deporte": f"FÚTBOL · {liga_name}",
                        "partido": r["PARTIDO"], "hora": r["HORA"], "mercado": r["MERCADO"], "apuesta": r["APUESTA"],
                        "cuota": cuota, "probabilidad": float(p_model), "edge": float(edge),
                        "kelly": float(kelly_fraction(edge, cuota, cfg["max_kelly"])),
                        "stake": stake if stake>0 else cfg["bankroll"]*kelly_fraction(edge, cuota, cfg["max_kelly"]),
                        "status": "PENDIENTE", "roi_real": np.nan, "pick_id": pick_id,
                    })
            # Tenis
            for _, r in df_tenis.iterrows():
                edge = edge_str_to_float(r["EDGE"])
                stake = stake_str_to_float(r["STAKE"])
                cuota = float(r["CUOTA"])
                p_model = edge + 1/cuota
                if edge < cfg["min_edge"]:
                    continue
                pick_id = make_pick_id(str(hoy), r["PARTIDO"], r["MERCADO"], r["APUESTA"])
                nuevos.append({
                    "fecha": str(hoy), "dia": "HOY", "deporte": "TENIS",
                    "partido": r["PARTIDO"], "hora": r["HORA"], "mercado": r["MERCADO"], "apuesta": r["APUESTA"],
                    "cuota": cuota, "probabilidad": float(p_model), "edge": float(edge),
                    "kelly": float(kelly_fraction(edge, cuota, cfg["max_kelly"])),
                    "stake": stake if stake>0 else cfg["bankroll"]*kelly_fraction(edge, cuota, cfg["max_kelly"]),
                    "status": "PENDIENTE", "roi_real": np.nan, "pick_id": pick_id,
                })

            if nuevos:
                new_df = pd.DataFrame(nuevos)
                # Prevención de duplicados
                if "pick_id" not in hist.columns:
                    hist["pick_id"] = hist.apply(lambda r: make_pick_id(r["fecha"], r["partido"], r["mercado"], r["apuesta"]), axis=1)
                exists = set(hist["pick_id"].astype(str).tolist())
                new_df = new_df[~new_df["pick_id"].astype(str).isin(exists)]
                if not new_df.empty:
                    hist = pd.concat([hist, new_df], ignore_index=True)
                    st.session_state.historial = hist
                    save_historial(hist)
                    st.success(f"{len(new_df)} apuestas añadidas al historial.")
                else:
                    st.info("No se añadieron apuestas: posibles duplicados.")
            else:
                st.info("No hay picks con edge suficiente para añadir.")

    # ---------- MAÑANA ----------
    with tab_man:
        # NBA mañana
        st.markdown("### 🏀 NBA - Juegos de mañana")
        df_man_nba = df_filtrado(data_home["manana"]["NBA"], cfg["min_edge"])
        if df_man_nba.empty:
            st.info("ℹ️ No se encontraron partidos o mercados de NBA para mañana con edge suficiente.")
        else:
            st.dataframe(df_man_nba, hide_index=True, use_container_width=True)

        # Fútbol mañana multiliga
        st.markdown("### ⚽ FÚTBOL - Múltiples ligas (mañana)")
        for liga_name, df_liga in data_home["manana"]["FÚTBOL"].items():
            st.markdown(f"#### {liga_name}")
            df_liga_f = df_filtrado(df_liga, cfg["min_edge"])
            if df_liga.empty:
                st.info(f"ℹ️ No se encontraron partidos ni mercados para {liga_name} mañana.")
            elif df_liga_f.empty:
                st.info(f"ℹ️ Hay partidos en {liga_name} mañana, pero ningún pick supera el edge mínimo configurado.")
            else:
                st.dataframe(df_liga_f, hide_index=True, use_container_width=True)

        # Tenis mañana
        st.markdown("### 🎾 TENIS - Principales eventos (mañana)")
        df_tenis_man = df_filtrado(data_home["manana"]["TENIS"], cfg["min_edge"])
        if df_tenis_man.empty:
            st.info("ℹ️ No se encontraron mercados de Tenis para mañana con edge suficiente.")
        else:
            st.dataframe(df_tenis_man, hide_index=True, use_container_width=True)

        # Enviar MAÑANA al historial
        if st.button("➕ Enviar pronósticos MAÑANA al historial"):
            nuevos = []
            # NBA
            for _, r in df_man_nba.iterrows():
                edge = edge_str_to_float(r["EDGE"])
                stake = stake_str_to_float(r["STAKE"])
                cuota = float(r["CUOTA"])
                p_model = edge + 1/cuota
                if edge < cfg["min_edge"]:
                    continue
                pick_id = make_pick_id(str(manana), r["PARTIDO"], r["MERCADO"], r["APUESTA"])
                nuevos.append({
                    "fecha": str(manana), "dia": "MAÑANA", "deporte": "NBA",
                    "partido": r["PARTIDO"], "hora": r["HORA"], "mercado": r["MERCADO"], "apuesta": r["APUESTA"],
                    "cuota": cuota, "probabilidad": float(p_model), "edge": float(edge),
                    "kelly": float(kelly_fraction(edge, cuota, cfg["max_kelly"])),
                    "stake": stake if stake>0 else cfg["bankroll"]*kelly_fraction(edge, cuota, cfg["max_kelly"]),
                    "status": "PENDIENTE", "roi_real": np.nan, "pick_id": pick_id,
                })
            # Fútbol por liga
            for liga_name, df_liga in data_home["manana"]["FÚTBOL"].items():
                df_liga_f = df_filtrado(df_liga, cfg["min_edge"])
                for _, r in df_liga_f.iterrows():
                    edge = edge_str_to_float(r["EDGE"])
                    stake = stake_str_to_float(r["STAKE"])
                    cuota = float(r["CUOTA"])
                    p_model = edge + 1/cuota
                    if edge < cfg["min_edge"]:
                        continue
                    pick_id = make_pick_id(str(manana), r["PARTIDO"], r["MERCADO"], r["APUESTA"])
                    nuevos.append({
                        "fecha": str(manana), "dia": "MAÑANA", "deporte": f"FÚTBOL · {liga_name}",
                        "partido": r["PARTIDO"], "hora": r["HORA"], "mercado": r["MERCADO"], "apuesta": r["APUESTA"],
                        "cuota": cuota, "probabilidad": float(p_model), "edge": float(edge),
                        "kelly": float(kelly_fraction(edge, cuota, cfg["max_kelly"])),
                        "stake": stake if stake>0 else cfg["bankroll"]*kelly_fraction(edge, cuota, cfg["max_kelly"]),
                        "status": "PENDIENTE", "roi_real": np.nan, "pick_id": pick_id,
                    })
            # Tenis
            for _, r in df_tenis_man.iterrows():
                edge = edge_str_to_float(r["EDGE"])
                stake = stake_str_to_float(r["STAKE"])
                cuota = float(r["CUOTA"])
                p_model = edge + 1/cuota
                if edge < cfg["min_edge"]:
                    continue
                pick_id = make_pick_id(str(manana), r["PARTIDO"], r["MERCADO"], r["APUESTA"])
                nuevos.append({
                    "fecha": str(manana), "dia": "MAÑANA", "deporte": "TENIS",
                    "partido": r["PARTIDO"], "hora": r["HORA"], "mercado": r["MERCADO"], "apuesta": r["APUESTA"],
                    "cuota": cuota, "probabilidad": float(p_model), "edge": float(edge),
                    "kelly": float(kelly_fraction(edge, cuota, cfg["max_kelly"])),
                    "stake": stake if stake>0 else cfg["bankroll"]*kelly_fraction(edge, cuota, cfg["max_kelly"]),
                    "status": "PENDIENTE", "roi_real": np.nan, "pick_id": pick_id,
                })

            if nuevos:
                new_df = pd.DataFrame(nuevos)
                if "pick_id" not in hist.columns:
                    hist["pick_id"] = hist.apply(lambda r: make_pick_id(r["fecha"], r["partido"], r["mercado"], r["apuesta"]), axis=1)
                exists = set(hist["pick_id"].astype(str).tolist())
                new_df = new_df[~new_df["pick_id"].astype(str).isin(exists)]
                if not new_df.empty:
                    hist = pd.concat([hist, new_df], ignore_index=True)
                    st.session_state.historial = hist
                    save_historial(hist)
                    st.success(f"{len(new_df)} apuestas añadidas al historial.")
                else:
                    st.info("No se añadieron apuestas: posibles duplicados.")
            else:
                st.info("No hay picks con edge suficiente para añadir.")

# =============================================================================
# RESULTADOS
# =============================================================================
with tab_res:
    st.subheader("✅ Marcar resultados")
    pendientes = hist[hist["status"]=="PENDIENTE"]
    if pendientes.empty:
        st.success("No hay apuestas pendientes.")
    else:
        for idx,row in pendientes.iterrows():
            c1,c2,c3,c4 = st.columns([3,1,1,1])
            c1.markdown(f"**{row['deporte']}** – {row['partido']} ({row['mercado']} · {row['apuesta']})")
            c1.caption(f"Stake: {fmt_money(row['stake'])} | Cuota: {row['cuota']:.2f}")
            if c2.button("GANÓ", key=f"w_{idx}"):
                hist.at[idx,"status"] = "GANADO"
                hist.at[idx,"roi_real"] = row["cuota"]-1
                save_historial(hist); st.session_state.historial = hist; st.rerun()
            if c3.button("PERDIÓ", key=f"l_{idx}"):
                hist.at[idx,"status"] = "PERDIDO"
                hist.at[idx,"roi_real"] = -1.0
                save_historial(hist); st.session_state.historial = hist; st.rerun()
            if c4.button("CASH OUT", key=f"c_{idx}"):
                hist.at[idx,"status"] = "CASH_OUT"
                hist.at[idx,"roi_real"] = cfg["auto_cashout"]
                save_historial(hist); st.session_state.historial = hist; st.rerun()

# =============================================================================
# REPORTES
# =============================================================================
with tab_rep:
    st.subheader("📈 Reportes")
    if hist.empty:
        st.info("Sin datos todavía.")
    else:
        comp = hist[hist["status"].isin(["GANADO","PERDIDO","CASH_OUT"])]
        if comp.empty:
            st.info("Aún no hay apuestas cerradas.")
        else:
            ganancia = (comp["roi_real"]*comp["stake"]).sum()
            winrate_rep = (comp["status"]=="GANADO").mean()
            roi_medio = comp["roi_real"].mean()
            c1,c2,c3 = st.columns(3)
            c1.metric("Winrate", f"{winrate_rep:.1%}")
            c2.metric("ROI medio", f"{roi_medio:.1%}")
            c3.metric("Ganancia total", fmt_money(ganancia))

            st.markdown("#### Por deporte")
            por_dep = (
                comp.groupby("deporte")
                .agg(
                    apuestas=("status","count"),
                    winrate=("status", lambda s:(s=="GANADO").mean()),
                    stake_total=("stake","sum"),
                    roi_medio=("roi_real","mean"),
                ).round(3)
            )
            st.dataframe(por_dep)

        if st.button("📥 Exportar historial a Excel"):
            hist.to_excel("betting_report.xlsx", index=False)
            st.success("Exportado a betting_report.xlsx")

# =============================================================================
# VALUE BETS
# =============================================================================
with tab_val:
    st.subheader("💎 Value Bets (The Odds API)")
    the_sports = fetch_theodds_sports(cfg)
    nba_keys = [s.get("key") for s in the_sports if "basketball_nba" in s.get("key","")]
    epl_keys = [s.get("key") for s in the_sports if "soccer_epl" in s.get("key","")]
    vb_df = pd.DataFrame()
    if nba_keys:
        vb_df = find_value_bets_theodds(cfg, nba_keys[0])
    elif epl_keys:
        vb_df = find_value_bets_theodds(cfg, epl_keys[0])

    if vb_df.empty:
        st.info("No se pudo construir Value Bets dinámicas (sin datos o claves). Revisa tus API keys.")
    else:
        st.dataframe(vb_df, hide_index=True, use_container_width=True)
