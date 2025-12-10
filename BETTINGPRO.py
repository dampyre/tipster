# =============================================================================
# BETTING PRO — v18.9 (Odds API + TheSportsDB + WagerLab, con mapeo y fallback)
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import os, json
import re

CONFIG_FILE = "betting_config.json"
HIST_FILE   = "betting_historial.csv"
APP_VERSION = "v18.9"

ODDS_SPORT_KEYS = {
    "NBA": "basketball_nba",
    "UCL": "soccer_uefa_champs_league",
    "EPL": "soccer_epl",
    "LA_LIGA": "soccer_spain_la_liga",
    "SERIE_A": "soccer_italy_serie_a",
    "BUNDESLIGA": "soccer_germany_bundesliga",
    "ATP": "tennis_atp",
}

CL_TZ = ZoneInfo("America/Santiago")

# ---------------- Configuración ----------------
def load_config():
    default = {
        "bankroll": 10000,
        "min_edge": 0.07,
        "max_kelly": 0.10,
        "auto_cashout": 0.30,
        "ODDS_API_KEY": "",
        "THE_ODDS_REGIONS": "eu",
        "THE_ODDS_MARKETS": "h2h,totals,btts",
        "THE_ODDS_FORMAT": "decimal",
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            default.update(data)
        except Exception:
            pass
    return default

def save_config(cfg):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

def init_historial():
    cols = ["fecha","dia","deporte","partido","hora","mercado","apuesta",
            "cuota","probabilidad","edge","kelly","stake","status","roi_real","pick_id"]
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
    df.to_csv(HIST_FILE, index=False)

# ---------------- Utilidades ----------------
def as_percent(x): return f"{x*100:.1f}%"
def fmt_money(x): return f"${x:,.0f}"
def kelly_fraction(edge, cuota, max_kelly):
    if edge <= 0 or cuota <= 1.0: return 0.0
    f = edge / (cuota - 1)
    return max(0.0, min(f, max_kelly))
def implied_prob_decimal(odds): return 1/float(odds) if odds and odds>0 else 0.0
def get_local_dates():
    hoy = datetime.now(CL_TZ).date()
    manana = hoy + timedelta(days=1)
    return hoy, manana
def calidad_apuesta(edge):
    if edge < 0: return "❌ Sin valor"
    elif edge < 0.03: return "⚠️ Bajo valor"
    elif edge < 0.07: return "✅ Valor moderado"
    else: return "💎 Alto valor"
def make_pick_id(fecha, partido, mercado, apuesta):
    return f"{fecha}|{partido}|{mercado}|{apuesta}"

# Normalizador de nombres para mapeo
def norm_name(s: str) -> str:
    if not s: return ""
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s\-]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

# ---------------- Odds API (principal) ----------------
@st.cache_data(ttl=300)
def odds_fetch_events(cfg, sport_key):
    params = {
        "apiKey": cfg.get("ODDS_API_KEY",""),
        "regions": cfg.get("THE_ODDS_REGIONS","eu"),
        "markets": cfg.get("THE_ODDS_MARKETS","h2h,totals,btts"),
        "oddsFormat": cfg.get("THE_ODDS_FORMAT","decimal"),
    }
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []

def odds_filter_by_date(events, fecha_iso):
    return [ev for ev in events if (ev.get("commence_time") or "").startswith(fecha_iso)]

# ---------------- TheSportsDB (fixtures actuales, fallback) ----------------
@st.cache_data(ttl=300)
def thesportsdb_fixtures(fecha: str, sport: str="Soccer"):
    url = f"https://www.thesportsdb.com/api/v1/json/1/eventsday.php?d={fecha}&s={sport}"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return r.json().get("events", [])
    except Exception:
        return []

# ---------------- WagerLab Odds API (cuotas fallback) ----------------
@st.cache_data(ttl=300)
def wagerlab_odds(sport: str):
    url = f"https://api.wagerlab.app/odds?sport={sport}"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []

# ---------------- Debug ----------------
def debug_log_events(events, label):
    st.caption(f"DEBUG: {label} → {len(events)} eventos recibidos")
    if events:
        st.json([
            {"home": ev.get("home_team"), "away": ev.get("away_team"), "time": ev.get("commence_time")}
            for ev in events
        ])

def debug_log_generic(items, label, max_items=5):
    st.caption(f"DEBUG: {label} → {len(items)} items")
    if items:
        st.json(items[:max_items])

# ---------------- Probabilidades ajustadas ----------------
def fair_probs_from_prices(prices):
    ps = [float(p) for p in prices if p and float(p) > 1.0]
    if not ps: return 0.0
    avg = sum(ps) / len(ps)
    best = max(ps)
    p_avg = implied_prob_decimal(avg)
    p_best = implied_prob_decimal(best)
    overround = max(0.0, p_avg - p_best)
    overround = min(overround, 0.05)
    return max(0.0, min(1.0, p_avg - overround))

def collect_prices_for_market(ev, key, flt):
    prices = []
    for bk in ev.get("bookmakers", []):
        for mk in bk.get("markets", []):
            if mk.get("key") == key:
                for outc in mk.get("outcomes", []):
                    if flt(outc):
                        pr = outc.get("price")
                        if pr:
                            prices.append(float(pr))
    return prices

# Modelos basados en consenso
def model_soccer_home_prob(ev):
    home = (ev.get("home_team") or "").lower()
    prices = collect_prices_for_market(
        ev, "h2h",
        lambda o: "home" in (o.get("name") or "").lower() or home in (o.get("name") or "").lower()
    )
    base = fair_probs_from_prices(prices)
    adj = 0.03 if base >= 0.60 else 0.015 if base >= 0.50 else 0.005
    return min(1.0, base + adj)

def model_soccer_btts_yes_prob(ev):
    prices = collect_prices_for_market(ev, "btts", lambda o: "yes" in (o.get("name") or "").lower())
    base = fair_probs_from_prices(prices) if prices else 0.55
    return 0.7 * base + 0.3 * 0.55

def model_nba_ml_home_prob(ev):
    home = (ev.get("home_team") or "").lower()
    prices = collect_prices_for_market(
        ev, "h2h",
        lambda o: "home" in (o.get("name") or "").lower() or home in (o.get("name") or "").lower()
    )
    base = fair_probs_from_prices(prices)
    adj = 0.02 if base >= 0.55 else 0.01
    return min(1.0, base + adj)

def model_nba_totals_over_prob(ev, total_line, over_price):
    prices = collect_prices_for_market(ev, "totals", lambda o: "over" in (o.get("description") or "").lower())
    p_over = fair_probs_from_prices(prices) if prices else implied_prob_decimal(over_price or 2.0)
    return max(0.35, min(0.65, p_over))

def model_tennis_p1_prob(ev):
    p1 = (ev.get("home_team") or "").lower()
    prices = collect_prices_for_market(
        ev, "h2h",
        lambda o: "home" in (o.get("name") or "").lower() or p1 in (o.get("name") or "").lower()
    )
    base = fair_probs_from_prices(prices)
    return 0.65 * base + 0.35 * 0.52

# ---------------- Extractores Odds API ----------------
def extract_soccer_events(cfg, fecha_iso, sport_keys):
    out = {}
    for sk in sport_keys:
        evs = odds_fetch_events(cfg, sk)
        evs = odds_filter_by_date(evs, fecha_iso)
        name = sk
        if sk == ODDS_SPORT_KEYS["UCL"]: name = "UEFA Champions League"
        elif sk == ODDS_SPORT_KEYS["EPL"]: name = "Premier League"
        elif sk == ODDS_SPORT_KEYS["LA_LIGA"]: name = "La Liga"
        elif sk == ODDS_SPORT_KEYS["SERIE_A"]: name = "Serie A"
        elif sk == ODDS_SPORT_KEYS["BUNDESLIGA"]: name = "Bundesliga"
        out[name] = evs
    return out

def extract_nba_events(cfg, fecha_iso):
    return odds_filter_by_date(odds_fetch_events(cfg, ODDS_SPORT_KEYS["NBA"]), fecha_iso)

def extract_tennis_events(cfg, fecha_iso):
    return odds_filter_by_date(odds_fetch_events(cfg, ODDS_SPORT_KEYS["ATP"]), fecha_iso)

# ---------------- Build DataFrames (Odds API) ----------------
def build_soccer_df(events, cfg):
    rows = []
    for ev in events:
        match = f"{ev.get('home_team')} vs {ev.get('away_team')}"
        hora = (ev.get("commence_time") or "")[11:16]
        # ML local
        ph = collect_prices_for_market(ev, "h2h", lambda o: "home" in (o.get("name") or "").lower())
        ml_home = max(ph) if ph else None
        if ml_home:
            p = model_soccer_home_prob(ev)
            edge = p - implied_prob_decimal(ml_home)
            k = kelly_fraction(edge, ml_home, cfg["max_kelly"])
            s = cfg["bankroll"] * max(k, 0)
            rows.append([hora, match, "Moneyline", "Local", round(ml_home,2),
                         as_percent(edge), fmt_money(s), "✅ Localía consenso", calidad_apuesta(edge)])
        # BTTS Yes
        py = collect_prices_for_market(ev, "btts", lambda o: "yes" in (o.get("name") or "").lower())
        btts_yes = max(py) if py else None
        if btts_yes:
            p = model_soccer_btts_yes_prob(ev)
            edge = p - implied_prob_decimal(btts_yes)
            k = kelly_fraction(edge, btts_yes, cfg["max_kelly"])
            s = cfg["bankroll"] * max(k, 0)
            rows.append([hora, match, "Ambos Anotan", "Sí", round(btts_yes,2),
                         as_percent(edge), fmt_money(s), "✅ Tendencia ofensiva", calidad_apuesta(edge)])
    cols = ["HORA","PARTIDO","MERCADO","APUESTA","CUOTA","EDGE","STAKE","RAZÓN","CALIDAD"]
    return pd.DataFrame(rows, columns=cols)

def build_nba_df(events, cfg):
    rows = []
    for ev in events:
        match = f"{ev.get('home_team')} vs {ev.get('away_team')}"
        hora = (ev.get("commence_time") or "")[11:16]
        # ML local
        ph = collect_prices_for_market(ev, "h2h", lambda o: "home" in (o.get("name") or "").lower())
        ml_home = max(ph) if ph else None
        if ml_home:
            p = model_nba_ml_home_prob(ev)
            edge = p - implied_prob_decimal(ml_home)
            k = kelly_fraction(edge, ml_home, cfg["max_kelly"])
            s = cfg["bankroll"] * max(k, 0)
            rows.append([hora, match, "Moneyline", "Local", round(ml_home,2),
                         as_percent(edge), fmt_money(s), "✅ Forma local", calidad_apuesta(edge)])
        # Totales Over
        po = collect_prices_for_market(ev, "totals", lambda o: "over" in (o.get("description") or "").lower())
        over = max(po) if po else None
        total_line = None
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                if mk.get("key") == "totals":
                    for outc in mk.get("outcomes", []):
                        if outc.get("point") is not None:
                            total_line = outc.get("point")
                            break
                if total_line is not None: break
            if total_line is not None: break
        if over and total_line is not None:
            p = model_nba_totals_over_prob(ev, total_line, over)
            edge = p - implied_prob_decimal(over)
            k = kelly_fraction(edge, over, cfg["max_kelly"])
            s = cfg["bankroll"] * max(k, 0)
            rows.append([hora, match, "Total Puntos", f"Over {total_line}", round(over,2),
                         as_percent(edge), fmt_money(s), "✅ Ritmo alto", calidad_apuesta(edge)])
    cols = ["HORA","PARTIDO","MERCADO","APUESTA","CUOTA","EDGE","STAKE","RAZÓN","CALIDAD"]
    return pd.DataFrame(rows, columns=cols)

def build_tennis_df(events, cfg):
    rows = []
    for ev in events:
        match = f"{ev.get('home_team')} vs {ev.get('away_team')}"
        hora = (ev.get("commence_time") or "")[11:16]
        p1_prices = collect_prices_for_market(ev, "h2h", lambda o: "home" in (o.get("name") or "").lower())
        p1_odds = max(p1_prices) if p1_prices else None
        if p1_odds:
            p = model_tennis_p1_prob(ev)
            edge = p - implied_prob_decimal(p1_odds)
            k = kelly_fraction(edge, p1_odds, cfg["max_kelly"])
            s = cfg["bankroll"] * max(k, 0)
            rows.append([hora, match, "Moneyline", match.split(" vs ")[0], round(p1_odds,2),
                         as_percent(edge), fmt_money(s), "✅ Consenso y forma", calidad_apuesta(edge)])
    cols = ["HORA","PARTIDO","MERCADO","APUESTA","CUOTA","EDGE","STAKE","RAZÓN","CALIDAD"]
    return pd.DataFrame(rows, columns=cols)

# ---------------- Mapeo WagerLab <-> TheSportsDB ----------------
def map_wagerlab_to_fixture_name(wager_item):
    # WagerLab estructuras comunes: {"sport":"nba","home":"LAL Lakers","away":"Boston Celtics","odds":{"home":1.80,"away":2.10}}
    home = norm_name(wager_item.get("home") or "")
    away = norm_name(wager_item.get("away") or "")
    return home, away

def map_thesportsdb_fixture(ev):
    # TheSportsDB: campos comunes "strEvent", "strHomeTeam", "strAwayTeam", "dateEvent", "strTime"
    home = norm_name(ev.get("strHomeTeam") or "")
    away = norm_name(ev.get("strAwayTeam") or "")
    hora = (ev.get("strTime") or "")
    match = f"{ev.get('strHomeTeam')} vs {ev.get('strAwayTeam')}"
    return home, away, hora, match

def join_fixtures_with_odds(fixtures, odds_list):
    # Devuelve lista de dicts con match, hora y cuotas home/away si hay match fuzzy
    out = []
    odds_index = {}
    for o in odds_list:
        h,a = map_wagerlab_to_fixture_name(o)
        if h and a:
            key = f"{h}|{a}"
            odds_index[key] = o
    for ev in fixtures:
        fh, fa, hora, match = map_thesportsdb_fixture(ev)
        key = f"{fh}|{fa}"
        o = odds_index.get(key)
        if o:
            # Extrae mejores precios disponibles
            odds_home = None
            odds_away = None
            if isinstance(o.get("odds"), dict):
                odds_home = o["odds"].get("home")
                odds_away = o["odds"].get("away")
            out.append({
                "hora": hora,
                "match": match,
                "home_team": ev.get("strHomeTeam"),
                "away_team": ev.get("strAwayTeam"),
                "odds_home": odds_home,
                "odds_away": odds_away
            })
    return out

# ---------------- Construcción DF desde fallback ----------------
def build_soccer_df_from_fallback(joined, cfg):
    rows = []
    for it in joined:
        hora = it["hora"] or ""
        match = it["match"]
        ml_home = it.get("odds_home")
        if ml_home and ml_home > 1.0:
            # Construimos un pseudo-evento para usar el mismo modelo
            ev = {
                "home_team": it["home_team"],
                "away_team": it["away_team"],
                "bookmakers": [
                    {"markets":[{"key":"h2h","outcomes":[
                        {"name":"Home","price":ml_home},
                        {"name":"Away","price":it.get("odds_away")}
                    ]}]}]
                }
            
            p = model_soccer_home_prob(ev)
            edge = p - implied_prob_decimal(ml_home)
            k = kelly_fraction(edge, ml_home, cfg["max_kelly"])
            s = cfg["bankroll"] * max(k, 0)
            rows.append([hora, match, "Moneyline", "Local", round(ml_home,2),
                         as_percent(edge), fmt_money(s), "✅ Fallback WagerLab + TheSportsDB", calidad_apuesta(edge)])
    cols = ["HORA","PARTIDO","MERCADO","APUESTA","CUOTA","EDGE","STAKE","RAZÓN","CALIDAD"]
    return pd.DataFrame(rows, columns=cols)

def build_nba_df_from_fallback(joined, cfg):
    rows = []
    for it in joined:
        hora = it["hora"] or ""
        match = it["match"]
        ml_home = it.get("odds_home")
        if ml_home and ml_home > 1.0:
            ev = {
                "home_team": it["home_team"],
                "away_team": it["away_team"],
                "bookmakers": [
                    {"markets":[{"key":"h2h","outcomes":[
                        {"name":"Home","price":ml_home},
                        {"name":"Away","price":it.get("odds_away")}
                    ]}]}]
                }
            
            p = model_nba_ml_home_prob(ev)
            edge = p - implied_prob_decimal(ml_home)
            k = kelly_fraction(edge, ml_home, cfg["max_kelly"])
            s = cfg["bankroll"] * max(k, 0)
            rows.append([hora, match, "Moneyline", "Local", round(ml_home,2),
                         as_percent(edge), fmt_money(s), "✅ Fallback WagerLab + TheSportsDB", calidad_apuesta(edge)])
    cols = ["HORA","PARTIDO","MERCADO","APUESTA","CUOTA","EDGE","STAKE","RAZÓN","CALIDAD"]
    return pd.DataFrame(rows, columns=cols)

def build_tennis_df_from_fallback(joined, cfg):
    rows = []
    for it in joined:
        hora = it["hora"] or ""
        match = it["match"]
        p1_odds = it.get("odds_home")
        if p1_odds and p1_odds > 1.0:
            ev = {
                "home_team": it["home_team"],
                "away_team": it["away_team"],
                "bookmakers": [
                    {"markets":[{"key":"h2h","outcomes":[
                        {"name":"Home","price":p1_odds},
                        {"name":"Away","price":it.get("odds_away")}
                    ]}]}]
                }
            
            p = model_tennis_p1_prob(ev)
            edge = p - implied_prob_decimal(p1_odds)
            k = kelly_fraction(edge, p1_odds, cfg["max_kelly"])
            s = cfg["bankroll"] * max(k, 0)
            rows.append([hora, match, "Moneyline", match.split(" vs ")[0], round(p1_odds,2),
                         as_percent(edge), fmt_money(s), "✅ Fallback WagerLab + TheSportsDB", calidad_apuesta(edge)])
    cols = ["HORA","PARTIDO","MERCADO","APUESTA","CUOTA","EDGE","STAKE","RAZÓN","CALIDAD"]
    return pd.DataFrame(rows, columns=cols)

# ---------------- App Streamlit ----------------
st.set_page_config(layout="wide", page_icon="🏆", page_title=f"BETTING PRO {APP_VERSION}")

if "config" not in st.session_state:
    st.session_state.config = load_config()
if "historial" not in st.session_state:
    st.session_state.historial = init_historial()
cfg = st.session_state.config
hist = st.session_state.historial

with st.sidebar:
    st.header("⚙️ Configuración")
    st.caption(f"BETTING PRO — {APP_VERSION}")
    cfg["bankroll"] = st.number_input("Bankroll base", 1000, 200000, int(cfg["bankroll"]), step=500)
    cfg["min_edge"] = st.slider("Edge mínimo (%)", 0, 20, int(cfg["min_edge"]*100)) / 100
    cfg["max_kelly"] = st.slider("Kelly máximo (%)", 5, 25, int(cfg["max_kelly"]*100)) / 100
    cfg["auto_cashout"] = st.slider("ROI Cash Out (%)", 10, 50, int(cfg["auto_cashout"]*100)) / 100

    st.markdown("---")
    st.subheader("🔑 The Odds API")
    cfg["ODDS_API_KEY"] = st.text_input("API key", value=cfg.get("ODDS_API_KEY",""))
    cfg["THE_ODDS_REGIONS"] = st.selectbox("Región", ["us","eu","uk","au"], index=["us","eu","uk","au"].index(cfg.get("THE_ODDS_REGIONS","eu")))
    cfg["THE_ODDS_MARKETS"] = st.text_input("Markets (coma)", value=cfg.get("THE_ODDS_MARKETS","h2h,totals,btts"))
    cfg["THE_ODDS_FORMAT"]  = st.selectbox("Formato de cuota", ["decimal","american"], index=["decimal","american"].index(cfg.get("THE_ODDS_FORMAT","decimal")))

    if st.button("💾 Guardar configuración", use_container_width=True):
        save_config(cfg); st.success("Configuración guardada.")

hoy, manana = get_local_dates()
hoy_str, manana_str = hoy.strftime("%Y-%m-%d"), manana.strftime("%Y-%m-%d")
st.markdown(f"<h2 style='text-align:center'>🏆 Pronósticos — {hoy.strftime('%d/%m/%Y')} y {manana.strftime('%d/%m/%Y')}</h2>", unsafe_allow_html=True)

total_bets = len(hist)
ganadas = len(hist[hist["status"]=="GANADO"]) if "status" in hist.columns else 0
winrate = ganadas/total_bets if total_bets>0 else 0
c1,c2,c3 = st.columns(3)
c1.metric("Bankroll base", fmt_money(cfg['bankroll']))
c2.metric("Winrate histórico", f"{winrate:.1%}")
c3.metric("Total apuestas", str(total_bets))

tab_home, tab_res, tab_rep, tab_val = st.tabs(["🏠 Home", "✅ Resultados", "📈 Reportes", "💎 Value Bets"])

# HOME
with tab_home:
    tab_hoy, tab_man = st.tabs([f"📅 HOY {hoy.strftime('%d/%m/%Y')}", f"📅 MAÑANA {manana.strftime('%d/%m/%Y')}"])

    # Config llaves de deportes para Odds API
    soccer_keys = [ODDS_SPORT_KEYS["UCL"], ODDS_SPORT_KEYS["EPL"], ODDS_SPORT_KEYS["LA_LIGA"],
                   ODDS_SPORT_KEYS["SERIE_A"], ODDS_SPORT_KEYS["BUNDESLIGA"]]

    # HOY
    with tab_hoy:
        # Fútbol
        st.markdown("### ⚽ Fútbol (Odds API / WagerLab / TheSportsDB)")
        fut_hoy = extract_soccer_events(cfg, hoy_str, soccer_keys)
        for liga, evs in fut_hoy.items():
            st.markdown(f"#### {liga}")
            debug_log_events(evs, f"{liga} HOY (Odds API)")
            if evs:
                df = build_soccer_df(evs, cfg)
                st.dataframe(df, hide_index=True, use_container_width=True)
            else:
                fixtures = thesportsdb_fixtures(hoy_str, "Soccer")
                wager_odds = wagerlab_odds("soccer")
                debug_log_generic(fixtures, f"{liga} HOY fixtures (TheSportsDB)")
                debug_log_generic(wager_odds, f"{liga} HOY odds (WagerLab)")
                joined = join_fixtures_with_odds(fixtures, wager_odds)
                df_fb = build_soccer_df_from_fallback(joined, cfg)
                if not df_fb.empty:
                    st.dataframe(df_fb, hide_index=True, use_container_width=True)
                else:
                    st.info("Fallback fútbol activo, pero no se pudo mapear fixtures con cuotas.")

        # NBA
        st.markdown("### 🏀 NBA (Odds API / WagerLab / TheSportsDB)")
        nba_hoy = extract_nba_events(cfg, hoy_str)
        debug_log_events(nba_hoy, "NBA HOY (Odds API)")
        if nba_hoy:
            df_nba = build_nba_df(nba_hoy, cfg)
            st.dataframe(df_nba, hide_index=True, use_container_width=True)
        else:
            nba_fixtures = thesportsdb_fixtures(hoy_str, "Basketball")
            nba_wager = wagerlab_odds("nba")
            debug_log_generic(nba_fixtures, "NBA HOY fixtures (TheSportsDB)")
            debug_log_generic(nba_wager, "NBA HOY odds (WagerLab)")
            joined_nba = join_fixtures_with_odds(nba_fixtures, nba_wager)
            df_nba_fb = build_nba_df_from_fallback(joined_nba, cfg)
            if not df_nba_fb.empty:
                st.dataframe(df_nba_fb, hide_index=True, use_container_width=True)
            else:
                st.info("Fallback NBA activo, pero no se pudo mapear fixtures con cuotas.")

        # Tenis
        st.markdown("### 🎾 Tenis (Odds API / WagerLab / TheSportsDB)")
        ten_hoy = extract_tennis_events(cfg, hoy_str)
        debug_log_events(ten_hoy, "Tenis HOY (Odds API)")
        if ten_hoy:
            df_ten = build_tennis_df(ten_hoy, cfg)
            st.dataframe(df_ten, hide_index=True, use_container_width=True)
        else:
            ten_fixtures = thesportsdb_fixtures(hoy_str, "Tennis")
            ten_wager = wagerlab_odds("tennis")
            debug_log_generic(ten_fixtures, "Tenis HOY fixtures (TheSportsDB)")
            debug_log_generic(ten_wager, "Tenis HOY odds (WagerLab)")
            joined_ten = join_fixtures_with_odds(ten_fixtures, ten_wager)
            df_ten_fb = build_tennis_df_from_fallback(joined_ten, cfg)
            if not df_ten_fb.empty:
                st.dataframe(df_ten_fb, hide_index=True, use_container_width=True)
            else:
                st.info("Fallback Tenis activo, pero no se pudo mapear fixtures con cuotas.")

    # MAÑANA
    with tab_man:
        # Fútbol
        st.markdown("### ⚽ Fútbol (Odds API / WagerLab / TheSportsDB)")
        fut_man = extract_soccer_events(cfg, manana_str, soccer_keys)
        for liga, evs in fut_man.items():
            st.markdown(f"#### {liga}")
            debug_log_events(evs, f"{liga} MAÑANA (Odds API)")
            if evs:
                df = build_soccer_df(evs, cfg)
                st.dataframe(df, hide_index=True, use_container_width=True)
            else:
                fixtures = thesportsdb_fixtures(manana_str, "Soccer")
                wager_odds = wagerlab_odds("soccer")
                debug_log_generic(fixtures, f"{liga} MAÑANA fixtures (TheSportsDB)")
                debug_log_generic(wager_odds, f"{liga} MAÑANA odds (WagerLab)")
                joined = join_fixtures_with_odds(fixtures, wager_odds)
                df_fb = build_soccer_df_from_fallback(joined, cfg)
                if not df_fb.empty:
                    st.dataframe(df_fb, hide_index=True, use_container_width=True)
                else:
                    st.info("Fallback fútbol activo (mañana), pero no se pudo mapear fixtures con cuotas.")

        # NBA
        st.markdown("### 🏀 NBA (Odds API / WagerLab / TheSportsDB)")
        nba_man = extract_nba_events(cfg, manana_str)
        debug_log_events(nba_man, "NBA MAÑANA (Odds API)")
        if nba_man:
            df_nba_m = build_nba_df(nba_man, cfg)
            st.dataframe(df_nba_m, hide_index=True, use_container_width=True)
        else:
            nba_fixtures_m = thesportsdb_fixtures(manana_str, "Basketball")
            nba_wager_m = wagerlab_odds("nba")
            debug_log_generic(nba_fixtures_m, "NBA MAÑANA fixtures (TheSportsDB)")
            debug_log_generic(nba_wager_m, "NBA MAÑANA odds (WagerLab)")
            joined_nba_m = join_fixtures_with_odds(nba_fixtures_m, nba_wager_m)
            df_nba_fb_m = build_nba_df_from_fallback(joined_nba_m, cfg)
            if not df_nba_fb_m.empty:
                st.dataframe(df_nba_fb_m, hide_index=True, use_container_width=True)
            else:
                st.info("Fallback NBA activo (mañana), pero no se pudo mapear fixtures con cuotas.")

        # Tenis
        st.markdown("### 🎾 Tenis (Odds API / WagerLab / TheSportsDB)")
        ten_man = extract_tennis_events(cfg, manana_str)
        debug_log_events(ten_man, "Tenis MAÑANA (Odds API)")
        if ten_man:
            df_ten_m = build_tennis_df(ten_man, cfg)
            st.dataframe(df_ten_m, hide_index=True, use_container_width=True)
        else:
            ten_fixtures_m = thesportsdb_fixtures(manana_str, "Tennis")
            ten_wager_m = wagerlab_odds("tennis")
            debug_log_generic(ten_fixtures_m, "Tenis MAÑANA fixtures (TheSportsDB)")
            debug_log_generic(ten_wager_m, "Tenis MAÑANA odds (WagerLab)")
            joined_ten_m = join_fixtures_with_odds(ten_fixtures_m, ten_wager_m)
            df_ten_fb_m = build_tennis_df_from_fallback(joined_ten_m, cfg)
            if not df_ten_fb_m.empty:
                st.dataframe(df_ten_fb_m, hide_index=True, use_container_width=True)
            else:
                st.info("Fallback Tenis activo (mañana), pero no se pudo mapear fixtures con cuotas.")

# RESULTADOS
with tab_res:
    st.subheader("✅ Marcar resultados")
    pendientes = hist[hist["status"]=="PENDIENTE"] if not hist.empty else pd.DataFrame()
    if pendientes.empty:
        st.success("No hay apuestas pendientes.")
    else:
        for idx,row in pendientes.iterrows():
            c1,c2,c3,c4 = st.columns([3,1,1,1])
            c1.markdown(f"**{row['deporte']}** – {row['partido']} ({row['mercado']} · {row['apuesta']})")
            c1.caption(f"Stake: {fmt_money(row.get('stake',0))} | Cuota: {float(row.get('cuota',1)):.2f}")
            if c2.button("GANÓ", key=f"w_{idx}"):
                hist.at[idx,"status"] = "GANADO"; hist.at[idx,"roi_real"] = float(row.get("cuota",1)) - 1
                save_historial(hist); st.session_state.historial = hist; st.rerun()
            if c3.button("PERDIÓ", key=f"l_{idx}"):
                hist.at[idx,"status"] = "PERDIDO"; hist.at[idx,"roi_real"] = -1.0
                save_historial(hist); st.session_state.historial = hist; st.rerun()
            if c4.button("CASH OUT", key=f"c_{idx}"):
                hist.at[idx,"status"] = "CASH_OUT"; hist.at[idx,"roi_real"] = cfg["auto_cashout"]
                save_historial(hist); st.session_state.historial = hist; st.rerun()

# REPORTES
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

# VALUE BETS (Odds API directo)
with tab_val:
    st.subheader("💎 Value Bets (The Odds API)")
    def value_bets_h2h(cfg, sport_key):
        data = odds_fetch_events(cfg, sport_key)
        rows = []
        for ev in data:
            match = f"{ev.get('home_team')} vs {ev.get('away_team')}"
            casas = {}
            for bk in ev.get("bookmakers", []):
                name = bk.get("title") or bk.get("key")
                for mk in bk.get("markets", []):
                    if mk.get("key") == "h2h":
                        for outc in mk.get("outcomes", []):
                            nm = (outc.get("name") or "").lower()
                            price = outc.get("price")
                            if price and "home" in nm:
                                casas[name] = float(price)
            if len(casas) >= 2:
                prices = list(casas.values())
                mejor = max(prices); promedio = sum(prices)/len(prices)
                edge_extra = (mejor - promedio)/promedio if promedio>0 else 0
                rows.append([match, len(casas), mejor, f"+{edge_extra*100:.1f}%"])
        return pd.DataFrame(rows, columns=["Partido","Casas consideradas","Mejor cuota home","Edge extra"])

    vb_nba = value_bets_h2h(cfg, ODDS_SPORT_KEYS["NBA"])
    vb_epl = value_bets_h2h(cfg, ODDS_SPORT_KEYS["EPL"])

    st.markdown("#### NBA")
    if not vb_nba.empty:
        st.dataframe(vb_nba, hide_index=True, use_container_width=True)
    else:
        st.info("No hay value bets detectadas en NBA.")
    st.markdown("#### Premier League")
    if not vb_epl.empty:
        st.dataframe(vb_epl, hide_index=True, use_container_width=True)
    else:
        st.info("No hay value bets detectadas en EPL.")
