"""
Configuración central del pipeline de portfolio.
Todo número/ruta "mágico" vive acá, no dentro de la lógica.
"""
from pathlib import Path

# --- Rutas ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

PRICES_PARQUET = PROCESSED_DIR / "precios_ajustados.parquet"
VOLUME_PARQUET = PROCESSED_DIR / "volumen.parquet"
UNIVERSE_CSV = PROCESSED_DIR / "universo.csv"
METRICS_JSON = PROCESSED_DIR / "metricas.json"

# --- Universo ---
N_ACCIONES_LIQUIDAS = 1500          # top N acciones por dollar-volume (dentro del rango 1000-2000 acordado)
N_MESES_LOOKBACK_LIQUIDEZ = 6        # ventana para calcular dollar-volume promedio y rankear
PATRONES_ETF_EXCLUIR = [             # apalancados / inversos, se excluyen por nombre
    "2X", "3X", "-1X", "ULTRA", "INVERSE", "BEAR", "BULL",
    "LEVERAGED", "DAILY 2X", "DAILY 3X",
]
EMISORES_APALANCADOS = ["Direxion", "ProShares Ultra", "ProShares Short"]

# --- Backfill histórico ---
BACKFILL_YEARS = 12
BATCH_SIZE = 75                       # tickers por request a yfinance
MAX_REINTENTOS = 4
BACKOFF_BASE_SEGUNDOS = 3             # backoff exponencial: base * (2 ** intento)
PAUSA_ENTRE_LOTES_SEGUNDOS = 2

# --- Update incremental (semanal) ---
DIAS_MARGEN_UPDATE = 10               # se re-descargan últimos N días por ajustes retroactivos de dividendos

# --- Risk-free rate (FRED) ---
FRED_SERIE_RISK_FREE = "DGS3MO"       # T-Bill 3 meses

# --- Bonos como ETFs (proxy, sin apalancamiento) ---
UNIVERSO_BONOS_ETF = [
    "IEF",   # Treasury 7-10Y (proxy del 10Y pedido)
    "SHY",   # Treasury 1-3Y (corto plazo)
    "TLT",   # Treasury 20Y+ (largo plazo)
    "LQD",   # Corporate Investment Grade
    "HYG",   # High Yield Corporate
    "AGG",   # Aggregate bond (broad market)
    "TIP",   # TIPS (inflation-protected)
]
