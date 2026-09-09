"""
Handler HTTP de la función serverless (Vercel).

Endpoint: POST /api/analyze
Body JSON esperado:
{
  "posiciones": [
    {"ticker": "AAPL", "peso": 0.4},
    {"ticker": "MSFT", "peso": null}   <- sin peso: se completa vía HRP
  ],
  "metodo_completar": "hrp" | "kelly"   <- solo si hay posiciones sin peso
}

Si TODAS las posiciones vienen con peso -> devuelve métricas del portfolio tal cual.
Si ALGUNA viene sin peso -> corre HRP/Kelly sobre el conjunto completo de tickers
pedidos (ignorando los pesos que sí vinieron) y devuelve los pesos sugeridos + métricas.

Nota de migración: esta es la misma lógica que corría en netlify/functions/analyze.py.
Netlify dejó de soportar funciones en Python, así que el handler HTTP se adaptó a la
convención de Vercel (BaseHTTPRequestHandler en vez de handler(event, context)).
_procesar_analisis() concentra toda la lógica de negocio y no cambió una línea.

Nota #2: portfolio_logic.py (HRP/Kelly/métricas) está fusionado acá abajo en vez de
importado como módulo separado. La convención de Vercel de "un .py en /api = una
función" no garantiza que los archivos hermanos queden en el sys.path del runtime
(fallaba con "ModuleNotFoundError: No module named 'portfolio_logic'" incluso estando
el archivo presente en la carpeta) -- un único archivo autocontenido lo evita del todo.
"""
import json
import os
from http.server import BaseHTTPRequestHandler

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

DIAS_TRADING_ANIO = 252


# ============================================================
# HRP (Hierarchical Risk Parity)
# ============================================================

def _distancia_correlacion(matriz_corr: pd.DataFrame) -> pd.DataFrame:
    """Distancia estándar de Lopez de Prado: sqrt(0.5 * (1 - corr))."""
    return np.sqrt(0.5 * (1 - matriz_corr))


def _cluster_var(cov: pd.DataFrame, items: list) -> float:
    """Varianza de un cluster, asumiendo pesos inverse-variance dentro de él."""
    sub_cov = cov.loc[items, items]
    ivp = 1.0 / np.diag(sub_cov)
    ivp /= ivp.sum()
    return float(ivp @ sub_cov @ ivp)


def _split_lista(items: list) -> tuple[list, list]:
    mitad = len(items) // 2
    return items[:mitad], items[mitad:]


def _recursive_bisection(cov: pd.DataFrame, orden: list) -> pd.Series:
    """Asignación recursiva de pesos, corazón del algoritmo HRP."""
    pesos = pd.Series(1.0, index=orden)
    clusters = [orden]

    while clusters:
        clusters = [c for c in clusters if len(c) > 1]
        if not clusters:
            break
        nuevos_clusters = []
        for cluster in clusters:
            izq, der = _split_lista(cluster)
            var_izq = _cluster_var(cov, izq)
            var_der = _cluster_var(cov, der)
            alpha = 1 - var_izq / (var_izq + var_der)
            pesos[izq] *= alpha
            pesos[der] *= (1 - alpha)
            nuevos_clusters.extend([izq, der])
        clusters = nuevos_clusters

    return pesos


def calcular_pesos_hrp(retornos: pd.DataFrame) -> pd.Series:
    """
    Punto de entrada HRP. retornos: DataFrame de retornos diarios,
    columnas = tickers. Devuelve Serie de pesos que suman 1.0.

    Requiere al menos 2 tickers con datos suficientes (>=30 obs) y sin
    varianza cero (activo constante).
    """
    retornos_validos = retornos.dropna(axis=0, how="any")
    if len(retornos_validos) < 30:
        raise ValueError(
            f"Solo {len(retornos_validos)} fechas con datos completos para todos los "
            f"tickers pedidos -- insuficiente para calcular covarianza confiable (mínimo 30)."
        )

    cov = retornos_validos.cov()
    corr = retornos_validos.corr()

    if cov.shape[0] < 2:
        raise ValueError("HRP requiere al menos 2 tickers")

    dist = _distancia_correlacion(corr)
    condensed = squareform(dist.values, checks=False)
    link = linkage(condensed, method="single")

    dend = dendrogram(link, no_plot=True, labels=list(cov.columns))
    orden = dend["ivl"]

    pesos = _recursive_bisection(cov, orden)
    pesos = pesos / pesos.sum()
    return pesos.reindex(retornos.columns)


# ============================================================
# Kelly fraccional
# ============================================================

def calcular_pesos_kelly(retornos: pd.DataFrame, rf_diario: float, fraccion: float = 0.5) -> pd.Series:
    """
    Kelly fraccional multi-activo: w* = fraccion * Sigma^-1 * (mu - rf)
    Se usa fracción (default 0.5 = half-Kelly) porque full Kelly es
    excesivamente volátil en la práctica -- estándar de la industria.

    Pesos negativos (posiciones short) se permiten en el cálculo crudo,
    pero se recortan a 0 y renormalizan si se pide long-only (default acá).
    """
    retornos_validos = retornos.dropna(axis=0, how="any")
    if len(retornos_validos) < 30:
        raise ValueError(f"Solo {len(retornos_validos)} fechas con datos completos -- insuficiente.")

    mu = retornos_validos.mean() * DIAS_TRADING_ANIO
    cov = retornos_validos.cov() * DIAS_TRADING_ANIO
    exceso = mu - rf_diario * DIAS_TRADING_ANIO

    try:
        cov_inv = np.linalg.pinv(cov.values)
    except np.linalg.LinAlgError:
        raise ValueError("Matriz de covarianza singular, no se puede invertir")

    w_crudo = fraccion * (cov_inv @ exceso.values)
    pesos = pd.Series(w_crudo, index=cov.columns)

    pesos = pesos.clip(lower=0)
    if pesos.sum() == 0:
        raise ValueError("Kelly no sugiere ninguna posición long-only positiva con estos activos")
    pesos = pesos / pesos.sum()
    return pesos.reindex(retornos.columns)


# ============================================================
# Métricas de portafolio combinado (dado un vector de pesos)
# ============================================================

def metricas_portfolio(
    retornos: pd.DataFrame,
    pesos: pd.Series,
    rf_diario: float,
    retornos_benchmark: pd.Series | None = None,
) -> dict:
    """Métricas del portafolio combinado con los pesos dados."""
    retornos_alineados = retornos[pesos.index]
    retorno_portfolio = (retornos_alineados * pesos).sum(axis=1)
    retorno_portfolio = retorno_portfolio.dropna()

    if len(retorno_portfolio) < 30:
        raise ValueError("Historia insuficiente del portafolio combinado (<30 obs)")

    retorno_anualizado = float(retorno_portfolio.mean() * DIAS_TRADING_ANIO)
    vol_anualizada = float(retorno_portfolio.std() * np.sqrt(DIAS_TRADING_ANIO))
    exceso = retorno_portfolio - rf_diario
    sharpe = float((exceso.mean() / exceso.std()) * np.sqrt(DIAS_TRADING_ANIO)) if exceso.std() > 0 else None

    downside = exceso[exceso < 0]
    sortino = float((exceso.mean() / downside.std()) * np.sqrt(DIAS_TRADING_ANIO)) if len(downside) > 0 and downside.std() > 0 else None

    var_95 = float(np.percentile(retorno_portfolio, 5))
    cola = retorno_portfolio[retorno_portfolio <= var_95]
    cvar_95 = float(cola.mean()) if len(cola) > 0 else var_95

    precio_acumulado = (1 + retorno_portfolio).cumprod()
    pico = precio_acumulado.cummax()
    mdd = float(((precio_acumulado - pico) / pico).min())

    resultado = {
        "retorno_anualizado": round(retorno_anualizado, 6),
        "volatilidad_anualizada": round(vol_anualizada, 6),
        "sharpe": round(sharpe, 4) if sharpe is not None else None,
        "sortino": round(sortino, 4) if sortino is not None else None,
        "var_95": round(var_95, 6),
        "cvar_95": round(cvar_95, 6),
        "max_drawdown": round(mdd, 6),
        "n_observaciones": len(retorno_portfolio),
    }

    if retornos_benchmark is not None:
        df = pd.concat([retorno_portfolio, retornos_benchmark], axis=1).dropna()
        if len(df) >= 30:
            df.columns = ["portfolio", "benchmark"]
            beta = float(df["portfolio"].cov(df["benchmark"]) / df["benchmark"].var())
            resultado["beta"] = round(beta, 4)

    return resultado


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PRICES_PATH = os.path.join(DATA_DIR, "precios_ajustados.parquet")
METRICS_PATH = os.path.join(DATA_DIR, "metricas.json")

BENCHMARK_TICKER = "SPY"


def _cargar_retornos_y_contexto():
    precios = pd.read_parquet(PRICES_PATH)
    retornos = np.log(precios / precios.shift(1)).dropna(how="all")

    with open(METRICS_PATH) as f:
        metricas_universo = json.load(f)
    rf_anual = metricas_universo["risk_free_rate"]
    rf_diario = rf_anual / 252

    return retornos, rf_diario


def _procesar_analisis(payload: dict) -> tuple[int, dict]:
    """Lógica de negocio pura: recibe el payload ya parseado, devuelve (status_code, body)."""
    posiciones = payload.get("posiciones", [])
    metodo_completar = payload.get("metodo_completar", "hrp")

    if not posiciones:
        return 400, {"error": "Se requiere al menos 1 posición en 'posiciones'"}

    tickers_pedidos = [p["ticker"].upper() for p in posiciones]
    tiene_peso_faltante = any(p.get("peso") is None for p in posiciones)

    try:
        retornos, rf_diario = _cargar_retornos_y_contexto()
    except FileNotFoundError as e:
        return 500, {"error": f"Datos del pipeline no encontrados: {e}"}

    tickers_no_encontrados = [t for t in tickers_pedidos if t not in retornos.columns]
    if tickers_no_encontrados:
        return 400, {
            "error": f"Tickers no encontrados en el universo: {tickers_no_encontrados}. "
                     f"Verificá que estén dentro del universo cubierto."
        }

    retornos_subset = retornos[tickers_pedidos]
    retornos_benchmark = retornos[BENCHMARK_TICKER] if BENCHMARK_TICKER in retornos.columns else None

    try:
        if tiene_peso_faltante:
            if metodo_completar == "kelly":
                pesos = calcular_pesos_kelly(retornos_subset, rf_diario)
            else:
                pesos = calcular_pesos_hrp(retornos_subset)
            metodo_usado = metodo_completar
        else:
            pesos_dict = {p["ticker"].upper(): p["peso"] for p in posiciones}
            suma = sum(pesos_dict.values())
            if abs(suma - 1.0) > 0.02:
                return 400, {
                    "error": f"Los pesos suman {suma:.4f}, deben sumar 1.0 (100%). "
                             f"Ajustá las posiciones o dejá 'peso': null para que se autocomplete."
                }
            pesos = pd.Series(pesos_dict)
            metodo_usado = "pesos_provistos_por_usuario"

        metricas = metricas_portfolio(retornos_subset, pesos, rf_diario, retornos_benchmark)

    except ValueError as e:
        return 400, {"error": str(e)}

    return 200, {
        "metodo": metodo_usado,
        "pesos": {t: round(float(w), 6) for t, w in pesos.items()},
        "metricas_portfolio": metricas,
    }


def _cors_headers(handler: BaseHTTPRequestHandler) -> None:
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        _cors_headers(self)
        self.end_headers()

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0) or 0)
        raw_body = self.rfile.read(content_length) if content_length else b"{}"

        try:
            payload = json.loads(raw_body or b"{}")
        except json.JSONDecodeError:
            status_code, body = 400, {"error": "Body no es JSON válido"}
        else:
            status_code, body = _procesar_analisis(payload)

        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        _cors_headers(self)
        self.end_headers()
        self.wfile.write(json.dumps(body).encode("utf-8"))
