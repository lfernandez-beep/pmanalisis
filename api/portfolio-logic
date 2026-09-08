"""
Lógica de portfolio construction: HRP, Kelly, métricas de portafolio combinado.

Separado del handler HTTP (handler.py) a propósito: esto es lógica pura,
testeable con datos sintéticos sin necesitar simular una request HTTP real.
"""
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
        # Solo dividir clusters con más de 1 elemento
        clusters = [c for c in clusters if len(c) > 1]
        if not clusters:
            break
        nuevos_clusters = []
        for cluster in clusters:
            izq, der = _split_lista(cluster)
            var_izq = _cluster_var(cov, izq)
            var_der = _cluster_var(cov, der)
            alpha = 1 - var_izq / (var_izq + var_der)  # más peso al cluster de MENOR varianza
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
    retornos_validos = retornos.dropna(axis=0, how="any")  # fechas donde TODOS tienen dato
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

    # Orden de hoja del dendrograma (quasi-diagonalización)
    dend = dendrogram(link, no_plot=True, labels=list(cov.columns))
    orden = dend["ivl"]

    pesos = _recursive_bisection(cov, orden)
    pesos = pesos / pesos.sum()  # normalizar por las dudas
    return pesos.reindex(retornos.columns)  # mismo orden que input


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
        cov_inv = np.linalg.pinv(cov.values)  # pseudo-inversa: más robusta que inv() si casi singular
    except np.linalg.LinAlgError:
        raise ValueError("Matriz de covarianza singular, no se puede invertir")

    w_crudo = fraccion * (cov_inv @ exceso.values)
    pesos = pd.Series(w_crudo, index=cov.columns)

    # Long-only: recortar negativos y renormalizar
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
