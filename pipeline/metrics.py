"""
Calcula métricas de riesgo/retorno por activo a partir de la matriz de precios
del backfill, y genera el JSON final que consume el dashboard.

Convención: todos los retornos son diarios (log returns) y las métricas se
anualizan con factor sqrt(252) para volatilidad, y 252 para retorno.
Aunque el UPDATE de datos sea semanal, el HISTÓRICO descargado es diario
(así lo definimos en backfill.py), así que estas fórmulas son correctas.
"""
import json
import logging

import numpy as np
import pandas as pd

from config import PRICES_PARQUET, UNIVERSE_CSV, METRICS_JSON
from risk_free import obtener_risk_free_rate

logger = logging.getLogger(__name__)

DIAS_TRADING_ANIO = 252
BENCHMARK_TICKER = "SPY"  # usado para beta y R²


def calcular_retornos_diarios(precios: pd.DataFrame) -> pd.DataFrame:
    """Log returns. Se usan log returns (no simples) porque son aditivos en el
    tiempo y estadísticamente mejor comportados para anualizar."""
    return np.log(precios / precios.shift(1)).dropna(how="all")


def sharpe_ratio(retornos: pd.Series, rf_diario: float) -> float:
    exceso = retornos - rf_diario
    if exceso.std() == 0 or exceso.isna().all():
        return np.nan
    return (exceso.mean() / exceso.std()) * np.sqrt(DIAS_TRADING_ANIO)


def sortino_ratio(retornos: pd.Series, rf_diario: float) -> float:
    exceso = retornos - rf_diario
    downside = exceso[exceso < 0]
    if len(downside) == 0 or downside.std() == 0:
        return np.nan
    downside_dev = downside.std()
    return (exceso.mean() / downside_dev) * np.sqrt(DIAS_TRADING_ANIO)


def beta_y_r2(retornos_activo: pd.Series, retornos_benchmark: pd.Series) -> tuple[float, float]:
    """Beta vía regresión OLS simple: cov(activo,bench)/var(bench). R² = correlación²."""
    df = pd.concat([retornos_activo, retornos_benchmark], axis=1).dropna()
    df.columns = ["activo", "benchmark"]
    if len(df) < 30 or df["benchmark"].var() == 0:
        return np.nan, np.nan
    covarianza = df["activo"].cov(df["benchmark"])
    varianza_bench = df["benchmark"].var()
    beta = covarianza / varianza_bench
    r2 = df["activo"].corr(df["benchmark"]) ** 2
    return float(beta), float(r2)


def max_drawdown(precios: pd.Series) -> float:
    precios_validos = precios.dropna()
    if len(precios_validos) < 2:
        return np.nan
    pico_acumulado = precios_validos.cummax()
    drawdown = (precios_validos - pico_acumulado) / pico_acumulado
    return float(drawdown.min())


def var_cvar_historico(retornos: pd.Series, nivel_confianza: float = 0.95) -> tuple[float, float]:
    """VaR y CVaR (Expected Shortfall) históricos, no paramétricos."""
    retornos_validos = retornos.dropna()
    if len(retornos_validos) < 30:
        return np.nan, np.nan
    var = float(np.percentile(retornos_validos, (1 - nivel_confianza) * 100))
    cola = retornos_validos[retornos_validos <= var]
    cvar = float(cola.mean()) if len(cola) > 0 else var
    return var, cvar


def calmar_ratio(retorno_anualizado: float, mdd: float) -> float:
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    return retorno_anualizado / abs(mdd)


def treynor_ratio(retorno_anualizado: float, rf_anual: float, beta: float) -> float:
    if beta == 0 or np.isnan(beta):
        return np.nan
    return (retorno_anualizado - rf_anual) / beta


def jensen_alpha(retorno_anualizado: float, rf_anual: float, beta: float, retorno_benchmark_anual: float) -> float:
    if np.isnan(beta):
        return np.nan
    retorno_esperado_capm = rf_anual + beta * (retorno_benchmark_anual - rf_anual)
    return retorno_anualizado - retorno_esperado_capm


def calcular_metricas_activo(
    ticker: str,
    precios: pd.Series,
    retornos_diarios: pd.Series,
    retornos_benchmark: pd.Series,
    rf_diario: float,
    rf_anual: float,
    retorno_benchmark_anual: float,
) -> dict:
    retornos_validos = retornos_diarios.dropna()
    n_obs = len(retornos_validos)

    if n_obs < 30:
        # Historia insuficiente (IPO reciente, ticker con datos parciales):
        # se devuelve estructura completa con NaN en vez de omitir el activo,
        # así el dashboard sabe que existe pero no tiene métricas confiables aún.
        logger.warning(f"{ticker}: solo {n_obs} observaciones válidas, métricas insuficientes")
        return {
            "ticker": ticker, "n_observaciones": n_obs,
            "retorno_anualizado": None, "volatilidad_anualizada": None,
            "sharpe": None, "sortino": None, "beta": None, "r_squared": None,
            "max_drawdown": None, "var_95": None, "cvar_95": None,
            "calmar": None, "treynor": None, "jensen_alpha": None,
        }

    retorno_anualizado = float(retornos_validos.mean() * DIAS_TRADING_ANIO)
    volatilidad_anualizada = float(retornos_validos.std() * np.sqrt(DIAS_TRADING_ANIO))
    beta, r2 = beta_y_r2(retornos_diarios, retornos_benchmark)
    mdd = max_drawdown(precios)
    var_95, cvar_95 = var_cvar_historico(retornos_validos)

    return {
        "ticker": ticker,
        "n_observaciones": n_obs,
        "retorno_anualizado": round(retorno_anualizado, 6),
        "volatilidad_anualizada": round(volatilidad_anualizada, 6),
        "sharpe": round(sharpe_ratio(retornos_validos, rf_diario), 4) if not np.isnan(sharpe_ratio(retornos_validos, rf_diario)) else None,
        "sortino": round(sortino_ratio(retornos_validos, rf_diario), 4) if not np.isnan(sortino_ratio(retornos_validos, rf_diario)) else None,
        "beta": round(beta, 4) if not np.isnan(beta) else None,
        "r_squared": round(r2, 4) if not np.isnan(r2) else None,
        "max_drawdown": round(mdd, 6) if not np.isnan(mdd) else None,
        "var_95": round(var_95, 6) if not np.isnan(var_95) else None,
        "cvar_95": round(cvar_95, 6) if not np.isnan(cvar_95) else None,
        "calmar": round(calmar_ratio(retorno_anualizado, mdd), 4) if mdd and not np.isnan(calmar_ratio(retorno_anualizado, mdd)) else None,
        "treynor": round(treynor_ratio(retorno_anualizado, rf_anual, beta), 4) if not np.isnan(treynor_ratio(retorno_anualizado, rf_anual, beta)) else None,
        "jensen_alpha": round(jensen_alpha(retorno_anualizado, rf_anual, beta, retorno_benchmark_anual), 6) if not np.isnan(beta) else None,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logger.info("Cargando precios y universo...")
    precios = pd.read_parquet(PRICES_PARQUET)
    universo = pd.read_csv(UNIVERSE_CSV)

    if BENCHMARK_TICKER not in precios.columns:
        raise ValueError(
            f"'{BENCHMARK_TICKER}' no está en la matriz de precios — es requerido "
            f"como benchmark para beta/R²/Treynor/Jensen. Verificar que esté en el universo."
        )

    rf_anual, rf_fecha = obtener_risk_free_rate()
    rf_diario = rf_anual / DIAS_TRADING_ANIO

    retornos = calcular_retornos_diarios(precios)
    retornos_benchmark = retornos[BENCHMARK_TICKER]
    retorno_benchmark_anual = float(retornos_benchmark.dropna().mean() * DIAS_TRADING_ANIO)

    tipo_por_ticker = dict(zip(universo["ticker"], universo["tipo_activo"]))

    resultados = []
    for ticker in precios.columns:
        metricas = calcular_metricas_activo(
            ticker=ticker,
            precios=precios[ticker],
            retornos_diarios=retornos[ticker],
            retornos_benchmark=retornos_benchmark,
            rf_diario=rf_diario,
            rf_anual=rf_anual,
            retorno_benchmark_anual=retorno_benchmark_anual,
        )
        metricas["tipo_activo"] = tipo_por_ticker.get(ticker, "desconocido")
        resultados.append(metricas)

    salida = {
        "fecha_actualizacion": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "benchmark": BENCHMARK_TICKER,
        "risk_free_rate": round(rf_anual, 6),
        "risk_free_fecha": rf_fecha,
        "universo_total": len(resultados),
        "activos": resultados,
    }

    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_JSON, "w") as f:
        json.dump(salida, f, indent=2)

    logger.info(f"Métricas calculadas para {len(resultados)} activos -> {METRICS_JSON}")


if __name__ == "__main__":
    main()
