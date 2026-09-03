"""
Construcción del universo de tickers.

Responsabilidad única de este módulo: decidir QUÉ tickers entran al análisis.
No descarga histórico profundo acá (eso es tarea de backfill.py) — solo lo
mínimo necesario (volumen reciente) para poder rankear por liquidez.
"""
import logging
import time
from dataclasses import dataclass

import pandas as pd
import yfinance as yf

from config import (
    N_ACCIONES_LIQUIDAS,
    N_ETFS_LIQUIDOS,
    N_MESES_LOOKBACK_LIQUIDEZ,
    PATRONES_ETF_EXCLUIR,
    UNIVERSO_BONOS_ETF,
)

logger = logging.getLogger(__name__)


@dataclass
class ActivoUniverso:
    ticker: str
    tipo_activo: str  # "equity" | "etf" | "bond_etf"
    dollar_volume_promedio: float = 0.0


def obtener_listado_nasdaq_nyse() -> pd.DataFrame:
    """
    Descarga el listado completo de símbolos de NASDAQ + NYSE/AMEX desde los
    archivos públicos que publica Nasdaq Trader (nasdaqtrader.com/trader.aspx?id=symbollookup).

    Retorna DataFrame con columnas: symbol, security_name, es_etf (bool).
    """
    url_nasdaq = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    url_other = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

    df_nasdaq = pd.read_csv(url_nasdaq, sep="|")
    df_nasdaq = df_nasdaq[df_nasdaq["Test Issue"] == "N"]
    df_nasdaq = df_nasdaq.rename(columns={"Symbol": "symbol", "Security Name": "security_name"})
    df_nasdaq["es_etf"] = df_nasdaq["ETF"] == "Y"

    df_other = pd.read_csv(url_other, sep="|")
    df_other = df_other[df_other["Test Issue"] == "N"]
    df_other = df_other.rename(columns={"ACT Symbol": "symbol", "Security Name": "security_name"})
    df_other["es_etf"] = df_other["ETF"] == "Y"

    cols = ["symbol", "security_name", "es_etf"]
    listado = pd.concat([df_nasdaq[cols], df_other[cols]], ignore_index=True)
    listado = listado.dropna(subset=["symbol"])
    listado = listado[~listado["symbol"].str.contains(r"[\.\$\^]", regex=True, na=False)]
    listado = listado.drop_duplicates(subset="symbol")

    logger.info(f"Listado completo obtenido: {len(listado)} símbolos "
                f"({listado['es_etf'].sum()} ETFs, {(~listado['es_etf']).sum()} acciones)")
    return listado.reset_index(drop=True)


def es_etf_apalancado_o_inverso(nombre_seguridad: str) -> bool:
    """Filtra por patrones en el nombre. Case-insensitive."""
    nombre_upper = str(nombre_seguridad).upper()
    return any(patron in nombre_upper for patron in PATRONES_ETF_EXCLUIR)


def calcular_dollar_volume_promedio(tickers: list[str], lote_size: int = 50) -> dict[str, float]:
    """
    Descarga N meses de histórico liviano (solo para rankear liquidez, no es el
    backfill completo) y calcula dollar-volume promedio = precio_cierre * volumen.

    Retorna dict ticker -> dollar_volume_promedio. Tickers que fallan quedan
    en 0.0 (se descartan naturalmente al rankear) en vez de romper la corrida.

    Usa reintento + backoff exponencial por lote (igual que backfill.py) porque
    Yahoo Finance bloquea temporalmente (HTTP 401 "Invalid Crumb") cuando detecta
    volumen alto de requests en poco tiempo desde IPs de datacenter (GitHub Actions,
    Colab). Sin esto, un bloqueo temprano tira 0.0 para TODOS los tickers restantes.
    """
    resultados: dict[str, float] = {}
    periodo = f"{N_MESES_LOOKBACK_LIQUIDEZ}mo"
    max_reintentos = 4
    backoff_base_segundos = 5

    for i in range(0, len(tickers), lote_size):
        lote = tickers[i:i + lote_size]
        datos = None

        for intento in range(max_reintentos):
            try:
                datos = yf.download(
                    lote, period=periodo, group_by="ticker",
                    auto_adjust=True, progress=False, threads=True,
                )
                if datos is not None and not datos.empty:
                    break
            except Exception as e:
                espera = backoff_base_segundos * (2 ** intento)
                logger.warning(
                    f"Intento {intento + 1}/{max_reintentos} falló para lote de liquidez "
                    f"({lote[0]}...{lote[-1]}): {e}. Reintentando en {espera}s"
                )
                time.sleep(espera)
                datos = None

        if datos is None or datos.empty:
            logger.warning(f"Lote de liquidez definitivamente falló tras {max_reintentos} intentos: "
                           f"{lote[0]}...{lote[-1]}. Se asigna 0.0, no rompe la corrida.")
            for t in lote:
                resultados[t] = 0.0
            time.sleep(backoff_base_segundos)  # pausa extra tras un fallo, por las dudas
            continue

        for t in lote:
            try:
                if len(lote) == 1:
                    serie_close = datos["Close"]
                    serie_vol = datos["Volume"]
                else:
                    serie_close = datos[t]["Close"]
                    serie_vol = datos[t]["Volume"]
                dv = (serie_close * serie_vol).mean()
                resultados[t] = float(dv) if pd.notna(dv) else 0.0
            except (KeyError, Exception):
                resultados[t] = 0.0

        time.sleep(2)  # cortesía entre lotes, evita rate limiting

    return resultados


def construir_universo() -> pd.DataFrame:
    """
    Punto de entrada principal. Devuelve DataFrame final del universo:
    columnas [ticker, tipo_activo, dollar_volume_promedio]
    """
    listado = obtener_listado_nasdaq_nyse()

    acciones = listado[~listado["es_etf"]].copy()
    etfs = listado[listado["es_etf"]].copy()
    etfs = etfs[~etfs["security_name"].apply(es_etf_apalancado_o_inverso)]

    logger.info(f"Calculando liquidez para {len(acciones)} acciones candidatas...")
    dv_acciones = calcular_dollar_volume_promedio(acciones["symbol"].tolist())
    acciones["dollar_volume_promedio"] = acciones["symbol"].map(dv_acciones).fillna(0.0)
    top_acciones = acciones.nlargest(N_ACCIONES_LIQUIDAS, "dollar_volume_promedio")

    logger.info(f"Calculando liquidez para {len(etfs)} ETFs candidatos (sin apalancados)...")
    dv_etfs = calcular_dollar_volume_promedio(etfs["symbol"].tolist())
    etfs["dollar_volume_promedio"] = etfs["symbol"].map(dv_etfs).fillna(0.0)
    etfs_liquidos = etfs.nlargest(N_ETFS_LIQUIDOS, "dollar_volume_promedio")
    etfs_liquidos = etfs_liquidos[etfs_liquidos["dollar_volume_promedio"] > 0]

    filas = []
    for _, row in top_acciones.iterrows():
        filas.append({"ticker": row["symbol"], "tipo_activo": "equity",
                       "dollar_volume_promedio": row["dollar_volume_promedio"]})
    for _, row in etfs_liquidos.iterrows():
        filas.append({"ticker": row["symbol"], "tipo_activo": "etf",
                       "dollar_volume_promedio": row["dollar_volume_promedio"]})
    for ticker in UNIVERSO_BONOS_ETF:
        filas.append({"ticker": ticker, "tipo_activo": "bond_etf",
                       "dollar_volume_promedio": None})

    universo = pd.DataFrame(filas).drop_duplicates(subset="ticker").reset_index(drop=True)
    logger.info(f"Universo final: {len(universo)} tickers "
                f"({(universo['tipo_activo'] == 'equity').sum()} acciones, "
                f"{(universo['tipo_activo'] == 'etf').sum()} ETFs, "
                f"{(universo['tipo_activo'] == 'bond_etf').sum()} bond ETFs)")
    return universo


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = construir_universo()
    from config import UNIVERSE_CSV
    UNIVERSE_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(UNIVERSE_CSV, index=False)
    print(f"Universo guardado en {UNIVERSE_CSV}")
