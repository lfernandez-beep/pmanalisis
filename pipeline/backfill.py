"""
Backfill histórico de precios (una sola vez, o para re-armar la base desde cero).

Descarga BACKFILL_YEARS de historia diaria para todo el universo, en lotes,
con reintento + backoff exponencial por lote. Un lote que falla definitivamente
después de MAX_REINTENTOS no rompe la corrida completa: se registra y se sigue.

Guarda dos matrices anchas (filas=fecha, columnas=ticker):
  - precios_ajustados.parquet  (Close ajustado por dividendos/splits)
  - volumen.parquet
"""
import logging
import time

import pandas as pd
import yfinance as yf

from config import (
    BACKFILL_YEARS, BATCH_SIZE, MAX_REINTENTOS, BACKOFF_BASE_SEGUNDOS,
    PAUSA_ENTRE_LOTES_SEGUNDOS, PRICES_PARQUET, VOLUME_PARQUET, UNIVERSE_CSV,
)

logger = logging.getLogger(__name__)


def descargar_lote_con_reintento(tickers: list[str], period: str) -> pd.DataFrame | None:
    """
    Descarga un lote de tickers con reintento y backoff exponencial.
    Retorna None si falla después de todos los reintentos (no lanza excepción,
    el caller decide cómo degradar).
    """
    for intento in range(MAX_REINTENTOS):
        try:
            datos = yf.download(
                tickers, period=period, auto_adjust=True,
                group_by="ticker", progress=False, threads=True,
            )
            if datos is None or datos.empty:
                raise ValueError("Descarga vacía")
            return datos
        except Exception as e:
            espera = BACKOFF_BASE_SEGUNDOS * (2 ** intento)
            logger.warning(
                f"Intento {intento + 1}/{MAX_REINTENTOS} falló para lote "
                f"({tickers[0]}...{tickers[-1]}): {e}. Reintentando en {espera}s"
            )
            time.sleep(espera)
    logger.error(f"Lote definitivamente falló tras {MAX_REINTENTOS} intentos: "
                 f"{tickers[0]}...{tickers[-1]}. Se omite, no rompe la corrida.")
    return None


def extraer_close_volume(datos_lote: pd.DataFrame, tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    yfinance devuelve estructura distinta según sea 1 o N tickers.
    Normaliza a dos DataFrames simples: close_df, volume_df (columnas = tickers).
    """
    close_cols, vol_cols = {}, {}
    es_multiindex = isinstance(datos_lote.columns, pd.MultiIndex)

    if not es_multiindex:
        # Solo ocurre si se pidió un único ticker como string (no lista).
        t = tickers[0]
        close_cols[t] = datos_lote["Close"]
        vol_cols[t] = datos_lote["Volume"]
    else:
        for t in tickers:
            try:
                close_cols[t] = datos_lote[t]["Close"]
                vol_cols[t] = datos_lote[t]["Volume"]
            except KeyError:
                continue  # ticker individual sin datos dentro de un lote exitoso

    close_df = pd.DataFrame(close_cols)
    vol_df = pd.DataFrame(vol_cols)
    return close_df, vol_df


def ejecutar_backfill(tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Orquesta la descarga completa en lotes. Devuelve (precios, volumen) como
    matrices anchas concatenadas de todos los lotes exitosos.
    """
    period = f"{BACKFILL_YEARS}y"
    todos_close, todos_vol = [], []
    lotes_fallidos = []

    n_lotes = (len(tickers) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(tickers), BATCH_SIZE):
        lote = tickers[i:i + BATCH_SIZE]
        n_actual = i // BATCH_SIZE + 1
        logger.info(f"Lote {n_actual}/{n_lotes} ({len(lote)} tickers)...")

        datos = descargar_lote_con_reintento(lote, period)
        if datos is None:
            lotes_fallidos.extend(lote)
            continue

        close_df, vol_df = extraer_close_volume(datos, lote)
        todos_close.append(close_df)
        todos_vol.append(vol_df)
        time.sleep(PAUSA_ENTRE_LOTES_SEGUNDOS)

    if lotes_fallidos:
        logger.warning(f"{len(lotes_fallidos)} tickers no se pudieron descargar: {lotes_fallidos[:20]}...")

    precios = pd.concat(todos_close, axis=1) if todos_close else pd.DataFrame()
    volumen = pd.concat(todos_vol, axis=1) if todos_vol else pd.DataFrame()
    precios = precios.sort_index()
    volumen = volumen.sort_index()
    return precios, volumen


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    universo = pd.read_csv(UNIVERSE_CSV)
    tickers = universo["ticker"].tolist()
    logger.info(f"Iniciando backfill de {len(tickers)} tickers, {BACKFILL_YEARS} años de historia")

    precios, volumen = ejecutar_backfill(tickers)

    PRICES_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    precios.to_parquet(PRICES_PARQUET)
    volumen.to_parquet(VOLUME_PARQUET)

    logger.info(f"Backfill completo: {precios.shape[0]} fechas x {precios.shape[1]} tickers")
    logger.info(f"Guardado en {PRICES_PARQUET} y {VOLUME_PARQUET}")


if __name__ == "__main__":
    main()
