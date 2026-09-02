"""
Update incremental (semanal). A diferencia de backfill.py, NO re-descarga
todo el histórico: solo los últimos DIAS_MARGEN_UPDATE días, y hace upsert
sobre la matriz parquet existente (los ajustes retroactivos por dividendos
recientes pisan los valores viejos de esas fechas; el resto del histórico
queda intacto).
"""
import logging

import pandas as pd

from config import DIAS_MARGEN_UPDATE, PRICES_PARQUET, VOLUME_PARQUET, UNIVERSE_CSV
from backfill import ejecutar_backfill

logger = logging.getLogger(__name__)


def hacer_upsert(matriz_existente: pd.DataFrame, matriz_nueva: pd.DataFrame) -> pd.DataFrame:
    """
    Combina la matriz vieja con la nueva. Para fechas que se solapan, gana el
    valor NUEVO (puede venir corregido por un split/dividendo declarado
    después de la corrida anterior). Para tickers nuevos que no estaban en
    la matriz vieja, se agregan como columna nueva.
    """
    combinada = matriz_nueva.combine_first(matriz_existente)
    # combine_first prioriza matriz_nueva en los índices que se solapan,
    # y rellena con matriz_existente donde matriz_nueva no tiene dato.
    # Para las fechas específicas que SÍ están en ambas, forzamos que gane la nueva:
    fechas_solapadas = matriz_existente.index.intersection(matriz_nueva.index)
    tickers_comunes = matriz_existente.columns.intersection(matriz_nueva.columns)
    combinada.loc[fechas_solapadas, tickers_comunes] = matriz_nueva.loc[fechas_solapadas, tickers_comunes]
    return combinada.sort_index()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    universo = pd.read_csv(UNIVERSE_CSV)
    tickers = universo["ticker"].tolist()

    logger.info(f"Update incremental: últimos {DIAS_MARGEN_UPDATE} días para {len(tickers)} tickers")

    # Reusa la misma lógica de descarga en lotes que el backfill, pero pidiendo
    # solo el período corto (se pasa period directamente vía monkeypatch del config
    # no es necesario: ejecutar_backfill ya usa BACKFILL_YEARS de config.py).
    # Para el update usamos un period distinto, así que llamamos la función interna
    # con el period correcto en vez de reusar ejecutar_backfill tal cual:
    from backfill import descargar_lote_con_reintento, extraer_close_volume
    from config import BATCH_SIZE, PAUSA_ENTRE_LOTES_SEGUNDOS
    import time

    period = f"{DIAS_MARGEN_UPDATE}d"
    todos_close, todos_vol = [], []

    n_lotes = (len(tickers) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(tickers), BATCH_SIZE):
        lote = tickers[i:i + BATCH_SIZE]
        n_actual = i // BATCH_SIZE + 1
        logger.info(f"Lote {n_actual}/{n_lotes} ({len(lote)} tickers)...")

        datos = descargar_lote_con_reintento(lote, period)
        if datos is None:
            continue

        close_df, vol_df = extraer_close_volume(datos, lote)
        todos_close.append(close_df)
        todos_vol.append(vol_df)
        time.sleep(PAUSA_ENTRE_LOTES_SEGUNDOS)

    precios_nuevos = pd.concat(todos_close, axis=1).sort_index() if todos_close else pd.DataFrame()
    volumen_nuevo = pd.concat(todos_vol, axis=1).sort_index() if todos_vol else pd.DataFrame()

    if precios_nuevos.empty:
        raise RuntimeError("Update incremental no trajo ningún dato nuevo — abortando sin tocar el parquet existente")

    precios_existentes = pd.read_parquet(PRICES_PARQUET)
    volumen_existente = pd.read_parquet(VOLUME_PARQUET)

    precios_final = hacer_upsert(precios_existentes, precios_nuevos)
    volumen_final = hacer_upsert(volumen_existente, volumen_nuevo)

    precios_final.to_parquet(PRICES_PARQUET)
    volumen_final.to_parquet(VOLUME_PARQUET)

    logger.info(f"Update completo: matriz final {precios_final.shape[0]} fechas x {precios_final.shape[1]} tickers")


if __name__ == "__main__":
    main()
