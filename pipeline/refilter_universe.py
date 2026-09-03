"""
Filtra el universo completo (ya descargado en precios_ajustados.parquet /
volumen.parquet) al top N por liquidez, usando los datos que YA TENEMOS.

Reemplaza el enfoque anterior de universe.py, que volvía a pegarle a Yahoo
Finance solo para calcular liquidez -- innecesario si ya bajamos el histórico
completo una vez. Esto corre en segundos, sin red, sin riesgo de bloqueo.

Uso: correr DESPUÉS de tener precios_ajustados.parquet y volumen.parquet ya
poblados (o sea, después del primer backfill completo, como ya hiciste).
"""
import logging

import pandas as pd

from config import (
    N_ACCIONES_LIQUIDAS, N_ETFS_LIQUIDOS, N_MESES_LOOKBACK_LIQUIDEZ,
    PATRONES_ETF_EXCLUIR, UNIVERSO_BONOS_ETF,
    PRICES_PARQUET, VOLUME_PARQUET, UNIVERSE_CSV,
)
from universe import obtener_listado_nasdaq_nyse, es_etf_apalancado_o_inverso

logger = logging.getLogger(__name__)


def calcular_dollar_volume_desde_parquet(precios: pd.DataFrame, volumen: pd.DataFrame) -> pd.Series:
    """
    Dollar-volume promedio de los últimos N_MESES_LOOKBACK_LIQUIDEZ, calculado
    directo de los datos ya descargados. ~21 días hábiles por mes como
    aproximación estándar de mercado.
    """
    dias_lookback = N_MESES_LOOKBACK_LIQUIDEZ * 21
    precios_recientes = precios.tail(dias_lookback)
    volumen_reciente = volumen.tail(dias_lookback)

    tickers_comunes = precios_recientes.columns.intersection(volumen_reciente.columns)
    dollar_volume = (precios_recientes[tickers_comunes] * volumen_reciente[tickers_comunes]).mean()
    return dollar_volume.fillna(0.0)


def refiltrar_universo():
    logger.info("Cargando parquets ya existentes (sin descargar nada nuevo)...")
    precios = pd.read_parquet(PRICES_PARQUET)
    volumen = pd.read_parquet(VOLUME_PARQUET)

    dollar_volume = calcular_dollar_volume_desde_parquet(precios, volumen)
    logger.info(f"Dollar-volume calculado para {len(dollar_volume)} tickers desde datos locales")

    # Necesitamos saber cuáles son ETF vs acción vs bond_etf -> el listado de
    # nasdaqtrader.com es liviano (nombres/flags, no precios), así que esta sí
    # es una descarga chica y rápida, no un problema.
    listado = obtener_listado_nasdaq_nyse()
    listado = listado.set_index("symbol")

    dv_df = dollar_volume.rename("dollar_volume_promedio").to_frame()
    dv_df = dv_df.join(listado[["es_etf", "security_name"]], how="left")
    dv_df["es_etf"] = dv_df["es_etf"].fillna(False)  # tickers no encontrados en listado (ej. bond ETFs ya conocidos)

    # Bond ETFs del universo fijo se excluyen del ranking, van aparte
    dv_df = dv_df[~dv_df.index.isin(UNIVERSO_BONOS_ETF)]

    acciones = dv_df[~dv_df["es_etf"]]
    etfs = dv_df[dv_df["es_etf"]]
    etfs = etfs[~etfs["security_name"].apply(es_etf_apalancado_o_inverso)]

    top_acciones = acciones.nlargest(N_ACCIONES_LIQUIDAS, "dollar_volume_promedio")
    top_etfs = etfs.nlargest(N_ETFS_LIQUIDOS, "dollar_volume_promedio")
    top_etfs = top_etfs[top_etfs["dollar_volume_promedio"] > 0]

    filas = []
    for ticker, row in top_acciones.iterrows():
        filas.append({"ticker": ticker, "tipo_activo": "equity",
                       "dollar_volume_promedio": row["dollar_volume_promedio"]})
    for ticker, row in top_etfs.iterrows():
        filas.append({"ticker": ticker, "tipo_activo": "etf",
                       "dollar_volume_promedio": row["dollar_volume_promedio"]})
    for ticker in UNIVERSO_BONOS_ETF:
        filas.append({"ticker": ticker, "tipo_activo": "bond_etf",
                       "dollar_volume_promedio": None})

    universo_final = pd.DataFrame(filas).drop_duplicates(subset="ticker").reset_index(drop=True)
    logger.info(f"Universo filtrado: {len(universo_final)} tickers "
                f"({(universo_final['tipo_activo'] == 'equity').sum()} acciones, "
                f"{(universo_final['tipo_activo'] == 'etf').sum()} ETFs, "
                f"{(universo_final['tipo_activo'] == 'bond_etf').sum()} bond ETFs)")

    # Recortar los parquets también, para que el archivo final sea liviano
    tickers_finales = universo_final["ticker"].tolist()
    tickers_presentes = [t for t in tickers_finales if t in precios.columns]
    faltantes = set(tickers_finales) - set(tickers_presentes)
    if faltantes:
        logger.warning(f"{len(faltantes)} tickers del universo filtrado no están en el parquet "
                       f"(no se bajaron en el backfill original): {list(faltantes)[:10]}...")

    precios_recortado = precios[tickers_presentes]
    volumen_recortado = volumen[tickers_presentes]

    return universo_final, precios_recortado, volumen_recortado


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    universo_final, precios_recortado, volumen_recortado = refiltrar_universo()

    universo_final.to_csv(UNIVERSE_CSV, index=False)
    precios_recortado.to_parquet(PRICES_PARQUET)
    volumen_recortado.to_parquet(VOLUME_PARQUET)

    logger.info(f"Guardado: universo.csv ({len(universo_final)} filas), "
                f"parquets recortados a {precios_recortado.shape[1]} columnas")


if __name__ == "__main__":
    main()
