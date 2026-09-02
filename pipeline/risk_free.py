"""
Tasa libre de riesgo desde FRED (Federal Reserve Economic Data).

Usa el endpoint público de descarga CSV de FRED, que no requiere API key
(a diferencia del endpoint JSON completo de la API de FRED). Suficiente para
una serie simple como T-Bill 3 meses.
"""
import logging

import pandas as pd

from config import FRED_SERIE_RISK_FREE

logger = logging.getLogger(__name__)

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={serie}"


def obtener_risk_free_rate(serie: str = FRED_SERIE_RISK_FREE) -> tuple[float, str]:
    """
    Descarga la serie completa y devuelve el último valor no-nulo disponible,
    junto con su fecha. FRED reporta en porcentaje (ej. 4.21), se devuelve
    como decimal (0.0421) para uso directo en fórmulas financieras.

    Si la descarga falla, lanza la excepción hacia arriba explícitamente:
    la risk-free rate es un input crítico para TODAS las métricas
    (Sharpe, Sortino, Treynor, Jensen), no tiene sentido "degradar con gracia"
    acá con un valor inventado — mejor que la corrida falle visible.
    """
    url = FRED_CSV_URL.format(serie=serie)
    df = pd.read_csv(url)

    # FRED ha cambiado el nombre de la columna de fecha entre versiones de su
    # plataforma (se vio 'DATE' y también 'observation_date'). En vez de
    # hardcodear un nombre, se detecta dinámicamente: la columna de fecha es
    # la que NO es la serie de valores (que sí tiene nombre fijo = el id de la serie).
    if serie not in df.columns:
        raise ValueError(
            f"La columna '{serie}' no está en la respuesta de FRED. "
            f"Columnas recibidas: {list(df.columns)}"
        )
    columnas_fecha = [c for c in df.columns if c != serie]
    if len(columnas_fecha) != 1:
        raise ValueError(
            f"Se esperaba exactamente 1 columna de fecha además de '{serie}', "
            f"se encontraron: {columnas_fecha}"
        )
    columna_fecha = columnas_fecha[0]

    df = df.rename(columns={columna_fecha: "fecha", serie: "valor"})
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")  # FRED usa "." para missing
    df = df.dropna(subset=["valor"])

    if df.empty:
        raise ValueError(f"Serie FRED '{serie}' no devolvió ningún valor válido")

    ultima_fila = df.iloc[-1]
    tasa_decimal = float(ultima_fila["valor"]) / 100.0
    fecha = str(ultima_fila["fecha"])

    logger.info(f"Risk-free rate ({serie}): {tasa_decimal:.4%} al {fecha}")
    return tasa_decimal, fecha


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    tasa, fecha = obtener_risk_free_rate()
    print(f"Risk-free rate: {tasa:.4%} (dato al {fecha})")
