"""
Tests con datos SINTÉTICOS que imitan la forma exacta en que yfinance devuelve
datos (MultiIndex de columnas para N tickers, columnas simples para 1 ticker).

Objetivo: validar la lógica de extraer_close_volume() y ejecutar_backfill()
sin depender de red. Esto es lo que evita bugs de lógica silenciosos.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

import numpy as np
import pandas as pd
from backfill import extraer_close_volume


def _generar_datos_multi_ticker(tickers: list[str], n_dias: int = 30, seed: int = 42) -> pd.DataFrame:
    """Simula exactamente el formato yf.download(tickers, group_by='ticker') para N>1 tickers."""
    rng = np.random.default_rng(seed)
    fechas = pd.bdate_range("2024-01-01", periods=n_dias)
    columnas = pd.MultiIndex.from_product(
        [tickers, ["Open", "High", "Low", "Close", "Volume"]], names=["Ticker", "Price"]
    )
    data = {}
    for t in tickers:
        precio_base = rng.uniform(50, 500)
        precios = precio_base + np.cumsum(rng.normal(0, 2, n_dias))
        data[(t, "Close")] = precios
        data[(t, "Open")] = precios - rng.normal(0, 1, n_dias)
        data[(t, "High")] = precios + abs(rng.normal(0, 1, n_dias))
        data[(t, "Low")] = precios - abs(rng.normal(0, 1, n_dias))
        data[(t, "Volume")] = rng.integers(1_000_000, 50_000_000, n_dias)
    df = pd.DataFrame(data, index=fechas, columns=columnas)
    return df


def _generar_datos_single_ticker(n_dias: int = 30, seed: int = 1) -> pd.DataFrame:
    """Simula exactamente el formato yf.download(['AAPL']) para un solo ticker (sin MultiIndex)."""
    rng = np.random.default_rng(seed)
    fechas = pd.bdate_range("2024-01-01", periods=n_dias)
    precios = 150 + np.cumsum(rng.normal(0, 2, n_dias))
    return pd.DataFrame({
        "Open": precios - 1, "High": precios + 1, "Low": precios - 1,
        "Close": precios, "Volume": rng.integers(1_000_000, 50_000_000, n_dias),
    }, index=fechas)


def test_extraer_close_volume_multi_ticker():
    tickers = ["AAA", "BBB", "CCC"]
    datos = _generar_datos_multi_ticker(tickers)
    close_df, vol_df = extraer_close_volume(datos, tickers)

    assert list(close_df.columns) == tickers, "Deben estar las 3 columnas de tickers"
    assert list(vol_df.columns) == tickers
    assert len(close_df) == 30, "Deben conservarse las 30 fechas"
    assert not close_df.isna().all().any(), "Ninguna columna debe quedar completamente vacía"
    # Chequeo de valores: el close de AAA en la primera fecha debe matchear la fuente sintética
    assert close_df["AAA"].iloc[0] == datos[("AAA", "Close")].iloc[0]
    print("OK: test_extraer_close_volume_multi_ticker")


def test_extraer_close_volume_single_ticker_string():
    """Caso real: yf.download('AAPL') con string (no lista) -> columnas planas, sin MultiIndex."""
    datos = _generar_datos_single_ticker()
    close_df, vol_df = extraer_close_volume(datos, ["AAPL"])

    assert list(close_df.columns) == ["AAPL"]
    assert len(close_df) == 30
    assert close_df["AAPL"].iloc[5] == datos["Close"].iloc[5]
    print("OK: test_extraer_close_volume_single_ticker_string")


def test_extraer_close_volume_lote_de_un_solo_ticker_como_lista():
    """
    Caso que backfill.py SIEMPRE produce: un lote de tamaño 1 pasado como lista
    (ej. último lote con 1 ticker sobrante). yfinance con group_by='ticker' y
    una LISTA de un elemento sigue devolviendo MultiIndex, a diferencia de pasar
    un string suelto. Este test existe porque el bug real apareció acá.
    """
    tickers = ["CCC"]
    datos = _generar_datos_multi_ticker(tickers, n_dias=10)  # MultiIndex, como hace yfinance con listas

    close_df, vol_df = extraer_close_volume(datos, tickers)

    assert list(close_df.columns) == ["CCC"]
    assert len(close_df) == 10
    assert close_df["CCC"].iloc[0] == datos[("CCC", "Close")].iloc[0]
    print("OK: test_extraer_close_volume_lote_de_un_solo_ticker_como_lista")


def test_extraer_close_volume_ticker_faltante_no_rompe():
    """Si un ticker del lote no vino en la respuesta (yfinance a veces omite
    silenciosamente tickers delisted), no debe explotar, debe omitirlo."""
    tickers_pedidos = ["AAA", "BBB", "ZZZ_NO_EXISTE"]
    datos = _generar_datos_multi_ticker(["AAA", "BBB"])  # ZZZ no vino en la respuesta

    close_df, vol_df = extraer_close_volume(datos, tickers_pedidos)

    assert "ZZZ_NO_EXISTE" not in close_df.columns, "Ticker faltante se omite, no rompe"
    assert set(close_df.columns) == {"AAA", "BBB"}
    print("OK: test_extraer_close_volume_ticker_faltante_no_rompe")


def test_matriz_ancha_concatena_bien_entre_lotes():
    """Simula 2 'lotes' con distintos tickers y valida que la concatenación final
    (axis=1, como hace ejecutar_backfill) arme una matriz ancha correcta,
    incluyendo el caso de longitudes de historia distintas (IPO reciente)."""
    lote1_tickers = ["AAA", "BBB"]
    lote2_tickers = ["CCC"]

    datos1 = _generar_datos_multi_ticker(lote1_tickers, n_dias=30)
    datos2 = _generar_datos_multi_ticker(lote2_tickers, n_dias=10)  # IPO reciente: menos historia

    close1, vol1 = extraer_close_volume(datos1, lote1_tickers)
    close2, vol2 = extraer_close_volume(datos2, lote2_tickers)

    precios_final = pd.concat([close1, close2], axis=1).sort_index()

    assert set(precios_final.columns) == {"AAA", "BBB", "CCC"}
    assert precios_final.shape[0] == 30, "El índice de fechas se une (outer join implícito de concat)"
    # CCC solo tiene 10 días de historia real → el resto debe ser NaN, NO ceros ni error
    assert precios_final["CCC"].isna().sum() == 20, "Días sin historia deben ser NaN, no 0 ni relleno falso"
    assert precios_final["AAA"].isna().sum() == 0, "AAA tiene historia completa"
    print("OK: test_matriz_ancha_concatena_bien_entre_lotes")


if __name__ == "__main__":
    test_extraer_close_volume_multi_ticker()
    test_extraer_close_volume_single_ticker_string()
    test_extraer_close_volume_lote_de_un_solo_ticker_como_lista()
    test_extraer_close_volume_ticker_faltante_no_rompe()
    test_matriz_ancha_concatena_bien_entre_lotes()
    print("\nTodos los tests pasaron.")
