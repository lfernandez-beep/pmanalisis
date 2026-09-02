import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

import pandas as pd
from update import hacer_upsert


def test_upsert_preserva_historico_viejo_no_solapado():
    """Fechas antiguas que no vienen en la descarga nueva deben quedar intactas."""
    existente = pd.DataFrame(
        {"AAA": [100, 101, 102]}, index=pd.bdate_range("2024-01-01", periods=3)
    )
    nueva = pd.DataFrame(
        {"AAA": [999]}, index=pd.bdate_range("2024-01-10", periods=1)
    )
    resultado = hacer_upsert(existente, nueva)

    assert resultado.loc["2024-01-01", "AAA"] == 100
    assert resultado.loc["2024-01-02", "AAA"] == 101
    assert resultado.loc["2024-01-03", "AAA"] == 102
    assert resultado.loc["2024-01-10", "AAA"] == 999
    print("OK: test_upsert_preserva_historico_viejo_no_solapado")


def test_upsert_dato_nuevo_pisa_al_viejo_en_fecha_solapada():
    """
    Caso clave: si una fecha se descarga de nuevo (dentro de la ventana de
    margen) con un valor DISTINTO al que ya teníamos (ej. ajuste retroactivo
    por dividendo declarado después), debe ganar el valor NUEVO.
    """
    fecha_solapada = pd.Timestamp("2024-01-03")
    existente = pd.DataFrame(
        {"AAA": [100, 101, 102]},
        index=pd.bdate_range("2024-01-01", periods=3)
    )
    nueva = pd.DataFrame(
        {"AAA": [102, 105, 106]},  # 2024-01-03 viene con valor corregido: 102 -> pero simulamos 999 para distinguir
        index=pd.bdate_range("2024-01-03", periods=3)
    )
    nueva.loc[fecha_solapada, "AAA"] = 999  # valor corregido retroactivamente

    resultado = hacer_upsert(existente, nueva)

    assert resultado.loc[fecha_solapada, "AAA"] == 999, "El valor nuevo/corregido debe ganar sobre el viejo"
    assert resultado.loc["2024-01-01", "AAA"] == 100, "Fecha no solapada, vieja, se preserva"
    print("OK: test_upsert_dato_nuevo_pisa_al_viejo_en_fecha_solapada")


def test_upsert_agrega_ticker_nuevo_sin_romper_los_existentes():
    """Si aparece un ticker que no estaba en la matriz vieja, se agrega como columna nueva."""
    existente = pd.DataFrame(
        {"AAA": [100, 101]}, index=pd.bdate_range("2024-01-01", periods=2)
    )
    nueva = pd.DataFrame(
        {"AAA": [102], "BBB": [50]}, index=pd.bdate_range("2024-01-03", periods=1)
    )
    resultado = hacer_upsert(existente, nueva)

    assert "BBB" in resultado.columns
    assert resultado.loc["2024-01-03", "BBB"] == 50
    assert pd.isna(resultado.loc["2024-01-01", "BBB"]), "BBB no existía en esa fecha vieja -> NaN, no error"
    print("OK: test_upsert_agrega_ticker_nuevo_sin_romper_los_existentes")


if __name__ == "__main__":
    test_upsert_preserva_historico_viejo_no_solapado()
    test_upsert_dato_nuevo_pisa_al_viejo_en_fecha_solapada()
    test_upsert_agrega_ticker_nuevo_sin_romper_los_existentes()
    print("\nTodos los tests de update pasaron.")
