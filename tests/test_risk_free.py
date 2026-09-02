import sys
from pathlib import Path
from unittest.mock import patch
from io import StringIO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

import pandas as pd
from risk_free import obtener_risk_free_rate


def test_obtener_risk_free_rate_parsea_bien_y_toma_ultimo_valor():
    """
    FRED marca días sin dato con '.' (fin de semana, feriado). La lógica debe
    ignorarlos y devolver el último valor NUMÉRICO real, no el último renglón
    del archivo (que podría ser un '.').
    """
    # Nombre de columna real confirmado contra FRED en vivo: 'observation_date'
    # (FRED cambió esto en algún momento, antes era 'DATE' -- por eso se detecta
    # dinámicamente en el código, no se hardcodea ninguno de los dos nombres)
    csv_sintetico = (
        "observation_date,DGS3MO\n"
        "2026-08-25,4.22\n"
        "2026-08-26,4.21\n"
        "2026-08-27,.\n"          # feriado simulado
        "2026-08-28,4.19\n"
        "2026-08-29,.\n"          # fin de semana simulado
    )

    with patch("pandas.read_csv", return_value=pd.read_csv(StringIO(csv_sintetico))):
        tasa, fecha = obtener_risk_free_rate(serie="DGS3MO")

    assert abs(tasa - 0.0419) < 1e-9, f"Debe tomar 4.19% -> 0.0419, se obtuvo {tasa}"
    assert fecha == "2026-08-28", "Debe tomar la fecha del último valor NUMÉRICO, no del último renglón"
    print("OK: test_obtener_risk_free_rate_parsea_bien_y_toma_ultimo_valor")


def test_obtener_risk_free_rate_funciona_con_nombre_de_columna_viejo_DATE():
    """
    Robustez: si FRED alguna vez vuelve al nombre viejo 'DATE' (o cualquier
    otro nombre), debe seguir funcionando porque la detección es dinámica,
    no hardcodeada a un string fijo.
    """
    csv_con_nombre_viejo = (
        "DATE,DGS3MO\n"
        "2026-08-25,4.22\n"
        "2026-08-28,4.19\n"
    )
    with patch("pandas.read_csv", return_value=pd.read_csv(StringIO(csv_con_nombre_viejo))):
        tasa, fecha = obtener_risk_free_rate(serie="DGS3MO")

    assert abs(tasa - 0.0419) < 1e-9
    assert fecha == "2026-08-28"
    print("OK: test_obtener_risk_free_rate_funciona_con_nombre_de_columna_viejo_DATE")


def test_obtener_risk_free_rate_serie_vacia_lanza_error():
    """Si todos los valores son '.', debe fallar explícitamente (no inventar 0.0)."""
    csv_vacio = "DATE,DGS3MO\n2026-08-25,.\n2026-08-26,.\n"

    with patch("pandas.read_csv", return_value=pd.read_csv(StringIO(csv_vacio))):
        try:
            obtener_risk_free_rate(serie="DGS3MO")
            assert False, "Debía lanzar ValueError"
        except ValueError:
            pass
    print("OK: test_obtener_risk_free_rate_serie_vacia_lanza_error")


if __name__ == "__main__":
    test_obtener_risk_free_rate_parsea_bien_y_toma_ultimo_valor()
    test_obtener_risk_free_rate_funciona_con_nombre_de_columna_viejo_DATE()
    test_obtener_risk_free_rate_serie_vacia_lanza_error()
    print("\nTodos los tests de risk_free pasaron.")
