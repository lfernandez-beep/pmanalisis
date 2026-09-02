import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

import numpy as np
import pandas as pd
from metrics import (
    calcular_retornos_diarios, sharpe_ratio, beta_y_r2, max_drawdown,
    var_cvar_historico, calcular_metricas_activo,
)


def test_beta_activo_identico_al_benchmark_es_uno():
    """Si el activo ES el benchmark, beta debe dar exactamente 1.0 y R² = 1.0."""
    rng = np.random.default_rng(0)
    retornos_bench = pd.Series(rng.normal(0.0005, 0.01, 500))
    beta, r2 = beta_y_r2(retornos_bench, retornos_bench)
    assert abs(beta - 1.0) < 1e-9, f"Beta debe ser 1.0, dio {beta}"
    assert abs(r2 - 1.0) < 1e-9, f"R2 debe ser 1.0, dio {r2}"
    print("OK: test_beta_activo_identico_al_benchmark_es_uno")


def test_beta_activo_2x_apalancado_da_dos():
    """Un activo sintético = 2x el benchmark (sin ruido) debe dar beta=2.0 exacto."""
    rng = np.random.default_rng(1)
    retornos_bench = pd.Series(rng.normal(0.0003, 0.01, 500))
    retornos_2x = retornos_bench * 2.0
    beta, r2 = beta_y_r2(retornos_2x, retornos_bench)
    assert abs(beta - 2.0) < 1e-9, f"Beta debe ser 2.0, dio {beta}"
    assert abs(r2 - 1.0) < 1e-6, f"R2 debe ser ~1.0 (relación lineal perfecta), dio {r2}"
    print("OK: test_beta_activo_2x_apalancado_da_dos")


def test_beta_activo_sin_relacion_da_cero_aprox():
    """Un activo con retornos totalmente independientes del benchmark debe dar beta ~0."""
    rng = np.random.default_rng(2)
    retornos_bench = pd.Series(rng.normal(0.0003, 0.01, 2000))
    retornos_independiente = pd.Series(rng.normal(0.0003, 0.01, 2000))  # otra semilla, sin relación
    beta, r2 = beta_y_r2(retornos_independiente, retornos_bench)
    assert abs(beta) < 0.15, f"Beta debe ser cercano a 0 (series independientes), dio {beta}"
    print("OK: test_beta_activo_sin_relacion_da_cero_aprox")


def test_sharpe_ratio_con_retorno_constante_conocido():
    """
    Serie con retorno diario constante conocido y volatilidad conocida:
    Sharpe debe matchear el cálculo manual exacto.
    """
    retorno_diario = 0.001
    vol_diaria = 0.01
    rf_diario = 0.0001
    n = 1000
    # Serie determinística: mismo retorno todos los días + ruido simétrico controlado
    rng = np.random.default_rng(3)
    ruido = rng.normal(0, vol_diaria, n)
    retornos = pd.Series(retorno_diario + ruido)

    sharpe_calculado = sharpe_ratio(retornos, rf_diario)
    exceso_manual = retornos - rf_diario
    sharpe_manual = (exceso_manual.mean() / exceso_manual.std()) * np.sqrt(252)

    assert abs(sharpe_calculado - sharpe_manual) < 1e-9
    # Con estos parámetros el Sharpe anualizado debe rondar valores razonables (no NaN, no infinito)
    assert not np.isnan(sharpe_calculado) and abs(sharpe_calculado) < 50
    print("OK: test_sharpe_ratio_con_retorno_constante_conocido")


def test_max_drawdown_con_caida_conocida():
    """Serie de precios diseñada con una caída del exactamente 50% desde el pico."""
    precios = pd.Series([100, 110, 120, 60, 70, 90])  # pico=120, valle=60 -> dd = -50%
    mdd = max_drawdown(precios)
    assert abs(mdd - (-0.5)) < 1e-9, f"Drawdown debe ser exactamente -0.5, dio {mdd}"
    print("OK: test_max_drawdown_con_caida_conocida")


def test_var_cvar_con_distribucion_conocida():
    """CVaR debe ser siempre <= VaR (la cola es peor o igual que el percentil de corte)."""
    rng = np.random.default_rng(4)
    retornos = pd.Series(rng.normal(0.0005, 0.02, 1000))
    var, cvar = var_cvar_historico(retornos, nivel_confianza=0.95)
    assert cvar <= var, f"CVaR ({cvar}) debe ser <= VaR ({var}), la cola siempre es peor"
    print("OK: test_var_cvar_con_distribucion_conocida")


def test_activo_con_historia_insuficiente_devuelve_none_no_error():
    """Ticker con < 30 observaciones (ej. IPO reciente) no debe romper el pipeline,
    debe devolver estructura con None explícito."""
    precios_cortos = pd.Series([100.0, 101.0, 99.0, 102.0])  # solo 4 días
    retornos_cortos = calcular_retornos_diarios(precios_cortos.to_frame("X"))["X"]
    retornos_bench = pd.Series(np.random.default_rng(5).normal(0.0003, 0.01, 500))

    resultado = calcular_metricas_activo(
        ticker="IPO_RECIENTE", precios=precios_cortos, retornos_diarios=retornos_cortos,
        retornos_benchmark=retornos_bench, rf_diario=0.0001, rf_anual=0.025,
        retorno_benchmark_anual=0.08,
    )
    assert resultado["sharpe"] is None
    assert resultado["ticker"] == "IPO_RECIENTE"
    print("OK: test_activo_con_historia_insuficiente_devuelve_none_no_error")


if __name__ == "__main__":
    test_beta_activo_identico_al_benchmark_es_uno()
    test_beta_activo_2x_apalancado_da_dos()
    test_beta_activo_sin_relacion_da_cero_aprox()
    test_sharpe_ratio_con_retorno_constante_conocido()
    test_max_drawdown_con_caida_conocida()
    test_var_cvar_con_distribucion_conocida()
    test_activo_con_historia_insuficiente_devuelve_none_no_error()
    print("\nTodos los tests de metrics pasaron.")
