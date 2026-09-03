"""
Handler HTTP de la Netlify Function.

Endpoint: POST /.netlify/functions/analyze
Body JSON esperado:
{
  "posiciones": [
    {"ticker": "AAPL", "peso": 0.4},
    {"ticker": "MSFT", "peso": null}   <- sin peso: se completa vía HRP
  ],
  "metodo_completar": "hrp" | "kelly"   <- solo si hay posiciones sin peso
}

Si TODAS las posiciones vienen con peso -> devuelve métricas del portfolio tal cual.
Si ALGUNA viene sin peso -> corre HRP/Kelly sobre el conjunto completo de tickers
pedidos (ignorando los pesos que sí vinieron) y devuelve los pesos sugeridos + métricas.
"""
import json
import os

import pandas as pd

from portfolio_logic import calcular_pesos_hrp, calcular_pesos_kelly, metricas_portfolio

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PRICES_PATH = os.path.join(DATA_DIR, "precios_ajustados.parquet")
METRICS_PATH = os.path.join(DATA_DIR, "metricas.json")

BENCHMARK_TICKER = "SPY"


def _cargar_retornos_y_contexto():
    precios = pd.read_parquet(PRICES_PATH)
    import numpy as np
    retornos = np.log(precios / precios.shift(1)).dropna(how="all")

    with open(METRICS_PATH) as f:
        metricas_universo = json.load(f)
    rf_anual = metricas_universo["risk_free_rate"]
    rf_diario = rf_anual / 252

    return retornos, rf_diario


def _respuesta(status_code: int, body: dict):
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(body),
    }


def handler(event, context):
    if event.get("httpMethod") == "OPTIONS":
        return _respuesta(200, {})

    try:
        payload = json.loads(event.get("body") or "{}")
    except json.JSONDecodeError:
        return _respuesta(400, {"error": "Body no es JSON válido"})

    posiciones = payload.get("posiciones", [])
    metodo_completar = payload.get("metodo_completar", "hrp")

    if not posiciones:
        return _respuesta(400, {"error": "Se requiere al menos 1 posición en 'posiciones'"})

    tickers_pedidos = [p["ticker"].upper() for p in posiciones]
    tiene_peso_faltante = any(p.get("peso") is None for p in posiciones)

    try:
        retornos, rf_diario = _cargar_retornos_y_contexto()
    except FileNotFoundError as e:
        return _respuesta(500, {"error": f"Datos del pipeline no encontrados: {e}"})

    tickers_no_encontrados = [t for t in tickers_pedidos if t not in retornos.columns]
    if tickers_no_encontrados:
        return _respuesta(400, {
            "error": f"Tickers no encontrados en el universo: {tickers_no_encontrados}. "
                     f"Verificá que estén dentro del universo de 1907 activos cubiertos."
        })

    retornos_subset = retornos[tickers_pedidos]
    retornos_benchmark = retornos[BENCHMARK_TICKER] if BENCHMARK_TICKER in retornos.columns else None

    try:
        if tiene_peso_faltante:
            if metodo_completar == "kelly":
                pesos = calcular_pesos_kelly(retornos_subset, rf_diario)
            else:
                pesos = calcular_pesos_hrp(retornos_subset)
            metodo_usado = metodo_completar
        else:
            pesos_dict = {p["ticker"].upper(): p["peso"] for p in posiciones}
            suma = sum(pesos_dict.values())
            if abs(suma - 1.0) > 0.02:
                return _respuesta(400, {
                    "error": f"Los pesos suman {suma:.4f}, deben sumar 1.0 (100%). "
                             f"Ajustá las posiciones o dejá 'peso': null para que se autocomplete."
                })
            pesos = pd.Series(pesos_dict)
            metodo_usado = "pesos_provistos_por_usuario"

        metricas = metricas_portfolio(retornos_subset, pesos, rf_diario, retornos_benchmark)

    except ValueError as e:
        return _respuesta(400, {"error": str(e)})

    return _respuesta(200, {
        "metodo": metodo_usado,
        "pesos": {t: round(float(w), 6) for t, w in pesos.items()},
        "metricas_portfolio": metricas,
    })
