import joblib
import pandas as pd
import os
import json
from .preprocessing import preprocess_user_input  # 👈 Importa la función de preprocesamiento

# 📍 Ruta al modelo entrenado (.pkl)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "modelo_tdahtool_5.pkl")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "metrics_4.json")
# 🚀 Cargar el modelo una sola vez cuando se importe el archivo
model = joblib.load(MODEL_PATH)

def cargar_metricas():
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"message": "metrics.json no encontrado"}
    
def predecir_probabilidades(data_dict, include_metrics):
    """
    Recibe un diccionario con los valores de entrada del usuario
    (por ejemplo, desde un formulario con textos), los codifica
    usando los mapeos definidos en preprocessing.py, y devuelve
    las probabilidades de TDAH y No TDAH, además de la predicción.

    Parámetros:
        data_dict (dict): Diccionario con valores del usuario en texto (sin codificar).

    Retorna:
        dict: Contiene:
            - prediccion: clase predicha (0 o 1)
            - probabilidad_no_tdah: en porcentaje
            - probabilidad_tdah: en porcentaje
    """
    try:
        # 🧼 Preprocesar y reordenar las columnas
        input_df = preprocess_user_input(data_dict)

        # 🔮 Calcular las probabilidades
        probabilities = model.predict_proba(input_df)[0]

        # 🏷️ Predecir clase (0 o 1)
        pred_clase = model.predict(input_df)[0]

        # 📦 Devolver resultados en formato legible
        resultado = {
            "prediccion": int(pred_clase),
            "probabilidad_no_tdah": round(probabilities[0] * 100, 2),
            "probabilidad_tdah": round(probabilities[1] * 100, 2)
        }

        if include_metrics:
            resultado["metrics"] = cargar_metricas()

        return resultado

    except Exception as e:
        # ❌ Devuelve el error si ocurre algo (por ejemplo, clave inválida o valor no mapeado)
        return {"error": str(e)}
