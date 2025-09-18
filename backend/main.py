# main.py
import json
from utils.predict import predecir_probabilidades,cargar_metricas, METRICS_PATH, MODEL_PATH, explicar_ruta_lineal
import os


# 1️⃣ Simulación de entrada textual del usuario
entrada_usuario = {
    "conducta_status_num": "Nunca diagnosticado",
    "sc_age_years": 5,
    "a1_age": 35,
    "educacion_especial_status_num": "Nunca ha tenido plan especial de educación",
    "hcability_num": "This child does not have any health conditions",
    "ansiedad_status_num": "Nunca diagnosticado",
    "k7q84_r_num": "Usually",
    "k8q31_num": "Sometimes",
    "k7q70_r_num": "Sometimes",
    "makefriend_num": "No difficulty",
    "sc_sex_bin": "Female",
    "outdoorswkday_clean_num": "1 hour per day",
}


# 2️⃣ Llamar a la función predictiva del backend
resultado = predecir_probabilidades(entrada_usuario, include_metrics=True)

metricas = resultado.get("metrics") or cargar_metricas()

# 🧭 Explicación tipo cadena con confianza por nodo
exp = explicar_ruta_lineal(entrada_usuario)

# 3️⃣ Mostrar el resultado
if "error" in resultado:
    print("❌ Error en la predicción:", resultado["error"])
else:
    print("\n✅ Predicción (0 = No TDAH, 1 = TDAH):", resultado["prediccion"])
    print("🧠 Probabilidad No TDAH:", resultado["probabilidad_no_tdah"], "%")
    print("🧠 Probabilidad TDAH:", resultado["probabilidad_tdah"], "%")

print("\n📊 Métricas del modelo:")
print(json.dumps(metricas, indent=2, ensure_ascii=False))

print("\n🧭 Explicación de la ruta (JSON):")
print(json.dumps(exp, indent=2, ensure_ascii=False))