# main.py
import json
from utils.predict import predecir_probabilidades
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

# 3️⃣ Mostrar el resultado
print("\n🔍 Resultado de predicción para entrada del usuario:")
if "error" in resultado:
    print("❌ Error:", resultado["error"])
else:
    print("✅ Predicción (0 = No TDAH, 1 = TDAH):", resultado["prediccion"])
    print("🧠 Probabilidad No TDAH:", resultado["probabilidad_no_tdah"], "%")
    print("🧠 Probabilidad TDAH:", resultado["probabilidad_tdah"], "%")

    if "metrics" in resultado:
        print("\n📊 Métricas del modelo (desde metrics.json):")
        print(json.dumps(resultado["metrics"], indent=2, ensure_ascii=False))
    else:
        print("\n⚠️ No llegaron métricas en la respuesta. Comprobando ruta...")
        print("Ruta METRICS_PATH:", METRICS_PATH)
        print("Existe el archivo?:", os.path.exists(METRICS_PATH))
        # Alternativa: leerlas directamente
        print("\n📊 Métricas (cargar_metricas()):")
        print(json.dumps(cargar_metricas(), indent=2, ensure_ascii=False))