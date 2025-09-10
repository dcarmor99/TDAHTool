# main.py
import json
from utils.predict import predecir_probabilidades
import os


# 1️⃣ Simulación de entrada textual del usuario
entrada_usuario = {
    "conducta_status_num": "Diagnosticado pero ya no lo tiene",
    "sc_age_years": 16,
    "educacion_especial_status_num": "Tiene plan y recibe servicios de educación especial",
    "hcability_num": "This child does not have any health conditions",
    "birth_yr": 2009,
    "ansiedad_status_num": "Diagnosticado y lo tiene actualmente",
    "k7q84_r_num": "Always",
    "k7q70_r_num": "Sometimes",
    "k8q31_num": "Never",
    "sc_sex_bin": "Female",
    "makefriend_num": "A lot of difficulty",
    "outdoorswkday_clean_num": "4 or more hours per day"
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