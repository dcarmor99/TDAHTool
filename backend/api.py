# backend/api.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
from .utils.predict import predecir_probabilidades, cargar_metricas

app = FastAPI(title="TDAH Tool API", version="1.0.0")

# CORS abierto para que el frontend (Streamlit) pueda llamar a la API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en prod, restringe esto
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== Esquema de entrada (valores textuales tal como los mapea preprocessing.py) ======
class UserInput(BaseModel):
    conducta_status_num: Literal[
        "Nunca diagnosticado",
        "Diagnosticado pero ya no lo tiene",
        "Alguna vez diagnosticado (estado actual desconocido)",
        "Lo tiene actualmente, leve",
        "Lo tiene actualmente, moderado",
        "Lo tiene actualmente, grave",
        "Diagnóstico actual, severidad desconocida",
    ]
    sc_age_years: int = Field(ge=0, le=18)
    a1_age: int
    educacion_especial_status_num: Literal[
        "Nunca ha tenido plan especial de educación",
        "Tuvo plan, ya no recibe servicios de educación especial",
        "Tiene plan y recibe servicios de educación especial",
    ]
    hcability_num: Literal[
        "This child does not have any health conditions",
        "Never", "Sometimes", "Usually", "Always",
    ]
    ansiedad_status_num: Literal[
        "Nunca diagnosticado",
        "Diagnosticado pero ya no lo tiene",
        "Diagnosticado pero estado actual desconocido",
        "Diagnosticado y lo tiene actualmente",
    ]
    k7q84_r_num: Literal["Never", "Sometimes", "Usually", "Always"]
    k8q31_num: Literal["Never", "Rarely", "Sometimes", "Usually", "Always"]
    k7q70_r_num: Literal["Never", "Sometimes", "Usually", "Always"]
    makefriend_num: Literal["A lot of difficulty", "A little difficulty", "No difficulty"]
    sc_sex_bin: Literal["Female", "Male"]
    outdoorswkday_clean_num: Literal[
        "Too young (<3 years)",
        "Less than 1 hour per day",
        "1 hour per day",
        "2 hours per day",
        "3 hours per day",
        "4 or more hours per day",
    ]

@app.get("/health")
def health():
    """Ping de salud para comprobar que la API está viva."""
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    """
    Devuelve las métricas oficiales del modelo que guardaste en backend/model/metrics.json.
    """
    mets = cargar_metricas()
    return mets

@app.post("/predict")
def predict(
    payload: UserInput,
    include_metrics: bool = Query(False, description="Incluir métricas globales en la respuesta"),
    use_optimal_threshold: bool = Query(False, description="Aplicar también predicción con umbral óptimo si existe"),
):
    """
    Recibe valores textuales, mapea y predice.
    - Predicción estándar con umbral 0.5.
    - Si include_metrics=True, añade metrics.json.
    - Si use_optimal_threshold=True, calcula también clase con umbral óptimo.
    """
    result = predecir_probabilidades(payload.dict(), include_metrics=include_metrics)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    if use_optimal_threshold:
        mets = cargar_metricas()
        thr = mets.get("threshold_optimo") or mets.get("threshold") or mets.get("umbral")
        if thr is not None:
            # Busca prob TDAH en el resultado (ajusta a tu estructura real)
            probs = result.get("probabilidades") or result.get("probs") or {}
            p_tdah = None
            if isinstance(probs, dict):
                for k in ("TDAH", "tdah", 1, "1"):
                    if k in probs:
                        p_tdah = probs[k]
                        break
            elif isinstance(probs, (list, tuple)) and len(probs) >= 2:
                p_tdah = probs[1]  # índice 1 = clase positiva

            if p_tdah is not None:
                result["prediccion_con_umbral_optimo"] = int(float(p_tdah) >= float(thr))
                result["umbral_optimo_usado"] = thr
            else:
                result["prediccion_con_umbral_optimo"] = None
                result["umbral_optimo_usado"] = thr
                result["nota"] = "No pude inferir la probabilidad de TDAH para aplicar el umbral óptimo."

    return result