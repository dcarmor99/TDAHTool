# frontend/app.py
import requests
import streamlit as st
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import os

# -------- Config --------
st.set_page_config(page_title="TDAH Tool — Frontend", layout="centered")
#API_URL = "http://127.0.0.1:8000"
API_URL = os.getenv("API_URL", "http://backend:8000")
# Ruta por defecto a la imagen del árbol
BASE_DIR = Path(__file__).resolve().parent  # .../frontend
TREE_IMAGE_PATH = BASE_DIR / "img" / "decision_tree_complete.png"

# Variables que usa el backend para predecir (12)
BACKEND_VARS_12 = [
    'conducta_status_num', 'sc_age_years', 'birth_yr',
    'educacion_especial_status_num', 'hcability_num', 'k8q31_num',
    'k7q70_r_num', 'ansiedad_status_num', 'k7q84_r_num',
    'makefriend_num', 'sc_sex_bin', 'outdoorswkday_clean_num'
]
# -------- Helpers --------
def api_health() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.ok and r.json().get("status") == "ok"
    except Exception:
        return False

@st.cache_data(ttl=30)
def get_metrics():
    try:
        r = requests.get(f"{API_URL}/metrics", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}

def call_predict(payload: dict, include_metrics: bool = True):
    params = {"include_metrics": include_metrics} if include_metrics else {}
    r = requests.post(f"{API_URL}/predict", json=payload, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def extract_pct(resp: dict):
    """Devuelve (tdah_pct, no_tdah_pct) en 0–100."""
    keys = {k.lower(): k for k in resp.keys()}
    if "probabilidad_tdah" in keys and "probabilidad_no_tdah" in keys:
        k_t = keys["probabilidad_tdah"]; k_n = keys["probabilidad_no_tdah"]
        return float(resp[k_t]), float(resp[k_n])
    probs = resp.get("probabilidades") or resp.get("probs") or resp.get("probability")
    if isinstance(probs, dict):
        for k in ("TDAH", "tdah", 1, "1"):
            if k in probs:
                p = float(probs[k]) * 100.0
                return p, 100.0 - p
    elif isinstance(probs, (list, tuple)) and len(probs) >= 2:
        p = float(probs[1]) * 100.0
        return p, 100.0 - p
    return None, None

def fmt_pct(x):
    return f"{float(x):.2f}%"

def pctify(v):
    try:
        v = float(v)
        return f"{(v*100 if v <= 1.0 else v):.2f}%"
    except Exception:
        return "-"

def get_first(d, names):
    for n in names:
        if n in d: return d[n]
    return None

def show_http_error(prefix: str, e: requests.HTTPError):
    msg = f"{prefix}: {e}"
    if e.response is not None:
        try:
            detail = e.response.json()
            msg += f"\nDetalle: {detail}"
        except Exception:
            msg += f"\nRespuesta: {e.response.text}"
    st.error(msg)

# =======================
#  Opciones EXACTAS que espera el backend (UserInput)
#  -> Algunas en ESPAÑOL, otras en INGLÉS
#  -> En el payload se envían TAL CUAL
# =======================
OP_CONDUCTA = [
    "Nunca diagnosticado",
    "Diagnosticado pero ya no lo tiene",
    "Alguna vez diagnosticado (estado actual desconocido)",
    "Lo tiene actualmente, leve",
    "Lo tiene actualmente, moderado",
    "Lo tiene actualmente, grave",
    "Diagnóstico actual, severidad desconocida",
]
OP_EDU_ESP = [
    "Nunca ha tenido plan especial de educación",
    "Tuvo plan, ya no recibe servicios de educación especial",
    "Tiene plan y recibe servicios de educación especial",
]
OP_HCABILITY = [
    "This child does not have any health conditions",
    "Never", "Sometimes", "Usually", "Always",
]
OP_NEVER_ALWAYS_5 = ["Never", "Rarely", "Sometimes", "Usually", "Always"]
OP_NEVER_ALWAYS_4 = ["Never", "Sometimes", "Usually", "Always"]
OP_ANSIEDAD = [
    "Nunca diagnosticado",
    "Diagnosticado pero ya no lo tiene",
    "Diagnosticado pero estado actual desconocido",
    "Diagnosticado y lo tiene actualmente",
]
OP_MAKEFRIEND = ["A lot of difficulty", "A little difficulty", "No difficulty"]
OP_SEX = ["Female", "Male"]
OP_OUTDOORS_WKDAY = [
    "Too young (<3 years)",
    "Less than 1 hour per day",
    "1 hour per day",
    "2 hours per day",
    "3 hours per day",
    "4 or more hours per day",
]

# Traducciones SOLO para mostrar (cuando la opción está en inglés)
L_HC_ES = {
    "This child does not have any health conditions": "El menor no presenta problemas de salud",
    "Never": "Nunca", "Sometimes": "A veces", "Usually": "Normalmente", "Always": "Siempre",
}
L_K8Q31_ES = {"Never": "Nunca", "Rarely": "Casi nunca", "Sometimes": "A veces", "Usually": "Normalmente", "Always": "Siempre"}
L_K7Q70R_ES = {"Never": "Nunca", "Sometimes": "A veces", "Usually": "Normalmente", "Always": "Siempre"}
L_K7Q84R_ES = {"Never": "Nunca", "Sometimes": "A veces", "Usually": "Normalmente", "Always": "Siempre"}
L_MAKEFRIEND_ES = {"A lot of difficulty": "Mucha dificultad", "A little difficulty": "Dificultad moderada", "No difficulty": "Sin dificultad"}
L_SEX_ES = {"Female": "Femenino", "Male": "Masculino"}
L_OUTDOORS_WKDAY_ES = {
    "Too young (<3 years)": "Muy joven (<3 años)",
    "Less than 1 hour per day": "Menos de 1 hora por día",
    "1 hour per day": "1 hora por día",
    "2 hours per day": "2 horas por día",
    "3 hours per day": "3 horas por día",
    "4 or more hours per day": "4 o más horas por día",
}
def tr(mapping):
    return lambda v: mapping.get(v, v)

# -------- UI: Cabecera --------
st.title("🧠 TDAH Tool — Demo")
if api_health():
    st.success("API conectada ✅")
else:
    st.error("API no disponible ❌ (revisa que esté corriendo en 127.0.0.1:8000)")

# -------- Form builder --------
def render_controls(prefix: str, use_sliders_for_numbers: bool = False, auto_birth_from_age: bool = True):
    col1, col2 = st.columns(2)
    with col1:
        # Campos que el backend espera en ESPAÑOL (ya vienen en español)
        conducta_status_num = st.selectbox(f"{prefix}Estado de conducta del paciente", OP_CONDUCTA, key=f"{prefix}conducta")

        # Numéricos
        sc_age_years = (
            st.slider(f"{prefix}Edad del paciente (años)", 0, 18, 10, 1, key=f"{prefix}age")
            if use_sliders_for_numbers else
            st.number_input(f"{prefix}Edad del paciente (años)", min_value=0, max_value=18, value=10, step=1, key=f"{prefix}age")
        )
        current_year = datetime.now().year
        birth_yr_default = int(current_year - int(sc_age_years)) if auto_birth_from_age else 2015
        birth_yr = (
            st.slider(f"{prefix}Año de nacimiento", current_year-25, current_year, birth_yr_default, 1, key=f"{prefix}by")
            if use_sliders_for_numbers else
            st.number_input(f"{prefix}Año de nacimiento", value=birth_yr_default, step=1, key=f"{prefix}by")
        )

        educacion_especial_status_num = st.selectbox(f"{prefix}Estado de servicios de educación especial", OP_EDU_ESP, key=f"{prefix}edu")

        # Campos que el backend espera en INGLÉS -> mostramos traducido con format_func
        hcability_num = st.selectbox(
            f"{prefix}Condiciones de salud presentes en el paciente",
            OP_HCABILITY, format_func=tr(L_HC_ES), key=f"{prefix}hca"
        )

    with col2:
        k8q31_num = st.selectbox(
            f"{prefix}Dificultad para cuidar al paciente",
            OP_NEVER_ALWAYS_5, format_func=tr(L_K8Q31_ES), key=f"{prefix}k831"
        )
        k7q70_r_num = st.selectbox(
            f"{prefix}El paciente discute mucho",
            OP_NEVER_ALWAYS_4, format_func=tr(L_K7Q70R_ES), key=f"{prefix}k770"
        )
        ansiedad_status_num = st.selectbox(
            f"{prefix}Estado de ansiedad en el paciente",
            OP_ANSIEDAD, key=f"{prefix}ans"  # estas opciones ya están en español, coinciden con el backend
        )
        k7q84_r_num = st.selectbox(
            f"{prefix}Persistencia del paciente para finalizar las tareas",
            OP_NEVER_ALWAYS_4, format_func=tr(L_K7Q84R_ES), key=f"{prefix}k784"
        )
        makefriend_num = st.selectbox(
            f"{prefix}Dificultad para hacer amigos",
            OP_MAKEFRIEND, format_func=tr(L_MAKEFRIEND_ES), key=f"{prefix}mf"
        )
        sc_sex_bin = st.selectbox(
            f"{prefix}Sexo del paciente",
            OP_SEX, format_func=tr(L_SEX_ES), key=f"{prefix}sex"
        )
        outdoorswkday_clean_num = st.selectbox(
            f"{prefix}Tiempo de juego del paciente (entre semana)",
            OP_OUTDOORS_WKDAY, format_func=tr(L_OUTDOORS_WKDAY_ES), key=f"{prefix}out"
        )

    # >>> IMPORTANTE: devolvemos EXACTAMENTE lo que espera el backend (strings/ints)
    return {
        "conducta_status_num": conducta_status_num,                       # str (ES)
        "sc_age_years": int(sc_age_years),                                # int
        "birth_yr": int(birth_yr),                                        # int
        "educacion_especial_status_num": educacion_especial_status_num,   # str (ES)
        "hcability_num": hcability_num,                                    # str (EN)
        "k8q31_num": k8q31_num,                                            # str (EN)
        "k7q70_r_num": k7q70_r_num,                                        # str (EN)
        "ansiedad_status_num": ansiedad_status_num,                        # str (ES)
        "k7q84_r_num": k7q84_r_num,                                        # str (EN)
        "makefriend_num": makefriend_num,                                  # str (EN)
        "sc_sex_bin": sc_sex_bin,                                          # str (EN)
        "outdoorswkday_clean_num": outdoorswkday_clean_num,                # str (EN)
    }

# -------- Tabs --------
tab_form, tab_explore, tab_model = st.tabs(["Formulario", "Explorar", "Modelo"])

# ======= TAB 1: FORMULARIO =======
with tab_form:
    st.subheader("Introducir datos y obtener predicción")
    payload = render_controls(prefix="F_", use_sliders_for_numbers=False, auto_birth_from_age=True)

    include_metrics = st.checkbox("Incluir métricas en la respuesta", value=True, key="F_incmet")

    # (Opcional) depuración rápida del payload
    with st.expander("Payload que se enviará a /predict"):
        st.json(payload)

    if st.button("Calcular (Formulario)", key="F_calc"):
        try:
            resp = call_predict(payload, include_metrics=include_metrics)
            tdah_pct, no_pct = extract_pct(resp)

            cls = resp.get("prediccion")
            if cls in (0, 1):
                st.subheader(f"Predicción del modelo: **{'TDAH' if cls==1 else 'NO TDAH'}**")

            if tdah_pct is not None:
                colA, colB = st.columns(2)
                colA.metric("Probabilidad TDAH", fmt_pct(tdah_pct))
                colB.metric("Probabilidad NO TDAH", fmt_pct(no_pct))

                fig, ax = plt.subplots()
                ax.bar(["NO TDAH", "TDAH"], [no_pct, tdah_pct])
                ax.set_ylabel("Porcentaje")
                ax.set_ylim(0, 100)
                for i, v in enumerate([no_pct, tdah_pct]):
                    ax.text(i, v + 1, f"{v:.2f}%", ha="center", va="bottom")
                st.pyplot(fig)
            else:
                st.info("No pude interpretar las probabilidades. Respuesta completa abajo.")

            mets = get_metrics()
            if mets:
                st.markdown("### Métricas del modelo (validación)")
                acc = mets.get("accuracy")
                rec = mets.get("recall_TDAH") or mets.get("recall_tdah")
                prec = mets.get("precision_TDAH") or mets.get("precision_tdah") or mets.get("precision")
                f1 = mets.get("f1_TDAH") or mets.get("f1") or mets.get("f1_score")
                spec = mets.get("specificity_NoTDAH") or mets.get("specificity") or mets.get("especificidad") or mets.get("tnr")
                auc = mets.get("roc_auc") or mets.get("auc")
                thr = mets.get("threshold_optimo") or mets.get("umbral_optimo") or mets.get("threshold") or mets.get("umbral")

                c1, c2, c3 = st.columns(3)
                c1.metric("Accuracy", pctify(acc) if acc is not None else "-")
                c2.metric("Recall (TDAH)", pctify(rec) if rec is not None else "-")
                c3.metric("Precision (TDAH)", pctify(prec) if prec is not None else "-")

                c4, c5, c6 = st.columns(3)
                c4.metric("F1 (TDAH)", pctify(f1) if f1 is not None else "-")
                c5.metric("Specificidad (NO TDAH)", pctify(spec) if spec is not None else "-")
                c6.metric("AUC (ROC)", pctify(auc) if auc is not None else "-")

                if thr is not None:
                    st.caption(f"**Umbral óptimo**: {pctify(thr)}")
            else:
                st.info("No se pudieron cargar las métricas.")

            with st.expander("Respuesta cruda de /predict"):
                st.json(resp)

        except requests.HTTPError as e:
            show_http_error("Error en /predict", e)
        except requests.RequestException as e:
            st.error(f"Error en /predict: {e}")

# ======= TAB 2: EXPLORAR =======
with tab_explore:
    st.subheader("Mover controles y visualizar cómo cambia la probabilidad")
    payload2 = render_controls(prefix="E_", use_sliders_for_numbers=True, auto_birth_from_age=True)

    if "hist_probs_pct" not in st.session_state:
        st.session_state.hist_probs_pct = []
    if "hist_probs_no_pct" not in st.session_state:
        st.session_state.hist_probs_no_pct = []
    if "hist_inputs" not in st.session_state:
        st.session_state.hist_inputs = []

    col1, col2 = st.columns([1,1])
    if col1.button("Calcular y registrar punto", key="E_calc"):
        try:
            resp = call_predict(payload2, include_metrics=False)
            tdah_pct, no_pct = extract_pct(resp)

            if tdah_pct is not None:
                st.session_state.hist_probs_pct.append(float(tdah_pct))
                no_val = float(no_pct) if no_pct is not None else 100.0 - float(tdah_pct)
                st.session_state.hist_probs_no_pct.append(no_val)
                st.session_state.hist_inputs.append(payload2.copy())
                st.success(f"Punto registrado: P(TDAH) = {fmt_pct(tdah_pct)} | P(No TDAH) = {fmt_pct(no_val)}")
            else:
                st.warning("No pude extraer P(TDAH). Respuesta completa abajo.")

            with st.expander("Respuesta cruda de /predict"):
                st.json(resp)

        except requests.HTTPError as e:
            show_http_error("Error en /predict", e)
        except requests.RequestException as e:
            st.error(f"Error en /predict: {e}")

    if col2.button("Reset historial", key="E_reset"):
        st.session_state.hist_probs_pct = []
        st.session_state.hist_probs_no_pct = []
        st.session_state.hist_inputs = []
        st.info("Historial reiniciado.")

    mets = get_metrics()
    thr = get_first(mets, ["threshold_optimo", "umbral_optimo", "threshold", "umbral"])
    thr_pct = None
    if isinstance(thr, (int, float)):
        thr_pct = thr*100 if thr <= 1.0 else float(thr)
    elif isinstance(thr, str) and thr.replace('.', '', 1).isdigit():
        val = float(thr); thr_pct = val*100 if val <= 1.0 else val

    if st.session_state.hist_probs_pct:
        st.subheader("Evolución de P(TDAH)")
        xs = list(range(1, len(st.session_state.hist_probs_pct) + 1))
        ys = st.session_state.hist_probs_pct

        fig, ax = plt.subplots()
        ax.plot(xs, ys, marker="o")
        ax.set_xlabel("Cálculo #")
        ax.set_ylabel("P(TDAH) (%)")
        ax.set_ylim(0, 100)
        if thr_pct is not None:
            ax.axhline(thr_pct, linestyle="--")
            ax.text(xs[-1] if xs else 1, thr_pct, " umbral", va="bottom", ha="right")
        st.pyplot(fig)

        last = ys[-1]
        delta = f"{(last - ys[-2]):.2f} pp" if len(ys) >= 2 else None
        st.metric("Última P(TDAH)", fmt_pct(last), delta=delta)

    if st.session_state.hist_probs_pct:
        st.subheader("Probabilidades por ejecución")
        n = len(st.session_state.hist_probs_pct)
        xs_idx = list(range(1, n + 1))

        ys_tdah = st.session_state.hist_probs_pct
        ys_no = st.session_state.hist_probs_no_pct if len(st.session_state.hist_probs_no_pct) == n else [100.0 - v for v in ys_tdah]

        width = 0.4
        lefts_tdah = [i - width/2 for i in xs_idx]
        lefts_no   = [i + width/2 for i in xs_idx]

        fig2, ax2 = plt.subplots()
        ax2.bar(lefts_tdah, ys_tdah, width, label="TDAH")
        ax2.bar(lefts_no, ys_no, width, label="No TDAH")
        ax2.set_xlabel("Cálculo #")
        ax2.set_ylabel("Probabilidad (%)")
        ax2.set_ylim(0, 100)
        ax2.set_xticks(xs_idx); ax2.set_xticklabels([str(i) for i in xs_idx])
        ax2.legend()

        for x, v in zip(lefts_tdah, ys_tdah):
            ax2.text(x, v + 1, f"{v:.2f}%", ha="center", va="bottom")
        for x, v in zip(lefts_no, ys_no):
            ax2.text(x, v + 1, f"{v:.2f}%", ha="center", va="bottom")

        st.pyplot(fig2)

    with st.expander("Ver historial de entradas"):
        if st.session_state.hist_inputs:
            st.write(f"Hay {len(st.session_state.hist_inputs)} registros.")
            st.json(st.session_state.hist_inputs)
        else:
            st.info("Aún no has registrado cálculos.")
# ======= TAB 3: MODELO =======
with tab_model:
    st.subheader("Árbol de decisión del experimento (EDA completo)")
    colL, colR = st.columns([2, 1])

    with colL:
        st.caption(
            "Este árbol es un **artefacto estático** para explicar qué variables se consideraron "
            "en el experimento con todas las variables del EDA (no sólo las 12 del backend)."
        )

        # Cargar imagen desde disco (ruta robusta con pathlib)
        if TREE_IMAGE_PATH.exists():
            st.image(str(TREE_IMAGE_PATH), caption="🌳 Árbol de decisión entrenado (sin fuga de información)", use_container_width=True)
        else:
            st.warning(f"No encontré la imagen del árbol en: {TREE_IMAGE_PATH}")
            uploaded = st.file_uploader("Sube aquí la imagen del árbol (png/jpg/svg)", type=["png", "jpg", "jpeg", "svg"])
            if uploaded is not None:
                st.image(uploaded, caption="🌳 Árbol de decisión (imagen subida)", use_column_width=True)


    with colR:
        st.markdown("### Variables usadas por el **backend** (12)")
        st.write(", ".join(BACKEND_VARS_12))
        # (Opcional) listado completo si dejaste un txt