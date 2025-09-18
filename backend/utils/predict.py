import joblib
import pandas as pd
import os
import json
from .preprocessing import preprocess_user_input, ORDERED_COLUMNS, MAPEOS  # 👈 Importa la función de preprocesamiento
import numpy as np

#-------------------------------------------------------------------------
from typing import Dict, Any, List
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
#-------------------------------------------------------------------------

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

#-------------------------------------------------------------------------
from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier

from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier

def _get_final_estimator(obj):
    if isinstance(obj, Pipeline):
        return obj.named_steps.get("model", obj.steps[-1][1])
    return obj

def _unwrap_decision_tree(est) -> Optional[DecisionTreeClassifier]:
    if isinstance(est, DecisionTreeClassifier):
        return est
    if isinstance(est, Pipeline):
        for _, step in reversed(est.steps):
            if isinstance(step, DecisionTreeClassifier):
                return step
    if hasattr(est, "tree_") and isinstance(est, DecisionTreeClassifier):
        return est
    return None

def _align_columns(df, feature_names: List[str]):
    if feature_names is None:
        return df
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas por el árbol: {missing}")
    return df.reindex(columns=feature_names)

def _unique_path_nodes(node_indicator, sample_index: int, leaf_id: int) -> List[int]:
    """
    Devuelve la lista de nodos visitados (excluyendo la hoja), en orden,
    sin duplicados, para el sample dado.
    """
    start = node_indicator.indptr[sample_index]
    end = node_indicator.indptr[sample_index + 1]
    path_nodes = node_indicator.indices[start:end]  # ya viene ordenado

    seen = set()
    ordered_unique = []
    for nid in path_nodes:
        if nid == leaf_id:
            continue  # la hoja se reporta aparte
        if nid not in seen:
            ordered_unique.append(int(nid))
            seen.add(nid)
    return ordered_unique

def _extract_path_with_node_conf(tree: DecisionTreeClassifier, x_df, pred_label: int):
    """
    Construye:
      - pasos (sin duplicados, sin la hoja)
      - hoja (UNA sola vez)
      - ruta_lineal
    Probabilidades de nodo = proporción de clases en ese nodo (no calibradas).
    """
    if not hasattr(tree, "tree_"):
        raise TypeError("El estimador base no es un árbol de decisión entrenado.")

    feat_names = list(tree.feature_names_in_) if hasattr(tree, "feature_names_in_") else list(x_df.columns)
    x_aligned = _align_columns(x_df, feat_names)

    t = tree.tree_
    node_indicator = tree.decision_path(x_aligned)
    # Para una sola fila (índice 0)
    leaf_id = int(tree.apply(x_aligned)[0])

    # Nodos únicos del camino (orden correcto, sin hoja)
    path_nodes = _unique_path_nodes(node_indicator, sample_index=0, leaf_id=leaf_id)

    pasos = []
    cadena = []

    for node_id in path_nodes:
        feat_idx = int(t.feature[node_id])
        thr = float(t.threshold[node_id])

        # Si fuera un nodo terminal raro (feature=-2), lo saltamos
        if feat_idx < 0:
            continue

        feat_name = feat_names[feat_idx]
        user_val = float(x_aligned.iloc[0, feat_idx])
        decision = "<=" if user_val <= thr else ">"

        counts = t.value[node_id][0]
        total = counts.sum() if counts.sum() > 0 else 1.0
        p_no = float(counts[0] / total)
        p_si = float(counts[1] / total)
        conf_node = float(p_si if pred_label == 1 else p_no)

        pasos.append({
            "node_id": int(node_id),
            "feature": feat_name,
            "threshold": thr,
            "user_value": user_val,
            "decision": decision,
            "node_probs": {"NoTDAH": p_no, "TDAH": p_si},
            "node_confidence_for_pred": conf_node
        })
        cadena.append(f"{feat_name} ({user_val} {decision} {thr})")

    # Hoja (una sola vez, NO incluida en pasos)
    counts_leaf = t.value[leaf_id][0]
    total_leaf = counts_leaf.sum() if counts_leaf.sum() > 0 else 1.0
    p_no_leaf = float(counts_leaf[0] / total_leaf)
    p_si_leaf = float(counts_leaf[1] / total_leaf)

    hoja = {
        "node_id": leaf_id,
        "node_probs": {"NoTDAH": p_no_leaf, "TDAH": p_si_leaf},
        "node_confidence_for_pred": float(p_si_leaf if pred_label == 1 else p_no_leaf)
    }

    return {
        "pasos": pasos,
        "hoja": hoja,
        "ruta_lineal": (" → ".join(cadena) + " → [leaf]") if cadena else "[leaf]"
    }

def _get_estimator_from_calibrated(cal_clf):
    # Compatibilidad entre versiones de sklearn
    for attr in ("estimator", "base_estimator", "classifier", "clf"):
        est = getattr(cal_clf, attr, None)
        if est is not None:
            return est
    return None

def explicar_ruta_lineal(data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    JSON final sin duplicados:
      - prediccion y probas calibradas (finales)
      - confidence final
      - source (fold más confiado / prefit / raw)
      - explicacion: ruta_lineal, pasos[], hoja{}
    """
    x_df = preprocess_user_input(data_dict)

    proba_final = model.predict_proba(x_df)[0]
    pred = int(model.predict(x_df)[0])
    confidence_final = float(max(proba_final))

    final_est = _get_final_estimator(model)
    source = {}
    tree_to_use = None
    x_for_tree = x_df

    if isinstance(final_est, CalibratedClassifierCV) and hasattr(final_est, "calibrated_classifiers_") and final_est.calibrated_classifiers_:
        best = {"fold": None, "conf": -1.0, "proba": None, "tree": None, "x": x_df, "cal": None}
        for i, cal_clf in enumerate(final_est.calibrated_classifiers_):
            est = _get_estimator_from_calibrated(cal_clf)
            tree = _unwrap_decision_tree(est)
            if tree is None:
                continue

            x_fold = _align_columns(x_df, list(tree.feature_names_in_)) if hasattr(tree, "feature_names_in_") else x_df
            p_fold = cal_clf.predict_proba(x_fold)[0]
            conf_fold = float(p_fold[pred])

            if conf_fold > best["conf"]:
                best.update({"fold": i, "conf": conf_fold, "proba": p_fold, "tree": tree, "x": x_fold, "cal": cal_clf})

        if best["tree"] is None:
            return {
                "prediccion": pred,
                "probabilidades": {"NoTDAH": float(proba_final[0]), "TDAH": float(proba_final[1])},
                "confidence": confidence_final,
                "error": "No se encontró un árbol dentro de los calibradores."
            }

        tree_to_use = best["tree"]
        x_for_tree = best["x"]
        source = {
            "type": "calibrated_fold_mas_confiado",
            "fold": int(best["fold"]),
            "fold_probabilities": {  # probas calibradas del fold elegido
                "NoTDAH": float(best["proba"][0]),
                "TDAH": float(best["proba"][1]),
            }
        }

    elif isinstance(final_est, CalibratedClassifierCV):
        est = getattr(final_est, "base_estimator", None) or getattr(final_est, "estimator", None)
        tree_to_use = _unwrap_decision_tree(est)
        if tree_to_use is None:
            return {
                "prediccion": pred,
                "probabilidades": {"NoTDAH": float(proba_final[0]), "TDAH": float(proba_final[1])},
                "confidence": confidence_final,
                "error": "No se pudo extraer el árbol del calibrador (prefit)."
            }
        source = {"type": "calibrated_prefit_base_estimator"}

    else:
        tree_to_use = _unwrap_decision_tree(final_est)
        if tree_to_use is None:
            return {
                "prediccion": pred,
                "probabilidades": {"NoTDAH": float(proba_final[0]), "TDAH": float(proba_final[1])},
                "confidence": confidence_final,
                "error": "El estimador final no contiene un árbol de decisión reconocible."
            }
        source = {"type": "raw_decision_tree"}

    ruta = _extract_path_with_node_conf(tree_to_use, x_for_tree, pred_label=pred)
# --- NUEVO: añadir calibradas del fold y finales a la hoja ---
    # Probabilidad calibrada del fold elegido (coherente con la ruta)
    if "cal" in best and best["cal"] is not None:
        p_fold = best["proba"]  # ya la calculaste
        ruta["hoja"]["calibrated_probabilities_fold"] = {
            "NoTDAH": float(p_fold[0]),
            "TDAH": float(p_fold[1]),
        }
        ruta["hoja"]["conf_calibrated_fold_for_pred"] = float(p_fold[pred])

    # Probabilidad final (promedio de folds) = lo que devuelve model.predict_proba
    ruta["hoja"]["calibrated_probabilities_final"] = {
        "NoTDAH": float(proba_final[0]),
        "TDAH": float(proba_final[1]),
    }
    ruta["hoja"]["conf_calibrated_final_for_pred"] = float(max(proba_final))        
    return {
        "prediccion": pred,
        "probabilidades": {"NoTDAH": float(proba_final[0]), "TDAH": float(proba_final[1])},
        "confidence": confidence_final,
        "explicacion": {
            "source": source,
            "ruta_lineal": ruta["ruta_lineal"],
            "pasos": ruta["pasos"],
            "hoja": ruta["hoja"]
        }
    }