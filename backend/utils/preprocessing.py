import pandas as pd

# Diccionarios de mapeo
MAPEOS = {
    "conducta_status_num": {
        "Nunca diagnosticado": 0,
        "Diagnosticado pero ya no lo tiene": 1,
        "Alguna vez diagnosticado (estado actual desconocido)": 2,
        "Lo tiene actualmente, leve": 3,
        "Lo tiene actualmente, moderado": 4,
        "Lo tiene actualmente, grave": 5,
        "Diagnóstico actual, severidad desconocida": 6
    },
    "educacion_especial_status_num": {
        "Nunca ha tenido plan especial de educación": 0,
        "Tuvo plan, ya no recibe servicios de educación especial": 1,
        "Tiene plan y recibe servicios de educación especial": 2
    },
    "hcability_num": {
        "This child does not have any health conditions": 0,
        "Never": 1,
        "Sometimes": 2,
        "Usually": 3,
        "Always": 4
    },
    "k8q31_num": {
        "Never": 0,
        "Rarely": 1,
        "Sometimes": 2,
        "Usually": 3,
        "Always": 4
    },
    "k7q70_r_num": {
        "Never": 0,
        "Sometimes": 1,
        "Usually": 2,
        "Always": 3
    },
    "ansiedad_status_num": {
        "Nunca diagnosticado": 0,
        "Diagnosticado pero ya no lo tiene": 1,
        "Diagnosticado pero estado actual desconocido": 2,
        "Diagnosticado y lo tiene actualmente": 3
    },
    "k7q84_r_num": {
        "Never": 0,
        "Sometimes": 1,
        "Usually": 2,
        "Always": 3
    },
    "makefriend_num": {
        "A lot of difficulty": 0,
        "A little difficulty": 1,
        "No difficulty": 2
    },
    "sc_sex_bin": {
        "Female": 0,
        "Male": 1
    },
    "outdoorswkday_clean_num": {
        "Too young (<3 years)": 0,
        "Less than 1 hour per day": 1,
        "1 hour per day": 2,
        "2 hours per day": 3,
        "3 hours per day": 4,
        "4 or more hours per day": 5
    }
}

# Orden de columnas esperadas por el modelo
ORDERED_COLUMNS = [
    "conducta_status_num", "sc_age_years", "a1_age","educacion_especial_status_num",
    "hcability_num", "ansiedad_status_num", "k7q84_r_num",
    "k8q31_num", "k7q70_r_num", "makefriend_num","sc_sex_bin", 
    "outdoorswkday_clean_num"
]

def preprocess_user_input(data: dict) -> pd.DataFrame:
    """
    Transforma un diccionario de datos del usuario en un DataFrame codificado y ordenado
    para pasar al modelo.
    """
    # Mapear valores de texto a números
    data_num = {
        'conducta_status_num': MAPEOS['conducta_status_num'].get(data['conducta_status_num']),
        'sc_age_years': data['sc_age_years'],
        'a1_age': data['a1_age'],
        'educacion_especial_status_num': MAPEOS['educacion_especial_status_num'].get(data['educacion_especial_status_num']),
        'hcability_num': MAPEOS['hcability_num'].get(data['hcability_num']),
        'ansiedad_status_num': MAPEOS['ansiedad_status_num'].get(data['ansiedad_status_num']),
        'k8q31_num': MAPEOS['k8q31_num'].get(data['k8q31_num']),
        'k7q84_r_num': MAPEOS['k7q84_r_num'].get(data['k7q84_r_num']),
        'k7q70_r_num': MAPEOS['k7q70_r_num'].get(data['k7q70_r_num']),
        'makefriend_num': MAPEOS['makefriend_num'].get(data['makefriend_num']),
        'sc_sex_bin': MAPEOS['sc_sex_bin'].get(data['sc_sex_bin']),
        'outdoorswkday_clean_num': MAPEOS['outdoorswkday_clean_num'].get(data['outdoorswkday_clean_num'])
    }

    # Convertir a DataFrame en el orden correcto
    df = pd.DataFrame([data_num])
    df = df[ORDERED_COLUMNS]

    return df
