# get_versions.py
import sys, platform, importlib
from importlib.metadata import version, PackageNotFoundError

# Paquetes que usas en tu app (backend + frontend)
# izquierda: nombre para import; derecha: nombre del paquete/distribución (pip)
TARGETS = {
    # Backend (FastAPI)
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "pydantic": "pydantic",
    "pandas": "pandas",
    "joblib": "joblib",
    "sklearn": "scikit-learn",         # OJO: import sklearn, paquete scikit-learn
    "numpy": "numpy",

    # Frontend (Streamlit)
    "streamlit": "streamlit",
    "requests": "requests",
    "matplotlib": "matplotlib",

    # Si estuvieras en Colab y quieres reportarlo
    "google.colab": "google-colab",
}

def get_import_version(import_name):
    try:
        m = importlib.import_module(import_name)
        return getattr(m, "__version__", "N/D")
    except Exception:
        return "No importable"

def get_dist_version(dist_name):
    try:
        return version(dist_name)
    except PackageNotFoundError:
        return "No instalado"

def main():
    print("=== Entorno ===")
    print("Python:", sys.version.split()[0])
    print("Sistema:", platform.platform())
    print()

    print("=== Versiones de paquetes ===")
    for import_name, dist_name in TARGETS.items():
        imp_v = get_import_version(import_name)
        dist_v = get_dist_version(dist_name)
        print(f"{import_name:14} (import): {imp_v:12} | {dist_name:15} (pip): {dist_v}")

    # Extra útil: crear un CSV con nombre, versión (pip)
    try:
        import csv
        with open("versions_report.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["component", "import_name", "pip_package", "import_version", "pip_version"])
            for import_name, dist_name in TARGETS.items():
                w.writerow(["backend/frontend", import_name, dist_name,
                            get_import_version(import_name), get_dist_version(dist_name)])
        print("\nArchivo generado: versions_report.csv")
    except Exception as e:
        print("\n(No se pudo crear CSV)", e)

if __name__ == "__main__":
    main()
