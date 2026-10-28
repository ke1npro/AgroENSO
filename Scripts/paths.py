"""
paths.py - Módulo de resolución de rutas dinámicas del proyecto.
Permite ejecutar scripts tanto desde la raíz del proyecto como desde la subcarpeta Scripts/
o cualquier subdirectorio sin romper las referencias relativas ni requerir rutas absolutas cableadas.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "Dataset" / "raw"
CLEAN_DIR = ROOT_DIR / "Dataset" / "clean"
RESULTS_DIR = ROOT_DIR / "Resultados"
SCRIPTS_DIR = ROOT_DIR / "Scripts"
EXPERIMENTAL_DIR = ROOT_DIR / "Experimental"
DASHBOARD_DIR = ROOT_DIR / "dashboard_tecnico"


def resolve_path(filename: str, create_dir_if_missing: bool = False) -> Path:
    """
    Resuelve la ubicación de un archivo en el proyecto.
    Si el archivo ya existe en alguna de las carpetas estándar, retorna su ruta exacta.
    Si no existe y parece un resultado (baselines_*, red_modular_*, *resultados*, *predicciones*),
    lo direcciona a Resultados/.
    Si parece un dato crudo (*EVA*.xlsx, *Base_Agricola*, ideam_normales*, oni_mensual*),
    lo direcciona a Dataset/raw/.
    En cualquier otro caso, lo direcciona a Dataset/clean/.
    """
    p = Path(filename)
    if p.is_absolute():
        return p

    # Búsqueda en rutas conocidas
    search_dirs = [CLEAN_DIR, RAW_DIR, RESULTS_DIR, DASHBOARD_DIR, ROOT_DIR, EXPERIMENTAL_DIR]
    for d in search_dirs:
        candidate = d / filename
        if candidate.exists():
            return candidate

    # Destino por convención para nuevos archivos
    fn_lower = filename.lower()
    if any(k in fn_lower for k in ["baseline", "red_modular", "resultados", "prediccion", "curvas", "importances"]):
        target_dir = RESULTS_DIR
    elif any(k in fn_lower for k in ["raw", ".xlsx", "base_agricola", "normales_climatologicas", "oni_mensual"]):
        target_dir = RAW_DIR
    elif "dashboard" in fn_lower:
        target_dir = DASHBOARD_DIR
    else:
        target_dir = CLEAN_DIR

    if create_dir_if_missing:
        target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / filename
