"""
generar_inventario.py
Genera el inventario exhaustivo de archivos de C:\Proyecto Semillero,
calculando hash SHA-256, tamaño en bytes, fecha de modificación,
conteo exacto de filas y columnas (para datasets tabulares), y clasificación de rol.

Salida: inventario_archivos.csv
"""

import os
import hashlib
from pathlib import Path
import datetime
import pandas as pd

ROOT = Path(r"C:\Proyecto Semillero")
OUT_CSV = ROOT / "inventario_archivos.csv"

# Clasificación funcional de archivos
ROLES = {
    # Crudos
    "Base Agrícola EVA 2007-2018_MADR.xlsx": ("Crudo", "Histórico agrícola municipal 2007-2018 (AgroNet)"),
    "Base EVA 2019-2024.xlsx": ("Crudo", "Histórico agrícola municipal 2019-2024 (UPRA / MADR)"),
    "Evaluaciones_Agropecuarias_Municipales_-_EVA._2019_-_2025._Base_Agrícola_20260829.csv": ("Crudo", "Descarga original Datos Abiertos EVA 2019-2025"),
    "ideam_normales_climatologicas_estaciones.csv": ("Crudo", "Normales Climatológicas Oficiales IDEAM (nsz2-kzcq)"),
    "oni_mensual_noaa.csv": ("Crudo", "Serie histórica mensual ONI 1950-2026 (NOAA CPC)"),
    
    # Intermedios
    "eva_2007_2024_consolidado.csv": ("Intermedio", "Dataset agrícola unificado 2007-2024 depurado (Fase 1)"),
    "eva_departamento_cultivo_anio.csv": ("Intermedio", "Serie agregada depto-cultivo-año para rendimiento potencial"),
    "eva_panel_balanceado.csv": ("Intermedio", "Subconjunto de municipios con reporte continuo 18 años"),
    "municipios_panel_final.csv": ("Intermedio", "Listado de los 191 municipios del panel balanceado"),
    "tabla_conteos_departamento_cultivo_anio.csv": ("Intermedio", "Matriz de completitud de reporte anual por departamento"),
    "municipios_coordenadas_altitud.csv": ("Intermedio", "Coordenadas DANE DIVIPOLA y altitud DEM de los 208 municipios"),
    "clima_mensual_municipios_2007_2024.csv": ("Intermedio", "Serie mensual de clima híbrido municipal (IDEAM + NASA POWER)"),
    "clima_anual_municipios_2007_2024.csv": ("Intermedio", "Serie anual de clima municipal con anomalías y rezagos"),
    "oni_anual_features.csv": ("Intermedio", "Variables anuales y fenológicas del índice ONI (1950-2025)"),
    "perfil_agronomico_cultivos.csv": ("Intermedio", "Metadatos ecofisiológicos de los 5 cultivos"),
    "eva_departamento_tendencia_potencial.csv": ("Intermedio", "Rendimiento potencial departamental suavizado (sin El Niño)"),
    
    # Finales / Modelado
    "dataset_maestro_entrenamiento.csv": ("Final", "Dataset maestro para modelado (18,425 filas x 114 columnas)"),
    "dataset_maestro_panel_balanceado.csv": ("Final", "Subconjunto de entrenamiento para panel balanceado"),
    "baselines_resultados_completos.csv": ("Final", "Métricas de evaluación de baselines (LOEO y ablación ENSO)"),
    "baselines_predicciones_test.csv": ("Final", "Predicciones individuales en test con residuos y metadata"),
    "baselines_feature_importances.csv": ("Final", "Importancias de características de Random Forest y XGBoost"),
    "baselines_curvas_aprendizaje.csv": ("Final", "Curvas de convergencia por iteración en XGBoost"),
    
    # Documentos
    "ESTADO_DEL_ARTE.md": ("Documento", "Revisión bibliográfica, antecedentes y análisis TimesFM 2.5"),
    "PLAN_DE_ACCION.md": ("Documento", "Plan metodológico y hoja de ruta consolidada de las 6 fases"),
    "HANDOFF.md": ("Documento", "Documento de transferencia y estado de verdad del proyecto"),
}


def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def contar_filas_columnas(filepath: Path):
    if filepath.suffix.lower() == ".csv":
        try:
            # Leer encabezado para columnas
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                first_line = f.readline()
                cols = len(first_line.rstrip("\r\n").split(","))
            # Contar filas totales con lectura eficiente
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                num_lines = sum(1 for _ in f)
            rows = max(0, num_lines - 1)  # Descontar encabezado
            return rows, cols
        except Exception:
            return None, None
    return None, None


def main():
    registros = []
    
    for item in sorted(ROOT.rglob("*")):
        if item.is_dir():
            continue
        # Excluir archivos temporales, git y pycache
        rel_str = str(item.relative_to(ROOT))
        if any(p in rel_str for p in [".git", "__pycache__", ".system_generated", "scratch"]):
            continue
        if item.name == "inventario_archivos.csv":
            continue

        size_bytes = item.stat().st_size
        mtime = datetime.datetime.fromtimestamp(item.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        sha = sha256_file(item)
        rows, cols = contar_filas_columnas(item)

        # Determinar rol y propósito
        if item.name in ROLES:
            rol, proposito = ROLES[item.name]
        elif item.suffix == ".py":
            rol = "Script"
            proposito = f"Script de procesamiento / modelado ({item.name})"
        elif "landing" in rel_str:
            rol = "Web/Landing"
            proposito = f"Componente de landing Tempero ({item.name})"
        elif "dashboard_tecnico" in rel_str:
            rol = "Web/Dashboard"
            proposito = f"Componente de Dashboard Técnico ({item.name})"
        else:
            rol = "Otro"
            proposito = item.name

        registros.append({
            "ruta_relativa": rel_str.replace("\\", "/"),
            "rol": rol,
            "proposito": proposito,
            "filas": rows if rows is not None else "",
            "columnas": cols if cols is not None else "",
            "tamano_bytes": size_bytes,
            "tamano_kb": round(size_bytes / 1024, 2),
            "sha256": sha,
            "fecha_modificacion": mtime,
        })

    df = pd.DataFrame(registros)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"Inventario generado exitosamente en: {OUT_CSV}")
    print(f"Total de archivos catalogados: {len(df)}")
    print(df[["ruta_relativa", "rol", "filas", "columnas", "tamano_kb"]].to_string())


if __name__ == "__main__":
    main()
