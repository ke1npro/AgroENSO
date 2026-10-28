"""
Scripts/preparar_huggingface.py
================================
Gestor de preparación, documentación y sincronización con Hugging Face Datasets.

Funcionalidades:
  1. Genera la Dataset Card oficial (Dataset/README.md) con metadatos YAML conformes a HF Datasets.
  2. Ofrece comando para verificar la completitud de los datos locales (--verificar).
  3. Ofrece comando para subir los datasets crudos y limpios al Hugging Face Hub (--subir).
  4. Ofrece comando para descargar e hidratar el proyecto desde Hugging Face (--descargar).

Uso:
  python Scripts/preparar_huggingface.py --verificar
  python Scripts/preparar_huggingface.py --subir --repo_id "usuario/colombia-agroclimatic-enso-eva"
  python Scripts/preparar_huggingface.py --descargar --repo_id "usuario/colombia-agroclimatic-enso-eva"
"""

import os
import sys
import argparse
from pathlib import Path

# Soporte para ejecución desde raíz o desde Scripts/
try:
    from paths import ROOT_DIR, RAW_DIR, CLEAN_DIR
except ImportError:
    from Scripts.paths import ROOT_DIR, RAW_DIR, CLEAN_DIR

DATASET_CARD_CONTENT = """---
language:
- es
license: cc-by-4.0
task_categories:
- tabular-regression
tags:
- climate
- agriculture
- enso
- el-nino
- colombia
- crop-yield
- deep-learning
size_categories:
- 10K<n<100K
---

# Dataset Agroclimatológico Multirregional de Colombia (2007–2024)
## Rendimiento Agrícola, Teleconexiones ENSO (NOAA ONI) y Clima Local Híbrido (IDEAM + NASA POWER)

Este repositorio contiene los datos oficiales, preprocesados y enriquecidos del **Semillero de Investigación en Inteligencia Artificial y Agroclimatología**.

### Descripción General
- **Rango temporal:** 2007 – 2024 (18 años continuos).
- **Cobertura geográfica:** 10 departamentos de Colombia (Región Andina: Antioquia, Huila, Tolima; Región Caribe: Atlántico, Bolívar, Cesar, Córdoba, La Guajira, Magdalena, Sucre).
- **Especies evaluadas:** Café, Cacao, Fríjol, Maíz Tecnificado y Maíz Tradicional.
- **Volumen consolidado:** 29,622 observaciones a nivel municipal-anual con 115 columnas de atributos biofísicos, ecofisiológicos y macroclimáticos.

### Estructura de Archivos en el Hub

```text
Dataset/
├── raw/                               # Archivos originales de fuentes oficiales
│   ├── Base Agrícola EVA 2007-2018_MADR.xlsx
│   ├── Base EVA 2019-2024.xlsx
│   ├── Evaluaciones_Agropecuarias_Municipales_–_EVA._2019_-_2025.csv
│   ├── ideam_normales_climatologicas_estaciones.csv
│   ├── ideam_normales_climatologicas_estaciones_caribe.csv
│   └── oni_mensual_noaa.csv
└── clean/                             # Tablas maestras e intermedias auditadas (0 NaNs en predictores)
    ├── dataset_maestro_andina_caribe.csv   (29,622 filas x 115 cols - Macro-regional)
    ├── dataset_maestro_entrenamiento.csv  (18,425 filas x 114 cols - Andina)
    ├── dataset_maestro_entrenamiento_caribe.csv (11,197 filas x 114 cols - Caribe)
    ├── clima_anual_municipios_2007_2024.csv
    ├── clima_mensual_municipios_2007_2024.csv
    ├── municipios_coordenadas_altitud.csv
    ├── oni_anual_features.csv
    └── perfil_agronomico_cultivos.csv
```

### Arquitectura de Columnas del Dataset Maestro (115 columnas)
1. **Identificadores y Llaves:** `cod_dane_mun`, `cod_dane_dep`, `municipio`, `departamento`, `año`, `cultivo_cat`.
2. **Variables de Salida (Targets):**
   - `rendimiento`: Rendimiento agrícola observado (t/ha).
   - `rendimiento_potencial_dep`: Media móvil centrada 5 años sin eventos El Niño severos.
   - `eficiencia`: Ratio rendimiento / potencial (máx 7.22, mediana 0.93).
   - `eficiencia_clip`: Target principal de entrenamiento, winsorizada al percentil 0.5% - 99.5% [0.248, 2.237].
3. **Rama 1 - Contexto Biofísico y Cultivo (26 predictores):**
   - Coordenadas DANE, altitud DEM, normales IDEAM 1981-2010.
   - Perfil ecofisiológico FAO/AGRIS: rangos óptimos de temperatura y precipitación, C3/C4, sensibilidades hídricas.
4. **Rama 2 - Señal Macroclimática y Clima Local (59 predictores):**
   - Índice ONI mensual y anual NOAA CPC, rachas de calentamiento, semestres y trimestres críticos.
   - Clima híbrido municipal (NASA POWER calibrado con normales IDEAM), anomalías y z-scores.
5. **Estrés e Interacciones Explícitas (6 predictores):**
   - Déficit y exceso hídrico, estrés térmico diurno/nocturno, interacción ONI x Anomalía de Lluvia.

### Particiones Canónicas de Validación (Leave-One-Event-Out - LOEO)
- **Split 1:** Test en El Niño Severo 2015–2016 (Entrenamiento en resto de años).
- **Split 2:** Test en El Niño Reciente 2023–2024 (Entrenamiento en 2007–2022).
- **Split 3:** Test secuencial cuatrienal 2021–2024 (Entrenamiento en 2007–2020).

### Citación
```bibtex
@misc{colombia_agroenso_2026,
  author = {Semillero de Investigación IA y Agroclimatología},
  title = {Dataset Agroclimatológico Multirregional y Teleconexiones ENSO en Colombia (2007-2024)},
  year = {2026},
  publisher = {Hugging Face Datasets},
  howpublished = {https://huggingface.co/datasets/semillero-agro/colombia-agroclimatic-enso-eva}
}
```
"""


def generar_dataset_card():
    card_path = ROOT_DIR / "Dataset" / "README.md"
    card_path.parent.mkdir(parents=True, exist_ok=True)
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(DATASET_CARD_CONTENT.strip() + "\n")
    print(f"Dataset Card generada exitosamente en: {card_path}")


def verificar_archivos_locales():
    print("=" * 80)
    print("VERIFICACIÓN DE DATASETS LOCALES (RAW Y CLEAN)")
    print("=" * 80)

    raw_esperados = [
        "ideam_normales_climatologicas_estaciones.csv",
        "ideam_normales_climatologicas_estaciones_caribe.csv",
        "oni_mensual_noaa.csv"
    ]
    clean_esperados = [
        "dataset_maestro_andina_caribe.csv",
        "dataset_maestro_entrenamiento.csv",
        "dataset_maestro_entrenamiento_caribe.csv",
        "clima_anual_municipios_2007_2024.csv",
        "clima_mensual_municipios_2007_2024.csv",
        "municipios_coordenadas_altitud.csv",
        "oni_anual_features.csv",
        "perfil_agronomico_cultivos.csv"
    ]

    print("\n[Directorio Dataset/raw/]")
    for fn in raw_esperados:
        p = RAW_DIR / fn
        if p.exists():
            print(f"  [OK] {fn} ({p.stat().st_size / 1024:.1f} KB)")
        else:
            print(f"  [--] {fn} (NO ENCONTRADO)")

    print("\n[Directorio Dataset/clean/]")
    for fn in clean_esperados:
        p = CLEAN_DIR / fn
        if p.exists():
            print(f"  [OK] {fn} ({p.stat().st_size / (1024*1024):.2f} MB)")
        else:
            print(f"  [--] {fn} (NO ENCONTRADO)")


def subir_a_huggingface(repo_id: str, token: str = None):
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Error: Requiere `pip install huggingface_hub`.")
        sys.exit(1)

    print(f"\nIniciando subida hacia Hugging Face Hub: {repo_id}...")
    api = HfApi(token=token)

    # Crear repo si no existe
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    dataset_folder = ROOT_DIR / "Dataset"
    print(f"Subiendo carpeta completa: {dataset_folder}")
    api.upload_folder(
        folder_path=str(dataset_folder),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Subida oficial datasets raw y clean (Andina + Caribe 2007-2024)"
    )
    print("Subida completada con éxito.")


def descargar_de_huggingface(repo_id: str, token: str = None):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: Requiere `pip install huggingface_hub`.")
        sys.exit(1)

    print(f"\nDescargando datasets desde {repo_id} hacia Dataset/...")
    dataset_folder = ROOT_DIR / "Dataset"
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dataset_folder),
        token=token
    )
    print("Descarga e hidratación completada con éxito.")


def main():
    parser = argparse.ArgumentParser(description="Gestor de sincronización con Hugging Face Datasets")
    parser.add_argument("--verificar", action="store_true", help="Verifica archivos locales raw y clean")
    parser.add_argument("--subir", action="store_true", help="Sube los datasets a Hugging Face Hub")
    parser.add_argument("--descargar", action="store_true", help="Descarga los datasets desde Hugging Face Hub")
    parser.add_argument("--repo_id", type=str, default="semillero-agro/colombia-agroclimatic-enso-eva", help="ID del dataset en Hugging Face")
    parser.add_argument("--token", type=str, default=None, help="Token HF_TOKEN de acceso (opcional si ya hizo login)")

    # Siempre regenerar la card actualizada
    generar_dataset_card()

    args = parser.parse_args()

    if args.verificar or (not args.subir and not args.descargar):
        verificar_archivos_locales()

    if args.subir:
        subir_a_huggingface(args.repo_id, args.token)

    if args.descargar:
        descargar_de_huggingface(args.repo_id, args.token)


if __name__ == "__main__":
    main()
