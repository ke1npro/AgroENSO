# AgroENSO: Red Neuronal Modular de Doble Rama para Predicción del Impacto Agroclimatológico de El Niño en Colombia (2007–2024)

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-orange)](https://huggingface.co/datasets/semillero-agro/colombia-agroclimatic-enso-eva)

Este repositorio contiene el código fuente, la suite experimental y las aplicaciones interactivas del proyecto de investigación enfocado en modelar la pérdida o ganancia de eficiencia agrícola municipal ante eventos extremos de **El Niño Oscilación del Sur (ENSO)** en Colombia.

---

## 📌 Aspectos Destacados del Proyecto

- **Panel Multirregional Continuo (2007–2024):** 18 años de historia agrícola oficial (EVA MADR / UPRA), meteorología híbrida (IDEAM + NASA POWER) e índices macroclimáticos (NOAA ONI).
- **Cobertura Territorial (10 Departamentos, 29,622 observaciones):**
  - *Región Andina:* Antioquia, Huila, Tolima (18,425 registros).
  - *Región Caribe:* Atlántico, Bolívar, Cesar, Córdoba, La Guajira, Magdalena, Sucre (11,197 registros).
- **Especies Analizadas:** Café, Cacao, Fríjol, Maíz Tecnificado y Maíz Tradicional (clasificados y tratados de forma independiente).
- **Validación Rigurosa Leave-One-Event-Out (LOEO):** Evaluación sin fuga temporal aislando los dos mayores eventos de El Niño del siglo XXI:
  - *Split 1:* El Niño Severo 2015–2016.
  - *Split 2:* El Niño Reciente 2023–2024.
- **Arquitectura Neuronal Modular (FiLM + MC Dropout):**
  - **Rama 1:** Contexto Zonal (altitud, coordenadas, normales IDEAM) y Ecofisiología del cultivo (C3/C4, óptimos térmicos e hídricos).
  - **Rama 2:** Clima Local (lluvia, temperatura, anomalías, rezagos) y Señal ENSO (ONI, rachas, fenología de eventos).
  - **Modulación Contextual FiLM:** $\gamma = 1 + \tanh(W_g h_{\text{ctx}} + b_g)$, $\beta = W_b h_{\text{ctx}} + b_b$, $z = \gamma \odot h_{\text{clim}} + \beta$.
  - **Incertidumbre Epistémica:** Inferencia estocástica con Monte Carlo Dropout ($S = 30$ pases), logrando una cobertura empírica PICP del 94.7% en intervalos al 95%.

---

## 🗂️ Estructura del Repositorio

El repositorio está organizado en carpetas maestras con clara separación de responsabilidades:

```text
├── Dataset/                   # Gestionado en Hugging Face Datasets (ignorado en Git)
│   ├── raw/                   # Excels originales MADR EVA y Socrata IDEAM
│   ├── clean/                 # Matrices maestras procesadas (0 NaNs en predictores)
│   └── README.md              # Tarjeta de metadatos oficial de Hugging Face
│
├── Scripts/                   # Pipeline canónico y scripts de modelado reproducible
│   ├── paths.py               # Resolución dinámica de rutas del proyecto
│   ├── pipeline_canonico.py   # Consolidación determinista de fuentes EVA A y B
│   ├── obtencion_clima.py     # Hibridación climática NASA POWER + Normales IDEAM
│   ├── descarga_oni_noaa.py   # Extracción de la serie histórica del índice ONI
│   ├── evaluar_baselines_fase4.py # Benchmarking LOEO (Ridge, Huber, RF, XGBoost)
│   ├── prototipo_red_modular_fase5.py # Red Modular FiLM en PyTorch
│   ├── auditoria_puntos_2_3_4.py # Verificación estricta de integridad y cero NaNs
│   └── preparar_huggingface.py # Gestor de sincronización con Hugging Face Hub
│
├── Resultados/                # Tablas consolidadas de predicciones y métricas
│   ├── baselines_resultados_completos.csv
│   ├── baselines_predicciones_test.csv
│   ├── red_modular_comparativa_loeo.csv
│   └── red_modular_curvas_entrenamiento.csv
│
├── Experimental/              # Iteraciones rápidas y experimentos secundarios
│   ├── ejecutar_experimentos.py # Runner de Antioquia vs Macro-regiones
│   ├── resultados_simplificados.csv # Métricas de las 3 iteraciones
│   └── index.html             # Visor web de resultados experimentales
│
├── dashboard_tecnico/         # Visor web SPA de resultados (HTML5 / Chart.js)
│   ├── index.html             # Dashboard interactivo con 4 pestañas de análisis
│   ├── exclusivo.html         # Mapa de custodia y gobernanza Git vs Hugging Face
│   ├── css/dashboard.css      # Hoja de estilos con paleta técnica oscura
│   └── js/dashboard.js        # Lógica de renderizado dinámico y filtros
│
├── HANDOFF.md                 # Estado de verdad no negociable y registro de decisiones
├── PLAN_DE_ACCION.md          # Plan metodológico consolidado de las 6 fases
├── requirements.txt           # Dependencias exactas de Python
└── .gitignore                 # Exclusión de binarios y datos pesados
```

---

## 🚀 Guía de Reproducción Rápida

### 1. Clonar el repositorio
```bash
git clone https://github.com/semillero-agroenso/modelo-predictivo.git
cd modelo-predictivo
```

### 2. Configurar el entorno virtual e instalar dependencias
```bash
python -m venv venv
# En Windows:
venv\Scripts\activate
# En Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Hidratar los datasets desde Hugging Face
```bash
python Scripts/preparar_huggingface.py --descargar
```

### 4. Auditar la integridad del dataset
```bash
python Scripts/auditoria_puntos_2_3_4.py
```

### 5. Ejecutar la suite experimental o abrir los visores
```bash
# Ejecutar iteración rápida:
python Experimental/ejecutar_experimentos.py

# Ver los resultados en el navegador:
# Abrir dashboard_tecnico/index.html o Experimental/index.html
```

---

## 📊 Resumen de Resultados Principales (Promedio LOEO)

| Ámbito Geográfico | Modelo | RMSE Test | MAE Test | R² Test | Pearson $r$ | Cobertura PICP 95% |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Antioquia Exclusivo** | XGBoost Regressor | 0.3002 | 0.2233 | **0.2398** | 0.5048 | N/A |
| **TODAS las Regiones (10 Dptos)** | XGBoost Regressor | 0.3062 | 0.2314 | 0.0701 | 0.3430 | N/A |
| **TODAS las Regiones (10 Dptos)** | **Red Modular FiLM (Ours)** | **0.3033** | **0.2215** | **0.0932** | 0.3375 | **94.67%** |

---

## 👥 Gobernanza de Datos y Exclusividad
Consulte [dashboard_tecnico/exclusivo.html](file:///c:/Proyecto%20Semillero/dashboard_tecnico/exclusivo.html) para conocer la separación de artefactos entre Git (código ligero y reproducible) y Hugging Face Datasets (matrices de datos crudos e intermedios).
