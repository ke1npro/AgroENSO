# HANDOFF: Estado del Proyecto, Decisiones y Transferencia Tecnica

Fecha de actualizacion verificada: `2026-11-07` [VERIFICADO: reloj del sistema y ejecucion de scripts].

## 1. Resumen Ejecutivo
El proyecto **AgroENSO** desarrolla y valida un marco computacional neuro-biofisico para predecir el impacto de la variabilidad climatica local y el ciclo ENSO (El Nino-Oscilacion del Sur) sobre la eficiencia de rendimiento agricola municipal en Colombia [VERIFICADO: PLAN_DE_ACCION.md]. A la fecha se encuentran consolidados 18 anos continuos de Evaluaciones Agropecuarias Municipales (EVA 2007-2024) con **29,622 registros validos** en **400 municipios** de **10 departamentos** (3 de la Region Andina: Antioquia, Huila y Tolima con 18,425 filas; 7 de la Region Caribe: Atlantico, Bolivar, Cesar, Cordoba, La Guajira, Magdalena y Sucre con 11,197 filas), acoplados con climatologia hibrida municipal (IDEAM + DEM SRTM + NASA POWER 2006-2024) y el indice mensual ONI de NOAA [VERIFICADO: Dataset/clean/dataset_maestro_andina_caribe.csv].

Se completaron cuatro familias de experimentos bajo validacion temporal estricta **Leave-One-Event-Out (LOEO)** sobre los dos episodios canonicos fuertes de El Nino (`2015-2016` y `2023-2024`):
1. **Baselines clasicos (Fase 4 / Sprint 7-8):** Ridge, Huber, Random Forest y XGBoost en Region Andina, Region Caribe y consolidado Macro-Regional [VERIFICADO: Resultados/baselines_*.csv].
2. **Red Neuronal Modular de Doble Rama en PyTorch (Fase 5 / Sprint 8):** Comparacion controlada entre compuerta Sigmoide y modulacion afín FiLM ($\gamma = 1 + \tanh, \beta$) con cuantificacion de incertidumbre mediante Monte Carlo Dropout [VERIFICADO: Scripts/prototipo_red_modular_fase5.py, Resultados/red_modular_comparativa_loeo.csv].
3. **Estudio paralelo multi-escala y radiografia de cultivos recomendados:** Evaluacion comparativa desde micro-regiones (`Antioquia`, `Cesar`, `Sierra Nevada de Santa Marta`) hasta el consolidado de 10 departamentos (`N = 29,622`), junto con el censo empirico en EVA cruda para *Palma de aceite*, *Arroz*, *Algodon*, *Cafe de la Sierra* y *Ganaderia* [VERIFICADO: Experimental/estudio_paralelo_recomendaciones.py].
4. **Ablacion de escalamiento neuronal y mejoras causales libres de fuga (Sprint 8-9):** Se demostro empiricamente que escalar por fuerza bruta el ancho/profundidad de la red (`hidden_dim` de `32` a `512`, hasta `1,765,889` parametros) apenas eleva el $R^2$ macro-regional de `0.0814` a `0.1153` por saturacion ante variables omitidas. En contraste, una red liviana (`~57k` parametros) con alineacion al semestre de cosecha activo (`A`/`B`), *Entity Embeddings* municipales (`cod_dane_mun`) y aprendizaje residual sobre un prior estructural Empirical Bayes estrictamente causal (`M3`) eleva el $R^2$ macro-regional a **`0.2395`** (`RMSE = 0.2775`, `MAE = 0.1991`, `PICP 95% = 95.63%`) y en Region Andina a **`0.2872`** (`RMSE = 0.2569`, `MAE = 0.1835`, `MAE agregado cultivo-ano = 0.0310`), superando ampliamente a `XGBoost` incluso cuando este recibe exactamente las mismas variables enriquecidas (`R^2 = 0.2395 vs 0.1370` en Macro; `0.2872 vs 0.2188` en Andina) [VERIFICADO: Experimental/ablacion_dimensiones_y_mejoras.py, Experimental/mejoras_precision_loeo.csv].

El repositorio fue modularizado en carpetas maestras (`Dataset/`, `Scripts/`, `Resultados/`, `Experimental/`, `dashboard_tecnico/`), versionado en GitHub (`https://github.com/ke1npro/AgroENSO`) y publicado en vivo mediante GitHub Pages (`https://ke1npro.github.io/AgroENSO/`), manteniendo los datos pesados segregados para custodia en Hugging Face Datasets [VERIFICADO: git status, preparar_huggingface.py].

---

## 2. Identidad, Alcance y Alineacion con el Cronograma Oficial
* **Titulo preliminar:** Modelo Neuronal Modular de Prediccion de Impacto Agricola por Ciclo ENSO en Colombia [ORIGEN POR CONFIRMAR].
* **Objetivo general:** Cuantificar y predecir la eficiencia de rendimiento agricola municipal condicionada por anomalias oceanicas globales (ENSO/ONI) e interacciones biofisicas y edafoclimaticas locales [VERIFICADO: PLAN_DE_ACCION.md].
* **Cultivos modelados en el nucleo canonico (5 categorias):** Cafe, Cacao, Frijol, Maiz tradicional y Maiz tecnificado (separados explicitamente por brecha tecnologica) [VERIFICADO: Dataset/clean/dataset_maestro_andina_caribe.csv].
* **Cobertura geografica consolidada (10 departamentos, 400 municipios):**
  * **Region Andina (3 departamentos, 208 municipios, 18,425 registros):** Antioquia (124), Tolima (47), Huila (37) [VERIFICADO: Dataset/clean/dataset_maestro_entrenamiento.csv].
  * **Region Caribe (7 departamentos, 192 municipios, 11,197 registros):** Atlantico, Bolivar, Cesar, Cordoba, La Guajira, Magdalena y Sucre [VERIFICADO: Dataset/clean/dataset_maestro_entrenamiento_caribe.csv].
* **Cobertura temporal:** 2007 a 2024 (18 anos agricolas continuos; incluye 2006 en series climaticas y ONI para rezagos observados $t-1$ sin perdida de muestra) [VERIFICADO: Dataset/clean/clima_anual_municipios_2007_2024.csv].
* **Unidad de analisis:** Registro `Municipio - Cultivo - Ano - Periodo (A/B/Anual)` [VERIFICADO: Dataset/clean/dataset_maestro_andina_caribe.csv].
* **Repositorio oficial GitHub:** `https://github.com/ke1npro/AgroENSO` (rama `master`) [VERIFICADO: git remote -v].
* **Despliegue interactivo GitHub Pages:**
  * Raiz / Dashboard principal: `https://ke1npro.github.io/AgroENSO/` y `https://ke1npro.github.io/AgroENSO/dashboard_tecnico/index.html` [VERIFICADO: HTTP 200].
  * Mapa de custodia Git vs Hugging Face: `https://ke1npro.github.io/AgroENSO/dashboard_tecnico/exclusivo.html` [VERIFICADO: HTTP 200].
  * Laboratorio Experimental simplificado y ablaciones: `https://ke1npro.github.io/AgroENSO/Experimental/index.html` [VERIFICADO: HTTP 200].
* **Revista candidata:** *Ingenieria y Competitividad* (Universidad del Valle) [ORIGEN POR CONFIRMAR, no decidida formalmente].

### 2.1 Mapeo Dual: Cronograma Institucional Oficial (5 Fases / 11 Sprints) vs Plan Tecnico Interno (6 Fases)
El cronograma oficial del semillero consta de 5 fases y 11 sprints semanales (del 14 de septiembre al 27 de noviembre) [VERIFICADO: imagenes oficiales del cronograma cargadas por el usuario]. Al `7 de noviembre`, el proyecto acaba de cerrar el **Sprint 8 (fin de Fase 3 oficial)** con la totalidad del desarrollo predictivo ejecutado y entra al **Sprint 9 (Fase 4 oficial: 9 al 13 de noviembre)** con la implementacion y validacion adelantadas.

| Fase Cronograma Oficial | Sprints y Fechas Oficiales | Entregable Institucional Exigido | Equivalencia Plan Interno | Artefactos Verificados en el Repositorio | Estado Real al 07-Nov |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **Fase 1.** Planteamiento y diseno | Sprints 1-3 (`14 sep - 2 oct`) | Problema, objetivos, estado del arte y propuesta metodologica | Fase 0 / Marco conceptual | `ESTADO_DEL_ARTE.md`, `PLAN_DE_ACCION.md` | Completada |
| **Fase 2.** Preparacion y analisis de datos | Sprints 4-5 (`5 - 16 oct`) | Dataset limpio documentado e informe exploratorio | Fases 1, 2 y 3 (EVA + Clima + Features) | `Dataset/clean/dataset_maestro_*.csv`, `Dataset/README.md`, `dashboard_tecnico/` | Completada |
| **Fase 3.** Desarrollo del enfoque predictivo | Sprints 6-8 (`19 oct - 6 nov`; hito `5 nov`) | Diseno experimental, modelos entrenados y seleccion del enfoque | Fase 4 (Baselines) + Fase 5 (Red Modular FiLM) | `Resultados/baselines_*.csv`, `Resultados/red_modular_*.csv`, `Experimental/` | Completada |
| **Fase 4.** Implementacion y validacion | Sprint 9 (`9 - 13 nov`; hito `12 nov`) | Solucion implementada y validada; limitaciones y aportes documentados | Ablacion `M0-M4`, `Experimental/index.html`, `HANDOFF.md` | `Experimental/ablacion_dimensiones_y_mejoras.py`, `HANDOFF.md`, despliegue GitHub Pages | Ejecutada (lista para cierre de Sprint 9) |
| **Fase 5.** Elaboracion del articulo cientifico | Sprints 10-11 (`16 - 27 nov`; hitos `19` y `26 nov`) | Primer borrador (`19 nov`) y articulo final ES/EN (`26 nov`) | Fase 6 (Manuscrito IMRaD) | Tablas/metricas listas en `Resultados/` y `Experimental/`; redaccion IMRaD pendiente | Pendiente (proxima ventana) |

---

## 3. Arquitectura del Repositorio y Custodia Git vs Hugging Face
El espacio de trabajo `C:\Proyecto Semillero` esta estructurado bajo separacion estricta entre codigo/resultados livianos (versionados en Git) y corpus tabulares masivos (custodiados localmente y preparados para Hugging Face Datasets) [DECISIÓN DEL USUARIO; VERIFICADO: .gitignore, Scripts/paths.py, dashboard_tecnico/exclusivo.html]:

* `Scripts/`: 27 scripts de ingesta, depuracion, ingenieria de caracteristicas, entrenamiento, auditoria e inventario, centralizados mediante el resolutor dinamico de rutas `Scripts/paths.py` [VERIFICADO].
* `Dataset/raw/` y `Dataset/clean/`: 6 insumos primarios crudos (`~81.5 MB`) y 22 tablas limpias/maestras (`~65.4 MB`), excluidos de Git mediante `.gitignore` (con excepcion de `Dataset/README.md`, que actua como Dataset Card oficial para Hugging Face) [VERIFICADO: Scripts/preparar_huggingface.py].
* `Resultados/`: 16 archivos CSV con metricas oficiales, curvas de aprendizaje, importancias de variables y predicciones fuera de muestra de las Fases 4 y 5 [VERIFICADO].
* `Experimental/`: Suite de iteracion rapida aislada de los hallazgos canonicos principales; aloja el comparador regional (`ejecutar_experimentos.py`), el estudio multi-escala (`estudio_paralelo_recomendaciones.py`), la ablacion de dimensiones y mejoras causales (`ablacion_dimensiones_y_mejoras.py`), sus salidas CSV/JSON y el visor web `Experimental/index.html` [VERIFICADO].
* `dashboard_tecnico/` e `index.html` (raiz) + `.nojekyll`: Plataforma interactiva HTML/CSS/JS autocontenida sin servidor y pagina `exclusivo.html` publicada directamente desde la rama `master` en GitHub Pages [VERIFICADO].

**Restriccion operativa critica de GitHub OAuth:** El token OAuth actual del cliente `gh` (`ke1npro`) posee alcances `'gist', 'read:org', 'repo'` pero **no** posee el alcance `'workflow'`. Por tanto, **no se deben crear ni modificar archivos bajo `.github/workflows/*.yml`**, ya que GitHub rechazara el `git push`. El despliegue de GitHub Pages opera nativamente desde la raiz (`/`) de la rama `master` con `.nojekyll` [VERIFICADO: configuracion API `repos/ke1npro/AgroENSO/pages`].

---

## 4. Inventario de Archivos Clave
Generado automaticamente mediante `python Scripts/generar_inventario.py` el `2026-11-07` [VERIFICADO: inventario_archivos.csv]. Total catalogado en el espacio de trabajo: **103 archivos** (mas `inventario_archivos.csv`). A continuacion se registran los artefactos troncales:

| Ruta Relativa | Rol | Filas | Columnas | Tamano (KB) | SHA-256 (prefijo 12 car.) | Fecha Modificacion | Proposito |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `Dataset/clean/dataset_maestro_andina_caribe.csv` | Final | 29,622 | 115 | 20,921.31 | `759d09c222e6` | 2026-10-20 17:54:25 | Dataset maestro Macro-Regional consolidado (10 departamentos) |
| `Dataset/clean/dataset_maestro_entrenamiento.csv` | Final | 18,425 | 114 | 12,867.52 | `827ed938cca8` | 2026-10-06 17:15:37 | Dataset maestro Region Andina (Antioquia, Huila, Tolima) |
| `Dataset/clean/dataset_maestro_entrenamiento_caribe.csv` | Final | 11,197 | 114 | 7,853.50 | `c5fe669c698a` | 2026-10-14 21:30:40 | Dataset maestro Region Caribe (7 departamentos) |
| `Dataset/clean/dataset_maestro_panel_balanceado.csv` | Final | 13,545 | 114 | 9,453.35 | `d75f661f3874` | 2026-10-06 17:15:41 | Subconjunto balanceado estricto Region Andina (191 municipios) |
| `Dataset/clean/dataset_maestro_panel_balanceado_caribe.csv` | Final | 7,141 | 114 | 5,001.55 | `9ff3d10568f5` | 2026-10-14 21:30:41 | Subconjunto balanceado estricto Region Caribe |
| `Dataset/clean/clima_anual_municipios_2007_2024.csv` | Intermedio | 3,744 | 35 | 862.66 | `02e23291a2d7` | 2026-10-06 11:29:20 | Clima anual calibrado Andina (208 municipios x 18 anos) |
| `Dataset/clean/clima_anual_municipios_2007_2024_caribe.csv` | Intermedio | 3,456 | 35 | 789.38 | `0679f5512dfd` | 2026-10-14 21:30:08 | Clima anual calibrado Caribe (192 municipios x 18 anos) |
| `Dataset/clean/oni_anual_features.csv` | Intermedio | 76 | 27 | 11.84 | `1d0dbe334f05` | 2026-10-06 11:14:28 | Variables anuales, semestrales, trimestrales y rezagos ONI (1950-2025) |
| `Dataset/clean/perfil_agronomico_cultivos.csv` | Intermedio | 5 | 17 | 0.87 | `a36ae5c6ef02` | 2026-10-06 11:29:44 | Parametros ecofisiologicos de los 5 cultivos del nucleo |
| `Resultados/baselines_resultados_completos.csv` | Final | 24 | 15 | 3.99 | `a8e002b4809d` | 2026-10-08 12:27:59 | Metricas de los 4 baselines en Region Andina (LOEO + ablacion ENSO) |
| `Resultados/baselines_resultados_caribe.csv` | Final | 24 | 16 | 4.22 | `34ec5baccd78` | 2026-10-14 21:34:47 | Metricas de los 4 baselines en Region Caribe (LOEO + ablacion ENSO) |
| `Resultados/baselines_xgboost_andina_caribe.csv` | Final | 18 | 18 | 3.83 | `8b6e7f71a71a` | 2026-10-20 17:54:39 | Metricas de XGBoost combinado Andina + Caribe desglosado |
| `Resultados/red_modular_comparativa_loeo.csv` | Final | 8 | 11 | 0.82 | `1d57430deb1b` | 2026-10-20 18:07:20 | Comparativa LOEO de Red Modular FiLM vs Sigmoide (Andina y Caribe) |
| `Experimental/resultados_simplificados.csv` | Experimental | 6 | 21 | 1.60 | `b8ee747dba21` | 2026-10-27 21:46:47 | Iteracion base: XGBoost Antioquia vs XGBoost Macro vs FiLM Macro |
| `Experimental/estudio_paralelo_resultados.csv` | Experimental | 6 | 9 | 0.74 | `6f79b48cb475` | 2026-11-06 18:20:44 | Benchmarking multi-escala LOEO (Macro, Andina, Caribe, Cesar, Sierra, Ant.) |
| `Experimental/ablacion_dimensiones_red.csv` | Experimental | 6 | 9 | 0.45 | `cb05bfa4408a` | 2026-11-07 10:32:41 | Curva de escalamiento de capacidad neuronal (`hidden_dim` 32 a 512) |
| `Experimental/mejoras_precision_loeo.csv` | Experimental | 16 | 16 | 3.09 | `50e4ccc4e21b` | 2026-11-07 10:40:40 | Escalera de mejoras causales libres de fuga (`M0` a `M4` y `ANDINA_M3`) |
| `Experimental/ablacion_dimensiones_y_mejoras.py` | Script | - | - | 29.97 | `48dc193b5cff` | 2026-11-07 10:32:17 | Script reproducible de escalamiento neuronal y mejoras causales LOEO |
| `dashboard_tecnico/index.html` | Web/Dashboard | - | - | 16.74 | `304ecba05350` | 2026-10-27 21:51:38 | Dashboard tecnico principal interactivo |
| `dashboard_tecnico/exclusivo.html` | Web/Dashboard | - | - | 12.33 | `2ebaa72789eb` | 2026-10-27 21:51:21 | Pagina Exclusivo: arquitectura de custodia Git vs Hugging Face |
| `Experimental/index.html` | Web/Experimental | - | - | 21.95 | `e873570db071` | 2026-11-07 10:42:12 | Visor web del Laboratorio Experimental y tablas de ablacion |

---

## 5. Como Reproducir
### 5.1 Entorno y Hardware Verificado
* **Sistema Operativo:** Windows 11 Pro 64-bit (`10.0.26200-SP0`) [VERIFICADO].
* **CPU / RAM:** AMD Ryzen (8 cores fisicos, 16 hilos logicos), 13.84 GB RAM [VERIFICADO].
* **Interprete Python:** `Python 3.13.13` (`C:\Users\Lenovo\AppData\Local\Programs\Python\Python313\python.exe`) [VERIFICADO].
* **Librerias verificadas en ejecucion:** `torch==2.14.1+cpu`, `pandas==3.0.5`, `numpy==2.5.2`, `scikit-learn==1.9.1`, `xgboost==3.4.1`, `scipy==1.18.1`, `requests==2.34.2`, `openpyxl==3.1.5` [VERIFICADO: import directo en sesion].

### 5.2 Orden de Ejecucion del Pipeline Completo
Desde la raiz `C:\Proyecto Semillero`:
1. `python Scripts/descarga_oni_noaa.py`: Descarga y procesa el indice ONI de NOAA CPC (`Dataset/raw/oni_mensual_noaa.csv` -> `Dataset/clean/oni_anual_features.csv`).
2. `python Scripts/pipeline_canonico.py` y `python Scripts/pipeline_eva_caribe.py`: Depuran EVA 2007-2024 para Region Andina (`18,591` filas consolidadas) y Region Caribe (`11,197` filas), aplicando la correccion numerica de cafe 2019 y el colapso de variedades de maiz post-2021.
3. `python Scripts/obtencion_clima.py` y `python Scripts/obtencion_clima_caribe.py`: Integran estaciones IDEAM (`nsz2-kzcq`), coordenadas DIVIPOLA (`pqwj-3fi4`), altitud DEM SRTM y reanalisis NASA POWER 2006-2024.
4. `python Scripts/construir_dataset_maestro.py` y `python Scripts/construir_dataset_maestro_caribe.py`: Calculan potencial departamental suavizado y ensamblan `dataset_maestro_entrenamiento.csv` (`18,425 x 114`), `dataset_maestro_entrenamiento_caribe.csv` (`11,197 x 114`) y `dataset_maestro_andina_caribe.csv` (`29,622 x 115`).
5. `python Scripts/evaluar_baselines_fase4.py`, `python Scripts/evaluar_baselines_caribe.py` y `python Scripts/evaluar_xgboost_andina_caribe.py`: Ejecutan el benchmarking LOEO de baselines clasicos (`seed=42`).
6. `python Scripts/prototipo_red_modular_fase5.py`: Entrena y evalua la Red Modular PyTorch (`FiLM` vs `Sigmoide`) con Monte Carlo Dropout en Andina y Caribe.
7. `python Scripts/exportar_metricas_dashboard.py` y `python Scripts/exportar_metricas_caribe.py`: Actualizan los payloads JS del visor `dashboard_tecnico/`.
8. `python Experimental/ejecutar_experimentos.py`, `python Experimental/estudio_paralelo_recomendaciones.py` y `python Experimental/ablacion_dimensiones_y_mejoras.py`: Reproducen todas las tablas del laboratorio `Experimental/` (`seed=42`; anadir `--incluir-escalado-completo` al ultimo script si se desea reentrenar desde cero los modelos pesados `hidden=256..512`).
9. `python Scripts/preparar_huggingface.py`: Verifica hashes e integridad de `Dataset/` y genera el paquete listo para `huggingface-cli upload`.
10. `python Scripts/generar_inventario.py`: Regenera `inventario_archivos.csv` con conteos y hashes SHA-256.

### 5.3 Chequeos de Forma e Integridad Automatizados
* `dataset_maestro_entrenamiento.csv` (Andina): `18,425` filas x `114` columnas; `208` municipios; `0` valores nulos en las 113 columnas numericas/categoricas operativas (unico campo con nulos: `subcategoria` descriptiva original) [VERIFICADO].
* `dataset_maestro_entrenamiento_caribe.csv` (Caribe): `11,197` filas x `114` columnas; `192` municipios; `0` valores nulos en predictores [VERIFICADO].
* `dataset_maestro_andina_caribe.csv` (Macro-Regional): `29,622` filas x `115` columnas (`114` base + `macro_region`); `400` municipios unicos (`cod_dane_mun` preservado como string de 5 digitos con cero inicial) [VERIFICADO].

---

## 6. Procedencia de Datos
| Fuente Primaria | Identificador / Endpoint | Fecha Descarga | Licencia / Uso | Cobertura Real Verificada en el Proyecto |
| :--- | :--- | :---: | :--- | :--- |
| EVA 2007-2018 | Portal AgroNet (MADR) `Base Agricola EVA 2007-2018_MADR.xlsx` (`header=1`) | 2026-08-29 | Publico institucional | `12,693` filas Andina + `7,544` filas Caribe en los 5 cultivos [VERIFICADO] |
| EVA 2019-2024 | Datos Abiertos Colombia / UPRA (`2019_-_2025`, `166,732` filas nacionales) | 2026-08-29 | Abierta gubernamental | `5,898` filas Andina + `3,653` filas Caribe validas (2019-2024) [VERIFICADO] |
| Normales IDEAM | Datos Abiertos Socrata `nsz2-kzcq` | 2026-10-06 / 10-14 | Datos Abiertos Colombia | `3,889` registros Andina + `1,833` registros Caribe [VERIFICADO] |
| Coordenadas DANE | Datos Abiertos Socrata `pqwj-3fi4` (DIVIPOLA) | 2026-10-06 / 10-14 | Datos Abiertos Colombia | `400` municipios (`208` Andina + `192` Caribe; corregido Mariquita `73443`) [VERIFICADO] |
| Elevacion DEM | Open-Meteo SRTM 90m (`/v1/elevation`) | 2026-10-06 / 10-14 | Abierta no comercial | `400` cabeceras municipales [VERIFICADO] |
| Clima Historico | NASA POWER Regional API (`/temporal/monthly/regional`) | 2026-10-06 / 10-14 | Abierta NASA LaRC | Series mensuales 2006-2024 calibradas con sesgo aditivo/multiplicativo IDEAM [VERIFICADO] |
| Indice ONI | NOAA CPC (`oni.ascii.txt`) | 2026-10-06 | Dominio publico NOAA | `920` meses (1950 a 2026) [VERIFICADO] |

---

## 7. Diccionario del Dataset Maestro y Extensiones Causales
El dataset consolidado `dataset_maestro_andina_caribe.csv` contiene `115` columnas agrupadas en cinco bloques base, a los cuales el modulo experimental de alta precision (`Experimental/ablacion_dimensiones_y_mejoras.py`) incorpora tres extensiones estrictamente causales:

1. **Grupo 1 - Metadatos, Llaves y Calidad (`21` columnas en Macro, excluidas como features crudas):** Geograficas (`departamento`, `municipio`, `cod_dane_dep`, `cod_dane_mun`, `macro_region`), taxonomicas (`cultivo`, `subcategoria`, `cultivo_cat`), temporales (`año`, `periodo`, `fuente`), `en_panel_balanceado`, metadatos de estacion IDEAM y etiquetas descriptivas ENSO (`fase_enso_anual`, `categoria_evento_anual`).
2. **Grupo 2 - Variables Agricolas Observadas y Targets (`9` columnas, excluidas de $X$ por prevencion de fuga):** `area_sembrada`, `area_cosechada`, `produccion`, `rendimiento`, `rendimiento_potencial_dep`, `eficiencia`, `rendimiento_base_municipal`, `eficiencia_rel_municipal` y el target canonico **`eficiencia_clip`** (recortado en `[0.05, 3.0]` y entrenado con perdida robusta Huber).
3. **Grupo 3 - Rama 1: Contexto Biofisico Zonal y Ecofisiologia del Cultivo (`26` features continuas):** `9` geografico-climatologicas (`latitud_dane`, `longitud_dane`, `altitud_msnm`, normales IDEAM y medias/desviaciones historicas municipales) y `17` parametros ecofisiologicos del cultivo (`es_permanente`, `via_fotosintetica_c4`, `nivel_tecnificacion`, duracion de ciclo, umbrales termicos/pluviometricos/altitudinales optimos y sensibilidad hidrica).
4. **Grupo 4 - Rama 2: Senal Climatica Local y Ciclo ENSO (`53` features continuas):** `29` de clima municipal anual/semestral/trimestral, anomalias, z-scores y rezagos $t-1$, mas `24` metricas del indice ONI (medias, extremos, acumulados, rachas, ventanas trimestrales/semestrales, rezagos $t-1$ e interacciones ONI x anomalia).
5. **Grupo 5 - Interacciones Explicitas de Estres Agroclimatico (`6` features continuas):** `estres_calor_c`, `estres_frio_c`, `desviacion_temp_optima_c`, `ratio_hidrico_cultivo`, `deficit_hidrico_mm`, `exceso_hidrico_mm`.
6. **Extensiones Causales Libres de Fuga (`M1` a `M3` en `ablacion_dimensiones_y_mejoras.py`):**
   * **Alineacion semestral (`M1`):** Indicadores `sem_A`, `sem_B` extraidos de `periodo`, mas `precip_sem_activo`, `temp_sem_activo`, `oni_sem_activo` (clima y ONI del semestre real en que crecio la cosecha transitoria) e interacciones regionales `oni_x_caribe`, `anom_precip_x_caribe`.
   * **Entity Embedding Municipal (`M2`):** Representacion densa aprendible $\mathbf{e}_m \in \mathbb{R}^8$ indexada por `cod_dane_mun` en la Rama 1 (captura fertilidad edafica, riego e infraestructura local sin tocar $y$).
   * **Prior Estructural Causal Empirical Bayes + Aprendizaje Residual (`M3`):** Media historica suavizada $\hat{\mu}_{m,c}^{\text{causal}} = \frac{\sum y + \lambda \bar{y}_{d,c}}{n + \lambda}$ ($\lambda = 5$) y dispersion $\hat{\sigma}_{m,c}^{\text{causal}}$ calculadas **exclusivamente con anos pasados `año < test_start_year` para los conjuntos de prueba** (`2007-2014` al evaluar `2015-2016`; `2007-2022` al evaluar `2023-2024`) y mediante **Leave-One-Year-Out** dentro del entrenamiento. La red predice la desviacion agroclimatica residual $\hat{y} = \max(0.05, \hat{\mu}_{m,c}^{\text{causal}} + f_{\text{FiLM}}(x_{\text{ctx}}, x_{\text{clim}}))$.

---

## 8. Sintesis Consolidada de Resultados Empiricos

### 8.1 Baselines Clasicos y Red Modular Inicial (Features Base, Promedio LOEO 2015-2016 y 2023-2024)
Fuente verificada: `Resultados/baselines_resultados_completos.csv`, `Resultados/baselines_resultados_caribe.csv`, `Resultados/red_modular_comparativa_loeo.csv` y `Experimental/resultados_simplificados.csv`.

| Ambito Geografico | N Total | Modelo (Configuracion `Con_ENSO`) | RMSE LOEO | MAE LOEO | $R^2$ LOEO | Pearson $r$ | PICP 95% |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Antioquia Exclusivo** | 9,296 | XGBoost Regressor | 0.3002 | 0.2233 | 0.2398 | 0.5048 | - |
| **Region Andina (3 Dptos)** | 18,425 | Random Forest (`160` arboles) | 0.2750 | 0.2014 | 0.1806 | 0.4450 | - |
| **Region Andina (3 Dptos)** | 18,425 | XGBoost (`220` arboles) | 0.2789 | 0.2068 | 0.1570 | 0.4235 | - |
| **Region Andina (3 Dptos)** | 18,425 | RedModular_Sigmoide (Fase 5 base) | 0.2848 | 0.2066 | 0.1216 | 0.3826 | 91.65% |
| **Region Andina (3 Dptos)** | 18,425 | RedModular_FiLM (Fase 5 base) | 0.2828 | 0.2026 | 0.1331 | 0.3950 | 91.19% |
| **Region Caribe (7 Dptos)** | 11,197 | Random Forest (`160` arboles) | 0.3262 | 0.2530 | 0.1055 | 0.3727 | - |
| **Region Caribe (7 Dptos)** | 11,197 | XGBoost (`220` arboles) | 0.3272 | 0.2544 | 0.0977 | 0.3724 | - |
| **Region Caribe (7 Dptos)** | 11,197 | RedModular_FiLM (Fase 5 base) | 0.3399 | 0.2525 | 0.0250 | 0.2810 | 89.30% |
| **Macro-Regional (10 Dptos)** | 29,622 | XGBoost Combinado (`EXP-2`) | 0.3062 | 0.2314 | 0.0701 | 0.3431 | - |
| **Macro-Regional (10 Dptos)** | 29,622 | RedModular_FiLM (`EXP-3` base) | 0.3033 | 0.2215 | 0.0933 | 0.3375 | 94.67% |

### 8.2 Estudio Paralelo Multi-Escala y Viabilidad de Cultivos de la Recomendacion
Fuente verificada: `Experimental/estudio_paralelo_recomendaciones.py`, `Experimental/estudio_paralelo_resultados.csv` y auditoria directa sobre `Dataset/raw/`.

1. **Disponibilidad real en EVA cruda (Caribe, 7 departamentos, 2007-2024) para la linea paralela recomendada:**
   * **Arroz:** `4,590` registros (`2,970` en 2007-2018 + `1,620` en 2019-2024) -> **Alta viabilidad inmediata**.
   * **Palma de aceite:** `1,241` registros (`661` en 2007-2018 + `580` en 2019-2024) -> **Viable**.
   * **Algodon:** `684` registros (`489` en 2007-2018 + `195` en 2019-2024) -> **Muestra reducida pero modelable con regularizacion**.
   * **Cafe de la Sierra Nevada** (Valledupar, Pueblo Bello, Agustin Codazzi, Santa Marta, Cienaga, Aracataca, Fundacion, Dibulla, San Juan del Cesar): `132` registros -> **Insuficiente como modelo aislado** (`R^2 = -0.1974` al aislar los 8 municipios con `N = 759` multicultivo en `EXP-SIERRA-NEVADA`), pero **valioso integrado mediante condicionamiento FiLM** dentro del panel macro-regional.
   * **Ganaderia e indice NDVI:** **Ausentes en EVA Agricola**. Ganaderia requiere Censo Nacional Agropecuario (CNA-DANE) / Encuesta Nacional Agropecuaria o inventario bovino ICA, y NDVI requiere extraccion satelital MODIS (`MOD13Q1`) / Sentinel-2 en Google Earth Engine. Se documentan como trabajo futuro de segunda generacion sin desviar el cierre de Sprints 9-11.
2. **Efecto de escala geografica con arboles planos (`estudio_paralelo_resultados.csv`):** Al aislar micro-dominios pequenos sin pooling regional (`Cesar Exclusivo`, `N = 2,479`, $R^2 = -0.2155$; `Sierra Nevada`, `N = 759`, $R^2 = -0.1974$), los arboles sobreajustan; al agregar toda la Region Andina (`N = 18,425`, $R^2 = 0.1605$) o Antioquia (`N = 9,296`, $R^2 = 0.2433$) mejora la estabilidad, pero al mezclar Andina + Caribe en un arbol plano (`N = 29,622`), el $R^2$ cae a `0.0732` por signos opuestos de teleconexion ENSO.

### 8.3 Diagnostico de Varianza y Prueba de Escalamiento de Dimensiones Neuronales
Antes de modificar las entradas, se descompuso la varianza total de `eficiencia_clip` en `dataset_maestro_andina_caribe.csv` (`Var total = 0.1004`, `std = 0.3169`):
* **35.2% (`0.0353`) es varianza estructural persistente entre pares `(municipio, cultivo_cat)`** (diferencias permanentes de suelo, riego y manejo frente al potencial departamental; autocorrelacion `lag-1` observada `r = 0.5790` en `N = 26,178`).
* **Colision por omision de semestre (`periodo`):** Existen `10,317` pares de cosecha semestral `A` y `B` en el mismo municipio-cultivo-ano con diferencia absoluta media `|efic(A) - efic(B)| = 0.0954`, que en los modelos base recibian vectores predictores $X$ **100% identicos** (error de Bayes irreducible).
* **Teleconexiones ENSO de signo opuesto:** `r(oni_media_anual, eficiencia_clip)` es `+0.0297` en Andina frente a `-0.0229` en Caribe (`Sucre -0.0706`), y dentro del propio Caribe la anomalia de lluvia correlaciona positivamente en zonas aridas (`Atlantico +0.0747`, `La Guajira +0.0546`) pero negativamente en depresiones inundables (`Magdalena -0.0712`, `Bolivar -0.0635`).

Al evaluar si **aumentar las dimensiones de la Red Modular FiLM** mejora la precision manteniendo fijas las features originales (`Experimental/ablacion_dimensiones_red.csv`, `N = 29,622`, `seed = 42`):

| `hidden_dim` | Capas / Rama | Parametros Entrenables | `RMSE_train` | `RMSE_test` LOEO | `MAE_test` LOEO | $R^2$ Test LOEO | Pearson $r$ | Tiempo CPU (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 32 | 2 | 8,289 | 0.2956 | 0.3071 | 0.2241 | 0.0713 | 0.2878 | 78.9 |
| **64 (Base)** | **2** | **25,793** | **0.2920** | **0.3055** | **0.2232** | **0.0814** | **0.2915** | **81.8** |
| 128 | 2 | 88,449 | 0.2779 | 0.3040 | 0.2225 | 0.0898 | 0.3162 | 151.8 |
| 256 | 2 | 324,353 | 0.2911 | 0.3060 | 0.2235 | 0.0779 | 0.2906 | 355.2 |
| 256 | 4 | 589,569 | 0.2908 | 0.3039 | 0.2208 | 0.0915 | 0.3067 | 553.4 |
| 512 | 3 | 1,765,889 | 0.2834 | 0.3000 | 0.2201 | 0.1153 | 0.3324 | 430.5 |

**Conclusion empirica sobre dimensiones:** Multiplicar los parametros por **68.5x** (`25,793 -> 1,765,889`) y el tiempo de computo por **5.3x** produce un avance marginal (`MAE` baja apenas `0.0031`, de `0.2232` a `0.2201`; $R^2$ pasa de `0.0814` a `0.1153`, e incluso retrocede en `256 x 2`). Aumentar dimensiones sin resolver la colision semestral ni la heterogeneidad municipal no es eficiente ni resuelve el techo predictivo [VERIFICADO: Experimental/ablacion_dimensiones_red.csv].

### 8.4 Escalera de Mejoras Estructurales y Causales Libres de Fuga (`M0` a `M4`)
Fuente verificada: `Experimental/mejoras_precision_loeo.csv` y `Experimental/resumen_mejoras_precision.json` (`seed = 42`).

| ID Modelo | Ambito | Arquitectura / Mejora Incorporada | Parametros | RMSE LOEO | MAE LOEO | $R^2$ LOEO | Pearson $r$ | MAPE (%) | PICP 95% | MAE Depto-Cultivo | MAE Cultivo-Ano |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `M0_FiLM_Base` | Macro (10 Dptos) | RedModularFiLM Base (`hidden=64`) | 25,793 | 0.3055 | 0.2232 | 0.0814 | 0.2915 | 27.40 | 96.56% | 0.0892 | 0.0406 |
| `M1_Semestre_Caribe` | Macro (10 Dptos) | + Semestre Activo (`A`/`B`) e Interacciones Caribe | 26,241 | 0.3032 | 0.2213 | 0.0954 | 0.3044 | 27.71 | 96.89% | 0.0849 | 0.0358 |
| `M2_Entity_Embedding` | Macro (10 Dptos) | + Entity Embedding Municipal (`cod_dane_mun -> R^8`, `hidden=96`) | 57,161 | 0.2919 | 0.2117 | 0.1637 | 0.4008 | 26.38 | 96.87% | 0.0907 | 0.0376 |
| **`M3_FiLM_Residual_Causal`** | **Macro (10 Dptos)** | **RedModularFiLM Residual + Embedding + Prior Causal Estricto** | **57,353** | **0.2775** | **0.1991** | **0.2395** | **0.4912** | **25.04** | **95.63%** | **0.0900** | **0.0406** |
| `XGB_M3_Enriquecido` | Macro (10 Dptos) | XGBoost Regressor con exactamente las mismas features `M3` | - | 0.2949 | 0.2176 | 0.1370 | 0.4423 | 28.52 | - | 0.1094 | 0.0621 |
| `M4_Ensamble_Hibrido` | Macro (10 Dptos) | Ensamble 50% `FiLM_M3` + 50% `XGB_M3` | 57,353 | 0.2816 | 0.2051 | 0.2162 | 0.4779 | 26.45 | - | 0.0961 | 0.0476 |
| **`ANDINA_M3_FiLM_Residual`** | **Andina (3 Dptos)** | **RedModularFiLM Residual + Prior Causal (`N = 18,425`)** | **54,857** | **0.2569** | **0.1835** | **0.2872** | **0.5318** | **23.31** | **95.84%** | **0.0517** | **0.0310** |
| `ANDINA_XGB_M3` | Andina (3 Dptos) | XGBoost Regressor con exactamente las mismas features `M3` | - | 0.2686 | 0.1949 | 0.2188 | 0.4843 | 25.41 | - | 0.0541 | 0.0353 |

**Hallazgos cientificos clave de la escalera `M0 -> M3`:**
1. Solo con **Entity Embeddings municipales (`M2`, sin usar ningun estadistico de $y$ en las entradas)**, el $R^2$ macro-regional se duplica de `0.0814` a `0.1637` (`r = 0.4008`), superando con `57k` parametros a la red bruta de `1.77M` parametros (`0.1153`).
2. Con **aprendizaje residual FiLM sobre el prior causal libre de fuga (`M3`)**, la Red Modular alcanza $R^2 = 0.2395$ en los 10 departamentos (`0.2408` en El Nino `2015-2016` y `0.2383` en El Nino `2023-2024`, frente a `0.0361` de `M0` en `2023-2024`) y **$R^2 = 0.2872$ en Region Andina** (`R^2 = 0.3636`, `RMSE = 0.2200`, `MAE = 0.1525` en El Nino `2015-2016`, reduciendo en **11.4%** el RMSE frente al XGBoost Fase 4 `0.2484`).
3. **Superioridad demostrada de la arquitectura neuronal modular frente a Gradient Boosting en igualdad de informacion:** Al entregar exactamente la misma matriz causal `M3` a `XGBoost`, los arboles particionan el prior continuo en escalones y sobreajustan (`R^2 = 0.1370` vs `0.2395` en Macro; `0.2188` vs `0.2872` en Andina), mientras que la conexion residual lineal `prior + f_FiLM(...)` permite a la red neuronal modular dedicar el 100% de su capacidad no lineal a estimar el choque agroclimatico $\Delta \eta$.
4. **Cumplimiento de criterios del `PLAN_DE_ACCION.md`:** La cobertura empirica al 95% se situa en **`95.63%`** (Macro) y **`95.84%`** (Andina), practicamente exacta sobre el nivel nominal, y el error agregado por cultivo-ano desciende a **`0.0406`** (Macro) y **`0.0310`** (Andina; `0.0517` a nivel departamento-cultivo-ano), cumpliendo holgadamente el umbral `MAE <= 0.08` a escala agregada.

---

## 9. Registro de Decisiones
| Decision Adoptada | Motivo Tecnico | Alternativas Descartadas | Origen | Evidencia |
| :--- | :--- | :--- | :---: | :--- |
| No modelar la dinamica oceanica ENSO internamente | Pronosticar SST del Pacifico excede el alcance agricola; NOAA publica ONI oficial | Red LSTM oceanica propia | [PROPUESTA ACEPTADA] | `PLAN_DE_ACCION.md` |
| Target principal `eficiencia_clip` con Huber Loss | Outliers de digitacion alcanzan `7.2252` frente a mediana `0.9297` | Rendimiento bruto t/ha o MSE sin recorte | [DECISIÓN DEL USUARIO] | `PLAN_DE_ACCION.md` Sec. 3.1 |
| Potencial tecnologico a nivel departamento-cultivo | Vacios municipales impiden estimar tendencia local estable en series cortas | Tendencia LOESS municipal individual | [PROPUESTA ACEPTADA] | `Scripts/construir_dataset_maestro.py` |
| Conciliacion EVA AgroNet (2007-2018) + UPRA (2019-2024) y colapso de maiz | Garantiza 18 anos continuos corrigiendo desglose de maiz blanco/amarillo post-2021 | Usar solo 2019-2024 o duplicar variedades | [DECISIÓN DEL USUARIO] | `Scripts/pipeline_canonico.py`, `correccion_maiz.py` |
| Clima hibrido (IDEAM + DEM SRTM + NASA POWER) | Estaciones IDEAM cubren directamente 62/208 municipios andinos; NASA aporta continuidad | Open-Meteo Archive (HTTP 429) o IDEAM crudo | [DECISIÓN DEL USUARIO] | `Scripts/obtencion_clima.py` |
| Validacion estricta Leave-One-Event-Out (LOEO) | Evita fuga temporal intra-episodio en El Nino `2015-2016` y `2023-2024` | K-Fold aleatorio | [PROPUESTA ACEPTADA] | `PLAN_DE_ACCION.md` Sec. 4.1 |
| Expansion multi-regional a 10 departamentos (Andina + Caribe) | Contrastar regimenes climaticos orograficos vs litorales (`N = 29,622`) | Limitar el estudio a 3 departamentos | [DECISIÓN DEL USUARIO] | `Dataset/clean/dataset_maestro_andina_caribe.csv` |
| Reestructuracion en carpetas maestras y segregacion Git vs Hugging Face | Mantener Git liviano y auditable; alojar `Dataset/` (~147 MB) en Hugging Face Datasets | Subir binarios pesados a Git | [DECISIÓN DEL USUARIO] | `.gitignore`, `dashboard_tecnico/exclusivo.html` |
| Despliegue directo de GitHub Pages desde rama `master` (`/`) con `.nojekyll` | El token OAuth `gh` carece de alcance `workflow` para subir `.github/workflows/*.yml` | GitHub Actions workflow YAML | [PROPUESTA ACEPTADA] | `index.html`, `.nojekyll`, API GitHub Pages |
| Mantener cultivos recomendados (Palma, Arroz, Algodon) como estudio paralelo | Preservar intacto el cronograma de Sprints 9-11 del nucleo de 5 cultivos ya validado | Reemplazar los 5 cultivos a 3 semanas de entrega | [DECISIÓN DEL USUARIO] | `Experimental/estudio_paralelo_recomendaciones.py` |
| Descartar escalamiento bruto `hidden=512` y adoptar arquitectura compacta `M3` (`hidden=96` residual causal) | `M3` (`57k` params) logra `R^2 = 0.2395` frente a `0.1153` de `hidden=512` (`1.77M` params) en 5x menos tiempo | Aumentar capas y neuronas sin corregir entradas | [PROPUESTA ACEPTADA] | `Experimental/ablacion_dimensiones_y_mejoras.py` |

---

## 10. Hallazgos, Errores Corregidos y Discrepancias Documentadas
1. **Bug de parsing numerico en cafe (2019):** Registros de Datos Abiertos mayores a 999 usaban punto de miles y coma decimal, generando `NaN` y una caida artificial >90%. Resuelto con `to_num_new` en `pipeline_canonico.py` [VERIFICADO].
2. **Descarte de cambio de unidad en cafe:** Se verifico contra fichas UPRA que no hubo cambio de cereza a pergamino en 2019; era exclusivamente el error de parsing [VERIFICADO: `Scripts/diag_cafe_unidad.py`].
3. **Colapso de variedades de maiz post-2021:** Sumatoria de areas y producciones de maiz blanco y amarillo por municipio-ano-semestre [VERIFICADO: `Scripts/correccion_maiz.py`].
4. **Hueco oficial de maiz tecnificado en Antioquia (2015-2018):** `0` registros en la fuente gubernamental primaria; los municipios reportaron bajo maiz tradicional en ese cuatrienio [VERIFICADO].
5. **Preservacion de codigos DANE y correccion de Mariquita (`73443`):** Forzado `str.zfill(5)` y correccion de coordenadas `525.0, -74916667.0` -> `5.1989, -74.8897` [VERIFICADO: `Scripts/obtencion_clima.py`].
6. **Correccion de codificacion consola Windows (`cp1252`) en `preparar_huggingface.py`:** Sustitucion de caracteres Unicode no soportados y correccion de argumento `type=str` en `argparse` [VERIFICADO].
7. **Correccion de conteos `N Test` en `Experimental/index.html`:** Se alinearon las celdas HTML con los conteos exactos de `resultados_simplificados.csv` (`1,022 / 1,099` en Antioquia y `3,533 / 2,937` en Macro) [VERIFICADO].
8. **Asimetria taxonomica detectada en `subcategoria` de `Frijol` (2007-2018 vs 2019-2024):** En `B_agronet_2007_2018`, `Frijol` presenta desglose entre `FRIJOL ARBUSTIVO` y `FRIJOL VOLUBLE` en `420` pares municipio-ano-semestre, mientras que `A_datosgov_2019_2025` reporta `Frijol` unificado. Por ello, **`subcategoria` nunca debe incluirse como predictor** al generalizar hacia `2023-2024` [VERIFICADO: auditoria directa sobre `dataset_maestro_andina_caribe.csv`].
9. **Discrepancia documental vigente en `ESTADO_DEL_ARTE.md`:** Las lineas `28`, `40` y `170-171` de `ESTADO_DEL_ARTE.md` aun contienen menciones textuales a `Romero (2014)` y `Stuepp et al. (2022)`, las cuales fueron auditadas como citas inexistentes/alucinadas en etapas tempranas. **Advertencia obligatoria para Sprint 10-11:** No trasladar dichas referencias al manuscrito IMRaD; emplear unicamente las referencias verificadas en la Seccion 13 de este documento [VERIFICADO: lectura directa de `ESTADO_DEL_ARTE.md`].

---

## 11. Callejones sin Salida Documentados
* **Escalamiento bruto de ancho/profundidad neuronal (`hidden=256..512`, `4` capas) sin enriquecimiento causal:** Eleva los parametros a `1.77M` y el tiempo de entrenamiento CPU a `7-9 min` por configuracion, pero se estanca en $R^2 \le 0.1153$ debido al error de Bayes inducido por semestres colisionados y heterogeneidad municipal latente [VERIFICADO: `Experimental/ablacion_dimensiones_red.csv`].
* **Modelos de arboles aislados en micro-dominios (`Cesar` o `Sierra Nevada` solos):** Producen $R^2$ negativo (`-0.2155` y `-0.1974`) en LOEO por tamano muestral insuficiente ante choques extremos [VERIFICADO: `Experimental/estudio_paralelo_resultados.csv`].
* **API Open-Meteo Archive masiva (`/v1/archive`):** Bloqueo HTTP 429 por cuota [VERIFICADO].
* **Telemetria cruda IDEAM en Socrata (`s54a-sgyg`, `sbwg-7ju4`) y portal DHIME:** Cobertura parcial (62/208 municipios en Andina), timeouts Socrata y CAPTCHAs en DHIME [VERIFICADO].
* **Workflows `.github/workflows/*.yml` con el token `gh` actual:** Rechazo remoto por ausencia del scope OAuth `workflow` [VERIFICADO].

---

## 12. Limitaciones y Riesgos Abiertos
1. **Distincion entre generalizacion temporal (municipios vistos) y extrapolacion espacial pura (municipios nuevos):** El modelo `M3` (`R^2 = 0.2395` Macro / `0.2872` Andina) utiliza *Entity Embeddings* y un prior historico estrictamente causal (`año < test_start_year`), lo cual es 100% valido y libre de fuga para pronosticar el impacto de un futuro evento ENSO sobre los 400 municipios monitoreados por EVA. Sin embargo, para un municipio completamente nuevo sin historial EVA previo ($n = 0$), el prior Empirical Bayes se contrae automaticamente al promedio departamental ($\hat{\mu} \to \bar{y}_{d,c}$) y el embedding cae al indice `0` (desconocido), reduciendo el desempeno hacia el regimen `M1` [VERIFICADO: `Experimental/ablacion_dimensiones_y_mejoras.py`].
2. **Incertidumbre climatica en municipios sin estacion IDEAM cercana:** En municipios interpolados espacialmente, el error honesto medido por validacion cruzada Leave-One-Station-Out (LOO-CV) es `RMSE = 45.95 mm/mes` ($R^2 = 0.8126$) en precipitacion y `RMSE = 0.8972 °C` ($R^2 = 0.9597$) en temperatura [VERIFICADO: `Scripts/validacion_ideam_vs_nasa_power.py`].
3. **Ventana movil centrada en `rendimiento_potencial_dep`:** El denominador departamental $Y^{\text{pot}}_{d,c,t}$ usa media movil centrada de 5 anos (`center=True`) excluyendo `[2009, 2015, 2016, 2023]`, pero aun incluye parcialmente `2024` en el borde final de la serie, lo que puede atenuar levemente la magnitud de anomalia en 2024 [VERIFICADO: `Scripts/construir_dataset_maestro.py`].
4. **Potencia estadistica de episodios extremos:** La ventana 2007-2024 abarca dos eventos El Nino fuertes (`2015-2016` y `2023-2024`), limite intrinseco del registro historico moderno de EVA [VERIFICADO].

---

## 13. Estado de Verificacion de Referencias Bibliograficas
| Referencia | Estado | Evidencia de Verificacion |
| :--- | :---: | :--- |
| Botero y Barnes (2022, *Food Security*) | [VERIFICADO] | Springer Food Security: impacto climatico agricola en Colombia |
| Justino et al. (2026, *Theor. Appl. Climatol.*) | [VERIFICADO] | Articulo/preprint TAC sobre ENSO y precipitacion andina |
| Jimenez Benavides y Urrea Urresti (2025, UNAD) | [VERIFICADO] | Tesis de maestria UNAD sobre rendimiento agricola y clima |
| Nobrega Passos, Kaur y Sarmiento Herrera (2023, Neuromatch) | [VERIFICADO] | Investigacion Neuromatch Academy sobre Deep Learning agroclimatico |
| Banco de la Republica (2026, Recuadro 1) | [VERIFICADO] | Documento tecnico oficial sobre choque de alimentos y El Nino |
| Posada y Alfaro (2025, IEEE CONCAPAN) | [VERIFICADO: DOI 10.1109/CONCAPAN66820.2025.11512570] | ML y ENSO para rendimiento de maiz en Centroamerica |
| Castro Hernandez y Duran Blanco (2023, UIS) | [VERIFICADO: Repositorio Noesis UIS] | Tesis maestria UIS: cafe y cacao en Santander con ML/DL |
| Gal y Ghahramani (2016, ICML) | [VERIFICADO: PMLR 48:1050-1059] | Monte Carlo Dropout como aproximacion bayesiana |
| Lobell y Field (2007, *Environ. Res. Lett.*) | [VERIFICADO: DOI 10.1088/1748-9326/2/1/014002] | Relaciones empiricas clima-rendimiento global |
| Perez et al. (2018, AAAI) | [VERIFICADO: AAAI-18] | Arquitectura fundacional FiLM (Feature-wise Linear Modulation) |
| Romero (2014) | **[DESACREDITADA / PROHIBIDA]** | Inexistente; aun aparece en texto antiguo de `ESTADO_DEL_ARTE.md` (no citar) |
| Stuepp et al. (2022) | **[DESACREDITADA / PROHIBIDA]** | Inexistente; aun aparece en texto antiguo de `ESTADO_DEL_ARTE.md` (no citar) |

---

## 14. Proximos Pasos (Sprints 9 a 11) y Decisiones para Confirmar con Keiner

### 14.1 Hoja de Ruta Inmediata segun Cronograma Oficial
1. **Cierre de Sprint 9 (`9 - 13 de noviembre`, Fase 4 oficial - Implementacion y validacion):**
   * Decidir si la arquitectura ganadora **`M3_FiLM_Residual_Causal`** (`R^2 = 0.2395` Macro / `0.2872` Andina) se promueve desde `Experimental/` hacia el dashboard principal `dashboard_tecnico/index.html` como modelo insignia final del proyecto.
   * Opcional: Subir el paquete ya validado por `Scripts/preparar_huggingface.py` al repositorio de Hugging Face Datasets cuando se defina el identificador de usuario/organizacion en HF.
2. **Sprint 10 (`16 - 20 de noviembre`, hito `19 nov` - Primer borrador del articulo IMRaD):**
   * Redactar Introduccion, Estado del Arte (depurando `Romero 2014` y `Stuepp 2022`), Metodologia (pipeline hibrido IDEAM+NASA, formulacion FiLM residual causal libre de fuga y protocolo LOEO) y Resultados/Discusion con las tablas verificadas de la Seccion 8.
3. **Sprint 11 (`23 - 27 de noviembre`, hito `26 nov` - Articulo final ES/EN):**
   * Elaborar resumen, *abstract*, palabras clave, conclusiones, formato de revista y revision bilingue espanol/ingles.

### 14.2 Puntos Abiertos para Confirmar con Keiner
1. **Promocion de `M3` al visor principal:** ¿Promovemos las metricas de `M3_FiLM_Residual_Causal` al Dashboard Tecnico principal o se mantienen en `Experimental/index.html` como hallazgo de optimizacion de Fase 4/5?
2. **Publicacion efectiva en Hugging Face Hub:** Confirmar el nombre del repositorio destino (ej. `<usuario>/AgroENSO-Colombia-2007-2024`) para ejecutar la subida de `Dataset/`.
3. **Plantilla de revista para Sprint 10:** Confirmar si el borrador IMRaD del `19 de noviembre` se formatea bajo normas IEEE (doble columna) o formato de *Ingenieria y Competitividad*.

---

## 15. Prompt de Arranque para el Proximo Agente
```text
INSTRUCCION DE CONTINUIDAD PARA EL PROXIMO AGENTE:
1. Lee este archivo HANDOFF.md completo antes de proponer cambios o redactar el manuscrito cientifico.
2. Ejecuta `python Scripts/generar_inventario.py` para verificar la integridad de los 103+ archivos del espacio de trabajo.
3. Reglas metodologicas inquebrantables:
   - Todo numero reportado en codigo, dashboards o en el articulo IMRaD (Sprints 10-11) debe provenir de los CSV verificados en `Resultados/` y `Experimental/`. Prohibido inventar o interpolar metricas.
   - Cero fuga temporal o de target: respetar estrictamente los splits LOEO (`2015-2016` y `2023-2024`) y el calculo causal `año < test_start_year` en cualquier prior historico.
   - NO crear ni modificar archivos en `.github/workflows/` (el token OAuth actual carece del scope `workflow`; GitHub Pages despliega directamente desde `master` raiz).
   - NO subir `Dataset/` a Git (excluido en `.gitignore` para custodia exclusiva en Hugging Face).
   - PROHIBIDO citar a `Romero (2014)` o `Stuepp et al. (2022)` en el manuscrito aunque sigan mencionados en `ESTADO_DEL_ARTE.md`.
```
