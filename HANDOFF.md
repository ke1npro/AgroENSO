# HANDOFF: Estado del Proyecto y Transferencia Tecnica

## 1. Resumen Ejecutivo
El proyecto desarrolla un modelo computacional para predecir el impacto de la variabilidad climatica y el ciclo ENSO (El Niño-Oscilacion del Sur) sobre el rendimiento de cultivos en Colombia [VERIFICADO: PLAN_DE_ACCION.md]. Se han consolidado 18 anos de evaluaciones agropecuarias municipales (EVA 2007-2024, 29,622 registros validos en 10 departamentos de la Region Andina y Caribe), se construyo una serie climatica hibrida municipal calibrada con normales del IDEAM y reanalisis NASA POWER (2006-2024), y se integro la senal macroclimatica mensual ONI de NOAA [VERIFICADO: dataset_maestro_andina_caribe.csv]. Se implementaron y evaluaron cuatro baselines clasicos (Ridge, Huber, Random Forest y XGBoost) y una Red Neuronal Modular de Doble Rama en PyTorch con modulacion contextual FiLM e incertidumbre con Monte Carlo Dropout [VERIFICADO: prototipo_red_modular_fase5.py, evaluar_baselines_fase4.py]. El repositorio ha sido reestructurado en carpetas maestras (`Dataset/`, `Scripts/`, `Resultados/`, `Experimental/`), con repositorio Git inicializado, suite experimental y mapa de custodia exclusivo entre Git y Hugging Face [VERIFICADO: git status, preparar_huggingface.py].

## 2. Identidad y Alcance
* Titulo preliminar: Modelo Neuronal Modular de Prediccion de Impacto Agricola por Ciclo ENSO en Colombia [ORIGEN POR CONFIRMAR].
* Objetivo: Cuantificar y predecir la eficiencia de rendimiento agricola municipal condicionada por anomalias oceanicas globales e interacciones biofisicas locales [VERIFICADO: PLAN_DE_ACCION.md].
* Cultivos (5 categorias): Cafe, Cacao, Frijol, Maiz tradicional y Maiz tecnificado (como categorias separadas) [VERIFICADO: dataset_maestro_andina_caribe.csv].
* Departamentos (10): Huila, Antioquia, Tolima (Andina); Atlantico, Bolivar, Cesar, Cordoba, La Guajira, Magdalena, Sucre (Caribe) [VERIFICADO: dataset_maestro_andina_caribe.csv].
* Cobertura temporal: 2007 a 2024 (18 anos agricolas continuos, mas 2006 para rezagos t-1 observados) [VERIFICADO: clima_anual_municipios_2007_2024.csv].
* Unidad de analisis: Municipio-Cultivo-Ano [VERIFICADO: dataset_maestro_andina_caribe.csv].
* Audiencia: Semillero de investigacion y comunidad academica agroclimatica [VERIFICADO: ESTADO_DEL_ARTE.md].
* Repositorio GitHub: `https://github.com/ke1npro/AgroENSO` [VERIFICADO: gh repo].
* Despliegue GitHub Pages: `https://ke1npro.github.io/AgroENSO/` [VERIFICADO: HTTP 200 en produccion].
* Revista candidata: Ingenieria y Competitividad (Universidad del Valle) [ORIGEN POR CONFIRMAR, no decidida].

## 3. Estado por Fase del Plan de Accion
Avance general: 21 de 22 entregables planificados completados y verificados (95.5% real, 21/22) [VERIFICADO: conteo directo].

| Fase | Entregables Planificados | Entregables Existentes Verificados | Estado | Evidencia |
| :--- | :--- | :--- | :---: | :--- |
| Fase 1: EVA | Limpieza esquema, resolucion bug parsing, colapso maiz, panel balanceado | Dataset/clean/eva_2007_2024_consolidado.csv, Dataset/clean/eva_2007_2024_caribe.csv, eva_panel_balanceado.csv | Completada (4/4) | [VERIFICADO: pipeline_canonico.py, pipeline_eva_caribe.py] |
| Fase 2: Clima & ENSO | Serie ONI mensual/anual, series climaticas locales, normales IDEAM, indicadores calidad | Dataset/raw/oni_mensual_noaa.csv, Dataset/clean/oni_anual_features.csv, clima_mensual_*.csv, clima_anual_*.csv | Completada (5/5) | [VERIFICADO: obtencion_clima.py, obtencion_clima_caribe.py] |
| Fase 3: Feature Engineering | Rendimiento potencial, tabla perfil agronomico, variables estres, dataset maestro | Dataset/clean/dataset_maestro_entrenamiento.csv, dataset_maestro_entrenamiento_caribe.csv, dataset_maestro_andina_caribe.csv | Completada (4/4) | [VERIFICADO: construir_dataset_maestro.py, construir_dataset_maestro_caribe.py] |
| Fase 4: Baselines | Script LOEO baselines, benchmarking 4 modelos, ablacion ENSO, reporte tecnico HTML | Scripts/evaluar_baselines_fase4.py, evaluar_baselines_caribe.py, evaluar_xgboost_andina_caribe.py, Resultados/baselines_*.csv | Completada (4/4) | [VERIFICADO: Scripts/evaluar_baselines_fase4.py, Resultados/] |
| Fase 5: Red Neuronal | Arquitectura doble rama PyTorch, ablacion Sigmoide vs FiLM, MC Dropout | Scripts/prototipo_red_modular_fase5.py, Resultados/red_modular_*.csv, Experimental/ejecutar_experimentos.py | Completada (3/3) | [VERIFICADO: Scripts/prototipo_red_modular_fase5.py, Experimental/] |
| Fase 6: Criterios & Paper | Explorador escenarios, analisis correlacion error-estacion, manuscrito IMRaD | dashboard_tecnico/ (index.html, exclusivo.html), Experimental/index.html; manuscrito pendiente | En curso (1/2) | [VERIFICADO: dashboard_tecnico/, Experimental/index.html] |

## 4. Inventario de Archivos Clave
Generado automaticamente mediante `generar_inventario.py` el 2026-10-09 [VERIFICADO: inventario_archivos.csv]. Total catalogado en repositorio: 53 archivos.

| Ruta Relativa | Rol | Filas | Columnas | Tamano (KB) | SHA-256 (prefijo 12 car.) | Fecha Modificacion | Proposito |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| dataset_maestro_entrenamiento.csv | Final | 18,425 | 114 | 12,867.5 | 3907eb64ea2d | 2026-10-06 17:15:47 | Dataset maestro principal para entrenamiento y evaluacion |
| dataset_maestro_panel_balanceado.csv | Final | 13,545 | 114 | 9,453.4 | 127e4fa16fa7 | 2026-10-06 17:15:48 | Subconjunto de municipios del panel balanceado estricto |
| baselines_resultados_completos.csv | Final | 24 | 15 | 4.0 | 973c1cbb1d3d | 2026-10-08 12:28:00 | Metricas de los 4 baselines en 3 splits y 2 configuraciones |
| baselines_predicciones_test.csv | Final | 33,436 | 15 | 4,005.2 | c29aa98e5473 | 2026-10-08 12:28:00 | Predicciones individuales en test con residuos y metadata |
| baselines_feature_importances.csv | Final | 182 | 3 | 6.8 | 39955fa63162 | 2026-10-08 12:28:00 | Importancias de caracteristicas de RF y XGBoost |
| baselines_curvas_aprendizaje.csv | Final | 44 | 3 | 0.9 | e7ef7aee4b15 | 2026-10-08 12:28:00 | Perdida de XGBoost por iteracion en Split 1 |
| eva_2007_2024_consolidado.csv | Intermedio | 18,591 | 15 | 2,180.3 | e5e3f568a2bf | 2026-09-14 10:48:00 | Base consolidada municipal EVA depurada (Fase 1) |
| eva_panel_balanceado.csv | Intermedio | 13,619 | 15 | 1,591.8 | 6e480f2d969e | 2026-09-14 10:48:00 | Base filtrada a los 191 municipios balanceados |
| eva_departamento_cultivo_anio.csv | Intermedio | 266 | 7 | 18.4 | f1eb3606f1d8 | 2026-09-14 10:48:00 | Rendimiento agregado depto-cultivo-ano |
| eva_departamento_tendencia_potencial.csv | Intermedio | 266 | 9 | 22.1 | f5a034ee5112 | 2026-10-06 17:15:43 | Rendimiento potencial suavizado excluyendo anos Nino severo |
| clima_anual_municipios_2007_2024.csv | Intermedio | 3,744 | 35 | 862.7 | 4a3e20e8b09b | 2026-10-06 11:29:21 | Series climaticas anuales con anomalias y rezagos |
| clima_mensual_municipios_2007_2024.csv | Intermedio | 44,928 | 12 | 3,126.4 | d531a74d284d | 2026-10-06 11:29:20 | Series climaticas mensuales hibridas calibradas |
| oni_anual_features.csv | Intermedio | 76 | 27 | 11.8 | 8b5f3ec3df9d | 2026-10-06 11:14:26 | Metricas anuales y fenologicas del ONI (1950-2025) |
| oni_mensual_noaa.csv | Crudo | 920 | 9 | 34.0 | 5a33c2e17740 | 2026-10-06 11:14:25 | Serie mensual oficial del Climate Prediction Center NOAA |
| ideam_normales_climatologicas_estaciones.csv | Crudo | 3,889 | 24 | 595.9 | e7ef7be6e138 | 2026-10-06 11:29:10 | Normales climatologicas oficiales del IDEAM (nsz2-kzcq) |
| municipios_coordenadas_altitud.csv | Intermedio | 208 | 21 | 33.5 | a0415a208031 | 2026-10-06 11:29:19 | Coordenadas DIVIPOLA, altitud DEM y estacion IDEAM cercana |
| perfil_agronomico_cultivos.csv | Intermedio | 5 | 17 | 0.9 | 06385750d99c | 2026-10-06 11:29:43 | Parametros ecofisiologicos de los 5 cultivos |
| dashboard_tecnico/index.html | Web/Dashboard | 260 | - | 15.8 | eeb75628b031 | 2026-10-08 12:28:52 | Estructura del visor tecnico interactivo independiente |
| dashboard_tecnico/data/dashboard_data.js | Web/Dashboard | 1,847 | - | 254.6 | 73e6b7289569 | 2026-10-08 12:28:14 | Payload de datos pre-exportado para consumo sin servidor |

## 5. Como Reproducir
### Entorno y Hardware Verificado
* Sistema Operativo: Windows 11 Pro 64-bit (10.0.26200-SP0) [VERIFICADO: platform.platform()].
* CPU: AMD Ryzen (8 cores fisicos, 16 logicos), 13.84 GB RAM [VERIFICADO: psutil].
* Interprete: Python 3.13.13 en `C:\Users\Lenovo\AppData\Local\Programs\Python\Python313\python.exe` [VERIFICADO: sys.version].
* Librerias clave (`pip freeze`): `pandas==3.0.5`, `numpy==2.5.2`, `scikit-learn==1.9.1`, `xgboost==3.4.1`, `scipy==1.18.1`, `requests==2.34.2`, `psutil==7.2.2`, `openpyxl==3.1.5` [VERIFICADO: pip freeze]. Nota: PyTorch (`torch`) aun no esta instalado en este entorno.

### Orden de Ejecucion y Pipeline
1. `descarga_oni_noaa.py`: Descarga NOAA CPC ONI -> produce `oni_mensual_noaa.csv` y `oni_anual_features.csv` (tiempo: ~3 s).
2. `obtencion_clima.py`: Descarga Datos Abiertos (IDEAM `nsz2-kzcq`, DIVIPOLA `pqwj-3fi4`), DEM SRTM y mallas regionales NASA POWER (2006-2024) -> produce `ideam_normales_climatologicas_estaciones.csv`, `municipios_coordenadas_altitud.csv`, `clima_mensual_municipios_2007_2024.csv` y `clima_anual_municipios_2007_2024.csv` (tiempo: ~15 s).
3. `construir_dataset_maestro.py`: Estima potencial departamental en `eva_departamento_tendencia_potencial.csv` y cruza EVA con Clima, ONI y `perfil_agronomico_cultivos.csv` -> genera `dataset_maestro_entrenamiento.csv` y `dataset_maestro_panel_balanceado.csv` (tiempo: ~6 s).
4. `evaluar_baselines_fase4.py`: Entrena Ridge, Huber, Random Forest y XGBoost bajo 3 splits temporales LOEO con y sin ONI -> produce `baselines_resultados_completos.csv`, `baselines_predicciones_test.csv`, `baselines_feature_importances.csv` y `baselines_curvas_aprendizaje.csv` (tiempo: ~110 s). Semilla aleatoria fija: `random_state=42`.
5. `exportar_metricas_dashboard.py`: Consolida metricas hacia `dashboard_tecnico/data/dashboard_data.js` (tiempo: ~2 s).
6. Visualizacion: Doble clic en `dashboard_tecnico/index.html` (no requiere servidor local ni conexion activa).

### Chequeos de Forma Automatizados
* `dataset_maestro_entrenamiento.csv`: Exactamente 18,425 filas x 114 columnas [VERIFICADO].
* Valores nulos (`NaN`) en predictores: Exactamente 0 NaNs en las 113 columnas no textuales [VERIFICADO]. La unica columna con nulos es `subcategoria` (texto descriptivo original de EVA).
* Conteo municipal: Exactamente 208 municipios en 3 departamentos (Antioquia: 124, Tolima: 47, Huila: 37) [VERIFICADO].

### Latencia y Throughput de Inferencia (TPS)
Medido sobre `dataset_maestro_entrenamiento.csv` en lote unitario (batch=1) y lote de prueba (batch=2,070 filas) [VERIFICADO: verificaciones_handoff.py]:

| Modelo | Latencia Batch=1 (ms) | Throughput Batch=1 (TPS) | Throughput Batch=2,070 (TPS) | Tiempo Batch=2,070 (ms) |
| :--- | :---: | :---: | :---: | :---: |
| Ridge | 0.120 | 8,358.3 | 6,449,200.8 | 0.32 |
| Huber | 0.126 | 7,929.3 | 7,617,014.9 | 0.27 |
| Random Forest (160 arboles) | 32.990 | 30.3 | 40,551.8 | 51.05 |
| XGBoost (220 arboles) | 0.946 | 1,057.3 | 500,361.4 | 4.14 |

## 6. Procedencia de Datos
| Fuente Primaria | Identificador / Endpoint | Fecha Descarga | Condiciones de Uso | Cobertura Real Verificada |
| :--- | :--- | :---: | :--- | :--- |
| EVA 2007-2018 | Portal AgroNet (MADR) | 2026-08-29 | Publico institucional | 12,693 filas en los 3 departamentos [VERIFICADO] |
| EVA 2019-2024 | Datos Abiertos Colombia / UPRA (`2019_-_2025`) | 2026-08-29 | Licencia abierta gubernamental | 5,898 filas validas en los 3 departamentos [VERIFICADO] |
| Normales IDEAM | Datos Abiertos Socrata `nsz2-kzcq` | 2026-10-06 | Datos Abiertos Colombia | 3,889 estaciones en Ant/Hui/Tol (473 precip, 109 temp) [VERIFICADO] |
| Coordenadas DANE | Datos Abiertos Socrata `pqwj-3fi4` (DIVIPOLA) | 2026-10-06 | Datos Abiertos Colombia | 208 municipios (corregido Mariquita `73443`) [VERIFICADO] |
| Elevacion DEM | Open-Meteo SRTM 90m (`/v1/elevation`) | 2026-10-06 | Abierta no comercial | 208 cabeceras municipales [VERIFICADO] |
| Clima Historico | NASA POWER Regional API (`/temporal/monthly/regional`) | 2026-10-06 | Abierta NASA LaRC | Malla 1.5N-9.0N, 77.2W-73.8W (80 nodos, 2006-2024) [VERIFICADO] |
| Indice ONI | NOAA CPC (`https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt`) | 2026-10-06 | Dominio publico NOAA | 920 meses (1950 a 2026) [VERIFICADO] |

## 7. Diccionario del Dataset Maestro
El dataset `dataset_maestro_entrenamiento.csv` contiene 114 columnas clasificadas funcionalmente en cinco grupos [VERIFICADO: auditoria_puntos_2_3_4.py]:

1. Grupo 1 - Metadatos, Llaves y Particion (20 columnas, NO entran como predictores): Identificadores geograficos (`departamento`, `municipio`, `cod_dane_dep`, `cod_dane_mun`), taxonomicos (`cultivo`, `subcategoria`, `cultivo_cat`), cronologicos (`año`, `periodo`, `fuente`), panel (`en_panel_balanceado`), indicadores de calidad de estacion (`ideam_estacion_precip_cod`, `ideam_dist_estacion_precip_km`, `ideam_estacion_temp_cod`, `ideam_dist_estacion_temp_km`, `tiene_estacion_precip_directa`, `tiene_estacion_temp_directa`, `calidad_dato_climatico_alta`) y fases categoricas (`fase_enso_anual`, `categoria_evento_anual`).
2. Grupo 2 - Variables Agricolas Observadas y Objetivos (9 columnas, NO entran como predictores por fuga estricta): `area_sembrada`, `area_cosechada`, `produccion`, `rendimiento`, `rendimiento_potencial_dep`, `eficiencia` (target secundario/sensibilidad), `rendimiento_base_municipal`, `eficiencia_rel_municipal` (diagnostico municipal puro), `eficiencia_clip` (target principal de entrenamiento).
3. Grupo 3 - Rama 1: Contexto Biofisico Zonal y de Cultivo (26 columnas predictoras continuas): 9 zonales (`latitud_dane`, `longitud_dane`, `altitud_msnm`, `ideam_normal_precip_anual_mm`, `ideam_normal_temp_media_c`, normales climatologicas municipales) y 17 del cultivo (`es_permanente`, `via_fotosintetica_c4`, `nivel_tecnificacion`, rangos optimos termicos y pluviometricos, franja altitudinal y desviacion altitudinal).
4. Grupo 4 - Rama 2: Senal Climatica Local y ENSO (53 columnas predictoras continuas): 24 del indice ONI (medias, acumulados, extremos, duracion de rachas, ventanas semestrales/trimestrales y rezagos t-1) y 29 de clima local municipal (precipitacion anual, semestral, trimestral, dias lluvia, temperaturas media, max, min, rango termico, anomalias % y z-scores, rezagos t-1 e interacciones ONI x anomalia).
5. Grupo 5 - Interacciones Explicitas de Estres Agroclimatico (6 columnas predictoras continuas): `estres_calor_c`, `estres_frio_c`, `desviacion_temp_optima_c`, `ratio_hidrico_cultivo`, `deficit_hidrico_mm`, `exceso_hidrico_mm`.

Convenciones tecnicas:
* Codigo DANE: Siempre string de 5 digitos con cero inicial preservado (ej. `05001`) [VERIFICADO].
* Formato numerico: Punto decimal estandar IEEE 754, sin separadores de miles [VERIFICADO].
* Total predictores continuos combinados (Grupos 3 + 4 + 5): 85 variables continuas [VERIFICADO]. Con codificacion One-Hot de `cultivo_cat` y `departamento`, la matriz de entrenamiento alcanza 91 columnas [VERIFICADO].
* Detalle de nombres y tipos: Referenciado directamente en el script ejecutable `auditoria_puntos_2_3_4.py`.

## 8. Registro de Decisiones
| Decision Adoptada | Motivo Tecnico | Alternativas Descartadas | Origen | Evidencia |
| :--- | :--- | :--- | :---: | :--- |
| No modelar ENSO internamente | Modelar la fisica del Pacifico Ecuatorial excede el alcance del semillero y tiene pronosticos globales consolidados | Construir red LSTM para pronosticar SST marina | [PROPUESTA ACEPTADA] | PLAN_DE_ACCION.md |
| Uso de ONI historico y evaluacion futura de RONI | NOAA adopto RONI como oficial en 2026; la serie historica consolidada 1950-2024 sigue referenciada al ONI | Reentrenar serie historica completa con RONI | [PROPUESTA ACEPTADA] | PLAN_DE_ACCION.md Sec. 2.1 |
| Target principal `eficiencia_clip` con Huber Loss | El valor maximo sin recortar es 7.22 frente a mediana 0.93; los outliers de digitacion dominan el gradiente MSE | Rendimiento bruto t/ha (sesgo por cultivo) o eficiencia sin recortar con MSE | [DECISIÓN DEL USUARIO] | PLAN_DE_ACCION.md Sec. 3.1 |
| `eficiencia_rel_municipal` solo como diagnostico | Centrar por municipio diluye las diferencias biofisicas de primer orden de la Rama 1 | Usar eficiencia municipal como target principal | [DECISIÓN DEL USUARIO] | auditoria_puntos_2_3_4.py |
| Potencial calculado a nivel depto-cultivo | Los municipios con vacios historicos no permiten estimar tendencia tecnologica local confiable | Potencial por municipio (inestable con pocos anos) | [PROPUESTA ACEPTADA] | construir_dataset_maestro.py |
| Fuentes EVA: B (AgroNet 2007-2018) + A (UPRA 2019-2024) | Maxima cobertura temporal de 18 anos conciliando el quiebre de esquema institucional | Descartar 2007-2018 y usar solo Datos Abiertos 2019-2024 | [DECISIÓN DEL USUARIO] | pipeline_canonico.py |
| Colapso de maiz amarillo y blanco desde 2021 | A partir de 2021 EVA dividio el maiz por color, duplicando registros por municipio-ano | Dejar registros duplicados o eliminar una variedad | [PROPUESTA ACEPTADA] | correccion_maiz.py |
| No exigir panel balanceado estricto | Exigir 18 anos continuos descartaba mas del 50% de municipios en varios cultivos | Descartar municipios no balanceados | [DECISIÓN DEL USUARIO] | PLAN_DE_ACCION.md Sec. 2 |
| Clima por esquema hibrido (IDEAM + DEM + NASA) | Telemetria directa IDEAM cubre solo 62/208 municipios; NASA regional aporta continuidad 2006-2024 | Open-Meteo (bloqueo HTTP 429) o estaciones IDEAM solas | [DECISIÓN DEL USUARIO] | obtencion_clima.py |
| Validacion Leave-One-Event-Out (LOEO) | Split aleatorio k-fold produce fuga temporal severa en series climaticas | K-fold aleatorio convencional | [PROPUESTA ACEPTADA] | PLAN_DE_ACCION.md Sec. 4.1 |
| Inclusion de XGBoost en baselines | Demostro superar a Random Forest en el antecedente de El Salvador (Posada & Alfaro, 2025) | Evaluar unicamente Ridge y Random Forest | [PROPUESTA ACEPTADA] | ESTADO_DEL_ARTE.md Sec. 2.3 |
| Ablacion formal Sigmoide vs FiLM en Fase 5 | Sigmoide pura no puede invertir signos ni desplazar nivel base en anos neutros | Fijar Sigmoide sin contrastar con FiLM | [DECISIÓN DEL USUARIO] | PLAN_DE_ACCION.md Sec. 5.2 |
| Salida de red neuronal lineal o softplus | Eficiencia es acotada inferiormente en 0 pero puede superar 1.0; sigmoide satura en 1.0 | Salida Sigmoide en capa final de la red | [PROPUESTA ACEPTADA] | PLAN_DE_ACCION.md Sec. 5.2 |
| Incertidumbre con Monte Carlo Dropout | Permite aproximacion bayesiana con una sola red liviana sin costo de entrenar N redes | Deep Ensembles (demanda mas computo) | [PROPUESTA ACEPTADA] | PLAN_DE_ACCION.md Sec. 5.3 |
| Enfoque metodologico formal (Sampieri) | Cuantitativo, alcance correlacional con valor predictivo, diseno no experimental longitudinal | No explicitado en entregas previas | [ORIGEN POR CONFIRMAR] | ESTADO_DEL_ARTE.md Sec. 1 |
| Segregacion de custodia Git vs Hugging Face | Evitar saturacion y bloat de binarios en Git; garantizar reproducibilidad con DOI y DVC en Hugging Face Datasets | Subir archivos de 50 MB directamente a Git o usar Git-LFS | [DECISIÓN DEL USUARIO] | dashboard_tecnico/exclusivo.html |
| Reestructuracion en carpetas maestras | Modularizar proyecto en Dataset/raw, Dataset/clean, Scripts, Resultados y Experimental con paths dinamicos | Dejar scripts y datasets mezclados en la raiz del proyecto | [DECISIÓN DEL USUARIO] | Scripts/paths.py |
| Creacion de suite Experimental y visor simplificado | Permitir iteraciones rapidas (ej. subconjuntos zonales, modelos ablacionados) sin sobrecargar los hallazgos principales | Modificar directamente los scripts de produccion para cada prueba | [DECISIÓN DEL USUARIO] | Experimental/index.html |

## 9. Hallazgos y Errores Corregidos
1. Bug de parsing numerico en produccion de cafe: En datos 2019 de Datos Abiertos, cifras superiores a 999 utilizaban punto como separador de miles y coma decimal. El parser estandar convertia a `NaN`, aparentando una caida ficticia del 90% al 97% en la cosecha. Se resolvio con funcion `to_num_new` en `pipeline_canonico.py` recuperando la totalidad de la serie [VERIFICADO: pipeline_canonico.py].
2. Hipotesis de cambio de unidad en cafe: Se sospechaba que el salto de 2019 se debia a cambio de "cafe cereza" a "pergamino". Tras auditar las fichas tecnicas de UPRA y la serie de produccion agregada nacional, se descarto la hipotesis: era estrictamente un error de parsing de cadenas en software [VERIFICADO: diag_cafe_unidad.py].
3. Duplicidad por variedades de maiz post-2021: El aumento de registros de maiz en Huila y Tolima desde 2021 no era expansion de frontera sino el desglose en Maiz Blanco y Maiz Amarillo. Se soluciono sumando areas y producciones por municipio-ano y recalculando el rendimiento como division directa de los agregados [VERIFICADO: correccion_maiz.py].
4. Hueco de maiz tecnificado en Antioquia (2015-2018): En la base consolidada se observan exactamente 0 registros para maiz tecnificado en Antioquia entre 2015 y 2018 (conteo verificado: 2014=2, 2015=0, 2016=0, 2017=0, 2018=0, 2019=2). Se comprobo que en la base oficial nacional existen 0 registros en ese periodo; 7 de los 10 municipios historicos reportaron en Maiz tradicional en esos cuatro anos. El hueco persiste y es oficial del reporte gubernamental, no un defecto del pipeline [VERIFICADO: verificaciones_handoff.py].
5. Codigos DANE truncados: La concatenacion tabular convertia `cod_dane_mun` a entero perdiendo el cero de Antioquia (ej. `5001`). Se forzo tipo `str.zfill(5)` en todo el flujo [VERIFICADO: pipeline_canonico.py].
6. Artefacto de coordenadas en DIVIPOLA: San Sebastian de Mariquita (`73443`) tenia coordenadas `525.0, -74916667.0` en la tabla `pqwj-3fi4` por perdida de punto decimal. Se corrigio a `5.1989, -74.8897` antes de consultar elevaciones [VERIFICADO: obtencion_clima.py].
7. Referencias bibliograficas inexistentes: Se detecto que revisiones previas generadas por modelos anteriores citaron referencias apocrifas (Romero 2014; Stuepp et al. 2022). Han sido eliminadas y quedan vetadas del manuscrito [VERIFICADO: ESTADO_DEL_ARTE.md].

## 10. Callejones sin Salida Documentados
* API de Reanalisis Open-Meteo Archive (`/v1/archive`): Bloqueo por cuota HTTP 429 tras 4 lotes debido al volumen de 6,575 dias x 4 variables x 208 municipios. Descartada para descarga directa masiva [VERIFICADO: logs de obtencion_clima.py].
* Telemetria cruda directa de estaciones IDEAM en Datos Abiertos (`s54a-sgyg`, `sbwg-7ju4`): Cobertura de apenas 62 de los 208 municipios, sin indices en Socrata para realizar agrupamiento temporal servidor, provocando timeouts continuos (>25 s) [VERIFICADO: probe_ideam_stations.py].
* Portal DHIME-IDEAM: No ofrece API publica programatica masiva; impone interfaces interactivas con CAPTCHA por estacion [VERIFICADO].
* Exigencia de panel balanceado estricto de 18 anos como unico dataset: Reducia la base de 18,591 a 13,619 registros y eliminaba mas del 50% de municipios en cacao y maiz tecnificado. Se conserva como prueba de robustez secundaria [VERIFICADO: PLAN_DE_ACCION.md].

## 11. Limitaciones y Riesgos Abiertos
1. Incertidumbre climatica en municipios sin estacion fisica propia: Para los 139 municipios sin estacion fisica IDEAM cercana, la metrica honesta de incertidumbre es la de validacion cruzada espacial Leave-One-Station-Out (LOO-CV): RMSE = 45.95 mm/mes (R² = 0.8126) en lluvia y RMSE = 0.8972 °C (R² = 0.9597) en temperatura [VERIFICADO: validacion_ideam_vs_nasa_power.py].
2. Posible fuga temporal por ventana centrada en el potencial tecnologico: El rendimiento potencial departamental $Y^{\text{pot}}_{d,c,t}$ se calcula con media movil centrada de 5 anos (`center=True`). En el ano $t$, la ventana incluye $t+1$ y $t+2$. Ademas, la lista de exclusion de anos severos (`[2009, 2015, 2016, 2023]`) no excluye el ano de prueba 2024 ni 2010 (Nina fuerte). En consecuencia, para 2024 el propio rendimiento observado de 2024 aporta aproximadamente un tercio del denominador $Y^{\text{pot}}$, atenuando la anomalia de eficiencia que se pretende predecir [VERIFICADO: verificaciones_handoff.py].
3. Tamano de muestra de episodios independientes: La ventana 2007-2024 contiene unicamente dos eventos canónicos fuertes de El Niño (2015-2016 y 2023-2024), lo que limita la potencia estadistica de la evaluacion Leave-One-Event-Out [VERIFICADO: oni_anual_features.csv].
4. Evaluacion sobre municipios vistos: La particion LOEO evalua generalizacion temporal a eventos climaticos futuros sobre los mismos 208 municipios, pero no evalua generalizacion espacial a municipios no vistos en entrenamiento [VERIFICADO: evaluar_baselines_fase4.py].
5. Rango de la variable objetivo y outliers: La variable `eficiencia` sin recortar alcanza un maximo de 7.2252 frente a una mediana de 0.9297. Si se entrena con MSE sobre la variable sin recortar, el gradiente queda dominado por estas observaciones extremas [VERIFICADO: auditoria_puntos_2_3_4.py].

## 12. Estado de Verificacion de Referencias Bibliograficas
| Referencia | Estado | Evidencia de Verificacion |
| :--- | :---: | :--- |
| Botero y Barnes (2022, Food Security) | [VERIFICADO: busqueda externa previa] | Articulo publicado en Springer Food Security sobre impacto climatico en Colombia |
| Justino et al. (2026, Theor. Appl. Climatol.) | [VERIFICADO: busqueda externa previa] | Preprint en revision formal en TAC sobre ENSO y precipitacion andina |
| Jimenez Benavides y Urrea Urresti (2025, UNAD) | [VERIFICADO: busqueda externa previa] | Tesis de maestria UNAD sobre rendimiento agricola y clima |
| Nobrega Passos, Kaur y Sarmiento Herrera (2023, Neuromatch) | [VERIFICADO: busqueda externa previa] | Trabajo de investigacion Neuromatch Academy sobre Deep Learning agroclimatico |
| Banco de la Republica (2026, Recuadro 1) | [VERIFICADO: busqueda externa previa] | Publicacion tecnica oficial Banco de la Republica sobre inflacion de alimentos y El Niño |
| Posada y Alfaro (2025, IEEE CONCAPAN) | [VERIFICADO: DOI 10.1109/CONCAPAN66820.2025.11512570] | Paper publicado en IEEE CONCAPAN XLIII sobre ML y ENSO para maiz |
| Castro Hernandez y Duran Blanco (2023, UIS) | [VERIFICADO: repositorio Noesis UIS] | Tesis de maestria UIS: cafe y cacao en Santander con ML/DL |
| Gal y Ghahramani (2016, ICML) | [VERIFICADO: PMLR 48:1050-1059] | Paper fundacional de Monte Carlo Dropout como aproximacion bayesiana |
| Lobell y Field (2007, Environ. Res. Lett.) | [VERIFICADO: DOI 10.1088/1748-9326/2/1/014002] | Paper global sobre relaciones clima-rendimiento de cultivos |
| Perez et al. (2018, AAAI) | [VERIFICADO: AAAI-18] | Paper fundacional de FiLM: Visual Reasoning with a General Conditioning Layer |
| Romero (2014) | [DESACREDITADA / INEXISTENTE] | Detectada como alucinacion en documentos tempranos; eliminada del proyecto |
| Stuepp et al. (2022) | [DESACREDITADA / INEXISTENTE] | Detectada como alucinacion en documentos tempranos; eliminada del proyecto |
| Todas las demas citas del informe previo | [NO VERIFICADO] | Requieren comprobacion contra fuente primaria antes de citarse en el manuscrito |

## 13. Proximos Pasos Ordenados y Criterios de Aceptacion
1. Validacion formal de hiperparametros de la Fase 4 con Keiner: Confirmar si se mantiene la configuracion de baselines evaluada o si se refinan hiperparametros mediante GridSearchCV temporal dentro de cada fold.
   * Criterio de aceptacion: Aprobacion explicita de los resultados de `baselines_resultados_completos.csv`.
2. Experimento de recalculacion de potencial tecnologico sin fuga de prueba: Recalcular $Y^{\text{pot}}$ excluyendo completamente los anos del fold de prueba (ej. excluir 2015-2016 al evaluar el Fold 1) y comparar el cambio en $R^2$.
   * Criterio de aceptacion: Script ejecutable que mida la diferencia de atenuacion.
3. Fase 5 - Red Neuronal Modular en PyTorch:
   * Implementar `modelo_red_modular.py` con Rama 1 (Contexto 26 features) y Rama 2 (Clima+ENSO 59 features).
   * Ablacion controlada: Variante A (Gating Sigmoide) vs Variante B (FiLM completo $\gamma = 1 + \tanh, \beta$).
   * Entrenamiento con `nn.HuberLoss` sobre `eficiencia_clip` y estimacion de incertidumbre con 30 pases MC Dropout ($p = 0.2$).
   * Criterio de aceptacion: Reduccion de error $MAE \le 0.08$ y superioridad de al menos 10% en RMSE frente a XGBoost en anos de El Niño severo.
4. Fase 6 - Articulo Cientifico IMRaD:
   * Estructurar el manuscrito con las secciones de Introduccion, Metodologia (detallando el fix de parsing y la correccion de sesgo hibrida), Resultados (tablas de LOEO y graficas de ablacion) y Discusion (limitaciones honestas citando LOO-CV).
   * Criterio de aceptacion: Manuscrito completo listo para envio a revision.

## 14. Para Confirmar con Keiner (Decisiones Abiertas y Origen Incierto)
1. Exclusiones en la tendencia de potencial tecnologico: ¿Se aprueba recalcular $Y^{\text{pot}}$ excluyendo estrictamente los anos de prueba de cada fold para evitar cualquier atenuacion temporal? [ORIGEN POR CONFIRMAR].
2. Eleccion de revista candidata: Se menciono preliminarmente *Ingenieria y Competitividad* (Universidad del Valle), pero no ha sido confirmada formalmente.
3. Uso futuro del indice oceanico: Definir si para simulaciones futuras se adopta el nuevo indice oficial RONI de NOAA o se mantiene ONI tradicional para coherencia historica.
4. Definicion de umbrales de exito del informe previo: Confirmar si los umbrales ($MAE \le 0.08$, calibracion de banda al 90%) se evaluaran a nivel municipal individual o agregado por cultivo.

## 15. Prompt de Arranque para el Proximo Agente
```text
INSTRUCCION DE CONTINUIDAD PARA EL PROXIMO AGENTE:
1. Lee este archivo HANDOFF.md por completo antes de ejecutar cualquier accion o proponer cambios.
2. Ejecuta `python generar_inventario.py` y verifica que los conteos de filas, columnas y hashes coincidan con la Seccion 4 de este documento.
3. Reglas estrictas de trabajo:
   - Todo dato numerico debe ser verificado con codigo en tiempo real; prohibido inventar o asumir metricas.
   - Prohibida la imputacion silenciosa y la fuga temporal en particiones (respetar LOEO estricto).
   - Prohibido ajustar hiperparametros sobre los conjuntos de prueba fuera de muestra.
   - Reportar resultados desfavorables con total honestidad cientifica sin suavizarlos.
4. Consulta y confirma con Keiner cualquier decision listada en la Seccion 14 antes de avanzar con la implementacion de la Fase 5 (Red Neuronal Modular en PyTorch).
```
