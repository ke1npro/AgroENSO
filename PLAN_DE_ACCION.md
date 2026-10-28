# Plan de Acción Consolidado
## Proyecto: Modelo Neuronal de Predicción de Impacto Agrícola por Fenómeno del Niño

**Marco:** Semillero de Investigación  
**Alcance Geográfico:** Departamentos de Huila, Antioquia y Tolima  
**Cultivos Seleccionados:** Café (principal/piloto), Cacao, Maíz (tradicional y tecnificado) y Frijol  
**Rango Temporal:** 2007 – 2024  
**Equipo:** 2 investigadores + Docente Asesor  
**Fecha de consolidación:** Septiembre 2026  

---

## 1. Justificación y Diferenciador Científico

El proyecto aborda la relación entre la variabilidad climática asociada al ciclo ENSO (El Niño-Oscilación del Sur) y el rendimiento agrícola en Colombia con un enfoque metodológico riguroso y publicable (formato IMRaD).

### Diferenciador frente al Estado del Arte
La revisión bibliográfica identificó que los antecedentes existentes en Colombia y la región presentan al menos una de las siguientes limitaciones:
1. **Modelos econométricos clásicos:** Cuantifican impacto macro con regresiones estándar sin capacidad de capturar relaciones altamente no lineales ni interacciones complejas cultivo-zona (ej. estudios de Banco de la República y MinAgricultura).
2. **Machine Learning sin señal ENSO explícita:** El antecedente técnico más cercano en Colombia (tesis UIS, 2023) compara ML y DL para café y cacao en Santander usando clima IDEAM general, pero **omite índices ENSO explícitos** como predictores directos.
3. **Modelos con ENSO fuera de Colombia o sin Deep Learning:** Estudios recientes en la región (como Posada & Alfaro, 2025 en El Salvador para maíz) usan índices de NOAA y modelos como XGBoost/Random Forest, pero no redes neuronales ni contexto colombiano.

**El aporte original y defendible del proyecto radica en:**
$$\text{Red Neuronal Modular} + \text{Índice ENSO Explícito (ONI)} + \text{Interacción Moduladora (Zona} \times \text{Cultivo)} + \text{Cuantificación de Incertidumbre (MC Dropout)}$$

---

## 2. Diagnóstico del Repositorio y Cierre de la Fase 1 (EVA)

La fase de recolección, depuración y validación de las **Evaluaciones Agropecuarias Municipales (EVA)** ha quedado formalmente completada y auditada en el repositorio.

### Resumen de Hallazgos y Correcciones Validadas

| Aspecto Evaluado | Estado Previo / Problema | Solución Implementada / Resultado Auditado | Estado |
| :--- | :--- | :--- | :---: |
| **Unificación de Fuentes** | Dispersión entre AgroNet (2007–2018) y Datos Abiertos/UPRA (2019–2024). | Conciliación de esquemas de columnas, codificación DANE y consolidación cronológica en `pipeline_canonico.py`. | ✅ Cerrado |
| **Bug de Parsing Numérico** | Caída aparente de $>90\%$ en producción de café en 2019 debido a que valores $>999$ se convertían en `NaN` por mal manejo de separadores colombianos (puntos de miles y comas decimales). | Implementación de parser numérico `to_num_new` que estandariza comas y puntos. Recuperación total de datos históricos sin pérdida por magnitud. | ✅ Resuelto |
| **Colapso de Variedades de Maíz** | A partir de 2021, EVA desglosó el maíz en variedades (blanco/amarillo), duplicando registros por municipio-año. | Colapso sistemático para años $\ge 2021$ sumando áreas y producciones, y recalculando el rendimiento como $\frac{\text{Producción}}{\text{Área cosechada}}$ (sin promediar tasas precalculadas). | ✅ Resuelto |
| **Hueco Maíz Tecnificado Antioquia (2015–2018)** | 0 registros observados para maíz tecnificado en Antioquia entre 2015 y 2018. Se sospechaba posible artefacto de software. | **Auditoría confirmada:** No es un bug. En la base oficial nacional existen exactamente 0 registros en ese cuatrienio. 7 de los 10 municipios que reportaban tecnificado antes de 2015 fueron reclasificados como maíz tradicional en ese periodo. | ✅ Explicado |
| **Estrategia de Panel** | Exigir panel balanceado estricto (18 años continuos por municipio) descartaba más del 50% de municipios en varios cultivos. | Se preserva la base consolidada completa (18,591 registros) para modelado general y se aísla el panel balanceado (13,619 registros) para pruebas de robustez. | ✅ Decidido |

### Entregables Agrícolas Disponibles en el Repositorio

1. `eva_2007_2024_consolidado.csv`: Dataset municipal limpio de 3 departamentos y 5 categorías de cultivo (18,591 filas).
2. `eva_departamento_cultivo_anio.csv`: Serie agregada departamento-cultivo-año (266 filas, base para calcular rendimientos potenciales).
3. `eva_panel_balanceado.csv`: Subconjunto con los 191 municipios que reportan ininterrumpidamente los 18 años.
4. `tabla_conteos_departamento_cultivo_anio.csv`: Matriz de completitud anual de reporte.

---

## 3. Plan de Acción por Fases

```
[Fase 1: EVA (Completada)]
       │
       ▼
[Fase 2: Clima & ENSO] ────────► Descarga ONI (NOAA) + Series IDEAM/CHIRPS
       │
       ▼
[Fase 3: Feature Engineering] ──► Variable Eficiencia + Perfil Agronómico + Dataset Maestro
       │
       ▼
[Fase 4: Baselines Clásicos] ───► Ridge + Random Forest + XGBoost (Split por Evento LOEO)
       │
       ▼
[Fase 5: Red Neuronal] ─────────► Arquitectura Modular (Condicionamiento ENSO) + MC Dropout
       │
       ▼
[Fase 6: Validación & Paper] ───► Evaluación Cuantitativa + Escenarios + Manuscrito IMRaD
```

---

### FASE 2: Ingesta de Señal ENSO y Variables Climáticas (Prioridad Inmediata)

#### 2.1. Índice Oceánico El Niño (ONI - NOAA CPC)
* **Objetivo:** Adquirir la señal causal central del modelo.
* **Acciones:**
  1. Descargar la serie histórica mensual del índice ONI (1950–presente) directamente del CPC/NOAA.
  2. Construir variables derivadas relevantes agronómicamente:
     * $ONI_t$: Valor mensual contemporáneo.
     * $ONI_{acum}$: Intensidad acumulada del evento durante el ciclo productivo.
     * Duración en meses del episodio cálido ($ONI \ge +0.5$) o frío ($ONI \le -0.5$).
     * Fase ENSO categórica/continua durante etapas fenológicas clave (floración y llenado de fruto).

#### 2.2. Datos Climáticos Locales (Esquema Híbrido IDEAM + DEM + NASA POWER)
* **Objetivo:** Capturar cómo la anomalía global se manifiesta localmente en precipitación y temperatura con $0\%$ de datos faltantes en los 208 municipios (2007–2024, más 2006 para rezagos reales $t-1$).
* **Variables requeridas:** Precipitación mensual y anual ($\text{mm}$), Días con lluvia/secos, Temperatura Media, Mínima y Máxima ($^\circ\text{C}$).
* **Metodología implementada y métrica honesta de incertidumbre (LOO-CV):**
  1. Se ancla la climatología base mensual de cada municipio mediante interpolación IDW ($k=3$) sobre las **Normales Climatológicas Oficiales del IDEAM (`nsz2-kzcq`, 3,889 registros)** y corrección por gradiente térmico vertical andino ($-0.006\,^\circ\text{C/m}$) con elevación DEM SRTM (90m), modulada por la serie mensual **NASA POWER Agroclimatology (2006–2024)**.
  2. **Incertidumbre climática reportada (Regla metodológica):** En todo reporte y sección de limitaciones se cita exclusivamente el error de **Validación Cruzada Espacial *Leave-One-Station-Out* (LOO-CV)** —ocultando la estación propia del municipio— y no el ajuste in-sample del pipeline final:
     * **Precipitación LOO-CV:** $\text{RMSE} = 45.95\text{ mm/mes}$ ($\text{MAE} = 31.28\text{ mm/mes}$, $R^2 = 0.8126$, $r = 0.9020$) y $\text{RMSE} = 468.01\text{ mm/año}$ ($R^2 = 0.7278$), frente a $92.82\text{ mm/mes}$ ($R^2 = 0.2352$) de NASA POWER sin calibrar.
     * **Temperatura Media LOO-CV:** $\text{RMSE} = 0.8972\,^\circ\text{C}$ ($\text{MAE} = 0.7024\,^\circ\text{C}$, $R^2 = 0.9597$, $r = 0.9805$), frente a $3.8969\,^\circ\text{C}$ ($R^2 = 0.2404$) de NASA POWER sin calibrar. En validación interanual contra telemetría automática cruda (`sbwg-7ju4`, $N = 2,661$), el $\text{RMSE}$ cae de $4.5218\,^\circ\text{C}$ ($R^2 = 0.3886$) a $2.8817\,^\circ\text{C}$ ($R^2 = 0.7517$).
  3. **Trazabilidad de calidad climática por municipio:** Se incorporan en el dataset maestro las columnas `ideam_dist_estacion_precip_km`, `ideam_dist_estacion_temp_km`, `tiene_estacion_precip_directa` ($\le 8\text{ km}$), `tiene_estacion_temp_directa` ($\le 10\text{ km}$) y `calidad_dato_climatico_alta` para auditar si los errores de predicción del modelo final correlacionan con la distancia a la estación física más cercana.

---

### FASE 3: Feature Engineering y Variable Objetivo

#### 3.1. Formulación de la Variable Objetivo: "Eficiencia de Rendimiento" y Manejo de Outliers
Para evitar el sesgo introducido por tecnologías de cultivo dispares y escalas no comparables:
* **No usar:** Rendimiento bruto en $t/\text{ha}$.
* **Definición propuesta (`eficiencia` — Eficiencia Departamental):**
  $$\text{Eficiencia}_{m, c, t} = \frac{Y_{m, c, t}}{Y^{\text{pot}}_{d, c, t}}$$
  Donde:
  * $Y_{m, c, t}$: Rendimiento observado en el municipio $m$, cultivo $c$, año $t$.
  * $Y^{\text{pot}}_{d, c, t}$: Rendimiento potencial o esperado para el departamento $d$ y cultivo $c$ en el año $t$, estimado mediante tendencia tecnológica histórica (media móvil centrada de 5 años excluyendo años de El Niño severo: 2009, 2015, 2016 y 2023).
  * **Estrategia de robustez frente a outliers de digitación en EVA:** Dado que `eficiencia` sin recortar presenta valores extremos puntuales de hasta $7.2252$ frente a una mediana de $0.9297$, se establece **`eficiencia_clip`** (winsorizada a los percentiles $0.5\,\%\text{–}99.5\,\%$, rango $[0.2481, 2.2368]$) como **objetivo principal de entrenamiento** acompañada de función de pérdida robusta (**Huber Loss**), reportando el desempeño sobre `eficiencia` sin recortar como análisis formal de sensibilidad. Asimismo, se conserva `eficiencia_rel_municipal` únicamente como variable auxiliar de diagnóstico local (nunca como predictor).
  * **Interpretación:**
    * $\text{Eficiencia} < 1.0$: Pérdida de rendimiento atribuible a estrés ambiental o menor aptitud local.
    * $\text{Eficiencia} \approx 1.0$: Rendimiento conforme al potencial tecnológico del departamento.
    * $\text{Eficiencia} > 1.0$: Rendimiento sobresaliente (condiciones favorables).

#### 3.2. Perfil Agronómico de Cultivos (Variables Continuas)
Estructuración de matriz de características de especie para habilitar transferencia multi-cultivo:
* Temperatura óptima mínima y máxima ($^\circ\text{C}$).
* Requerimiento hídrico óptimo anual ($\text{mm}$).
* Rango altitudinal recomendado ($\text{m.s.n.m.}$).
* Ciclo fenológico: Permanente (Café, Cacao) vs. Transitorio (Maíz, Frijol).

#### 3.3. Características Biofísicas Zonales
* Altitud media municipal ($\text{m.s.n.m.}$).
* Clima base histórico (precipitación y temperatura media climatológica IDEAM).

---

### FASE 4: Estrategia de Validación y Modelos de Referencia (Baselines)

#### 4.1. Partición Temporal: Leave-One-Event-Out (LOEO)
* **Regla estricta:** Queda prohibido el split aleatorio ($k$-fold estándar), dado que genera fuga de información temporal y sobreestima el rendimiento real.
* **Estrategia LOEO:**
  * Se entrena con todos los años neutros y eventos previos.
  * Se evalúa la capacidad de generalización testeando en un evento completo de El Niño fuera de muestra (ej. Test en evento 2015–2016 o 2023–2024).

#### 4.2. Benchmarking Obligatorio
Antes de validar la red neuronal, se entrenarán e hiper-parametrizarán sobre `eficiencia_clip` (evaluando también contra `eficiencia` sin recortar):
1. **Regresión Lineal Regularizada (Ridge / HuberRegressor):** Baseline de interpretabilidad directa.
2. **Random Forest Regressor:** Modelo no lineal de ensamble.
3. **XGBoost Regressor (con pérdida `reg:pseudohubererror` / `reg:squarederror`):** Baseline competitivo del estado del arte (demostró superar a RF en El Salvador 2025).

---

### FASE 5: Diseño del Modelo Neuronal y Cuantificación de Incertidumbre

#### 5.1. Selección de Framework
* **Decisión:** **PyTorch** con función de pérdida robusta **Huber Loss (`nn.HuberLoss`)**.
* **Motivo:** Máxima flexibilidad para arquitecturas de doble rama y mecanismos de modulación personalizados; compatibilidad directa en laptops, Colab y Kaggle; estándar de facto en publicaciones científicas actuales.

#### 5.2. Principio de Arquitectura: Modulación de la Señal ENSO (Ablación Explícita Gating Sigmoide vs. FiLM)
Las variables de zona y cultivo no son causas independientes simétricas; son **moduladores** de la señal climática.
* **Rama 1 (Contexto Agronómico y Zonal, 26 predictores):**
  $$\mathbf{h}_{\text{ctx}} = f_{\text{MLP}}(\mathbf{x}_{\text{cultivo}}, \mathbf{x}_{\text{zona}})$$
* **Rama 2 (Señal Climática y ENSO, 53 predictores + 6 de estrés):**
  $$\mathbf{h}_{\text{clima}} = g_{\text{MLP}}(\mathbf{x}_{\text{ONI}}, \mathbf{x}_{\text{clima\_local}})$$
* **Capa de Interacción / Modulación (Decisión explícita con estudio de ablación):**
  Se evalúan formalmente dos mecanismos de condicionamiento en validación LOEO:
  1. **Variante A — Compuerta Multiplicativa Sigmoide:**
     $$\mathbf{z} = \mathbf{h}_{\text{clima}} \odot \sigma(\mathbf{W}_\gamma \mathbf{h}_{\text{ctx}} + \mathbf{b}_\gamma)$$
  2. **Variante B — Modulación Lineal FiLM Completa (*Feature-wise Linear Modulation*):**
     $$\mathbf{z} = \bigl(1 + \tanh(\mathbf{W}_\gamma \mathbf{h}_{\text{ctx}} + \mathbf{b}_\gamma)\bigr) \odot \mathbf{h}_{\text{clima}} + (\mathbf{W}_\beta \mathbf{h}_{\text{ctx}} + \mathbf{b}_\beta)$$
     *(Permite tanto escalar/invertir la sensibilidad al choque climático mediante $\gamma$ como preservar el nivel base de aptitud biofísica del municipio en años neutros mediante el desplazamiento aditivo $\beta$).*
* **Salida:**
  $$\hat{y} = \text{Dense}(\mathbf{z}) \quad \longrightarrow \quad \widehat{\text{Eficiencia}}$$

#### 5.3. Cuantificación de Incertidumbre
* **Técnica:** **Monte Carlo Dropout** (activado en inferencia).
* Durante la predicción se ejecutan $S = 30$ pases hacia adelante:
  $$\mu = \frac{1}{S} \sum_{s=1}^S \hat{y}^{(s)}, \qquad \sigma^2 = \frac{1}{S} \sum_{s=1}^S (\hat{y}^{(s)} - \mu)^2$$
* Toda predicción reportará: **Valor esperado $\mu \pm 1.96\sigma$ (Intervalo de confianza al 95%)**.

---

### FASE 6: Criterios de Éxito, Simulación y Publicación

#### 6.1. Criterios Numéricos de Éxito
Para considerar que el modelo es viable para publicación y defensa ante el semillero:
1. **Reducción de error:** $MAE \le 0.08$ en la escala de eficiencia (error absoluto medio menor al $8\%$ del rendimiento potencial).
2. **Superioridad en eventos severos:** La red neuronal debe reducir el $RMSE$ al menos en un $10\%$ respecto a XGBoost en años de El Niño fuerte ($ONI > 1.5$).
3. **Calibración del intervalo:** Al menos el $90\%$ de las observaciones reales deben quedar contenidas dentro de la banda de incertidumbre del $95\%$ generada por MC Dropout.
4. **Auditoría de error por calidad climática:** Evaluar y reportar la correlación entre el error residual $|\hat{y} - y|$ y la distancia a la estación física del IDEAM (`ideam_dist_estacion_precip_km`, `calidad_dato_climatico_alta`).

#### 6.2. Explorador de Escenarios (Uso del Pronóstico IRI/CPC)
* Construcción de una función de inferencia para simular el impacto en los 4 cultivos ante diferentes escenarios del pronóstico probabilístico de la NOAA/IRI (escenario Niño Fuerte, Moderado, Débil o Neutro).

#### 6.3. Redacción del Artículo Científico (Estructura IMRaD)
* **Introducción:** Vulnerabilidad agrícola andina ante ENSO; vacío de literatura en redes neuronales con modulación explícita.
* **Metodología:** Limpieza y resolución de artefactos en EVA; formulación de eficiencia (`eficiencia_clip` + Huber Loss); arquitectura modular PyTorch (ablación Gating Sigmoide vs. FiLM) con MC Dropout; validación Leave-One-Event-Out.
* **Resultados:** Tabla comparativa vs. Baselines (Ridge, RF, XGBoost); curvas de calibración de incertidumbre; análisis de error por calidad de estación climática.
* **Discusión y Limitaciones:** Cobertura espacial municipal; incertidumbre climática honesta medida por validación cruzada espacial **LOO-CV** ($\text{RMSE} = 45.95\text{ mm/mes}$ en precipitación y $0.8972\,^\circ\text{C}$ en temperatura); dependencia del reporte oficial; límites en la extrapolación a eventos climáticos sin precedentes.

---

## 4. Hoja de Ruta de Ejecución Inmediata

| Tarea Inmediata | Entregable Concreto | Herramienta / Fuente |
| :--- | :--- | :--- |
| **Paso 1** | Script de descarga y parseo de serie histórica mensual ONI (1950–2024). | Python (`requests`, `pandas`), NOAA CPC. |
| **Paso 2** | Script de cálculo de variable objetivo `eficiencia` sobre `eva_departamento_cultivo_anio.csv`. | Python (`numpy`, `pandas`). |
| **Paso 3** | Definición de tabla agronómica de requerimientos por cultivo. | Archivo de metadatos `perfil_agronomico_cultivos.csv`. |
| **Paso 4** | Integración de precipitación/temperatura mensual (IDEAM / CHIRPS). | Script `obtencion_clima.py`. |
| **Paso 5** | Consolidación del dataset unificado `dataset_maestro_entrenamiento.csv`. | Pipeline de cruce tabular. |
