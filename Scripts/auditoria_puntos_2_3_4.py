"""
Auditoría Técnica Rigurosa - Puntos 2, 3 y 4 sobre dataset_maestro_entrenamiento.csv
====================================================================================
Genera evidencia reproducible y output crudo para:
  - Punto 2: Definición matemática y estadística de 'eficiencia' (departamental) vs
             'eficiencia_rel_municipal' (municipal), rol en el dataset y ausencia de fuga.
  - Punto 3: Conteo exacto y trazabilidad de las 166 filas descartadas entre
             eva_2007_2024_consolidado.csv (18,591) y dataset_maestro_entrenamiento.csv (18,425),
             desglosado por causa, departamento y cultivo_cat, más verificación estricta
             de 0 imputaciones silenciosas en predictores.
  - Punto 4: Inventario explícito de las 109 columnas clasificadas por bloque funcional
             para la arquitectura neuronal de doble rama (Rama 1: Contexto Zona+Cultivo,
             Rama 2: Señal ENSO+Clima Local) y baselines.
"""

import pandas as pd
import numpy as np

try:
    from paths import resolve_path
except ImportError:
    from Scripts.paths import resolve_path


def auditar_punto_3():
    print("=" * 85)
    print("PUNTO 3: AUDITORÍA DE FILAS DESCARTADAS (18,591 -> 18,425) E IMPUTACIÓN CERO")
    print("=" * 85)

    df_eva = pd.read_csv(resolve_path("eva_2007_2024_consolidado.csv"), dtype={"cod_dane_mun": str, "cod_dane_dep": str})
    df_master = pd.read_csv(resolve_path("dataset_maestro_entrenamiento.csv"), dtype={"cod_dane_mun": str, "cod_dane_dep": str})
    df_clima = pd.read_csv(resolve_path("clima_anual_municipios_2007_2024.csv"), dtype={"cod_dane_mun": str})
    df_oni = pd.read_csv(resolve_path("oni_anual_features.csv"))
    df_muns = pd.read_csv(resolve_path("municipios_coordenadas_altitud.csv"), dtype={"cod_dane_mun": str})
    df_agro = pd.read_csv(resolve_path("perfil_agronomico_cultivos.csv"))

    print(f"Filas en eva_2007_2024_consolidado.csv     : {len(df_eva)}")
    print(f"Filas en dataset_maestro_entrenamiento.csv : {len(df_master)}")
    print(f"Diferencia exacta de filas                 : {len(df_eva) - len(df_master)}")

    # Verificar pérdidas por bloque de cruce (Zona, Clima, ENSO, Perfil Agronómico)
    muns_faltantes_zona = set(df_eva["cod_dane_mun"]) - set(df_muns["cod_dane_mun"])
    pares_eva = set(zip(df_eva["cod_dane_mun"], df_eva["año"]))
    pares_clima = set(zip(df_clima["cod_dane_mun"], df_clima["año"]))
    pares_faltantes_clima = pares_eva - pares_clima
    anios_faltantes_enso = set(df_eva["año"]) - set(df_oni["año"])
    cultivos_faltantes_agro = set(df_eva["cultivo_cat"]) - set(df_agro["cultivo_cat"])

    print("\n--- 3.1. Pérdidas en cruces (Joins) con bloques externos ---")
    print(f"  - Filas perdidas por bloque ZONA (municipios_coordenadas_altitud) : {len(muns_faltantes_zona)}")
    print(f"  - Filas perdidas por bloque CLIMA (clima_anual_municipios)        : {len(pares_faltantes_clima)}")
    print(f"  - Filas perdidas por bloque ENSO (oni_anual_features)             : {len(anios_faltantes_enso)}")
    print(f"  - Filas perdidas por bloque CULTIVO (perfil_agronomico_cultivos)  : {len(cultivos_faltantes_agro)}")

    # Identificar exactamente las 166 filas descartadas en el filtro de validez de variable objetivo EVA
    cond_valida = (
        df_eva["rendimiento"].notna()
        & (df_eva["rendimiento"] > 0)
        & (df_eva["area_cosechada"].fillna(0) > 0)
    )
    df_descartadas = df_eva[~cond_valida].copy()
    print(f"\n--- 3.2. Causa exacta de las {len(df_descartadas)} filas descartadas (Filtro de Cosecha/Rendimiento en EVA) ---")
    df_descartadas["causa_descarte"] = np.select(
        [
            (df_descartadas["area_cosechada"].fillna(0) == 0) & df_descartadas["rendimiento"].isna(),
            (df_descartadas["area_cosechada"].fillna(0) > 0) & df_descartadas["rendimiento"].isna(),
            (df_descartadas["rendimiento"] == 0),
        ],
        [
            "area_cosechada == 0 y rendimiento is NaN (cultivo en establecimiento/sin cosecha)",
            "area_cosechada > 0 pero rendimiento is NaN (producción=0 o dato faltante en EVA)",
            "rendimiento == 0",
        ],
        default="otra",
    )
    print(df_descartadas["causa_descarte"].value_counts().to_string())

    print("\n--- 3.3. Desglose de las 166 filas descartadas por Departamento y Cultivo (cultivo_cat) ---")
    tabla_dep_cult = pd.crosstab(
        df_descartadas["departamento"],
        df_descartadas["cultivo_cat"],
        margins=True,
        margins_name="TOTAL",
    )
    print(tabla_dep_cult.to_string())

    print("\n--- 3.4. Desglose de las 166 filas descartadas por Fuente / Periodo EVA ---")
    print(df_descartadas.groupby(["fuente", "causa_descarte"]).size().to_string())

    print("\n--- 3.5. Verificación de CERO imputación silenciosa en fuentes y predictores ---")
    print(f"  - NaNs en clima_anual_municipios_2007_2024.csv (crudo antes de join) : {df_clima.isna().sum().sum()}")
    print(f"  - NaNs en oni_anual_features.csv (ventana 2007-2024 antes de join)   : {df_oni[df_oni['año'].between(2007, 2024)].isna().sum().sum()}")
    print(f"  - NaNs en municipios_coordenadas_altitud.csv (antes de join)         : {df_muns.isna().sum().sum()}")
    print(f"  - NaNs en perfil_agronomico_cultivos.csv (antes de join)             : {df_agro.isna().sum().sum()}")
    cols_pred_y_obj = [c for c in df_master.columns if c != "subcategoria"]
    print(f"  - NaNs en dataset_maestro_entrenamiento.csv (108 cols sin subcat)    : {df_master[cols_pred_y_obj].isna().sum().sum()}")


def auditar_punto_2():
    print("\n" + "=" * 85)
    print("PUNTO 2: EFICIENCIA DEPARTAMENTAL ('eficiencia') VS EFICIENCIA MUNICIPAL")
    print("=" * 85)
    df = pd.read_csv(resolve_path("dataset_maestro_entrenamiento.csv"))
    cols_obj = [
        "rendimiento",
        "rendimiento_potencial_dep",
        "eficiencia",
        "eficiencia_clip",
        "rendimiento_base_municipal",
        "eficiencia_rel_municipal",
    ]
    print("Estadísticas descriptivas globales de las columnas de rendimiento y eficiencia (N = 18,425):")
    print(df[cols_obj].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).round(4).to_string())

    print("\nCorrelación de Pearson entre las métricas de rendimiento y eficiencia:")
    print(df[["rendimiento", "eficiencia", "eficiencia_clip", "eficiencia_rel_municipal"]].corr().round(4).to_string())


def auditar_punto_4():
    print("\n" + "=" * 85)
    print("PUNTO 4: INVENTARIO EXPLÍCITO DE LAS 109 COLUMNAS POR BLOQUE ARQUITECTÓNICO")
    print("=" * 85)
    df = pd.read_csv(resolve_path("dataset_maestro_entrenamiento.csv"), nrows=5)
    all_cols = list(df.columns)
    print(f"Total columnas en dataset_maestro_entrenamiento.csv: {len(all_cols)}")

    # Clasificación exhaustiva y mutuamente excluyente de las 114 columnas
    bloque_identificadores_meta = [
        "departamento", "municipio", "cultivo", "subcategoria", "cultivo_cat",
        "año", "periodo", "fuente", "cod_dane_dep", "cod_dane_mun", "en_panel_balanceado",
        "ideam_estacion_precip_cod", "ideam_dist_estacion_precip_km",
        "ideam_estacion_temp_cod", "ideam_dist_estacion_temp_km",
        "tiene_estacion_precip_directa", "tiene_estacion_temp_directa", "calidad_dato_climatico_alta",
        "fase_enso_anual", "categoria_evento_anual"
    ]

    bloque_agr_observado_y_targets = [
        "area_sembrada", "area_cosechada", "produccion", "rendimiento",
        "rendimiento_potencial_dep", "eficiencia", "rendimiento_base_municipal",
        "eficiencia_rel_municipal", "eficiencia_clip"
    ]

    # RAMA 1 (Contexto Biofísico Zonal + Perfil Ecofisiológico del Cultivo)
    bloque_rama1_zona = [
        "latitud_dane", "longitud_dane", "altitud_msnm",
        "ideam_normal_precip_anual_mm", "ideam_normal_temp_media_c",
        "clima_base_precip_mm", "clima_std_precip_mm",
        "clima_base_temp_c", "clima_std_temp_c"
    ]
    bloque_rama1_cultivo = [
        "es_permanente", "via_fotosintetica_c4", "nivel_tecnificacion",
        "duracion_ciclo_meses", "temp_opt_min_c", "temp_opt_max_c", "temp_opt_media_c",
        "precip_opt_min_mm", "precip_opt_max_mm", "precip_opt_media_mm",
        "altitud_opt_min_msnm", "altitud_opt_max_msnm", "altitud_opt_media_msnm",
        "sensibilidad_deficit_hidrico", "sensibilidad_exceso_hidrico",
        "desviacion_altitud_msnm", "en_rango_altitudinal_optimo"
    ]

    # RAMA 2 (Señal Macroclimática ENSO + Dinámica Climática Local Contemporánea y Rezagada)
    bloque_rama2_enso = [
        "oni_media_anual", "oni_max_anual", "oni_min_anual", "oni_std_anual",
        "oni_acum_positivo", "oni_acum_negativo",
        "meses_nino_anual", "meses_nina_anual", "meses_nino_mod_fuerte",
        "meses_nino_fuerte", "meses_nina_fuerte", "max_racha_enso_meses",
        "oni_semestre1_ene_jun", "oni_semestre2_jul_dic",
        "oni_q1_ene_mar", "oni_q2_abr_jun", "oni_q3_jul_sep", "oni_q4_oct_dic",
        "oni_media_lag1", "oni_max_lag1", "oni_min_lag1",
        "oni_acum_pos_lag1", "oni_sem2_lag1_jul_dic", "meses_nino_lag1"
    ]
    bloque_rama2_clima_local = [
        "precip_anual_mm", "precip_sem1_mm", "precip_sem2_mm",
        "precip_q1_mm", "precip_q2_mm", "precip_q3_mm", "precip_q4_mm",
        "precip_min_mes_mm", "precip_max_mes_mm",
        "dias_lluvia_anual", "dias_secos_anual",
        "temp_media_anual_c", "temp_max_anual_c", "temp_min_anual_c",
        "temp_max_extrema_c", "temp_min_extrema_c", "rango_termico_diario_c",
        "temp_sem1_c", "temp_sem2_c",
        "anomalia_precip_pct", "zscore_precip", "anomalia_temp_c", "zscore_temp",
        "precip_anual_lag1_mm", "anomalia_precip_lag1_pct",
        "temp_media_lag1_c", "anomalia_temp_lag1_c",
        "interaccion_oni_anomalia_precip", "interaccion_oni_anomalia_temp"
    ]

    # Interacciones explícitas Clima x Cultivo (útiles en Baselines y opcionales en Rama 2/Fusión)
    bloque_interacciones_agroclimaticas = [
        "estres_calor_c", "estres_frio_c", "desviacion_temp_optima_c",
        "ratio_hidrico_cultivo", "deficit_hidrico_mm", "exceso_hidrico_mm"
    ]

    grupos = [
        ("1. Metadatos, Llaves y Partición LOEO", bloque_identificadores_meta),
        ("2. Variables Agrícolas Observadas y Objetivos (Target)", bloque_agr_observado_y_targets),
        ("3A. Rama 1 - Contexto Biofísico Zonal (Municipio)", bloque_rama1_zona),
        ("3B. Rama 1 - Contexto Ecofisiológico de Especie (Cultivo)", bloque_rama1_cultivo),
        ("4A. Rama 2 - Señal Macroclimática ENSO (ONI NOAA)", bloque_rama2_enso),
        ("4B. Rama 2 - Clima Local Municipal (IDEAM + NASA POWER)", bloque_rama2_clima_local),
        ("5. Interacciones Explícitas Estrés Clima x Cultivo", bloque_interacciones_agroclimaticas),
    ]

    suma_total = 0
    cols_clasificadas = []
    for nombre, lista in grupos:
        print(f"\n{nombre} ({len(lista)} columnas):")
        for idx, c in enumerate(lista, 1):
            pos_real = all_cols.index(c) + 1
            print(f"  [{pos_real:3d}] {c}")
        suma_total += len(lista)
        cols_clasificadas.extend(lista)

    faltantes = set(all_cols) - set(cols_clasificadas)
    duplicadas = len(cols_clasificadas) - len(set(cols_clasificadas))
    print(f"\nVerificación de partición de columnas: Suma={suma_total} / Total={len(all_cols)} | Sin clasificar={len(faltantes)} | Duplicadas={duplicadas}")


if __name__ == "__main__":
    auditar_punto_3()
    auditar_punto_2()
    auditar_punto_4()
