"""
Fase 3 - Feature Engineering, Variable Objetivo 'Eficiencia' y Dataset Maestro
==============================================================================
Implementa los Pasos 2, 3 y 5 de la Hoja de Ruta del Plan de Acción:
  1. Calcula el rendimiento potencial departamental (Y_pot_{d,c,t}) sobre
     eva_departamento_cultivo_anio.csv mediante tendencia tecnológica suavizada
     (excluyendo eventos de El Niño severo para estimar la línea base).
  2. Calcula la variable objetivo 'eficiencia' = Y_{m,c,t} / Y_pot_{d,c,t}
     y también 'eficiencia_rel_municipal' (respecto a la mediana histórica del
     propio municipio-cultivo).
  3. Cruza la base agrícola EVA con:
     - Índice ONI mensual/anual/fenológico (oni_anual_features.csv)
     - Clima local municipal e indicadores IDEAM (clima_anual_municipios_2007_2024.csv
       y municipios_coordenadas_altitud.csv)
     - Perfil ecofisiológico de cada cultivo (perfil_agronomico_cultivos.csv)
  4. Construye indicadores biofísicos de estrés térmico, déficit/exceso hídrico
     y aptitud altitudinal (interacción Zona x Cultivo x Clima).

Salidas:
  - eva_departamento_tendencia_potencial.csv
  - dataset_maestro_entrenamiento.csv (base completa lista para modelado)
  - dataset_maestro_panel_balanceado.csv (subconjunto de panel balanceado)
"""

import pandas as pd
import numpy as np


def estimar_rendimiento_potencial_departamental() -> pd.DataFrame:
    print("[1/4] Estimando rendimiento potencial departamental (Y_pot_{d,c,t})...")
    df_dep = pd.read_csv("eva_departamento_cultivo_anio.csv")
    df_dep = df_dep.sort_values(["departamento", "cultivo_cat", "año"]).reset_index(drop=True)

    # Años de El Niño Fuerte (ONI max >= 1.5) se enmascaran al ajustar la tendencia tecnológica
    anios_nino_fuerte = {2009, 2015, 2016, 2023}

    resultados = []
    for (dep, cult), g in df_dep.groupby(["departamento", "cultivo_cat"]):
        g = g.sort_values("año").copy()
        # Serie base sin años Niño severo para estimar potencial tecnológico
        y_sin_nino = g["rendimiento"].where(~g["año"].isin(anios_nino_fuerte), np.nan)
        # Interpolación lineal + media móvil centrada de 5 años
        y_interp = y_sin_nino.interpolate(method="linear", limit_direction="both")
        y_smooth = y_interp.rolling(window=5, min_periods=1, center=True).mean()

        # Tendencia lineal robusta de respaldo si hubiera pocos puntos
        if y_smooth.isna().any():
            y_smooth = y_Smooth_fallback = g["rendimiento"].rolling(window=5, min_periods=1, center=True).mean()

        g["rendimiento_potencial_dep"] = y_smooth.round(5)
        g["eficiencia_agregada_dep"] = (g["rendimiento"] / g["rendimiento_potencial_dep"]).round(4)
        resultados.append(g)

    df_pot = pd.concat(resultados, ignore_index=True)
    df_pot.to_csv("eva_departamento_tendencia_potencial.csv", index=False, encoding="utf-8")
    print(f"    -> Guardado 'eva_departamento_tendencia_potencial.csv' ({len(df_pot)} filas)")
    return df_pot


def construir_dataset_maestro(df_pot: pd.DataFrame):
    print("[2/4] Cargando EVA municipal, ONI, Clima Local e Indicadores Agronómicos...")
    df_eva = pd.read_csv(
        "eva_2007_2024_consolidado.csv",
        dtype={"cod_dane_mun": str, "cod_dane_dep": str},
    )
    df_oni = pd.read_csv("oni_anual_features.csv")
    df_clima = pd.read_csv("clima_anual_municipios_2007_2024.csv", dtype={"cod_dane_mun": str})
    df_muns = pd.read_csv("municipios_coordenadas_altitud.csv", dtype={"cod_dane_mun": str})
    df_agro = pd.read_csv("perfil_agronomico_cultivos.csv")

    # Filtrar registros con área cosechada > 0 y rendimiento válido (> 0)
    n_inicial = len(df_eva)
    df_valid = df_eva[
        (df_eva["rendimiento"].notna())
        & (df_eva["rendimiento"] > 0)
        & (df_eva["area_cosechada"].fillna(0) > 0)
    ].copy()
    print(f"    -> Registros EVA totales: {n_inicial} | Con cosecha y rendimiento válido (>0): {len(df_valid)}")

    # 1. Cruzar con rendimiento potencial departamental
    df_master = df_valid.merge(
        df_pot[["departamento", "cultivo_cat", "año", "rendimiento_potencial_dep"]],
        on=["departamento", "cultivo_cat", "año"],
        how="left",
    )

    # Calcular Eficiencia Departamental (definición del Plan de Acción 3.1)
    df_master["eficiencia"] = (df_master["rendimiento"] / df_master["rendimiento_potencial_dep"]).round(4)

    # Calcular también potencial local municipal (mediana histórica sin Niño fuerte de cada municipio-cultivo)
    anios_nino_fuerte = {2009, 2015, 2016, 2023}
    base_mun = (
        df_master[~df_master["año"].isin(anios_nino_fuerte)]
        .groupby(["cod_dane_mun", "cultivo_cat"], as_index=False)["rendimiento"]
        .median()
        .rename(columns={"rendimiento": "rendimiento_base_municipal"})
    )
    df_master = df_master.merge(base_mun, on=["cod_dane_mun", "cultivo_cat"], how="left")
    df_master["rendimiento_base_municipal"] = df_master["rendimiento_base_municipal"].fillna(
        df_master["rendimiento_potencial_dep"]
    ).round(4)
    df_master["eficiencia_rel_municipal"] = (
        df_master["rendimiento"] / df_master["rendimiento_base_municipal"]
    ).round(4)

    # Winsorización suave (percentil 1% - 99%) en variable 'eficiencia_clip' para evitar outliers de digitación
    p01, p99 = df_master["eficiencia"].quantile(0.005), df_master["eficiencia"].quantile(0.995)
    df_master["eficiencia_clip"] = df_master["eficiencia"].clip(lower=round(p01, 4), upper=round(p99, 4)).round(4)

    # 2. Cruzar con coordenadas, altitud y referencias IDEAM
    cols_mun = [
        "cod_dane_mun", "latitud_dane", "longitud_dane", "altitud_msnm",
        "ideam_estacion_precip_cod", "ideam_dist_estacion_precip_km",
        "ideam_estacion_temp_cod", "ideam_dist_estacion_temp_km",
        "ideam_normal_precip_anual_mm", "ideam_normal_temp_media_c"
    ]
    df_master = df_master.merge(df_muns[cols_mun], on="cod_dane_mun", how="left")

    # Indicadores explícitos de calidad de cobertura climática por municipio (para auditoría de error en Fase 6)
    df_master["tiene_estacion_precip_directa"] = (df_master["ideam_dist_estacion_precip_km"] <= 8.0).astype(int)
    df_master["tiene_estacion_temp_directa"] = (df_master["ideam_dist_estacion_temp_km"] <= 10.0).astype(int)
    df_master["calidad_dato_climatico_alta"] = (
        (df_master["tiene_estacion_precip_directa"] == 1)
        & (df_master["tiene_estacion_temp_directa"] == 1)
    ).astype(int)

    # 3. Cruzar con Clima Anual Municipal (2007-2024)
    cols_clima = [c for c in df_clima.columns if c not in ("departamento", "municipio")]
    df_master = df_master.merge(df_clima[cols_clima], on=["cod_dane_mun", "año"], how="left")

    # Verificación estricta: 0 imputación en variables climáticas (incluyendo rezagos 2006->2007)
    nans_clima = int(df_master[cols_clima].isna().sum().sum())
    assert nans_clima == 0, f"Error: se detectaron {nans_clima} NaNs en el bloque climático."

    # 4. Cruzar con Señal Macroclimática ONI (NOAA CPC)
    df_master = df_master.merge(df_oni, on="año", how="left")

    # 5. Cruzar con Perfil Agronómico de Cultivos
    cols_agro = [c for c in df_agro.columns if c != "fuente_referencia"]
    df_master = df_master.merge(df_agro[cols_agro], on="cultivo_cat", how="left")

    print("[3/4] Calculando interacciones biofísicas (Estrés Térmico, Hídrico y Aptitud Altitudinal)...")
    # Estrés térmico respecto al rango óptimo de la especie
    df_master["estres_calor_c"] = np.maximum(0.0, df_master["temp_media_anual_c"] - df_master["temp_opt_max_c"]).round(3)
    df_master["estres_frio_c"] = np.maximum(0.0, df_master["temp_opt_min_c"] - df_master["temp_media_anual_c"]).round(3)
    df_master["desviacion_temp_optima_c"] = (df_master["estres_calor_c"] + df_master["estres_frio_c"]).round(3)

    # Balance hídrico respecto al requerimiento del cultivo
    df_master["ratio_hidrico_cultivo"] = (df_master["precip_anual_mm"] / df_master["precip_opt_media_mm"]).round(4)
    df_master["deficit_hidrico_mm"] = np.maximum(0.0, df_master["precip_opt_min_mm"] - df_master["precip_anual_mm"]).round(2)
    df_master["exceso_hidrico_mm"] = np.maximum(0.0, df_master["precip_anual_mm"] - df_master["precip_opt_max_mm"]).round(2)

    # Desviación altitudinal respecto a la franja óptima del cultivo
    df_master["desviacion_altitud_msnm"] = (
        np.maximum(0.0, df_master["altitud_msnm"] - df_master["altitud_opt_max_msnm"])
        + np.maximum(0.0, df_master["altitud_opt_min_msnm"] - df_master["altitud_msnm"])
    ).round(1)
    df_master["en_rango_altitudinal_optimo"] = (df_master["desviacion_altitud_msnm"] == 0).astype(int)

    # Interacción explícita ONI x Déficit/Anomalía local
    df_master["interaccion_oni_anomalia_precip"] = (
        df_master["oni_media_anual"] * df_master["anomalia_precip_pct"]
    ).round(3)
    df_master["interaccion_oni_anomalia_temp"] = (
        df_master["oni_media_anual"] * df_master["anomalia_temp_c"]
    ).round(4)

    # Ordenar cronológica y geográficamente
    df_master = df_master.sort_values(["departamento", "cod_dane_mun", "cultivo_cat", "año"]).reset_index(drop=True)

    df_master.to_csv("dataset_maestro_entrenamiento.csv", index=False, encoding="utf-8")
    df_panel = df_master[df_master["en_panel_balanceado"] == True].reset_index(drop=True)
    df_panel.to_csv("dataset_maestro_panel_balanceado.csv", index=False, encoding="utf-8")

    print("\n[4/4] === RESUMEN DEL DATASET MAESTRO (FASE 3 COMPLETADA) ===")
    print(f"  - dataset_maestro_entrenamiento.csv    : {len(df_master)} filas x {df_master.shape[1]} columnas")
    print(f"  - dataset_maestro_panel_balanceado.csv : {len(df_panel)} filas x {df_panel.shape[1]} columnas")
    print(f"  - Municipios representados             : {df_master['cod_dane_mun'].nunique()}")
    print(f"  - Cultivos representados               : {df_master['cultivo_cat'].unique().tolist()}")
    print(f"  - Rango de años                        : {df_master['año'].min()} - {df_master['año'].max()}")
    print(f"  - Total NaNs en variables predictoras  : {df_master.drop(columns=['subcategoria']).isna().sum().sum()}")
    print("\nEstadísticas de la variable objetivo (eficiencia y eficiencia_rel_municipal) por cultivo:")
    resumen_obj = df_master.groupby("cultivo_cat").agg(
        registros=("eficiencia", "count"),
        rend_medio_t_ha=("rendimiento", "mean"),
        efic_dep_media=("eficiencia", "mean"),
        efic_dep_mediana=("eficiencia", "median"),
        efic_mun_media=("eficiencia_rel_municipal", "mean"),
        efic_mun_std=("eficiencia_rel_municipal", "std"),
    ).round(3)
    print(resumen_obj.to_string())


def main():
    df_pot = estimar_rendimiento_potencial_departamental()
    construir_dataset_maestro(df_pot)


if __name__ == "__main__":
    main()
