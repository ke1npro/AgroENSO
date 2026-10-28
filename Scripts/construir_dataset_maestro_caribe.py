"""
Fase 3 - Feature Engineering, Variable Objetivo 'Eficiencia' y Dataset Maestro - Caribe
=======================================================================================
Integra los datos agrícolas EVA de los 7 departamentos de la Región Caribe con:
  - Rendimiento potencial departamental (eva_departamento_tendencia_potencial_caribe.csv)
  - Clima local municipal e indicadores IDEAM (clima_anual_municipios_2007_2024_caribe.csv)
  - Coordenadas, altitud y calidad de estación (municipios_coordenadas_altitud_caribe.csv)
  - Índice ONI macroclimático (oni_anual_features.csv)
  - Perfil ecofisiológico de cultivos (perfil_agronomico_cultivos.csv)
  - Indicadores biofísicos de estrés térmico, hídrico y aptitud altitudinal.

Salidas:
  - dataset_maestro_entrenamiento_caribe.csv
  - dataset_maestro_panel_balanceado_caribe.csv
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = r"C:\Proyecto Semillero"


def construir_dataset_maestro_caribe():
    print("=" * 80)
    print("CONSTRUYENDO DATASET MAESTRO DE ENTRENAMIENTO - REGIÓN CARIBE")
    print("=" * 80)

    # 1. Cargar fuentes
    print("\n[1/4] Cargando archivos intermedios de Caribe, ONI y Perfil Agronómico...")
    df_eva = pd.read_csv(
        os.path.join(BASE_DIR, "eva_2007_2024_caribe.csv"),
        dtype={"cod_dane_mun": str, "cod_dane_dep": str},
    )
    df_oni = pd.read_csv(os.path.join(BASE_DIR, "oni_anual_features.csv"))
    df_clima = pd.read_csv(
        os.path.join(BASE_DIR, "clima_anual_municipios_2007_2024_caribe.csv"),
        dtype={"cod_dane_mun": str},
    )
    df_muns = pd.read_csv(
        os.path.join(BASE_DIR, "municipios_coordenadas_altitud_caribe.csv"),
        dtype={"cod_dane_mun": str},
    )
    df_agro = pd.read_csv(os.path.join(BASE_DIR, "perfil_agronomico_cultivos.csv"))

    # 2. Cruzar con coordenadas, altitud y referencias IDEAM
    print("[2/4] Cruzando variables espaciales, de calidad climática y series climáticas...")
    cols_mun = [
        "cod_dane_mun", "latitud_dane", "longitud_dane", "altitud_msnm",
        "ideam_estacion_precip_cod", "ideam_dist_estacion_precip_km",
        "ideam_estacion_temp_cod", "ideam_dist_estacion_temp_km",
        "ideam_normal_precip_anual_mm", "ideam_normal_temp_media_c",
        "tiene_estacion_precip_directa", "tiene_estacion_temp_directa",
        "calidad_dato_climatico_alta",
    ]
    df_master = df_eva.merge(df_muns[cols_mun].drop_duplicates("cod_dane_mun"), on="cod_dane_mun", how="left")

    # 3. Cruzar con Clima Anual Municipal (2007-2024)
    cols_clima = [c for c in df_clima.columns if c not in ("departamento", "municipio")]
    df_master = df_master.merge(df_clima[cols_clima], on=["cod_dane_mun", "año"], how="left")

    # Verificación de completitud climática
    nans_clima = int(df_master[cols_clima].isna().sum().sum())
    assert nans_clima == 0, f"Error: se detectaron {nans_clima} NaNs en el bloque climático de Caribe."

    # 4. Cruzar con Señal Macroclimática ONI (NOAA CPC)
    df_master = df_master.merge(df_oni, on="año", how="left")

    # 5. Cruzar con Perfil Agronómico de Cultivos
    cols_agro = [c for c in df_agro.columns if c != "fuente_referencia"]
    df_master = df_master.merge(df_agro[cols_agro], on="cultivo_cat", how="left")

    # 6. Interacciones biofísicas
    print("[3/4] Calculando interacciones biofísicas (Estrés Térmico, Hídrico y Aptitud Altitudinal)...")
    # Estrés térmico
    df_master["estres_calor_c"] = np.maximum(0.0, df_master["temp_media_anual_c"] - df_master["temp_opt_max_c"]).round(3)
    df_master["estres_frio_c"] = np.maximum(0.0, df_master["temp_opt_min_c"] - df_master["temp_media_anual_c"]).round(3)
    df_master["desviacion_temp_optima_c"] = (df_master["estres_calor_c"] + df_master["estres_frio_c"]).round(3)

    # Balance hídrico
    df_master["ratio_hidrico_cultivo"] = (df_master["precip_anual_mm"] / df_master["precip_opt_media_mm"]).round(4)
    df_master["deficit_hidrico_mm"] = np.maximum(0.0, df_master["precip_opt_min_mm"] - df_master["precip_anual_mm"]).round(2)
    df_master["exceso_hidrico_mm"] = np.maximum(0.0, df_master["precip_anual_mm"] - df_master["precip_opt_max_mm"]).round(2)

    # Desviación altitudinal
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

    out_master = os.path.join(BASE_DIR, "dataset_maestro_entrenamiento_caribe.csv")
    out_panel = os.path.join(BASE_DIR, "dataset_maestro_panel_balanceado_caribe.csv")

    df_master.to_csv(out_master, index=False, encoding="utf-8")
    df_panel = df_master[df_master["en_panel_balanceado"] == True].reset_index(drop=True)
    df_panel.to_csv(out_panel, index=False, encoding="utf-8")

    print("\n[4/4] === RESUMEN DEL DATASET MAESTRO CARIBE (FASE 3 COMPLETADA) ===")
    print(f"  - dataset_maestro_entrenamiento_caribe.csv    : {len(df_master)} filas x {df_master.shape[1]} columnas")
    print(f"  - dataset_maestro_panel_balanceado_caribe.csv : {len(df_panel)} filas x {df_panel.shape[1]} columnas")
    print(f"  - Municipios representados                     : {df_master['cod_dane_mun'].nunique()}")
    print(f"  - Cultivos representados                       : {df_master['cultivo_cat'].unique().tolist()}")
    print(f"  - Rango de años                                : {df_master['año'].min()} - {df_master['año'].max()}")
    
    # Chequeo estricto de NaNs en predictores
    cols_no_predictor = [
        "subcategoria", "rendimiento", "produccion", "area_cosechada", "area_sembrada",
        "eficiencia", "eficiencia_clip", "eficiencia_rel_municipal", "rendimiento_base_municipal"
    ]
    pred_cols = [c for c in df_master.columns if c not in cols_no_predictor and df_master[c].dtype in (np.float64, np.int64, float, int)]
    nans_pred = int(df_master[pred_cols].isna().sum().sum())
    print(f"  - Total columnas predictoras numéricas         : {len(pred_cols)}")
    print(f"  - Total NaNs en predictores numéricos          : {nans_pred}")
    assert nans_pred == 0, f"Error: Se encontraron {nans_pred} NaNs en predictores numéricos."

    print("\nEstadísticas de la variable objetivo (eficiencia) por cultivo en Caribe:")
    resumen_obj = df_master.groupby("cultivo_cat").agg(
        registros=("eficiencia", "count"),
        rend_medio_t_ha=("rendimiento", "mean"),
        efic_media=("eficiencia", "mean"),
        efic_mediana=("eficiencia", "median"),
        efic_clip_media=("eficiencia_clip", "mean"),
        efic_std=("eficiencia", "std"),
    ).round(3)
    print(resumen_obj.to_string())


if __name__ == "__main__":
    construir_dataset_maestro_caribe()
