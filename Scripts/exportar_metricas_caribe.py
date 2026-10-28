"""
Exportador de Datos y Métricas de la Región Caribe para el Dashboard Técnico
============================================================================
Exporta los datos agrícolas, climáticos, baselines (Fase 4), Red Modular (Fase 5)
y XGBoost Macro-Regional Combinado para la Región Caribe:
  - dataset_maestro_entrenamiento_caribe.csv
  - oni_anual_features.csv
  - baselines_resultados_caribe.csv
  - red_modular_resultados_caribe.csv
  - baselines_xgboost_andina_caribe.csv
  - baselines_predicciones_caribe.csv
  - baselines_feature_importances_caribe.csv
  - baselines_curvas_aprendizaje_caribe.csv

Genera:
  - dashboard_tecnico/data/dashboard_data_caribe.js (como window.DASHBOARD_DATA_CARIBE)
"""

import os
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Soporte para ejecución desde raíz o desde Scripts/
try:
    from paths import resolve_path, ROOT_DIR
except ImportError:
    from Scripts.paths import resolve_path, ROOT_DIR


def main():
    print("[1/4] Cargando datasets y resultados de Caribe...")
    df_master = pd.read_csv(resolve_path("dataset_maestro_entrenamiento_caribe.csv"))
    df_oni = pd.read_csv(resolve_path("oni_anual_features.csv"))
    df_res_base = pd.read_csv(resolve_path("baselines_resultados_caribe.csv"))
    df_res_nn = pd.read_csv(resolve_path("red_modular_resultados_caribe.csv"))
    df_res_comb = pd.read_csv(resolve_path("baselines_xgboost_andina_caribe.csv"))
    df_preds = pd.read_csv(resolve_path("baselines_predicciones_caribe.csv"))
    df_preds_nn = pd.read_csv(resolve_path("red_modular_predicciones_caribe.csv"))
    df_imp = pd.read_csv(resolve_path("baselines_feature_importances_caribe.csv"))
    df_curv = pd.read_csv(resolve_path("baselines_curvas_aprendizaje_caribe.csv"))

    cols_res = [
        "split", "split_descripcion", "configuracion_enso", "modelo", "tiempo_seg",
        "RMSE_test", "MAE_test", "R2_test", "Pearson_r_test", "Bias_test",
        "RMSE_test_raw", "MAE_test_raw", "RMSE_train", "MAE_train", "R2_train"
    ]
    df_comb_car = df_res_comb[df_res_comb["ambito_evaluacion"] == "Subconjunto_Caribe"].copy()
    df_res = pd.concat([df_res_base[cols_res], df_res_nn[cols_res], df_comb_car[cols_res]], ignore_index=True)

    print("[2/4] Procesando datos del Explorador de Datos y Clima Caribe...")
    stats_cultivo = []
    for cult, g in df_master.groupby("cultivo_cat"):
        stats_cultivo.append({
            "cultivo": cult,
            "registros": len(g),
            "rend_medio": round(float(g["rendimiento"].mean()), 3),
            "rend_mediana": round(float(g["rendimiento"].median()), 3),
            "efic_media": round(float(g["eficiencia"].mean()), 3),
            "efic_mediana": round(float(g["eficiencia"].median()), 3),
            "efic_clip_media": round(float(g["eficiencia_clip"].mean()), 3),
            "efic_std": round(float(g["eficiencia"].std()), 3),
            "area_media_ha": round(float(g["area_cosechada"].mean()), 1),
            "prod_media_t": round(float(g["produccion"].mean()), 1),
        })

    serie_oni = []
    for _, row in df_oni.sort_values("año").iterrows():
        serie_oni.append({
            "año": int(row["año"]),
            "oni_media": round(float(row["oni_media_anual"]), 3),
            "oni_max": round(float(row["oni_max_anual"]), 3),
            "oni_min": round(float(row["oni_min_anual"]), 3),
            "meses_nino": int(row["meses_nino_anual"]),
            "meses_nina": int(row["meses_nina_anual"]),
            "categoria": str(row["categoria_evento_anual"]),
            "fase": str(row["fase_enso_anual"]),
        })

    clima_deptos = []
    for (dep, anio), g in df_master.groupby(["departamento", "año"]):
        clima_deptos.append({
            "departamento": dep,
            "año": int(anio),
            "precip_anual_mm": round(float(g["precip_anual_mm"].mean()), 1),
            "temp_media_c": round(float(g["temp_media_anual_c"].mean()), 2),
            "anomalia_precip_pct": round(float(g["anomalia_precip_pct"].mean()), 2),
            "anomalia_temp_c": round(float(g["anomalia_temp_c"].mean()), 3),
            "oni_media": round(float(g["oni_media_anual"].iloc[0]), 3),
        })

    print("[3/4] Procesando resultados de Baselines y Red Modular Caribe...")
    tabla_baselines = df_res.to_dict(orient="records")

    df_loeo = df_res[df_res["split"].isin(["Split_1_LOEO_2015_2016", "Split_2_LOEO_2023_2024"])]
    prom_loeo = (
        df_loeo.groupby(["modelo", "configuracion_enso"])[["RMSE_test", "MAE_test", "R2_test", "Pearson_r_test"]]
        .mean()
        .round(4)
        .reset_index()
        .to_dict(orient="records")
    )

    sub_sp1_base = df_preds[df_preds["split"] == "Split_1_LOEO_2015_2016"][["modelo", "cultivo_cat", "y_real_clip", "residual"]]
    sub_sp1_nn = df_preds_nn[df_preds_nn["split"] == "Split_1_LOEO_2015_2016"][["modelo", "cultivo_cat", "y_real_clip", "residual"]]
    sub_sp1 = pd.concat([sub_sp1_base, sub_sp1_nn], ignore_index=True)

    error_por_cultivo = []
    for (mod, cult), g in sub_sp1.groupby(["modelo", "cultivo_cat"]):
        err = g["residual"].values
        denom = np.sum((g["y_real_clip"] - g["y_real_clip"].mean()) ** 2)
        r2_val = round(float(1.0 - (np.sum(err ** 2) / denom)), 4) if denom > 0 else np.nan
        error_por_cultivo.append({
            "modelo": mod,
            "cultivo": cult,
            "N": len(g),
            "RMSE": round(float(np.sqrt(np.mean(err ** 2))), 4),
            "MAE": round(float(np.mean(np.abs(err))), 4),
            "R2": r2_val,
        })

    importances_dict = {}
    for mod in ["XGBoost", "RandomForest"]:
        sub_imp = df_imp[df_imp["modelo"] == mod].sort_values("importance", ascending=False).head(25)
        importances_dict[mod] = sub_imp.to_dict(orient="records")

    curvas_aprendizaje = df_curv.to_dict(orient="records")

    sub_xgb_sp1 = df_preds[(df_preds["split"] == "Split_1_LOEO_2015_2016") & (df_preds["modelo"] == "XGBoost")]
    sample_preds = (
        sub_xgb_sp1.sample(n=min(600, len(sub_xgb_sp1)), random_state=42)
        [["departamento", "municipio", "cultivo_cat", "año", "y_real_clip", "y_pred", "residual", "error_abs", "calidad_dato_climatico_alta", "ideam_dist_estacion_precip_km"]]
        .to_dict(orient="records")
    )

    calidad_estaciones_stats = []
    for cal_val, g in sub_xgb_sp1.groupby("calidad_dato_climatico_alta"):
        calidad_estaciones_stats.append({
            "calidad_alta": int(cal_val),
            "etiqueta": "Alta Calidad (Estación IDEAM <= 8-10 km)" if cal_val == 1 else "Estación Regional / Lejana (> 8-10 km)",
            "N": len(g),
            "MAE": round(float(g["error_abs"].mean()), 4),
            "RMSE": round(float(np.sqrt(np.mean(g["residual"] ** 2))), 4),
            "mediana_dist_km": round(float(g["ideam_dist_estacion_precip_km"].median()), 2),
        })

    data_payload = {
        "metadata": {
            "nombre": "AgroENSO · Semillero de Investigación en IA y Agroclimatología",
            "alcance": "Región Caribe (Atlántico, Bolívar, Cesar, Córdoba, La Guajira, Magdalena, Sucre)",
            "periodo": "2007 - 2024",
            "registros_totales": len(df_master),
            "municipios": int(df_master["cod_dane_mun"].nunique()),
            "cultivos": ["Maíz tradicional", "Frijol", "Maíz tecnificado", "Cacao", "Café"],
            "fecha_generacion": "Octubre 2026",
        },
        "stats_cultivo": stats_cultivo,
        "serie_oni": serie_oni,
        "clima_deptos": clima_deptos,
        "tabla_baselines": tabla_baselines,
        "promedio_loeo": prom_loeo,
        "error_por_cultivo": error_por_cultivo,
        "feature_importances": importances_dict,
        "curvas_aprendizaje": curvas_aprendizaje,
        "sample_predicciones": sample_preds,
        "calidad_estaciones": calidad_estaciones_stats,
    }

    out_dir = ROOT_DIR / "dashboard_tecnico" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "dashboard_data_caribe.js"
    json_str = json.dumps(data_payload, ensure_ascii=False, indent=2)

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("/* Datos consolidados para el Dashboard Técnico - Región Caribe */\n")
        f.write("window.DASHBOARD_DATA_CARIBE = " + json_str + ";\n")

    print(f"[4/4] Archivo exportado exitosamente: {out_file} ({os.path.getsize(out_file) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
