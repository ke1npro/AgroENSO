"""
Fase 4 - Evaluación Rigurosa de Baselines Clásicos - Región Caribe
==================================================================
Entrena y evalúa los 4 modelos de referencia de la literatura agroclimática para
los 7 departamentos del Caribe colombiano (11,197 registros, 192 municipios):
  1. Ridge Regression (Baseline lineal L2)
  2. Huber Regressor (Baseline lineal robusto a outliers)
  3. Random Forest Regressor (Ensamble no lineal)
  4. XGBoost Regressor (Gradient Boosting con pérdida pseudo-Huber)

Esquema experimental (100% equivalente a la Región Andina):
  - Variable objetivo principal: 'eficiencia_clip' (winsorizada [0.311, 2.521])
  - Análisis de sensibilidad: 'eficiencia' sin recortar.
  - Particiones de prueba:
      * Split 1 (LOEO 2015-2016): Test en El Niño Severo 2015-2016.
      * Split 2 (LOEO 2023-2024): Test en El Niño Reciente 2023-2024.
      * Split 3 (Secuencial 2021-2024): Test en cuatrienio reciente 2021-2024.
      * Promedio conjunto LOEO entre eventos de El Niño.
  - Ablación científica ENSO:
      * Sin_ENSO: Solo Zona + Cultivo + Clima Local.
      * Con_ENSO: Zona + Cultivo + Clima Local + Señal ONI NOAA.

Salidas:
  - baselines_resultados_caribe.csv
  - baselines_predicciones_caribe.csv
  - baselines_feature_importances_caribe.csv
  - baselines_curvas_aprendizaje_caribe.csv
"""

import os
import time
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

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

BASE_DIR = r"C:\Proyecto Semillero"


def calcular_metricas(y_true, y_pred, y_true_raw=None):
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - (float(np.sum(err ** 2)) / ss_tot) if ss_tot > 0 else np.nan
    r_corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 2 else np.nan

    res = {
        "N": len(y_true),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "Bias": round(bias, 4),
        "R2": round(r2, 4),
        "Pearson_r": round(r_corr, 4),
    }

    if y_true_raw is not None:
        err_raw = y_pred - y_true_raw
        res["RMSE_raw"] = round(float(np.sqrt(np.mean(err_raw ** 2))), 4)
        res["MAE_raw"] = round(float(np.mean(np.abs(err_raw))), 4)

    return res


def obtener_bloques_features():
    cols_zona = [
        "latitud_dane", "longitud_dane", "altitud_msnm",
        "ideam_normal_precip_anual_mm", "ideam_normal_temp_media_c",
        "clima_base_precip_mm", "clima_std_precip_mm",
        "clima_base_temp_c", "clima_std_temp_c"
    ]
    cols_cultivo = [
        "es_permanente", "via_fotosintetica_c4", "nivel_tecnificacion",
        "duracion_ciclo_meses", "temp_opt_min_c", "temp_opt_max_c", "temp_opt_media_c",
        "precip_opt_min_mm", "precip_opt_max_mm", "precip_opt_media_mm",
        "altitud_opt_min_msnm", "altitud_opt_max_msnm", "altitud_opt_media_msnm",
        "sensibilidad_deficit_hidrico", "sensibilidad_exceso_hidrico",
        "desviacion_altitud_msnm", "en_rango_altitudinal_optimo"
    ]
    cols_clima_local = [
        "precip_anual_mm", "precip_sem1_mm", "precip_sem2_mm",
        "precip_q1_mm", "precip_q2_mm", "precip_q3_mm", "precip_q4_mm",
        "precip_min_mes_mm", "precip_max_mes_mm",
        "dias_lluvia_anual", "dias_secos_anual",
        "temp_media_anual_c", "temp_max_anual_c", "temp_min_anual_c",
        "temp_max_extrema_c", "temp_min_extrema_c", "rango_termico_diario_c",
        "temp_sem1_c", "temp_sem2_c",
        "anomalia_precip_pct", "zscore_precip", "anomalia_temp_c", "zscore_temp",
        "precip_anual_lag1_mm", "anomalia_precip_lag1_pct",
        "temp_media_lag1_c", "anomalia_temp_lag1_c"
    ]
    cols_estres = [
        "estres_calor_c", "estres_frio_c", "desviacion_temp_optima_c",
        "ratio_hidrico_cultivo", "deficit_hidrico_mm", "exceso_hidrico_mm"
    ]
    cols_enso = [
        "oni_media_anual", "oni_max_anual", "oni_min_anual", "oni_std_anual",
        "oni_acum_positivo", "oni_acum_negativo",
        "meses_nino_anual", "meses_nina_anual", "meses_nino_mod_fuerte",
        "meses_nino_fuerte", "meses_nina_fuerte", "max_racha_enso_meses",
        "oni_semestre1_ene_jun", "oni_semestre2_jul_dic",
        "oni_q1_ene_mar", "oni_q2_abr_jun", "oni_q3_jul_sep", "oni_q4_oct_dic",
        "oni_media_lag1", "oni_max_lag1", "oni_min_lag1",
        "oni_acum_pos_lag1", "oni_sem2_lag1_jul_dic", "meses_nino_lag1",
        "interaccion_oni_anomalia_precip", "interaccion_oni_anomalia_temp"
    ]

    sin_enso = cols_zona + cols_cultivo + cols_clima_local + cols_estres
    con_enso = sin_enso + cols_enso
    return sin_enso, con_enso


def main():
    print("=" * 90)
    print("FASE 4: EVALUACIÓN DE BASELINES CLÁSICOS - REGIÓN CARIBE (LOEO + ABLACIÓN ENSO)")
    print("=" * 90)

    df = pd.read_csv(resolve_path("dataset_maestro_entrenamiento_caribe.csv"))
    print(f"Cargado dataset maestro Caribe: {len(df)} registros x {df.shape[1]} columnas")
    print(f"Rango de años: {df['año'].min()} - {df['año'].max()} | Cultivos: {df['cultivo_cat'].nunique()}")

    feats_sin_enso, feats_con_enso = obtener_bloques_features()
    print(f"Predictores continuos: Sin ENSO = {len(feats_sin_enso)} | Con ENSO = {len(feats_con_enso)}")

    df_dummies = pd.get_dummies(df[["cultivo_cat", "departamento"]], drop_first=True, dtype=float)
    dummy_cols = list(df_dummies.columns)

    X_all_con = pd.concat([df[feats_con_enso], df_dummies], axis=1)
    X_all_sin = pd.concat([df[feats_sin_enso], df_dummies], axis=1)

    y_target = df["eficiencia_clip"].values
    y_raw = df["eficiencia"].values

    splits = {
        "Split_1_LOEO_2015_2016": {
            "test_mask": df["año"].isin([2015, 2016]),
            "descripcion": "El Niño Severo 2015-2016 (LOEO Canónico)"
        },
        "Split_2_LOEO_2023_2024": {
            "test_mask": df["año"].isin([2023, 2024]),
            "train_mask": df["año"] <= 2022,
            "descripcion": "El Niño Reciente 2023-2024 (LOEO Reciente)"
        },
        "Split_3_Secuencial_2021_2024": {
            "test_mask": df["año"] >= 2021,
            "train_mask": df["año"] <= 2020,
            "descripcion": "Cuatrienio Reciente 2021-2024 (Secuencial Puro)"
        },
    }

    modelos_dict = {
        "Ridge": lambda: Ridge(alpha=25.0),
        "Huber": lambda: HuberRegressor(epsilon=1.35, max_iter=1000, tol=1e-3),
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=160, max_depth=14, min_samples_leaf=3, random_state=42, n_jobs=-1
        ),
        "XGBoost": lambda: XGBRegressor(
            n_estimators=220, max_depth=6, learning_rate=0.04,
            subsample=0.85, colsample_bytree=0.85,
            objective="reg:pseudohubererror", random_state=42, n_jobs=-1
        ),
    }

    resultados = []
    predicciones_test_rows = []
    importances_rows = []
    curvas_rows = []

    for split_nombre, s_info in splits.items():
        print(f"\n>>> Ejecutando {split_nombre} ({s_info['descripcion']})")
        test_mask = s_info["test_mask"].values
        train_mask = s_info.get("train_mask", ~s_info["test_mask"]).values

        print(f"    Train: {train_mask.sum()} filas | Test: {test_mask.sum()} filas")

        for enso_cfg, X_mat, feats_list in [("Sin_ENSO", X_all_sin, feats_sin_enso + dummy_cols),
                                            ("Con_ENSO", X_all_con, feats_con_enso + dummy_cols)]:
            X_tr, X_te = X_mat.iloc[train_mask].values, X_mat.iloc[test_mask].values
            y_tr, y_te = y_target[train_mask], y_target[test_mask]
            y_te_raw = y_raw[test_mask]

            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_te_sc = scaler.transform(X_te)

            for m_nombre, m_fn in modelos_dict.items():
                t0 = time.time()
                m = m_fn()
                if m_nombre in ("Ridge", "Huber"):
                    m.fit(X_tr_sc, y_tr)
                    y_pred = m.predict(X_te_sc)
                    y_pred_tr = m.predict(X_tr_sc)
                else:
                    m.fit(X_tr, y_tr)
                    y_pred = m.predict(X_te)
                    y_pred_tr = m.predict(X_tr)
                dt = time.time() - t0

                # TPS e inferencia
                t_inf_start = time.perf_counter()
                if m_nombre in ("Ridge", "Huber"):
                    _ = m.predict(X_te_sc)
                else:
                    _ = m.predict(X_te)
                t_inf = time.perf_counter() - t_inf_start
                tps = round(len(y_te) / max(t_inf, 1e-6), 1)

                m_test = calcular_metricas(y_te, y_pred, y_te_raw)
                m_train = calcular_metricas(y_tr, y_pred_tr)

                reg = {
                    "split": split_nombre,
                    "split_descripcion": s_info["descripcion"],
                    "configuracion_enso": enso_cfg,
                    "modelo": m_nombre,
                    "tiempo_seg": round(dt, 2),
                    "tps_inferencia": tps,
                    "RMSE_test": m_test["RMSE"],
                    "MAE_test": m_test["MAE"],
                    "R2_test": m_test["R2"],
                    "Pearson_r_test": m_test["Pearson_r"],
                    "Bias_test": m_test["Bias"],
                    "RMSE_test_raw": m_test.get("RMSE_raw", np.nan),
                    "MAE_test_raw": m_test.get("MAE_raw", np.nan),
                    "RMSE_train": m_train["RMSE"],
                    "MAE_train": m_train["MAE"],
                    "R2_train": m_train["R2"],
                }
                resultados.append(reg)

                if enso_cfg == "Con_ENSO":
                    df_te = df.iloc[test_mask].copy()
                    for idx_row, (_, row_te) in enumerate(df_te.iterrows()):
                        predicciones_test_rows.append({
                            "split": split_nombre,
                            "modelo": m_nombre,
                            "cod_dane_mun": row_te["cod_dane_mun"],
                            "departamento": row_te["departamento"],
                            "municipio": row_te["municipio"],
                            "cultivo_cat": row_te["cultivo_cat"],
                            "año": int(row_te["año"]),
                            "y_real_clip": round(float(y_te[idx_row]), 4),
                            "y_real_raw": round(float(y_te_raw[idx_row]), 4),
                            "y_pred": round(float(y_pred[idx_row]), 4),
                            "residual": round(float(y_pred[idx_row] - y_te[idx_row]), 4),
                            "error_abs": round(float(abs(y_pred[idx_row] - y_te[idx_row])), 4),
                            "calidad_dato_climatico_alta": int(row_te["calidad_dato_climatico_alta"]),
                            "ideam_dist_estacion_precip_km": round(float(row_te["ideam_dist_estacion_precip_km"]), 2),
                            "oni_media_anual": round(float(row_te["oni_media_anual"]), 3),
                        })

                if enso_cfg == "Con_ENSO" and split_nombre == "Split_1_LOEO_2015_2016" and hasattr(m, "feature_importances_"):
                    for feat, imp in zip(feats_list, m.feature_importances_):
                        importances_rows.append({
                            "modelo": m_nombre,
                            "feature": feat,
                            "importance": round(float(imp), 5)
                        })

                print(f"      [{m_nombre:<12} | {enso_cfg:<8}] RMSE={m_test['RMSE']:.4f} | MAE={m_test['MAE']:.4f} | R2={m_test['R2']:.4f} | r={m_test['Pearson_r']:.4f} ({tps:,.0f} tps)")

        if split_nombre == "Split_1_LOEO_2015_2016":
            print("\n    -> Generando datos de curvas de convergencia (n_estimators vs Loss)...")
            X_tr_xgb = X_all_con.iloc[train_mask].values
            y_tr_xgb = y_target[train_mask]
            X_te_xgb = X_all_con.iloc[test_mask].values
            y_te_xgb = y_target[test_mask]
            xgb_mon = XGBRegressor(
                n_estimators=220, max_depth=6, learning_rate=0.04,
                subsample=0.85, colsample_bytree=0.85,
                objective="reg:pseudohubererror", eval_metric="rmse",
                random_state=42, n_jobs=-1
            )
            xgb_mon.fit(X_tr_xgb, y_tr_xgb, eval_set=[(X_tr_xgb, y_tr_xgb), (X_te_xgb, y_te_xgb)], verbose=False)
            evals = xgb_mon.evals_result()
            metric_key = list(evals["validation_0"].keys())[0]
            for epoch_i in range(0, len(evals["validation_0"][metric_key]), 5):
                curvas_rows.append({
                    "iteracion": epoch_i + 1,
                    "train_loss": round(float(evals["validation_0"][metric_key][epoch_i]), 5),
                    "test_loss": round(float(evals["validation_1"][metric_key][epoch_i]), 5),
                })

    df_res = pd.DataFrame(resultados)
    df_res.to_csv(resolve_path("baselines_resultados_caribe.csv"), index=False, encoding="utf-8")
    print("\nGuardado 'baselines_resultados_caribe.csv'")

    df_preds = pd.DataFrame(predicciones_test_rows)
    df_preds.to_csv(resolve_path("baselines_predicciones_caribe.csv"), index=False, encoding="utf-8")
    print(f"Guardado 'baselines_predicciones_caribe.csv' ({len(df_preds)} predicciones)")

    df_imp = pd.DataFrame(importances_rows)
    df_imp.to_csv(resolve_path("baselines_feature_importances_caribe.csv"), index=False, encoding="utf-8")
    print("Guardado 'baselines_feature_importances_caribe.csv'")

    df_curv = pd.DataFrame(curvas_rows)
    df_curv.to_csv(resolve_path("baselines_curvas_aprendizaje_caribe.csv"), index=False, encoding="utf-8")
    print("Guardado 'baselines_curvas_aprendizaje_caribe.csv'")

    # Resumen comparativo en consola
    print("\n" + "=" * 95)
    print("RESUMEN COMPARATIVO DE BASELINES CARIBE: CON ENSO VS SIN ENSO")
    print("=" * 95)
    cols_tabla = ["split", "modelo", "configuracion_enso", "RMSE_test", "MAE_test", "R2_test", "Pearson_r_test"]
    print(df_res[cols_tabla].to_string(index=False))

    df_loeo = df_res[df_res["split"].isin(["Split_1_LOEO_2015_2016", "Split_2_LOEO_2023_2024"])]
    prom_loeo = df_loeo.groupby(["modelo", "configuracion_enso"])[["RMSE_test", "MAE_test", "R2_test", "Pearson_r_test"]].mean().round(4)
    print("\n" + "=" * 95)
    print("PROMEDIO CONJUNTO LOEO CARIBE (EL NIÑO 2015-2016 + EL NIÑO 2023-2024):")
    print("=" * 95)
    print(prom_loeo.to_string())

    print("\n" + "=" * 95)
    print("AUDITORÍA DE ERROR SEGÚN CALIDAD CLIMÁTICA MUNICIPAL CARIBE (XGBoost en Split 1 2015-2016):")
    print("=" * 95)
    sub_xgb = df_preds[(df_preds["modelo"] == "XGBoost") & (df_preds["split"] == "Split_1_LOEO_2015_2016")]
    cal_res = sub_xgb.groupby("calidad_dato_climatico_alta")[["error_abs"]].agg(["count", "mean", "median"]).round(4)
    print(cal_res.to_string())
    r_dist = float(np.corrcoef(sub_xgb["ideam_dist_estacion_precip_km"], sub_xgb["error_abs"])[0, 1])
    print(f"Correlación de Pearson entre distancia a estación IDEAM y error absoluto: r = {r_dist:.4f}")


if __name__ == "__main__":
    main()
