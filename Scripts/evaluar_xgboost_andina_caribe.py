"""
Evaluación de XGBoost Macro-Regional Combinado (Región Andina + Región Caribe)
==============================================================================
Consolida los datasets maestros de ambas macro-regiones:
  - dataset_maestro_entrenamiento.csv        (Andina: 18,425 filas, 208 municipios, 3 dptos)
  - dataset_maestro_entrenamiento_caribe.csv (Caribe: 11,197 filas, 192 municipios, 7 dptos)
  -> Total combinado: 29,622 filas, 400 municipios, 10 departamentos, 5 cultivos.

Evalúa XGBoost (con pérdida pseudo-Huber) bajo los 3 esquemas temporales LOEO
(2015-2016, 2023-2024, Secuencial 2021-2024) y ablación ENSO (Con_ENSO vs Sin_ENSO),
reportando métricas tanto globales como desagregadas por sub-región (Andina y Caribe)
para contrastar el modelo conjunto frente a los modelos exclusivos regionales.

Salidas:
  - dataset_maestro_andina_caribe.csv
  - baselines_xgboost_andina_caribe.csv
  - baselines_predicciones_andina_caribe.csv
"""

import os
import time
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# Soporte para ejecución desde raíz o desde Scripts/
try:
    from paths import resolve_path, ROOT_DIR
except ImportError:
    from Scripts.paths import resolve_path, ROOT_DIR


def calcular_metricas(y_true, y_pred, y_true_raw=None):
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - (float(np.sum(err ** 2)) / ss_tot) if ss_tot > 0 else np.nan
    r_corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 2 else np.nan

    res = {
        "N": int(len(y_true)),
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
    print("=" * 95)
    print("ENTRENAMIENTO Y EVALUACIÓN DE XGBOOST MACRO-REGIONAL (ANDINA + CARIBE)")
    print("=" * 95)

    df_and = pd.read_csv(resolve_path("dataset_maestro_entrenamiento.csv"), dtype={"cod_dane_mun": str, "cod_dane_dep": str})
    df_car = pd.read_csv(resolve_path("dataset_maestro_entrenamiento_caribe.csv"), dtype={"cod_dane_mun": str, "cod_dane_dep": str})

    df_and["macro_region"] = "Andina"
    df_car["macro_region"] = "Caribe"

    # Alinear columnas exactas
    cols_comunes = list(df_and.columns)
    df_car = df_car[cols_comunes]

    df_comb = pd.concat([df_and, df_car], ignore_index=True)
    out_comb = resolve_path("dataset_maestro_andina_caribe.csv")
    df_comb.to_csv(out_comb, index=False, encoding="utf-8")

    print(f"Dataset Andina        : {len(df_and):,} filas | {df_and['cod_dane_mun'].nunique()} municipios | {df_and['departamento'].nunique()} dptos")
    print(f"Dataset Caribe        : {len(df_car):,} filas | {df_car['cod_dane_mun'].nunique()} municipios | {df_car['departamento'].nunique()} dptos")
    print(f"Dataset Combinado     : {len(df_comb):,} filas | {df_comb['cod_dane_mun'].nunique()} municipios | {df_comb['departamento'].nunique()} dptos")

    feats_sin_enso, feats_con_enso = obtener_bloques_features()
    df_dummies = pd.get_dummies(df_comb[["cultivo_cat", "departamento", "macro_region"]], drop_first=True, dtype=float)
    dummy_cols = list(df_dummies.columns)

    X_all_con = pd.concat([df_comb[feats_con_enso], df_dummies], axis=1)
    X_all_sin = pd.concat([df_comb[feats_sin_enso], df_dummies], axis=1)

    y_target = df_comb["eficiencia_clip"].values
    y_raw = df_comb["eficiencia"].values
    regiones = df_comb["macro_region"].values

    splits = {
        "Split_1_LOEO_2015_2016": {
            "test_mask": df_comb["año"].isin([2015, 2016]),
            "descripcion": "El Niño Severo 2015-2016 (LOEO Canónico)"
        },
        "Split_2_LOEO_2023_2024": {
            "test_mask": df_comb["año"].isin([2023, 2024]),
            "train_mask": df_comb["año"] <= 2022,
            "descripcion": "El Niño Reciente 2023-2024 (LOEO Reciente)"
        },
        "Split_3_Secuencial_2021_2024": {
            "test_mask": df_comb["año"] >= 2021,
            "train_mask": df_comb["año"] <= 2020,
            "descripcion": "Cuatrienio Reciente 2021-2024 (Secuencial Puro)"
        },
    }

    resultados = []
    predicciones_rows = []

    for split_nombre, s_info in splits.items():
        print(f"\n>>> Ejecutando {split_nombre} ({s_info['descripcion']})")
        test_mask = s_info["test_mask"].values
        train_mask = s_info.get("train_mask", ~s_info["test_mask"]).values
        print(f"    Train Combinado: {train_mask.sum():,} filas | Test Combinado: {test_mask.sum():,} filas")

        for enso_cfg, X_mat in [("Sin_ENSO", X_all_sin), ("Con_ENSO", X_all_con)]:
            X_tr, X_te = X_mat.iloc[train_mask].values, X_mat.iloc[test_mask].values
            y_tr, y_te = y_target[train_mask], y_target[test_mask]
            y_te_raw = y_raw[test_mask]
            reg_te = regiones[test_mask]

            t0 = time.time()
            model = XGBRegressor(
                n_estimators=220,
                max_depth=6,
                learning_rate=0.04,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="reg:pseudohubererror",
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            y_pred_tr = model.predict(X_tr)
            dt = time.time() - t0

            t_inf0 = time.perf_counter()
            _ = model.predict(X_te)
            t_inf = time.perf_counter() - t_inf0
            tps = round(len(y_te) / max(t_inf, 1e-6), 1)

            m_train = calcular_metricas(y_tr, y_pred_tr)

            # Evaluar en 3 ámbitos de test: Global (Andina+Caribe), Subconjunto Andina, Subconjunto Caribe
            for eval_scope, mask_sc in [
                ("Global_Andina+Caribe", np.ones(len(y_te), dtype=bool)),
                ("Subconjunto_Andina", reg_te == "Andina"),
                ("Subconjunto_Caribe", reg_te == "Caribe"),
            ]:
                m_te = calcular_metricas(y_te[mask_sc], y_pred[mask_sc], y_te_raw[mask_sc])
                resultados.append({
                    "split": split_nombre,
                    "split_descripcion": s_info["descripcion"],
                    "configuracion_enso": enso_cfg,
                    "modelo": "XGBoost_Combinado",
                    "ambito_evaluacion": eval_scope,
                    "N_test": m_te["N"],
                    "tiempo_seg": round(dt, 2),
                    "tps_inferencia": tps,
                    "RMSE_test": m_te["RMSE"],
                    "MAE_test": m_te["MAE"],
                    "R2_test": m_te["R2"],
                    "Pearson_r_test": m_te["Pearson_r"],
                    "Bias_test": m_te["Bias"],
                    "RMSE_test_raw": m_te["RMSE_raw"],
                    "MAE_test_raw": m_te["MAE_raw"],
                    "RMSE_train": m_train["RMSE"],
                    "MAE_train": m_train["MAE"],
                    "R2_train": m_train["R2"],
                })
                print(
                    f"      [{enso_cfg:<8} | {eval_scope:<22}] N={m_te['N']:<4d} | RMSE={m_te['RMSE']:.4f} | "
                    f"MAE={m_te['MAE']:.4f} | R2={m_te['R2']:.4f} | r={m_te['Pearson_r']:.4f}"
                )

            if enso_cfg == "Con_ENSO":
                df_te = df_comb.iloc[test_mask].copy()
                for idx_r, (_, r_te) in enumerate(df_te.iterrows()):
                    predicciones_rows.append({
                        "split": split_nombre,
                        "macro_region": r_te["macro_region"],
                        "departamento": r_te["departamento"],
                        "municipio": r_te["municipio"],
                        "cod_dane_mun": r_te["cod_dane_mun"],
                        "cultivo_cat": r_te["cultivo_cat"],
                        "año": int(r_te["año"]),
                        "y_real_clip": round(float(y_te[idx_r]), 4),
                        "y_real_raw": round(float(y_te_raw[idx_r]), 4),
                        "y_pred": round(float(y_pred[idx_r]), 4),
                        "residual": round(float(y_pred[idx_r] - y_te[idx_r]), 4),
                        "error_abs": round(float(abs(y_pred[idx_r] - y_te[idx_r])), 4),
                    })

    df_res = pd.DataFrame(resultados)
    df_res.to_csv(resolve_path("baselines_xgboost_andina_caribe.csv"), index=False, encoding="utf-8")
    df_preds = pd.DataFrame(predicciones_rows)
    df_preds.to_csv(resolve_path("baselines_predicciones_andina_caribe.csv"), index=False, encoding="utf-8")

    # Promedio LOEO (Splits 1 y 2)
    df_loeo = df_res[df_res["split"].isin(["Split_1_LOEO_2015_2016", "Split_2_LOEO_2023_2024"])]
    prom_loeo = (
        df_loeo.groupby(["ambito_evaluacion", "configuracion_enso"])[["RMSE_test", "MAE_test", "R2_test", "Pearson_r_test"]]
        .mean()
        .round(4)
    )
    print("\n" + "=" * 95)
    print("PROMEDIO CONJUNTO LOEO (2015-2016 + 2023-2024) - XGBOOST ENTRENADO EN ANDINA + CARIBE:")
    print("=" * 95)
    print(prom_loeo.to_string())


if __name__ == "__main__":
    main()
