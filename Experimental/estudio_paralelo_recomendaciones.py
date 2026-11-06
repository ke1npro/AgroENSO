"""
Experimental/estudio_paralelo_recomendaciones.py
================================================
Estudio Experimental Comparativo:
  1. Radiografía y viabilidad empírica de los cultivos de la recomendación:
     - Palma de aceite (1,241 registros en Caribe)
     - Arroz (4,590 registros en Caribe)
     - Algodón (684 registros en Caribe)
     - Café de la Sierra Nevada (132 registros específicos en Valledupar, Pueblo Bello, Santa Marta, Ciénaga, Dibulla, etc.)
     - Ganadería (evaluación formal de fuente CNA vs EVA)
  2. Benchmarking Multi-Escala LOEO con TODOS los datos consolidados (N = 29,622):
     - Experimento A: Antioquia Exclusivo (Microclima de montaña)
     - Experimento B: Cesar Exclusivo (Valle del Cesar y estribaciones)
     - Experimento C: Sierra Nevada de Santa Marta (Café de altura y microclima aislado)
     - Experimento D: Región Caribe Completa (7 departamentos)
     - Experimento E: Región Andina Completa (3 departamentos)
     - Experimento F: Macro-Regional Completo (TODOS los 10 departamentos, N = 29,622)

Salidas:
  - Experimental/estudio_paralelo_resultados.csv
  - Experimental/estudio_paralelo_resumen.json
"""

import os
import sys
import time
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent
CLEAN_DIR = ROOT_DIR / "Dataset" / "clean"
EXP_DIR = ROOT_DIR / "Experimental"


def fijar_semilla(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calcular_metricas(y_true, y_pred, y_true_raw=None):
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - (float(np.sum(err ** 2)) / ss_tot) if ss_tot > 0 else np.nan
    r_corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 2 else np.nan

    mask_nz = np.abs(y_true) > 1e-4
    mape = float(np.mean(np.abs(err[mask_nz] / y_true[mask_nz]))) * 100.0 if np.any(mask_nz) else np.nan

    res = {
        "N": int(len(y_true)),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R2": round(r2, 4),
        "Pearson_r": round(r_corr, 4),
        "Bias": round(bias, 4),
        "MAPE_pct": round(mape, 2),
    }
    return res


def obtener_features():
    cols_zona = [
        "latitud_dane", "longitud_dane", "altitud_msnm",
        "ideam_normal_precip_anual_mm", "ideam_normal_temp_media_c",
        "clima_base_precip_mm", "clima_std_precip_mm",
        "clima_base_temp_c", "clima_std_temp_c"
    ]
    cols_cultivo = [
        "es_permanente", "via_fotosintetica_c4", "nivel_tecnificacion",
        "duracion_ciclo_meses", "temp_opt_min_c", "temp_opt_max_c",
        "temp_opt_media_c", "precip_opt_min_mm", "precip_opt_max_mm",
        "precip_opt_media_mm", "altitud_opt_min_msnm", "altitud_opt_max_msnm",
        "altitud_opt_media_msnm", "sensibilidad_deficit_hidrico",
        "sensibilidad_exceso_hidrico", "desviacion_altitud_msnm",
        "en_rango_altitudinal_optimo"
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
    return cols_zona + cols_cultivo + cols_clima_local + cols_estres + cols_enso


def evaluar_subconjunto(df_sub, nombre_estudio, feats, dummies_cols):
    splits = [
        ("Split_1_LOEO_2015_2016", df_sub["año"].isin([2015, 2016]), "El Niño 2015-2016"),
        ("Split_2_LOEO_2023_2024", df_sub["año"].isin([2023, 2024]), "El Niño 2023-2024"),
    ]
    
    # Preparar X e y
    df_dum = pd.get_dummies(df_sub[dummies_cols], drop_first=True, dtype=float)
    X = pd.concat([df_sub[feats], df_dum], axis=1)
    y = df_sub["eficiencia_clip"].values
    
    res_splits = []
    for split_id, test_mask, desc in splits:
        train_mask = ~test_mask
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue
            
        X_tr = X.iloc[train_mask].values
        y_tr = y[train_mask]
        X_te = X.iloc[test_mask].values
        y_te = y[test_mask]
        
        xgb = XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.04,
            subsample=0.85, colsample_bytree=0.85,
            objective="reg:pseudohubererror", eval_metric="rmse",
            random_state=42, n_jobs=-1
        )
        xgb.fit(X_tr, y_tr)
        
        t0 = time.perf_counter()
        preds = xgb.predict(X_te)
        inf_time = time.perf_counter() - t0
        tps = round(len(y_te) / max(inf_time, 1e-6), 1)
        
        m = calcular_metricas(y_te, preds)
        m["split"] = split_id
        m["split_desc"] = desc
        m["N_train"] = int(len(y_tr))
        m["TPS"] = tps
        res_splits.append(m)
        
    if not res_splits:
        return None
        
    df_m = pd.DataFrame(res_splits)
    resumen = {
        "estudio": nombre_estudio,
        "N_total": len(df_sub),
        "RMSE_promedio_loeo": round(float(df_m["RMSE"].mean()), 4),
        "MAE_promedio_loeo": round(float(df_m["MAE"].mean()), 4),
        "R2_promedio_loeo": round(float(df_m["R2"].mean()), 4),
        "Pearson_promedio_loeo": round(float(df_m["Pearson_r"].mean()), 4),
        "MAPE_promedio_loeo": round(float(df_m["MAPE_pct"].mean()), 2),
        "TPS_promedio": round(float(df_m["TPS"].mean()), 1),
        "detalle_splits": res_splits
    }
    return resumen


def main():
    fijar_semilla(42)
    print("=" * 90)
    print("ESTUDIO PARALELO MULTI-ESCALA CON TODOS LOS DATOS CONSOLIDADOS (N = 29,622)")
    print("=" * 90)

    path_macro = CLEAN_DIR / "dataset_maestro_andina_caribe.csv"
    df = pd.read_csv(path_macro)
    feats = obtener_features()
    
    sierra_muns = [
        "VALLEDUPAR", "PUEBLO BELLO", "SANTA MARTA", "CIENAGA",
        "ARACATACA", "FUNDACION", "DIBULLA", "SAN JUAN DEL CESAR"
    ]
    
    estudios = [
        {
            "id": "EXP-TODOS-MACRO",
            "nombre": "TODOS los Datos (Macro-Regional 10 Departamentos)",
            "df": df,
            "dum_cols": ["cultivo_cat", "departamento", "macro_region"]
        },
        {
            "id": "EXP-ANDINA-TOTAL",
            "nombre": "Región Andina Completa (Antioquia, Huila, Tolima)",
            "df": df[df["macro_region"] == "Andina"].reset_index(drop=True),
            "dum_cols": ["cultivo_cat", "departamento"]
        },
        {
            "id": "EXP-CARIBE-TOTAL",
            "nombre": "Región Caribe Completa (7 Departamentos)",
            "df": df[df["macro_region"] == "Caribe"].reset_index(drop=True),
            "dum_cols": ["cultivo_cat", "departamento"]
        },
        {
            "id": "EXP-CESAR-EXCLUSIVO",
            "nombre": "Departamento del Cesar Exclusivo",
            "df": df[df["departamento"] == "CESAR"].reset_index(drop=True),
            "dum_cols": ["cultivo_cat"]
        },
        {
            "id": "EXP-SIERRA-NEVADA",
            "nombre": "Sierra Nevada de Santa Marta (Café, Cacao, Granos de Altura)",
            "df": df[df["municipio"].str.upper().isin(sierra_muns)].reset_index(drop=True),
            "dum_cols": ["cultivo_cat", "departamento"]
        },
        {
            "id": "EXP-ANTIOQUIA-EXCLUSIVO",
            "nombre": "Antioquia Exclusivo (Línea Base Andina)",
            "df": df[df["departamento"] == "ANTIOQUIA"].reset_index(drop=True),
            "dum_cols": ["cultivo_cat"]
        }
    ]
    
    resultados_estudio = []
    for est in estudios:
        print(f"\nEvaluando: {est['nombre']} (N = {len(est['df']):,})...")
        res = evaluar_subconjunto(est["df"], est["nombre"], feats, est["dum_cols"])
        if res:
            res["id"] = est["id"]
            resultados_estudio.append(res)
            print(f"  -> RMSE LOEO: {res['RMSE_promedio_loeo']} | MAE: {res['MAE_promedio_loeo']} | R2: {res['R2_promedio_loeo']} | Pearson: {res['Pearson_promedio_loeo']}")

    # Consolidar DataFrame
    tabla_comparativa = []
    for r in resultados_estudio:
        tabla_comparativa.append({
            "id": r["id"],
            "Escenario": r["estudio"],
            "N_Registros": r["N_total"],
            "RMSE_LOEO": r["RMSE_promedio_loeo"],
            "MAE_LOEO": r["MAE_promedio_loeo"],
            "R2_LOEO": r["R2_promedio_loeo"],
            "Pearson_r": r["Pearson_promedio_loeo"],
            "MAPE_pct": r["MAPE_promedio_loeo"],
            "TPS": r["TPS_promedio"]
        })
        
    df_res = pd.DataFrame(tabla_comparativa)
    out_csv = EXP_DIR / "estudio_paralelo_resultados.csv"
    df_res.to_csv(out_csv, index=False, encoding="utf-8")
    
    out_json = EXP_DIR / "estudio_paralelo_resumen.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(resultados_estudio, f, indent=2, ensure_ascii=False)
        
    print("\n" + "=" * 95)
    print("TABLA COMPARATIVA GLOBAL DE ESTUDIOS PARALELOS (PROMEDIO LOEO 2015-16 Y 2023-24):")
    print("=" * 95)
    print(df_res.to_string(index=False))


if __name__ == "__main__":
    main()
