"""
Experimental/ejecutar_experimentos.py
=====================================
Suite de experimentación rápida e iteración técnica.
Ejecuta las tres configuraciones clave solicitadas:
  1. Baseline con Región Antioquia (XGBoost + Ridge, LOEO)
  2. Baseline con TODAS las Regiones (10 departamentos: Andina + Caribe, N = 29,622, LOEO)
  3. Modelo de Red Modular FiLM con TODAS las Regiones (Doble rama + FiLM + MC Dropout, LOEO)

Guarda:
  - Experimental/resultados_simplificados.csv
  - Experimental/datos_experimentales.json
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
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Rutas dinámicas del proyecto
ROOT_DIR = Path(__file__).resolve().parent.parent
CLEAN_DIR = ROOT_DIR / "Dataset" / "clean"
RESULTS_DIR = ROOT_DIR / "Resultados"
EXP_DIR = ROOT_DIR / "Experimental"


def fijar_semilla(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calcular_metricas(y_true, y_pred, y_true_raw=None, sigma_mc=None, sigma_total=None):
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - (float(np.sum(err ** 2)) / ss_tot) if ss_tot > 0 else np.nan
    r_corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 2 else np.nan

    # MAPE sobre eficiencia_clip
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

    if y_true_raw is not None:
        err_raw = y_pred - y_true_raw
        res["RMSE_raw"] = round(float(np.sqrt(np.mean(err_raw ** 2))), 4)
        res["MAE_raw"] = round(float(np.mean(np.abs(err_raw))), 4)

    if sigma_mc is not None and sigma_total is not None:
        ic_inf = y_pred - 1.96 * sigma_total
        ic_sup = y_pred + 1.96 * sigma_total
        picp = float(np.mean((y_true >= ic_inf) & (y_true <= ic_sup))) * 100.0
        res["sigma_mc_media"] = round(float(np.mean(sigma_mc)), 4)
        res["PICP_95_pct"] = round(picp, 2)

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
    return cols_zona, cols_cultivo, cols_clima_local, cols_estres, cols_enso


# Definición de Arquitectura Red Modular FiLM
class RedModularFiLM(nn.Module):
    def __init__(self, dim_ctx: int, dim_clim: int, dim_hidden: int = 64, dropout_rate: float = 0.20):
        super().__init__()
        self.dropout_rate = dropout_rate

        # Rama 1: Contexto Biofísico Zonal y Ecofisiológico
        self.branch_ctx = nn.Sequential(
            nn.Linear(dim_ctx, dim_hidden),
            nn.BatchNorm1d(dim_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_hidden, dim_hidden),
            nn.BatchNorm1d(dim_hidden),
            nn.ReLU()
        )

        # Rama 2: Señal Climática Local y ENSO
        self.branch_clim = nn.Sequential(
            nn.Linear(dim_clim, dim_hidden),
            nn.BatchNorm1d(dim_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_hidden, dim_hidden),
            nn.BatchNorm1d(dim_hidden),
            nn.ReLU()
        )

        # Capa FiLM
        self.film_gamma = nn.Linear(dim_hidden, dim_hidden)
        self.film_beta = nn.Linear(dim_hidden, dim_hidden)

        # Cabeza Regresora
        self.head = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_hidden // 2, 1)
        )

    def forward(self, x_ctx, x_clim):
        h_ctx = self.branch_ctx(x_ctx)
        h_clim = self.branch_clim(x_clim)

        gamma = 1.0 + torch.tanh(self.film_gamma(h_ctx))
        beta = self.film_beta(h_ctx)
        z = gamma * h_clim + beta

        y_hat = self.head(z).squeeze(-1)
        return y_hat


def predecir_mc_dropout(model, t_ctx, t_clim, n_passes: int = 30):
    model.train()
    passes = []
    with torch.no_grad():
        for _ in range(n_passes):
            pred = model(t_ctx, t_clim).cpu().numpy()
            passes.append(pred)
    arr = np.array(passes)
    mu = np.mean(arr, axis=0)
    sigma = np.std(arr, axis=0)
    return mu, sigma


def main():
    fijar_semilla(42)
    print("=" * 90)
    print("EJECUTANDO SUITE EXPERIMENTAL: ANTIOQUIA, TODAS REGIONES (BASELINE) Y TODAS (RED MODULAR)")
    print("=" * 90)

    # 1. Cargar datos
    path_and = CLEAN_DIR / "dataset_maestro_entrenamiento.csv"
    path_car = CLEAN_DIR / "dataset_maestro_entrenamiento_caribe.csv"
    path_comb = CLEAN_DIR / "dataset_maestro_andina_caribe.csv"

    if not path_comb.exists():
        print("Generando dataset_maestro_andina_caribe.csv...")
        df_and = pd.read_csv(path_and, dtype={"cod_dane_mun": str, "cod_dane_dep": str})
        df_car = pd.read_csv(path_car, dtype={"cod_dane_mun": str, "cod_dane_dep": str})
        df_and["macro_region"] = "Andina"
        df_car["macro_region"] = "Caribe"
        cols_comunes = list(df_and.columns)
        df_comb = pd.concat([df_and, df_car[cols_comunes]], ignore_index=True)
        df_comb.to_csv(path_comb, index=False, encoding="utf-8")
    else:
        df_comb = pd.read_csv(path_comb, dtype={"cod_dane_mun": str, "cod_dane_dep": str})

    cols_zona, cols_cultivo, cols_clima_local, cols_estres, cols_enso = obtener_bloques_features()
    feats_con_enso = cols_zona + cols_cultivo + cols_clima_local + cols_estres + cols_enso
    feats_sin_enso = cols_zona + cols_cultivo + cols_clima_local + cols_estres

    resultados_lista = []

    # -------------------------------------------------------------
    # EXPERIMENTO 1: BASELINE CON REGIÓN ANTIOQUIA
    # -------------------------------------------------------------
    print("\n>>> EXPERIMENTO 1: Baseline con Región Antioquia (Exclusivo)")
    df_ant = df_comb[df_comb["departamento"].str.upper() == "ANTIOQUIA"].copy().reset_index(drop=True)
    print(f"Registros Antioquia: {len(df_ant):,} filas | Cultivos: {df_ant['cultivo_cat'].nunique()}")

    # Dummies para Antioquia (solo cultivo_cat)
    df_ant_dum = pd.get_dummies(df_ant[["cultivo_cat"]], drop_first=True, dtype=float)
    X_ant_con = pd.concat([df_ant[feats_con_enso], df_ant_dum], axis=1)
    y_ant_clip = df_ant["eficiencia_clip"].values
    y_ant_raw = df_ant["eficiencia"].values

    splits_loeo_ant = [
        ("Split_1_LOEO_2015_2016", df_ant["año"].isin([2015, 2016]), "El Niño Severo 2015-2016"),
        ("Split_2_LOEO_2023_2024", df_ant["año"].isin([2023, 2024]), "El Niño Reciente 2023-2024")
    ]

    for split_nombre, test_mask, desc in splits_loeo_ant:
        train_mask = ~test_mask
        X_tr = X_ant_con.iloc[train_mask].values
        y_tr = y_ant_clip[train_mask]
        X_te = X_ant_con.iloc[test_mask].values
        y_te = y_ant_clip[test_mask]
        y_te_raw = y_ant_raw[test_mask]

        # XGBoost
        t0 = time.time()
        xgb = XGBRegressor(
            n_estimators=220, max_depth=6, learning_rate=0.04,
            subsample=0.85, colsample_bytree=0.85,
            objective="reg:pseudohubererror", eval_metric="rmse",
            random_state=42, n_jobs=-1
        )
        xgb.fit(X_tr, y_tr)
        t_fit = time.time() - t0

        t_inf0 = time.perf_counter()
        y_pred = xgb.predict(X_te)
        t_inf = time.perf_counter() - t_inf0
        tps = round(len(y_te) / max(t_inf, 1e-6), 1)

        m = calcular_metricas(y_te, y_pred, y_te_raw)
        resultados_lista.append({
            "experimento_id": "EXP-1-ANT-XGB",
            "ambito": "Antioquia Exclusivo",
            "tipo_modelo": "Baseline Árbol (XGBoost)",
            "modelo": "XGBoost Regressor",
            "split": split_nombre,
            "split_descripcion": desc,
            "configuracion_enso": "Con_ENSO",
            "N_train": int(len(y_tr)),
            "N_test": m["N"],
            "tiempo_seg": round(t_fit, 2),
            "TPS": tps,
            "RMSE_test": m["RMSE"],
            "MAE_test": m["MAE"],
            "R2_test": m["R2"],
            "Pearson_r_test": m["Pearson_r"],
            "Bias_test": m["Bias"],
            "MAPE_test_pct": m["MAPE_pct"],
            "RMSE_test_raw": m["RMSE_raw"],
            "MAE_test_raw": m["MAE_raw"],
            "sigma_mc_media": np.nan,
            "PICP_95_pct": np.nan,
        })

    # -------------------------------------------------------------
    # EXPERIMENTO 2: BASELINE CON TODAS LAS REGIONES (10 Dptos)
    # -------------------------------------------------------------
    print("\n>>> EXPERIMENTO 2: Baseline con TODAS las Regiones (10 Departamentos, N = 29,622)")
    df_dummies_all = pd.get_dummies(df_comb[["cultivo_cat", "departamento", "macro_region"]], drop_first=True, dtype=float)
    X_all_con = pd.concat([df_comb[feats_con_enso], df_dummies_all], axis=1)
    y_all_clip = df_comb["eficiencia_clip"].values
    y_all_raw = df_comb["eficiencia"].values

    splits_loeo_all = [
        ("Split_1_LOEO_2015_2016", df_comb["año"].isin([2015, 2016]), "El Niño Severo 2015-2016"),
        ("Split_2_LOEO_2023_2024", df_comb["año"].isin([2023, 2024]), "El Niño Reciente 2023-2024")
    ]

    for split_nombre, test_mask, desc in splits_loeo_all:
        train_mask = ~test_mask
        X_tr = X_all_con.iloc[train_mask].values
        y_tr = y_all_clip[train_mask]
        X_te = X_all_con.iloc[test_mask].values
        y_te = y_all_clip[test_mask]
        y_te_raw = y_all_raw[test_mask]

        t0 = time.time()
        xgb = XGBRegressor(
            n_estimators=220, max_depth=6, learning_rate=0.04,
            subsample=0.85, colsample_bytree=0.85,
            objective="reg:pseudohubererror", eval_metric="rmse",
            random_state=42, n_jobs=-1
        )
        xgb.fit(X_tr, y_tr)
        t_fit = time.time() - t0

        t_inf0 = time.perf_counter()
        y_pred = xgb.predict(X_te)
        t_inf = time.perf_counter() - t_inf0
        tps = round(len(y_te) / max(t_inf, 1e-6), 1)

        m = calcular_metricas(y_te, y_pred, y_te_raw)
        resultados_lista.append({
            "experimento_id": "EXP-2-ALL-XGB",
            "ambito": "TODAS las Regiones (10 Dptos)",
            "tipo_modelo": "Baseline Árbol (XGBoost)",
            "modelo": "XGBoost Regressor",
            "split": split_nombre,
            "split_descripcion": desc,
            "configuracion_enso": "Con_ENSO",
            "N_train": int(len(y_tr)),
            "N_test": m["N"],
            "tiempo_seg": round(t_fit, 2),
            "TPS": tps,
            "RMSE_test": m["RMSE"],
            "MAE_test": m["MAE"],
            "R2_test": m["R2"],
            "Pearson_r_test": m["Pearson_r"],
            "Bias_test": m["Bias"],
            "MAPE_test_pct": m["MAPE_pct"],
            "RMSE_test_raw": m["RMSE_raw"],
            "MAE_test_raw": m["MAE_raw"],
            "sigma_mc_media": np.nan,
            "PICP_95_pct": np.nan,
        })

    # -------------------------------------------------------------
    # EXPERIMENTO 3: MODELO RED MODULAR FiLM CON TODAS LAS REGIONES
    # -------------------------------------------------------------
    print("\n>>> EXPERIMENTO 3: Red Neuronal Modular FiLM con TODAS las Regiones (10 Dptos)")
    cols_ctx_num = cols_zona + cols_cultivo
    cols_clim_num = cols_clima_local + cols_estres + cols_enso
    dummy_cols_list = list(df_dummies_all.columns)

    for split_nombre, test_mask, desc in splits_loeo_all:
        train_mask = ~test_mask
        idx_train_full = np.where(train_mask)[0]
        idx_test = np.where(test_mask)[0]

        np.random.seed(42)
        perm = np.random.permutation(len(idx_train_full))
        n_val = int(len(idx_train_full) * 0.15)
        idx_val = idx_train_full[perm[:n_val]]
        idx_train = idx_train_full[perm[n_val:]]

        scaler_ctx = StandardScaler()
        scaler_clim = StandardScaler()

        X_ctx_tr_num = scaler_ctx.fit_transform(df_comb.iloc[idx_train][cols_ctx_num])
        X_ctx_val_num = scaler_ctx.transform(df_comb.iloc[idx_val][cols_ctx_num])
        X_ctx_te_num = scaler_ctx.transform(df_comb.iloc[idx_test][cols_ctx_num])
        X_ctx_trfull_num = scaler_ctx.transform(df_comb.iloc[idx_train_full][cols_ctx_num])

        X_clim_tr_num = scaler_clim.fit_transform(df_comb.iloc[idx_train][cols_clim_num])
        X_clim_val_num = scaler_clim.transform(df_comb.iloc[idx_val][cols_clim_num])
        X_clim_te_num = scaler_clim.transform(df_comb.iloc[idx_test][cols_clim_num])
        X_clim_trfull_num = scaler_clim.transform(df_comb.iloc[idx_train_full][cols_clim_num])

        X_dum_tr = df_dummies_all.iloc[idx_train].values
        X_dum_val = df_dummies_all.iloc[idx_val].values
        X_dum_te = df_dummies_all.iloc[idx_test].values
        X_dum_trfull = df_dummies_all.iloc[idx_train_full].values

        X_ctx_tr = np.hstack([X_ctx_tr_num, X_dum_tr])
        X_ctx_val = np.hstack([X_ctx_val_num, X_dum_val])
        X_ctx_te = np.hstack([X_ctx_te_num, X_dum_te])
        X_ctx_trfull = np.hstack([X_ctx_trfull_num, X_dum_trfull])

        y_tr = y_all_clip[idx_train].astype(np.float32)
        y_val = y_all_clip[idx_val].astype(np.float32)
        y_te = y_all_clip[idx_test].astype(np.float32)
        y_tr_full = y_all_clip[idx_train_full].astype(np.float32)
        y_te_raw = y_all_raw[idx_test]

        t_ctx_tr = torch.tensor(X_ctx_tr, dtype=torch.float32)
        t_clim_tr = torch.tensor(X_clim_tr_num, dtype=torch.float32)
        t_y_tr = torch.tensor(y_tr, dtype=torch.float32)

        t_ctx_val = torch.tensor(X_ctx_val, dtype=torch.float32)
        t_clim_val = torch.tensor(X_clim_val_num, dtype=torch.float32)
        t_y_val = torch.tensor(y_val, dtype=torch.float32)

        t_ctx_te = torch.tensor(X_ctx_te, dtype=torch.float32)
        t_clim_te = torch.tensor(X_clim_te_num, dtype=torch.float32)

        t_ctx_trfull = torch.tensor(X_ctx_trfull, dtype=torch.float32)
        t_clim_trfull = torch.tensor(X_clim_trfull_num, dtype=torch.float32)

        dataset_tr = TensorDataset(t_ctx_tr, t_clim_tr, t_y_tr)
        loader_tr = DataLoader(dataset_tr, batch_size=128, shuffle=True)

        model = RedModularFiLM(
            dim_ctx=X_ctx_tr.shape[1],
            dim_clim=X_clim_tr_num.shape[1],
            dim_hidden=64,
            dropout_rate=0.20
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
        criterion = nn.HuberLoss(delta=0.5)

        t0 = time.time()
        best_val_loss = float("inf")
        patience, p_counter = 8, 0

        for epoch in range(1, 41):
            model.train()
            for b_ctx, b_clim, b_y in loader_tr:
                optimizer.zero_grad()
                pred = model(b_ctx, b_clim)
                loss = criterion(pred, b_y)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(t_ctx_val, t_clim_val)
                val_loss = float(criterion(val_pred, t_y_val).item())

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                p_counter = 0
            else:
                p_counter += 1
                if p_counter >= patience:
                    break

        t_fit = time.time() - t0

        # Incertidumbre con MC Dropout (30 pases)
        t_inf0 = time.perf_counter()
        y_pred_mc, sigma_mc = predecir_mc_dropout(model, t_ctx_te, t_clim_te, n_passes=30)
        t_inf = time.perf_counter() - t_inf0
        tps = round(len(y_te) / max(t_inf, 1e-6), 1)

        model.eval()
        with torch.no_grad():
            y_pred_val = model(t_ctx_val, t_clim_val).numpy()
        sigma_aleat = float(np.std(y_val - y_pred_val))
        sigma_total = np.sqrt(sigma_mc ** 2 + sigma_aleat ** 2)

        m = calcular_metricas(y_te, y_pred_mc, y_te_raw, sigma_mc=sigma_mc, sigma_total=sigma_total)

        resultados_lista.append({
            "experimento_id": "EXP-3-ALL-FILM",
            "ambito": "TODAS las Regiones (10 Dptos)",
            "tipo_modelo": "Deep Learning Modular (FiLM)",
            "modelo": "Red Modular FiLM (MC Dropout)",
            "split": split_nombre,
            "split_descripcion": desc,
            "configuracion_enso": "Con_ENSO",
            "N_train": int(len(idx_train_full)),
            "N_test": m["N"],
            "tiempo_seg": round(t_fit, 2),
            "TPS": tps,
            "RMSE_test": m["RMSE"],
            "MAE_test": m["MAE"],
            "R2_test": m["R2"],
            "Pearson_r_test": m["Pearson_r"],
            "Bias_test": m["Bias"],
            "MAPE_test_pct": m["MAPE_pct"],
            "RMSE_test_raw": m["RMSE_raw"],
            "MAE_test_raw": m["MAE_raw"],
            "sigma_mc_media": m["sigma_mc_media"],
            "PICP_95_pct": m["PICP_95_pct"],
        })

    # Consolidar Dataframe
    df_res = pd.DataFrame(resultados_lista)
    out_csv = EXP_DIR / "resultados_simplificados.csv"
    df_res.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\nResultados guardados exitosamente en: {out_csv}")

    # Tabla resumen por experimento
    resumen = df_res.groupby(["experimento_id", "ambito", "tipo_modelo", "modelo"])[
        ["RMSE_test", "MAE_test", "R2_test", "Pearson_r_test", "MAPE_test_pct", "TPS"]
    ].mean().round(4).reset_index()

    print("\n" + "=" * 95)
    print("RESUMEN SIMPLIFICADO DE LOS 3 EXPERIMENTOS (PROMEDIO LOEO: 2015-2016 Y 2023-2024)")
    print("=" * 95)
    print(resumen.to_string(index=False))

    # Guardar JSON estructurado para el visor web simplificado
    json_payload = {
        "fecha_ejecucion": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_experimentos": len(resumen),
        "resumen_ejecutivo": resumen.to_dict(orient="records"),
        "detalle_splits": df_res.to_dict(orient="records"),
        "comparativa": [
            {
                "id": "antioquia_baseline",
                "titulo": "1. Baseline Antioquia Exclusivo",
                "subtitulo": "XGBoost entrenado únicamente sobre los municipios de Antioquia (N = 10,729)",
                "rmse": float(resumen[resumen["experimento_id"] == "EXP-1-ANT-XGB"]["RMSE_test"].values[0]),
                "mae": float(resumen[resumen["experimento_id"] == "EXP-1-ANT-XGB"]["MAE_test"].values[0]),
                "r2": float(resumen[resumen["experimento_id"] == "EXP-1-ANT-XGB"]["R2_test"].values[0]),
                "tps": float(resumen[resumen["experimento_id"] == "EXP-1-ANT-XGB"]["TPS"].values[0]),
                "ventaja": "Alta especialización local; captura la orografía y microclimas de la cordillera central y occidental antioqueña.",
                "desafio": "Menor diversidad agroclimática para extrapolar a llanuras o valles cálidos del Caribe."
            },
            {
                "id": "todas_baseline",
                "titulo": "2. Baseline TODAS las Regiones",
                "subtitulo": "XGBoost macro-regional entrenado sobre los 10 departamentos (N = 29,622)",
                "rmse": float(resumen[resumen["experimento_id"] == "EXP-2-ALL-XGB"]["RMSE_test"].values[0]),
                "mae": float(resumen[resumen["experimento_id"] == "EXP-2-ALL-XGB"]["MAE_test"].values[0]),
                "r2": float(resumen[resumen["experimento_id"] == "EXP-2-ALL-XGB"]["R2_test"].values[0]),
                "tps": float(resumen[resumen["experimento_id"] == "EXP-2-ALL-XGB"]["TPS"].values[0]),
                "ventaja": "Aprovecha el volumen masivo de datos; aprende interacciones climáticas compartidas a nivel nacional.",
                "desafio": "Riesgo de sesgo por asimetría ENSO entre el Caribe seco y el interior andino si no se modulan los pesos zonalmente."
            },
            {
                "id": "todas_red_modular",
                "titulo": "3. Red Modular FiLM TODAS las Regiones",
                "subtitulo": "Deep Learning de doble rama con modulación FiLM y Monte Carlo Dropout (N = 29,622)",
                "rmse": float(resumen[resumen["experimento_id"] == "EXP-3-ALL-FILM"]["RMSE_test"].values[0]),
                "mae": float(resumen[resumen["experimento_id"] == "EXP-3-ALL-FILM"]["MAE_test"].values[0]),
                "r2": float(resumen[resumen["experimento_id"] == "EXP-3-ALL-FILM"]["R2_test"].values[0]),
                "tps": float(resumen[resumen["experimento_id"] == "EXP-3-ALL-FILM"]["TPS"].values[0]),
                "ventaja": "Arquitectura biofísica guiada; condiciona la respuesta al clima según la altitud y ecofisiología. Provee incertidumbre probabilística (PICP 95%).",
                "desafio": "Mayor costo computacional en inferencia estocástica (30 pases MC), balanceado con altísima representatividad física."
            }
        ]
    }

    out_json = EXP_DIR / "datos_experimentales.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, ensure_ascii=False)
    print(f"JSON experimental guardado en: {out_json}")


if __name__ == "__main__":
    main()
