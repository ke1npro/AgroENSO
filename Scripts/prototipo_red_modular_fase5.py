"""
Fase 5 - Prototipo de Red Neuronal Modular de Doble Rama (Andina Exclusivo y Caribe Exclusivo)
==============================================================================================
Implementa en PyTorch la arquitectura modular propuesta en el Plan de Acción (Sección 5):
  - Rama 1 (Contexto Biofísico Zonal y Ecofisiológico del Cultivo):
      h_ctx = f_MLP(x_zona, x_cultivo, dummies_depto_cultivo)
  - Rama 2 (Señal Climática Local, Estrés y ENSO ONI):
      h_clima = g_MLP(x_clima_local, x_estres, [x_ONI])
  - Capa de Modulación Contextual (Ablación formal de diseño):
      * Variante A (Gating Sigmoide):
          z = h_clima * sigmoid(W_g * h_ctx + b_g)
      * Variante B (FiLM Completo - Feature-wise Linear Modulation):
          gamma = 1.0 + tanh(W_g * h_ctx + b_g)
          beta  = W_b * h_ctx + b_b
          z = gamma * h_clima + beta
  - Cuantificación de Incertidumbre (Gal & Ghahramani, 2016):
      Monte Carlo Dropout con S = 30 pases estocásticos en inferencia (p = 0.20).

Entrenamientos ejecutados de forma 100% independiente (exclusivos por región):
  1. Prototipo Exclusivo Región Andina (dataset_maestro_entrenamiento.csv, N = 18,425)
  2. Prototipo Exclusivo Región Caribe (dataset_maestro_entrenamiento_caribe.csv, N = 11,197)

Salidas:
  - red_modular_resultados_andina.csv
  - red_modular_resultados_caribe.csv
  - red_modular_predicciones_andina.csv
  - red_modular_predicciones_caribe.csv
  - red_modular_curvas_entrenamiento.csv
"""

import os
import time
import random
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

# Soporte para ejecución desde raíz o desde Scripts/
try:
    from paths import resolve_path, ROOT_DIR
except ImportError:
    from Scripts.paths import resolve_path, ROOT_DIR


def fijar_semilla(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def obtener_bloques_features():
    # Rama 1: Contexto Zonal y Cultivo (26 continuas)
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
    cols_rama1 = cols_zona + cols_cultivo

    # Rama 2: Clima Local + Estrés + ENSO (59 continuas Con_ENSO / 33 Sin_ENSO)
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

    rama2_sin_enso = cols_clima_local + cols_estres
    rama2_con_enso = cols_clima_local + cols_estres + cols_enso
    return cols_rama1, rama2_sin_enso, rama2_con_enso


class RedModularAgroENSO(nn.Module):
    """
    Arquitectura Neuronal de Doble Rama con Modulación Contextual (Sigmoide vs FiLM).
    """

    def __init__(self, dim_ctx: int, dim_clima: int, hidden_dim: int = 64, modo_modulacion: str = "FiLM", dropout_p: float = 0.20):
        super().__init__()
        self.modo_modulacion = modo_modulacion
        self.dropout_p = dropout_p

        # Rama 1: Codificador de Contexto Biofísico Zonal y de Cultivo
        self.rama_ctx = nn.Sequential(
            nn.Linear(dim_ctx, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Rama 2: Codificador de Dinámica Climática y ENSO
        self.rama_clima = nn.Sequential(
            nn.Linear(dim_clima, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
        )

        # Proyecciones de modulación desde el contexto hacia la señal climática
        self.proj_gamma = nn.Linear(hidden_dim, hidden_dim)
        if modo_modulacion == "FiLM":
            self.proj_beta = nn.Linear(hidden_dim, hidden_dim)

        # Cabeza regresora final con salida Softplus acotada inferiormente en 0
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),
        )

    def forward(self, x_ctx: torch.Tensor, x_clima: torch.Tensor) -> torch.Tensor:
        h_ctx = self.rama_ctx(x_ctx)
        h_clima = self.rama_clima(x_clima)

        if self.modo_modulacion == "Sigmoide":
            # Variante A: Compuerta Multiplicativa Sigmoide pura
            gate = torch.sigmoid(self.proj_gamma(h_ctx))
            z = h_clima * gate
        else:
            # Variante B: Modulación Lineal FiLM completa (gamma = 1 + tanh, beta)
            gamma = 1.0 + torch.tanh(self.proj_gamma(h_ctx))
            beta = self.proj_beta(h_ctx)
            z = gamma * h_clima + beta

        out = self.head(z).squeeze(-1)
        return out


def predecir_mc_dropout(model: nn.Module, x_ctx: torch.Tensor, x_clima: torch.Tensor, n_passes: int = 30):
    """
    Ejecuta S pases hacia adelante manteniendo activo el Dropout (Monte Carlo Dropout)
    para obtener la media predictiva y la desviación estándar epistémica.
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    preds = []
    with torch.no_grad():
        for _ in range(n_passes):
            y_hat = model(x_ctx, x_clima).cpu().numpy()
            preds.append(y_hat)

    preds_arr = np.stack(preds, axis=0)  # (S, N)
    mu = preds_arr.mean(axis=0)
    sigma_mc = preds_arr.std(axis=0)
    return mu, sigma_mc


def calcular_metricas(y_true, y_pred, y_true_raw=None, sigma_mc=None, sigma_total=None):
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

    if sigma_mc is not None:
        res["sigma_mc_media"] = round(float(np.mean(sigma_mc)), 4)

    if sigma_total is not None:
        lower = y_pred - 1.96 * sigma_total
        upper = y_pred + 1.96 * sigma_total
        picp_95 = float(np.mean((y_true >= lower) & (y_true <= upper)) * 100.0)
        res["PICP_95_pct"] = round(picp_95, 2)

    return res


def entrenar_y_evaluar_region(
    nombre_region: str,
    csv_path: str,
    out_res_csv: str,
    out_preds_csv: str,
    curvas_acum: list,
    epochs: int = 45,
    batch_size: int = 256,
):
    print("\n" + "=" * 95)
    print(f"ENTRENAMIENTO PROTOTIPO RED MODULAR FASE 5 - REGIÓN: {nombre_region.upper()} (EXCLUSIVO)")
    print("=" * 95)

    df = pd.read_csv(csv_path, dtype={"cod_dane_mun": str, "cod_dane_dep": str})
    print(f"Dataset {nombre_region}: {len(df):,} filas x {df.shape[1]} columnas | {df['cod_dane_mun'].nunique()} municipios")

    cols_rama1, rama2_sin_enso, rama2_con_enso = obtener_bloques_features()
    df_dummies = pd.get_dummies(df[["cultivo_cat", "departamento"]], drop_first=True, dtype=float)
    X_ctx_df = pd.concat([df[cols_rama1], df_dummies], axis=1)

    y_all = df["eficiencia_clip"].values.astype(np.float32)
    y_raw_all = df["eficiencia"].values.astype(np.float32)
    anios = df["año"].values

    splits = {
        "Split_1_LOEO_2015_2016": {
            "test_mask": df["año"].isin([2015, 2016]).values,
            "train_mask": (~df["año"].isin([2015, 2016])).values,
            "val_year": 2014,
            "descripcion": "El Niño Severo 2015-2016 (LOEO Canónico)",
        },
        "Split_2_LOEO_2023_2024": {
            "test_mask": df["año"].isin([2023, 2024]).values,
            "train_mask": (df["año"] <= 2022).values,
            "val_year": 2022,
            "descripcion": "El Niño Reciente 2023-2024 (LOEO Reciente)",
        },
        "Split_3_Secuencial_2021_2024": {
            "test_mask": (df["año"] >= 2021).values,
            "train_mask": (df["año"] <= 2020).values,
            "val_year": 2020,
            "descripcion": "Cuatrienio Reciente 2021-2024 (Secuencial Puro)",
        },
    }

    resultados = []
    predicciones_rows = []

    for split_nombre, s_info in splits.items():
        print(f"\n>>> [{nombre_region}] {split_nombre} ({s_info['descripcion']})")
        test_mask = s_info["test_mask"]
        train_full_mask = s_info["train_mask"]
        val_mask = train_full_mask & (anios == s_info["val_year"])
        subtrain_mask = train_full_mask & (~val_mask)

        for enso_cfg, cols_r2 in [("Sin_ENSO", rama2_sin_enso), ("Con_ENSO", rama2_con_enso)]:
            X_clim_df = df[cols_r2]

            # Escalado sin fuga ajustado solo en train_full_mask
            sc_ctx = StandardScaler()
            sc_clim = StandardScaler()

            X_ctx_tr_full = sc_ctx.fit_transform(X_ctx_df.iloc[train_full_mask].values).astype(np.float32)
            X_ctx_te = sc_ctx.transform(X_ctx_df.iloc[test_mask].values).astype(np.float32)

            X_clim_tr_full = sc_clim.fit_transform(X_clim_df.iloc[train_full_mask].values).astype(np.float32)
            X_clim_te = sc_clim.transform(X_clim_df.iloc[test_mask].values).astype(np.float32)

            # Sub-partición temporal interna para calibración de incertidumbre residual
            X_ctx_sub = sc_ctx.transform(X_ctx_df.iloc[subtrain_mask].values).astype(np.float32)
            X_clim_sub = sc_clim.transform(X_clim_df.iloc[subtrain_mask].values).astype(np.float32)
            y_sub = y_all[subtrain_mask]

            X_ctx_val = sc_ctx.transform(X_ctx_df.iloc[val_mask].values).astype(np.float32)
            X_clim_val = sc_clim.transform(X_clim_df.iloc[val_mask].values).astype(np.float32)
            y_val = y_all[val_mask]

            y_tr_full = y_all[train_full_mask]
            y_te = y_all[test_mask]
            y_te_raw = y_raw_all[test_mask]

            for variante in ["Sigmoide", "FiLM"]:
                fijar_semilla(42)
                t0 = time.time()

                model = RedModularAgroENSO(
                    dim_ctx=X_ctx_tr_full.shape[1],
                    dim_clima=X_clim_tr_full.shape[1],
                    hidden_dim=64,
                    modo_modulacion=variante,
                    dropout_p=0.20,
                )

                criterion = nn.HuberLoss(delta=0.25)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=2e-3)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

                ds_train = TensorDataset(
                    torch.from_numpy(X_ctx_tr_full),
                    torch.from_numpy(X_clim_tr_full),
                    torch.from_numpy(y_tr_full),
                )
                loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)

                t_ctx_te = torch.from_numpy(X_ctx_te)
                t_clim_te = torch.from_numpy(X_clim_te)
                t_ctx_tr = torch.from_numpy(X_ctx_tr_full)
                t_clim_tr = torch.from_numpy(X_clim_tr_full)
                t_ctx_val = torch.from_numpy(X_ctx_val)
                t_clim_val = torch.from_numpy(X_clim_val)

                for ep in range(1, epochs + 1):
                    model.train()
                    for b_ctx, b_clim, b_y in loader:
                        optimizer.zero_grad()
                        pred = model(b_ctx, b_clim)
                        loss = criterion(pred, b_y)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                        optimizer.step()
                    scheduler.step()

                    if split_nombre == "Split_1_LOEO_2015_2016" and enso_cfg == "Con_ENSO" and (ep == 1 or ep % 5 == 0):
                        model.eval()
                        with torch.no_grad():
                            p_tr_ep = model(t_ctx_tr, t_clim_tr).numpy()
                            p_te_ep = model(t_ctx_te, t_clim_te).numpy()
                        rmse_tr_ep = float(np.sqrt(np.mean((p_tr_ep - y_tr_full) ** 2)))
                        rmse_te_ep = float(np.sqrt(np.mean((p_te_ep - y_te) ** 2)))
                        curvas_acum.append({
                            "region": nombre_region,
                            "variante": f"RedModular_{variante}",
                            "epoca": ep,
                            "train_rmse": round(rmse_tr_ep, 5),
                            "test_rmse": round(rmse_te_ep, 5),
                        })

                dt = time.time() - t0

                # Estimación de varianza residual en validación interna para intervalo al 95% (Gal & Ghahramani 2016)
                model.eval()
                with torch.no_grad():
                    y_pred_tr = model(t_ctx_tr, t_clim_tr).numpy()
                    y_pred_val = model(t_ctx_val, t_clim_val).numpy()
                sigma_aleat = float(np.std(y_val - y_pred_val))

                # Inferencia Monte Carlo Dropout (S = 30 pases) y medición de TPS
                t_inf0 = time.perf_counter()
                y_pred_mc, sigma_mc = predecir_mc_dropout(model, t_ctx_te, t_clim_te, n_passes=30)
                t_inf = time.perf_counter() - t_inf0
                tps_mc30 = round(len(y_te) / max(t_inf, 1e-6), 1)

                sigma_total = np.sqrt(sigma_mc ** 2 + sigma_aleat ** 2)

                m_test = calcular_metricas(y_te, y_pred_mc, y_te_raw, sigma_mc=sigma_mc, sigma_total=sigma_total)
                m_train = calcular_metricas(y_tr_full, y_pred_tr)

                modelo_etiqueta = f"RedModular_{variante}"
                resultados.append({
                    "region": nombre_region,
                    "split": split_nombre,
                    "split_descripcion": s_info["descripcion"],
                    "configuracion_enso": enso_cfg,
                    "modelo": modelo_etiqueta,
                    "variante_modulacion": variante,
                    "N_train": int(len(y_tr_full)),
                    "N_test": m_test["N"],
                    "tiempo_seg": round(dt, 2),
                    "tps_mc30": tps_mc30,
                    "RMSE_test": m_test["RMSE"],
                    "MAE_test": m_test["MAE"],
                    "R2_test": m_test["R2"],
                    "Pearson_r_test": m_test["Pearson_r"],
                    "Bias_test": m_test["Bias"],
                    "RMSE_test_raw": m_test["RMSE_raw"],
                    "MAE_test_raw": m_test["MAE_raw"],
                    "sigma_mc_media": m_test["sigma_mc_media"],
                    "PICP_95_pct": m_test["PICP_95_pct"],
                    "RMSE_train": m_train["RMSE"],
                    "MAE_train": m_train["MAE"],
                    "R2_train": m_train["R2"],
                })

                print(
                    f"      [{modelo_etiqueta:<19} | {enso_cfg:<8}] RMSE={m_test['RMSE']:.4f} | MAE={m_test['MAE']:.4f} | "
                    f"R2={m_test['R2']:.4f} | r={m_test['Pearson_r']:.4f} | PICP95={m_test['PICP_95_pct']:.1f}% ({dt:.1f}s)"
                )

                if enso_cfg == "Con_ENSO":
                    df_te = df.iloc[test_mask].copy()
                    for idx_r, (_, r_te) in enumerate(df_te.iterrows()):
                        predicciones_rows.append({
                            "region": nombre_region,
                            "split": split_nombre,
                            "modelo": modelo_etiqueta,
                            "cod_dane_mun": r_te["cod_dane_mun"],
                            "departamento": r_te["departamento"],
                            "municipio": r_te["municipio"],
                            "cultivo_cat": r_te["cultivo_cat"],
                            "año": int(r_te["año"]),
                            "y_real_clip": round(float(y_te[idx_r]), 4),
                            "y_real_raw": round(float(y_te_raw[idx_r]), 4),
                            "y_pred_mc": round(float(y_pred_mc[idx_r]), 4),
                            "sigma_mc": round(float(sigma_mc[idx_r]), 4),
                            "ic95_inf": round(float(y_pred_mc[idx_r] - 1.96 * sigma_total[idx_r]), 4),
                            "ic95_sup": round(float(y_pred_mc[idx_r] + 1.96 * sigma_total[idx_r]), 4),
                            "residual": round(float(y_pred_mc[idx_r] - y_te[idx_r]), 4),
                            "error_abs": round(float(abs(y_pred_mc[idx_r] - y_te[idx_r])), 4),
                        })

    df_res = pd.DataFrame(resultados)
    df_res.to_csv(out_res_csv, index=False, encoding="utf-8")
    df_preds = pd.DataFrame(predicciones_rows)
    df_preds.to_csv(out_preds_csv, index=False, encoding="utf-8")
    return df_res


def main():
    curvas_acum = []

    # 1. Prototipo inicial exclusivo con datos de la Región Andina
    df_res_andina = entrenar_y_evaluar_region(
        nombre_region="Andina",
        csv_path=resolve_path("dataset_maestro_entrenamiento.csv"),
        out_res_csv=resolve_path("red_modular_resultados_andina.csv"),
        out_preds_csv=resolve_path("red_modular_predicciones_andina.csv"),
        curvas_acum=curvas_acum,
    )

    # 2. Entrenamiento exclusivo con datos de la Región Caribe
    df_res_caribe = entrenar_y_evaluar_region(
        nombre_region="Caribe",
        csv_path=resolve_path("dataset_maestro_entrenamiento_caribe.csv"),
        out_res_csv=resolve_path("red_modular_resultados_caribe.csv"),
        out_preds_csv=resolve_path("red_modular_predicciones_caribe.csv"),
        curvas_acum=curvas_acum,
    )

    df_curvas = pd.DataFrame(curvas_acum)
    df_curvas.to_csv(resolve_path("red_modular_curvas_entrenamiento.csv"), index=False, encoding="utf-8")

    # 3. Resumen comparativo LOEO (El Niño 2015-2016 + El Niño 2023-2024) entre Andina y Caribe
    df_ambos = pd.concat([df_res_andina, df_res_caribe], ignore_index=True)
    df_loeo = df_ambos[df_ambos["split"].isin(["Split_1_LOEO_2015_2016", "Split_2_LOEO_2023_2024"])]
    resumen_loeo = (
        df_loeo.groupby(["region", "modelo", "configuracion_enso"])[
            ["RMSE_test", "MAE_test", "R2_test", "Pearson_r_test", "RMSE_test_raw", "MAE_test_raw", "sigma_mc_media", "PICP_95_pct"]
        ]
        .mean()
        .round(4)
        .reset_index()
    )
    resumen_loeo.to_csv(resolve_path("red_modular_comparativa_loeo.csv"), index=False, encoding="utf-8")

    print("\n" + "=" * 95)
    print("RESUMEN COMPARATIVO PROMEDIO LOEO (2015-2016 + 2023-2024): ANDINA EXCLUSIVO VS CARIBE EXCLUSIVO")
    print("=" * 95)
    print(resumen_loeo.to_string(index=False))


if __name__ == "__main__":
    main()
