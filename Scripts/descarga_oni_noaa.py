"""
Fase 2.1 - Descarga y Procesamiento del Índice Oceánico El Niño (ONI - NOAA CPC)
================================================================================
Descarga la serie histórica oficial del Climate Prediction Center (CPC - NOAA),
clasifica episodios oficiales de El Niño / La Niña (criterio NOAA >= 5 trimestres
móviles consecutivos con |ONI| >= 0.5) y construye el dataset de variables
macroclimáticas anuales, semestrales, fenológicas y rezagadas (1950-2025).

Salidas:
  1. oni_mensual_noaa.csv   : Serie mensual detallada (1950-presente).
  2. oni_anual_features.csv : Variables agregadas por año agrícola (incluye 2007-2024).
"""

import requests
import pandas as pd
import numpy as np

NOAA_ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"

SEAS_TO_MONTH = {
    "DJF": 1,
    "JFM": 2,
    "FMA": 3,
    "MAM": 4,
    "AMJ": 5,
    "MJJ": 6,
    "JJA": 7,
    "JAS": 8,
    "ASO": 9,
    "SON": 10,
    "OND": 11,
    "NDJ": 12,
}


def descargar_oni_mensual(url: str = NOAA_ONI_URL) -> pd.DataFrame:
    print(f"[1/3] Descargando serie oficial ONI desde NOAA CPC: {url}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    rows = []
    for line in resp.text.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0] in SEAS_TO_MONTH:
            seas, yr, total, anom = parts
            rows.append({
                "año": int(yr),
                "mes": SEAS_TO_MONTH[seas],
                "trimestre_movil": seas,
                "sst_nino34_total": float(total),
                "oni": float(anom),
            })

    df = pd.DataFrame(rows).sort_values(["año", "mes"]).reset_index(drop=True)

    # Identificación de episodios oficiales NOAA (>= 5 trimestres móviles consecutivos)
    cond_nino = df["oni"] >= 0.5
    cond_nina = df["oni"] <= -0.5

    # Grupos consecutivos para El Niño
    grp_nino = (~cond_nino).cumsum()
    run_len_nino = cond_nino.groupby(grp_nino).transform("sum")
    df["episodio_nino_oficial"] = (cond_nino & (run_len_nino >= 5)).astype(int)

    # Grupos consecutivos para La Niña
    grp_nina = (~cond_nina).cumsum()
    run_len_nina = cond_nina.groupby(grp_nina).transform("sum")
    df["episodio_nina_oficial"] = (cond_nina & (run_len_nina >= 5)).astype(int)

    # Duración acumulada en meses dentro del episodio actual (cálido o frío)
    duracion_actual = []
    cnt = 0
    estado_prev = 0
    for _, row in df.iterrows():
        estado = 1 if row["oni"] >= 0.5 else (-1 if row["oni"] <= -0.5 else 0)
        if estado != 0 and estado == estado_prev:
            cnt += 1
        elif estado != 0:
            cnt = 1
        else:
            cnt = 0
        estado_prev = estado
        duracion_actual.append(cnt)
    df["duracion_racha_meses"] = duracion_actual

    # Fase mensual simple
    df["fase_enso_mes"] = np.select(
        [df["oni"] >= 0.5, df["oni"] <= -0.5],
        ["El Niño", "La Niña"],
        default="Neutro",
    )

    return df


def construir_features_anuales(df_mes: pd.DataFrame) -> pd.DataFrame:
    print("[2/3] Calculando variables anuales, fenológicas y de rezago (1950-2025)...")
    # Filtramos años completos (con los 12 meses disponibles)
    conteo_meses = df_mes.groupby("año")["mes"].count()
    anios_completos = conteo_meses[conteo_meses == 12].index
    df_full = df_mes[df_mes["año"].isin(anios_completos)].copy()

    registros = []
    for anio, g in df_full.groupby("año"):
        g = g.sort_values("mes")
        oni_vals = g["oni"].values

        # Métricas anuales contemporáneas
        oni_media = float(np.mean(oni_vals))
        oni_max = float(np.max(oni_vals))
        oni_min = float(np.min(oni_vals))
        oni_std = float(np.std(oni_vals, ddof=0))

        # Intensidad acumulada (grado-meses de anomalía)
        oni_acum_pos = float(np.sum(np.maximum(oni_vals, 0.0)))
        oni_acum_neg = float(np.sum(np.maximum(-oni_vals, 0.0)))

        # Conteo de meses por umbral
        meses_nino = int(np.sum(oni_vals >= 0.5))
        meses_nina = int(np.sum(oni_vals <= -0.5))
        meses_nino_mod_fuerte = int(np.sum(oni_vals >= 1.0))
        meses_nino_fuerte = int(np.sum(oni_vals >= 1.5))
        meses_nina_fuerte = int(np.sum(oni_vals <= -1.5))
        max_racha_meses = int(g["duracion_racha_meses"].max())

        # Ventanas semestrales y trimestrales (etapas fenológicas andinas)
        oni_sem1 = float(g[g["mes"].isin([1, 2, 3, 4, 5, 6])]["oni"].mean())
        oni_sem2 = float(g[g["mes"].isin([7, 8, 9, 10, 11, 12])]["oni"].mean())
        oni_q1_jfm = float(g[g["mes"].isin([1, 2, 3])]["oni"].mean())
        oni_q2_amj = float(g[g["mes"].isin([4, 5, 6])]["oni"].mean())
        oni_q3_jas = float(g[g["mes"].isin([7, 8, 9])]["oni"].mean())
        oni_q4_ond = float(g[g["mes"].isin([10, 11, 12])]["oni"].mean())

        # Etiquetas para validación Leave-One-Event-Out (LOEO)
        if meses_nino >= 5 and oni_max >= 1.5:
            evento_anual = "El Niño Fuerte"
            fase_anual = "El Niño"
        elif meses_nino >= 4:
            evento_anual = "El Niño Moderado/Débil"
            fase_anual = "El Niño"
        elif meses_nina >= 5 and oni_min <= -1.5:
            evento_anual = "La Niña Fuerte"
            fase_anual = "La Niña"
        elif meses_nina >= 4:
            evento_anual = "La Niña Moderada/Débil"
            fase_anual = "La Niña"
        else:
            evento_anual = "Neutro"
            fase_anual = "Neutro"

        registros.append({
            "año": int(anio),
            "oni_media_anual": round(oni_media, 4),
            "oni_max_anual": round(oni_max, 4),
            "oni_min_anual": round(oni_min, 4),
            "oni_std_anual": round(oni_std, 4),
            "oni_acum_positivo": round(oni_acum_pos, 4),
            "oni_acum_negativo": round(oni_acum_neg, 4),
            "meses_nino_anual": meses_nino,
            "meses_nina_anual": meses_nina,
            "meses_nino_mod_fuerte": meses_nino_mod_fuerte,
            "meses_nino_fuerte": meses_nino_fuerte,
            "meses_nina_fuerte": meses_nina_fuerte,
            "max_racha_enso_meses": max_racha_meses,
            "oni_semestre1_ene_jun": round(oni_sem1, 4),
            "oni_semestre2_jul_dic": round(oni_sem2, 4),
            "oni_q1_ene_mar": round(oni_q1_jfm, 4),
            "oni_q2_abr_jun": round(oni_q2_amj, 4),
            "oni_q3_jul_sep": round(oni_q3_jas, 4),
            "oni_q4_oct_dic": round(oni_q4_ond, 4),
            "fase_enso_anual": fase_anual,
            "categoria_evento_anual": evento_anual,
        })

    df_anual = pd.DataFrame(registros).sort_values("año").reset_index(drop=True)

    # Añadir variables de rezago (t-1), esenciales para floración y desarrollo de café y cacao
    df_anual["oni_media_lag1"] = df_anual["oni_media_anual"].shift(1).round(4)
    df_anual["oni_max_lag1"] = df_anual["oni_max_anual"].shift(1).round(4)
    df_anual["oni_min_lag1"] = df_anual["oni_min_anual"].shift(1).round(4)
    df_anual["oni_acum_pos_lag1"] = df_anual["oni_acum_positivo"].shift(1).round(4)
    df_anual["oni_sem2_lag1_jul_dic"] = df_anual["oni_semestre2_jul_dic"].shift(1).round(4)
    df_anual["meses_nino_lag1"] = df_anual["meses_nino_anual"].shift(1)

    return df_anual


def main():
    df_mes = descargar_oni_mensual()
    df_mes.to_csv("oni_mensual_noaa.csv", index=False, encoding="utf-8")
    print(f"    -> Guardado 'oni_mensual_noaa.csv' ({len(df_mes)} registros mensuales, {df_mes['año'].min()}-{df_mes['año'].max()})")

    df_anual = construir_features_anuales(df_mes)
    df_anual.to_csv("oni_anual_features.csv", index=False, encoding="utf-8")
    print(f"    -> Guardado 'oni_anual_features.csv' ({len(df_anual)} años completos)")

    print("\n[3/3] Resumen ventana del proyecto (2007-2024):")
    cols_vista = [
        "año", "oni_media_anual", "oni_max_anual", "oni_min_anual",
        "oni_acum_positivo", "meses_nino_anual", "meses_nina_anual", "categoria_evento_anual"
    ]
    df_ventana = df_anual[(df_anual["año"] >= 2007) & (df_anual["año"] <= 2024)][cols_vista]
    print(df_ventana.to_string(index=False))


if __name__ == "__main__":
    main()
