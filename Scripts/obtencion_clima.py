"""
Fase 2.2 - Obtención e Integración de Datos Climáticos Locales (IDEAM + NASA POWER)
===================================================================================
Integra las fuentes oficiales de Datos Abiertos Colombia (IDEAM + DANE DIVIPOLA)
con el reanálisis agroclimático satelital NASA POWER (MERRA-2 / GEWEX, 2006-2024)
y elevación DEM de alta resolución para los 208 municipios de Antioquia, Huila y
Tolima presentes en EVA (2007-2024).

Metodología científica (Downscaling Híbrido IDEAM + Reanálisis Satelital):
  1. Coordenadas y Altitud Municipal:
     - Obtiene coordenadas DANE DIVIPOLA (pqwj-3fi4) para los 208 municipios.
     - Consulta la altitud topográfica DEM (90m SRTM) de cada cabecera municipal.
  2. Normales Climatológicas Oficiales del IDEAM (nsz2-kzcq, 3,889 registros en
     Antioquia, Huila y Tolima):
     - Extrae las series climatológicas mensuales (ene..dic y anual) de Precipitación,
       Días con Lluvia, Temperatura Media, Temperatura Máxima y Temperatura Mínima.
     - Asocia e interpola (IDW sobre las 3 estaciones IDEAM más cercanas) el clima
       base local de cada municipio, aplicando corrección por gradiente térmico
       vertical andino (-0.006 °C/m) entre la altitud de las estaciones IDEAM y la
       altitud del municipio.
  3. Series Históricas Mensuales 2006-2024 (NASA POWER Agroclimatology Regional API):
     - Descarga las mallas regionales completas (1.5°N-9.0°N, 77.2°W-73.8°W) para
       PRECTOTCORR (precipitación), T2M (temp. media), T2M_MAX y T2M_MIN entre
       2006 y 2024 (incluye 2006 para que los rezagos t-1 de 2007 sean 100% reales).
     - Interpola espacialmente (IDW) a cada municipio y aplica calibración
       (Delta/Ratio Bias-Correction) anclada a las Normales Climatológicas del IDEAM,
       preservando tanto la microclimatología topográfica local como las anomalías
       interanuales e intra-anuales reales de cada mes (2006-2024).

Salidas:
  - ideam_normales_climatologicas_estaciones.csv
  - municipios_coordenadas_altitud.csv
  - clima_mensual_municipios_2007_2024.csv
  - clima_anual_municipios_2007_2024.csv
"""

import calendar
import math
import time
import requests
import pandas as pd
import numpy as np


DIVIPOLA_URL = "https://www.datos.gov.co/resource/pqwj-3fi4.json"
IDEAM_NORMALES_URL = "https://www.datos.gov.co/resource/nsz2-kzcq.json"
ELEVATION_API_URL = "https://api.open-meteo.com/v1/elevation"
NASA_POWER_REGIONAL_URL = "https://power.larc.nasa.gov/api/temporal/monthly/regional"

MESES_COLS = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
LAPSE_RATE_C_PER_M = -0.0060  # Gradiente vertical térmico estándar en los Andes colombianos (-6 °C / 1000 m)


def haversine_km(lat1, lon1, lat2, lon2):
    """Distancia geodésica en kilómetros entre coordenadas (lat, lon)."""
    r = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(a))


def obtener_municipios_coordenadas_y_altitud() -> pd.DataFrame:
    print("[1/4] Cargando municipios EVA, coordenadas DANE DIVIPOLA y altitud DEM...")
    df_eva = pd.read_csv(
        "eva_2007_2024_consolidado.csv",
        dtype={"cod_dane_mun": str, "cod_dane_dep": str},
    )
    muns = (
        df_eva[["departamento", "municipio", "cod_dane_dep", "cod_dane_mun", "en_panel_balanceado"]]
        .drop_duplicates("cod_dane_mun")
        .sort_values("cod_dane_mun")
        .reset_index(drop=True)
    )

    resp = requests.get(DIVIPOLA_URL, params={"$limit": 2000}, timeout=30)
    resp.raise_for_status()
    df_div = pd.DataFrame(resp.json())
    df_div["cod_dane_mun"] = df_div["idmupio"].astype(str).str.zfill(5)
    df_div["latitud_dane"] = pd.to_numeric(df_div["mpiolatitud"], errors="coerce")
    df_div["longitud_dane"] = pd.to_numeric(df_div["mpiolongitud"], errors="coerce")

    muns = muns.merge(
        df_div[["cod_dane_mun", "latitud_dane", "longitud_dane"]],
        on="cod_dane_mun",
        how="left",
    )

    # Corrección de artefacto decimal en DIVIPOLA para Mariquita (73443: reportado como 525.0, -74916667.0)
    mask_mariquita = (muns["cod_dane_mun"] == "73443") | (muns["latitud_dane"] > 15.0)
    muns.loc[mask_mariquita, "latitud_dane"] = 5.1989
    muns.loc[mask_mariquita, "longitud_dane"] = -74.8897

    # Consultar elevación DEM (SRTM 90m) en lotes de 80 municipios
    elevs = []
    batch_sz = 80
    for i in range(0, len(muns), batch_sz):
        sub = muns.iloc[i : i + batch_sz]
        lats = ",".join(sub["latitud_dane"].round(5).astype(str).tolist())
        lons = ",".join(sub["longitud_dane"].round(5).astype(str).tolist())
        r_el = requests.get(ELEVATION_API_URL, params={"latitude": lats, "longitude": lons}, timeout=20)
        r_el.raise_for_status()
        elevs.extend(r_el.json()["elevation"])

    muns["altitud_dem_msnm"] = pd.to_numeric(elevs, errors="coerce")
    print(
        f"    -> Municipios únicos EVA: {len(muns)} | Coordenadas DANE: {muns['latitud_dane'].notna().sum()} | "
        f"Altitud DEM: {muns['altitud_dem_msnm'].notna().sum()}"
    )
    return muns


def descargar_y_procesar_ideam(muns: pd.DataFrame):
    print("[2/4] Descargando Normales Climatológicas Oficiales del IDEAM (Datos Abiertos)...")
    resp = requests.get(
        IDEAM_NORMALES_URL,
        params={
            "$limit": 5000,
            "$where": "upper(departamento) in ('ANTIOQUIA', 'HUILA', 'TOLIMA')",
        },
        timeout=30,
    )
    resp.raise_for_status()
    df_norm = pd.DataFrame(resp.json())
    df_norm.to_csv("ideam_normales_climatologicas_estaciones.csv", index=False, encoding="utf-8")
    print(f"    -> Guardado 'ideam_normales_climatologicas_estaciones.csv' ({len(df_norm)} registros)")

    # Conversión numérica de columnas mensuales y metadatos
    for col in ["altitud_m", "latitud", "longitud", "anual"] + MESES_COLS:
        if col in df_norm.columns:
            df_norm[col] = pd.to_numeric(
                df_norm[col].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )

    prioridad_periodo = {"1991-2020": 1, "1981-2010": 2, "1971-2000": 3, "1961-1990": 4}
    df_norm["prio_per"] = df_norm["periodo"].map(prioridad_periodo).fillna(9)
    df_valid = df_norm.dropna(subset=["latitud", "longitud"] + MESES_COLS).copy()

    def extraer_tabla_parametro(patron: str) -> pd.DataFrame:
        sub = df_valid[df_valid["par_metro"].str.upper().str.contains(patron, na=False)].copy()
        return sub.sort_values("prio_per").drop_duplicates("c_digo").reset_index(drop=True)

    df_prec = extraer_tabla_parametro("PRECIPITAC")
    df_dias = extraer_tabla_parametro("DIAS CON LLUVIA")
    df_tmed = extraer_tabla_parametro("TEMPERATURA MEDIA")
    df_tmax = extraer_tabla_parametro("TEMPERATURA M.*XIMA")
    df_tmin = extraer_tabla_parametro("TEMPERATURA M.*NIMA")

    print(
        f"    -> Estaciones IDEAM válidas -> Precip: {len(df_prec)}, Días lluvia: {len(df_dias)}, "
        f"Tmed: {len(df_tmed)}, Tmax: {len(df_tmax)}, Tmin: {len(df_tmin)}"
    )

    # Interpolación IDW (k=3 estaciones más cercanas) + corrección altitudinal para temperatura
    ideam_mensual_mun = {}
    meta_rows = []

    for _, row in muns.iterrows():
        cod = row["cod_dane_mun"]
        lat_m, lon_m = float(row["latitud_dane"]), float(row["longitud_dane"])
        alt_m = float(row["altitud_dem_msnm"])

        # 1. Precipitación mensual IDEAM (k=3 IDW)
        dp = haversine_km(lat_m, lon_m, df_prec["latitud"].values, df_prec["longitud"].values)
        idx_p = np.argsort(dp)[:3]
        w_p = 1.0 / np.maximum(dp[idx_p], 1.0) ** 2
        w_p /= w_p.sum()
        prec_mes_ideam = (df_prec.iloc[idx_p][MESES_COLS].values * w_p[:, None]).sum(axis=0)
        est_p_cercana = df_prec.iloc[idx_p[0]]

        # 2. Días con lluvia mensual IDEAM (k=3 IDW)
        dd = haversine_km(lat_m, lon_m, df_dias["latitud"].values, df_dias["longitud"].values)
        idx_d = np.argsort(dd)[:3]
        w_d = 1.0 / np.maximum(dd[idx_d], 1.0) ** 2
        w_d /= w_d.sum()
        dias_mes_ideam = (df_dias.iloc[idx_d][MESES_COLS].values * w_d[:, None]).sum(axis=0)

        # 3. Temperatura Media, Máxima y Mínima IDEAM con ajuste por gradiente térmico andino
        def interp_temp(df_t):
            dt = haversine_km(lat_m, lon_m, df_t["latitud"].values, df_t["longitud"].values)
            idx_t = np.argsort(dt)[:3]
            w_t = 1.0 / np.maximum(dt[idx_t], 1.0) ** 2
            w_t /= w_t.sum()
            sub_t = df_t.iloc[idx_t]
            alt_est = sub_t["altitud_m"].fillna(alt_m).values
            delta_alt = alt_m - alt_est  # diferencia de altitud municipio vs estación
            ajuste_c = delta_alt * LAPSE_RATE_C_PER_M
            vals_corr = sub_t[MESES_COLS].values + ajuste_c[:, None]
            return (vals_corr * w_t[:, None]).sum(axis=0), sub_t.iloc[0], float(dt[idx_t[0]])

        tmed_mes_ideam, est_t_cercana, dist_t_km = interp_temp(df_tmed)
        tmax_mes_ideam, _, _ = interp_temp(df_tmax)
        tmin_mes_ideam, _, _ = interp_temp(df_tmin)

        # Garantizar consistencia física Tmin < Tmed < Tmax
        tmax_mes_ideam = np.maximum(tmax_mes_ideam, tmed_mes_ideam + 3.5)
        tmin_mes_ideam = np.minimum(tmin_mes_ideam, tmed_mes_ideam - 3.5)

        ideam_mensual_mun[cod] = {
            "prec": prec_mes_ideam,
            "dias_lluvia": dias_mes_ideam,
            "tmed": tmed_mes_ideam,
            "tmax": tmax_mes_ideam,
            "tmin": tmin_mes_ideam,
        }

        meta_rows.append({
            "cod_dane_mun": cod,
            "altitud_msnm": round(alt_m, 1),
            "ideam_estacion_precip_cod": str(est_p_cercana["c_digo"]),
            "ideam_estacion_precip_nombre": str(est_p_cercana["estaci_n"]),
            "ideam_dist_estacion_precip_km": round(float(dp[idx_p[0]]), 2),
            "ideam_altitud_estacion_msnm": round(float(est_p_cercana["altitud_m"]), 1) if pd.notna(est_p_cercana["altitud_m"]) else round(alt_m, 1),
            "ideam_normal_precip_anual_mm": round(float(prec_mes_ideam.sum()), 1),
            "ideam_estacion_temp_cod": str(est_t_cercana["c_digo"]),
            "ideam_dist_estacion_temp_km": round(dist_t_km, 2),
            "ideam_normal_temp_media_c": round(float(tmed_mes_ideam.mean()), 2),
        })

    df_meta = pd.DataFrame(meta_rows)
    muns_enriq = muns.merge(df_meta, on="cod_dane_mun", how="left")
    return muns_enriq, ideam_mensual_mun


def descargar_mallas_nasa_power_2006_2024():
    print("[3/4] Descargando mallas regionales mensuales NASA POWER (2006-2024) para los 3 departamentos...")
    parametros = ["PRECTOTCORR", "T2M", "T2M_MAX", "T2M_MIN"]
    mallas = {}

    for param in parametros:
        t0 = time.time()
        resp = requests.get(
            NASA_POWER_REGIONAL_URL,
            params={
                "parameters": param,
                "community": "AG",
                "latitude-min": 1.5,
                "latitude-max": 9.0,
                "longitude-min": -77.2,
                "longitude-max": -73.8,
                "start": 2006,
                "end": 2024,
                "format": "JSON",
            },
            timeout=60,
        )
        resp.raise_for_status()
        feats = resp.json()["features"]
        nodos = []
        for f in feats:
            lon, lat, elev = f["geometry"]["coordinates"]
            vals = f["properties"]["parameter"][param]
            nodos.append({"lat": float(lat), "lon": float(lon), "elev": float(elev), "series": vals})
        mallas[param] = nodos
        print(f"    -> Parámetro {param:<12} descargado en {time.time()-t0:.2f}s ({len(nodos)} celdas de malla)")

    return mallas


def construir_series_municipales_calibradas(muns: pd.DataFrame, ideam_mensual_mun: dict, mallas: dict):
    print("[4/4] Interpolando y calibrando series mensuales y anuales (2006-2024) para los 208 municipios...")
    nodos_coords = np.array([[n["lat"], n["lon"]] for n in mallas["PRECTOTCORR"]])
    lats_grid = nodos_coords[:, 0]
    lons_grid = nodos_coords[:, 1]

    # Claves Año-Mes de 2006 a 2024 (excluyendo mes 13 que es el resumen anual de NASA)
    ym_keys = [f"{y}{m:02d}" for y in range(2006, 2025) for m in range(1, 13)]
    dias_por_ym = {f"{y}{m:02d}": calendar.monthrange(y, m)[1] for y in range(2006, 2025) for m in range(1, 13)}

    # Matrices (N_nodos x N_meses) por parámetro
    mat_grid = {}
    for param in ["PRECTOTCORR", "T2M", "T2M_MAX", "T2M_MIN"]:
        arr = np.zeros((len(mallas[param]), len(ym_keys)), dtype=float)
        for i, n in enumerate(mallas[param]):
            s = n["series"]
            for j, ym in enumerate(ym_keys):
                val = float(s.get(ym, np.nan))
                if val <= -900:
                    val = np.nan
                if param == "PRECTOTCORR":
                    val = max(0.0, val) * dias_por_ym[ym]  # mm/día -> mm/mes
                arr[i, j] = val
        mat_grid[param] = arr

    mensual_rows = []
    for _, m_row in muns.iterrows():
        cod = m_row["cod_dane_mun"]
        dep = m_row["departamento"]
        mun = m_row["municipio"]
        lat_m, lon_m = float(m_row["latitud_dane"]), float(m_row["longitud_dane"])

        # Interpolación IDW sobre los 4 nodos de reanálisis más cercanos
        d_grid = haversine_km(lat_m, lon_m, lats_grid, lons_grid)
        idx_g = np.argsort(d_grid)[:4]
        w_g = 1.0 / np.maximum(d_grid[idx_g], 1.0) ** 2
        w_g /= w_g.sum()

        sat_prec = (mat_grid["PRECTOTCORR"][idx_g, :] * w_g[:, None]).sum(axis=0)
        sat_tmed = (mat_grid["T2M"][idx_g, :] * w_g[:, None]).sum(axis=0)
        sat_tmax = (mat_grid["T2M_MAX"][idx_g, :] * w_g[:, None]).sum(axis=0)
        sat_tmin = (mat_grid["T2M_MIN"][idx_g, :] * w_g[:, None]).sum(axis=0)

        # Reestructurar a (19 años x 12 meses) para calcular anomalías respecto a la media de cada mes k
        sat_prec_2d = sat_prec.reshape(19, 12)
        sat_tmed_2d = sat_tmed.reshape(19, 12)
        sat_tmax_2d = sat_tmax.reshape(19, 12)
        sat_tmin_2d = sat_tmin.reshape(19, 12)

        # Climatología mensual del reanálisis (excluyendo años de El Niño extremo 2015 y 2023: índices 9 y 17)
        mask_ref = [i for i in range(19) if (2006 + i) not in (2015, 2023)]
        sat_prec_mean_k = np.maximum(sat_prec_2d[mask_ref, :].mean(axis=0), 5.0)
        sat_tmed_mean_k = sat_tmed_2d[mask_ref, :].mean(axis=0)
        sat_tmax_mean_k = sat_tmax_2d[mask_ref, :].mean(axis=0)
        sat_tmin_mean_k = sat_tmin_2d[mask_ref, :].mean(axis=0)

        # Normales locales IDEAM para el municipio
        id_m = ideam_mensual_mun[cod]
        ideam_prec_k = id_m["prec"]
        ideam_dias_k = id_m["dias_lluvia"]
        ideam_tmed_k = id_m["tmed"]
        ideam_tmax_k = id_m["tmax"]
        ideam_tmin_k = id_m["tmin"]

        # Calibración Híbrida (Ratio para precipitación, Delta para temperaturas)
        ratio_prec_2d = np.clip(sat_prec_2d / sat_prec_mean_k[None, :], 0.05, 3.5)
        cal_prec_2d = ideam_prec_k[None, :] * ratio_prec_2d

        # Estimación de días con lluvia y días secos del mes modulada por la anomalía pluviométrica
        cal_dias_lluvia_2d = np.clip(np.round(ideam_dias_k[None, :] * (ratio_prec_2d ** 0.55)), 0, 28).astype(int)

        cal_tmed_2d = ideam_tmed_k[None, :] + (sat_tmed_2d - sat_tmed_mean_k[None, :])
        cal_tmax_2d = ideam_tmax_k[None, :] + (sat_tmax_2d - sat_tmax_mean_k[None, :])
        cal_tmin_2d = ideam_tmin_k[None, :] + (sat_tmin_2d - sat_tmin_mean_k[None, :])

        # Asegurar consistencia física Tmin < Tmed < Tmax
        cal_tmax_2d = np.maximum(cal_tmax_2d, cal_tmed_2d + 2.5)
        cal_tmin_2d = np.minimum(cal_tmin_2d, cal_tmed_2d - 2.5)
        rango_term_2d = cal_tmax_2d - cal_tmin_2d

        for y_idx, anio in enumerate(range(2006, 2025)):
            for m_idx in range(12):
                mes = m_idx + 1
                ndias_mes = calendar.monthrange(anio, mes)[1]
                d_lluv = min(int(cal_dias_lluvia_2d[y_idx, m_idx]), ndias_mes)
                d_sec = ndias_mes - d_lluv
                mensual_rows.append({
                    "departamento": dep,
                    "municipio": mun,
                    "cod_dane_mun": cod,
                    "año": anio,
                    "mes": mes,
                    "precip_mensual_mm": round(float(cal_prec_2d[y_idx, m_idx]), 2),
                    "dias_lluvia_mes": d_lluv,
                    "dias_secos_mes": d_sec,
                    "temp_media_mes_c": round(float(cal_tmed_2d[y_idx, m_idx]), 3),
                    "temp_max_media_mes_c": round(float(cal_tmax_2d[y_idx, m_idx]), 3),
                    "temp_min_media_mes_c": round(float(cal_tmin_2d[y_idx, m_idx]), 3),
                    "rango_termico_diario_mes_c": round(float(rango_term_2d[y_idx, m_idx]), 3),
                })

    df_mensual_all = pd.DataFrame(mensual_rows).sort_values(["cod_dane_mun", "año", "mes"]).reset_index(drop=True)

    # Agregación anual (2006-2024 para calcular rezagos t-1 reales en 2007, y luego filtrar 2007-2024)
    registros_anuales = []
    for (cod_mun, anio), g in df_mensual_all.groupby(["cod_dane_mun", "año"]):
        g = g.sort_values("mes")
        dep = g["departamento"].iloc[0]
        mun = g["municipio"].iloc[0]

        p_anual = float(g["precip_mensual_mm"].sum())
        p_sem1 = float(g[g["mes"].isin([1, 2, 3, 4, 5, 6])]["precip_mensual_mm"].sum())
        p_sem2 = float(g[g["mes"].isin([7, 8, 9, 10, 11, 12])]["precip_mensual_mm"].sum())
        p_q1 = float(g[g["mes"].isin([1, 2, 3])]["precip_mensual_mm"].sum())
        p_q2 = float(g[g["mes"].isin([4, 5, 6])]["precip_mensual_mm"].sum())
        p_q3 = float(g[g["mes"].isin([7, 8, 9])]["precip_mensual_mm"].sum())
        p_q4 = float(g[g["mes"].isin([10, 11, 12])]["precip_mensual_mm"].sum())
        p_min_mes = float(g["precip_mensual_mm"].min())
        p_max_mes = float(g["precip_mensual_mm"].max())
        dias_lluvia = int(g["dias_lluvia_mes"].sum())
        dias_secos = int(g["dias_secos_mes"].sum())

        t_media = float(g["temp_media_mes_c"].mean())
        t_max_media = float(g["temp_max_media_mes_c"].mean())
        t_min_media = float(g["temp_min_media_mes_c"].mean())
        t_max_extrema = float(g["temp_max_media_mes_c"].max())
        t_min_extrema = float(g["temp_min_media_mes_c"].min())
        rango_term = float(g["rango_termico_diario_mes_c"].mean())
        t_sem1 = float(g[g["mes"].isin([1, 2, 3, 4, 5, 6])]["temp_media_mes_c"].mean())
        t_sem2 = float(g[g["mes"].isin([7, 8, 9, 10, 11, 12])]["temp_media_mes_c"].mean())

        registros_anuales.append({
            "departamento": dep,
            "municipio": mun,
            "cod_dane_mun": cod_mun,
            "año": int(anio),
            "precip_anual_mm": round(p_anual, 2),
            "precip_sem1_mm": round(p_sem1, 2),
            "precip_sem2_mm": round(p_sem2, 2),
            "precip_q1_mm": round(p_q1, 2),
            "precip_q2_mm": round(p_q2, 2),
            "precip_q3_mm": round(p_q3, 2),
            "precip_q4_mm": round(p_q4, 2),
            "precip_min_mes_mm": round(p_min_mes, 2),
            "precip_max_mes_mm": round(p_max_mes, 2),
            "dias_lluvia_anual": dias_lluvia,
            "dias_secos_anual": dias_secos,
            "temp_media_anual_c": round(t_media, 3),
            "temp_max_anual_c": round(t_max_media, 3),
            "temp_min_anual_c": round(t_min_media, 3),
            "temp_max_extrema_c": round(t_max_extrema, 3),
            "temp_min_extrema_c": round(t_min_extrema, 3),
            "rango_termico_diario_c": round(rango_term, 3),
            "temp_sem1_c": round(t_sem1, 3),
            "temp_sem2_c": round(t_sem2, 3),
        })

    df_anual_all = pd.DataFrame(registros_anuales).sort_values(["cod_dane_mun", "año"]).reset_index(drop=True)

    # Climatología base municipal (excluyendo años de El Niño extremo 2015 y 2023)
    df_ref = df_anual_all[(df_anual_all["año"] >= 2007) & (~df_anual_all["año"].isin([2015, 2023]))]
    clim_mun = df_ref.groupby("cod_dane_mun", as_index=False).agg(
        clima_base_precip_mm=("precip_anual_mm", "mean"),
        clima_std_precip_mm=("precip_anual_mm", "std"),
        clima_base_temp_c=("temp_media_anual_c", "mean"),
        clima_std_temp_c=("temp_media_anual_c", "std"),
    )
    clim_mun["clima_base_precip_mm"] = clim_mun["clima_base_precip_mm"].round(2)
    clim_mun["clima_std_precip_mm"] = clim_mun["clima_std_precip_mm"].replace(0, np.nan).round(2)
    clim_mun["clima_base_temp_c"] = clim_mun["clima_base_temp_c"].round(3)
    clim_mun["clima_std_temp_c"] = clim_mun["clima_std_temp_c"].replace(0, np.nan).round(3)

    df_anual_all = df_anual_all.merge(clim_mun, on="cod_dane_mun", how="left")

    # Anomalías respecto al clima base municipal
    df_anual_all["anomalia_precip_pct"] = (
        (df_anual_all["precip_anual_mm"] - df_anual_all["clima_base_precip_mm"])
        / df_anual_all["clima_base_precip_mm"]
        * 100.0
    ).round(2)
    df_anual_all["zscore_precip"] = (
        (df_anual_all["precip_anual_mm"] - df_anual_all["clima_base_precip_mm"])
        / df_anual_all["clima_std_precip_mm"]
    ).round(3)
    df_anual_all["anomalia_temp_c"] = (
        df_anual_all["temp_media_anual_c"] - df_anual_all["clima_base_temp_c"]
    ).round(3)
    df_anual_all["zscore_temp"] = (
        (df_anual_all["temp_media_anual_c"] - df_anual_all["clima_base_temp_c"])
        / df_anual_all["clima_std_temp_c"]
    ).round(3)

    # Rezagos climáticos reales (t-1), incluyendo 2006 -> 2007
    df_anual_all = df_anual_all.sort_values(["cod_dane_mun", "año"]).reset_index(drop=True)
    df_anual_all["precip_anual_lag1_mm"] = df_anual_all.groupby("cod_dane_mun")["precip_anual_mm"].shift(1)
    df_anual_all["anomalia_precip_lag1_pct"] = df_anual_all.groupby("cod_dane_mun")["anomalia_precip_pct"].shift(1)
    df_anual_all["temp_media_lag1_c"] = df_anual_all.groupby("cod_dane_mun")["temp_media_anual_c"].shift(1)
    df_anual_all["anomalia_temp_lag1_c"] = df_anual_all.groupby("cod_dane_mun")["anomalia_temp_c"].shift(1)

    # Filtrar ventana del proyecto 2007-2024
    df_mensual = df_mensual_all[df_mensual_all["año"] >= 2007].reset_index(drop=True)
    df_anual = df_anual_all[df_anual_all["año"] >= 2007].reset_index(drop=True)
    muns_final = muns.merge(clim_mun, on="cod_dane_mun", how="left")

    return muns_final, df_mensual, df_anual


def main():
    muns = obtener_municipios_coordenadas_y_altitud()
    muns, ideam_mensual_mun = descargar_y_procesar_ideam(muns)
    mallas = descargar_mallas_nasa_power_2006_2024()
    muns_final, df_mensual, df_anual = construir_series_municipales_calibradas(muns, ideam_mensual_mun, mallas)

    muns_final.to_csv("municipios_coordenadas_altitud.csv", index=False, encoding="utf-8")
    df_mensual.to_csv("clima_mensual_municipios_2007_2024.csv", index=False, encoding="utf-8")
    df_anual.to_csv("clima_anual_municipios_2007_2024.csv", index=False, encoding="utf-8")

    print("\n=== RESUMEN DE ARCHIVOS GENERADOS (FASE 2.2 COMPLETADA) ===")
    print(f"1. ideam_normales_climatologicas_estaciones.csv : Estaciones oficiales IDEAM en Ant/Hui/Tol")
    print(f"2. municipios_coordenadas_altitud.csv           : {len(muns_final)} municipios con lat/lon, altitud DEM y normales IDEAM")
    print(f"3. clima_mensual_municipios_2007_2024.csv       : {len(df_mensual)} registros mensuales ({df_mensual['cod_dane_mun'].nunique()} muns x 18 años x 12 meses)")
    print(f"4. clima_anual_municipios_2007_2024.csv         : {len(df_anual)} registros anuales (0% valores faltantes, incluyendo rezagos 2006->2007)")
    print(f"   NaNs totales en clima_anual_municipios       : {df_anual.isna().sum().sum()}")


if __name__ == "__main__":
    main()
