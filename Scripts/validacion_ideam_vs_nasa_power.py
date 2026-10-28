"""
Auditoría Punto 1 - Validación Empírica: NASA POWER (Crudo vs. Calibrado) contra Observaciones IDEAM
====================================================================================================
Ejecuta pruebas cuantitativas reproducibles que:
  1. Evidencian por qué las tablas crudas sub-horarias de Socrata (sbwg-7ju4, s54a-sgyg)
     no cubren los 208 municipios x 216 meses de forma directa (cobertura espacial parcial
     de estaciones automáticas y ausencia de índice para GROUP BY masivo).
  2. Evalúan el error (RMSE, MAE, Sesgo/Bias, R² y r de Pearson) comparando:
       a) NASA POWER Crudo (sin calibrar)
       b) NASA POWER Calibrado (nuestro pipeline híbrido IDEAM + DEM + NASA POWER)
     frente a las observaciones reales del IDEAM en:
       - Validación 1: Series interanuales/mensuales observadas en estaciones automáticas
         IDEAM (sbwg-7ju4, 2007-2024) en Antioquia, Huila y Tolima.
       - Validación 2: Series mensuales y anuales completas de estaciones físicas IDEAM
         ubicadas en los municipios EVA (nsz2-kzcq, Precipitación en mm y Temperatura en °C),
         incluyendo validación cruzada espacial estricta Leave-One-Station-Out (LOO-CV).
"""

import calendar
import unicodedata
import requests
import pandas as pd
import numpy as np


IDEAM_OBS_TEMP_URL = "https://www.datos.gov.co/resource/sbwg-7ju4.json"
NASA_POWER_REGIONAL_URL = "https://power.larc.nasa.gov/api/temporal/monthly/regional"
MESES_COLS = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
LAPSE_RATE_C_PER_M = -0.0060


def normalizar_texto(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s.upper().strip())
    return "".join(c for c in s if not unicodedata.combining(c))


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(a))


def metricas(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    err = yp - yt
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    r_pearson = float(np.corrcoef(yt, yp)[0, 1]) if len(yt) > 2 else np.nan
    return {"N": len(yt), "RMSE": rmse, "MAE": mae, "Bias": bias, "R2": r2, "Pearson_r": r_pearson}


def reconstruir_nasa_crudo_municipios(muns: pd.DataFrame) -> pd.DataFrame:
    """Reconstruye la serie mensual y anual 2007-2024 de NASA POWER CRUDO (sin calibrar con IDEAM)."""
    mallas = {}
    for param in ["PRECTOTCORR", "T2M"]:
        resp = requests.get(
            NASA_POWER_REGIONAL_URL,
            params={
                "parameters": param,
                "community": "AG",
                "latitude-min": 1.5,
                "latitude-max": 9.0,
                "longitude-min": -77.2,
                "longitude-max": -73.8,
                "start": 2007,
                "end": 2024,
                "format": "JSON",
            },
            timeout=60,
        )
        resp.raise_for_status()
        nodos = []
        for f in resp.json()["features"]:
            lon, lat, elev = f["geometry"]["coordinates"]
            nodos.append({"lat": float(lat), "lon": float(lon), "series": f["properties"]["parameter"][param]})
        mallas[param] = nodos

    lats_grid = np.array([n["lat"] for n in mallas["PRECTOTCORR"]])
    lons_grid = np.array([n["lon"] for n in mallas["PRECTOTCORR"]])
    ym_keys = [f"{y}{m:02d}" for y in range(2007, 2025) for m in range(1, 13)]
    dias_por_ym = {f"{y}{m:02d}": calendar.monthrange(y, m)[1] for y in range(2007, 2025) for m in range(1, 13)}

    mat_prec = np.zeros((len(mallas["PRECTOTCORR"]), len(ym_keys)))
    mat_tmed = np.zeros((len(mallas["T2M"]), len(ym_keys)))
    for i, n in enumerate(mallas["PRECTOTCORR"]):
        for j, ym in enumerate(ym_keys):
            v = float(n["series"].get(ym, np.nan))
            mat_prec[i, j] = max(0.0, v) * dias_por_ym[ym] if v > -900 else np.nan
    for i, n in enumerate(mallas["T2M"]):
        for j, ym in enumerate(ym_keys):
            v = float(n["series"].get(ym, np.nan))
            mat_tmed[i, j] = v if v > -900 else np.nan

    rows = []
    for _, r in muns.iterrows():
        cod = r["cod_dane_mun"]
        d_grid = haversine_km(float(r["latitud_dane"]), float(r["longitud_dane"]), lats_grid, lons_grid)
        idx_g = np.argsort(d_grid)[:4]
        w_g = 1.0 / np.maximum(d_grid[idx_g], 1.0) ** 2
        w_g /= w_g.sum()
        p_raw = (mat_prec[idx_g, :] * w_g[:, None]).sum(axis=0).reshape(18, 12)
        t_raw = (mat_tmed[idx_g, :] * w_g[:, None]).sum(axis=0).reshape(18, 12)
        for y_i, anio in enumerate(range(2007, 2025)):
            for m_i in range(12):
                rows.append({
                    "cod_dane_mun": cod,
                    "año": anio,
                    "mes": m_i + 1,
                    "nasa_crudo_precip_mm": float(p_raw[y_i, m_i]),
                    "nasa_crudo_temp_c": float(t_raw[y_i, m_i]),
                })
    return pd.DataFrame(rows)


def main():
    print("=" * 90)
    print("PUNTO 1: AUDITORÍA METODOLÓGICA Y VALIDACIÓN CUANTITATIVA (IDEAM VS NASA POWER)")
    print("=" * 90)

    muns = pd.read_csv("municipios_coordenadas_altitud.csv", dtype={"cod_dane_mun": str})
    df_cal_mes = pd.read_csv("clima_mensual_municipios_2007_2024.csv", dtype={"cod_dane_mun": str})
    df_cal_anual = pd.read_csv("clima_anual_municipios_2007_2024.csv", dtype={"cod_dane_mun": str})
    df_norm = pd.read_csv("ideam_normales_climatologicas_estaciones.csv")

    print("\n[1/3] Reconstruyendo serie NASA POWER Cruda (sin calibrar) 2007-2024 para comparación directa...")
    df_raw_mes = reconstruir_nasa_crudo_municipios(muns)
    df_comp_mes = df_cal_mes.merge(df_raw_mes, on=["cod_dane_mun", "año", "mes"], how="left")

    # -------------------------------------------------------------------------
    # VALIDACIÓN 1: Observaciones interanuales/mensuales en estaciones IDEAM (sbwg-7ju4, 2007-2024)
    # -------------------------------------------------------------------------
    print("\n[2/3] Validación 1: Comparación contra observaciones históricas de estaciones IDEAM (sbwg-7ju4)...")
    dfs_obs = []
    for depto in ["ANTIOQUIA", "HUILA", "TOLIMA"]:
        r = requests.get(
            IDEAM_OBS_TEMP_URL,
            params={"$where": f"departamento = '{depto}'", "$limit": 35000},
            timeout=30,
        )
        if r.status_code == 200 and r.json():
            dfs_obs.append(pd.DataFrame(r.json()))

    df_obs = pd.concat(dfs_obs, ignore_index=True)
    df_obs["val"] = pd.to_numeric(df_obs["valorobservado"], errors="coerce")
    df_obs = df_obs[(df_obs["val"] >= 2.0) & (df_obs["val"] <= 42.0)].copy()
    df_obs["año"] = pd.to_numeric(df_obs["fechaobservacion"].str.slice(0, 4), errors="coerce")
    df_obs["mes"] = pd.to_numeric(df_obs["fechaobservacion"].str.slice(5, 7), errors="coerce")
    df_obs = df_obs[df_obs["año"].between(2007, 2024)].copy()
    df_obs["lat_est"] = pd.to_numeric(df_obs["latitud"], errors="coerce")
    df_obs["lon_est"] = pd.to_numeric(df_obs["longitud"], errors="coerce")

    # Cruzar municipio por nombre normalizado y departamento
    muns["key_mun"] = muns["departamento"].apply(normalizar_texto) + "|" + muns["municipio"].apply(normalizar_texto)
    df_obs["key_mun"] = df_obs["departamento"].apply(normalizar_texto) + "|" + df_obs["municipio"].apply(normalizar_texto)
    df_obs = df_obs.merge(
        muns[["key_mun", "cod_dane_mun", "latitud_dane", "longitud_dane", "altitud_msnm"]],
        on="key_mun",
        how="inner",
    )

    # Filtrar estaciones cercanas a la cabecera municipal (<= 12 km) para que la comparación sea representativa
    df_obs["dist_cabecera_km"] = haversine_km(
        df_obs["lat_est"].values, df_obs["lon_est"].values,
        df_obs["latitud_dane"].values, df_obs["longitud_dane"].values
    )
    df_obs_cercanas = df_obs[df_obs["dist_cabecera_km"] <= 12.0].copy()

    # Promedio observado por municipio-año-mes
    obs_mes = (
        df_obs_cercanas.groupby(["cod_dane_mun", "departamento", "municipio", "año", "mes"], as_index=False)
        .agg(temp_obs_ideam_c=("val", "mean"), n_lecturas=("val", "count"))
    )
    obs_mes_comp = obs_mes.merge(
        df_comp_mes[["cod_dane_mun", "año", "mes", "temp_media_mes_c", "nasa_crudo_temp_c"]],
        on=["cod_dane_mun", "año", "mes"],
        how="inner",
    )

    m_raw_obs = metricas(obs_mes_comp["temp_obs_ideam_c"].values, obs_mes_comp["nasa_crudo_temp_c"].values)
    m_cal_obs = metricas(obs_mes_comp["temp_obs_ideam_c"].values, obs_mes_comp["temp_media_mes_c"].values)

    print(f"  -> Municipios con estación automática en muestra sbwg-7ju4 (<=12 km cabecera): {obs_mes_comp['cod_dane_mun'].nunique()} de 208")
    print(f"  -> Pares Municipio-Año-Mes evaluados (2007-2024): N = {len(obs_mes_comp)}")
    df_res_v1 = pd.DataFrame([
        {"Serie Evaluada": "NASA POWER Crudo (Sin Calibrar)", **m_raw_obs},
        {"Serie Evaluada": "NASA POWER Calibrado con IDEAM (Nuestro Pipeline)", **m_cal_obs},
    ])
    print(df_res_v1.round(4).to_string(index=False))

    # -------------------------------------------------------------------------
    # VALIDACIÓN 2: Series Mensuales y Anuales Completas de Estaciones Físicas IDEAM (nsz2-kzcq)
    # -------------------------------------------------------------------------
    print("\n[3/3] Validación 2: Series completas (12 meses y anual) contra Estaciones Oficiales IDEAM en Municipios EVA...")
    for col in ["altitud_m", "latitud", "longitud", "anual"] + MESES_COLS:
        if col in df_norm.columns:
            df_norm[col] = pd.to_numeric(
                df_norm[col].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )
    prio = {"1991-2020": 1, "1981-2010": 2, "1971-2000": 3, "1961-1990": 4}
    df_norm["prio"] = df_norm["periodo"].map(prio).fillna(9)
    df_norm_v = df_norm.dropna(subset=["latitud", "longitud", "anual"] + MESES_COLS).copy()

    df_prec_est = df_norm_v[df_norm_v["par_metro"].str.upper().str.contains("PRECIPITAC", na=False)].sort_values("prio").drop_duplicates("c_digo").reset_index(drop=True)
    df_tmed_est = df_norm_v[df_norm_v["par_metro"].str.upper().str.contains("TEMPERATURA MEDIA", na=False)].sort_values("prio").drop_duplicates("c_digo").reset_index(drop=True)

    # Climatología mensual 2007-2024 de NASA Crudo vs NASA Calibrado por municipio
    clim_mes_mun = df_comp_mes.groupby(["cod_dane_mun", "mes"], as_index=False).agg(
        cal_prec_mes=("precip_mensual_mm", "mean"),
        raw_prec_mes=("nasa_crudo_precip_mm", "mean"),
        cal_temp_mes=("temp_media_mes_c", "mean"),
        raw_temp_mes=("nasa_crudo_temp_c", "mean"),
    )

    # Evaluar en los municipios donde existe una estación física IDEAM dentro del municipio (dist <= 8 km)
    # Además calculamos predicción Leave-One-Station-Out (LOO-CV: excluyendo la estación propia del municipio)
    rows_prec_val = []
    for _, mr in muns.iterrows():
        cod = mr["cod_dane_mun"]
        lat_m, lon_m = float(mr["latitud_dane"]), float(mr["longitud_dane"])
        dists = haversine_km(lat_m, lon_m, df_prec_est["latitud"].values, df_prec_est["longitud"].values)
        idx_sorted = np.argsort(dists)
        if dists[idx_sorted[0]] <= 8.0:
            est_propia = df_prec_est.iloc[idx_sorted[0]]
            # Leave-One-Station-Out: usar las siguientes 3 estaciones (excluyendo idx_sorted[0])
            idx_loo = idx_sorted[1:4]
            w_loo = 1.0 / np.maximum(dists[idx_loo], 1.0) ** 2
            w_loo /= w_loo.sum()
            loo_prec_12 = (df_prec_est.iloc[idx_loo][MESES_COLS].values * w_loo[:, None]).sum(axis=0)

            sub_cm = clim_mes_mun[clim_mes_mun["cod_dane_mun"] == cod].sort_values("mes")
            for m_i, m_col in enumerate(MESES_COLS):
                rows_prec_val.append({
                    "cod_dane_mun": cod,
                    "mes": m_i + 1,
                    "ideam_obs_mm": float(est_propia[m_col]),
                    "nasa_crudo_mm": float(sub_cm.iloc[m_i]["raw_prec_mes"]),
                    "pipeline_cal_mm": float(sub_cm.iloc[m_i]["cal_prec_mes"]),
                    "loo_cv_mm": float(loo_prec_12[m_i]),
                })

    df_pv = pd.DataFrame(rows_prec_val)
    df_pv_anual = df_pv.groupby("cod_dane_mun", as_index=False).sum()

    rows_temp_val = []
    for _, mr in muns.iterrows():
        cod = mr["cod_dane_mun"]
        lat_m, lon_m, alt_m = float(mr["latitud_dane"]), float(mr["longitud_dane"]), float(mr["altitud_msnm"])
        dists = haversine_km(lat_m, lon_m, df_tmed_est["latitud"].values, df_tmed_est["longitud"].values)
        idx_sorted = np.argsort(dists)
        if dists[idx_sorted[0]] <= 10.0:
            est_propia = df_tmed_est.iloc[idx_sorted[0]]
            alt_est_propia = float(est_propia["altitud_m"]) if pd.notna(est_propia["altitud_m"]) else alt_m
            obs_corr_12 = est_propia[MESES_COLS].values.astype(float) + (alt_m - alt_est_propia) * LAPSE_RATE_C_PER_M

            # Leave-One-Station-Out (excluyendo estación propia idx_sorted[0])
            idx_loo = idx_sorted[1:4]
            w_loo = 1.0 / np.maximum(dists[idx_loo], 1.0) ** 2
            w_loo /= w_loo.sum()
            sub_loo = df_tmed_est.iloc[idx_loo]
            alt_loo = sub_loo["altitud_m"].fillna(alt_m).values
            vals_loo = sub_loo[MESES_COLS].values + ((alt_m - alt_loo) * LAPSE_RATE_C_PER_M)[:, None]
            loo_temp_12 = (vals_loo * w_loo[:, None]).sum(axis=0)

            sub_cm = clim_mes_mun[clim_mes_mun["cod_dane_mun"] == cod].sort_values("mes")
            for m_i in range(12):
                rows_temp_val.append({
                    "cod_dane_mun": cod,
                    "mes": m_i + 1,
                    "ideam_obs_c": float(obs_corr_12[m_i]),
                    "nasa_crudo_c": float(sub_cm.iloc[m_i]["raw_temp_mes"]),
                    "pipeline_cal_c": float(sub_cm.iloc[m_i]["cal_temp_mes"]),
                    "loo_cv_c": float(loo_temp_12[m_i]),
                })

    df_tv = pd.DataFrame(rows_temp_val)

    print(f"\n--- 2A. Precipitación Mensual (mm/mes) en {df_pv['cod_dane_mun'].nunique()} municipios con estación IDEAM directa (N = {len(df_pv)} meses) ---")
    df_res_pm = pd.DataFrame([
        {"Modelo / Fuente": "NASA POWER Crudo (Sin Calibrar)", **metricas(df_pv["ideam_obs_mm"].values, df_pv["nasa_crudo_mm"].values)},
        {"Modelo / Fuente": "Validación Cruzada LOO-CV (Ocultando estación local)", **metricas(df_pv["ideam_obs_mm"].values, df_pv["loo_cv_mm"].values)},
        {"Modelo / Fuente": "NASA POWER Calibrado con IDEAM (Pipeline Final)", **metricas(df_pv["ideam_obs_mm"].values, df_pv["pipeline_cal_mm"].values)},
    ])
    print(df_res_pm.round(4).to_string(index=False))

    print(f"\n--- 2B. Precipitación Anual (mm/año) en { len(df_pv_anual) } municipios con estación IDEAM directa ---")
    df_res_pa = pd.DataFrame([
        {"Modelo / Fuente": "NASA POWER Crudo (Sin Calibrar)", **metricas(df_pv_anual["ideam_obs_mm"].values, df_pv_anual["nasa_crudo_mm"].values)},
        {"Modelo / Fuente": "Validación Cruzada LOO-CV (Ocultando estación local)", **metricas(df_pv_anual["ideam_obs_mm"].values, df_pv_anual["loo_cv_mm"].values)},
        {"Modelo / Fuente": "NASA POWER Calibrado con IDEAM (Pipeline Final)", **metricas(df_pv_anual["ideam_obs_mm"].values, df_pv_anual["pipeline_cal_mm"].values)},
    ])
    print(df_res_pa.round(4).to_string(index=False))

    print(f"\n--- 2C. Temperatura Media Mensual (°C) en {df_tv['cod_dane_mun'].nunique()} municipios con estación IDEAM directa (N = {len(df_tv)} meses) ---")
    df_res_tm = pd.DataFrame([
        {"Modelo / Fuente": "NASA POWER Crudo (Sin Calibrar)", **metricas(df_tv["ideam_obs_c"].values, df_tv["nasa_crudo_c"].values)},
        {"Modelo / Fuente": "Validación Cruzada LOO-CV (Ocultando estación local)", **metricas(df_tv["ideam_obs_c"].values, df_tv["loo_cv_c"].values)},
        {"Modelo / Fuente": "NASA POWER Calibrado con IDEAM (Pipeline Final)", **metricas(df_tv["ideam_obs_c"].values, df_tv["pipeline_cal_c"].values)},
    ])
    print(df_res_tm.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
