"""
Pipeline Agrícola EVA - Región Caribe (2007-2024)
==================================================
Procesa y estandariza los registros de producción agrícola municipal para los 7
departamentos del Caribe colombiano:
  - Atlántico (08)
  - Bolívar (13)
  - Cesar (20)
  - Córdoba (23)
  - La Guajira (44)
  - Magdalena (47)
  - Sucre (70)

Cultivos objetivo (garantizando paridad metodológica con Región Andina):
  - Maíz tradicional
  - Maíz tecnificado
  - Fríjol
  - Cacao
  - Café

Reglas de estandarización:
  1. Parser numérico robusto (comas decimales y puntos de miles).
  2. Colapso de variedades de maíz (amarillo/blanco) desde 2021 por municipio-año-periodo.
  3. Cálculo de tendencia tecnológica departamental (Y_pot_{d,c,t}) con media móvil
     centrada de 5 años excluyendo años de El Niño fuerte (2009, 2015, 2016, 2023).
  4. Filtrado de registros con área cosechada > 0 y rendimiento válido > 0.
"""

import os
import glob
import unicodedata
import numpy as np
import pandas as pd

BASE_DIR = r"C:\Proyecto Semillero"
A_PATH = glob.glob(os.path.join(BASE_DIR, "*Base_Agr*cola*.csv"))[0]
B_PATH = glob.glob(os.path.join(BASE_DIR, "*2007-2018*"))[0]

DEPS_CARIBE = ["ATLANTICO", "BOLIVAR", "CESAR", "CORDOBA", "LA GUAJIRA", "MAGDALENA", "SUCRE"]
CROPS_MAP = {"CAFE": "Café", "CACAO": "Cacao", "FRIJOL": "Frijol", "MAIZ": "Maíz"}


def norm_str(s):
    if s is None or pd.isna(s):
        return ""
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c)).upper()


def to_num_new(s):
    if s is None or pd.isna(s):
        return np.nan
    s = str(s).strip()
    if s == "" or s.lower() == "nan":
        return np.nan
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return np.nan


def categorizar_cultivo(c_raw, d_raw):
    c = norm_str(c_raw)
    d = norm_str(d_raw)
    if c == "MAIZ":
        if "TECNIFICADO" in d:
            return "Maíz tecnificado"
        if "TRADICIONAL" in d:
            return "Maíz tradicional"
        return "Maíz tradicional"
    return CROPS_MAP.get(c, None)


def ejecutar_pipeline_eva_caribe():
    print("=" * 80)
    print("INICIANDO PIPELINE AGRÍCOLA EVA - REGIÓN CARIBE (2007-2024)")
    print("=" * 80)

    # 1. Cargar Archivo A (2019-2025)
    print("\n[1/5] Cargando y filtrando Archivo A (Datos Abiertos, 2019-2025)...")
    map_a = {
        "Departamento": "departamento",
        "Municipio": "municipio",
        "Cultivo": "cultivo",
        "Desagregación cultivo": "subcategoria",
        "Año": "año",
        "Periodo": "periodo",
        "Área sembrada": "area_sembrada",
        "Área cosechada": "area_cosechada",
        "Producción": "produccion",
        "Rendimiento": "rendimiento",
        "Código Dane departamento": "cod_dane_dep",
        "Código Dane municipio": "cod_dane_mun",
    }
    dfA = pd.read_csv(A_PATH, dtype=str, encoding="utf-8").rename(columns=map_a)
    for c in ["area_sembrada", "area_cosechada", "produccion", "rendimiento"]:
        dfA[c] = dfA[c].map(to_num_new)
    dfA["año"] = pd.to_numeric(dfA["año"], errors="coerce")
    dfA["periodo"] = dfA["periodo"].astype(str).str.strip()
    dfA["fuente"] = "A_datosgov_2019_2025"

    # 2. Cargar Archivo B (2007-2018)
    print("[2/5] Cargando y filtrando Archivo B (AgroNet, 2007-2018)...")
    map_b = {
        "DEPARTAMENTO": "departamento",
        "MUNICIPIO": "municipio",
        "CULTIVO": "cultivo",
        "DESAGREGACIÓN REGIONAL Y/O SISTEMA PRODUCTIVO": "subcategoria",
        "AÑO": "año",
        "PERIODO": "periodo",
        "Área Sembrada\n(ha)": "area_sembrada",
        "Área Cosechada\n(ha)": "area_cosechada",
        "Producción\n(t)": "produccion",
        "Rendimiento\n(t/ha)": "rendimiento",
        "CÓD. \nDEP.": "cod_dane_dep",
        "CÓD. MUN.": "cod_dane_mun",
    }
    dfB = pd.read_excel(B_PATH, header=1, dtype=str).rename(columns=map_b)
    for c in ["area_sembrada", "area_cosechada", "produccion", "rendimiento"]:
        dfB[c] = dfB[c].map(to_num_new)
    dfB["año"] = pd.to_numeric(dfB["año"], errors="coerce")
    dfB["periodo"] = dfB["periodo"].astype(str).str.strip()
    dfB["fuente"] = "B_agronet_2007_2018"

    # Estandarizar departamentos, municipios, códigos DANE y categorías
    for df in (dfA, dfB):
        df["departamento"] = df["departamento"].astype(str).str.strip().str.upper()
        df["municipio"] = df["municipio"].astype(str).str.strip().str.upper()
        df["cod_dane_mun"] = df["cod_dane_mun"].astype(str).str.strip().str.zfill(5)
        df["cod_dane_dep"] = df["cod_dane_dep"].astype(str).str.strip().str.zfill(2)
        df["cultivo_cat"] = [
            categorizar_cultivo(c, d) for c, d in zip(df["cultivo"], df["subcategoria"])
        ]

    maskA = dfA["departamento"].isin(DEPS_CARIBE) & dfA["cultivo_cat"].notna()
    maskB = dfB["departamento"].isin(DEPS_CARIBE) & dfB["cultivo_cat"].notna()
    fA = dfA[maskA].copy()
    fB = dfB[maskB].copy()

    cols_export = [
        "departamento", "municipio", "cultivo", "subcategoria", "cultivo_cat",
        "año", "periodo", "area_sembrada", "area_cosechada", "produccion",
        "rendimiento", "fuente", "cod_dane_dep", "cod_dane_mun"
    ]
    base = pd.concat([fA[cols_export], fB[cols_export]], ignore_index=True)
    base = base[(base["año"] >= 2007) & (base["año"] <= 2024)].copy()
    base["año"] = base["año"].astype(int)
    print(f"    -> Filas brutas Caribe (2007-2024): {len(base)} (A: {len(base[base.fuente=='A_datosgov_2019_2025'])}, B: {len(base[base.fuente=='B_agronet_2007_2018'])})")

    # 3. Colapso de variedades de maíz >= 2021
    print("\n[3/5] Aplicando colapso de variedades de maíz (>=2021)...")
    cond_maiz = base["cultivo_cat"].isin(["Maíz tecnificado", "Maíz tradicional"]) & (base["año"] >= 2021)
    df_maiz_ge2021 = base[cond_maiz].copy()
    df_resto = base[~cond_maiz].copy()

    gkeys = ["cod_dane_mun", "cultivo_cat", "año", "periodo"]
    collapsed = df_maiz_ge2021.groupby(gkeys, as_index=False, dropna=False).agg(
        departamento=("departamento", "first"),
        municipio=("municipio", "first"),
        cultivo=("cultivo", "first"),
        area_sembrada=("area_sembrada", "sum"),
        area_cosechada=("area_cosechada", "sum"),
        produccion=("produccion", "sum"),
        fuente=("fuente", "first"),
        cod_dane_dep=("cod_dane_dep", "first"),
    )
    collapsed["subcategoria"] = collapsed["cultivo_cat"]
    collapsed["rendimiento"] = np.where(
        collapsed["area_cosechada"] > 0,
        collapsed["produccion"] / collapsed["area_cosechada"],
        np.nan,
    )
    collapsed = collapsed[cols_export]

    df_consolidado = pd.concat([df_resto, collapsed], ignore_index=True)
    df_consolidado = df_consolidado.sort_values(
        ["departamento", "cultivo_cat", "año", "cod_dane_mun", "periodo"]
    ).reset_index(drop=True)
    print(f"    -> Filas tras colapso maíz: {len(df_consolidado)} (reducción de {len(base) - len(df_consolidado)} filas desagregadas)")

    # 4. Agregado Departamental y Tendencia Potencial Tecnológica
    print("\n[4/5] Estimando rendimiento potencial departamental (Y_pot_{d,c,t})...")
    # Agregación departamental anual
    agg_dep = df_consolidado.groupby(["departamento", "cultivo_cat", "año"], as_index=False).agg(
        area_sembrada=("area_sembrada", "sum"),
        area_cosechada=("area_cosechada", "sum"),
        produccion=("produccion", "sum"),
    )
    agg_dep["rendimiento"] = np.where(
        agg_dep["area_cosechada"] > 0,
        agg_dep["produccion"] / agg_dep["area_cosechada"],
        np.nan,
    )
    agg_dep = agg_dep.sort_values(["departamento", "cultivo_cat", "año"]).reset_index(drop=True)
    agg_dep.to_csv(os.path.join(BASE_DIR, "eva_departamento_cultivo_anio_caribe.csv"), index=False, encoding="utf-8")

    anios_nino_fuerte = {2009, 2015, 2016, 2023}
    resultados_pot = []
    for (dep, cult), g in agg_dep.groupby(["departamento", "cultivo_cat"]):
        g = g.sort_values("año").copy()
        y_sin_nino = g["rendimiento"].where(~g["año"].isin(anios_nino_fuerte), np.nan)
        y_interp = y_sin_nino.interpolate(method="linear", limit_direction="both")
        y_smooth = y_interp.rolling(window=5, min_periods=1, center=True).mean()
        if y_smooth.isna().any():
            y_smooth = g["rendimiento"].rolling(window=5, min_periods=1, center=True).mean().bfill().ffill()

        g["rendimiento_potencial_dep"] = y_smooth.round(5)
        g["eficiencia_agregada_dep"] = (g["rendimiento"] / g["rendimiento_potencial_dep"]).round(4)
        resultados_pot.append(g)

    df_pot = pd.concat(resultados_pot, ignore_index=True)
    df_pot.to_csv(os.path.join(BASE_DIR, "eva_departamento_tendencia_potencial_caribe.csv"), index=False, encoding="utf-8")
    print(f"    -> Guardado 'eva_departamento_tendencia_potencial_caribe.csv' ({len(df_pot)} filas)")

    # 5. Filtrar registros válidos y calcular eficiencias
    print("\n[5/5] Calculando eficiencia y guardando base consolidada Caribe...")
    df_valid = df_consolidado[
        (df_consolidado["rendimiento"].notna())
        & (df_consolidado["rendimiento"] > 0)
        & (df_consolidado["area_cosechada"].fillna(0) > 0)
    ].copy()
    print(f"    -> Filas con cosecha y rendimiento válido (>0): {len(df_valid)} de {len(df_consolidado)}")

    # Merge con rendimiento potencial departamental
    df_valid = df_valid.merge(
        df_pot[["departamento", "cultivo_cat", "año", "rendimiento_potencial_dep"]],
        on=["departamento", "cultivo_cat", "año"],
        how="left",
    )
    df_valid["eficiencia"] = (df_valid["rendimiento"] / df_valid["rendimiento_potencial_dep"]).round(4)

    # Base municipal histórica (mediana sin El Niño)
    base_mun = (
        df_valid[~df_valid["año"].isin(anios_nino_fuerte)]
        .groupby(["cod_dane_mun", "cultivo_cat"], as_index=False)["rendimiento"]
        .median()
        .rename(columns={"rendimiento": "rendimiento_base_municipal"})
    )
    df_valid = df_valid.merge(base_mun, on=["cod_dane_mun", "cultivo_cat"], how="left")
    df_valid["rendimiento_base_municipal"] = df_valid["rendimiento_base_municipal"].fillna(
        df_valid["rendimiento_potencial_dep"]
    )
    df_valid["eficiencia_rel_municipal"] = (
        df_valid["rendimiento"] / df_valid["rendimiento_base_municipal"]
    ).round(4)

    # Eficiencia clip (winsorizada 0.5% - 99.5%)
    q_low = float(df_valid["eficiencia"].quantile(0.005))
    q_high = float(df_valid["eficiencia"].quantile(0.995))
    df_valid["eficiencia_clip"] = df_valid["eficiencia"].clip(lower=q_low, upper=q_high).round(4)

    # Flag de panel balanceado (18 años)
    yrs_mun = df_valid.groupby(["departamento", "cultivo_cat", "cod_dane_mun"])["año"].apply(set)
    mun_bal = set(m for (dep, cat, m), ys in yrs_mun.items() if len(ys) == 18 and set(range(2007, 2025)).issubset(ys))
    df_valid["en_panel_balanceado"] = df_valid["cod_dane_mun"].isin(mun_bal)

    out_path = os.path.join(BASE_DIR, "eva_2007_2024_caribe.csv")
    df_valid.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nConsolidado final Caribe guardado: {out_path} ({len(df_valid)} filas)")
    print(f"Rango de eficiencia sin recortar: [{df_valid['eficiencia'].min():.3f}, {df_valid['eficiencia'].max():.3f}] | Mediana: {df_valid['eficiencia'].median():.3f}")
    print(f"Rango de eficiencia_clip (0.5%-99.5%): [{q_low:.3f}, {q_high:.3f}]")
    print("\nDesglose de filas válidas por cultivo:")
    print(df_valid["cultivo_cat"].value_counts())
    print("\nDesglose de filas válidas por departamento:")
    print(df_valid["departamento"].value_counts())


if __name__ == "__main__":
    ejecutar_pipeline_eva_caribe()
