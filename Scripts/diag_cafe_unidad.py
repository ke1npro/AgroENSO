import pandas as pd, numpy as np, glob, os, unicodedata, openpyxl
base=r"C:\Proyecto Semillero"
pd.set_option('display.max_columns',None)
pd.set_option('display.width',300)
pd.set_option('display.max_colwidth',60)

# 1. TABLA AGREGADA 2015-2022
agg=pd.read_csv(os.path.join(base,"eva_departamento_cultivo_anio.csv"))
print("=== AGG eva_departamento_cultivo_anio.csv HEAD ===")
print(agg.head(3).to_string())
print("\n=== CAFE Huila y Tolima 2015-2022 ===")
for dep in ["HUILA","TOLIMA"]:
    print(f"\n--- {dep} CAFÉ ---")
    sub=agg[(agg.departamento==dep)&(agg.cultivo_cat=="Café")&(agg.año.between(2015,2022))].sort_values("año")
    print(sub.to_string(index=False, float_format=lambda x: f"{x:.2f}" if pd.notna(x) else "NaN"))
    # calcular ratio
    try:
        p2018=sub[sub.año==2018].produccion.values[0]
        p2019=sub[sub.año==2019].produccion.values[0]
        r2018=sub[sub.año==2018].rendimiento.values[0]
        r2019=sub[sub.año==2019].rendimiento.values[0]
        print(f" Ratio 2019/2018 prod={p2019/p2018:.3f}  rend 2018={r2018:.3f} 2019={r2019:.3f}")
    except: pass

print("\n=== CACAO y FRIJOL 2015-2022 (3 deps) ===")
for cult in ["Cacao","Frijol"]:
    print(f"\n--- {cult.upper()} ---")
    sub=agg[(agg.cultivo_cat==cult)&(agg.año.between(2015,2022))].sort_values(["departamento","año"])
    # pivot para ver rendimiento año a año
    print(sub.to_string(index=False, float_format=lambda x: f"{x:.2f}" if pd.notna(x) else "NaN"))

# 2. INSPECCION ESTADO FISICO EN FUENTES ORIGINALES
A_path=glob.glob(os.path.join(base,"*Base_Agr*cola*.csv"))[0]
B_path=glob.glob(os.path.join(base,"*2007-2018*"))[0]
print("\n=== INSPECCION ARCHIVO A: columnas ===")
dfA_raw=pd.read_csv(A_path, nrows=0, encoding="utf-8", dtype=str)
print(dfA_raw.columns.tolist())
# cargar solo columnas relevantes para cafe
usecolsA=["Departamento","Cultivo","Desagregación cultivo","Año","Estado físico del cultivo","Producción","Rendimiento","Área sembrada","Área cosechada"]
dfA=pd.read_csv(A_path, usecols=usecolsA, dtype=str, encoding="utf-8")
# filtrar cafe
dfA_cafe=dfA[dfA["Cultivo"]=="Café"]
print("\n A - Estado físico único para CAFÉ:")
print(dfA_cafe["Estado físico del cultivo"].dropna().unique().tolist())
print(" A - muestra café 2019-2024 (Huila,Tolima):")
print(dfA_cafe[dfA_cafe["Departamento"].isin(["Huila","Tolima"])].head(8).to_string(index=False))
print("\n A - Estado físico para CACAO:")
print(dfA[dfA["Cultivo"]=="Cacao"]["Estado físico del cultivo"].dropna().unique().tolist()[:5])
print(" A - Estado físico para FRIJOL:")
print(dfA[dfA["Cultivo"]=="Frijol"]["Estado físico del cultivo"].dropna().unique().tolist()[:10])

print("\n=== INSPECCION ARCHIVO B: hojas ===")
import openpyxl
wb=openpyxl.load_workbook(B_path, read_only=True, data_only=True)
print("Hojas:", wb.sheetnames)
# leer hoja FINAL header fila 2? ya sabemos
dfB=pd.read_excel(B_path, header=1, dtype=str)
print("\n B - columnas:", dfB.columns.tolist()[:10])
# filtrar cafe
dfB_cafe=dfB[dfB["CULTIVO"]=="CAFE"]
print("\n B - ESTADO FISICO PRODUCCION único para CAFE:")
print(dfB_cafe["ESTADO FISICO PRODUCCION"].dropna().unique().tolist())
print(" B - muestra café (Huila,Tolima) primeras filas:")
print(dfB_cafe[dfB_cafe["DEPARTAMENTO"].isin(["HUILA","TOLIMA"])].head(6).to_string(index=False))
print("\n B - ESTADO FISICO para CACAO:")
print(dfB[dfB["CULTIVO"]=="CACAO"]["ESTADO FISICO PRODUCCION"].dropna().unique().tolist()[:5])
print(" B - ESTADO FISICO para FRIJOL:")
print(dfB[dfB["CULTIVO"]=="FRIJOL"]["ESTADO FISICO PRODUCCION"].dropna().unique().tolist()[:10])

# leer definiciones
print("\n=== HOJA DEFINICIONES BÁSICAS (B) ===")
try:
    dfDef=pd.read_excel(B_path, sheet_name="DEFINICIONES BÁSICAS", header=None, dtype=str)
    print(f"Shape definiciones: {dfDef.shape}")
    # mostrar primeras 40 filas crudas
    for i,row in dfDef.head(60).iterrows():
        vals=[str(v) for v in row.values if pd.notna(v)]
        if vals:
            print(f"R{i}: {' | '.join(vals)}")
except Exception as e:
    print("Error leyendo definiciones:", e)

print("\n=== HOJA CLASIFICACIÓN AGRÍCOLA (B) ===")
try:
    dfClas=pd.read_excel(B_path, sheet_name="CLASIFICACIÓN AGRÍCOLA", header=None, dtype=str)
    print(f"Shape clasificacion: {dfClas.shape}")
    for i,row in dfClas.head(60).iterrows():
        vals=[str(v) for v in row.values if pd.notna(v)]
        if vals:
            print(f"R{i}: {' | '.join(vals)}")
except Exception as e:
    print("Error leyendo clasificacion:", e)

# 3. Rendimientos tipicos chequeo
print("\n=== CHEQUEO RENDIMIENTOS TIPICOS CAFÉ 0.5-1.3 t/ha ===")
for dep in ["HUILA","TOLIMA"]:
    sub=agg[(agg.departamento==dep)&(agg.cultivo_cat=="Café")&(agg.año.between(2015,2022))]
    print(f"\n{dep} Café rendimiento 2015-2022:")
    for _,r in sub.iterrows():
        flag="OK" if 0.5<=r.rendimiento<=1.3 else ("BAJO" if r.rendimiento<0.5 else "ALTO")
        print(f" {int(r.año)}: {r.rendimiento:.3f} t/ha -> {flag}  (prod {r.produccion:.0f} t / area_cosech {r.area_cosechada:.0f} ha)")

# 4. Misma para comparar con eva_departamento antes de colapso? No.
# Buscar documentación en archivo A descripción - no hay
# Inspeccionar area sembrada también colapsa?
print("\n=== AREA SEMBRADA/COSECHADA CAFÉ Huila,Tolima 2015-2022 para ver si área también colapsa ===")
# ya impreso arriba, pero recalcular ratio area
for dep in ["HUILA","TOLIMA"]:
    sub=agg[(agg.departamento==dep)&(agg.cultivo_cat=="Café")&(agg.año.between(2015,2022))].sort_values("año")
    a2018=sub[sub.año==2018].area_cosechada.values[0] if len(sub[sub.año==2018]) else np.nan
    a2019=sub[sub.año==2019].area_cosechada.values[0] if len(sub[sub.año==2019]) else np.nan
    print(f"{dep} area_cosechada 2018={a2018:.0f} 2019={a2019:.0f} ratio={a2019/a2018:.3f}" if pd.notna(a2018) and pd.notna(a2019) else f"{dep} NA")

