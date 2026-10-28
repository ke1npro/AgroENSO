import pandas as pd, glob, os
base=r"C:\Proyecto Semillero"
cons=pd.read_csv(os.path.join(base,"eva_2007_2024_consolidado.csv"), dtype=str, encoding="utf-8")
cons["año"]=pd.to_numeric(cons["año"], errors="coerce").astype(int)
cons["cod_dane_mun"]=cons["cod_dane_mun"].astype(str).str.zfill(5)
cons["municipio"]=cons["municipio"].astype(str).str.upper()
for c in ["area_sembrada","area_cosechada","produccion","rendimiento"]:
    cons[c]=pd.to_numeric(cons[c], errors="coerce")

# contar municipios por año para café Huila/Tolima/Antioquia
print("=== N MUNICIPIOS REPORTANTES CAFÉ por año ===")
for dep in ["HUILA","TOLIMA","ANTIOQUIA"]:
    sub=cons[(cons.departamento==dep)&(cons.cultivo_cat=="Café")]
    tbl=sub.groupby("año")["cod_dane_mun"].nunique().reindex(range(2015,2023))
    print(f"{dep}: {tbl.to_dict()}")

print("\n=== AREA SEMBRADA / COSECHADA / PROD CAFÉ Huila detalle año ===")
agg=pd.read_csv(os.path.join(base,"eva_departamento_cultivo_anio.csv"))
for dep in ["HUILA","TOLIMA","ANTIOQUIA"]:
    sub=agg[(agg.departamento==dep)&(agg.cultivo_cat=="Café")&(agg.año.between(2015,2022))].sort_values("año")
    print(sub.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

# verificar si café en Huila 2019 son pocos municipios con area pequeña
print("\n=== HUILA CAFÉ 2018 vs 2019: lista municipios (codigo, area, prod) ===")
for yr in [2018,2019]:
    s=cons[(cons.departamento=="HUILA")&(cons.cultivo_cat=="Café")&(cons.año==yr)]
    print(f"\n-- {yr} Huila café: {len(s)} filas, {s.cod_dane_mun.nunique()} municipios --")
    print(s[["cod_dane_mun","municipio","area_sembrada","area_cosechada","produccion","rendimiento"]].head(10).to_string(index=False))
    print(f" Suma area_sembrada {s.area_sembrada.sum():.1f} area_cosechada {s.area_cosechada.sum():.1f} prod {s.produccion.sum():.1f}")

# comparar cacao cacao Huila 2018 vs 2019
print("\n=== CACAO Huila 2018 vs 2019 ===")
for yr in [2018,2019]:
    s=cons[(cons.departamento=="HUILA")&(cons.cultivo_cat=="Cacao")&(cons.año==yr)]
    print(f"{yr}: filas {len(s)} mun {s.cod_dane_mun.nunique()} area {s.area_sembrada.sum():.1f} prod {s.produccion.sum():.1f} rend {s.produccion.sum()/s.area_cosechada.sum():.3f}")

# Frijol
print("\n=== FRIJOL Huila 2018 vs 2019 ===")
for yr in [2018,2019]:
    s=cons[(cons.departamento=="HUILA")&(cons.cultivo_cat=="Frijol")&(cons.año==yr)]
    print(f"{yr}: filas {len(s)} mun {s.cod_dane_mun.nunique()} area {s.area_sembrada.sum():.1f} prod {s.produccion.sum():.1f} rend {s.produccion.sum()/s.area_cosechada.sum():.3f}")

# buscar en definiciones si mencionan café verde vs pergamino
print("\n=== BÚSQUEDA TEXTO 'CAFE' en DEFINICIONES y CLASIFICACIÓN ===")
import openpyxl
B_path=glob.glob(os.path.join(base,"*2007-2018*"))[0]
for sheet in ["DEFINICIONES BÁSICAS","CLASIFICACIÓN AGRÍCOLA"]:
    df=pd.read_excel(B_path, sheet_name=sheet, header=None, dtype=str)
    flat=" ".join([str(v) for v in df.values.flatten() if pd.notna(v)])
    if "CAFE" in flat.upper():
        print(f"Hoja {sheet} contiene CAFE: ...{flat[flat.upper().find('CAFE')-100:flat.upper().find('CAFE')+200]}...")
    else:
        print(f"Hoja {sheet} NO menciona CAFE separado")

# Inspeccionar A metadata: buscar diccionario
print("\n=== A: revisar si hay diccionario de datos en CSV header? No, pero revisar produccion unidad ===")
A_path=glob.glob(os.path.join(base,"*Base_Agr*cola*.csv"))[0]
dfA_head=pd.read_csv(A_path, nrows=1, dtype=str, encoding="utf-8")
print("Columnas A producción:", [c for c in dfA_head.columns if "Producci" in c or "Rendimiento" in c])
# ver descripción datos.gov.co ya sabemos

# Calcular factor hipotético si fuera conversión verde->pergamino
# verde = pergamino * 0.80 => pergamino = verde /0.80 =1.25*verde
# Si B es verde y A es pergamino, A debería ser ~0.8 * B? Wait verde es menos que pergamino
# B verde 154k t, A pergamino 2.6k t => A =0.017*B, no 1.25*B
# Si B es cereza (cereza->pergamino ~5), B cereza 154k => pergamino 30k, no 2.6k
# entonces no cuadra
print("\n=== FACTORES HIPOTÉTICOS ===")
for dep in ["HUILA","TOLIMA","ANTIOQUIA"]:
    agg=pd.read_csv(os.path.join(base,"eva_departamento_cultivo_anio.csv"))
    sub=agg[(agg.departamento==dep)&(agg.cultivo_cat=="Café")]
    p2018=sub[sub.año==2018].produccion.values[0] if len(sub[sub.año==2018]) else None
    p2019=sub[sub.año==2019].produccion.values[0] if len(sub[sub.año==2019]) else None
    if p2018 and p2019:
        ratio=p2019/p2018
        print(f"{dep} ratio 2019/2018={ratio:.3f} factor inverso {1/ratio:.1f}x")
        print(f"  Si fuera cereza->pergamino (5x), esperado ratio ~0.20, observado {ratio:.3f} => {'NO cuadra' if abs(ratio-0.20)>0.05 else 'posible'}")
        print(f"  Si fuera pergamino->verde 0.8x, esperado ratio ~1.25, observado {ratio:.3f} => NO cuadra")
