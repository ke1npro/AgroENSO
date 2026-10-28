import pandas as pd, glob, os, unicodedata
base=r"C:\Proyecto Semillero"
A_path=glob.glob(os.path.join(base,"*Base_Agr*cola*.csv"))[0]

def to_num_old(s):
    if s is None: return float('nan')
    s=str(s).strip().replace(',', '.')
    if s=='' or s.lower()=='nan': return float('nan')
    try: return float(s)
    except: return float('nan')

def to_num_new(s):
    if s is None: return float('nan')
    s=str(s).strip()
    if s=='' or s.lower()=='nan': return float('nan')
    # Si contiene ',' es formato colombiano: '.' miles, ',' decimal
    if ',' in s:
        s=s.replace('.','').replace(',', '.')
    # si no, ya es formato inglés con '.' decimal
    try: return float(s)
    except: return float('nan')

# test
for val in ["4.327,40","4.848,50","94.5","94,5","1.42","1,42","13.985,80","11315.79"]:
    print(val, "old:", to_num_old(val), "new:", to_num_new(val))

# cargar A café Huila
dfA=pd.read_csv(A_path, dtype=str, encoding="utf-8")
# filtrar café Huila
mask=(dfA["Departamento"]=="Huila")&(dfA["Cultivo"]=="Café")
df=dfA[mask].copy()
print("\n=== MUESTRA RAW A café Huila ===")
print(df[["Municipio","Año","Área sembrada","Área cosechada","Producción","Rendimiento"]].head(10).to_string(index=False))

df["area_old"]=df["Área sembrada"].map(to_num_old)
df["area_new"]=df["Área sembrada"].map(to_num_new)
df["prod_old"]=df["Producción"].map(to_num_old)
df["prod_new"]=df["Producción"].map(to_num_new)
print("\nComparación suma 2019 Huila café:")
for yr in [2018,2019]:
    # 2018 no está en A, solo en B, pero para 2019 en A
    sub=df[df["Año"]==str(yr)]
    print(f"Año {yr}: filas {len(sub)} sum old area {sub.area_old.sum():.1f} new {sub.area_new.sum():.1f} prod old {sub.prod_old.sum():.1f} new {sub.prod_new.sum():.1f}")

# Reconstruir agregado con nuevo parsing
import numpy as np
def norm(s):
    return ''.join(c for c in unicodedata.normalize('NFKD', str(s)) if not unicodedata.combining(c)).upper()

CROPS={'CAFE':'Café','CACAO':'Cacao','FRIJOL':'Frijol','MAIZ':'Maíz'}
def cat(r):
    c=norm(r["Cultivo"]); d=norm(r["Desagregación cultivo"])
    if c=='MAIZ':
        if 'TECNIFICADO' in d: return 'Maíz tecnificado'
        if 'TRADICIONAL' in d: return 'Maíz tradicional'
        return 'Maíz'
    return CROPS.get(c)

# Recargar ambos con nuevo parsing
B_path=glob.glob(os.path.join(base,"*2007-2018*"))[0]
dfA2=pd.read_csv(A_path, dtype=str, encoding="utf-8")
for c in ["Área sembrada","Área cosechada","Producción","Rendimiento"]:
    dfA2[c]=dfA2[c].map(to_num_new)
dfA2["año"]=pd.to_numeric(dfA2["Año"], errors="coerce")
dfA2["departamento"]=dfA2["Departamento"].str.upper()
dfA2["municipio"]=dfA2["Municipio"].str.upper()
dfA2["cultivo_cat"]=dfA2.apply(cat, axis=1)
dfA2["cod_dane_mun"]=dfA2["Código Dane municipio"].astype(str).str.zfill(5)
dfA2["periodo"]=dfA2["Periodo"]

dfB=pd.read_excel(B_path, header=1, dtype=str)
for c in ["Área Sembrada\n(ha)","Área Cosechada\n(ha)","Producción\n(t)","Rendimiento\n(t/ha)"]:
    dfB[c]=dfB[c].map(to_num_new)
dfB["año"]=pd.to_numeric(dfB["AÑO"], errors="coerce")
dfB["departamento"]=dfB["DEPARTAMENTO"].str.upper()
dfB["municipio"]=dfB["MUNICIPIO"].str.upper()
dfB["cultivo"]=dfB["CULTIVO"]
dfB["subcategoria"]=dfB["DESAGREGACIÓN REGIONAL Y/O SISTEMA PRODUCTIVO"]
dfB["cultivo_cat"]=dfB.apply(lambda r: cat({"Cultivo":r["CULTIVO"],"Desagregación cultivo":r["DESAGREGACIÓN REGIONAL Y/O SISTEMA PRODUCTIVO"]}), axis=1)
dfB["cod_dane_mun"]=dfB["CÓD. MUN."].astype(str).str.zfill(5)
dfB["periodo"]=dfB["PERIODO"]
# renombrar
dfA2=dfA2.rename(columns={"Área sembrada":"area_sembrada","Área cosechada":"area_cosechada","Producción":"produccion","Rendimiento":"rendimiento"})
dfB=dfB.rename(columns={"Área Sembrada\n(ha)":"area_sembrada","Área Cosechada\n(ha)":"area_cosechada","Producción\n(t)":"produccion","Rendimiento\n(t/ha)":"rendimiento"})

# filtrar deps
DEPS=["HUILA","ANTIOQUIA","TOLIMA"]
cols=["departamento","municipio","cultivo_cat","año","periodo","area_sembrada","area_cosechada","produccion","cod_dane_mun"]
base2=pd.concat([dfA2[cols], dfB[cols]], ignore_index=True)
base2=base2[base2["año"].between(2007,2024) & base2["cultivo_cat"].isin(["Café","Cacao","Frijol","Maíz tecnificado","Maíz tradicional"]) & base2["departamento"].isin(DEPS)]
# colapso maíz >=2021 (igual que antes)
cond=base2["cultivo_cat"].isin(["Maíz tecnificado","Maíz tradicional"]) & (base2["año"]>=2021)
df_ge=base2[cond].copy()
df_rest=base2[~cond].copy()
if len(df_ge):
    collapsed=df_ge.groupby(["cod_dane_mun","cultivo_cat","año","periodo"], as_index=False, dropna=False).agg(
        departamento=("departamento","first"), municipio=("municipio","first"),
        area_sembrada=("area_sembrada","sum"), area_cosechada=("area_cosechada","sum"), produccion=("produccion","sum")
    )
    # para base2 no tenemos subcat etc, simplificado
    corrected2=pd.concat([df_rest, collapsed], ignore_index=True)
else:
    corrected2=base2.copy()
agg2=corrected2.groupby(["departamento","cultivo_cat","año"], as_index=False).agg(
    area_sembrada=("area_sembrada","sum"), area_cosechada=("area_cosechada","sum"), produccion=("produccion","sum")
)
agg2["rendimiento"]=np.where(agg2["area_cosechada"]>0, agg2["produccion"]/agg2["area_cosechada"], np.nan)
print("\n=== AGREGADO CORREGIDO con parsing nuevo 2015-2022 CAFE ===")
for dep in ["HUILA","TOLIMA","ANTIOQUIA"]:
    print(f"\n{dep}")
    print(agg2[(agg2.departamento==dep)&(agg2.cultivo_cat=="Café")&(agg2.año.between(2015,2022))].to_string(index=False, float_format=lambda x: f"{x:.1f}"))
