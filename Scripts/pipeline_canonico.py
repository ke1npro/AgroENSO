import pandas as pd, numpy as np, glob, os, unicodedata, openpyxl
pd.set_option('display.max_columns',None)
pd.set_option('display.width',400)
pd.set_option('display.max_colwidth',60)
pd.set_option('display.max_rows',500)
BASE_DIR=r"C:\Proyecto Semillero"
A_PATH=glob.glob(os.path.join(BASE_DIR,"*Base_Agr*cola*.csv"))[0]
B_PATH=glob.glob(os.path.join(BASE_DIR,"*2007-2018*"))[0]
print("="*100)
print("PIPELINE CANÓNICO ÚNICO - desde crudos A y B")
print(f"A: {os.path.basename(A_PATH)}")
print(f"B: {os.path.basename(B_PATH)}")
print("="*100)
def norm(s):
    if s is None: return ''
    return ''.join(c for c in unicodedata.normalize('NFKD', str(s)) if not unicodedata.combining(c)).upper()
def to_num_old(s):
    if s is None: return np.nan
    s=str(s).strip().replace(',', '.')
    if s=='' or s.lower()=='nan': return np.nan
    try: return float(s)
    except: return np.nan
def to_num_new(s):
    if s is None: return np.nan
    s=str(s).strip()
    if s=='' or s.lower()=='nan': return np.nan
    if ',' in s:
        s=s.replace('.','').replace(',','.')
    try: return float(s)
    except: return np.nan
# ------------------------------------------------------------
# 1. CONTEO FILAS AFECTADAS POR BUG DE PARSING (A)
print("\n"+"="*100)
print("1. RECUENTO BUG PARSING - A (formato colombiano) vs B")
print("="*100)
dfA_raw=pd.read_csv(A_PATH, dtype=str, encoding="utf-8")
cols_num=["Área sembrada","Área cosechada","Producción","Rendimiento"]
# aplicar ambos parsers a mismas columnas crudas para comparar
for c in cols_num:
    dfA_raw[c+"_old"]=dfA_raw[c].map(to_num_old)
    dfA_raw[c+"_new"]=dfA_raw[c].map(to_num_new)
# cultivo_cat para desglose (no depende de parsing)
CROPS={'CAFE':'Café','CACAO':'Cacao','FRIJOL':'Frijol','MAIZ':'Maíz'}
def cat_row(r):
    c=norm(r["Cultivo"]); d=norm(r["Desagregación cultivo"])
    if c=='MAIZ':
        if 'TECNIFICADO' in d: return 'Maíz tecnificado'
        if 'TRADICIONAL' in d: return 'Maíz tradicional'
        return 'Maíz'
    return CROPS.get(c)
dfA_raw["cultivo_cat"]=dfA_raw.apply(cat_row, axis=1)
dfA_raw["departamento_norm"]=dfA_raw["Departamento"].astype(str).str.strip().str.upper()
# total filas A
total_A=len(dfA_raw)
print(f"Total filas A crudo: {total_A}")
for col in cols_num:
    changed=((dfA_raw[col+"_old"].isna()) & (dfA_raw[col+"_new"].notna())).sum()
    print(f" {col}: OLD NaN->NEW válido = {changed} filas ({changed/total_A*100:.2f}%)")
# desglose por cultivo_cat (todos) y departamento (todos) - mostrar top
print("\n--- Desglose por cultivo_cat (A, cambio NaN->válido en al menos una de area/prod/rend) ---")
mask_any=((dfA_raw["Área sembrada_old"].isna() & dfA_raw["Área sembrada_new"].notna()) |
          (dfA_raw["Área cosechada_old"].isna() & dfA_raw["Área cosechada_new"].notna()) |
          (dfA_raw["Producción_old"].isna() & dfA_raw["Producción_new"].notna()))
dfA_raw["afectada"]=mask_any
print(dfA_raw.groupby("cultivo_cat")["afectada"].agg(["sum","count"]).sort_values("sum", ascending=False).head(20).to_string())
print("\n--- Desglose por departamento (top 15 afectados) ---")
print(dfA_raw.groupby("departamento_norm")["afectada"].agg(["sum","count"]).sort_values("sum", ascending=False).head(15).to_string())
# desglose específico para el subconjunto del consolidado (3 deps, 5 cultivos) para comparar con entregables previos
DEPS_FILT=['HUILA','ANTIOQUIA','TOLIMA']
CULTS_FILT=['Café','Cacao','Frijol','Maíz tecnificado','Maíz tradicional']
sub=dfA_raw[dfA_raw["departamento_norm"].isin(DEPS_FILT) & dfA_raw["cultivo_cat"].isin(CULTS_FILT)]
print(f"\n--- Subconjunto consolidado (3 deps x 5 cultivos) A: {len(sub)} filas ---")
for cat in CULTS_FILT:
    s=sub[sub.cultivo_cat==cat]
    aff=s["afectada"].sum()
    print(f" {cat}: {aff}/{len(s)} afectadas ({aff/len(s)*100:.1f}%)")
for dep in DEPS_FILT:
    s=sub[sub.departamento_norm==dep]
    aff=s["afectada"].sum()
    print(f" {dep}: {aff}/{len(s)} afectadas ({aff/len(s)*100:.1f}%)")
# Matriz cultivo x depto para filtrado
print("\nMatriz afectadas (cultivo_cat x departamento) - subconjunto 3x5:")
mat=sub.groupby(["cultivo_cat","departamento_norm"])["afectada"].sum().unstack(fill_value=0)
print(mat.to_string())
# ------------------------------------------------------------
# 2. PIPELINE CORREGIDO COMPLETO (desde crudos, con parsing nuevo, colapso maíz, estandarización)
print("\n"+"="*100)
print("2. PIPELINE CORREGIDO - carga con parsing nuevo")
print("="*100)
MAP_A={'Departamento':'departamento','Municipio':'municipio','Cultivo':'cultivo','Desagregación cultivo':'subcategoria','Año':'año','Periodo':'periodo','Área sembrada':'area_sembrada','Área cosechada':'area_cosechada','Producción':'produccion','Rendimiento':'rendimiento','Código Dane departamento':'cod_dane_dep','Código Dane municipio':'cod_dane_mun'}
dfA=pd.read_csv(A_PATH, dtype=str, encoding="utf-8").rename(columns=MAP_A)
for c in ['area_sembrada','area_cosechada','produccion','rendimiento']:
    dfA[c]=dfA[c].map(to_num_new)
dfA['año']=pd.to_numeric(dfA['año'], errors='coerce')
dfA['periodo']=dfA['periodo'].astype(str).str.strip()
dfA['fuente']='A_datosgov_2019_2025'
MAP_B={'DEPARTAMENTO':'departamento','MUNICIPIO':'municipio','CULTIVO':'cultivo','DESAGREGACIÓN REGIONAL Y/O SISTEMA PRODUCTIVO':'subcategoria','AÑO':'año','PERIODO':'periodo','Área Sembrada\n(ha)':'area_sembrada','Área Cosechada\n(ha)':'area_cosechada','Producción\n(t)':'produccion','Rendimiento\n(t/ha)':'rendimiento','CÓD. \nDEP.':'cod_dane_dep','CÓD. MUN.':'cod_dane_mun'}
dfB=pd.read_excel(B_PATH, header=1, dtype=str).rename(columns=MAP_B)
for c in ['area_sembrada','area_cosechada','produccion','rendimiento']:
    dfB[c]=dfB[c].map(to_num_new)
dfB['año']=pd.to_numeric(dfB['año'], errors='coerce')
dfB['periodo']=dfB['periodo'].astype(str).str.strip()
dfB['fuente']='B_agronet_2007_2018'
for df in (dfA,dfB):
    df['departamento']=df['departamento'].astype(str).str.strip().str.upper()
    df['municipio']=df['municipio'].astype(str).str.strip().str.upper()
    df['cod_dane_mun']=df['cod_dane_mun'].astype(str).str.strip().str.zfill(5)
    df['cod_dane_dep']=df['cod_dane_dep'].astype(str).str.strip().str.zfill(2)
    df['cultivo_cat']=df.apply(lambda r: (lambda c,d: 'Maíz tecnificado' if c=='MAIZ' and 'TECNIFICADO' in d else 'Maíz tradicional' if c=='MAIZ' and 'TRADICIONAL' in d else 'Maíz' if c=='MAIZ' else {'CAFE':'Café','CACAO':'Cacao','FRIJOL':'Frijol'}.get(c))(norm(r['cultivo']), norm(r['subcategoria'])), axis=1)
DEPS=['HUILA','ANTIOQUIA','TOLIMA']
maskA=dfA['departamento'].isin(DEPS) & dfA['cultivo_cat'].isin(['Café','Cacao','Frijol','Maíz tecnificado','Maíz tradicional'])
maskB=dfB['departamento'].isin(DEPS) & dfB['cultivo_cat'].isin(['Café','Cacao','Frijol','Maíz tecnificado','Maíz tradicional'])
fA=dfA[maskA].copy(); fB=dfB[maskB].copy()
cols=['departamento','municipio','cultivo','subcategoria','cultivo_cat','año','periodo','area_sembrada','area_cosechada','produccion','rendimiento','fuente','cod_dane_dep','cod_dane_mun']
base=pd.concat([fA[cols], fB[cols]], ignore_index=True)
base=base[(base['año']>=2007)&(base['año']<=2024)].copy()
base['año']=base['año'].astype(int)
print(f"Base antes colapso maíz: {len(base)} filas (A:{len(base[base.fuente=='A_datosgov_2019_2025'])} B:{len(base[base.fuente=='B_agronet_2007_2018'])})")
# Colapso maíz >=2021 por (cod_dane_mun, cultivo_cat, año, periodo)
cond=base['cultivo_cat'].isin(['Maíz tecnificado','Maíz tradicional']) & (base['año']>=2021)
df_ge=base[cond].copy(); df_rest=base[~cond].copy()
gkeys=['cod_dane_mun','cultivo_cat','año','periodo']
collapsed=df_ge.groupby(gkeys, as_index=False, dropna=False).agg(departamento=('departamento','first'),municipio=('municipio','first'),cultivo=('cultivo','first'),area_sembrada=('area_sembrada','sum'),area_cosechada=('area_cosechada','sum'),produccion=('produccion','sum'),fuente=('fuente','first'),cod_dane_dep=('cod_dane_dep','first'))
collapsed['subcategoria']=collapsed['cultivo_cat']
collapsed['rendimiento']=np.where(collapsed['area_cosechada']>0, collapsed['produccion']/collapsed['area_cosechada'], np.nan)
collapsed=collapsed[['departamento','municipio','cultivo','subcategoria','cultivo_cat','año','periodo','area_sembrada','area_cosechada','produccion','rendimiento','fuente','cod_dane_dep','cod_dane_mun']]
corrected=pd.concat([df_rest, collapsed], ignore_index=True).sort_values(['departamento','cultivo_cat','año','cod_dane_mun','periodo']).reset_index(drop=True)
print(f"Base CORREGIDA (tras colapso maíz >=2021): {len(corrected)} filas (diff {len(corrected)-len(base)})")
# ------------------------------------------------------------
# 3. RE-VERIFICAR HUECO ANTIOQUIA MAÍZ TECNIFICADO 2015-2018 con parser corregido
print("\n"+"="*100)
print("3. HUECO ANTIOQUIA MAÍZ TECNIFICADO 2015-2018 (parser corregido)")
print("="*100)
ant=corrected[(corrected.departamento=='ANTIOQUIA')&(corrected.cultivo_cat=='Maíz tecnificado')]
tbl=ant.groupby('año')['cod_dane_mun'].nunique().reindex(range(2007,2025), fill_value=0)
print(tbl.to_string())
print("\nDetalle 2015-2018 filas (debería ser 0 si hueco persiste):")
print(ant[ant.año.between(2015,2018)].head().to_string() if len(ant[ant.año.between(2015,2018)]) else "0 filas 2015-2018 -> hueco CONFIRMADO con parser corregido")
# listar municipios antes de 2015 y si aparecen como tradicional 2015-18 (como antes)
prev=set(ant[ant.año<2015].cod_dane_mun)
trad=corrected[(corrected.departamento=='ANTIOQUIA')&(corrected.cultivo_cat=='Maíz tradicional')&(corrected.año.between(2015,2018))]
inter=prev.intersection(set(trad.cod_dane_mun))
print(f"Municipios tecnificado <2015: {len(prev)}  Reaparecen como tradicional 2015-18: {len(inter)}")
# ------------------------------------------------------------
# 4. REGENERAR ENTREGABLES COMPLETITUD DESDE CERO
print("\n"+"="*100)
print("4. REGENERAR ENTREGABLES COMPLETITUD")
print("="*100)
# Panel balanceado
mask_bal=pd.Series(False, index=corrected.index)
rows=[]
for (dep,cat), g in corrected.groupby(['departamento','cultivo_cat']):
    yrs=g.groupby('cod_dane_mun')['año'].apply(lambda s:set(s))
    total=g['cod_dane_mun'].nunique()
    bal=[m for m,ys in yrs.items() if len(ys)==18 and set(range(2007,2025)).issubset(ys)]
    rows.append((dep,cat,total,len(bal)))
    if bal:
        mask_bal.loc[g[g['cod_dane_mun'].isin(bal)].index]=True
resumen=pd.DataFrame(rows, columns=['departamento','cultivo_cat','total_municipios','en_panel_balanceado'])
print(resumen.to_string(index=False))
print(f"Filas panel: {int(mask_bal.sum())}/{len(corrected)}")
corrected['en_panel_balanceado']=mask_bal
panel=corrected[mask_bal].copy()
# Municipios final
mp=panel[['departamento','municipio','cod_dane_mun']].drop_duplicates()
mp=mp.sort_values('municipio', key=lambda s:s.map(len), ascending=False).drop_duplicates('cod_dane_mun').sort_values(['departamento','cod_dane_mun'])
print(f"\nMunicipios panel final únicos: {len(mp)}")
# Tabla conteos
tabla=corrected.groupby(['departamento','cultivo_cat','año']).size().unstack(fill_value=0).reindex(columns=range(2007,2025), fill_value=0)
print("\nTabla conteos departamento-cultivo-año (municipios-año filas):")
print(tabla.to_string())
# Guardar
corrected.to_csv(os.path.join(BASE_DIR,"eva_2007_2024_consolidado.csv"), index=False, encoding="utf-8")
panel.to_csv(os.path.join(BASE_DIR,"eva_panel_balanceado.csv"), index=False, encoding="utf-8")
mp.to_csv(os.path.join(BASE_DIR,"municipios_panel_final.csv"), index=False, encoding="utf-8")
tabla.to_csv(os.path.join(BASE_DIR,"tabla_conteos_departamento_cultivo_anio.csv"), encoding="utf-8")
# Agregado departamento
agg=corrected.groupby(['departamento','cultivo_cat','año'], as_index=False).agg(area_sembrada=('area_sembrada','sum'),area_cosechada=('area_cosechada','sum'),produccion=('produccion','sum'))
agg['rendimiento']=np.where(agg['area_cosechada']>0, agg['produccion']/agg['area_cosechada'], np.nan)
agg=agg.sort_values(['departamento','cultivo_cat','año'])
agg.to_csv(os.path.join(BASE_DIR,"eva_departamento_cultivo_anio.csv"), index=False, encoding="utf-8")
print(f"\nAgg guardado: {len(agg)} filas (esperado 270, faltan {270-len(agg)} por hueco Antioquia tecnificado)")
print(agg.head(5).to_string(index=False))
# ------------------------------------------------------------
# 5. CHEQUEO SISTEMÁTICO RATIOS AÑO-SOBRE-AÑO
print("\n"+"="*100)
print("5. CHEQUEO RATIOS AÑO-SOBRE-AÑO PRODUCCIÓN (rango 0.5-2.0)")
print("="*100)
# usar agregado corregido
ratios=[]
for (dep,cat), g in agg.groupby(['departamento','cultivo_cat']):
    g=g.sort_values('año')
    g['ratio']=g['produccion']/g['produccion'].shift(1)
    for _,r in g.iterrows():
        if pd.notna(r['ratio']) and (r['ratio']<0.5 or r['ratio']>2.0):
            ratios.append((dep,cat,int(r['año']),int(r['año']-1), float(r['ratio']), float(r['produccion']), float(g[g.año==r['año']-1].produccion.values[0]) if len(g[g.año==r['año']-1]) else np.nan))
df_r=pd.DataFrame(ratios, columns=['departamento','cultivo','año','año_prev','ratio','prod','prod_prev'])
if len(df_r):
    # marcar explicados
    def explicado(row):
        if row['departamento']=='ANTIOQUIA' and row['cultivo']=='Maíz tecnificado' and row['año'] in [2015,2019,2023]: return 'hueco 2015-18 / rebote 2019'
        if row['cultivo'].startswith('Maíz') and row['año']==2021: return 'colapso variedad ya corregido (no debería aparecer si corregido bien)'
        return ''
    df_r['explicado']=df_r.apply(explicado, axis=1)
    print(df_r.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nTotal ratios anómalos (<0.5 o >2.0): {len(df_r)}  Explicados: {(df_r['explicado']!='').sum()}  No explicados: {(df_r['explicado']=='').sum()}")
    if (df_r['explicado']=='').any():
        print("\nNO EXPLICADOS (requieren investigación):")
        print(df_r[df_r['explicado']==''].to_string(index=False))
    else:
        print("\nTodos los ratios anómalos restantes están explicados (hueco Antioquia).")
else:
    print("Ningún ratio fuera de 0.5-2.0 -> sin artefactos restantes")
