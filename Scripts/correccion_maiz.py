import pandas as pd, numpy as np, glob, os, unicodedata

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 400)
pd.set_option('display.max_colwidth', 50)

base = r"C:\Proyecto Semillero"
A_PATH = glob.glob(os.path.join(base, "*Base_Agr*cola*.csv"))[0]
B_PATH = glob.glob(os.path.join(base, "*2007-2018*"))[0]

def norm(s):
    if s is None: return ''
    return ''.join(c for c in unicodedata.normalize('NFKD', str(s)) if not unicodedata.combining(c)).upper()

def to_num(s):
    if s is None: return np.nan
    s = str(s).strip()
    if s == '' or s.lower() == 'nan': return np.nan
    # Formato colombiano: '.' = separador miles, ',' = decimal. Si hay ',', eliminar '.' miles y ',' -> '.'
    if ',' in s:
        s = s.replace('.','').replace(',','.')
    try: return float(s)
    except: return np.nan

print("="*100)
print("TAREA 1 - RECONSTRUCCION BASE CON PERIODO (ANTES DEL COLAPSO)")
print("="*100)
# ---- Cargar A ----
dfA_raw = pd.read_csv(A_PATH, dtype=str, encoding='utf-8')
MAP_A = {
    'Departamento':'departamento','Municipio':'municipio','Cultivo':'cultivo',
    'Desagregación cultivo':'subcategoria','Año':'año','Periodo':'periodo',
    'Área sembrada':'area_sembrada','Área cosechada':'area_cosechada',
    'Producción':'produccion','Rendimiento':'rendimiento',
    'Código Dane departamento':'cod_dane_dep','Código Dane municipio':'cod_dane_mun',
}
dfA = dfA_raw.rename(columns=MAP_A)
for c in ['area_sembrada','area_cosechada','produccion','rendimiento']:
    dfA[c]=dfA[c].map(to_num)
dfA['año']=pd.to_numeric(dfA['año'], errors='coerce')
dfA['periodo']=dfA['periodo'].astype(str).str.strip()
dfA['fuente']='A_datosgov_2019_2025'

# ---- Cargar B ----
dfB_raw = pd.read_excel(B_PATH, header=1, dtype=str)
# Ver columnas reales de B para mapear periodo correctamente
MAP_B = {
    'DEPARTAMENTO':'departamento','MUNICIPIO':'municipio','CULTIVO':'cultivo',
    'DESAGREGACIÓN REGIONAL Y/O SISTEMA PRODUCTIVO':'subcategoria','AÑO':'año','PERIODO':'periodo',
    'Área Sembrada\n(ha)':'area_sembrada','Área Cosechada\n(ha)':'area_cosechada',
    'Producción\n(t)':'produccion','Rendimiento\n(t/ha)':'rendimiento',
    'CÓD. \nDEP.':'cod_dane_dep','CÓD. MUN.':'cod_dane_mun',
}
dfB = dfB_raw.rename(columns=MAP_B)
for c in ['area_sembrada','area_cosechada','produccion','rendimiento']:
    dfB[c]=dfB[c].map(to_num)
dfB['año']=pd.to_numeric(dfB['año'], errors='coerce')
dfB['periodo']=dfB['periodo'].astype(str).str.strip()
dfB['fuente']='B_agronet_2007_2018'

# Normalizar
for df in (dfA, dfB):
    df['departamento']=df['departamento'].astype(str).str.strip().str.upper()
    df['municipio']=df['municipio'].astype(str).str.strip().str.upper()
    df['cod_dane_mun']=df['cod_dane_mun'].astype(str).str.strip().str.zfill(5)
    df['cod_dane_dep']=df['cod_dane_dep'].astype(str).str.strip().str.zfill(2)

CROPS = {'CAFE':'Café','CACAO':'Cacao','FRIJOL':'Frijol','MAIZ':'Maíz'}
def cultivo_cat_fn(row):
    c=norm(row['cultivo']); d=norm(row['subcategoria'])
    if c=='MAIZ':
        if 'TECNIFICADO' in d: return 'Maíz tecnificado'
        if 'TRADICIONAL' in d: return 'Maíz tradicional'
        return 'Maíz'
    if c in CROPS: return CROPS[c]
    return None

for df in (dfA, dfB):
    df['cultivo_cat']=df.apply(cultivo_cat_fn, axis=1)

DEPS=['HUILA','ANTIOQUIA','TOLIMA']
maskA = dfA['departamento'].isin(DEPS) & dfA['cultivo_cat'].isin(['Café','Cacao','Frijol','Maíz tecnificado','Maíz tradicional'])
maskB = dfB['departamento'].isin(DEPS) & dfB['cultivo_cat'].isin(['Café','Cacao','Frijol','Maíz tecnificado','Maíz tradicional'])

fA=dfA[maskA].copy(); fB=dfB[maskB].copy()
cols=['departamento','municipio','cultivo','subcategoria','cultivo_cat','año','periodo','area_sembrada','area_cosechada','produccion','rendimiento','fuente','cod_dane_dep','cod_dane_mun']
base_df=pd.concat([fA[cols], fB[cols]], ignore_index=True)
base_df=base_df[(base_df['año']>=2007)&(base_df['año']<=2024)].copy()
base_df['año']=base_df['año'].astype(int)
print(f"Base reconstruida (antes del colapso): {len(base_df)} filas")
print(f"  A filas: {len(fA)} -> filtrada 2007-2024: {len(base_df[base_df['fuente']=='A_datosgov_2019_2025'])}")
print(f"  B filas: {len(fB)} -> filtrada: {len(base_df[base_df['fuente']=='B_agronet_2007_2018'])}")
print(f"  Periodos únicos: {sorted(base_df['periodo'].dropna().unique())[:10]} ... total {base_df['periodo'].nunique()}")

# Contar por categoria antes
print("\n--- Conteo filas por cultivo_cat ANTES del colapso ---")
print(base_df['cultivo_cat'].value_counts().to_string())
print("\n--- Conteo por año para maíz >=2021 ANTES ---")
print(base_df[(base_df['cultivo_cat'].isin(['Maíz tecnificado','Maíz tradicional']))&(base_df['año']>=2021)].groupby(['año','cultivo_cat']).size().to_string())

# ============ TAREA 1 - COLAPSO MAIZ >=2021 ============
print("\n"+"="*100)
print("TAREA 1 - COLAPSO MAIZ >=2021 por (cod_dane_mun, cultivo_cat, año, periodo)")
print("="*100)
cond_maiz_ge = base_df['cultivo_cat'].isin(['Maíz tecnificado','Maíz tradicional']) & (base_df['año']>=2021)
df_maiz_ge = base_df[cond_maiz_ge].copy()
df_rest = base_df[~cond_maiz_ge].copy()
print(f"Filas maíz >=2021 ANTES: {len(df_maiz_ge)}")
print(f"Filas resto (no tocado): {len(df_rest)}")
print(f"  de las cuales maíz <2021: {len(base_df[(base_df['cultivo_cat'].isin(['Maíz tecnificado','Maíz tradicional']))&(base_df['año']<2021)])}")
print(f"  café/cacao/fríjol total: {len(base_df[base_df['cultivo_cat'].isin(['Café','Cacao','Frijol'])])}")

# Agrupar maíz >=2021
group_keys=['cod_dane_mun','cultivo_cat','año','periodo']
# Para verificar después, guardar detalle de grupos con 2 filas
g_sizes=df_maiz_ge.groupby(group_keys).size()
print(f"\nDistribución de filas por grupo (maíz >=2021):")
print(g_sizes.value_counts().sort_index().to_string())
print(f"Grupos con 2 filas (amarillo+blanco): {(g_sizes==2).sum()}  con 1 fila: {(g_sizes==1).sum()}")

collapsed = df_maiz_ge.groupby(group_keys, as_index=False, dropna=False).agg(
    departamento=('departamento','first'),
    municipio=('municipio','first'),
    cultivo=('cultivo','first'),
    area_sembrada=('area_sembrada','sum'),
    area_cosechada=('area_cosechada','sum'),
    produccion=('produccion','sum'),
    fuente=('fuente','first'),
    cod_dane_dep=('cod_dane_dep','first')
)
# subcategoria colapsada a generica (cultivo_cat) y rendimiento recalculado
collapsed['subcategoria'] = collapsed['cultivo_cat']
collapsed['rendimiento'] = np.where(collapsed['area_cosechada']>0, collapsed['produccion']/collapsed['area_cosechada'], np.nan)
# reordenar columnas para que coincida con base_df
collapsed = collapsed[['departamento','municipio','cultivo','subcategoria','cultivo_cat','año','periodo','area_sembrada','area_cosechada','produccion','rendimiento','fuente','cod_dane_dep','cod_dane_mun']]
print(f"Filas maíz >=2021 DESPUÉS del colapso: {len(collapsed)}")
print(f"Reducción: {len(df_maiz_ge)} -> {len(collapsed)}  (ahorro {len(df_maiz_ge)-len(collapsed)} filas)")
# Verificar que nunca promediamos tasas: ejemplo diferencia
test_groups = df_maiz_ge.groupby(group_keys).size()
print(f"Grupos colapsados: {len(collapsed)} (de {len(test_groups)} grupos originales)")

# Unir
corrected = pd.concat([df_rest, collapsed], ignore_index=True)
corrected = corrected.sort_values(['departamento','cultivo_cat','año','cod_dane_mun','periodo']).reset_index(drop=True)
print(f"\nBase CORREGIDA total filas: {len(corrected)}  (antes {len(base_df)}, diff {len(corrected)-len(base_df)})")

# Verificar que resto no cambió
print("\n--- Verificación: conteo por cultivo_cat DESPUÉS ---")
print(corrected['cultivo_cat'].value_counts().to_string())
# café/cacao/fríjol no tocados
for cat in ['Café','Cacao','Frijol']:
    antes=len(base_df[base_df['cultivo_cat']==cat])
    desp=len(corrected[corrected['cultivo_cat']==cat])
    print(f"  {cat}: antes={antes} después={desp} {'OK' if antes==desp else 'CAMBIO INESPERADO'}")
# maíz <2021
antes_m_lt=len(base_df[(base_df['cultivo_cat'].isin(['Maíz tecnificado','Maíz tradicional']))&(base_df['año']<2021)])
desp_m_lt=len(corrected[(corrected['cultivo_cat'].isin(['Maíz tecnificado','Maíz tradicional']))&(corrected['año']<2021)])
print(f"  Maíz <2021: antes={antes_m_lt} después={desp_m_lt} {'OK' if antes_m_lt==desp_m_lt else 'ERROR'}")

# ============ 5 EJEMPLOS CONCRETOS ============
print("\n"+"="*100)
print("5 EJEMPLOS CONCRETOS: antes (amarillo+blanco) vs después (colapsado) + verificacion suma manual")
print("="*100)
# Elegir 5 grupos con 2 filas al azar (primeros 5 con 2)
groups_2 = g_sizes[g_sizes==2].index.tolist()[:5]
for idx, key in enumerate(groups_2, 1):
    cod, cat, anio, per = key
    orig = df_maiz_ge[(df_maiz_ge['cod_dane_mun']==cod)&(df_maiz_ge['cultivo_cat']==cat)&(df_maiz_ge['año']==anio)&(df_maiz_ge['periodo']==per)]
    coll = collapsed[(collapsed['cod_dane_mun']==cod)&(collapsed['cultivo_cat']==cat)&(collapsed['año']==anio)&(collapsed['periodo']==per)].iloc[0]
    print(f"\n--- EJEMPLO {idx}: {cod} {cat} {anio} {per}  municipio={orig['municipio'].iloc[0]} dep={orig['departamento'].iloc[0]} ---")
    print("ANTES (2 filas originales):")
    print(orig[['subcategoria','periodo','area_sembrada','area_cosechada','produccion','rendimiento']].to_string(index=False))
    suma_s = orig['area_sembrada'].sum()
    suma_c = orig['area_cosechada'].sum()
    suma_p = orig['produccion'].sum()
    prom_rend = orig['rendimiento'].mean()
    recalc = suma_p/suma_c if suma_c>0 else np.nan
    print(f"  Suma manual: area_s={suma_s:.2f} area_c={suma_c:.2f} produccion={suma_p:.2f}  rendimiento recalc=sum/prod={recalc:.4f}  promedio tasas (ERRÓNEO)={prom_rend:.4f}")
    print("DESPUÉS (1 fila colapsada):")
    print(pd.DataFrame([coll])[['subcategoria','periodo','area_sembrada','area_cosechada','produccion','rendimiento']].to_string(index=False))
    ok_s = np.isclose(coll['area_sembrada'], suma_s, atol=1e-6)
    ok_c = np.isclose(coll['area_cosechada'], suma_c, atol=1e-6)
    ok_p = np.isclose(coll['produccion'], suma_p, atol=1e-6)
    ok_r = np.isclose(coll['rendimiento'], recalc, atol=1e-6) if pd.notna(recalc) else pd.isna(coll['rendimiento'])
    print(f"  Verificación: area_s {ok_s}  area_c {ok_c}  prod {ok_p}  rend {ok_r}  -> {'OK' if all([ok_s,ok_c,ok_p,ok_r]) else 'FALLO'}")
    if not np.isclose(prom_rend, recalc, atol=1e-4):
        print(f"  NOTA: promediar tasas ({prom_rend:.4f}) != dividir sumas ({recalc:.4f})  -> confirma que promediar es erróneo.")

# ============ RECALCULAR PANEL BALANCEADO SOBRE BASE CORREGIDA ============
print("\n"+"="*100)
print("RECALCULO PANEL BALANCEADO (sobre base corregida, clave cod_dane_mun, 18 años 2007-2024)")
print("="*100)
mask_bal=pd.Series(False, index=corrected.index)
res=[]
for (dep,cat), g in corrected.groupby(['departamento','cultivo_cat']):
    yrs=g.groupby('cod_dane_mun')['año'].apply(lambda s:set(s))
    total_mun=g['cod_dane_mun'].nunique()
    bal=[m for m,ys in yrs.items() if len(ys)==18 and set(range(2007,2025)).issubset(ys)]
    res.append((dep,cat,total_mun,len(bal)))
    if bal:
        idx=g[g['cod_dane_mun'].isin(bal)].index
        mask_bal.loc[idx]=True
resumen=pd.DataFrame(res, columns=['departamento','cultivo_cat','total_municipios','en_panel_balanceado'])
print(resumen.to_string(index=False))
print(f"\nFilas en panel balanceado: {int(mask_bal.sum())} de {len(corrected)}")
corrected['en_panel_balanceado']=mask_bal
panel=corrected[mask_bal].copy()

# Guardar corregidos
corrected.to_csv(os.path.join(base,'eva_2007_2024_consolidado.csv'), index=False, encoding='utf-8')
panel.to_csv(os.path.join(base,'eva_panel_balanceado.csv'), index=False, encoding='utf-8')
print("Guardados eva_2007_2024_consolidado.csv (corregido) y eva_panel_balanceado.csv (recalculado)")

# ============ TAREA 2 - AGREGACION DEPARTAMENTO-CULTIVO-AÑO ============
print("\n"+"="*100)
print("TAREA 2 - eva_departamento_cultivo_anio.csv (agregado sobre corregido, todos los datos)")
print("="*100)
agg = corrected.groupby(['departamento','cultivo_cat','año'], dropna=False).agg(
    area_sembrada=('area_sembrada','sum'),
    area_cosechada=('area_cosechada','sum'),
    produccion=('produccion','sum')
).reset_index()
agg['rendimiento'] = np.where(agg['area_cosechada']>0, agg['produccion']/agg['area_cosechada'], np.nan)
agg = agg.sort_values(['departamento','cultivo_cat','año'])
agg.to_csv(os.path.join(base,'eva_departamento_cultivo_anio.csv'), index=False, encoding='utf-8')
print(f"Filas agregadas: {len(agg)}  (3 deps × 5 cultivos × 18 años = 270 teóricas, obtenidas {len(agg)})")
print(agg.head(10).to_string(index=False))

# Matriz de completitud agregada
print("\n--- MATRIZ COMPLETITUD AGREGADA (departamento×cultivo×año) ---")
# pivot de produccion
pivot_prod = agg.pivot_table(index=['departamento','cultivo_cat'], columns='año', values='produccion', aggfunc='sum')
pivot_prod = pivot_prod.reindex(columns=range(2007,2025))
print("Produccion total por dept-cultivo-año (t):")
print(pivot_prod.to_string(float_format=lambda x: f"{x:,.0f}" if pd.notna(x) else " -"))
# presencia binaria
pivot_pres = agg.pivot_table(index=['departamento','cultivo_cat'], columns='año', values='area_sembrada', aggfunc='count')
pivot_pres = pivot_pres.reindex(columns=range(2007,2025))
# contar años con dato
for (dep,cat), row in pivot_pres.iterrows():
    presentes=int(row.notna().sum())
    print(f"  {dep:10s} {cat:18s}: {presentes}/18 años con dato")

# Señalar producciones anormalmente bajas
print("\n--- ALERTA: producciones anormalmente bajas en agregado ---")
for (dep,cat), g in agg.groupby(['departamento','cultivo_cat']):
    vals=g.set_index('año')['produccion']
    # definir umbral: producción < 10% de la mediana (excluyendo ceros/nulos) o < 20% del max
    med=vals.median()
    if pd.isna(med) or med==0:
        continue
    for y, prod in vals.items():
        if pd.isna(prod): continue
        if prod < med*0.10:  # <10% mediana
            print(f"  ALERTA {dep} {cat} {y}: produccion={prod:,.1f} t  mediana={med:,.1f}  ({prod/med*100:.1f}% de mediana)")
# también señalar años faltantes (sin fila)
print("\n--- AÑOS FALTANTES (sin fila agregada) por dept-cultivo ---")
for (dep,cat) in agg[['departamento','cultivo_cat']].drop_duplicates().itertuples(index=False):
    presentes=set(agg[(agg['departamento']==dep)&(agg['cultivo_cat']==cat)]['año'])
    faltantes=sorted(set(range(2007,2025))-presentes)
    if faltantes:
        print(f"  {dep} {cat}: faltan {faltantes}")

print("\n--- Verificación rendimiento agregado (no promediar tasas) ejemplo ---")
# tomar un ejemplo agregado y mostrar que promediar rendimientos municipales daría distinto
ex = corrected[(corrected['departamento']=='HUILA')&(corrected['cultivo_cat']=='Maíz tradicional')&(corrected['año']==2021)]
# produccion total / area total vs mean(rendimiento)
prod_tot=ex['produccion'].sum(); area_tot=ex['area_cosechada'].sum(); rend_recalc=prod_tot/area_tot if area_tot>0 else np.nan
rend_mean=ex['rendimiento'].mean()
print(f" HUILA Maíz tradicional 2021: prod_tot={prod_tot:,.1f} area_tot={area_tot:,.1f} rend_recalc={rend_recalc:.4f}  mean(rend_municipal)={rend_mean:.4f}  diff={abs(rend_recalc-rend_mean):.4f} -> promediar tasas es erróneo")
