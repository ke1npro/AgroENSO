import pandas as pd
import numpy as np
import unicodedata, glob, os

base = r"C:\Proyecto Semillero"
A_PATH = glob.glob(os.path.join(base, "*Base_Agr*cola*.csv"))[0]
B_PATH = glob.glob(os.path.join(base, "*2007-2018*"))[0]

def norm(s):
    if s is None: return ''
    return ''.join(c for c in unicodedata.normalize('NFKD', str(s)) if not unicodedata.combining(c)).upper()

def to_num(s):
    if s is None: return np.nan
    s = str(s).strip().replace(',', '.')
    if s == '' or s.lower() == 'nan': return np.nan
    try: return float(s)
    except Exception: return np.nan

# ========================= PASO 1 + 2: CARGA Y MAPEO =========================
# --- Archivo A ---
dfA = pd.read_csv(A_PATH, dtype=str, encoding='utf-8')
MAP_A = {
    'Departamento':'departamento','Municipio':'municipio','Cultivo':'cultivo',
    'Desagregación cultivo':'subcategoria','Año':'año',
    'Área sembrada':'area_sembrada','Área cosechada':'area_cosechada',
    'Producción':'produccion','Rendimiento':'rendimiento',
    'Código Dane departamento':'cod_dane_dep','Código Dane municipio':'cod_dane_mun',
}
dfA = dfA.rename(columns=MAP_A)
numcols = ['area_sembrada','area_cosechada','produccion','rendimiento']
for c in numcols: dfA[c] = dfA[c].map(to_num)
dfA['año'] = pd.to_numeric(dfA['año'], errors='coerce')
dfA['fuente'] = 'A_datosgov_2019_2025'

# --- Archivo B ---
dfB = pd.read_excel(B_PATH, header=1, dtype=str)
MAP_B = {
    'DEPARTAMENTO':'departamento','MUNICIPIO':'municipio','CULTIVO':'cultivo',
    'DESAGREGACIÓN REGIONAL Y/O SISTEMA PRODUCTIVO':'subcategoria','AÑO':'año',
    'Área Sembrada\n(ha)':'area_sembrada','Área Cosechada\n(ha)':'area_cosechada',
    'Producción\n(t)':'produccion','Rendimiento\n(t/ha)':'rendimiento',
    'CÓD. \nDEP.':'cod_dane_dep','CÓD. MUN.':'cod_dane_mun',
}
dfB = dfB.rename(columns=MAP_B)
for c in numcols: dfB[c] = dfB[c].map(to_num)
dfB['año'] = pd.to_numeric(dfB['año'], errors='coerce')
dfB['fuente'] = 'B_agronet_2007_2018'

print("MAPEO FINAL USADO:")
print(" A ->", MAP_A)
print(" B ->", MAP_B)

# ========================= PASO 3: ETIQUETADO Y FILTRO =========================
DEPS = ['HUILA','ANTIOQUIA','TOLIMA']
CROPS = {'CAFE':'Café','CACAO':'Cacao','FRIJOL':'Frijol','MAIZ':'Maíz'}

def cultiv_cat(row):
    c = norm(row['cultivo']); d = norm(row['subcategoria'])
    if c == 'MAIZ':
        if 'TECNIFICADO' in d: return 'Maíz tecnificado'
        if 'TRADICIONAL' in d: return 'Maíz tradicional'
        return 'Maíz'
    if c in CROPS: return CROPS[c]
    return None

for df in (dfA, dfB):
    df['departamento'] = df['departamento'].astype(str).str.strip().str.upper()
    df['cultivo_cat'] = df.apply(cultiv_cat, axis=1)

# Mostrar valores unicos en columna cultivo/subcategoria ANTES de filtrar (evidencia)
print("\n--- Valores unicos 'cultivo_cat' resultantes (A y B combinados) ---")
tmp = pd.concat([dfA[['cultivo_cat']], dfB[['cultivo_cat']]], ignore_index=True)
print(sorted([x for x in tmp['cultivo_cat'].dropna().unique() if x is not None]))

maskA = dfA['departamento'].isin(DEPS) & dfA['cultivo_cat'].notna()
maskB = dfB['departamento'].isin(DEPS) & dfB['cultivo_cat'].notna()
fA = dfA[maskA].copy(); fB = dfB[maskB].copy()
print(f"\nFilas A filtradas: {len(fA)} | Filas B filtradas: {len(fB)}")
print("Años A filtrado:", sorted(fA['año'].dropna().unique().astype(int).tolist()))
print("Años B filtrado:", sorted(fB['año'].dropna().unique().astype(int).tolist()))

# ========================= PASO 4: CONCAT + TABLA DE CONTEOS =========================
cols = ['departamento','municipio','cultivo','subcategoria','cultivo_cat','año',
        'area_sembrada','area_cosechada','produccion','rendimiento','fuente',
        'cod_dane_dep','cod_dane_mun']
cons = pd.concat([fA[cols], fB[cols]], ignore_index=True)
# años canonical 2007-2024 (18 años)
cons = cons[(cons['año']>=2007) & (cons['año']<=2024)].copy()
cons['año'] = cons['año'].astype(int)
cons.to_csv(os.path.join(base,'eva_2007_2024_consolidado.csv'), index=False, encoding='utf-8')
print(f"\nConsolidado escrito: eva_2007_2024_consolidado.csv  ({len(cons)} filas)")

tabla = cons.groupby(['departamento','cultivo_cat','año']).size().unstack(fill_value=0)
tabla = tabla.reindex(columns=range(2007,2025), fill_value=0)
print("\n========== TABLA CRUDA DE CONTEOS (departamento x cultivo x año) ==========")
print(tabla.to_string())

# años con al menos un registro y alerta <15 de 18
anios_tot = 18
resumen = []
for (dep, cul), row in tabla.iterrows():
    presentes = int((row > 0).sum())
    alerta = "ALERTA" if presentes < 15 else "ok"
    resumen.append((dep, cul, presentes, alerta))
resumen = pd.DataFrame(resumen, columns=['departamento','cultivo','años_con_registro','estado'])
print("\n---------- Resumen completitud (de 18 años 2007-2024) ----------")
print(resumen.to_string(index=False))
print("\nCombinaciones en ALERTA (<15 años):")
print(resumen[resumen['estado']=="ALERTA"].to_string(index=False))

tabla.to_csv(os.path.join(base,'tabla_conteos_departamento_cultivo_anio.csv'), encoding='utf-8')

# ========================= PASO 5: TRANSICION 2018->2019 (Antioquia maíz tecnificado) =========================
mt = cons[(cons['departamento']=='ANTIOQUIA') & (cons['cultivo_cat']=='Maíz tecnificado')].copy()
print("\n========== PASO 5: Municipios reportantes MAIZ TECNIFICADO en ANTIOQUIA por año (2015-2022) ==========")
p5 = mt[mt['año'].between(2015,2022)].groupby('año')['municipio'].nunique()
for y in range(2015,2023):
    print(f"  {y}: {int(p5.get(y,0))} municipios")
print("Diferencia 2018->2019:", int(p5.get(2019,0)) - int(p5.get(2018,0)),
      "(quiebre real)" if abs(int(p5.get(2019,0))-int(p5.get(2018,0)))>0 else "(sin quiebre aparente)")

# ========================= PASO 6: CEROS vs NULOS en area cosechada =========================
print("\n========== PASO 6: área cosechada = 0 vs nula (consolidado filtrado) ==========")
ac = cons['area_cosechada']
cero = int((ac==0).sum()); nulo = int(ac.isna().sum()); positivo = int((ac>0).sum())
print(f"  Filas con area_cosechada == 0 : {cero}")
print(f"  Filas con area_cosechada nula : {nulo}")
print(f"  Filas con area_cosechada >  0 : {positivo}")
# recalcular rendimiento excluyendo div por cero y NaN
cons['rend_calc'] = np.where((cons['area_cosechada']>0) & cons['area_cosechada'].notna(),
                             cons['produccion']/cons['area_cosechada'], np.nan)
div_zero = int(((cons['area_cosechada']==0) & cons['produccion'].notna()).sum())
print(f"  Filas donde produccion/area_cosechada habria division por cero (area=0): {div_zero}")
print(f"  Valores NaN en rend_calc (excluidos correctamente): {int(cons['rend_calc'].isna().sum())}")
# comparar con rendimiento provisto
cmp = cons.dropna(subset=['rendimiento','rend_calc'])
dif = (cmp['rendimiento']-cmp['rend_calc']).abs()
print(f"  Max diferencia |rendimiento provisto - recalculado| (filas con ambos): {dif.max():.4f}")
print("  => La division por cero se excluye correctamente (rend_calc NaN cuando area_cosechada<=0).")
