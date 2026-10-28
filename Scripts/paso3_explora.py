import pandas as pd
import numpy as np
import glob, os

pd.set_option('display.max_rows', None)
pd.set_option('display.width', 250)

base = r"C:\Proyecto Semillero"
A_PATH = glob.glob(os.path.join(base, "*Base_Agr*cola*.csv"))[0]
B_PATH = glob.glob(os.path.join(base, "*2007-2018*"))[0]

# ---------- Cargar A ----------
dfA = pd.read_csv(A_PATH, dtype=str, encoding='utf-8')
# ---------- Cargar B (header en fila 1, 0-indexed) ----------
dfB = pd.read_excel(B_PATH, header=1, dtype=str)

def to_num(s):
    if s is None:
        return np.nan
    s = str(s).strip().replace(',', '.')
    if s == '' or s.lower() == 'nan':
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan

# Limpiar numericas A
for col in ['Área sembrada','Área cosechada','Producción','Rendimiento']:
    dfA[col] = dfA[col].map(to_num)
dfA['Año'] = pd.to_numeric(dfA['Año'], errors='coerce')

# Limpiar numericas B
for col in ['Área Sembrada\n(ha)','Área Cosechada\n(ha)','Producción\n(t)','Rendimiento\n(t/ha)']:
    dfB[col] = dfB[col].map(to_num)
dfB['AÑO'] = pd.to_numeric(dfB['AÑO'], errors='coerce')

print("=== A shape:", dfA.shape, " | años:", sorted(dfA['Año'].dropna().unique().astype(int).tolist()))
print("=== B shape:", dfB.shape, " | años:", sorted(dfB['AÑO'].dropna().unique().astype(int).tolist()))

# Unicos de columnas de cultivo en A y B
print("\n=== A: unique 'Cultivo' (muestra que contiene MAIZ/CAFE/CACAO/FRIJOL) ===")
for v in dfA['Cultivo'].dropna().unique():
    if any(k in str(v).upper() for k in ['MAIZ','MAÍZ','CAFE','CACAO','FRIJOL','FRÍJOL']):
        print("  Cultivo:", repr(v))
print("\n=== A: unique 'Desagregación cultivo' (filtro maiz/cafe...) ===")
for v in dfA['Desagregación cultivo'].dropna().unique():
    if any(k in str(v).upper() for k in ['MAIZ','MAÍZ','CAFE','CACAO','FRIJOL','FRÍJOL','TECNIF','TRADIC']):
        print("  Desagr:", repr(v))

print("\n=== B: unique 'CULTIVO' (maiz/cafe/cacao/frijol) ===")
for v in dfB['CULTIVO'].dropna().unique():
    if any(k in str(v).upper() for k in ['MAIZ','MAÍZ','CAFE','CACAO','FRIJOL','FRÍJOL']):
        print("  Cultivo:", repr(v))
print("\n=== B: unique 'DESAGREGACIÓN REGIONAL Y/O SISTEMA PRODUCTIVO' (maiz...) ===")
for v in dfB['DESAGREGACIÓN REGIONAL Y/O SISTEMA PRODUCTIVO'].dropna().unique():
    if any(k in str(v).upper() for k in ['MAIZ','MAÍZ','CAFE','CACAO','FRIJOL','FRÍJOL','TECNIF','TRADIC']):
        print("  Desagr:", repr(v))
