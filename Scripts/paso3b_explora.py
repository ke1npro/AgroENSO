import pandas as pd
import unicodedata
import glob, os

base = r"C:\Proyecto Semillero"
A_PATH = glob.glob(os.path.join(base, "*Base_Agr*cola*.csv"))[0]
B_PATH = glob.glob(os.path.join(base, "*2007-2018*"))[0]

def norm(s):
    if s is None: return ''
    return ''.join(c for c in unicodedata.normalize('NFKD', str(s)) if not unicodedata.combining(c)).upper()

dfA = pd.read_csv(A_PATH, dtype=str, encoding='utf-8')
dfB = pd.read_excel(B_PATH, header=1, dtype=str)

print("=== A: TODOS los Cultivo unicos (normalized show) ===")
for v in sorted(dfA['Cultivo'].dropna().unique(), key=norm):
    print("  ", repr(v))
print("\n=== A: Departamento unicos (los 3 de interes) ===")
for v in dfA['Departamento'].dropna().unique():
    if norm(v) in ['HUILA','ANTIOQUIA','TOLIMA']:
        print("  ", repr(v))

print("\n=== B: TODOS los CULTIVO unicos ===")
for v in sorted(dfB['CULTIVO'].dropna().unique(), key=norm):
    if norm(v) in ['CAFE','CACAO','FRIJOL','MAIZ','MAIZ FORRAJERO']:
        print("  ", repr(v))
print("\n=== B: DESAGREGACION unicos para MAIZ ===")
for v in dfB['DESAGREGACIÓN REGIONAL Y/O SISTEMA PRODUCTIVO'].dropna().unique():
    if 'MAIZ' in norm(v):
        print("  ", repr(v))
print("\n=== B: Departamento unicos (los 3) ===")
for v in dfB['DEPARTAMENTO'].dropna().unique():
    if norm(v) in ['HUILA','ANTIOQUIA','TOLIMA']:
        print("  ", repr(v))
