import pandas as pd, numpy as np, os
base=r"C:\Proyecto Semillero"
cons=pd.read_csv(os.path.join(base,'eva_2007_2024_consolidado.csv'),dtype=str,encoding='utf-8')
def to_num(s):
    if s is None: return np.nan
    s=str(s).strip().replace(',','.')
    if s=='' or s.lower()=='nan': return np.nan
    try: return float(s)
    except: return np.nan
cons['año']=pd.to_numeric(cons['año'],errors='coerce').astype(int)

# --- Diagnostico Tarea 1: por que se duplica el conteo 2021? revisar subcategoria y filas por municipio ---
print("=== HUILA Maiz tradicional: subcategorias distintas por año (2020 vs 2021) ===")
for y in [2020,2021]:
    sub=cons[(cons['departamento']=='HUILA')&(cons['cultivo_cat']=='Maíz tradicional')&(cons['año']==y)]
    print(f" año {y}: n filas={len(sub)}, municipios unicos={sub['municipio'].nunique()}")
    print("   subcategorias:", sorted(sub['subcategoria'].dropna().unique().tolist()))
    # filas promedio por municipio
    print("   filas/municipio (media):", round(len(sub)/sub['municipio'].nunique(),2))
    print("   municipios con >1 fila:", (sub.groupby('municipio').size()>1).sum())

print("\n=== HUILA Café: subcategorias 2020 vs 2021 (control, no deberia cambiar) ===")
for y in [2020,2021]:
    sub=cons[(cons['departamento']=='HUILA')&(cons['cultivo_cat']=='Café')&(cons['año']==y)]
    print(f" año {y}: n filas={len(sub)}, municipios={sub['municipio'].nunique()}, subcats={sorted(sub['subcategoria'].dropna().unique().tolist())}")

# --- Diagnostico Tarea 3: porque panel balanceado = 0? revisar solapamiento 2018 vs 2019 ---
print("\n=== Solapamiento municipios 2018 vs 2019 (HUILA, cada cultivo_cat) ===")
for cat in ['Café','Cacao','Frijol','Maíz tradicional','Maíz tecnificado']:
    g=cons[cons['departamento']=='HUILA']
    m18=set(g[(g['cultivo_cat']==cat)&(g['año']==2018)]['municipio'])
    m19=set(g[(g['cultivo_cat']==cat)&(g['año']==2019)]['municipio'])
    print(f" {cat:18s}: 2018={len(m18)}  2019={len(m19)}  interseccion={len(m18&m19)}")

print("\n=== Distribucion de # años por municipio (HUILA Café) ===")
g=cons[(cons['departamento']=='HUILA')&(cons['cultivo_cat']=='Café')]
ny=g.groupby('municipio')['año'].nunique()
print(" max años:", ny.max(), " | # municipios con 18 años:", (ny==18).sum(), " | top:", ny.sort_values(ascending=False).head(5).to_dict())
print(" años presentes globalmente para HUILA Café:", sorted(g['año'].unique().tolist()))
