import pandas as pd
import numpy as np
import glob, os, unicodedata

base = r"C:\Proyecto Semillero"
cons = pd.read_csv(os.path.join(base,'eva_2007_2024_consolidado.csv'), dtype=str, encoding='utf-8')
def to_num(s):
    if s is None: return np.nan
    s=str(s).strip().replace(',','.')
    if s=='' or s.lower()=='nan': return np.nan
    try: return float(s)
    except: return np.nan
for c in ['area_sembrada','area_cosechada','produccion','rendimiento']:
    cons[c]=cons[c].map(to_num)
cons['año']=pd.to_numeric(cons['año'],errors='coerce').astype(int)
cons['cod_dane_mun']=cons['cod_dane_mun'].astype(str).str.strip().str.zfill(5)
cons['municipio']=cons['municipio'].astype(str).str.strip().str.upper()
maiz_tech=['Maíz tecnificado','Maíz tradicional']

# ---- Diagnostico Tarea 1: causa del salto 2021 (subcategoria) ----
print("=== DIAG T1: filas/municipio y subcategorias HUILA Maiz tradicional 2020 vs 2021 ===")
for y in [2020,2021]:
    sub=cons[(cons['departamento']=='HUILA')&(cons['cultivo_cat']=='Maíz tradicional')&(cons['año']==y)]
    print(f" {y}: filas={len(sub)} mun={sub['municipio'].nunique()} subcats={sorted(sub['subcategoria'].dropna().unique().tolist())} filas/mun={len(sub)/sub['municipio'].nunique():.2f}")


# ============ TAREA 1: SALTO 2021 MAIZ (HUILA, TOLIMA) ============
print("="*90)
print("TAREA 1 - Municipios 'nuevos' 2021 vs 2020 (maiz, Huila+Tolima)")
print("="*90)
for dep in ['HUILA','TOLIMA']:
    for cat in maiz_tech:
        sub=cons[(cons['departamento']==dep)&(cons['cultivo_cat']==cat)]
        m20=set(sub[sub['año']==2020]['municipio'])
        m21=set(sub[sub['año']==2021]['municipio'])
        nuevos=sorted(m21-m20)
        print(f"\n--- {dep} / {cat}: nuevos 2021 (no estaban en 2020) = {len(nuevos)} ---")
        fila=sub[(sub['año']==2021)&(sub['municipio'].isin(nuevos))]
        if len(nuevos):
            out=fila[['municipio','area_sembrada','produccion']].copy()
            out=out.sort_values('municipio')
            print(out.to_string(index=False))
        # comparacion atipicos: area/produccion nuevos vs continuos
        cont=sorted(m20 & m21)
        if nuevos and cont:
            a_nue=sub[(sub['año']==2021)&(sub['municipio'].isin(nuevos))]['area_sembrada']
            p_nue=sub[(sub['año']==2021)&(sub['municipio'].isin(nuevos))]['produccion']
            a_con=sub[(sub['año']==2021)&(sub['municipio'].isin(cont))]['area_sembrada']
            p_con=sub[(sub['año']==2021)&(sub['municipio'].isin(cont))]['produccion']
            print(f"   area_sembrada 2021 -> nuevos median={a_nue.median():.1f} mean={a_nue.mean():.1f} | continuos median={a_con.median():.1f} mean={a_con.mean():.1f}")
            print(f"   produccion   2021 -> nuevos median={p_nue.median():.1f} mean={p_nue.mean():.1f} | continuos median={p_con.median():.1f} mean={p_con.mean():.1f}")

print("\n--- Conteo real de municipios por año: cafe/cacao/frijol en HUILA y TOLIMA (¿salto 2021?) ---")
for dep in ['HUILA','TOLIMA']:
    for cat in ['Café','Cacao','Frijol']:
        sub=cons[(cons['departamento']==dep)&(cons['cultivo_cat']==cat)]
        c20=sub[sub['año']==2020]['municipio'].nunique()
        c21=sub[sub['año']==2021]['municipio'].nunique()
        print(f"   {dep:8s} {cat:8s}: 2020={c20:3d} municipios  2021={c21:3d} municipios  (delta {c21-c20:+d})")

# ============ TAREA 2: ANTIOQUIA MAIZ TECNIFICADO (hueco 2015-2018) ============
print("\n"+"="*90)
print("TAREA 2 - Antioquia Maiz tecnificado: municipios previos a 2015 y reclasificacion")
print("="*90)
at=cons[(cons['departamento']=='ANTIOQUIA')&(cons['cultivo_cat']=='Maíz tecnificado')]
prev=set(at[at['año']<2015]['municipio'])
print(f"Municipios con Maiz tecnificado antes de 2015: {sorted(prev)}")
# ¿alguno reaparece como Maiz tradicional en 2015-2018?
atr=cons[(cons['departamento']=='ANTIOQUIA')&(cons['cultivo_cat']=='Maíz tradicional')&(cons['año'].between(2015,2018))]
reclas=sorted(set(prev)&set(atr['municipio']))
print(f"De ellos, los que SI reportan Maiz tradicional en 2015-2018: {reclas if reclas else '(ninguno)'}")
print("Detalle años tradicional 2015-2018 para esos municipios:")
if reclas:
    print(atr[atr['municipio'].isin(reclas)][['municipio','año','area_sembrada','produccion']].to_string(index=False))

print("\nVersion (a) TAL CUAL - municipios/año Maiz tecnificado Antioquia 2007-2024:")
serie_a=at.groupby('año')['municipio'].nunique().reindex(range(2007,2025),fill_value=0)
print(serie_a.to_string())
print("\nVersion (b) 2015-2018 como FALTANTE (NaN), no cero:")
serie_b=serie_a.copy().astype('float')
serie_b.loc[2015:2018]=np.nan
print(serie_b.to_string())

# ============ TAREA 3: PANEL BALANCEADO ============
print("\n"+"="*90)
print("TAREA 3 - Panel balanceado (municipio presente en los 18 años 2007-2024)")
print("="*90)
res=[]
mask_bal=pd.Series(False,index=cons.index)
for (dep,cat),g in cons.groupby(['departamento','cultivo_cat']):
    # clave robusta: codigo DANE de municipio (evita diferencias de case B vs A)
    yrs=g.groupby('cod_dane_mun')['año'].apply(lambda s:set(s))
    total_mun=g['cod_dane_mun'].nunique()
    bal_mun=[m for m,ys in yrs.items() if len(ys)==18 and set(range(2007,2025)).issubset(ys)]
    res.append((dep,cat,total_mun,len(bal_mun)))
    if bal_mun:
        idx=g[g['cod_dane_mun'].isin(bal_mun)].index
        mask_bal.loc[idx]=True
resumen=pd.DataFrame(res,columns=['departamento','cultivo_cat','total_municipios','en_panel_balanceado'])
print(resumen.to_string(index=False))
print(f"\nTotal filas en panel balanceado: {int(mask_bal.sum())} de {len(cons)}")

cons['en_panel_balanceado']=mask_bal
bal=cons[mask_bal].copy()
cons.to_csv(os.path.join(base,'eva_2007_2024_consolidado.csv'),index=False,encoding='utf-8')
bal.to_csv(os.path.join(base,'eva_panel_balanceado.csv'),index=False,encoding='utf-8')
print("Actualizado eva_2007_2024_consolidado.csv (col en_panel_balanceado) y escrito eva_panel_balanceado.csv")

# ============ TAREA 4: LISTA UNICA MUNICIPIOS PARA IDEAM ============
print("\n"+"="*90)
print("TAREA 4 - municipios_panel_final.csv (unicos del panel balanceado)")
print("="*90)
mp=bal[['departamento','municipio','cod_dane_mun']].drop_duplicates()
# dedupe por cod_dane_mun: hay inconsistencia de tildes entre B y A (ej ELIAS/ELÍAS)
# se elige el nombre mas largo (el acentuado, ortografia oficial)
mp=mp.sort_values('municipio',key=lambda s:s.map(len),ascending=False).drop_duplicates('cod_dane_mun')
mp=mp.sort_values(['departamento','cod_dane_mun'])
mp.to_csv(os.path.join(base,'municipios_panel_final.csv'),index=False,encoding='utf-8')
print(f"Municipios unicos en panel balanceado (dedup por cod_dane_mun): {len(mp)}")
print(mp.to_string(index=False))
