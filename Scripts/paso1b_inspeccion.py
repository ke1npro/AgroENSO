import pandas as pd
import glob, os

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 250)
pd.set_option('display.max_colwidth', 40)

base = r"C:\Proyecto Semillero"
A_PATH = glob.glob(os.path.join(base, "*Base_Agr*cola*.csv"))[0]
B_PATH = glob.glob(os.path.join(base, "*2007-2018*"))[0]

# A: print raw repr of column names (to see real unicode)
dfA = pd.read_csv(A_PATH, nrows=5, encoding='utf-8')
print("=== A real column names (repr) ===")
for c in dfA.columns:
    print(repr(c))
print("\n=== A head (5 rows) ===")
print(dfA.head(5).to_string())

# B: read raw first 12 rows, no header
print("\n\n=== B raw first 12 rows (header=None) ===")
dfBraw = pd.read_excel(B_PATH, header=None, nrows=12)
for i, row in dfBraw.iterrows():
    print(f"--- row {i} ---")
    print(list(row.values))
