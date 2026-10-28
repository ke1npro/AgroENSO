import pandas as pd
import glob, os

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

base = r"C:\Proyecto Semillero"
A_PATH = glob.glob(os.path.join(base, "*Base_Agr*cola*.csv"))[0]
B_PATH = glob.glob(os.path.join(base, "*2007-2018*"))[0]

print("A_PATH =", A_PATH)
print("B_PATH =", B_PATH)

print("\n" + "=" * 80)
print("PASO 1 - ARCHIVO A (CSV, datos.gov.co, 2019-2024)")
print("=" * 80)
dfA = pd.read_csv(A_PATH, nrows=0)
print("\n--- A.columns.tolist() ---")
print(dfA.columns.tolist())
print("\n--- A.dtypes ---")
print(dfA.dtypes)

print("\n" + "=" * 80)
print("PASO 1 - ARCHIVO B (Excel, AgroNet, 2007-2018)")
print("=" * 80)
xl = pd.ExcelFile(B_PATH)
print("\n--- Hojas en B:", xl.sheet_names)
dfB = pd.read_excel(B_PATH, nrows=0)
print("\n--- B.columns.tolist() ---")
print(dfB.columns.tolist())
print("\n--- B.dtypes ---")
print(dfB.dtypes)
