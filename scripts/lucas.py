from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

CURRENT_DIR = Path(__file__).parent

CSV_PATH = CURRENT_DIR / "credito_banco.csv"


df_credito = pd.read_csv(CSV_PATH)

adimplentes = df_credito[df_credito["inadimplente"] == 0]
inadimplentes = df_credito[df_credito["inadimplente"] == 1]

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(adimplentes.renda_anual, color='purple', edgecolor='white')
plt.title("Distribuição de Renda de Adimplentes")
plt.xlabel("Renda (R$)")
plt.ylabel("Frequência")

plt.subplot(1, 2, 2)
plt.hist(inadimplentes.renda_anual, color='orange', edgecolor='white')
plt.title("Distribuição de Renda de Inadimplentes")
plt.xlabel("Renda (R$)")
plt.ylabel("Frequência")

plt.show()