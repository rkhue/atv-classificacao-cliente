from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

CURRENT_DIR = Path(__file__).parent

CSV_PATH = CURRENT_DIR / "credito_banco.csv"

df_credito = pd.read_csv(CSV_PATH)

colunas = ["renda_anual", "idade", "divida_atual", "score_externo"]

correlacao = df_credito[colunas].corr()

fig, ax = plt.subplots(figsize=(8, 6))

cax = ax.matshow(correlacao, cmap="coolwarm")

fig.colorbar(cax)

ax.set_xticks(range(len(colunas)))
ax.set_yticks(range(len(colunas)))

ax.set_xticklabels(colunas, rotation=45)
ax.set_yticklabels(colunas)

for i in range(len(colunas)):
    for j in range(len(colunas)):
        ax.text(
            j,
            i,
            f"{correlacao.iloc[i, j]:.2f}",
            va="center",
            ha="center"
        )

plt.title("Heatmap de Correlação", pad=20)
plt.tight_layout()
plt.show()