from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Criar diretório do DUMP (arquivos temporários)
# -----------------------------------------------------------------------

## 1.1 Criar diretório para salvar os arquivos temporários (DUMP)
CURRENT_DIR = Path(__file__).parent
DUMP_DIR = CURRENT_DIR / 'out'
DUMP_DIR.mkdir(exist_ok=True)

## 2. Definir o caminho do arquivo CSV
CSV_PATH = CURRENT_DIR / 'assets' / 'credito_banco.csv'

# 2. Ler o dataset
# -----------------------------------------------------------------------

# 2.1 Leitura
df_gerado = pd.read_csv(CSV_PATH)

# 2.3 Classe de colunas (base em credito_banco.csv)
class Col:
    ID_CLIENTE = 'id_cliente'
    IDADE = 'idade'
    RENDA_ANUAL = 'renda_anual'
    DIVIDA_ATUAL = 'divida_atual'
    HISTORICO_PAGAMENTO = 'historico_pagamento'
    SCORE_EXTERNO = 'score_externo'
    INADIMPLENTE = 'inadimplente'

    # Score ajustado
    SCORE_AJUSTADO = 'score_ajustado'

# 3. Limpeza e pré-processamento
# -----------------------------------------------------------------------

## 3.1 Limpeza de duplicados
df_credito = df_gerado.drop_duplicates()

## 3.2 Limpeza de valores nulos (preencher com a média da coluna)
for column in [Col.IDADE, Col.RENDA_ANUAL, Col.DIVIDA_ATUAL, Col.HISTORICO_PAGAMENTO, Col.SCORE_EXTERNO]:
    df_credito[column].fillna(df_credito[column].mean(), inplace=True)

## 3.3 Remover outliers (baseado no IQR - Interquartile Range)
def remove_outliers(df, column_name):
    q1 = df[column_name].quantile(0.25)
    q3 = df[column_name].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]
    return df

## Somente idade e renda
df_credito = remove_outliers(df_credito, Col.IDADE)
df_credito = remove_outliers(df_credito, Col.RENDA_ANUAL) 


## 3.4 Colunas para o cálculo do score ajustado
renda = df_credito[Col.RENDA_ANUAL].values
idade = df_credito[Col.IDADE].values
score_externo = df_credito[Col.SCORE_EXTERNO].values

## 3.5 Definir Fatores de Renda e Idade para ajuste do score

# Escolhemos quando a renda for maior que 150.000 recebe 1.2 a mais no score
# Quando a renda estiver entre 50.000 e 150.000 mantém o score
# Quando a renda for menor que 50.000 recebe 0.8 a menos no score
fator_renda = np.where(renda > 150000, 2, np.where(renda >= 50000, 1.0, 0.5))

# Escolhemos quando a idade for maior que 50 recebe 1.2 a mais no score
# Quando a idade estiver entre 25 e 50 mantém o score
# Quando a idade for menor que 25 recebe 0.8 a menos no score
fator_idade = np.where(idade > 50, 2, np.where(idade >= 25, 1.0, 0.5))

score_ajustado = score_externo * fator_renda * fator_idade

## 3.6 Adicionar a coluna de score ajustado ao DataFrame
df_credito[Col.SCORE_AJUSTADO] = score_ajustado


# 4. Machine Learning - Classificação
# -----------------------------------------------------------------------


# 8. Montagem de gráficos com matplotlib
# -----------------------------------------------------------------------

# Separando adimplentes e inadimplentes
adimplentes = df_credito[df_credito[Col.INADIMPLENTE] == 0]
inadimplentes = df_credito[df_credito[Col.INADIMPLENTE] == 1]

# Criar figura com 3 gráficos
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# -----------------------------------------------------------------------
# Gráfico 1 - Adimplentes
# -----------------------------------------------------------------------

axes[0].hist(
    adimplentes.renda_anual,
    color='purple',
    edgecolor='white'
)

axes[0].set_title("Renda de Adimplentes")
axes[0].set_xlabel("Renda (R$)")
axes[0].set_ylabel("Frequência")

# -----------------------------------------------------------------------
# Gráfico 2 - Inadimplentes
# -----------------------------------------------------------------------

axes[1].hist(
    inadimplentes.renda_anual,
    color='orange',
    edgecolor='white'
)

axes[1].set_title("Renda de Inadimplentes")
axes[1].set_xlabel("Renda (R$)")
axes[1].set_ylabel("Frequência")

# -----------------------------------------------------------------------
# Gráfico 3 - Heatmap
# -----------------------------------------------------------------------

colunas = [
    "renda_anual",
    "idade",
    "divida_atual",
    "score_ajustado"
]

correlacao = df_credito[colunas].corr()

cax = axes[2].matshow(
    correlacao,
    cmap="coolwarm"
)

# Barra lateral do heatmap
fig.colorbar(cax, ax=axes[2], fraction=0.046, pad=0.04)

axes[2].set_xticks(range(len(colunas)))
axes[2].set_yticks(range(len(colunas)))

axes[2].set_xticklabels(colunas, rotation=45)
axes[2].set_yticklabels(colunas)

# Valores da correlação
for i in range(len(colunas)):
    for j in range(len(colunas)):
        axes[2].text(
            j,
            i,
            f"{correlacao.iloc[i, j]:.2f}",
            va="center",
            ha="center"
        )

axes[2].set_title("Heatmap de Correlação")

# -----------------------------------------------------------------------

plt.tight_layout()
plt.show()