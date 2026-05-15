from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report

# 1. Criar diretório do DUMP (arquivos temporários)
# -----------------------------------------------------------------------

## 1.1 Criar diretório para salvar os arquivos temporários (DUMP)
CURRENT_DIR = Path(__file__).parent
DUMP_DIR = CURRENT_DIR / 'out'
DUMP_DIR.mkdir(exist_ok=True)

## 2. Definir o caminho do arquivo CSV
CSV_PATH = CURRENT_DIR / 'assets' / 'credito_banco_realista.csv'

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

    # Colunas novas
    SCORE_AJUSTADO = 'score_ajustado'
    CHANCE_INADIMPLENCIA = 'Chance_Inadimplencia(%)'

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

## 4.1 Separando dados de treinamento para o modelo
x = df_credito.drop(columns=[Col.INADIMPLENTE])
y = df_credito[Col.INADIMPLENTE]

## 4.2 Definindo os dados de treino e teste
x_treino, x_teste, y_treino, y_teste = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
)

## 4.3 Definindo modelo Classifier com algumas informações que ajudaram na sua eficiêmcia
modelo = RandomForestClassifier(
    n_estimators=300,
    # max_depth=10, 
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42
)

## 4.4 Treinando modelo
modelo.fit(x_treino, y_treino)

## 4.5 Previsões do modelo 0 e 1
previsoes = modelo.predict(x_teste)

## 4.6 Mostrando taxa de acuracy, f1_score e recall do modelo
accuracy = accuracy_score(y_teste, previsoes)
recall = recall_score(y_teste, previsoes)
f1 = f1_score(y_teste, previsoes)

print("Accuracy:", accuracy)
print("Recall:", recall)
print("F1-score:", f1)

## 4.7 Mostrando a probavilidade em porcentagem de um usuário específico 

df_inadimplencia = df_credito.copy()

df_inadimplencia[Col.CHANCE_INADIMPLENCIA] = modelo.predict_proba(
    x
)[:, 1]*100

print(df_inadimplencia)


# 5. RPA - Exportar clientes de alto risco para Excel no DUMP
# -----------------------------------------------------------------------
alto_risco = df_inadimplencia[df_inadimplencia[Col.CHANCE_INADIMPLENCIA] >= 50]
data_atual = datetime.now().strftime("%Y-%m-%d")
nome_arquivo = f"clientes_alto_risco_{data_atual}.xlsx"
caminho_arquivo = DUMP_DIR / nome_arquivo
alto_risco.to_excel(caminho_arquivo, index=False)
print(f"\nLista de clientes de alto risco exportada para: {caminho_arquivo}\n")

# 6. Montagem de gráficos com matplotlib e seaborn
# -----------------------------------------------------------------------

# 6.1 Separando adimplentes e inadimplentes
adimplentes = df_credito[df_credito[Col.INADIMPLENTE] == 0]
inadimplentes = df_credito[df_credito[Col.INADIMPLENTE] == 1]

# 6.2 Criar figura com 3 gráficos
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 6.3 Montagem do Gráfico 1 - Histograma da Renda de Adimplentes
axes[0].hist(
    adimplentes.renda_anual,
    color='purple',
    edgecolor='white'
)

axes[0].set_title("Renda de Adimplentes")
axes[0].set_xlabel("Renda (R$)")
axes[0].set_ylabel("Frequência")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 6.4 Montagem do Gráfico 2 - Histograma da Renda de Inadimplentes
axes[1].hist(
    inadimplentes.renda_anual,
    color='orange',
    edgecolor='white'
)

axes[1].set_title("Renda de Inadimplentes")
axes[1].set_xlabel("Renda (R$)")
axes[1].set_ylabel("Frequência")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 6.5 Montagem do Gráfico 3 - Heatmap de Correlação entre Renda, Idade, Dívida e Score Ajustado
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

# 6.5.2 Barra lateral do heatmap
fig.colorbar(cax, ax=axes[2], fraction=0.046, pad=0.04)

axes[2].set_xticks(range(len(colunas)))
axes[2].set_yticks(range(len(colunas)))

axes[2].set_xticklabels(colunas, rotation=45)
axes[2].set_yticklabels(colunas)

# 6.5.3 Valores da correlação
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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 6.6 Ajustar layout e mostrar gráficos
plt.tight_layout()
plt.show()