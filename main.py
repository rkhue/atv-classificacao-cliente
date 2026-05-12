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
df_credito = pd.read_csv(CSV_PATH)

# 2.3 Classe de colunas (base em credito_banco.csv)
class Col:
    ID_CLIENTE = 'id_cliente'
    IDADE = 'idade'
    RENDA_ANUAL = 'renda_anual'
    DIVIDA_ATUAL = 'divida_atual'
    HISTORICO_PAGAMENTO = 'historico_pagamento'
    SCORE_EXTERNO = 'score_externo'
    INADIMPLENTE = 'inadimplente'


# 3. Limpeza e pré-processamento
# -----------------------------------------------------------------------

## 3.1 Limpeza de duplicados
df = df_credito.drop_duplicates()

## 3.2 Limpeza de valores nulos (preencher com a média da coluna)
for column in [Col.IDADE, Col.RENDA_ANUAL, Col.DIVIDA_ATUAL, Col.HISTORICO_PAGAMENTO, Col.SCORE_EXTERNO]:
    df[column].fillna(df[column].mean(), inplace=True)

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
df = remove_outliers(df, Col.IDADE)
df = remove_outliers(df, Col.RENDA_ANUAL) 

# 4. Cálculo Score Ajustado
# -----------------------------------------------------------------------

# Colunas para o cálculo do score ajustado
renda = df[Col.RENDA_ANUAL].values
idade = df[Col.IDADE].values
score_externo = df[Col.SCORE_EXTERNO].values

# Definir Fatores Renda

# Escolhemos quando a renda for maior que 150.000 recebe 1.2 a mais no score
# Quando a renda estiver entre 50.000 e 150.000 mantém o score
# Quando a renda for menor que 50.000 recebe 0.8 a menos no score
fator_renda = np.where(renda > 150000, 1.2, np.where(renda >= 50000, 1.0, 0.8))

# Escolhemos quando a idade for maior que 50 recebe 1.2 a mais no score
# Quando a idade estiver entre 25 e 50 mantém o score
# Quando a idade for menor que 25 recebe 0.8 a menos no score
fator_idade = np.where(idade > 50, 1.2, np.where(idade >= 25, 1.0, 0.8))

score_ajustado = score_externo * fator_renda * fator_idade

df['score_ajustado'] = score_ajustado

df.to_csv(CSV_PATH, index=False)

# 8. Montagem de gráficos com matplotlib
# -----------------------------------------------------------------------

# Separando adimplentes e inadimplentes
adimplentes = df_credito[df_credito[Col.INADIMPLENTE] == 0]
inadimplentes = df_credito[df_credito[Col.INADIMPLENTE] == 1]

# Visualização da distribuição de renda para adimplentes e inadimplentes
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