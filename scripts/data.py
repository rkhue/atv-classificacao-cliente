from pathlib import Path
import pandas as pd
import numpy as np
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
CSV_PATH = CURRENT_DIR / 'credito_banco_realista.csv'

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



## Separando dados de treinamento para o modelo
x = df.drop(columns=['inadimplente'])
y = df['inadimplente']

## Realizando testes
x_treino, x_teste, y_treino, y_teste = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
)

## Definindo modelo Classifier com algumas informações que ajudaram na sua eficiêmcia
modelo = RandomForestClassifier(
    n_estimators=300,
    # max_depth=10, 
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42
)

## Treinando modelo
modelo.fit(x_treino, y_treino)

## Previsões do modelo 0 e 1
previsoes = modelo.predict(x_teste)

## Mostrando taxa de acuracy, f1_score e recall do modelo
accuracy = accuracy_score(y_teste, previsoes)
recall = recall_score(y_teste, previsoes)
f1 = f1_score(y_teste, previsoes)

print("Accuracy:", accuracy)
print("Recall:", recall)
print("F1-score:", f1)

## Mostrando a probavilidade em porcentagem de um usuário específico 
print("Chance de inadimplente:\t",modelo.predict_proba(x_teste)[:, 1][0])
