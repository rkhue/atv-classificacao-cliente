from pathlib import Path
import pandas as pd
import numpy as np


# 1. Criar diretório do DUMP (arquivos temporários)
# -----------------------------------------------------------------------

## 1.1 Criar diretório para salvar os arquivos temporários (DUMP)
CURRENT_DIR = Path(__file__).parent
DUMP_DIR = CURRENT_DIR / 'out'
DUMP_DIR.mkdir(exist_ok=True)

## 2. Definir o caminho do arquivo CSV
CSV_PATH = CURRENT_DIR / 'credito_banco.csv'

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
