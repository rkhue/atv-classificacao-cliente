import pandas as pd
import numpy as np
import os

diretorio = os.path.dirname(os.path.abspath(__file__))
    
credito_banco = os.path.join(diretorio, 'credito_banco.csv')
credito_banco_ajustado = os.path.join(diretorio, 'credito_banco_ajustado.csv')

df = pd.read_csv(credito_banco)

renda = df['renda_anual'].values
idade = df['idade'].values
score_externo = df['score_externo'].values

#Escolhemos quando a renda for maior que 150.000 recebe 1.2 a mais no score
#Quando a renda estiver entre 50.000 e 150.000 mantém o score
#Quando a renda for menor que 50.000 recebe 0.8 a menos no score
fator_renda = np.where(renda > 150000, 1.2, np.where(renda >= 50000, 1.0, 0.8))

#Escolhemos quando a idade for maior que 50 recebe 1.2 a mais no score
#Quando a idade estiver entre 25 e 50 mantém o score
#Quando a idade for menor que 25 recebe 0.8 a menos no score
fator_idade = np.where(idade > 50, 1.2, np.where(idade >= 25, 1.0, 0.8))

score_ajustado = score_externo * fator_renda * fator_idade

df['score_ajustado'] = score_ajustado

df.to_csv(credito_banco_ajustado, index=False)