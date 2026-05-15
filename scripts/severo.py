import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report


df = pd.read_csv(
    r"C:\Users\danielsevero-ieg\OneDrive - Instituto Germinare\Área de Trabalho\Escola\TECH\Terceirão\AD\PriemeiroSemestre\Atividade em Grupo\atv-classificacao-cliente\scripts\credito_banco_realista.csv"
)


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
