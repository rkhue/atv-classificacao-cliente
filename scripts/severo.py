import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report


df = pd.read_csv(
    r"C:\Users\danielsevero-ieg\OneDrive - Instituto Germinare\Área de Trabalho\Escola\TECH\Terceirão\AD\PriemeiroSemestre\Atividade em Grupo\atv-classificacao-cliente\scripts\credito_banco.csv"
)



x = df.drop(columns=['inadimplente'])
y = df['inadimplente']


x_treino, x_teste, y_treino, y_teste = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)


modelo = RandomForestClassifier(
    random_state=42
)


modelo.fit(x_treino, y_treino)


previsoes = modelo.predict(x_teste)


accuracy = accuracy_score(y_teste, previsoes)
recall = recall_score(y_teste, previsoes)
f1 = f1_score(y_teste, previsoes)

print("Accuracy:", accuracy)
print("Recall:", recall)
print("F1-score:", f1)


print(previsoes)