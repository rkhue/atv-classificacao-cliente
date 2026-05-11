import csv
import random
import math
import os

# Random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

n_records = 1000

data = []
# Header
data.append(['id_cliente', 'idade', 'renda_anual', 'divida_atual', 'historico_pagamento', 'score_externo', 'inadimplente'])

inadimplentes_count = 0

# Base parameters
intercept = -0.5
weight_renda = -2.5
weight_historico = -3.0
weight_divida = 3.5

for i in range(1, n_records + 1):
    id_cliente = i
    idade = random.randint(18, 80)
    
    # Renda anual: usando uma distribuição que puxe mais para rendas menores, mas respeitando os limites
    renda_anual = round(random.uniform(15000, 500000), 2)
    
    # Dívida atual
    divida_atual = round(random.uniform(0, 50000), 2)
    
    # Histórico de pagamento
    historico_pagamento = random.randint(0, 100)
    
    # Score externo
    score_externo = round(random.uniform(0, 1), 4)
    
    # Lógica de inadimplência
    norm_renda = (renda_anual - 15000) / (500000 - 15000)
    norm_historico = historico_pagamento / 100.0
    norm_divida = divida_atual / 50000.0
    
    logit = intercept + weight_renda * norm_renda + weight_historico * norm_historico + weight_divida * norm_divida
    prob = 1 / (1 + math.exp(-logit))
    
    inadimplente = 1 if random.random() < prob else 0
    inadimplentes_count += inadimplente
    
    data.append([id_cliente, idade, renda_anual, divida_atual, historico_pagamento, score_externo, inadimplente])

# Ajustar para 15-20% se não estiver na faixa
while not (0.15 <= inadimplentes_count / n_records <= 0.20):
    if inadimplentes_count / n_records < 0.15:
        intercept += 0.1
    else:
        intercept -= 0.1
        
    inadimplentes_count = 0
    for row in data[1:]:
        renda_anual = row[2]
        divida_atual = row[3]
        historico_pagamento = row[4]
        
        norm_renda = (renda_anual - 15000) / (500000 - 15000)
        norm_historico = historico_pagamento / 100.0
        norm_divida = divida_atual / 50000.0
        
        logit = intercept + weight_renda * norm_renda + weight_historico * norm_historico + weight_divida * norm_divida
        prob = 1 / (1 + math.exp(-logit))
        
        inadimplente = 1 if random.random() < prob else 0
        row[6] = inadimplente
        inadimplentes_count += inadimplente

print(f"Taxa de inadimplência gerada: {inadimplentes_count / n_records * 100:.2f}%")

output_path = os.path.join(os.path.dirname(__file__), 'credito_banco.csv')
with open(output_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(data)

print(f"Arquivo 'credito_banco.csv' criado com sucesso em: {output_path}")
