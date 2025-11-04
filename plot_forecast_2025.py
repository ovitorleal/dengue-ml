"""
plot_forecast_2025.py
Gera um gráfico simples mostrando a previsão semanal de casos de dengue para 2025.
Requer o arquivo 'outputs/forecast_2025.csv' gerado pelo script main.py.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Caminho do arquivo gerado pelo modelo
csv_path = os.path.join("outputs", "forecast_2025.csv")

# Verifica se o arquivo existe
if not os.path.exists(csv_path):
    print("❌ O arquivo 'outputs/forecast_2025.csv' não foi encontrado. Execute o main.py antes.")
    exit()

# Carrega as previsões
df = pd.read_csv(csv_path)

# Configuração de estilo
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))

# Gera gráfico de linha com marcadores
plt.plot(df["semana_num"], df["predicted_casos"], marker="o", linestyle="-", label="Casos previstos (2025)")

# Adiciona rótulos e título
plt.title("Previsão semanal de casos de dengue — 2025", fontsize=14)
plt.xlabel("Semana epidemiológica", fontsize=12)
plt.ylabel("Casos previstos", fontsize=12)
plt.legend()
plt.tight_layout()

# Cria a pasta outputs (caso não exista)
os.makedirs("outputs", exist_ok=True)

# Salva o gráfico
plot_path = os.path.join("outputs", "forecast_2025_barras.png")
plt.savefig(plot_path, bbox_inches="tight")

# Exibe no terminal
plt.show()

print(f"✅ Gráfico gerado com sucesso: {plot_path}")
