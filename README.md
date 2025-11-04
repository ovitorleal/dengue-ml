# 🦠 Previsão Semanal de Casos de Dengue — 2025

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-RandomForest-orange?logo=scikitlearn)
![Status](https://img.shields.io/badge/Status-Finalizado-brightgreen)
![Licença](https://img.shields.io/badge/Licença-Livre-lightgrey)

---

## 🎯 Objetivo

Este projeto foi desenvolvido por **Vitor da Silva Leal**, profissional da área de **Vigilância Epidemiológica**, com o propósito de aplicar a tecnologia para **prever o número de casos semanais de dengue no ano de 2025**.  

A ideia é simples, mas poderosa: usar dados reais de notificações anteriores para **antecipar possíveis aumentos de casos**, ajudando a equipe de vigilância a **planejar ações preventivas** com mais eficiência.

---

## 💡 Motivação

> “Quem trabalha com vigilância sabe: o segredo é se antecipar.”

Com base nessa ideia, este projeto transforma dados históricos do **SINAN (Sistema Nacional de Agravos de Notificação)** em uma previsão das 52 semanas de 2025.  
O modelo aprende com os padrões de anos anteriores (2019 a 2024) e gera uma projeção que pode servir de apoio ao planejamento de campanhas e decisões estratégicas na saúde pública.

---

## 🧩 Estrutura do Projeto

📂 dengue-ml
┣ 📜 main.py → código principal que treina o modelo e gera a previsão
┣ 📜 plot_forecast_2025.py → script para gerar o gráfico das previsões semanais
┣ 📜 requirements.txt → lista de bibliotecas necessárias
┣ 📜 dengue.csv → base de dados original (extraída do SINAN)
┣ 📂 outputs
┃ ┣ forecast_2025.csv → tabela com as 52 previsões semanais
┃ ┣ forecast_plot.png → gráfico histórico + projeção 2025
┃ ┗ forecast_2025_barras.png → gráfico em barras das previsões semanais
┗ 📜 README.md → este arquivo

yaml
Copiar código

---

## ⚙️ Como Executar

1. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
Execute o modelo para gerar as previsões:

bash
Copiar código
python main.py --input dengue.csv --output outputs --seed 42
Isso treina o modelo e gera o arquivo outputs/forecast_2025.csv.

Gere o gráfico semanal:

bash
Copiar código
python plot_forecast_2025.py
O resultado será salvo em outputs/forecast_2025_barras.png.

---

## 📊 Resultados
Arquivo: forecast_2025.csv → contém as previsões semanais de 2025.

Gráficos:

forecast_plot.png: histórico de 2019–2024 + projeção de 2025.

forecast_2025_barras.png: evolução semanal prevista para 2025.

Essas previsões permitem visualizar picos e quedas ao longo do ano, auxiliando o trabalho de planejamento e vigilância em saúde.

---

## 💬 Explicação Simples
O projeto pega o histórico de casos e ensina o computador a entender como eles se comportam ao longo das semanas.
Depois, ele tenta “adivinhar” como será o comportamento no próximo ano, com base nos padrões observados.

O objetivo não é prever números exatos, mas entender tendências — e isso já é muito valioso para a vigilância epidemiológica.

---

## 🧠 Tecnologias Utilizadas
Categoria	Ferramenta
Linguagem	🐍 Python
Análise de Dados	📊 Pandas, NumPy
Modelagem	🤖 Scikit-learn (Random Forest)
Visualização	📈 Matplotlib, Seaborn
Armazenamento de Modelo	💾 Joblib

---

## 🤝 Apoio do ChatGPT
Durante o desenvolvimento, o ChatGPT foi utilizado como assistente técnico e organizacional.

O ChatGPT atuou como ferramenta de apoio, e todo o conteúdo foi adaptado à realidade e experiência da vigilância epidemiológica.

---

## 🚀 Próximos Passos
Incluir dados de chuva e temperatura para melhorar a precisão.

Implementar uma previsão iterativa, onde cada semana prevista influencia a próxima.

Expandir o modelo para outras doenças de notificação, como chikungunya e zika.

---

## ✨ Autor
👨‍💻 Vitor da Silva Leal
Profissional de Vigilância Epidemiológica
📍 Volta Redonda — RJ
