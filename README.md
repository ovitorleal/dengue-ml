# 🦠 Previsão Semanal de Casos de Dengue — 2025

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-RandomForest-orange?logo=scikitlearn)
![Status](https://img.shields.io/badge/status-Em_desenvolvimento-yellow)
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

Abaixo está a estrutura de arquivos e pastas do projeto:

```bash
📂 dengue-ml/
┣ 📜 main.py                     # Código principal que treina o modelo e gera a previsão
┣ 📜 plot_forecast_2025.py       # Script para gerar os gráficos de visualização
┣ 📜 requirements.txt            # Lista de bibliotecas Python necessárias
┣ 📜 dengue.csv                  # Base de dados original extraída do SINAN
┣ 📂 outputs/
┃ ┣ 📜 forecast_2025.csv         # Tabela com as 52 previsões semanais para 2025
┃ ┣ 🖼️ forecast_plot.png         # Gráfico com histórico + projeção para 2025
┃ ┗ 🖼️ forecast_2025_barras.png  # Gráfico em barras das previsões semanais
┗ 📜 README.md                   # Este arquivo de documentação
````

-----

## ⚙️ Como Executar

Siga os passos abaixo para executar o projeto localmente.

1.  **Prepare o ambiente e instale as dependências:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Treine o modelo e gere as previsões:** 
    ```bash
    python main.py --input dengue.csv --output outputs --seed 42
    ```
    Este comando irá ler o arquivo `dengue.csv`, treinar o modelo e salvar o resultado `forecast_2025.csv` na pasta `outputs`.


3.  **Gere os gráficos de visualização:**
    ```bash
    python plot_forecast_2025.py
    ```
    Este comando utilizará o arquivo de previsão gerado no passo anterior para criar e salvar os gráficos na pasta `outputs`.


-----

## 📊 Resultados

O projeto gera os seguintes arquivos na pasta `outputs`:

### Arquivo de Previsão

  * **`forecast_2025.csv`**: Tabela de dados contendo as previsões do número de casos de dengue para cada uma das 52 semanas epidemiológicas de 2025.

### Visualizações Geradas

  * **`forecast_plot.png`**: Gráfico de linha mostrando a série histórica de casos (2019–2024) em conjunto com a projeção para 2025, permitindo uma análise comparativa.
  * **`forecast_2025_barras.png`**: Gráfico de barras que detalha a evolução semanal dos casos previstos para 2025, ideal para identificar picos e quedas sazonais.

Essas saídas permitem visualizar tendências, auxiliando diretamente o trabalho de planejamento e vigilância em saúde.

-----

## 💬 Explicação Simples

O projeto analisa o histórico de casos de dengue e "ensina" um modelo de computador a reconhecer os padrões de aumento e queda ao longo das semanas e anos. Com base nesse aprendizado, o modelo tenta "adivinhar" como esses padrões se comportarão no próximo ano.

O objetivo principal não é acertar o número exato de casos, mas sim **identificar as tendências** — o que já é uma ferramenta extremamente valiosa para a vigilância epidemiológica se preparar.

-----

## 🧠 Tecnologias Utilizadas

| Categoria                | Ferramenta / Biblioteca               |
| :----------------------- | :------------------------------------ |
| **Linguagem** | 🐍 Python                             |
| **Análise de Dados** | 📊 Pandas, NumPy                      |
| **Modelagem de ML** | 🤖 Scikit-learn (Random Forest)       |
| **Visualização de Dados**| 📈 Matplotlib, Seaborn                |
| **Armazenamento** | 💾 Joblib (para salvar o modelo)      |

-----

## 🤝 Apoio do ChatGPT

Durante o desenvolvimento deste projeto, o ChatGPT foi utilizado como uma ferramenta de assistência técnica e organizacional. Ele auxiliou na estruturação do código, na depuração de erros e na elaboração desta documentação. Todo o conteúdo técnico e a lógica do projeto foram adaptados e validados com base na realidade e na experiência profissional em vigilância epidemiológica.

-----

## 🚀 Próximos Passos

  * **Enriquecer o Modelo**: Incluir variáveis externas, como dados de chuva e temperatura, para melhorar a precisão das previsões.
  * **Implementar Previsão Iterativa**: Desenvolver um sistema onde a previsão de uma semana possa ser usada como dado de entrada para prever a semana seguinte.
  * **Expandir para Outras Doenças**: Adaptar o modelo para prever casos de outras doenças de notificação compulsória, como Chikungunya e Zika.

-----

## ✨ Autor

**👨‍💻 Vitor da Silva Leal**
<br>
Profissional de Vigilância Epidemiológica
<br>
📍 Volta Redonda — RJ
