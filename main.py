"""
main.py
Treina um RandomForestRegressor para prever casos semanais de dengue e produz previsões para 2025.

Uso:
    python main.py --input dengue.csv --output outputs --seed 42
"""

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin1", sep=";", skiprows=4, na_values='-')
    df = df.rename(columns=lambda c: c.strip())
    semana_cols = [c for c in df.columns if c.lower().startswith('semana')]
    if 'Ano notificação' not in df.columns:
        raise ValueError('Coluna "Ano notificação" não encontrada. Verifique o arquivo CSV.')
    df_long = df[['Ano notificação'] + semana_cols].melt(
        id_vars=['Ano notificação'],
        value_vars=semana_cols,
        var_name='semana',
        value_name='casos'
    )
    df_long['ano'] = df_long['Ano notificação'].astype(str).str.replace('"', '').str.strip()
    df_long = df_long[df_long['ano'].str.match(r'^\d{4}$')]
    df_long['semana_num'] = df_long['semana'].str.extract(r'(\d+)').astype(int)
    df_long['casos'] = pd.to_numeric(df_long['casos'], errors='coerce').fillna(0).astype(int)
    df_long = df_long.sort_values(['ano', 'semana_num']).reset_index(drop=True)
    anos_ordenados = sorted(df_long['ano'].unique(), key=lambda x: int(x))
    first_year = int(anos_ordenados[0])
    df_long['time_idx'] = (df_long['ano'].astype(int) - first_year) * 52 + (df_long['semana_num'] - 1)
    df_out = df_long[['ano', 'semana_num', 'casos', 'time_idx']].copy()
    df_out['ano'] = df_out['ano'].astype(int)
    return df_out


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df[['time_idx', 'ano', 'semana_num', 'casos']].copy()
    for lag in [1, 2, 3, 52]:
        out[f'lag_{lag}'] = out['casos'].shift(lag)
    out['rolling_mean_4'] = out['casos'].shift(1).rolling(window=4, min_periods=1).mean()
    out['rolling_std_4'] = out['casos'].shift(1).rolling(window=4, min_periods=1).std().fillna(0)
    out['semana_sin'] = np.sin(2 * np.pi * out['semana_num'] / 52)
    out['semana_cos'] = np.cos(2 * np.pi * out['semana_num'] / 52)
    out = out.dropna(subset=[f'lag_{l}' for l in [1, 2, 3, 52]]).reset_index(drop=True)
    return out


def train_and_evaluate(df_features: pd.DataFrame, seed: int = 42, output_dir: str = 'outputs'):
    # IMPORTANTE: Poderia remover time_idx das variáveis usadas no treinamento (pois ele é semelhante a um índice)
    X = df_features.drop(columns=['casos', 'ano', 'time_idx'])
    y = df_features['casos']

    test_size = 26
    if len(X) <= test_size * 2:
        test_size = max(6, int(len(X) * 0.2))

    X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

    rf = RandomForestRegressor(random_state=seed, n_jobs=-1)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [6, 10, None],
        'min_samples_split': [2, 5]
    }
    tss = TimeSeriesSplit(n_splits=3)
    gscv = GridSearchCV(rf, param_grid, cv=tss, scoring='neg_mean_absolute_error', n_jobs=-1)
    gscv.fit(X_train, y_train)

    best = gscv.best_estimator_
    y_pred = best.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))  
    r2 = r2_score(y_test, y_pred)

    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(best, os.path.join(output_dir, 'model.pkl'))

    metrics = {'mae': mae, 'rmse': rmse, 'r2': r2, 'best_params': gscv.best_params_}
    return best, metrics, X_train, X_test, y_train, y_test, y_pred


def forecast_2025(df_original: pd.DataFrame, df_features: pd.DataFrame, model, output_dir: str = 'outputs'):
    first_year = int(sorted(df_original['ano'].unique())[0])
    target_year = 2025
    weeks = list(range(1, 53))

    df_feats = df_features.set_index('time_idx')
    forecast_rows = []
    for w in weeks:
        t_idx = (target_year - first_year) * 52 + (w - 1)
        row = {
            'time_idx': t_idx,
            'semana_num': w,
            'semana_sin': np.sin(2 * np.pi * w / 52),
            'semana_cos': np.cos(2 * np.pi * w / 52),
        }
        for lag in [1, 2, 3, 52]:
            lag_idx = t_idx - lag
            if lag_idx in df_feats.index:
                row[f'lag_{lag}'] = int(df_feats.loc[lag_idx, 'casos']) 
            else:
                row[f'lag_{lag}'] = int(df_feats['casos'].mean())

        prev_idxs = [t_idx - i for i in range(1, 5)]
        vals = [df_feats.loc[i, 'casos'] if i in df_feats.index else df_feats['casos'].mean() for i in prev_idxs]
        row['rolling_mean_4'] = float(np.mean(vals))
        row['rolling_std_4'] = float(np.std(vals))

        forecast_rows.append(row)

    X_forecast = pd.DataFrame(forecast_rows)

#   Use exatamente os nomes das variáveis (features) com os quais o modelo foi treinado
    if hasattr(model, "feature_names_in_"):
        feature_cols = list(model.feature_names_in_)
    else:
        #  alternativa: use todas as colunas, exceto 'time_idx'
        feature_cols = [c for c in X_forecast.columns if c != 'time_idx']

#   Garanta que X_forecast tenha todas as colunas necessárias (preencha com a média caso alguma esteja faltando)
    for col in feature_cols:
        if col not in X_forecast.columns:
            X_forecast[col] = df_features[col].mean()

   #Faz previsão usando a mesma ordem das colunas
    preds = model.predict(X_forecast[feature_cols])
    X_forecast['predicted_casos'] = np.round(preds).astype(int)
    X_forecast['ano'] = target_year

    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, 'forecast_2025.csv')
    X_forecast.to_csv(out_csv, index=False)

    # Plot histórico + previsão
    plt.figure(figsize=(12, 6))
    hist_plot = df_original.groupby(['time_idx'])['casos'].sum().reset_index()
    plt.plot(hist_plot['time_idx'], hist_plot['casos'], label='Histórico (2019-2024)')
    plt.plot(X_forecast['time_idx'], X_forecast['predicted_casos'], marker='o', linestyle='--', label='Previsão 2025')
    plt.xlabel('time_idx (semanas desde primeiro ano)')
    plt.ylabel('Casos')
    plt.legend()
    plt.title('Casos semanais de dengue — Histórico e previsão para 2025')
    plt.savefig(os.path.join(output_dir, 'forecast_plot.png'), bbox_inches='tight')
    plt.close()

    return out_csv


def main(args):
    print("🔍 Carregando e limpando dados...")
    df_long = load_and_clean(args.input)
    print(f"✅ Dados carregados: {len(df_long)} registros válidos\n")

    print("⚙️  Criando features temporais...")
    df_features = create_features(df_long)

    print("🧠 Treinando modelo Random Forest...")
    model, metrics, *_ = train_and_evaluate(df_features, seed=args.seed, output_dir=args.output)

    print("\n📊 Métricas no conjunto de teste:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\n🔮 Gerando previsão para 2025...")
    out_csv = forecast_2025(df_long, df_features, model, output_dir=args.output)

    print("\n✅ Arquivos gerados:")
    print(f" - {out_csv}")
    print(f" - {os.path.join(args.output, 'forecast_plot.png')}")
    print(f" - {os.path.join(args.output, 'model.pkl')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Treina Random Forest e prevê 2025')
    parser.add_argument('--input', type=str, default='dengue.csv', help='Caminho para o CSV (formato do SINAN)')
    parser.add_argument('--output', type=str, default='outputs', help='Diretório de saída')
    parser.add_argument('--seed', type=int, default=42, help='Seed para reprodutibilidade')
    args = parser.parse_args()
    main(args)
