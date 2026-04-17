# Libs
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from metricas_plots import PlotsMetricas

# Variáveis globais
p = PlotsMetricas()
larguras = p.targets[:4]
cov_pairs = [
    (larguras[0], larguras[1]),
    (larguras[0], larguras[2]),
    (larguras[0], larguras[3]),
    (larguras[1], larguras[2]),
    (larguras[1], larguras[3]),
    (larguras[2], larguras[3]),
]

# Fixando seeds para reprodutibilidade
# seeds = [int(x) for x in np.random.uniform(1000, 9999, 10)]
seeds = [1637, 8334, 7910, 3576, 2737, 9809, 4655, 7819, 1802, 3788]

# Dados
dados = pd.read_csv("dados/ariel_limpo_log10.csv.gz", compression="gzip")
larguras = p.targets[:4]
train, test = train_test_split(
    dados[p.features + larguras], test_size=0.25, random_state=4321
)

# Configuração do Operon
config_operon = {
    "random_state": 4321,
    "population_size": 2000,
    "generations": 2000,
    "allowed_symbols": "add,sub,mul,aq,constant,variable,pow,exp,tanh",
    "max_length": 25,
    "max_depth": 100,
    "optimizer_iterations": 1000,
    "model_selection_criterion": "bayesian_information_criterion",
    "objectives": ["r2", "length"],
    "n_threads": 12,
}


def treinar_modelos_operon(hyper, X_train_bins, items):
    modelos = {}
    for model_key, col_name in items:
        y = train_lhc[col_name].values.astype(np.float64)
        modelo = p.treinar_operon(hyper, X_train_bins, y)
        modelos[model_key] = modelo
    return modelos


bic_todos = {}  # Dicionário para armazenar todos os BICs

for seed in tqdm(seeds):
    train_lhc = p.lhs_subsample_with_stats(
        train, p.features, larguras, n=2000, k_neighbors=100, seed=seed
    )
    X_train_bins = train_lhc[p.features].values.astype(np.float64)

    # Modelos para os estimadores da Normal
    items = (
        [(f"{l}_mean", f"{l}_mean") for l in larguras]
        + [(f"{l}_std", f"{l}_std") for l in larguras]
        + [(f"cov_{l1}_{l2}", f"cov_{l1}_{l2}") for l1, l2 in p.cov_pairs]
    )

    modelos = treinar_modelos_operon(config_operon, X_train_bins, items)

    for model_key, modelo in modelos.items():
        bic_todos.setdefault(model_key, []).append(modelo.stats_["model_bic"])


# Salvando JSON para Debug
with open("results/bics_operon.json", "w") as f:
    json.dump(bic_todos, f, indent=2)

# Gerando e salvando DataFrame com os resultados
df_bic = pd.DataFrame(
    {
        "modelo": list(bic_todos.keys()),
        "bic_mean": [np.mean(v) for v in bic_todos.values()],
        "bic_std": [np.std(v) for v in bic_todos.values()],
    }
)
df_bic.to_csv("results/bic_operon.csv", index=False)
