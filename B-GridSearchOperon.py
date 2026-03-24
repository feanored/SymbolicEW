# Libs
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import multivariate_normal, spearmanr
from metricas_plots import PlotsMetricas, T, F

# Variáveis globais
p = PlotsMetricas()
larguras = p.targets[:4]
dados = pd.read_csv("dados/ariel_limpo_log10.csv.gz", compression="gzip")

# Covariâncias (matriz triângular)
cov_pairs = [
    (larguras[0], larguras[1]),
    (larguras[0], larguras[2]),
    (larguras[0], larguras[3]),
    (larguras[1], larguras[2]),
    (larguras[1], larguras[3]),
    (larguras[2], larguras[3]),
]


def gera_stats(dados_split, col_x):
    stats_list = []
    for bin_name, group in dados_split.groupby(f"{col_x}_mean"):
        if len(group) < 5:  # Pular bins com poucos dados
            continue

        # Ponto central de cada bin
        col_center = group[col_x].mean()

        # Calcular média e desvio padrão
        dados_larguras = group[larguras].values
        means = dados_larguras.mean(axis=0)
        stds = dados_larguras.std(axis=0)

        # Calcular matriz de covariância 4x4
        cov_matrix = np.cov(dados_larguras, rowvar=False)

        # Montar dicionário com todas as estatísticas
        stat_dict = {col_x: col_center}
        for i, largura in enumerate(larguras):
            stat_dict[f"{largura}_mean"] = means[i]
            stat_dict[f"{largura}_std"] = stds[i]
        for l1, l2 in cov_pairs:
            i1 = larguras.index(l1)
            i2 = larguras.index(l2)
            stat_dict[f"cov_{l1}_{l2}"] = cov_matrix[i1, i2]

        stats_list.append(stat_dict)
    return stats_list


def grid_search(col_x, SEED=4321, popsize=1000, select="by_bic"):
    if select != "by_bic" and select != "by_r2":
        raise ("Modo de seleção deve ser by_bic ou by_r2")

    # Faz o split e o agrupamento dos dados
    train, test = train_test_split(
        dados[[col_x] + larguras], test_size=0.3, random_state=4321
    )

    # Cria subintervalos baseados em frequências (qcut)
    train[f"{col_x}_mean"] = pd.qcut(train[col_x], 150)
    test[f"{col_x}_mean"] = pd.qcut(test[col_x], 150)
    train_por_bin = pd.DataFrame(gera_stats(train, col_x))
    test_por_bin = pd.DataFrame(gera_stats(test, col_x))

    ### Treinar Operons para Média, Desvio Padrão e Covariâncias

    # Selecionar feature para o treinamento
    X_train_bins = train_por_bin[col_x].values.reshape(-1, 1).astype(np.float64)
    X_test_bins = test_por_bin[col_x].values.reshape(-1, 1).astype(np.float64)

    modelos = {}  # Dicionário para armazenar os modelos treinados

    # Configuração do Operon
    hyper = {
        "allowed_symbols": "add,sub,mul,aq,constant,variable,pow,exp,tanh",
        "random_state": SEED,
        "population_size": popsize,
        "max_length": 25,
        "max_depth": 25,
        "optimizer_iterations": 100,
        "model_selection_criterion": "bayesian_information_criterion",
        "objectives": ["r2", "length"],
        "n_threads": 12,
    }

    for largura in larguras:
        y = train_por_bin[f"{largura}_mean"].values.astype(np.float64)
        modelo = p.treinar_operon(hyper, X_train_bins, y)
        if select == "by_r2":
            p._operon_select_by_r2(modelo)
        modelos[f"{largura}_mean"] = modelo

    for largura in larguras:
        y = train_por_bin[f"{largura}_std"].values.astype(np.float64)
        modelo = p.treinar_operon(hyper, X_train_bins, y)
        if select == "by_r2":
            p._operon_select_by_r2(modelo)
        modelos[f"{largura}_std"] = modelo

    for l1, l2 in cov_pairs:
        y = train_por_bin[f"cov_{l1}_{l2}"].values.astype(np.float64)
        modelo = p.treinar_operon(hyper, X_train_bins, y)
        if select == "by_r2":
            p._operon_select_by_r2(modelo)
        modelos[f"cov_{l1}_{l2}"] = modelo

    metrics_data = []
    for model_name, model in modelos.items():
        # Predictions on train
        y_train_pred = model.predict(X_train_bins)
        y_train_true = train_por_bin[model_name].values.astype(np.float64)
        mse_train = mean_squared_error(y_train_true, y_train_pred)

        # Predictions on test
        y_test_pred = model.predict(X_test_bins)
        y_test_pred = np.nan_to_num(y_test_pred, nan=0.0, posinf=0.0, neginf=0.0)
        y_test_true = test_por_bin[model_name].values.astype(np.float64)
        mse_test = mean_squared_error(y_test_true, y_test_pred)
        r2_test = r2_score(y_test_true, y_test_pred)

        metrics_data.append(
            {
                "Model": model_name,
                "popsize": popsize,
                "complexity": model.stats_["model_complexity"],
                "bic": model.stats_["model_bic"],
                "R2 Train": model.stats_["model_r2"],
                "MSE Train": mse_train,
                "R2 Test": r2_test,
                "MSE Test": mse_test,
            }
        )

    df_metrics = pd.DataFrame(metrics_data)
    csv_out = f"results/{select}/metrics_{col_x}_{popsize}.csv"
    df_metrics.to_csv(csv_out, index=False)


### Gerar Amostras do Conjunto de Teste com Normal Multivariada
def gerar_amostras(col_x, SEED=4321):
    # Faz o split e o agrupamento dos dados
    train, test = train_test_split(
        dados[[col_x] + larguras], test_size=0.3, random_state=4321
    )

    # Cria subintervalos baseados em frequências (qcut)
    train[f"{col_x}_mean"] = pd.qcut(train[col_x], 150)
    test[f"{col_x}_mean"] = pd.qcut(test[col_x], 150)
    train_por_bin = pd.DataFrame(gera_stats(train, col_x))

    ### Treinar Operons para Média, Desvio Padrão e Covariâncias
    df_metrics_bic = pd.read_csv(f"results/by_bic/best_popsizes_by_bic.csv")
    modelos = {}

    # Selecionar feature para o treinamento
    X_train_bins = train_por_bin[col_x].values.reshape(-1, 1).astype(np.float64)
    X_test = test[[col_x]].astype(np.float64)
    n_samples = len(X_test)

    # Configuração do Operon
    hyper = {
        "random_state": 4321,
        "population_size": 1000,
        "allowed_symbols": "add,sub,mul,aq,constant,variable,pow,exp,tanh",
        "max_length": 25,
        "max_depth": 25,
        "optimizer_iterations": 100,
        "model_selection_criterion": "bayesian_information_criterion",
        "objectives": ["r2", "length"],
        "n_threads": 12,
    }

    for largura in larguras:
        y = train_por_bin[f"{largura}_mean"].values.astype(np.float64)
        df = df_metrics_bic.loc[
            (df_metrics_bic["model"] == f"{largura}_mean")
            & (df_metrics_bic["col_x"] == col_x)
        ]
        hyper["population_size"] = df["best_popsize"].values[0]
        modelo = p.treinar_operon(hyper, X_train_bins, y)
        modelos[f"{largura}_mean"] = modelo

    for largura in larguras:
        y = train_por_bin[f"{largura}_std"].values.astype(np.float64)
        df = df_metrics_bic.loc[
            (df_metrics_bic["model"] == f"{largura}_std")
            & (df_metrics_bic["col_x"] == col_x)
        ]
        hyper["population_size"] = df["best_popsize"].values[0]
        modelo = p.treinar_operon(hyper, X_train_bins, y)
        modelos[f"{largura}_std"] = modelo

    for l1, l2 in p.cov_pairs:
        y = train_por_bin[f"cov_{l1}_{l2}"].values.astype(np.float64)
        df = df_metrics_bic.loc[
            (df_metrics_bic["model"] == f"cov_{l1}_{l2}")
            & (df_metrics_bic["col_x"] == col_x)
        ]
        hyper["population_size"] = df["best_popsize"].values[0]
        modelo = p.treinar_operon(hyper, X_train_bins, y)
        modelos[f"cov_{l1}_{l2}"] = modelo

    # Fazer predições dos estimadores
    means_all = np.column_stack(
        [modelos[f"{nome}_mean"].predict(X_test) for nome in larguras]
    )
    stds_all = np.column_stack(
        [np.maximum(modelos[f"{nome}_std"].predict(X_test), 1e-6) for nome in larguras]
    )
    covs_all = {}
    for l1, l2 in cov_pairs:
        covs_all[(l1, l2)] = modelos[f"cov_{l1}_{l2}"].predict(X_test)

    n_correcoes = 0
    amostras_multivariadas = np.zeros((n_samples, 4))
    idx_map = {nome: j for j, nome in enumerate(larguras)}

    for i in range(n_samples):

        # Montar matriz de covariância
        mean_vector = means_all[i]
        stds = stds_all[i]
        cov_matrix = np.diag(stds**2)

        # Preencher covariâncias com validação
        for (l1_nome, l2_nome), cov_vals in covs_all.items():
            i1 = idx_map[l1_nome]
            i2 = idx_map[l2_nome]
            if np.isfinite(cov_vals[i]):
                cov_matrix[i1, i2] = cov_vals[i]
                cov_matrix[i2, i1] = cov_vals[i]

        # Remover infs e NaNs e garantir simetria
        cov_matrix = np.nan_to_num(cov_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        cov_matrix = (cov_matrix + cov_matrix.T) / 2

        # Regularização robusta
        corrigiu = False
        try:
            if not np.all(np.isfinite(cov_matrix)):
                raise ValueError("Matriz ainda contém valores não-finitos")

            # Corrigir autovalores negativos ou muito pequenos
            min_eigenval = 1e-8
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            if np.any(eigenvalues < min_eigenval):
                eigenvalues = np.maximum(eigenvalues, min_eigenval)
                cov_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
                corrigiu = True

            sample = multivariate_normal(
                mean=mean_vector, cov=cov_matrix, allow_singular=True
            ).rvs()

        except Exception as e:
            # Se tudo falhar, usar matriz diagonal
            cov_matrix = np.diag(stds**2)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            if not np.all(np.isfinite(cov_matrix)) or np.any(
                eigenvalues < min_eigenval
            ):
                corrigiu = False
                n_samples -= 1
                continue  # esquece e passa pra próxima combinação
            sample = multivariate_normal(
                mean=mean_vector, cov=cov_matrix, allow_singular=False
            ).rvs()
            corrigiu = True

        finally:
            if corrigiu:
                n_correcoes += 1

        amostras_multivariadas[i] = sample

    amostras = {
        col_x: X_test.values.flatten(),
        larguras[0]: amostras_multivariadas[:, 0],
        larguras[1]: amostras_multivariadas[:, 1],
        larguras[2]: amostras_multivariadas[:, 2],
        larguras[3]: amostras_multivariadas[:, 3],
    }

    # Salvando amostras em CSV
    df_amostras = pd.DataFrame(amostras)
    df_amostras = df_amostras.dropna()
    df_amostras = df_amostras.reset_index(drop=True)
    df_amostras.to_csv(f"results/amostras_{col_x}_{SEED}.csv", index=False)

    # Calcular razões do BPT
    test[T.nii_ha.value] = test[T.nii.value].values - test[T.ha.value].values
    test[T.oiii_hb.value] = test[T.oiii.value].values - test[T.hb.value].values
    df_amostras[T.nii_ha.value] = df_amostras[T.nii.value] - df_amostras[T.ha.value]
    df_amostras[T.oiii_hb.value] = df_amostras[T.oiii.value] - df_amostras[T.hb.value]

    # Matriz de correlação das amostras geradas
    corr_gerada = spearmanr(df_amostras[larguras]).correlation
    df_corrger = pd.DataFrame(corr_gerada, index=larguras, columns=larguras)

    # Matriz de correlação dos dados reais (test set)
    test_linhas = test[larguras].values
    corr_real = spearmanr(test_linhas).correlation
    df_corrtest = pd.DataFrame(corr_real, index=larguras, columns=larguras)

    # Diferença total absoluta
    diff_df = (df_corrger - df_corrtest).abs()
    diff = diff_df.values[np.triu_indices(4, k=1)].sum()

    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    p.plot_corr(
        corr_gerada,
        ax,
        larguras,
        title="Matriz de correlação das amostras normais\n"
        + r"Correções: %.1f%%, $\Delta$=%.4f" % (100 * n_correcoes / n_samples, diff),
        type="inf",
    )
    fig.savefig(f"results/corr_{col_x}_{SEED}.png")

    # Diagramas de diagnóstico coloridos pela feature
    p.show_bpt(
        df_amostras,
        col_x,
        title="Estimadores Operon + Amostras Normal4D",
        densities=True,
    )
    plt.savefig(f"results/bpt_{col_x}_{SEED}.png")

    p.show_whan(
        df_amostras,
        col_x,
        title="Estimadores Operon + Amostras Normal4D",
        densities=True,
    )
    plt.savefig(f"results/whan_{col_x}_{SEED}.png")


# Busca pelo popsize ideal enquanto treina as funções-resumo
def busca_equacoes():
    for pop in tqdm(range(100, 1001, 10)):
        grid_search(F.azmass.value, popsize=pop)
    for pop in tqdm(range(100, 1001, 10)):
        grid_search(F.atflux.value, popsize=pop)
    for pop in tqdm(range(100, 1001, 10)):
        grid_search(F.mass.value, popsize=pop)


# Gera pontos amostrais das normais, com equações já escolhidas
def busca_amostras():
    for _ in range(10):
        seed = int(np.random.random() * 1e6)
        gerar_amostras(F.azmass.value, seed)
        gerar_amostras(F.atflux.value, seed)
        gerar_amostras(F.mass.value, seed)


### MAIN
if __name__ == "__main__":
    busca_equacoes()
