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
modelos = {}  # Dicionário para armazenar os modelos treinados

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
        try:
            r2_train = model.stats_["model_r2"]
        except:
            r2_train = r2_score(y_train_true, y_train_pred)
            model.stats_["model_r2"] = r2_train

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
                "R2 Train": r2_train,
                "MSE Train": mse_train,
                "R2 Test": r2_test,
                "MSE Test": mse_test,
            }
        )

    df_metrics = pd.DataFrame(metrics_data)
    csv_out = f"results/{select}/metrics_{col_x}_{popsize}.csv"
    df_metrics.to_csv(csv_out, index=False)


# ## Gerar Amostras do Conjunto de Teste com Normal Multivariada
def gerar_amostras(col_x, SEED=4321):
    # Faz o split e o agrupamento dos dados
    train, test = train_test_split(
        dados[[col_x] + larguras], test_size=0.3, random_state=4321
    )
    X_test = test[[col_x]].astype(np.float64)
    n_samples = len(X_test)

    print(
        f"\nEstimando parâmetros de {n_samples} pontos do conjunto de validação para a Normal Multivariada..."
    )
    means_all = np.column_stack(
        [modelos[f"{nome}_mean"].predict(X_test) for nome in larguras]
    )
    stds_all = np.column_stack(
        [np.maximum(modelos[f"{nome}_std"].predict(X_test), 1e-6) for nome in larguras]
    )
    covs_all = {}
    for l1, l2 in cov_pairs:
        covs_all[(l1, l2)] = modelos[f"cov_{l1}_{l2}"].predict(X_test)
    print("Predições dos estimadores concluídas!")

    n_correcoes = 0
    amostras_multivariadas = np.zeros((n_samples, 4))
    idx_map = {nome: j for j, nome in enumerate(larguras)}
    print("\nAmostragem multivariada iniciada..\n")

    for i in range(n_samples):
        if i % 5000 == 0 and i > 0:
            print(f"  Progresso: {i}/{n_samples} ({100*i/n_samples:.1f}%)")

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
                print(
                    "Abortando combinação problemática, desvios diagonais inválidos!\n"
                )
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

    print("\nAmostragem multivariada concluída!")
    print(f"   - Amostras: {n_samples}")
    print(
        f"   - Matrizes corrigidas: {n_correcoes} ({100*n_correcoes/n_samples:.1f}%)\n"
    )
    df_amostras = pd.DataFrame(amostras)
    df_amostras.to_csv(f"results/amostras_{col_x}_{SEED}.csv", index=False)

    print(f"Estatísticas:\n")
    for nome in larguras:
        valores = df_amostras[nome]
        print(
            f"  {nome:12s}: média={valores.mean():7.3f}, std={valores.std():6.3f}, "
            f"min={valores.min():7.3f}, max={valores.max():7.3f}"
        )

    df_amostras = df_amostras.dropna()
    df_amostras = df_amostras.reset_index(drop=True)

    # Calcular razões do BPT
    test[T.nii_ha.value] = test[T.nii.value].values - test[T.ha.value].values
    test[T.oiii_hb.value] = test[T.oiii.value].values - test[T.hb.value].values
    df_amostras[T.nii_ha.value] = df_amostras[T.nii.value] - df_amostras[T.ha.value]
    df_amostras[T.oiii_hb.value] = df_amostras[T.oiii.value] - df_amostras[T.hb.value]

    # Verificar se as correlações foram preservadas
    print("\nVERIFICAÇÃO: Correlações das amostras geradas")
    print("=" * 80)

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

    print("\nDiferença Absoluta - (Amostras - Teste):")
    print(diff_df.round(3))
    print(f"\nDiferença Absoluta: {diff:.4f}")
    print("=" * 80)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    p.plot_corr(
        corr_real,
        axes[0],
        larguras,
        title="Matriz de correlação dos dados de teste",
        type="inf",
    )
    p.plot_corr(
        corr_gerada,
        axes[1],
        larguras,
        title=r"Matriz de correlação das amostras normais, $\Delta$=%.4f" % diff,
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
    pops = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    for pop in tqdm(pops):
        grid_search(F.azmass.value, popsize=pop)
        grid_search(F.atflux.value, popsize=pop)
        grid_search(F.mass.value, popsize=pop)


# Gera pontos amostrais das normais, com equações já escolhidas
def busca_amostras():
    pass


### MAIN
if __name__ == "__main__":
    busca_equacoes()
