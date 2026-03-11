# Libs
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal
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
    (larguras[2], larguras[3])
]

def gera_stats(dados_split, col_x):
    stats_list = []
    for bin_name, group in dados_split.groupby(f'{col_x}_mean'):
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
            stat_dict[f'{largura}_mean'] = means[i]
            stat_dict[f'{largura}_std'] = stds[i]
        for l1, l2 in cov_pairs:
            i1 = larguras.index(l1)
            i2 = larguras.index(l2)
            stat_dict[f'cov_{l1}_{l2}'] = cov_matrix[i1, i2]
        
        stats_list.append(stat_dict)
    return stats_list

def random_search(col_x, SEED=4321, popsize=2000, gens=1000):
    # Faz o split e o agrupamento dos dados
    train, test = train_test_split(dados[[col_x] + larguras], test_size=0.3, random_state=4321)
    
    # Cria subintervalos baseados em frequências (qcut)
    train[f'{col_x}_mean'] = pd.qcut(train[col_x], 150)
    test[f'{col_x}_mean'] = pd.qcut(test[col_x], 150)
    train_por_bin = pd.DataFrame(gera_stats(train, col_x))
    #test_por_bin = pd.DataFrame(gera_stats(test, col_x))

    # ## Treinar Operons para Média, Desvio Padrão e Covariâncias

    # Selecionar feature para o treinamento
    X_train_bins = train_por_bin[col_x].values.reshape(-1, 1).astype(np.float64)

    # Configuração do Operon
    hyper = {
        'allowed_symbols': "add,sub,mul,aq,constant,variable,square,pow,abs,sqrt,exp,log,tanh",
        'random_state': SEED,
        'population_size': popsize,
        'generations': gens,
        'optimizer_iterations': 500,
        'max_depth': 20,
        'n_threads': 12,
        'model_selection_criterion': 'bayesian_information_criterion',
        'objectives': ['r2', 'length']
    }

    # Dicionário para armazenar os modelos
    modelos = {}
    print("\n[1/3] TREINANDO MODELOS DE MÉDIA")
    print("-" * 80)
    for largura in larguras:
        print(f"Treinando MÉDIA para {largura}...")
        y = train_por_bin[f'{largura}_mean'].values.astype(np.float64)
        modelo = p.treinar_operon(hyper, X_train_bins, y)
        modelos[f'{largura}_mean'] = modelo
    
    print("\n[2/3] TREINANDO MODELOS DE DESVIO PADRÃO")
    print("-" * 80)
    for largura in larguras:
        print(f"Treinando DESVIO PADRÃO para {largura}...")
        y = train_por_bin[f'{largura}_std'].values.astype(np.float64)
        modelo = p.treinar_operon(hyper, X_train_bins, y)
        modelos[f'{largura}_std'] = modelo

    print("\n[3/3] TREINANDO MODELOS DE COVARIÂNCIA")
    print("-" * 80)
    for l1, l2 in cov_pairs:
        print(f"Treinando COVARIÂNCIA para {l1} x {l2}...")
        y = train_por_bin[f'cov_{l1}_{l2}'].values.astype(np.float64)
        modelo = p.treinar_operon(hyper, X_train_bins, y)
        modelos[f'cov_{l1}_{l2}'] = modelo

    p.salva_equacoes_html(modelos, col_x, f"n4d_{col_x}_{SEED}_{popsize}_{gens}")

    # ## Gerar Amostras do Conjunto de Teste com Normal Multivariada
    #X_test_bins = test_por_bin[col_x].values.reshape(-1, 1)
    X_test = test[[col_x]].astype(np.float64)
    n_samples = len(X_test)

    print(f"\nEstimando parâmetros de {n_samples} pontos do conjunto de validação para a Normal Multivariada...")
    means_all = np.column_stack([modelos[f'{nome}_mean'].predict(X_test) for nome in larguras])
    stds_all = np.column_stack([np.maximum(modelos[f'{nome}_std'].predict(X_test), 1e-6) for nome in larguras])
    covs_all = {}
    for l1, l2 in cov_pairs:
        covs_all[(l1, l2)] = modelos[f'cov_{l1}_{l2}'].predict(X_test)
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
        for (l1_nome, l2_nome), cov_vals in covs_all.items():
            i1 = idx_map[l1_nome]
            i2 = idx_map[l2_nome]
            cov_matrix[i1, i2] = cov_vals[i]
            cov_matrix[i2, i1] = cov_vals[i]
        
        # Regularização robusta
        cov_matrix = np.nan_to_num(cov_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        corrigiu = False
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            if np.any(eigenvalues < 1e-8):
                eigenvalues = np.maximum(eigenvalues, 1e-8)
                cov_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
                corrigiu = True
            sample = multivariate_normal(mean=mean_vector, cov=cov_matrix, allow_singular=True).rvs()
        except:
            cov_matrix = np.diag(stds**2)
            sample = multivariate_normal(mean=mean_vector, cov=cov_matrix, allow_singular=True).rvs()
            corrigiu = True
        finally:
            if corrigiu: n_correcoes += 1
        
        amostras_multivariadas[i] = sample

    amostras = {
        col_x: X_test,
        larguras[0]: amostras_multivariadas[:, 0],
        larguras[1]: amostras_multivariadas[:, 1],
        larguras[2]: amostras_multivariadas[:, 2],
        larguras[3]: amostras_multivariadas[:, 3]
    }

    print("\nAmostragem multivariada concluída!")
    print(f"   - Amostras: {n_samples}")
    print(f"   - Matrizes corrigidas: {n_correcoes} ({100*n_correcoes/n_samples:.1f}%)\n")

    print(f"Estatísticas para SEED {SEED}:\n")
    for nome in larguras:
        valores = amostras[nome]
        print(f"  {nome:12s}: média={valores.mean():7.3f}, std={valores.std():6.3f}, "
            f"min={valores.min():7.3f}, max={valores.max():7.3f}")

    # ## Calcular razões do BPT
    test[T.nii_ha.value] = test[T.nii.value].values - test[T.ha.value].values
    test[T.oiii_hb.value] = test[T.oiii.value].values - test[T.hb.value].values
    amostras[T.nii_ha.value] = amostras[T.nii.value] - amostras[T.ha.value]
    amostras[T.oiii_hb.value] = amostras[T.oiii.value] - amostras[T.hb.value]

    # Verificar se as correlações foram preservadas
    print("\nVERIFICAÇÃO: Correlações das amostras geradas")
    print("="*80)

    # Matriz de correlação das amostras geradas
    amostras_array = np.column_stack([amostras[nome] for nome in larguras])
    corr_gerada = np.corrcoef(amostras_array, rowvar=False)
    df_amostras = pd.DataFrame(corr_gerada, index=larguras, columns=larguras)

    # Matriz de correlação dos dados reais (test set)
    test_linhas = test[larguras].values
    corr_real = np.corrcoef(test_linhas, rowvar=False)
    df_test = pd.DataFrame(corr_real, index=larguras, columns=larguras)

    print("\nDiferença Absoluta - (Amostras - Teste):")
    diff_df = (df_amostras - df_test).abs()
    print(diff_df.round(3))

    diff = diff_df.values[np.triu_indices(4, k=1)].sum()
    print(f"\nDiferença Absoluta Média: {diff:.4f}")
    print("="*80)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    p.plot_corr(corr_real, axes[0], larguras, title='Matriz de correlação dos dados de teste', type='inf')
    p.plot_corr(corr_gerada, axes[1], larguras, title=r'Matriz de correlação das amostras normais, $\Delta$=%.4f'%diff, type='inf')
    fig.savefig(f"results/corr_{col_x}_{SEED}_{popsize}_{gens}.png")

    # ### Diagramas de diagnóstico coloridos pela feature

    p.show_bpt(amostras, col_x, title="Estimadores Operon + Amostras Normal4D")
    plt.savefig(f"results/bpt_{col_x}_{SEED}_{popsize}_{gens}.png")

    p.show_whan(amostras, col_x, title="Estimadores Operon + Amostras Normal4D")
    plt.savefig(f"results/whan_{col_x}_{SEED}_{popsize}_{gens}.png")


# ### MAIN
if __name__ == "__main__":
    for pop in tqdm(range(1000, 5001, 250)):
        for gen in range(1000, 5001, 250):
            random_search(F.azmass.value, 4321, pop, gen)
            random_search(F.atflux.value, 4321, pop, gen)
            random_search(F.mass.value, 4321, pop, gen)
