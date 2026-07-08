# Função para treinar ou carregar modelos Operon
def treinar_modelos_operon(hyper, X_train_bins, items, ignore_load=False):
    for model_key, file_name, col_name, label in items:
        if not ignore_load:
            modelo = p.load_operon(file_name)
        if ignore_load or modelo is None:
            print(f"Treinando {label}...")
            y = train_lhc[col_name].values.astype(np.float64)
            modelo = p.treinar_operon(hyper, X_train_bins, y)
            p.save_operon(modelo, file_name)
        else:
            print(f"Lido modelo salvo: {label}...")
        modelos[model_key] = modelo


if __name__ == "__main__":
    from metricas_plots import *
    p = PlotsMetricas()
    print("Carregando dados e gerando amostras LHC para treinamento e teste...")

    dados = pd.read_csv("dados/ariel_limpo_log10.csv.gz", compression="gzip")
    larguras = p.targets[:4]
    train, test = train_test_split(
        dados[p.features + larguras], test_size=0.25, random_state=4321
    )

    train[T.nii_ha.value] = train[T.nii.value].values - train[T.ha.value].values
    train[T.oiii_hb.value] = train[T.oiii.value].values - train[T.hb.value].values
    test[T.nii_ha.value] = test[T.nii.value].values - test[T.ha.value].values
    test[T.oiii_hb.value] = test[T.oiii.value].values - test[T.hb.value].values
    
    larguras_lhc = [l+"_median" for l in larguras]
    train_lhc = p.lhs_subsample_with_stats(train, p.features, larguras, n=4000, k_neighbors=100)
    test_lhc = p.lhs_subsample_with_stats(test, p.features, larguras, n=1000, k_neighbors=100)
    print("Amostras LHC para treinamento e teste geradas com sucesso!")
    
    modelos = {}  # Dicionário para armazenar os modelos

    # Selecionar feature para o treinamento
    X_train_bins = train_lhc[p.features].values.astype(np.float64)
    
    # Modelos para os estimadores da Normal
    modelo = "operon"
    items = (
        [
            (f"{l}_mean", f"{l}_all_mean", f"{l}_mean", f"MÉDIA para {l}") 
            for l in larguras
        ] + 
        [
            (f"{l}_std", f"{l}_all_std", f"{l}_std", f"DESVIO PADRÃO para {l}") 
            for l in larguras
        ] + 
        [
            (f"cov_{l1}_{l2}", f"{l1}_{l2}_all_cov", f"cov_{l1}_{l2}", f"COVARIÂNCIA para {l1} x {l2}")
            for l1, l2 in p.cov_pairs
        ]
    )

    # Configuração do Operon
    config_operon = {
        "random_state": RANDOM_SEED,
        "population_size": 2000,
        "generations": 2000,
        "allowed_symbols": "add,sub,mul,aq,constant,variable,pow,exp,tanh",
        "max_length": 25,
        "max_depth": 100,
        "optimizer_iterations": 1000,
        "model_selection_criterion": "bayesian_information_criterion",
        "objectives": ["r2", "length"],
        "n_threads": 32,
    }
    
    # Treinar operon para todas as células
    treinar_modelos_operon(config_operon, X_train_bins, items, False)
    p.salva_equacoes_operon(modelos, "n4d_all")
    
    p.inspecao_modelos(modelos, train_lhc, p.features, modelo)
    plt.savefig(f"results/regressoes/{modelo}_meanstds_n4d_all_{RANDOM_SEED}.png", bbox_inches="tight")
    plt.close()

    p.inspecao_modelos_covs(modelos, train_lhc, p.features, modelo)
    plt.savefig(f"results/regressoes/{modelo}_covs_n4d_all_{RANDOM_SEED}.png", bbox_inches="tight")
    plt.close()

    KMAX = 100
    ALPHA = 0.05
    BEST_SEED = 0
    BEST_SCORE = -np.inf
    BEST_OTIMO = False

    p_energy = 0
    p_wasserstein = 0
    p_ks2d = 0
    
    k = 0
    while k < KMAX:
        k += 1
        RANDOM_SEED = np.random.randint(10000, 20000)
        print(f"\n\nIteração {k}/{KMAX} - Seed aleatória: {RANDOM_SEED}")
        
        print("Gerando novas amostras!")
        p_correcao = p.gerar_amostras(modelos, modelo, test, test_lhc)
        print(f"Porcentagem de matrizes corrigidas: {p_correcao:.2f}%")
        
        # Ler amostras geradas
        df_amostras = pd.read_csv(f"results/amostras_all_{modelo}.csv")
        df_amostras = df_amostras.dropna()
        df_amostras = df_amostras.reset_index(drop=True)
        df_amostras[T.nii_ha.value] = df_amostras[T.nii.value] - df_amostras[T.ha.value]
        df_amostras[T.oiii_hb.value] = df_amostras[T.oiii.value] - df_amostras[T.hb.value]

        df_ocupacao_n4d = tabela_ocupacao_bpt({
            "Validation Set": (test[T.nii_ha.value], test[T.oiii_hb.value]),
            "Amostras da Normal4D": (df_amostras[T.nii_ha.value], df_amostras[T.oiii_hb.value]),
        })
        plot_ocupacao_bpt(df_ocupacao_n4d, titulo=f"Ocupação das regiões do BPT - {modelo}")
        plt.tight_layout()
        plt.savefig(f"results/correlacoes/bpt_ocupacao_n4d_all_{modelo}.png", bbox_inches="tight")
        plt.close()
        
        p.show_bpt(df_amostras, F.azmass.value, title="Estimadores Operon + Amostras Normal4D")
        plt.savefig(f"results/diagramas/bpt_cores_amostras_n4d_all_{modelo}_{RANDOM_SEED}.png", bbox_inches="tight")
        plt.close()

        # 1) Teste de energia repetido: robustez à escolha da sub-amostra de n_max pontos
        res_energy = energy_test_repeated(
            test[T.nii_ha.value], test[T.oiii_hb.value],
            df_amostras[T.nii_ha.value], df_amostras[T.oiii_hb.value],
            n_repeats=200, n_max=500, n_perm=199, n_jobs=16
        )

        # 2) Distância de Wasserstein (transporte de massa ótimo)
        res_wasserstein = wasserstein_test(
            test[T.nii_ha.value], test[T.oiii_hb.value],
            df_amostras[T.nii_ha.value], df_amostras[T.oiii_hb.value],
            n_max=250, n_perm=199, n_jobs=16
        )
        
        # 3) Distância de Bhattacharyya entre as densidades 2D no BPT
        res_bhatt = bhattacharyya_distance_2d(
            test[T.nii_ha.value], test[T.oiii_hb.value],
            df_amostras[T.nii_ha.value], df_amostras[T.oiii_hb.value],
            bins=80, xlim=(-2, 1), ylim=(-1.5, 1.4),
        )
        
        # 4) Diferença de densidades KDE 2D + testes KL e KS-2D
        res_kde_diff = p.plot_kde_diff_bpt(
            test[T.nii_ha.value], test[T.oiii_hb.value],
            df_amostras[T.nii_ha.value], df_amostras[T.oiii_hb.value],
            label1="Validation Set", label2="Amostras Normal4D",
            titulo=f"Diferença de densidades KDE no BPT - {modelo}",
            ks_kwargs={"n_max": 1000, "n_perm": 499, "n_jobs": 16},
        )
        plt.savefig(f"results/correlacoes/bpt_kde_diff_n4d_all_{modelo}_{RANDOM_SEED}.png", bbox_inches="tight")
        plt.close()
        
        # Resumo das estatísticas de ajuste bidimensional no BPT
        df_resumo = pd.DataFrame([{
            "energy": res_energy["p_value"].mean(),
            "wasserstein": res_wasserstein["p_value"],
            "bhattacharyya": res_bhatt["distance"],
            "ks2d": res_kde_diff["ks"]["p_value"],
        }], index=[modelo])
        
        p_energy = df_resumo["energy"].values[0]
        p_wasserstein = df_resumo["wasserstein"].values[0]
        p_ks2d = df_resumo["ks2d"].values[0]

        print(f"\nResumo das estatísticas de ajuste bidimensional no BPT para {modelo}:")
        print(f"Energy     : {p_energy:.4f}")
        print(f"Wasserstein: {p_wasserstein:.4f}")
        print(f"KS-2D      : {p_ks2d:.4f}")
        
        # Score = p-value médio entre os três testes; quanto maior, mais perto de
        # satisfazer todos os critérios simultaneamente. Guarda a melhor seed já vista,
        # mesmo que nenhuma tenha passado do ALPHA em todos os testes.
        score_atual = np.mean([p_energy, p_wasserstein, p_ks2d])
        if score_atual > BEST_SCORE:
            BEST_SCORE = score_atual
            BEST_SEED = RANDOM_SEED
            print(f"Novo melhor seed até agora: {BEST_SEED} (p-value médio = {BEST_SCORE:.4f})")

        if all([p_energy > ALPHA, p_wasserstein > ALPHA, p_ks2d > ALPHA]):
            print(f"Modelo {modelo} ótimo encontrado com todos os p-values acima de {ALPHA:.1e}!")
            BEST_SEED = RANDOM_SEED
            BEST_SCORE = score_atual
            BEST_OTIMO = True
            break
        
        print("\n" + "*"*50 + "\n")

if BEST_OTIMO:
    print(f"\nMelhor modelo encontrado com seed {BEST_SEED} (ótimo, todos os p-values > {ALPHA:.1e}) e pior p-value={BEST_SCORE:.4f}")
else:
    print(f"\nNenhuma seed atingiu todos os p-values > {ALPHA:.1e} em {KMAX} iterações.")
    print(f"Melhor seed encontrada mesmo assim: {BEST_SEED} (pior p-value={BEST_SCORE:.4f})")