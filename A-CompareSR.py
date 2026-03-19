import os
import time
import sympy as sp
import numpy as np
import pandas as pd
from datetime import datetime
import smplotlib  # type: ignore
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from metricas_plots import PlotsMetricas, T, F

np.seterr(all="ignore")
p = PlotsMetricas()

rf_config = {
    "algorithm": "RandomForest",
    "kwargs": {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 2,
        "random_state": 4321,
        "n_jobs": 12,
    },
}

operon_config = {
    "algorithm": "Operon",
    "kwargs": {
        "random_state": 4321,
        "population_size": 1000,
        "allowed_symbols": "add,sub,mul,div,constant,variable,square,exp,tanh",
        "max_length": 25,
        "max_depth": 25,
        "optimizer": "lbfgs",
        "model_selection_criterion": "bayesian_information_criterion",
        "objectives": ["r2", "length"],
        "n_threads": 12,
    },
}

pysr_config = {
    "algorithm": "PySR",
    "kwargs": {
        "populations": 10,
        "population_size": 100,
        "binary_operators": ["+", "-", "*", "/"],
        "unary_operators": ["square", "exp", "tanh"],
        "maxdepth": 25,
        "model_selection": "accuracy",
        "parallelism": "multiprocessing",
        "procs": 12,
        "verbosity": 1,
    },
}

eggp_config = {
    "algorithm": "EGGP",
    "kwargs": {
        "gen": 100,
        "nPop": 100,
        "maxSize": 25,
        "nTournament": 5,
        "nonterminals": "add,sub,mul,div,square,exp,tanh",
        "loss": "MSE"
    },
}


def get_feature_target_names():
    features = [F.azmass.value, F.atflux.value, F.mass.value]
    targets = [T.nii.value, T.ha.value, T.oiii.value, T.hb.value]
    return features, targets


def train_rf_models(X_train, y_train, config):
    from sklearn.ensemble import RandomForestRegressor

    rf_kwargs = config["kwargs"]
    models = []
    elapseds = []
    for i in range(y_train.shape[1]):
        start = time.perf_counter()
        model = RandomForestRegressor(**rf_kwargs)
        model.fit(X_train, y_train[:, i])
        elapsed = time.perf_counter() - start
        elapseds.append(elapsed)
        models.append(model)
    return models, elapseds


def train_operon_models(X_train, y_train, config):
    from pyoperon.sklearn import SymbolicRegressor

    operon_cfg = config["kwargs"]
    models = []
    elapseds = []
    for i in range(y_train.shape[1]):
        start = time.perf_counter()  # segundos
        modelo = SymbolicRegressor(**operon_cfg)
        modelo.fit(X_train, y_train[:, i])
        elapsed = time.perf_counter() - start
        elapseds.append(elapsed)
        models.append(modelo)
    return models, elapseds


def train_pysr_models(X_train, y_train, config):
    from pysr import PySRRegressor

    pysr_kwargs = config["kwargs"]
    models = []
    elapseds = []
    for i in range(y_train.shape[1]):
        start = time.perf_counter()  # segundos
        model = PySRRegressor(
            **pysr_kwargs
            # , extra_sympy_mappings={
            # "sqrtabs": lambda x: sp.sqrt(sp.Abs(x)),
            # "logabs": lambda x: sp.log(sp.Abs(x))}
        )
        model.fit(X_train, y_train[:, i])
        elapsed = time.perf_counter() - start
        elapseds.append(elapsed)
        models.append(model)
    return models, elapseds


def read_pysr_models(config, targets, prefix=""):
    from pysr import PySRRegressor

    models = []
    elapseds = []
    for target in targets:
        start = time.perf_counter()  # segundos
        model = PySRRegressor.from_file(
            run_directory=f"results/pysr/{prefix}{target}",
            model_selection=config["kwargs"]["model_selection"],
        )
        elapsed = time.perf_counter() - start
        elapseds.append(elapsed)
        models.append(model)
    return models, elapseds


# Divide X e y em n_bins subintervalos de igual frequência baseado em y.
def split_random_stratified(X, y, n_bins=10, random_state=4321):
    rng = np.random.default_rng(random_state)

    # Estratos baseados na distribuição de y
    strata = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")

    # Para cada estrato, distribui as amostras aleatoriamente entre os n_bins subconjuntos
    indices = [[] for _ in range(n_bins)]
    for s in range(strata.max() + 1):
        stratum_idx = np.where(strata == s)[0]
        rng.shuffle(stratum_idx)
        for i, idx in enumerate(stratum_idx):
            indices[i % n_bins].append(idx)

    Xs, ys = [], []
    for idx in indices:
        idx = np.array(idx)
        Xs.append(X[idx])
        ys.append(y[idx])

    return Xs, ys


def train_eggp_models(X_train, y_train, config):
    from eggp import EGGP

    models = []
    elapseds = []
    for i in range(y_train.shape[1]):
        # Partição em 10 ilhas
        Xs_train, ys_train = split_random_stratified(X_train, y_train[:, i], n_bins=10)

        # Verifica balanço das ilhas
        for j, (Xi, yi) in enumerate(zip(Xs_train, ys_train)):
            print(f"Ilha {j:2d}: n={len(yi):6d}  y=[{yi.min():.3f}, {yi.max():.3f}]")

        # Treino multi-view
        start = time.perf_counter()  # segundos
        model = EGGP(**config["kwargs"])
        model.fit_mvsr(Xs_train, ys_train)
        elapsed = time.perf_counter() - start
        elapseds.append(elapsed)
        models.append(model)
    return models, elapseds


def evaluate_models(
    models, predict_fn, X_train, X_test, y_train, y_test, algorithm, targets, elapsed
):
    rows = []
    if models is None:
        return rows

    for i, model in enumerate(models):
        # Previsões no conjunto de treinamento
        y_pred_train = predict_fn(model, X_train)
        y_pred_train = np.nan_to_num(y_pred_train, nan=0.0, posinf=0.0, neginf=0.0)

        # Previsões no conjunto de validação
        y_pred_test = predict_fn(model, X_test)
        y_pred_test = np.nan_to_num(y_pred_test, nan=0.0, posinf=0.0, neginf=0.0)

        # Calcula métricas
        mse_train = mean_squared_error(y_train[:, i], y_pred_train)
        mse_test = mean_squared_error(y_test[:, i], y_pred_test)
        r2_train = r2_score(y_train[:, i], y_pred_train)
        r2_test = r2_score(y_test[:, i], y_pred_test)

        # Obtém equações e complexidade
        if algorithm == "PySR":
            complexy = model.get_best()["complexity"]
            expr = model.sympy()
            equations = expr.xreplace({n: round(n, 6) for n in expr.atoms(sp.Float)})
        elif algorithm == "Operon":
            complexy = model.stats_["model_complexity"]
            equations = model.get_model_string(model.model_, precision=6).replace(
                "^", "**"
            )
        elif algorithm == "RandomForest":
            avg_depth = int(np.mean([e.get_depth() for e in model.estimators_]))
            complexy = model.n_estimators * avg_depth
            equations = "N/A"
        elif algorithm == "egGP":
            best = model.results.iloc[-1]
            complexy = best.size
            equations = best.Expression

        # Exibe gráfico real X predito
        plt.subplots(figsize=(7, 7))
        plt.scatter(
            y_test[:, i], y_pred_test, alpha=0.3, s=2, edgecolors=None, color="gray"
        )
        plt.plot(
            [y_test[:, i].min(), y_test[:, i].max()],
            [y_test[:, i].min(), y_test[:, i].max()],
            "r--",
        )
        p.curvas_densidade(y_test[:, i], y_pred_test)
        plt.xlabel("Real")
        plt.ylabel("Predito")
        plt.title(p.unidades[targets[i]])
        plt.xlim([y_test[:, i].min(), y_test[:, i].max()])
        plt.ylim([y_test[:, i].min(), y_test[:, i].max()])
        plt.savefig(f"results/compare_sr/fit_{algorithm}_{targets[i]}.png")
        plt.close()

        # Salva amostras geradas para posterior diagrama BPT
        df_amostra = pd.DataFrame(y_pred_test, columns=[targets[i]])
        df_amostra.to_csv(
            f"results/compare_sr/amostras_{algorithm}_{targets[i]}.csv", index=False
        )

        # Retorna métricas
        rows.append(
            {
                "algorithm": algorithm,
                "time": f"{elapsed[i]:.0f}",
                "target": targets[i],
                "r2_train": r2_train,
                "mse_train": mse_train,
                "r2_test": r2_test,
                "mse_test": mse_test,
                "complexity": complexy,
                "equation": equations,
            }
        )
    return rows


def operon_predict(model, X):
    return model.predict(X)


def pysr_predict(model, X):
    return model.predict(X)


def rf_predict(model, X):
    return model.predict(X)


def eggp_predict(model, X):
    return model.predict(X)


def run_comparison(pysr=None, operon=False, rf=False, eggp=False):
    os.makedirs("results/compare_sr", exist_ok=True)

    print("\n Carregando dados...")
    df = pd.read_csv("dados/ariel_limpo_log10.csv.gz", compression="gzip")
    features, targets = get_feature_target_names()
    for col in features + targets:
        if col not in df.columns:
            raise ValueError(f"Coluna esperada nao encontrada no arquivo: {col}")

    X = df[features].astype(float).values
    y = df[targets].astype(float).values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=4321
    )
    scaler_x = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)
    results_rows = []

    if rf:
        print("\n Treinando RandomForest...")
        rf_models, elapsed = train_rf_models(X_train_scaled, y_train, rf_config)
        results_rows.extend(
            evaluate_models(
                rf_models,
                rf_predict,
                X_train_scaled,
                X_test_scaled,
                y_train,
                y_test,
                "RandomForest",
                targets,
                elapsed,
            )
        )
        gerar_diagramas("RandomForest")
        combinar_fits("RandomForest")

    if operon:
        print("\n Treinando Operon...")
        operon_models, elapsed = train_operon_models(
            X_train_scaled, y_train, operon_config
        )
        results_rows.extend(
            evaluate_models(
                operon_models,
                operon_predict,
                X_train_scaled,
                X_test_scaled,
                y_train,
                y_test,
                "Operon",
                targets,
                elapsed,
            )
        )
        gerar_diagramas("Operon")
        combinar_fits("Operon")

    if eggp:
        print("\n Treinando egGP...")
        eggp_models, elapsed = train_eggp_models(X_train_scaled, y_train, eggp_config)
        results_rows.extend(
            evaluate_models(
                eggp_models,
                eggp_predict,
                X_train_scaled,
                X_test_scaled,
                y_train,
                y_test,
                "egGP",
                targets,
                elapsed,
            )
        )
        gerar_diagramas("egGP")
        combinar_fits("egGP")

    if pysr is not None:
        print("\n Treinando PySR...")
        if pysr == "train":
            pysr_models, elapsed = train_pysr_models(
                X_train_scaled, y_train, pysr_config
            )
        elif pysr == "read":
            pysr_models, elapsed = read_pysr_models(pysr_config, targets, "20260315_")
        if pysr_models is not None:
            results_rows.extend(
                evaluate_models(
                    pysr_models,
                    pysr_predict,
                    X_train_scaled,
                    X_test_scaled,
                    y_train,
                    y_test,
                    "PySR",
                    targets,
                    elapsed,
                )
            )
            gerar_diagramas("PySR")
            combinar_fits("PySR")

    if rf or operon or eggp or (pysr is not None and pysr != "read"):
        df_results = pd.DataFrame(results_rows)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join("results/compare_sr", f"results_{timestamp}.csv")
        df_results.to_csv(output_path, index=False)
        print("\nResultados salvos em:", output_path)
    else:
        print("Nenhum modelo foi ativado para o teste!")


def histogramas_validation():
    # Histogramas do conjunto de validação
    df = pd.read_csv("dados/ariel_limpo_log10.csv.gz", compression="gzip")
    features, targets = get_feature_target_names()
    X = df[features].astype(float).values
    y = df[targets].astype(float).values
    _, _, _, y_test = train_test_split(X, y, test_size=0.3, random_state=4321)
    _, ax = plt.subplots(2, 2, figsize=(16, 12))
    p.histogram_v(
        y_test[:, 0],
        "Validation Set for " + p.unidades[targets[0]],
        ax[0, 0],
        cor="darkblue",
        bins=80,
        lim=(-1, 2.5),
    )
    p.histogram_v(
        y_test[:, 1],
        "Validation Set for " + p.unidades[targets[1]],
        ax[0, 1],
        cor="darkblue",
        bins=80,
        lim=(-1, 2.5),
    )
    p.histogram_v(
        y_test[:, 2],
        "Validation Set for " + p.unidades[targets[2]],
        ax[1, 0],
        cor="darkblue",
        bins=80,
        lim=(-1, 2.5),
    )
    p.histogram_v(
        y_test[:, 3],
        "Validation Set for " + p.unidades[targets[3]],
        ax[1, 1],
        cor="darkblue",
        bins=80,
        lim=(-1, 2.5),
    )
    plt.savefig(f"results/compare_sr/histogram_validation.png")
    plt.close()


def gerar_diagramas(algo):
    try:
        series = []
        _, targets = get_feature_target_names()
        for target in targets:
            s = pd.read_csv(f"results/compare_sr/amostras_{algo}_{target}.csv").iloc[
                :, 0
            ]
            s.name = target
            series.append(s)
        df_amostras = pd.concat(series, axis=1)
        df_amostras["nii_halpha_ew"] = (
            df_amostras["nii_6584_ew"] - df_amostras["halpha_ew"]
        )
        df_amostras["oiii_hbeta_ew"] = (
            df_amostras["oiii_5007_ew"] - df_amostras["hbeta_ew"]
        )
        df_amostras.to_csv(f"results/compare_sr/amostras_{algo}.csv", index=False)
        for target in targets:
            os.remove(f"results/compare_sr/amostras_{algo}_{target}.csv")
    except:
        print(f"Ocorreu um erro ao juntar as amostras do {algo}!")
        try:
            df_amostras = pd.read_csv(f"results/compare_sr/amostras_{algo}.csv")
        except:
            return

    # Histogramas
    _, ax = plt.subplots(2, 2, figsize=(16, 12))
    p.histogram_v(
        df_amostras[T.nii.value],
        f"{algo} Sample for " + p.unidades[T.nii.value],
        ax[0, 0],
        cor="darkgreen",
        bins=80,
        lim=(-1, 2.5),
    )
    p.histogram_v(
        df_amostras[T.ha.value],
        f"{algo} Sample for " + p.unidades[T.ha.value],
        ax[0, 1],
        cor="darkgreen",
        bins=80,
        lim=(-1, 2.5),
    )
    p.histogram_v(
        df_amostras[T.oiii.value],
        f"{algo} Sample for " + p.unidades[T.oiii.value],
        ax[1, 0],
        cor="darkgreen",
        bins=80,
        lim=(-1, 2.5),
    )
    p.histogram_v(
        df_amostras[T.hb.value],
        f"{algo} Sample for " + p.unidades[T.hb.value],
        ax[1, 1],
        cor="darkgreen",
        bins=80,
        lim=(-1, 2.5),
    )
    plt.savefig(f"results/compare_sr/histogram_{algo}.png")
    plt.close()

    # Diagramas
    p.show_bpt(df_amostras, title=f"Amostras do {algo}", densities=True, show=False)
    plt.savefig(f"results/compare_sr/bpt_{algo}.png")
    plt.close()
    p.show_whan(df_amostras, title=f"Amostras do {algo}", densities=True, show=False)
    plt.savefig(f"results/compare_sr/whan_{algo}.png")
    plt.close()


def combinar_fits(algorithm):
    from PIL import Image

    _, targets = get_feature_target_names()
    # targets = [nii, ha, oiii, hb] → layout 2x2 já na ordem correta
    ordem = [targets[0], targets[1], targets[2], targets[3]]  # nii, ha, oiii, hb

    imgs = []
    for target in ordem:
        path = f"results/compare_sr/fit_{algorithm}_{target}.png"
        imgs.append(Image.open(path))

    w, h = imgs[0].size
    combined = Image.new("RGB", (w * 2, h * 2), color="white")
    positions = [(0, 0), (w, 0), (0, h), (w, h)]
    for img, pos in zip(imgs, positions):
        combined.paste(img, pos)

    out_path = f"results/compare_sr/fits_{algorithm}.png"
    combined.save(out_path, dpi=(300, 300))
    print(f"Figura combinada salva em: {out_path}")

    for target in ordem:
        path = f"results/compare_sr/fit_{algorithm}_{target}.png"
        os.remove(path) if os.path.exists(out_path) else None


def avaliar_amostras(algo):
    df = pd.read_csv("dados/ariel_limpo_log10.csv.gz", compression="gzip")
    df_amostras = pd.read_csv(f"results/compare_sr/amostras_{algo}.csv")
    features, targets = get_feature_target_names()
    X = df[features].astype(float).values
    y = df[targets].astype(float).values
    _, _, _, y_test = train_test_split(X, y, test_size=0.3, random_state=4321)

    for i, t in enumerate(targets):
        mean = np.mean(y_test[:, i])
        sqa = np.mean((y_test[:, i] - df_amostras[t]) ** 2)
        sqt = np.mean((y_test[:, i] - mean) ** 2)
        r2 = 1 - sqa / sqt
        print(f"\n{t}:")
        print(f"MSE: {sqa:.3f}")
        print(f"R2: {r2:.3f}")


if __name__ == "__main__":
    # histogramas_validation()
    run_comparison(pysr='train', operon=True, rf=True, eggp=False)
