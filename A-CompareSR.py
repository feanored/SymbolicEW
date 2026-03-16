import os
import time
import sympy as sp
import numpy as np
import pandas as pd
import smplotlib # type: ignore
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from metricas_plots import PlotsMetricas, T, F
np.seterr(all='ignore')
p = PlotsMetricas()

rf_config = {
    "algorithm": "RandomForest",
    "kwargs": {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 2,
        "random_state": 4321,
        "n_jobs": 12
    }
}

operon_config = {
    "algorithm": "Operon",
    "kwargs": {
        "population_size": 1000,
        "optimizer_iterations": 16, # para se igualar ao padrão do PySR
        "allowed_symbols": "add,sub,mul,aq,constant,variable,square,exp,tanh",
        "max_depth": 50,
        "model_selection_criterion": "bayesian_information_criterion",
        "n_threads": 12,
        "objectives": ["mse", "length"]
    }
}

pysr_multi = {
    "algorithm": "PySR",
    "kwargs": {
        "populations": 10,
        "population_size": 100,
        "binary_operators": ["+", "-", "*", "/"],
        "unary_operators": ["square", "exp", "tanh"],
        "maxdepth": 50,
        "model_selection": "accuracy",
        "parallelism": "multiprocessing",
        "procs": 12,
        "verbosity": 1
    }
}

pysr_single = {
    "algorithm": "PySR",
    "kwargs": {
        "populations": 1, # for single-island
        "population_size": 1000,
        "binary_operators": ["+", "-", "*", "/"],
        "unary_operators": [
            "square", "exp", "tanh",
            "sqrtabs(x) = sqrt(abs(x))",
            "logabs(x) = log(abs(x))",
        ],
        "maxdepth": 50,
        "model_selection": "accuracy",
        "parallelism": "multithreading",
        "procs": 12,
        "verbosity": 1
    }
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
        start = time.perf_counter() # segundos
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
        start = time.perf_counter() # segundos
        model = PySRRegressor(**pysr_kwargs
            # , extra_sympy_mappings={
            # "sqrtabs": lambda x: sp.sqrt(sp.Abs(x)),
            # "logabs": lambda x: sp.log(sp.Abs(x))}
        )
        model.fit(X_train, y_train[:, i])
        elapsed = time.perf_counter() - start
        elapseds.append(elapsed)
        models.append(model)
    return models, elapseds

def read_pysr_models(config, targets, prefix=''):
    from pysr import PySRRegressor
    models = []
    elapseds = []
    for target in targets:
        start = time.perf_counter() # segundos
        model = PySRRegressor.from_file(run_directory=f'results/pysr/{prefix}{target}', model_selection=config["kwargs"]["model_selection"])
        elapsed = time.perf_counter() - start
        elapseds.append(elapsed)
        models.append(model)
    return models, elapseds

def evaluate_models(models, predict_fn, X_train, X_test, y_train, y_test, algorithm, targets, elapsed):
    rows = []
    if models is None:
        return rows
    
    for i, model in enumerate(models):
        # Calcula métricas
        y_pred_train = predict_fn(model, X_train)
        y_pred_test = predict_fn(model, X_test)
        y_pred_train = np.nan_to_num(y_pred_train, nan=0.0, posinf=0.0, neginf=0.0)
        y_pred_test = np.nan_to_num(y_pred_test, nan=0.0, posinf=0.0, neginf=0.0)
        mse_train = mean_squared_error(y_train[:, i], y_pred_train)
        mse_test = mean_squared_error(y_test[:, i], y_pred_test)
        r2_train = r2_score(y_train[:, i], y_pred_train)
        r2_test = r2_score(y_test[:, i], y_pred_test)
        
        # Obtém equações e complexidade
        if algorithm == 'PySR':
            complexy = model.get_best()['complexity']
            expr = model.sympy()
            equations = expr.xreplace({n: round(n, 6) for n in expr.atoms(sp.Float)})
        elif algorithm == 'Operon':
            complexy = model.stats_['model_complexity']
            equations = model.get_model_string(model.model_, precision=6).replace('^', '**')
        elif algorithm == 'RandomForest':
            avg_depth = int(np.mean([e.get_depth() for e in model.estimators_]))
            complexy = model.n_estimators * avg_depth
            equations = "N/A"

        # Exibe gráfico real X predito
        plt.subplots(figsize=(7, 7))
        plt.scatter(y_test[:, i], y_pred_test, alpha=0.3, s=2, edgecolors=None, color='gray')
        plt.plot([y_test[:, i].min(), y_test[:, i].max()], [y_test[:, i].min(), y_test[:, i].max()], 'r--')
        p.curvas_densidade(y_test[:, i], y_pred_test)
        plt.xlabel("Real")
        plt.ylabel("Predito")
        plt.title(p.unidades[targets[i]])
        plt.xlim([y_test[:, i].min(), y_test[:, i].max()])
        plt.ylim([y_test[:, i].min(), y_test[:, i].max()])
        plt.savefig(f'results/compare_sr/fit_{algorithm}_{targets[i]}.png')
        plt.close()

        # Salva amostras geradas para posterior diagrama BPT
        df_amostra = pd.DataFrame(y_pred_test, columns=[targets[i]])
        df_amostra.to_csv(f'results/compare_sr/amostras_{algorithm}_{targets[i]}.csv', index=False)
        
        # Retorna métricas
        rows.append({
            "algorithm": algorithm,
            "time": f'{elapsed[i]:.0f}',
            "target": targets[i],
            "r2_train": r2_train,
            "mse_train": mse_train,
            "r2_test": r2_test,
            "mse_test": mse_test,
            "complexity": complexy,
            "equation": equations
        })
    return rows

def operon_predict(model, X):
    return model.predict(X)

def pysr_predict(model, X):
    return model.predict(X)

def rf_predict(model, X):
    return model.predict(X)

def run_comparison(pysr=None, operon=False, rf=False):
    os.makedirs("results/compare_sr", exist_ok=True)
    output_path = os.path.join("results/compare_sr", "compare_sr_results.csv")
    
    print("\n Carregando dados...")
    df = pd.read_csv("dados/ariel_limpo_log10.csv.gz", compression="gzip")
    features, targets = get_feature_target_names()
    for col in features + targets:
        if col not in df.columns:
            raise ValueError(f"Coluna esperada nao encontrada no arquivo: {col}")

    X = df[features].astype(float).values
    y = df[targets].astype(float).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4321)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    results_rows = []

    if rf:
        print("\n Treinando RandomForest...")
        rf_models, elapsed = train_rf_models(X_train_scaled, y_train, rf_config)
        results_rows.extend(evaluate_models(rf_models, rf_predict, X_train_scaled, X_test_scaled, y_train, y_test, "RandomForest", targets, elapsed))
        gerar_diagramas("RandomForest")

    if operon:
        print("\n Treinando Operon...")
        operon_models, elapsed = train_operon_models(X_train_scaled, y_train, operon_config)
        results_rows.extend(evaluate_models(operon_models, operon_predict, X_train_scaled, X_test_scaled, y_train, y_test, "Operon", targets, elapsed))
        gerar_diagramas("Operon")

    if pysr is not None:
        print("\n Treinando PySR...")
        if pysr == 'single': # Escolher aqui se é single ou multi, threading
            pysr_models, elapsed = train_pysr_models(X_train_scaled, y_train, pysr_single)
        elif pysr == 'multi':
            pysr_models, elapsed = train_pysr_models(X_train_scaled, y_train, pysr_multi)
        elif pysr == 'read':
            pysr_models, elapsed = read_pysr_models(pysr_multi, targets, '20260315_')
        if pysr_models is not None:
            results_rows.extend(evaluate_models(pysr_models, pysr_predict, X_train_scaled, X_test_scaled, y_train, y_test, "PySR", targets, elapsed))
            gerar_diagramas("PySR")

    if rf or operon or (pysr is not None and pysr != 'read'):
        df_results = pd.DataFrame(results_rows)
        df_results.to_csv(output_path, index=False)
        print("\nResultados salvos em:", output_path)
    else:
        print("Nenhum modelo foi ativado para o teste!")

def gerar_diagramas(algo):
    series = []
    _, targets = get_feature_target_names()
    for target in targets:
        s = pd.read_csv(f"results/compare_sr/amostras_{algo}_{target}.csv").iloc[:, 0]
        s.name = target
        series.append(s)
    df_amostras = pd.concat(series, axis=1)
    df_amostras['nii_halpha_ew'] = df_amostras['nii_6584_ew'] - df_amostras['halpha_ew']
    df_amostras['oiii_hbeta_ew'] = df_amostras['oiii_5007_ew'] - df_amostras['hbeta_ew']
    p.show_bpt(df_amostras, title=f"Amostras do {algo}", densities=True, show=False)
    plt.savefig(f'results/compare_sr/bpt_{algo}.png')
    plt.close()
    p.show_whan(df_amostras, title=f"Amostras do {algo}", densities=True, show=False)
    plt.savefig(f'results/compare_sr/whan_{algo}.png')
    plt.close()


if __name__ == "__main__":
    run_comparison(pysr='multi', operon=True, rf=True)