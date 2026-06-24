import os
import pickle
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from pyoperon.sklearn import SymbolicRegressor
from metricas_plots import PlotsMetricas, OperonModelWrapper, T, F

np.seterr(all="ignore")
p = PlotsMetricas()

OPERON_CONFIG = {
    "random_state": 4321,
    "population_size": 1000,
    "allowed_symbols": "add,sub,mul,div,constant,variable,square,exp,tanh",
    "max_length": 25,
    "max_depth": 25,
    "optimizer": "lbfgs",
    "model_selection_criterion": "bayesian_information_criterion",
    "objectives": ["r2", "length"],
    "n_threads": 12,
}

TARGETS = [T.nii.value, T.ha.value, T.oiii.value, T.hb.value]
FEATURES = [F.azmass.value, F.atflux.value, F.mass.value]
TARGET_LABELS = ["NII", r"H$\alpha$", "OIII", r"H$\beta$"]
TARGET_COLORS = ["steelblue", "tomato", "seagreen", "darkorange"]
OUTPUT_DIR = "results/pareto_combos"
FRONTS_DIR = f"{OUTPUT_DIR}/pareto_fronts"


def load_data():
    df = pd.read_csv("dados/ariel_limpo_log10.csv.gz", compression="gzip")
    X = df[FEATURES].astype(float).values
    y = df[TARGETS].astype(float).values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=4321
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    return X_train_sc, X_test_sc, y_train, y_test


def _build_pareto_wrappers(operon_model):
    """
    Converte toda a fronteira de Pareto de um SymbolicRegressor em uma lista de
    dicts {wrapper, complexity, r2, mse}, usando OperonModelWrapper para
    serialização segura (pickle do raw SymbolicRegressor não funciona).
    """
    selected_complexity = int(operon_model.stats_["model_complexity"])
    entries = []
    for entry in operon_model.pareto_front_:
        operon_model.model_ = entry["tree"]
        wrapper = OperonModelWrapper(operon_model)
        entries.append({
            "wrapper": wrapper,
            "complexity": entry["complexity"],
            "r2": -entry["objective_values"][0],
            "mse": entry["mean_squared_error"],
        })
    return entries, selected_complexity


def load_or_train_fronts(X_train, y_train):
    """
    Para cada target treina (ou carrega de .pkl) a fronteira de Pareto completa
    como lista de OperonModelWrapper. Salva sem filtro para reruns flexíveis.
    """
    os.makedirs(FRONTS_DIR, exist_ok=True)
    all_fronts = []
    operon_complexities = []
    for i, target in enumerate(TARGETS):
        path = f"{FRONTS_DIR}/pareto_{target}.pkl"
        if os.path.exists(path):
            print(f"  Carregando fronteira: {target}")
            with open(path, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                front = data["front"]
                sel_c = data["operon_best_complexity"]
            else:
                front = data  # formato antigo sem info de seleção
                sel_c = None
        else:
            print(f"  Treinando Operon para: {target}")
            model = SymbolicRegressor(**OPERON_CONFIG)
            model.fit(X_train, y_train[:, i])
            front, sel_c = _build_pareto_wrappers(model)
            with open(path, "wb") as f:
                pickle.dump({"front": front, "operon_best_complexity": sel_c}, f)
            print(f"  Salvo em {path}  ({len(front)} equações)")
        all_fronts.append(front)
        operon_complexities.append(sel_c)
    return all_fronts, operon_complexities


def _filter_front(front, complexities):
    """Filtra por complexidade (menor MSE por complexidade)."""
    seen = {}
    for e in front:
        c = e["complexity"]
        if c < complexities[0] or c > complexities[1]:
            continue
        if c not in seen or e["mse"] < seen[c]["mse"]:
            seen[c] = e
    return sorted(seen.values(), key=lambda x: x["complexity"])


def plot_combo_bpt(preds, y_test, complexities, r2s, save_path):
    nii_pred, ha_pred, oiii_pred, hb_pred = preds
    nii_ha = nii_pred - ha_pred
    oiii_hb = oiii_pred - hb_pred

    mses_test = [mean_squared_error(y_test[:, i], preds[i]) for i in range(4)]
    combo_str = "-".join(str(c) for c in complexities)
    r2_str = ", ".join(f"{r:.3f}" for r in r2s)

    fig, (ax_bpt, ax_mse) = plt.subplots(
        1, 2,
        figsize=(12.8, 7.2),
        gridspec_kw={"width_ratios": [3, 1]},
    )

    # --- BPT panel ---
    plt.sca(ax_bpt)

    nii_ha_true = y_test[:, 0] - y_test[:, 1]
    oiii_hb_true = y_test[:, 2] - y_test[:, 3]
    ax_bpt.scatter(
        nii_ha_true,
        oiii_hb_true,
        color="blue",
        alpha=0.25,
        s=5,
        edgecolors="none",
        zorder=1,
        label="Validation Set"
    )
    ax_bpt.scatter(
        nii_ha,
        oiii_hb,
        color="red",
        alpha=0.8,
        s=5,
        edgecolors="none",
        zorder=2,
        label="Sampled Set"
    )
    # p.curvas_densidade(nii_ha, oiii_hb)

    p.plot_KeKa()
    ax_bpt.set_xlabel(r"$\log_{10}$(EW[NII] / EWH$\alpha$)", fontsize="large")
    ax_bpt.set_ylabel(r"$\log_{10}$(EW[OIII] / EWH$\beta$)", fontsize="large")
    ax_bpt.set_title(
        f"Diagrama BPT - Fronteira de Pareto\nComplexidades: {combo_str}  |  $R^2$: {r2_str}"
    )
    ax_bpt.set_xlim(-2, 1)
    ax_bpt.set_ylim(-1.5, 1.4)
    ax_bpt.grid(True, alpha=0.2)
    ax_bpt.legend(loc="lower left", fontsize="small")

    # --- MSE row ---
    bars = ax_mse.bar(range(4), mses_test, color=TARGET_COLORS, width=0.6)
    ax_mse.set_xticks(range(4))
    ax_mse.set_xticklabels(TARGET_LABELS)
    ax_mse.set_ylabel("MSE")
    ax_mse.set_title("MSE no conjunto de teste por EW")
    ymax = max(mses_test) if max(mses_test) > 0 else 1.0
    for j, (bar, val) in enumerate(zip(bars, mses_test)):
        ax_mse.text(
            j, val + ymax * 0.04, f"{val:.4f}",
            ha="center", va="bottom",
        )
    ax_mse.set_ylim(0, ymax * 1.35)
    ax_mse.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)
    return mses_test


def run_pareto_combos(complexities=[10, 50]):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\nCarregando dados...")
    X_train, X_test, y_train, y_test = load_data()

    print("\nCarregando/treinando fronteiras de Pareto...")
    raw_fronts, operon_complexities = load_or_train_fronts(X_train, y_train)

    print("\nFiltrando fronteiras de Pareto...")
    fronts = [_filter_front(f, complexities) for f in raw_fronts]
    n_combos = 1
    for f in fronts:
        n_combos *= len(f)

    for target, front in zip(TARGETS, fronts):
        complexities = [e["complexity"] for e in front]
        print(f"  {target}: {len(front)} equações — complexidades {complexities}")

    print(f"\nTotal de combinações: {n_combos}")
    print(f"Salvando BPTs em: {OUTPUT_DIR}/\n")

    all_rows = []
    for combo in tqdm(itertools.product(*fronts), total=n_combos, desc="Combinações"):
        complexities = [e["complexity"] for e in combo]
        r2s = [e["r2"] for e in combo]

        preds = [
            np.nan_to_num(e["wrapper"].predict(X_test), nan=0.0, posinf=0.0, neginf=0.0)
            for e in combo
        ]

        combo_str = "-".join(str(c) for c in complexities)
        save_path = f"{OUTPUT_DIR}/bpt_combo_{combo_str}.png"

        mses_test = plot_combo_bpt(preds, y_test, complexities, r2s, save_path)

        all_rows.append({
            "combo": combo_str,
            **{f"complexity_{t}": c for t, c in zip(TARGETS, complexities)},
            **{f"r2_{t}": r for t, r in zip(TARGETS, r2s)},
            **{f"mse_test_{t}": m for t, m in zip(TARGETS, mses_test)},
            "mse_total": sum(mses_test),
        })

    df_results = pd.DataFrame(all_rows)
    df_results = df_results.sort_values("mse_total")
    df_results.to_csv(f"{OUTPUT_DIR}/mse_summary.csv", index=False)

    print(f"\nConcluído! {n_combos} diagramas gerados.")
    print(f"Resumo de MSEs salvo em {OUTPUT_DIR}/mse_summary.csv")
    print("\nTop 5 combinações por MSE total:")
    df_ranked = df_results.reset_index(drop=True)
    print(df_ranked[["combo", "mse_total"]].head().to_string(index=False))

    if all(c is not None for c in operon_complexities):
        operon_combo = "-".join(str(c) for c in operon_complexities)
        print(f"\nCombinação selecionada pelo Operon (BIC): {operon_combo}")
        match = df_ranked[df_ranked["combo"] == operon_combo]
        if not match.empty:
            rank = match.index[0] + 1
            mse_total = match.iloc[0]["mse_total"]
            print(f"  MSE total: {mse_total:.6f}  (posição #{rank} no ranking)")
        else:
            print("  (complexidade fora do intervalo avaliado)")
    else:
        print("\nCombinação selecionada pelo Operon (BIC): não disponível (reprocessar fronteiras)")



if __name__ == "__main__":
    run_pareto_combos([38, 47])
