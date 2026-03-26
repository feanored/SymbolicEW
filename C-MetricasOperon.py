import os
import numpy as np
import pandas as pd
import smplotlib  # type: ignore
import matplotlib.pyplot as plt
from metricas_plots import T


def load_metrics_for_col(results_subfolder, col_x):
    csv_files = [
        f
        for f in os.listdir(results_subfolder)
        if f.startswith(f"metrics_{col_x}_") and f.endswith(".csv")
    ]
    if not csv_files:
        df = pd.read_csv(os.path.join(results_subfolder, f"metrics_{col_x}.csv"))
        if df.empty:
            raise ("Não há métricas disponíveis!")
        return df
    dfs = [pd.read_csv(os.path.join(results_subfolder, f)) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(os.path.join(results_subfolder, f"metrics_{col_x}.csv"), index=False)
    for csv in csv_files:
        os.remove(os.path.join(results_subfolder, csv))
    return df


def compare_folders(folder_a, folder_b, col_x="", save_prefix="comp_bic_r2"):
    df_a = load_metrics_for_col(folder_a, col_x)
    df_b = load_metrics_for_col(folder_b, col_x)
    if df_a.empty or df_b.empty:
        print("Uma das pastas não contém CSVs válidos para a coluna", col_x)
        return

    agg_a = (
        df_a.groupby("popsize")
        .agg({"complexity": "median", "R2 Test": "median", "bic": "median"})
        .reset_index()
        .rename(
            columns={
                "complexity": "complexity_a",
                "R2 Test": "r2_test_a",
                "bic": "bic_test_a",
            }
        )
    )
    agg_b = (
        df_b.groupby("popsize")
        .agg({"complexity": "median", "R2 Test": "median", "bic": "median"})
        .reset_index()
        .rename(
            columns={
                "complexity": "complexity_b",
                "R2 Test": "r2_test_b",
                "bic": "bic_test_b",
            }
        )
    )
    merged = pd.merge(agg_a, agg_b, on="popsize", how="inner")

    print(f"Comparação {folder_a} x {folder_b} para {col_x or 'todas as cols'}")

    _, axes = plt.subplots(3, 1, figsize=(8, 16))

    axes[0].plot(merged["popsize"], merged["complexity_a"], marker="o", label="Min-BIC")
    axes[0].plot(merged["popsize"], merged["complexity_b"], marker="x", label="Max-R2")
    axes[0].set_title("Complexity")
    axes[0].set_xticks(merged["popsize"])
    axes[0].legend()

    axes[1].plot(merged["popsize"], merged["r2_test_a"], marker="o", label="Min-BIC")
    axes[1].plot(merged["popsize"], merged["r2_test_b"], marker="x", label="Max-R2")
    axes[1].set_title("R2 Validation Set")
    axes[1].set_xticks(merged["popsize"])
    axes[1].legend()

    axes[2].plot(merged["popsize"], merged["bic_test_a"], marker="o", label="Min-BIC")
    axes[2].plot(merged["popsize"], merged["bic_test_b"], marker="x", label="Max-R2")
    axes[2].set_title("BIC Validation Set")
    axes[2].set_xticks(merged["popsize"])
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(folder_a, f'{save_prefix}_{col_x or "all"}.png'))
    plt.close()


def compare_popsizes_by_model(folder, col_x="", save_prefix="comp_popsizes"):
    df = load_metrics_for_col(folder, col_x)
    if df.empty:
        print("A pasta não contém CSVs válidos para a coluna ", col_x)
        return

    popsizes = sorted(set(df["popsize"].unique()))
    # markers = ["o", "s", "^", "D", "v", "P", "*", "X", "h"]
    linestyles = ["-", "--", "-.", ":"]

    metrics = [
        ("complexity", "Complexity"),
        ("R2 Test", "R2 Validation Set"),
        ("bic", "BIC Validation Set"),
    ]

    _, axes = plt.subplots(len(metrics), 1, figsize=(8, 18))

    # Determinar ordem dos modelos a partir da menor popsize disponível
    first_pop = sorted(df["popsize"].unique())[0]
    model_order = df[df["popsize"] == first_pop].sort_values("Model")["Model"].tolist()
    if not model_order:
        model_order = sorted(df["Model"].unique())

    x = range(len(model_order))
    idx = np.linspace(0, len(popsizes) - 1, 10, dtype=int)

    for row_idx, (metric_col, metric_label) in enumerate(metrics):
        ax = axes[row_idx]

        for pop_idx, popsize in enumerate(np.array(popsizes)[idx]):
            df_pop = df[df["popsize"] == popsize].copy()
            if df_pop.empty:
                continue
            if df_pop.duplicated(subset=["Model"]).any():
                df_pop = df_pop.groupby("Model", as_index=False).mean(numeric_only=True)

            df_pop = df_pop.set_index("Model").reindex(model_order)
            # marker = markers[pop_idx % len(markers)]
            ls = linestyles[pop_idx % len(linestyles)]
            ax.plot(
                x,
                df_pop[metric_col].values,
                marker="",
                linestyle=ls,
                label=f"pop={popsize}",
            )

        ax.set_title(f"{metric_label}")
        ax.set_xticks(x)
        ax.set_xticklabels(model_order, rotation=45, ha="right", fontsize=8)
        if row_idx == 2:
            ax.legend(fontsize="small", loc="upper left")

    plt.tight_layout()
    col_suffix = f"_{col_x}" if col_x else ""
    out_path = os.path.join(folder, f"{save_prefix}{col_suffix}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Salvo: {out_path}")


def compare_model_by_popsizes(folder, col_x="", model=None):
    save_prefix = "comp_model_popsizes"
    df = load_metrics_for_col(folder, col_x)
    if df.empty:
        print("A pasta não contém CSVs válidos para a coluna ", col_x)
        return

    if model is None:
        print(
            "Informe o parâmetro 'model'. Modelos disponíveis:",
            sorted(df["Model"].unique()),
        )
        return

    df = df[df["Model"] == model]
    if df.empty:
        print(f"Modelo '{model}' não encontrado na pasta.")
        return

    metrics = [
        ("complexity", "Complexity"),
        ("R2 Test", "R2 Validation Set"),
        ("bic", "BIC Validation Set"),
    ]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 4 * len(metrics)))

    # popsizes comuns para alinhar eixo x
    popsizes = sorted(set(df["popsize"].unique()))
    idx = np.linspace(0, len(popsizes) - 1, 10, dtype=int)

    for row_idx, (metric_col, metric_label) in enumerate(metrics):
        ax = axes[row_idx]

        for df, label, marker in [(df, metric_label, "")]:
            df_grouped = (
                df.groupby("popsize", as_index=False)
                .mean(numeric_only=True)
                .sort_values("popsize")
                .set_index("popsize")
                .reindex(popsizes)
            )
            ax.plot(
                popsizes,
                df_grouped[metric_col].values,
                marker=marker,
                linestyle="-",
                label=label,
            )
        ax.set_xticks(np.array(popsizes)[idx])
        ax.set_xticklabels(np.array(popsizes)[idx], fontsize="medium")
        ax.set_xlabel("popsize")
        ax.legend()

    fig.suptitle(f'Modelo: {model} ~ {col_x or "all"}', fontsize="large", y=1.01)
    plt.tight_layout()
    col_suffix = f"_{col_x}" if col_x else ""
    model_suffix = f"_{model.replace(' ', '_')}"
    out_path = os.path.join(folder, f"{save_prefix}{col_suffix}{model_suffix}.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Salvo: {out_path}")


def save_best_popsizes_by_bic(folder):
    # Extrai col_x únicos a partir dos nomes dos arquivos metrics_<col_x>_<popsize>.csv
    col_x_values = sorted(
        {
            "_".join(f[len("metrics") : -len(".csv")].split("_")[1:])
            for f in os.listdir(folder)
            if f.startswith("metrics_") and f.endswith(".csv")
        }
    )

    rows = []
    for col_x in col_x_values:
        df = load_metrics_for_col(folder, col_x)
        if df.empty:
            continue
        for model in sorted(df["Model"].unique()):
            df_model = df[df["Model"] == model]
            best_row = df_model.loc[df_model["bic"].idxmin()]
            rows.append(
                {
                    "col_x": col_x,
                    "model": model,
                    "popsize": int(best_row["popsize"]),
                }
            )

    if not rows:
        print("Nenhum dado encontrado em", folder)
        return

    out = pd.DataFrame(rows)
    csv_path = os.path.join(folder, "best_popsizes_by_bic.csv")
    out.to_csv(csv_path, index=False)
    print(f"Salvo: {csv_path}")


if __name__ == "__main__":
    # Comparação de duas pastas
    compare_folders("results/by_bic", "results/by_r2", "atflux")
    compare_folders("results/by_bic", "results/by_r2", "aZmass")
    compare_folders("results/by_bic", "results/by_r2", "mass")

    # Evolução por popsize para um modelo específico
    compare_model_by_popsizes("results/by_bic", "mass", T.oiii.value + "_mean")

    # Comparando todos os popsizes sobrepostos por modelo
    compare_popsizes_by_model("results/by_bic", "atflux")
    compare_popsizes_by_model("results/by_bic", "aZmass")
    compare_popsizes_by_model("results/by_bic", "mass")
    save_best_popsizes_by_bic("results/by_bic")
