import os
import pandas as pd
import matplotlib.pyplot as plt


# Agrega todos os CSVs de métricas em uma subpasta de results e analisa as métricas por popsize.
def aggregate_and_analyze_metrics(results_subfolder, col_x=""):
    # Lista todos os arquivos CSV na subpasta
    csv_files = [
        f
        for f in os.listdir(results_subfolder)
        if f.startswith(f"metrics_{col_x}") and f.endswith(".csv")
    ]
    if not csv_files:
        print("Nenhum arquivo CSV de métricas encontrado na subpasta.")
        return

    # Concatenar todos os CSVs
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(results_subfolder, csv_file))
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    # Verificar se o header está correto
    expected_columns = [
        "Model",
        "popsize",
        "complexity",
        "R2 Train",
        "MSE Train",
        "R2 Test",
        "MSE Test",
    ]
    if list(combined_df.columns) != expected_columns:
        print("Erro: Headers dos CSVs não correspondem ao esperado.")
        return

    # Agrupar por popsize e calcular médias das métricas
    grouped = (
        combined_df.groupby(["popsize"])
        .agg(
            {
                "R2 Train": "mean",
                "R2 Test": "mean",
                "MSE Train": "mean",
                "MSE Test": "mean",
                "complexity": "mean",
            }
        )
        .reset_index()
    )

    # Determinar o melhor popsize baseado em R2 Test (maior) e MSE Test (menor)
    best_r2_test = grouped.loc[grouped["R2 Test"].idxmax()]
    best_mse_test = grouped.loc[grouped["MSE Test"].idxmin()]

    print(
        f"\nMelhor popsize baseado em R2 Test (maior): {best_r2_test['popsize']} com R2 Test = {best_r2_test['R2 Test']:.4f}"
    )
    print(
        f"Melhor popsize baseado em MSE Test (menor): {best_mse_test['popsize']} com MSE Test = {best_mse_test['MSE Test']:.4f}"
    )

    # Se ambos concordarem, é o melhor; senão, escolher baseado em critério combinado
    if best_r2_test["popsize"] == best_mse_test["popsize"]:
        print(f"Popsize recomendado: {best_r2_test['popsize']}")
    else:
        # Critério simples: normalizar e somar (maior R2 e menor MSE)
        grouped["score"] = (grouped["R2 Test"] - grouped["R2 Test"].min()) / (
            grouped["R2 Test"].max() - grouped["R2 Test"].min()
        ) - (grouped["MSE Test"] - grouped["MSE Test"].min()) / (
            grouped["MSE Test"].max() - grouped["MSE Test"].min()
        )
        best_combined = grouped.loc[grouped["score"].idxmax()]
        print(
            f"Popsize recomendado (critério combinado): {best_combined['popsize']} com score = {best_combined['score']:.4f}"
        )


def load_metrics_for_col(results_subfolder, col_x=""):
    csv_files = [
        f
        for f in os.listdir(results_subfolder)
        if f.startswith(f"metrics_{col_x}") and f.endswith(".csv")
    ]
    if not csv_files:
        return pd.DataFrame()
    dfs = [pd.read_csv(os.path.join(results_subfolder, f)) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)
    expected_columns = [
        "Model",
        "popsize",
        "complexity",
        "R2 Train",
        "MSE Train",
        "R2 Test",
        "MSE Test",
    ]
    if list(df.columns) != expected_columns:
        raise ValueError(
            f"Header inconsistente em {results_subfolder}: {list(df.columns)}"
        )
    return df


def compare_folders(folder_a, folder_b, col_x="", save_prefix="comp_bic_r2"):
    df_a = load_metrics_for_col(folder_a, col_x)
    df_b = load_metrics_for_col(folder_b, col_x)
    if df_a.empty or df_b.empty:
        print("Uma das pastas não contém CSVs válidos para a coluna", col_x)
        return

    agg_a = (
        df_a.groupby("popsize")
        .agg({"complexity": "median", "R2 Test": "median", "MSE Test": "median"})
        .reset_index()
        .rename(
            columns={
                "complexity": "complexity_a",
                "R2 Test": "r2_test_a",
                "MSE Test": "mse_test_a",
            }
        )
    )
    agg_b = (
        df_b.groupby("popsize")
        .agg({"complexity": "median", "R2 Test": "median", "MSE Test": "median"})
        .reset_index()
        .rename(
            columns={
                "complexity": "complexity_b",
                "R2 Test": "r2_test_b",
                "MSE Test": "mse_test_b",
            }
        )
    )
    merged = pd.merge(agg_a, agg_b, on="popsize", how="inner")

    merged["delta_r2_test"] = merged["r2_test_b"] - merged["r2_test_a"]
    merged["delta_mse_test"] = merged["mse_test_b"] - merged["mse_test_a"]
    merged["delta_complexity"] = merged["complexity_b"] - merged["complexity_a"]

    print(f"\nComparação {folder_a} x {folder_b} para {col_x or 'todas as cols'}")

    fig, axes = plt.subplots(3, 1, figsize=(8, 16))
    ticks = [merged["popsize"].iloc[t] for t in range(0, len(merged["popsize"]), 3)]

    axes[0].plot(merged["popsize"], merged["complexity_a"], marker="o", label="Min-BIC")
    axes[0].plot(merged["popsize"], merged["complexity_b"], marker="x", label="Max-R2")
    axes[0].set_title("Complexity")
    axes[0].set_xscale("log")
    axes[0].set_xticks(ticks)
    axes[0].set_xticklabels(ticks)
    axes[0].legend()

    axes[1].plot(merged["popsize"], merged["mse_test_a"], marker="o", label="Min-BIC")
    axes[1].plot(merged["popsize"], merged["mse_test_b"], marker="x", label="Max-R2")
    axes[1].set_title("MSE Validation Set")
    axes[1].set_xscale("log")
    axes[1].set_xticks(ticks)
    axes[1].set_xticklabels(ticks)
    axes[1].legend()

    axes[2].plot(merged["popsize"], merged["r2_test_a"], marker="o", label="Min-BIC")
    axes[2].plot(merged["popsize"], merged["r2_test_b"], marker="x", label="Max-R2")
    axes[2].set_title("R2 Validation Set")
    axes[2].set_xscale("log")
    axes[2].set_xticks(ticks)
    axes[2].set_xticklabels(ticks)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(folder_a, f'{save_prefix}_{col_x or "all"}.png'))
    plt.close()


def compare_popsizes_by_model(
    folder_a, folder_b, col_x="", save_prefix="comp_popsizes"
):
    """Para cada métrica, cria um gráfico com uma linha por popsize (eixo x = modelos),
    sobreposto num mesmo subplot — permite avaliar qual popsize é mais adequado."""
    df_a = load_metrics_for_col(folder_a, col_x)
    df_b = load_metrics_for_col(folder_b, col_x)
    if df_a.empty or df_b.empty:
        print("Uma das pastas não contém CSVs válidos para a coluna", col_x)
        return

    popsizes = sorted(set(df_a["popsize"].unique()) | set(df_b["popsize"].unique()))
    markers = ["o", "s", "^", "D", "v", "P", "*", "X", "h"]
    linestyles = ["-", "--", "-.", ":"]

    metrics = [
        ("complexity", "Complexity"),
        ("MSE Test", "MSE Validation Set"),
        ("R2 Test", "R2 Validation Set"),
    ]

    fig, axes = plt.subplots(len(metrics), 2, figsize=(16, 4 * len(metrics)))

    for col_idx, (folder, df, label) in enumerate(
        [
            (folder_a, df_a, "Min-BIC"),
            (folder_b, df_b, "Max-R2"),
        ]
    ):
        # Determinar ordem dos modelos a partir da menor popsize disponível
        first_pop = sorted(df["popsize"].unique())[0]
        model_order = (
            df[df["popsize"] == first_pop].sort_values("Model")["Model"].tolist()
        )
        if not model_order:
            model_order = sorted(df["Model"].unique())

        x = range(len(model_order))

        for row_idx, (metric_col, metric_label) in enumerate(metrics):
            ax = axes[row_idx][col_idx]

            for pop_idx, popsize in enumerate(popsizes):
                df_pop = df[df["popsize"] == popsize].copy()
                if df_pop.empty:
                    continue
                if df_pop.duplicated(subset=["Model"]).any():
                    df_pop = df_pop.groupby("Model", as_index=False).mean(
                        numeric_only=True
                    )

                df_pop = df_pop.set_index("Model").reindex(model_order)
                marker = markers[pop_idx % len(markers)]
                ls = linestyles[pop_idx % len(linestyles)]
                ax.plot(
                    x,
                    df_pop[metric_col].values,
                    marker=marker,
                    linestyle=ls,
                    label=f"pop={popsize}",
                )

            ax.set_title(f"{metric_label} — {label}")
            ax.set_xticks(list(x))
            if row_idx == len(metrics) - 1:
                ax.set_xticklabels(model_order, rotation=45, ha="right", fontsize=8)
            else:
                ax.set_xticklabels(["" for _ in x])
            if row_idx == 1 and col_idx == 1:
                ax.legend(fontsize=7)

    plt.tight_layout()
    col_suffix = f"_{col_x}" if col_x else ""
    out_path = os.path.join(folder_a, f"{save_prefix}{col_suffix}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Salvo: {out_path}")


def compare_model_by_popsizes(
    folder_a, folder_b, col_x="", model=None, save_prefix="comp_model_popsizes"
):
    """Para um modelo específico, plota cada métrica em função das popsizes (eixo x = popsize),
    com colunas Min-BIC e Max-R2 — permite ver o impacto do popsize nesse modelo."""
    df_a = load_metrics_for_col(folder_a, col_x)
    df_b = load_metrics_for_col(folder_b, col_x)
    if df_a.empty or df_b.empty:
        print("Uma das pastas não contém CSVs válidos para a coluna", col_x)
        return

    if model is None:
        print(
            "Informe o parâmetro 'model'. Modelos disponíveis:",
            sorted(df_a["Model"].unique()),
        )
        return

    df_a = df_a[df_a["Model"] == model]
    df_b = df_b[df_b["Model"] == model]
    if df_a.empty or df_b.empty:
        print(f"Modelo '{model}' não encontrado em uma das pastas.")
        return

    metrics = [
        ("complexity", "Complexity"),
        ("MSE Test", "MSE Validation Set"),
        ("R2 Test", "R2 Validation Set"),
    ]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 4 * len(metrics)))

    # popsizes comuns para alinhar eixo x
    popsizes = sorted(set(df_a["popsize"].unique()) | set(df_b["popsize"].unique()))
    x_pos = range(len(popsizes))

    for row_idx, (metric_col, metric_label) in enumerate(metrics):
        ax = axes[row_idx]

        for df, label, color, marker in [
            (df_a, "Min-BIC", "tab:blue", "o"),
            (df_b, "Max-R2", "tab:orange", "x"),
        ]:
            df_grouped = (
                df.groupby("popsize", as_index=False)
                .mean(numeric_only=True)
                .sort_values("popsize")
                .set_index("popsize")
                .reindex(popsizes)
            )
            ax.plot(
                list(x_pos),
                df_grouped[metric_col].values,
                marker=marker,
                linestyle="-",
                color=color,
                label=label,
            )

        ax.set_title(metric_label)
        ax.set_xticks(list(x_pos))
        if row_idx == len(metrics) - 1:
            ax.set_xticklabels(popsizes, rotation=45, ha="right", fontsize=8)
            ax.set_xlabel("popsize")
        else:
            ax.set_xticklabels(["" for _ in x_pos])
        ax.legend()

    fig.suptitle(f'Modelo: {model}  |  col_x: {col_x or "all"}', fontsize=11, y=1.01)
    plt.tight_layout()
    col_suffix = f"_{col_x}" if col_x else ""
    model_suffix = f"_{model.replace(' ', '_')}"
    out_path = os.path.join(folder_a, f"{save_prefix}{col_suffix}{model_suffix}.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Salvo: {out_path}")


if __name__ == "__main__":
    # Analisando cada pasta e feature
    aggregate_and_analyze_metrics("results/by_bic", "atflux")
    aggregate_and_analyze_metrics("results/by_bic", "aZmass")
    aggregate_and_analyze_metrics("results/by_bic", "mass")

    # Comparação de duas pastas
    compare_folders("results/by_bic", "results/by_r2", "atflux")
    compare_folders("results/by_bic", "results/by_r2", "aZmass")
    compare_folders("results/by_bic", "results/by_r2", "mass")

    # Comparando todos os popsizes sobrepostos por modelo
    compare_popsizes_by_model("results/by_bic", "results/by_r2", "atflux")
    compare_popsizes_by_model("results/by_bic", "results/by_r2", "aZmass")
    compare_popsizes_by_model("results/by_bic", "results/by_r2", "mass")

    # Evolução por popsize para um modelo específico
    compare_model_by_popsizes(
        "results/by_bic", "results/by_r2", "atflux", model="oiii_5007_ew_mean"
    )
    compare_model_by_popsizes(
        "results/by_bic", "results/by_r2", "aZmass", model="oiii_5007_ew_mean"
    )
    compare_model_by_popsizes(
        "results/by_bic", "results/by_r2", "mass", model="oiii_5007_ew_mean"
    )
