# SymbolicEW 🔭

> **Regressão Simbólica aplicada às Larguras Equivalentes de Linhas de Emissão de Galáxias**

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](./Dockerfile)

---

## 📖 Visão Geral

**SymbolicEW** é um projeto de pesquisa em astrofísica computacional que aplica **Regressão Simbólica (SR)** para modelar as **larguras equivalentes (EW)** de linhas de emissão de galáxias. O objetivo é descobrir expressões matemáticas interpretáveis e compactas que descrevam as relações entre propriedades espectroscópicas, em vez de ajustar formas paramétricas predefinidas.

São modeladas 4 linhas de emissão — **Hα**, **Hβ**, **[NII]6584** e **[OIII]5007** — a partir de 3 features galáticas: **massa estelar**, **fluxo total** e **massa azimutal**, todos os dados em escala logarítmica (log₁₀).

---

## 🧠 Algoritmos Comparados

O projeto realiza um benchmark rigoroso entre quatro abordagens:

| Algoritmo | Biblioteca | Papel no estudo |
|---|---|---|
| **Operon** | `pyoperon==0.6.1` | SR com seleção por BIC e fronteira de Pareto (R² × complexidade) |
| **PySR** | `pysr==1.5.9` | SR via programação genética com backend Julia |
| **RandomForest** | `scikit-learn` | Baseline não-simbólico para referência de desempenho |

---

## 🗂️ Estrutura do Projeto
```
SymbolicEW/
│
├── dados/
│   └── ariel_limpo_log10.csv.gz        # Dataset espectroscópico (log₁₀, comprimido)
│
├── results/                             # Saídas: métricas, gráficos, modelos
│
├── 1-DataPrep.ipynb                     # Limpeza e pré-processamento dos dados
├── 2-Analysis.ipynb                     # Análise exploratória, diagramas BPT e WHAN,
│                                        #   matrizes de correlação
├── 3-Modelagem univariada independente.ipynb  # SR por alvo, sem dependência entre EWs
├── 4-Modelagem univariada dependente.ipynb    # SR por alvo, com dependência entre EWs
│
├── A-CompareSR.py          # Benchmark completo: RF × Operon × PySR × egGP
├── B-GridSearchOperon.py   # Grid search de hiperparâmetros do Operon
├── C-MetricasOperon.py     # Extração e avaliação dos modelos Pareto-ótimos do Operon
├── D-BIC_Operon.py         # Executa uma rodada de amostras LH e calcula a significância da métrica BIC
│
├── metricas_plots.py       # Funções compartilhadas: métricas, histogramas, BPT, WHAN
├── fileserver.py           # Servidor de arquivos para workflows remotos
├── Dockerfile              # Ambiente Conda reprodutível com JupyterLab
└── requirements.txt        # Dependências Python
```

---

## ⚙️ Instalação

### Via Docker (recomendado)

O ambiente completo é encapsulado em uma imagem Miniconda com Python 3.13 e todas as dependências instaladas.
```bash
docker build -t symbolicew .
docker run -it --rm -p 8888:8888 -v $(pwd):/workspace symbolicew
```

Acesse o JupyterLab em `http://localhost:8888` com o token padrão `capeta` (configurável via `--build-arg TOKEN=seu_token`).

### Via pip (ambiente local)
```bash
git clone https://github.com/feanored/SymbolicEW.git
cd SymbolicEW
pip install -r requirements.txt
```

> ⚠️ O **PySR** requer Julia. Na primeira execução, instalará automaticamente Julia e dependências via `juliacall`.
> O **pyoperon** pode requerer instalação manual dependendo do sistema operacional — consulte [heal-research/operon](https://github.com/heal-research/operon).

---

## 🚀 Como Usar

Execute os notebooks em sequência para o pipeline completo de análise:
```
1-DataPrep.ipynb  →  2-Analysis.ipynb  →  3-Modelagem univariada.ipynb →  4-Modelagem multivariada.ipynb
```

Para executar o benchmark entre algoritmos via linha de comando:
```bash
# Treina todos os modelos ativos e salva resultados com timestamp
python A-CompareSR.py
```

Os modelos ativos são controlados dentro do script em `run_comparison(pysr="train", operon=True, rf=True)`.

---

## 📐 Pipeline

**Features (input):** `mass`, `atflux`, `azmass` — em log₁₀

**Targets (output):** `halpha_ew`, `hbeta_ew`, `nii_6584_ew`, `oiii_5007_ew` — em log₁₀

**Divisão dos dados:** 70% treino / 30% teste, `random_state=4321`, `StandardScaler` nas features

**Avaliação:** MSE, R² (treino e teste) e divergência KL entre a distribuição amostrada pelo modelo e o conjunto de validação real.

**Outputs gerados por algoritmo:**
- Gráfico real × predito por linha de emissão
- Histogramas comparativos com KL divergence
- Diagramas de diagnóstico **BPT** e **WHAN** das amostras geradas
- CSV com amostras e métricas (com timestamp)

---

## 📦 Dependências Principais
```
pyoperon==0.6.1
pysr==1.5.9
numpy
pandas
matplotlib
scikit-learn
sympy
tqdm
jupyterlab
```

---

## 📄 Licença

Este projeto está licenciado sob a **GNU General Public License v3.0**. Veja o arquivo [LICENSE](LICENSE) para detalhes.

---

## 👤 Autor

**Eduardo Galvani Massino** · [github.com/feanored](https://github.com/feanored)