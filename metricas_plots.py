import numpy as np
import sympy as sp
import pandas as pd
import smplotlib # type: ignore
import seaborn as sns
from scipy import stats
from IPython.display import display, Latex
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from scipy.optimize import minimize
from scipy.stats import linregress, truncnorm
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import median_abs_deviation
from scipy.stats import gaussian_kde
import statsmodels.api as sm
from scipy.special import gammaln
from statsmodels.stats.diagnostic import het_breuschpagan
from enum import Enum
from types import SimpleNamespace as Struct
from pyoperon.sklearn import SymbolicRegressor
from mapie.metrics.regression import regression_coverage_score
from mapie.regression import SplitConformalRegressor
from mapie.regression import ConformalizedQuantileRegressor
from mapie.utils import train_conformalize_test_split
import warnings
warnings.filterwarnings("ignore")

#--------------------------------------------------------------------------------------------------
# Definindo constantes úteis

RANDOM_SEED = 4321
CONF_LEVEL = 0.975

class T(Enum):
    nii = "nii_6584_ew"
    ha = "halpha_ew"
    oiii = "oiii_5007_ew"
    hb = "hbeta_ew"
    nii_ha = "nii_halpha_ew"
    oiii_hb = "oiii_hbeta_ew"
    nii_ha_f = "nii_halpha_flux"
    oiii_hb_f = "oiii_hbeta_flux"

class F(Enum):
    atflux = "atflux"
    atmass = "atmass"
    azflux = "aZflux"
    azmass = "aZmass"
    mass = "mass"
    av = "Av"

#--------------------------------------------------------------------------------------------------

class PlotsMetricas(object):
    def __init__(self):
        self.features = [f.value for f in F]
        self.targets = [t.value for t in T]
        self.phi = 1.618
        np.random.seed(RANDOM_SEED)
        self.unidades = {}
        self.setUnidades()
        self.setOptions()

    def __str__(self):
        return self.targets

    def setUnidades(self):
        self.unidades[T.nii.value]    = r'$ \log EW([NII]6584) ~(\AA) $'
        self.unidades[T.oiii.value]   = r'$ \log EW([OIII]5007) ~(\AA) $'
        self.unidades[T.ha.value]     = r'$ \log EW(H\alpha) ~(\AA) $'
        self.unidades[T.hb.value]     = r'$ \log EW(H\beta) ~(\AA) $'
        self.unidades[F.atflux.value] = r'$ \log t_{flux} ~(yr) $'
        self.unidades[F.atmass.value] = r'$ \log t_{mass} ~(yr) $'
        self.unidades[F.azflux.value] = r'$ \log (Z_{ergs}/F_\odot) $'
        self.unidades[F.azmass.value] = r'$ \log (Z_{mass}/Z_\odot) $'
        self.unidades[F.mass.value]   = r'$ \log mass ~(M_\odot) $'
    
    def setOptions(self):
        #if platform.system() == "Windows":
        #    pio.renderers.default = 'browser'
        #else:
        pio.renderers.default = 'png'
        pio.templates["plotly"].layout.width = 900
        pio.templates["plotly"].layout.height = 600
        pio.templates["plotly"].layout.xaxis.showgrid = False
        pio.templates["plotly"].layout.yaxis.showgrid = False
        pio.templates["plotly"].layout.plot_bgcolor = '#f5f5f5'
        pio.templates["plotly"].layout.xaxis.zeroline = True
        pio.templates["plotly"].layout.yaxis.zeroline = True
            
    # split
    def split_dados(self, dados, col_y, features):
        # Separar variáveis de entrada e saída
        X = dados[features].values
        y = dados[col_y].values
        return train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    
    # split for conformal regression
    def split_conformal(self, dados, col_y, features):
        X = dados[features].values
        y = dados[col_y].values
        (X_train, X_conform, X_test, y_train, y_conform, y_test) = train_conformalize_test_split(
            X, y, random_state=RANDOM_SEED,
            train_size=0.6, conformalize_size=0.2, test_size=0.2
        )
        return X_train, X_conform, X_test, y_train, y_conform, y_test
    
    # gráficos de correlações multiplas
    def plot_corr(self, correlacoes, ax, lbls, title):
        mask = np.triu(np.ones_like(correlacoes, dtype=bool), k=1) # triu debaixo, tril acima
        sns.heatmap(correlacoes, mask=mask, ax=ax,
                    annot = True, fmt = '.3f',
                    square=True, linewidths=0.5,
                    vmin=0, vmax=1, cmap='coolwarm')
        ax.set_xticklabels(lbls, rotation=-15)
        ax.set_yticklabels(lbls, rotation=45)
        ax.set_title(title)

    # Definir treinamento do modelo simbólico
    def operon_regression(self, X_train, y_train, selection='minimum_description_length'):
        # Treinar o modelo
        model = SymbolicRegressor(
            random_state=RANDOM_SEED,
            allowed_symbols="add,sub,mul,div,constant,variable,square,pow,abs,sqrt,exp,log,tanh",
            model_selection_criterion=selection,
            population_size=5000,
            generations=2000,
            optimizer_iterations=2000,
            max_length=50,
            n_threads=1, # 12
            objectives= ['r2'] #['mse', 'length']
        )
        model.fit(X_train, y_train)
        self._operon_sort_by_parsimony(model, X_train, y_train)
        return model
    
    # Aplica L_eff = MSE + λ * N_terms sobre o Pareto front
    def _operon_sort_by_parsimony(self, operon, X_train, y_train, lam=1e-4):
        backup = operon.model_
        for model in operon.pareto_front_:
            operon.model_ = model['tree']
            y_pred = operon.predict(X_train)
            mse    = np.mean((y_train - y_pred) ** 2)
            n_terms = model['tree'].Length
            l_eff  = mse + lam * n_terms
            model['l_eff'] = l_eff
        operon.model_ = backup

    # Treinar Operon passando configuração
    def treinar_operon(self, operon_config, X, y, plot=False):
        modelo = SymbolicRegressor(**operon_config)
        modelo.fit(X, y)
        self._operon_select_by_r2(modelo)
        if plot: self.plot_entropy(modelo)
        return modelo

    def plot_entropy(self, modelo):
        df_operon = self.process_metricas(pd.DataFrame(modelo.pareto_front_))
        self.plotly_entropy(df_operon)

    # Calcula R2 e seleciona o modelo com o maior valor
    def _operon_select_by_r2(self, operon):
        for model in operon.pareto_front_:
            model['r2'] = -model['objective_values'][0] # deve ser o primeiro!
        best = max(operon.pareto_front_, key=lambda x: x['r2'])
        operon.model_ = best['tree']
        operon.stats_['model_r2'] = best['r2']
        operon.stats_['model_length'] = best['length']
        operon.stats_['model_complexity'] = best['complexity']
    
    # Selecionar modelo do Operon com a complexidade desejada
    def select_by_complexity(self, df_operon, complexity):
        selected_model = df_operon[df_operon['complexity'] == complexity]['tree']
        
        if selected_model.empty:
            print(f"Modelo com complexidade {complexity} não encontrado.")
            return None
        else:
            return selected_model.iloc[0]

    def mostrar_equacao(self, operon, col_x, formato='texto'):
        """
        Mostra a equação de um modelo do pyoperon.

        Parameters:
        -----------
        operon : SymbolicRegressor
            O modelo do pyoperon
        formato : str, {'jupyter', 'latex'}
            'jupyter': Mostra formatado no Jupyter com IPython.display
            'latex': Retorna string LaTeX pura para copiar
        """
        # for model_dict in operon.pareto_front_:
        #     if model_dict['tree'] is operon.model_:
        #         complexity = model_dict['complexity']
        #         break
        # print("Complexity: %d"%complexity)
        equation_str = operon.get_model_string(operon.model_, names=[col_x], precision=6)        
        from sympy import latex
        eq_latex = latex(equation_str)

        if formato == 'jupyter':
            display(Latex(f'$${eq_latex}$$'))
            
        elif formato == 'latex':
            print(f"\nLaTeX: \n{eq_latex}")
        
        elif formato == 'texto':
            print(f"Equação: \n{equation_str.replace('^', '**')}")
        
        else:
            raise ValueError("formato deve ser 'jupyter', 'latex' ou 'texto'")

    # Definir treinamento do modelo conformal
    def conformal_regression_q(self, X_conform, y_conform, regressor, level=CONF_LEVEL):
        mapie = ConformalizedQuantileRegressor(
            estimator=regressor, confidence_level=level, prefit=True
        )
        mapie.conformalize(X_conform.reshape(-1, 1), y_conform)
        return mapie
    
    def conformal_regression_s(self, X_conform, y_conform, regressor, level=CONF_LEVEL):
        mapie = SplitConformalRegressor(
            estimator=regressor, confidence_level=level, prefit=True
        )
        mapie.conformalize(X_conform.reshape(-1, 1), y_conform)
        return mapie

    # Evaluate prediction and coverage level on testing set
    def conformal_score(self, X_test, y_test, mapie):
        y_pred, y_pis = mapie.predict_interval(X_test.reshape(-1, 1))
        coverage = regression_coverage_score(y_test, y_pis)[0]
        return coverage, y_pred, y_pis.reshape(-1, 2)
    
    # Gera pontos amostrais dentro do intervalo de conformidade
    def gerar_amostras_normais(self, df_medias, df_quantis, qtd=3, sigma_factor=12):
        dados = []
        
        for i, (media, (q_inf, q_sup)) in enumerate(zip(df_medias, df_quantis)):
            #std = (q_sup - q_inf) / sigma_factor
            std = 1 / sigma_factor # 4*qtd
            a = (q_inf - media) / std
            b = (q_sup - media) / std
            amostras = truncnorm.rvs(a, b, loc=media, scale=std, size=qtd)
            
            for j, amostra in enumerate(amostras):
                dados.append({'indice': i, 'amostra_id': j, 'amostra': amostra})
        
        return pd.DataFrame(dados)

    def conformal_plot(self, X_test, y_test, y_pred, y_pci, coverage, col_x, col_y):
        order = np.argsort(X_test)
        plt.figure(figsize=(6*self.phi, 6))
        plt.plot(X_test[order], y_pred[order], label=r"$\text{Mediana}_{\text{Operon}}$", color="green")
        title = "Operon com Split-Conformal"
        label = "Amostras no CI"
        if coverage > 0:
            plt.fill_between(
                X_test[order],
                y_pci[:, 0][order],
                y_pci[:, 1][order],
                alpha=0.4,
                label="Split Conformal Interval",
                color="green",
            )
            title += f", coverage={coverage:.3f}"
            label = "Conjunto de teste"
        plt.title(title)
        plt.scatter(X_test, y_test, color="red", alpha=0.6, label=label, s=2)
        plt.xlabel(col_x)
        plt.ylabel(col_y)
        plt.ylim([-1, 3])
        plt.legend(loc="lower left", fancybox=True, shadow=True)
        plt.show()

    # Juntar amostras num dataframe só
    def juntar_amostras(self, col_x):
        df_nii = pd.read_csv(f'amostras_{col_x}_{T.nii.value}.csv')
        df_oiii = pd.read_csv(f'amostras_{col_x}_{T.oiii.value}.csv')
        df_ha = pd.read_csv(f'amostras_{col_x}_{T.ha.value}.csv')
        df_hb = pd.read_csv(f'amostras_{col_x}_{T.hb.value}.csv')
        df_diagramas = df_nii.drop(columns=[T.nii.value]).merge(df_oiii[['index']], on=['index'])
        df_diagramas = df_diagramas.merge(df_nii[['index', T.nii.value]], on=['index'])
        df_diagramas = df_diagramas.merge(df_oiii[['index', T.oiii.value]], on=['index'])
        df_diagramas = df_diagramas.merge(df_ha[['index', T.ha.value]], on=['index'])
        df_diagramas = df_diagramas.merge(df_hb[['index', T.hb.value]], on=['index'])
        df_diagramas = df_diagramas.drop(columns=['indice', 'amostra_id'])
        df_diagramas[T.nii_ha.value] = df_diagramas[T.nii.value] - df_diagramas[T.ha.value]
        df_diagramas[T.oiii_hb.value] = df_diagramas[T.oiii.value] - df_diagramas[T.hb.value]
        return df_diagramas

    # calcula norma R2 no espaço BPT
    def mean_norma_r2(self, x1, y1, x2, y2):
        assert len(x1) == len(x2)
        assert len(x2) == len(y1)
        assert len(y1) == len(y2)
        N = len(x1)
        norma = 0.0
        for i in range(N):
            norma += np.sqrt((x2[i] - x1[i]) ** 2 + (y2[i] - y1[i]) ** 2)
        return norma / N

    # Write scores on disk
    def get_scores(self, col, scores, funcao):
        txt = col+"\n"
        txt += "MSE(train): %.6f\n"%scores["mse_train"]
        txt += "R2(train): %.6f\n"%scores["r2_train"]
        txt += "MSE: %.6f\n"%scores["mse"]
        txt += "R2: %.6f\n"%scores["r2"]
        txt += "%s\n"%funcao
        return txt

    def write_header(self, nome):
        txt = "col;mse_train;r2_train;mse;r2;complexity;funcao\n"
        file = open("metricas_%s.csv"%nome, "w")
        file.write(txt)
        file.close()

    def write_scores(self, nome, col, complexity, scores, funcao):
        txt = "%s;"%col
        txt += "%.6f;"%scores["mse_train"]
        txt += "%.6f;"%scores["r2_train"]
        txt += "%.6f;"%scores["mse"]
        txt += "%.6f;"%scores["r2"]
        txt += "%d;"%complexity
        txt += "%s\n"%funcao
        file = open("metricas_%s.csv"%nome, "a")
        file.write(txt)
        file.close()

    def calc_scores(self, y_train, y_test, y_pred_train, y_pred_test):
        # Avaliar os modelos
        mse_train = mean_squared_error(y_train, y_pred_train)
        r2_train = r2_score(y_train, y_pred_train)
        mse = mean_squared_error(y_test, y_pred_test)
        r2 = r2_score(y_test, y_pred_test)
        scores = {
            "mse_train": mse_train,
            "r2_train": r2_train,
            "mse": mse,
            "r2": r2
        }
        return scores
    
    # Pega combinações de funções do arquivo
    def get_symbols(self):
        symbols = []
        with open("funcoes.txt", "r") as file:
            line = file.readline()
            while line != "":
                if(not line.startswith('!')):
                    symbols.append(line.replace('\n', ''))
                line = file.readline()
        return symbols
    
    # Gerar bins e respectivas medianas de colunas
    def bins_and_medians(self, x, y, M):
        # Verifique se x e y têm o mesmo comprimento
        if len(x) != len(y):
            raise ValueError("Vectors x and y must have the same length.")
        
        # Definir as bordas dos bins e os centros dos bins
        x_min = min(x)
        x_max = max(x)
        bin_edges = np.linspace(x_min, x_max, M + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Centros dos bins
        
        # Inicializa vetores para armazenar as estatisticas
        medianas = np.empty(M)
        mads = np.zeros(M)
        
        # Loop sobre cada bin e estimar a mediana dos valores de y no bin
        for i in range(M):
            # Identificar os índices dos valores de x que estão no bin atual
            bin_indices = np.where((x >= bin_edges[i]) & (x < bin_edges[i + 1]))[0]
            
            # Se o bin não estiver vazio, calcular a mediana dos valores de y
            #qmin[i] = np.quantile(y_values_in_bin, 0.025)
            if len(bin_indices) > 0:
                y_values_in_bin = y[bin_indices]
                medianas[i] = np.median(y_values_in_bin)
                mads[i] = median_abs_deviation(y_values_in_bin)
            else:
                medianas[i] = np.nan
                mads[i] = np.nan
        
        # Retornar um DataFrame com os centros dos bins e as medianas
        return pd.DataFrame(
            {'center': bin_centers, 
             'median': medianas,
             'mads': mads
             }
            ).dropna().reset_index(drop=True)

    # Expandir funções salvas pelo Operon
    def expand_f(self, func_, features):
        func = func_.replace("^", "**")
        nps = ["exp", "log", "sqrt", "abs", "sin", "tan", "asin", "atan", "acos", "cosh"]
        for j in range(len(nps)):
            func = func.replace(nps[j], "np." + nps[j])
        func = func.replace("np.np.", "np.")
        func = func.replace("anp.sin", "np.asin")
        func = func.replace("anp.tan", "np.atan")
        for j in range(len(features)):
            func = func.replace("X%d" % (j + 1), features[j])
        return func

    def predict_operon(self, dados, func_, features):
        func = self.expand_f(func_, features)
        df = pd.DataFrame(dados, columns=features)
        try:
            pred = df.apply(lambda row: eval(func, {"np": np}, row), axis=1)
            return pred
        except:
            print(func)

    def process_metricas(self, metricas):
        metricas["loss"] = metricas["mean_squared_error"]
        metricas["equations"] = metricas["model"]
        #metricas["complexity"] = metricas["objective_values"].apply(lambda x: int(x[1]))        
        #metricas.drop(["tree"], axis=1, inplace=True)
        metricas.drop(["model"], axis=1, inplace=True)
        metricas.drop(["variables"], axis=1, inplace=True)
        metricas.drop(["objective_values"], axis=1, inplace=True)
        metricas.drop(["mean_squared_error"], axis=1, inplace=True)
        bests = metricas.loc[
            metricas.groupby("complexity")["loss"].idxmin()
        ]
        bests = bests[bests["complexity"] > 10].reset_index(drop=True)
        return bests
        
        
    def get_operon_func(self, target, feature, complexity):
        df_operon = self.get_scores_operon(target, feature)
        if complexity == -1: complexity = np.max(df_operon["complexity"])
        scores_operon = df_operon[df_operon["complexity"] == complexity].iloc[0]
        operon_expr = sp.latex(sp.N(sp.sympify(scores_operon["equations"].replace('X1', 'x')), 4))
        operon_func = sp.lambdify(sp.symbols('x'), sp.sympify(
            scores_operon["equations"].replace('X1', 'x')), "numpy")
        return scores_operon, operon_expr, operon_func
    '''
    def get_pysr_func(self, path, complexity):
        df_pysr = PySRRegressor.from_file(run_directory="scores_pysr/"+path).equations_
        scores_pysr = df_pysr[df_pysr["complexity"] == complexity].iloc[0]
        pysr_expr = sp.latex(sp.N(sp.sympify(scores_pysr["equation"].replace('x0', 'x')), 4))
        pysr_func = sp.lambdify(sp.symbols('x'), sp.sympify(
            scores_pysr["equation"].replace('x0', 'x')), "numpy")
        return scores_pysr, pysr_expr, pysr_func
    '''  
    def get_scores_operon(self, target, feature, save=False):
        file = target.split('_')[0] + '_' + feature.split('_')[0]
        df = pd.read_csv("scores_operon/%s.csv"%file)
        if save:
            txt = open('./regressoes/%s_%s_equations.tex'%(target, feature), 'w')
            txt.writelines(self.get_latex_table(df, feature, target))
            txt.close()
        return df
    
    def get_latex_table(self, df, col_x, col_y):
        titulo = 'Equações do Operon para: $%s \\times %s$'%(col_y, col_x)
        titulo = titulo.replace('_', r'\_')
        
        latex = "\\begin{table}[h!]\n\\centering\n"
        latex += f"\\caption{{{titulo}}}\n"
        latex += "\\begin{tabular}{|c|" + ("c|" * 2) + "}\n"
        latex += "\\hline\n"
        
        headers = ["Equação"]
        headers.extend(['complexity', 'loss'])
        latex += " & ".join(headers) + " \\\\\n\\hline\n"
        for _, row in df.iterrows():
            linha = [f"${sp.latex(sp.N(sp.sympify(row['equations'].replace('X1', 'x')), 3))}$"]
            linha.extend(['%d'%(row['complexity'])])
            linha.extend(['%.4f'%(row['loss'])])
            latex += " & ".join(linha) + " \\\\\n"
        
        latex += "\\hline\n\\end{tabular}\n\\end{table}"
        return latex

    # Realiza testes de hipótese do tipo goodness-of-fit
    def pdf_tests(self, esperado, modelado, metrica=''):
        if metrica == 'ks':
            ks = stats.ks_2samp(esperado, modelado, method='asymp')
            return "KS p-value: %.3e"%(ks.pvalue)
        elif metrica == 'mw':
            mw = stats.mannwhitneyu(esperado, modelado, method='asymptotic')
            return "MW p-value: %.4f"%(mw.pvalue)
        elif metrica == 'ws':
            ws = stats.wilcoxon(esperado, modelado, method='asymptotic')
            return "WS p-value: %.4f"%(ws.pvalue)
        return ''

    # Histogramas dos dados
    def histogram(self, esperado, modelado, label1, label2, title):
        legend = "%s \n"%(label1) + self.pdf_tests(esperado, modelado)
        plt.subplots(figsize=(10, 4))
        plt.hist(esperado, bins=50, alpha=0.95, label="Síntese"+label2, density=False)
        plt.hist(modelado, bins=50, color="red", alpha=0.65, label=legend, density=False)
        plt.xlim([-2, 1.5])
        plt.tight_layout()
        plt.title(title)
        plt.ylabel('counts')
        plt.legend()
        plt.show()

    # Plot de PDF estimado dos dados
    def kde_plot(self, esperado, modelado, label1, label2, title, metrica):
        legend = "%s \n"%(label1) + self.pdf_tests(esperado, modelado, metrica)
        plt.subplots(figsize=(10, 4))
        sns.kdeplot(data=esperado, label='Síntese'+label2, alpha=0.6)
        sns.kdeplot(data=modelado, label=legend, alpha=0.6)
        #plt.xlim([-2, 1.5])
        plt.title(f'{title} PDF')
        plt.xlabel('')
        plt.ylabel('density')
        plt.legend()
        plt.savefig('KDE '+title+'.png', dpi=150)
        plt.close()
        
    def get_densities(self, x, y):
        # Estimar a densidade 2D
        xy = np.vstack([x, y])
        kde = gaussian_kde(xy, bw_method=0.3)
        
        # Criar grade para contorno
        xgrid = np.linspace(x.min(), x.max()+1, 100)
        ygrid = np.linspace(y.min()-1, y.max()+1, 100)
        X, Y = np.meshgrid(xgrid, ygrid)
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = kde(positions).reshape(X.shape)
        
        return X, Y, Z, kde
    
    def get_levels(self, x, y):
        X, Y, Z, kde = self.get_densities(x, y)
        densities = kde([x, y])
        lvls = np.quantile(densities, [0.1, 0.2, 0.4, 0.6])
        return X, Y, Z, lvls

    # Processar dados das métricas dos testes para k=10 repetições ou len(symbols)=32 funções
    def plot_metricas(self, dados_r, col, hyper, metrica, cor):
        if col == None:
            dados = dados_r
        else: 
            dados = dados_r.loc[(dados_r["col"] == col)]
        #m_min = min(dados[metrica])
        #arg_min = dados.loc[dados[metrica] == m_min][hyper].iloc[0]
        #label_min = "min(%s) = %.4f\nargmin(%s) = %s" % (metrica, m_min, metrica, arg_min)
        label_mean = (
            r"$\mu$(%s) = %.4f" % (metrica, np.mean(dados[metrica])) + "\n" + 
            r"$\sigma$(%s) = %.4f" % (metrica, np.std(dados[metrica]))
        )
        plt.subplots(figsize=(10, 5))
        if col == None:
            plt.title("Métricas")
        else:
            plt.title("Métricas: %s" % (col))
        plt.plot(range(1, len(dados[metrica])+1), dados[metrica], "%s-" % cor, label=label_mean)
        plt.ylabel(metrica.upper())
        plt.xticks(range(1, len(dados[metrica])+1), rotation=90, fontsize=7)
        plt.tick_params(axis="y", labelcolor=cor)
        plt.legend()
        # plt.savefig("metricas_operon_%s_%s.png"%(col, metrica), dpi=150)
        plt.show()
        plt.close()

    # Métricas das razões indiretas
    def plot_metrica_ind(self, title, mse, lbl=2):
        label_min = "min = %.4f" % (min(mse))
        label_mean = (
            r"$\mu$ = %.4f" % (np.mean(mse)) + "\n" +
            r"$\sigma$ = %.4f" % (np.std(mse))
        )
        label = ""
        if lbl == 1:
            label = label_min
        else:
            label = label_mean
        plt.subplots(figsize=(10, 5))
        plt.title(title)
        plt.plot(range(1, 33), mse, "r-", label=label)
        plt.ylabel("MSE")
        plt.xticks(range(1, 33), rotation=90, fontsize=7)
        plt.tick_params(axis="y", labelcolor="r")
        plt.legend()
        plt.show()

    # Plotar métricas comparando parâmetro symbolic do Operon
    def plot_metrica_symbolic(self, metricas):
        dados_true = metricas.loc[(metricas["symbolic"] == True)]
        dados_false = metricas.loc[(metricas["symbolic"] == False)]
        metrica = "mse"

        plt.subplots(figsize=(10, 5))

        label_mean = (
            "Symbolic: True \n" +
            r"$\mu$(%s) = %.4f" % (metrica, np.mean(dados_true[metrica])) + "\n" + 
            r"$\sigma$(%s) = %.4f" % (metrica, np.std(dados_true[metrica]))
        )
        plt.plot(self.targets, dados_true[metrica], "b-", label=label_mean)

        label_mean = (
            "Symbolic: False \n" +
            r"$\mu$(%s) = %.4f" % (metrica, np.mean(dados_false[metrica])) + "\n" + 
            r"$\sigma$(%s) = %.4f" % (metrica, np.std(dados_false[metrica]))
        )
        plt.plot(self.targets, dados_false[metrica], "r-", label=label_mean)

        plt.title("Métricas Operon")
        plt.ylabel("MSE")
        plt.xticks(range(0, len(dados_true[metrica])), rotation=45, fontsize=7)
        plt.tick_params(axis="y", labelcolor="r")
        plt.legend()
        plt.show()

    def plot_fit(self, y_test, y_pred, mse, coluna, modelo):
        plt.figure(figsize=(8, 6))
        plt.hexbin(y_test, y_pred, gridsize=50, cmap='magma', mincnt=0)
        plt.colorbar(label='pred_count')

        linha = [min(y_test), max(y_test)]
        plt.plot(linha, linha, "r--", label="Real\nMSE: %.4f"%mse, alpha=0.6)
        plt.xlim(min(y_pred), max(y_pred))
        plt.ylim(min(y_pred), max(y_pred))

        plt.xlabel('Real')
        plt.ylabel('Predito')
        plt.title("%s - %s"%(modelo, coluna))
        plt.legend()
        plt.tight_layout()
        plt.savefig("y_test_%s_%s.png"%(modelo, coluna), dpi=150)
        plt.close()

    def plot_xy(self, x, y, lim=None):
        plt.scatter(x=x, y=y, alpha=0.2, s=1)
        if lim is not None:
            plt.xlim(lim)
            plt.ylim(lim)
        plt.xlabel(x.name)
        plt.ylabel(y.name)
        plt.show()
    
    def histogram_v(self, dados, title, ax, density=False,  lim=None, cor=None):
        ax.set_xlabel('Valor')
        ax.set_ylabel('Contagem')
        if lim is not None: ax.xlim(lim)
        ax.hist(dados.astype("float"), bins=100, density=density, color=cor)
        ax.set_title(title)

    def histogram_h(self, dados, title, density=False, lim=None, cor=None):
        plt.subplots(figsize=(5*self.phi, 5))
        plt.ylabel('Valor')
        plt.xlabel('Contagem')
        if lim is not None: plt.ylim(lim)
        plt.hist(dados, bins=100, orientation='horizontal', density=density, color=cor)
        plt.tight_layout()
        plt.title(title)
        plt.show()
        
    def filtra_median(self, df_median, filtro):
        df_median = df_median[(df_median['center'] >= filtro[0]) & (df_median['center'] <= filtro[1])]
        X_train = np.array(df_median["center"].values).reshape(-1, 1)
        return df_median, X_train
        
    # Plots das medianas dos conjuntos de dados
    def plot_median(self, dados, col_x, col_y, df_medians=None):
        if df_medians is None:
            df_medians = self.bins_and_medians(dados[col_x], dados[col_y], 1000)
        plt.figure(figsize=(5*self.phi, 5))
        plt.scatter(x=dados[col_x], y=dados[col_y], s=0.5, color='gray', alpha=0.5, label='Dados')
        plt.scatter(x=df_medians["center"], y=df_medians["median"], s=3, color="red", label="Medianas")
        plt.xlabel(col_x, fontsize='large'); plt.ylabel(col_y, fontsize='large')
        plt.legend(fontsize='large', loc='lower left')
        plt.show()
    
    def calcula_pvalues(self, medians, corte):
        errors_1 = (medians[medians["center"] < corte]["mads"])**2
        X1 = sm.add_constant(medians[medians["center"] < corte]["center"])
        stat_1, pvalue_1, _, _ = het_breuschpagan(1 / (errors_1 + 0.01), X1)
        errors_2 = (medians[medians["center"] >= corte]["mads"])**2
        X2 = sm.add_constant(medians[medians["center"] >= corte]["center"])
        stat_2, pvalue_2, _, _ = het_breuschpagan(1 / (errors_2 + 0.01), X2)
        return pvalue_1, pvalue_2
        
    def calcula_corte_cedasticidade(self, medians, plot=False, col_x='', col_y=''):
        M = int(np.size(medians["center"]) / 2)-5
        pvalues = np.zeros([2, M-5])
        for i in range(5, M):
            pvalues[:,i-5] = self.calcula_pvalues(medians, medians["center"].iloc[i])

        corte1 = medians["center"].iloc[5+np.argmin(pvalues[0,:int(M//1.5)])]
        corte2 = medians["center"].iloc[5+np.argmax(pvalues[1,:M])]
        corte = (corte1 + corte2) / 2
        
        if plot:
            plt.figure(figsize=(8, 8/3*2))
            plt.plot(medians["center"][5:M], pvalues[0,:M-5], 
                     label="P-value à esquerda", color='red')
            plt.plot(medians["center"][5:M], pvalues[1,:M-5], 
                      label="P-value à direita", color='green')
            plt.vlines(corte, -0.1, 1.1, 'purple', label="Corte: %.2f"%corte)
            plt.hlines(0.05, min(medians["center"][5:M]), max(medians["center"][5:M]), 
                       'blue', 'dashed', label="Nível de significância = 0.05")
            plt.xlabel(col_x)
            plt.ylabel(col_y)
            plt.title("Teste de Breusch-Pagan para Homocedasticidade")
            plt.legend()
            plt.grid(True)
            plt.show()
        
        return corte
    
    def calcula_corte_mad(self, medians, limiar=0.3):
        meio = np.median(medians["center"])
        corte = np.max(medians[(medians["mads"] > limiar) & 
                               (medians["center"] < meio) & 
                               (medians["mads"] > 0)]["center"])
        return corte
    
    def calcula_corte_outliers(self, medians, limiar=1):
        meio = np.median(medians["center"])
        corte = np.max(medians[(np.abs(3*medians["mads"]/medians["median"]) > limiar) & 
                               (medians["center"] < meio) & (medians["mads"] > 0)]["center"])
        return corte
    
    def calcula_corte_bayes(self, df, alpha0=1e-3, beta0=1e-3, prior='uniform'):
        """
        Compute posterior over a single variance change at index k (1..N-1).
        Returns a DataFrame with columns: k, x0, log_post, posterior.
        """
        x = np.asarray(df["center"], dtype=float)
        V = np.asarray(df["median"], dtype=float)
        if x.shape != V.shape:
            raise ValueError("x and V must have the same shape.")
        N = V.size
        if N < 3:
            raise ValueError("Need at least 3 points.")
        
        # 1) sort by x
        ord_idx = np.argsort(x)
        x = x[ord_idx]
        V = V[ord_idx]
        
        # 2) prefix sums of squares for O(1) segment SSE
        V2 = V**2
        csum = np.concatenate(([0.0], np.cumsum(V2)))
        
        # 3) grid of change indices and midpoint locations
        k_grid = np.arange(1, N)
        x0_grid = 0.5 * (x[:-1] + x[1:])
        
        # 4) prior on k
        if prior == 'uniform':
            log_prior_k = -np.log(N - 1) * np.ones(N - 1)
        elif prior == 'length_weighted':
            gaps = x[1:] - x[:-1]
            w = gaps / gaps.sum()
            log_prior_k = np.log(np.maximum(w, np.finfo(float).tiny))
        else:
            raise ValueError("prior must be 'uniform' or 'length_weighted'.")
    
        # 5) log-marginal likelihood as a function of k
        def log_marg(k):
            n1 = k
            n2 = N - k
            S1 = csum[k] - csum[0]
            S2 = csum[N] - csum[k]
            if alpha0 == 0 and beta0 == 0:
                # Jeffreys' limit: proportional to S1^{-n1/2} * S2^{-n2/2}
                if S1 <= 0 or S2 <= 0:
                    return -np.inf
                return -(n1 / 2.0) * np.log(S1) - (n2 / 2.0) * np.log(S2)
            else:
                lg1 = gammaln(alpha0 + n1 / 2.0) - (alpha0 + n1 / 2.0) * np.log(beta0 + S1 / 2.0)
                lg2 = gammaln(alpha0 + n2 / 2.0) - (alpha0 + n2 / 2.0) * np.log(beta0 + S2 / 2.0)
            return lg1 + lg2
    
        lmarg = np.array([log_marg(int(k)) for k in k_grid])
        log_post = lmarg + log_prior_k
        m = np.max(log_post)
        post = np.exp(log_post - m)
        post /= post.sum()
    
        posterior_df = pd.DataFrame({'x0': x0_grid, 'log_post': log_post, 'posterior': post})
        #print(posterior_df.head())
        
        # MAP and equal-tail 90% CI for x0 from the discrete posterior.
        df = posterior_df.sort_values('x0').reset_index(drop=True)
        cdf = df['posterior'].cumsum().to_numpy()
        x0 = df['x0'].to_numpy()
    
        map_idx = posterior_df['posterior'].idxmax()
        x0_map = posterior_df.loc[map_idx, 'x0']
    
        # interpolate quantiles on the discrete grid
        q025 = np.interp(0.025, cdf, x0)
        q975 = np.interp(0.975, cdf, x0)
    
        return x0_map, q025, q975
    
    def plot_tstudent_fit(self, medians):
        meio = np.median(medians["center"])
        medians["razoes"] = np.abs(medians["mads"]/medians["median"])
        dados = medians[(medians["center"] < meio) & (medians["mads"] > 0)]["razoes"]

        # Fitar tStudent
        df, loc, scale = stats.t.fit(dados)
        print(f"Parâmetros estimados: graus de liberdade={df:.3f}, loc={loc:.3f}, scale={scale:.3f}")

        # Visualizar o ajuste
        plt.hist(dados, bins=30, density=True, histtype='step', color='blue', alpha=0.5, label='Dados')
        x = np.linspace(min(dados), 2, 100)
        pdf = stats.t.pdf(x, df, loc, scale)
        plt.plot(x, pdf, 'r-', label='Distribuição t-Student Ajustada')
        plt.xlabel('Valor')
        plt.ylabel('Densidade')
        plt.title('Ajuste da Distribuição t-Student')
        plt.xlim(min(dados), 2)
        plt.legend()
        plt.show()
    
    def plotly_entropy(self, df, loss="loss"):
        fig = px.scatter(df, x="complexity", y=loss, title="Fronteira de Pareto")
        fig.update_xaxes(
            tickvals=df["complexity"]
        )
        # Calcular a inclinação para cada par de pontos consecutivos
        inclinacoes = []
        x_mids = []
        y_mids = []
        
        for i in range(1, len(df)):
            x1, y1 = df["complexity"].iloc[i-1], df[loss].iloc[i-1]
            x2, y2 = df["complexity"].iloc[i], df[loss].iloc[i]
            
            # Calcular a inclinação
            if x2 - x1 != 0:  # Evitar divisão por zero
                inclinacao = -np.arctan((y2 - y1) / (x2 - x1) / max(df[loss]))*180/np.pi
            else:
                inclinacao = 0
            
            inclinacoes.append(inclinacao)
            
            # Ponto médio para posicionar a anotação
            x_mids.append((x1 + x2) / 2)
            y_mids.append((y1 + y2) / 2)
            
            # Adicionar uma linha entre os pontos
            fig.add_trace(
                go.Scatter(
                    x=[x1, x2],
                    y=[y1, y2],
                    mode='lines',
                    line=dict(color='red', dash='dash', width=1),
                    showlegend=False
                )
            )
        
        # Adicionar anotações com os valores das inclinações
        for i in range(1, len(inclinacoes)):
            if (inclinacoes[i] < np.quantile(inclinacoes, 0.50) or inclinacoes[i] < 0 or inclinacoes[i-1] < -1):
                continue
            fig.add_annotation(
                x=x_mids[i],
                y=y_mids[i],
                text=f"θ = {inclinacoes[i]:.2f}°",
                bgcolor="white",
                opacity=1,
                showarrow=True,
                arrowhead=1,
                axref="pixel",
                ayref="pixel",
                ax=25,
                ay=-50,
                font=dict(size=14)
            )
        
        fig.update_xaxes(
            tickvals=df["complexity"]
        )
        fig.update_layout(
            showlegend=False,
            xaxis=dict(showgrid=True)
        )
        fig.show(renderer="notebook")
            
    def reg_linear(self, x, y):
        # Converter x e y para arrays numpy
        x = np.array(x)
        y = np.array(y)
        
        # Função para calcular a soma dos quadrados dos resíduos
        def objective(slope):
            intercept = y[-1] - slope * x[-1]
            y_pred = slope * x + intercept
            return np.sum((y_pred - y) ** 2)
        
        # Otimizar para encontrar o melhor slope
        result = minimize(objective, x0=0.0)  # x0 é o chute inicial para o slope
        slope = result.x[0]
        
        # Calcular o intercepto com base no último ponto
        intercept = y[-1] - slope * x[-1]
        
        # Retorna coeficientes da reta
        return Struct(A=slope, B=intercept)
    
    def reg_quantilica(self, x, y, quantile):
        # Converter x e y para arrays numpy
        x = np.array(x)
        y = np.array(y)
        
        qr = QuantileRegressor(quantile=quantile, alpha=0)
        qr.fit(x.reshape(-1, 1), y.reshape(-1, 1))
        
        # Calcular o intercepto com base no último ponto
        slope = qr.coef_[0]
        intercept = y[-1] - slope * x[-1]
        
        # Retorna coeficientes da reta
        return Struct(A=slope, B=intercept)
    
    def reg_linear_last_y_fixed(self, X, Y, funcao, quantil):
        last_index = Y.index[-1]
        Y.loc[last_index] = funcao(X.loc[last_index]) ## fixa Y final na função
        if quantil == None:
            return self.reg_linear(X, Y)
        else:
            return self.reg_quantilica(X, Y, quantil)
    
    def plot_median_corte(self, dados, x, y, corte):
        median = self.bins_and_medians(dados[x], dados[y], 1000)
        ajuste = median['center'] >= corte
        
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        ax.scatter(x=dados[x], y=dados[y], s=1, color='lightgray', label="Dados")
        
        ax.scatter(x=median["center"][ajuste], 
                      y=median["median"][ajuste],
                      s=2, color="red", label="Medianas")
        ax.scatter(x=median["center"][~ajuste],
                      y=median["median"][~ajuste],
                      s=2, color="darkred", alpha=0.5)
        ax.vlines(corte, np.min(dados[y]), np.max(dados[y]), 
                  color="purple", label="%s ~ %.2f"%(x, corte))
        
        ax.set_xlabel(self.unidades[x], fontsize='x-large')
        ax.set_ylabel(self.unidades[y], fontsize='x-large')
        ax.set_title("Dados & Medianas", fontsize='xx-large')
        ax.legend(fontsize='large')
    
        plt.show()
        plt.close()
    
    def plot_median_fn(self, dados, x, y, complexity, quantil=None, corte=None,
                       titulo="", save=False):
        median = self.bins_and_medians(dados[x], dados[y], 1000)
        _, _, funcao = self.get_operon_func(y, x, complexity)
        if corte is None:
            print("Calculando corte automaticamente..")
            corte = self.calcula_corte_cedasticidade(median)
        
        fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, 
                               figsize=(8.5, 5.5), sharex=True)
        # Gráfico original (parte superior)
        ax[0].scatter(x=dados[x], y=dados[y], s=1, color='lightgray')
        
        ajuste = median['center'] >= corte
        linear = median['center'] <  corte
        
        ax[0].scatter(x=median["center"][ajuste], 
                      y=median["median"][ajuste],
                      s=2, color="red")
        ax[0].scatter(x=median["center"][~ajuste],
                      y=median["median"][~ajuste],
                      s=2, color="darkred", alpha=0.5)
        
        X = median["center"][ajuste]
        Y = funcao(X)
        
        if (len(median[linear]) > 0):
            X_ = median["center"][linear].copy()
            Y_ = median["median"][linear].copy()
            reta = self.reg_linear_last_y_fixed(X_, Y_, funcao, quantil)
            Y_ = reta.A * X_ + reta.B
            ax[0].plot(X_, Y_, color="green")
            Y = np.concatenate((Y_, Y))
            X = np.concatenate((X_, X))
        
        mae = np.mean(np.abs(median["median"] - Y))
        mse = np.mean((median["median"] - Y)**2)
        ax[0].plot(X, Y, color="green", 
                   label='Complex.: %d | Mse: %.4f | Mae: %.4f'
                   %(complexity, mse, mae))
        
        ax[0].set_ylabel(self.unidades[y], fontsize='x-large')
        if titulo != "": ax[0].set_title(titulo, fontsize='xx-large')
        ax[0].legend(fontsize='large')
    
        # Gráfico de resíduos (parte inferior)
        residuos = median["median"] - Y
        ax[1].scatter(median["center"], residuos, s=2, color='blue')
        ax[1].axhline(0, color='black', linestyle='dashed')
        
        ax[1].set_xlabel(self.unidades[x], fontsize='x-large')
        ax[1].set_ylabel("Resíduos", fontsize='x-large')
    
        if save:
            plt.savefig("./regressoes/reg_%s_%s_%d.png"%(x, y, complexity), dpi=100)
        else:
            plt.show()
        plt.close()
        
        return X, Y

    def plot_median_mad_movel(self, medians, col_x, col_y, corte=None):
        if corte is None:
            corte = self.calcula_corte_bayes(medians)
        pvalue_antes, pvalue_depois = self.calcula_pvalues(medians, corte)
        #mad_m10 = medians['mads'].rolling(window=10).mean()
        # -----------------------------
        plt.figure(figsize=(8, 8/3*2))
        plt.plot(medians["center"], medians["median"], color="blue", label="Medianas")
        plt.scatter(x=medians["center"], y=medians["median"]+medians["mads"], 
                    s=2, label=r"$\pm$MAD", color="red")
        plt.scatter(x=medians["center"], y=medians["median"]-medians["mads"], 
                    s=2, color="red")
        #plt.plot(medians["center"], mad_m10, color="red", label="Média-móvel do MAD")
        plt.vlines(corte, np.min(medians["median"]), np.max(medians["median"]), 
                   color="purple", label="%s ~ %.2f \nBP p-value: %.4f"
                   %(col_x, corte, pvalue_antes))
        plt.xlabel(col_x)
        plt.ylabel(col_y)
        plt.title("Median absolute deviation per bin")
        plt.legend()
        plt.show()
        
    def plot_col_cortes(self, medians, col_x, col_y, cortes=None):
        # -----------------------------
        plt.figure(figsize=(8, 8/3*2))
        plt.scatter(x=medians["center"], y=medians["median"], 
                    s=2, label="Medianas", color="red")
        if cortes is not None:
            plt.vlines(cortes[0], np.min(medians["median"]), np.max(medians["median"]), 
                       color="purple", label="Corte: %.2f"%(cortes[0]))
            plt.vlines(cortes[1], np.min(medians["median"]), np.max(medians["median"]), 
                       color="blue", label="CI-95%%: [%.2f,%.2f]"%(cortes[1], cortes[2]))
            plt.vlines(cortes[2], np.min(medians["median"]), np.max(medians["median"]), 
                       color="blue")
        plt.xlabel(self.unidades[col_x])
        plt.ylabel(self.unidades[col_y])
        plt.title("Medianas e regimes de desvios-padrão")
        plt.legend()
        plt.show()
       
    def plot_stats(self, dados, col_x, col_y, df_medians):
        plt.figure(figsize=(8, 8/3*2))
        plt.plot(df_medians["center"], df_medians["std"], color="red", label="Desvios-padrão")
        plt.plot(df_medians["center"], df_medians["mad"], color="blue", label="Median-abs-dev")
        plt.xlabel(col_x, fontsize='x-large'); plt.ylabel(col_y, fontsize='x-large')
        plt.legend(fontsize='x-large')
        plt.show()
    
    # Gráficos de barras das métricas
    def plot_bar(self, forests, knn, operon, linhas, metrica, vmax):
        linhas.reverse()
        forests.reverse()
        knn.reverse()
        if operon is not None: operon.reverse()
        x = np.arange(len(linhas))
        width = 0.2

        _,ax = plt.subplots(figsize=(10, 6))

        ax.barh(x + width, forests, width, label='Forests')
        ax.barh(x, knn, width, label='KNN')
        if operon is not None: ax.barh(x - width, operon, width, label='Operon')

        for i in range(len(forests)):
            ax.text(forests[i]+2e-4, x[i]+width, f"{forests[i]:.4f}", color='black', va='center', ha='left')
            ax.text(knn[i]+2e-4, x[i], f"{knn[i]:.4f}", color='black', va='center', ha='left')
            if operon is not None: 
                ax.text(operon[i]+2e-4, x[i]-width, f"{operon[i]:.4f}", color='black', va='center', ha='left')

        ax.set_yticks(x)
        ax.set_yticklabels(linhas)
        ax.set_xlabel(metrica)
        ax.legend()

        plt.title('%s do conjunto de validação'%metrica)
        plt.xlim(0, vmax)
        plt.subplots_adjust(left=0.15, right=0.95, bottom=0.1, top=0.9)
        plt.show()
        
    # Plot ajuste de sigmas
    def plot_ajuste_sigma(dados, col_x, col_y, medians_50):
        plt.plot(medians_50['center'], medians_50['std'], 
                 color='red', label=r'$\sigma $ por bin')
        plt.hlines(np.std(dados[col_x]), np.min(medians_50['center']), 
                   np.max(medians_50['center']), label=r'$\sigma $ geral')
        plt.legend()
        plt.ylabel(r'$\sigma $(%s)'%col_y)
        plt.xlabel(col_x)
        plt.show()
        
    def std_linear(self, x, y, col_x, col_y):
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        
        # Plot the data and the fit
        plt.scatter(x, y, s=5, label="Desvio por bin", color="blue")
        plt.plot(x, slope * x + intercept, color="red", 
                 label="Linear fit, R-squared: %.4f"%(r_value**2))
        plt.xlabel(col_x)
        plt.ylabel("Std(%s)"%col_y)
        plt.legend()
        plt.show()
        
        return slope, intercept

    def std_poly(self, x, y, col_x, col_y, deg=3):
        # Fit a degree-3 polynomial
        coefficients = np.polyfit(x, y, deg=deg)
        polynomial = np.poly1d(coefficients)
        
        # Generate points for smooth plotting
        x_fit = np.linspace(min(x), max(x), 100)
        y_fit = polynomial(x_fit)
        
        # Plot original data and fitted curve
        plt.scatter(x, y, s=5, label="Desvio por bin", color="blue")
        plt.plot(x_fit, y_fit, label="Cubic fit", color="red")
        plt.xlabel(col_x)
        plt.ylabel("Std(%s)"%col_y)
        plt.legend()
        plt.show()
        
        # Return the polynomial function and coefficients
        return polynomial
    
    def salvar_grids(self, X, Y, Z, filename):
        """Salva grids X, Y, Z em formato relacional CSV"""
        df = pd.DataFrame({
            'X': X.ravel(),
            'Y': Y.ravel(),
            'Z': Z.ravel()
        })
        df.to_csv(filename, index=False)
        print(f"Grids salvos em {filename}")
    
    def ler_grids(self, filename):
        """Lê CSV e reconstrói os grids X, Y, Z"""
        df = pd.read_csv(filename)
        
        # Descobrir dimensões do grid
        n_unique_x = df['X'].nunique()
        n_unique_y = df['Y'].nunique()
        
        # Reconstruir grids
        X = df['X'].values.reshape(n_unique_y, n_unique_x)
        Y = df['Y'].values.reshape(n_unique_y, n_unique_x)
        Z = df['Z'].values.reshape(n_unique_y, n_unique_x)
        
        return X, Y, Z

    def plot_KeKa(self):
        # Linhas de separação
        x_ke01 = np.arange(-2, 0.3, 0.01)
        x_ka03 = np.arange(-2, 0, 0.01)
        ke_01 = 0.61 / (x_ke01 - 0.47) + 1.19
        ka_03 = 0.61 / (x_ka03 - 0.05) + 1.3
        plt.plot(x_ke01, ke_01, "--", color="black", alpha=0.5, label="Ke01")#: extreme starburst >")
        plt.plot(x_ka03, ka_03, "-", color="green", alpha=0.8, label="Ka03")#: pure star formation <")
        plt.text(-1.5, -0.5, "SF")
        plt.text(-0.1, -1.0, "Mix.")
        plt.text(0.6, 0.6, "AGN")

    # Plot BPT Diagrams
    def plot_bpt(self, x_test, y_test, bptx_pred=[], bpty_pred=[], 
                 qmin=[], qmax=[], label="", tipo="ambos", save=False):
        plt.subplots(figsize=(6*self.phi, 6))
        self.plot_KeKa()
        
        if tipo == "symb":
            alpha = 0.4
            lbl = " Operon + Normal4D"
        else:
            if tipo == "sintese":
                alpha = 0.3
            else:
                alpha = 0.3
            lbl = "Síntese"
        
        # Quantis
        if len(qmin) > 0 and len(qmin) == len(qmax):
            #yerr = [y_test - qmin, qmax - y_test]
            #plt.errorbar(x_test, y_test, yerr=yerr, fmt='o', color='darkblue', 
            #             ecolor='purple', label=lbl, alpha=alpha, 
            #             markersize=3, capsize=5)
            plt.plot(x_test, qmin, color='purple', alpha=alpha)
            plt.plot(x_test, qmax, color='purple', alpha=alpha)
            plt.fill_between(x_test, qmin, qmax, color='purple', alpha=0.06)
            plt.plot(x_test, y_test, '-o', color='darkblue',  markersize=3, label=lbl, alpha=alpha)
        else:
            plt.scatter(x=x_test, y=y_test, s=0.5, color="gray", alpha=alpha, label=lbl)
        
        # Estimativas da regressao
        if (tipo == "sintese"):
            self.curvas_densidade(x_test, y_test)
        else:
            plt.scatter(bptx_pred, bpty_pred, s=1, color=(0.571, 0.704, 0.997), alpha=alpha, label="%s"%label)
            self.curvas_densidade(bptx_pred, bpty_pred)
        
        # Configurações do gráfico
        plt.title("Diagrama BPT", fontsize='xx-large')
        plt.xlabel(r"$log_{10}$(W[NII] / WH$\alpha$)", fontsize='large')
        plt.ylabel(r"$log_{10}$(W[OIII] / WH$\beta$)", fontsize='large')
        
        plt.xticks(fontsize='medium')
        plt.yticks(fontsize='medium')
        plt.xlim(-2, 1)
        plt.ylim(-1.2, 1.5)
        plt.legend(fontsize='medium', markerscale=4)
        plt.tight_layout()
        if save:
            plt.savefig("%sbpt_%s.png"%('./bpts/', label.replace('(','_').replace(')','_')), dpi=100)
        else:
            plt.show()
        plt.close()

    def bpt_sintese(self, dados):
        self.plot_bpt(dados[T.nii_ha.value], dados[T.oiii_hb.value], tipo="sintese")
        
    def plot_KeKa06(self):
        # Linhas de separação
        plt.vlines(-0.4, np.log10(0.5) + 0.4, 3, alpha=0.8, label="S06 SF/AGN", color="darkblue")
        plt.hlines(np.log10(6), -0.4, 1, alpha=0.8, label="Ke06 Seyfert/LINER", color="green")
        plt.hlines(np.log10(3), -0.4, 1, alpha=0.8, label="Fe11 wAGN/RG", color="darkred")
        plt.hlines(np.log10(0.5), -2, 0, alpha=0.8, linestyles="--", color="black")
        plt.hlines(np.log10(0.5), 0, 1, alpha=0.8, linestyles=":", color="black")
        x = np.arange(-2, 0, 0.01)
        y = np.log10(0.5) - x
        plt.plot(x, y, ":", alpha=0.8, color="black")
        x = np.arange(0, np.log10(0.5)+0.8, 0.01)
        y = np.log10(0.5) - x
        plt.plot(x, y, "--", alpha=0.8, color="black")
        plt.text(-0.7, 2.5, "SF")
        plt.text(0, 1.5, "Seyfert")
        plt.text(0.4, 0.6, "wAGN")
        plt.text(0.4, 0, "RG")
        plt.text(-0.3, -0.55, "Passive")
    
    def whan_config(self, titulo="Diagrama WHAN", sfx=''):
        self.plot_KeKa06() # linhas de separação teóricas
        plt.title(titulo, fontsize='x-large')
        plt.xlabel(r"$log_{10}$(%s[NII] / %sH$\alpha$)"%(sfx, sfx), fontsize='large')
        plt.ylabel(r"$log_{10}$EW H$\alpha$)", fontsize='large')
        plt.xticks(fontsize='medium')
        plt.yticks(fontsize='medium')
        plt.xlim(-2, 1)
        plt.ylim(-0.8, 3)
        plt.grid(True, alpha=0.2)
        plt.legend(loc='upper right', fontsize='small')
        plt.tight_layout()
        plt.show()
        
    def curvas_densidade(self, dados_x, dados_y, levels=True):
        if levels:
            X, Y, Z, lvls = self.get_levels(dados_x, dados_y)
            #print(lvls)
        else:
            X, Y, Z, _ = self.get_densities(dados_x, dados_y)
            lvls = [0.24204035, 0.46083535, 1.138656, 1.85108968] # bpt dados reais
        plt.contour(X, Y, Z, colors='purple', alpha=0.4, levels=lvls)
        #sns.kdeplot(x=dados_x, y=dados_y, cmap=cor, fill=False, levels=(0.1, 0.2, 0.4, 0.6))
        plt.plot([], [], '-', color='purple', alpha=0.4, linewidth=1, label='Níveis de densidade numérica')

    def bpt_config(self, title='Diagrama BPT', sfx=''):
        self.plot_KeKa() # Linhas de separação teóricas
        plt.xlabel(r"$log_{10}$(%s[NII] / %sH$\alpha$)"%(sfx, sfx), fontsize='large')
        plt.ylabel(r"$log_{10}$(%s[OIII] / %sH$\beta$)"%(sfx, sfx), fontsize='large')
        plt.title(title)
        plt.xlim(-2, 1)
        plt.ylim(-1.5, 1.4)
        plt.grid(True, alpha=0.2)
        plt.legend(loc='lower left', fontsize='small')
        plt.tight_layout()
        plt.show()

    def bpt_pontos_reg(self, col_x, dados, X_, bptx, bpty, 
                       complexity=[], titulo="Dados & Operon + Quantílica"):
        # Todos os dados da síntese
        X_sin = dados[col_x]
        sintx = dados[T.nii_ha.value]
        sinty = dados[T.oiii_hb.value]
        
        titulo = "%s ~%s"%(titulo, col_x)
        if len(complexity) > 0:
            label = "Operon(%s)(%d-%d-%d-%d)"%(
                       col_x, 
                       complexity[0], complexity[1],
                       complexity[2], complexity[3])
        else:
            label = "Medianas ~%s"%col_x
        
        plt.subplots(figsize=(12,8))
        self.plot_KeKa()
        xy = plt.scatter(x=sintx, y=sinty, c=X_sin, cmap='viridis', s=2.5)
        plt.colorbar(xy).set_label(self.unidades[col_x], fontsize='xx-large')
        plt.scatter(x=bptx, y=bpty, color='black', s=3, label=label)
        
        plt.title("Diagrama BPT: %s"%titulo, fontsize='xx-large')
        plt.xlabel(r"$log_{10}$(NII / H$\alpha$)", fontsize='xx-large')
        plt.ylabel(r"$log_{10}$(OIII / H$\beta$)", fontsize='xx-large')
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlim(-2.5, 1)
        plt.ylim(-1.5, 1.5)
        plt.legend(fontsize=16, markerscale=4)
        plt.tight_layout()
        plt.show()
    
    def bpt_medianas_reg(self, col_x, dados, M, X_, bptx, bpty, 
                         complexity=[], titulo="Medianas & Operon + Quantílica"):
        
        medians_nii = self.bins_and_medians(dados[col_x], dados[T.nii.value], M)
        medians_ha = self.bins_and_medians(dados[col_x], dados[T.ha.value], M)
        medians_oiii = self.bins_and_medians(dados[col_x], dados[T.oiii.value], M)
        medians_hb = self.bins_and_medians(dados[col_x], dados[T.hb.value], M)

        X_sin = medians_nii["center"]
        sintx = medians_nii["median"] - medians_ha["median"]
        sinty = medians_oiii["median"] - medians_hb["median"]
        
        titulo = "%s ~%s"%(titulo, col_x)
        if len(complexity) > 0:
            label = "Operon(%s)(%d-%d-%d-%d)"%(
                       col_x, 
                       complexity[0], complexity[1],
                       complexity[2], complexity[3])
        else:
            label = "Medianas ~%s"%col_x
        
        plt.subplots(figsize=(12,8))
        self.plot_KeKa()
        xy = plt.scatter(x=sintx, y=sinty, c=X_sin, cmap='viridis', s=2.5)
        plt.colorbar(xy).set_label(self.unidades[col_x], fontsize='xx-large')
        plt.scatter(x=bptx, y=bpty, color='black', s=3, label=label)
        
        plt.title("Diagrama BPT: %s"%titulo, fontsize='xx-large')
        plt.xlabel(r"$log_{10}$(NII / H$\alpha$)", fontsize='xx-large')
        plt.ylabel(r"$log_{10}$(OIII / H$\beta$)", fontsize='xx-large')
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlim(-2.5, 1)
        plt.ylim(-1.5, 1.5)
        plt.legend(fontsize=16, markerscale=4)
        plt.tight_layout()
        plt.show()    
    

    def bpt_mediana_ind(self, dados, col_x, complexity, filtro=(-100,100), save=False):
        # Síntese
        _, _, _, sintx_test = self.split_dados(dados, T.nii_ha.value, col_x)
        _, _, _, sinty_test = self.split_dados(dados, T.oiii_hb.value, col_x)
        
        # Funções dos algoritmos
        scores_nii,_,operon_f_nii = self.get_operon_func(T.nii.value, col_x, complexity[0])
        scores_ha,_,operon_f_ha = self.get_operon_func(T.ha.value, col_x, complexity[1])
        scores_oiii,_,operon_f_oiii = self.get_operon_func(T.oiii.value, col_x, complexity[2])
        scores_hb,_,operon_f_hb = self.get_operon_func(T.hb.value, col_x, complexity[3])
        
        ajuste = (dados[col_x] >= filtro[0]) & (dados[col_x] <= filtro[1])
        linear = ~(dados[col_x] > filtro[0]) & (dados[col_x] < filtro[1])
        
        xnum = operon_f_nii(dados[col_x])
        xden = operon_f_ha(dados[col_x])
        ynum = operon_f_oiii(dados[col_x][ajuste])
        yden = operon_f_hb(dados[col_x])
        
        if (len(dados[linear]) > 0):
            X = dados[col_x][linear].copy()
            reta = self.reg_linear_last_y_fixed(X, operon_f_oiii(X), operon_f_oiii)
            Rynum = reta.A*X + reta.B
            ynum = np.concatenate((ynum, Rynum))
        
        bptx_test = xnum - xden
        bpty_test = ynum - yden

        self.plot_bpt("Medianas estimadas pelo Operon", 
                   sintx_test, sinty_test, bptx_test, bpty_test, 
                   label="Operon(%s)(%d-%d-%d-%d)"%(
                       col_x, 
                       scores_nii["complexity"],
                       scores_ha["complexity"],
                       scores_oiii["complexity"],
                       scores_hb["complexity"]),
                   save=save)
        
    def bpt_mediana_sym(self, dados, col_x, complexity, save=True):
        # Funções dos algoritmos
        scores_nii,_,operon_f_nii = self.get_operon_func(T.nii.value, col_x, complexity[0])
        scores_ha,_,operon_f_ha = self.get_operon_func(T.ha.value, col_x, complexity[1])
        scores_oiii,_,operon_f_oiii = self.get_operon_func(T.oiii.value, col_x, complexity[2])
        scores_hb,_,operon_f_hb = self.get_operon_func(T.hb.value, col_x, complexity[3])
        
        xnum = operon_f_nii(dados[col_x])
        xden = operon_f_ha(dados[col_x])
        ynum = operon_f_oiii(dados[col_x])
        yden = operon_f_hb(dados[col_x])
        bptx = xnum - xden
        bpty = ynum - yden
        
        # Dados originais e estatísticas
        _, _, _, sintx = self.split_dados(dados, T.nii_ha.value, col_x)
        _, _, _, sinty = self.split_dados(dados, T.oiii_hb.value, col_x)
        #mae = np.mean(np.abs(bptx - sintx) + np.abs(bpty - sinty))
        
        medians = self.bins_and_medians(sintx, sinty, 100)
        mediansx = medians['center']
        mediansy = medians['median']
        qmin = medians["qmin"]
        qmax = medians["qmax"]

        self.plot_bpt("Medianas estimadas pelo Operon", 
                   mediansx, mediansy, bptx, bpty, qmin, qmax,
                   label="Operon(%s)(%d-%d-%d-%d)"%(
                       col_x, 
                       scores_nii["complexity"],
                       scores_ha["complexity"],
                       scores_oiii["complexity"],
                       scores_hb["complexity"]), 
                   tipo="symb", save=save)
        
    def bpt_amostras_normais(self, dados, amostras):
        X_  = []
        for x in amostras[0]:
            for _ in range(100):
                X_.append(x)

        X_bpt = amostras[1] - amostras[2]
        Y_bpt = amostras[3] - amostras[4]
        X_sint = dados[T.nii.value] - dados[T.ha.value]
        Y_sint = dados[T.oiii.value] - dados[T.hb.value]

        plt.subplots(figsize=(10,6))
        self.plot_KeKa()
        plt.scatter(x=X_sint, y=Y_sint, color='gray', alpha=0.2, s=0.25)
        xy = plt.scatter(x=X_bpt, y=Y_bpt, c=X_, cmap='rainbow', alpha=0.6, s=0.2)
        plt.colorbar(xy, label="atflux")
        plt.title('BPT of Symbolic Regression and Normal deviation')
        plt.xlim(-2.5, 1)
        plt.ylim(-1.5, 1.5)
        plt.xlabel(r"$log_{10}$(NII / H$\alpha$)", fontsize=18)
        plt.ylabel(r"$log_{10}$(OIII / H$\beta$)", fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(markerscale=8)
        plt.tight_layout()
        plt.legend()
        plt.show()


if __name__ == "__main__":
    m = PlotsMetricas()
    
    # Gerar gráficos das métricas -- INICIO
    #p.parte_ks()
    #p.parte_combs()
    #p.parte_bootstrap()
    # Gerar gráficos das métricas -- FIM
    
    ### Configurar o Plotly -- INICIO
    df = px.data.iris()
    fig = px.scatter(df, x="sepal_width", y="sepal_length")
    fig.show()
    ### Configurar o Plotly -- FIM
