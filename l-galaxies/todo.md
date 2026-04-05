# Lista de Afazeres: L-Galaxies
## Download, Instalação, Execução e Obtenção de Resultados

---

## PARTE 1 — Onde baixar o L-Galaxies

### 1.1 Escolher a versão adequada
- [ ] Decidir qual versão usar. Para o seu projeto (implementar modelos de linhas de emissão), a versão **Henriques et al. (2015)** é a mais indicada para começar, pois é mais simples e tem documentação completa. A versão **Henriques et al. (2020)** adiciona discos resolvidos radialmente e enriquecimento químico detalhado, mas a documentação assume familiaridade com a versão 2015.
- [ ] A versão 2020 tem duas *branches* no GitHub: **"master"** (Yates+21a) e **"Yates2023"** (com evolução estelar binária e poeira).

### 1.2 Baixar o código-fonte
- [ ] Acessar o repositório GitHub:
  - **Henriques 2015:** https://github.com/LGalaxiesPublicRelease/LGalaxies_PublicRepository
  - **Henriques 2020:** https://github.com/LGalaxiesPublicRelease/LGalaxies2020_PublicRepository
- [ ] Clonar via git (recomendado):
  ```bash
  git clone --branch master --single-branch \
    https://github.com/LGalaxiesPublicRelease/LGalaxies2020_PublicRepository
  ```
  Ou baixar o ZIP pelo navegador e descompactar.
- [ ] Criar a pasta de saída dentro do diretório-raiz do projeto:
  ```bash
  cd LGalaxies2020_PublicRepository
  mkdir output
  ```

### 1.3 Baixar as tabelas espectrofotométricas (~1.6 GB)
- [ ] Baixar de: https://uhhpc.herts.ac.uk/~ry22aas/SpecPhotTables.tar
  ```bash
  wget -P ./ https://uhhpc.herts.ac.uk/~ry22aas/SpecPhotTables.tar
  ```
- [ ] Mover para o diretório-raiz do projeto e descompactar:
  ```bash
  tar -xf ./SpecPhotTables.tar
  ```

### 1.4 Baixar os merger trees (árvores de fusão do Millennium)
- [ ] Escolher o conjunto de treefiles:
  - **Mínimo para teste (Millennium-I):** Treefile 5 (~804 MB)
    https://uhhpc.herts.ac.uk/~ry22aas/MR_MergerTrees_5.tar
  - **Teste estatístico (Millennium-I):** Treefiles 0-9 (~7.5 GB)
    https://uhhpc.herts.ac.uk/~ry22aas/MR_MergerTrees_0-9.tar
  - **Volume completo (Millennium-I):** Treefiles 0-511 (~237 GB comprimido, ~401 GB descomprimido)
    https://uhhpc.herts.ac.uk/~ry22aas/MR_allMergerTrees.tar
- [ ] Baixar e mover para a pasta `/MergerTrees/` dentro do diretório-raiz:
  ```bash
  wget https://uhhpc.herts.ac.uk/~ry22aas/MR_MergerTrees_5.tar
  mv MR_MergerTrees_5.tar ./MergerTrees/
  cd MergerTrees
  tar -xzf MR_MergerTrees_5.tar
  ```

### 1.5 Documentação e referências
- [ ] Ler a documentação oficial: https://lgalaxiespublicrelease.github.io/
- [ ] Consultar a página de instalação Linux: https://lgalaxiespublicrelease.github.io/codeInstall_linux.html
- [ ] Artigos de referência: Henriques et al. (2015, MNRAS, 451, 2663) e Henriques et al. (2020)

---

## PARTE 2 — Como compilar e rodar o L-Galaxies

### 2.1 Verificar pré-requisitos do sistema
- [ ] Verificar se o compilador C (gcc) está instalado:
  ```bash
  which gcc
  ```
- [ ] Verificar se o `make` está instalado:
  ```bash
  which make
  ```
- [ ] Verificar se a GSL (GNU Scientific Library) está instalada. Os headers ficam tipicamente em `/usr/local/include/gsl` e as bibliotecas em `/usr/local/lib/`.
- [ ] Se faltar algo, instalar com:
  ```bash
  sudo apt update
  sudo apt-get install build-essential
  sudo apt-get install libgsl-dev libgsl-dbg
  ```
- [ ] Se estiver em um cluster com módulos, carregar o GSL:
  ```bash
  module load gsl
  ```

### 2.2 Configurar o compilador
- [ ] Abrir o arquivo `My_Makefile_compilers` no diretório-raiz do projeto.
- [ ] Confirmar que `SYSTYPE = "Linux"` está ativo (descomentar se necessário).
- [ ] Verificar que os caminhos `GSL_INCL` e `GSL_LIBS` apontam para o lugar certo. Se a GSL foi instalada via apt, pode ser necessário ajustar para `/usr/include` e `/usr/lib` em vez de `/usr/local/...`.

### 2.3 Configurar as opções do modelo
- [ ] Abrir o arquivo `My_Makefile_options` no diretório-raiz.
- [ ] Para rodar a versão Henriques2015 padrão, usar as opções default.
- [ ] Para a versão Henriques2020, ativar `H2_AND_RINGS` e `DETAILED_METALS_AND_MASS_RETURN`. Um exemplo está em `./Makefile_options/Makefile_options_Henriques20`.
- [ ] Ajustar `NOUT` para o número de redshifts de saída desejados.

### 2.4 Compilar o código
- [ ] No terminal, navegar até o diretório-raiz e compilar:
  ```bash
  make
  ```
- [ ] Se der erro, consultar a seção de "Common compiler issues" na documentação. Problemas comuns incluem: necessidade de adicionar `-std=C99`, `-fcommon`, `-fPIC`, ou trocar `-static` por `-shared` no `My_Makefile_compilers`.

### 2.5 Rodar o código
- [ ] Executar com o arquivo de entrada desejado:
  ```bash
  ./L-Galaxies ./input/input_MR_W1_PLANCK_LGals2020_DM.par
  ```
- [ ] Ajustar `FirstFile` e `LastFile` no arquivo `.par` para controlar quantos treefiles processar (treefile 5 sozinho já é suficiente para teste).
- [ ] Os arquivos de saída serão gravados na pasta `/output/`.

### 2.6 Integrar seus modelos de regressão simbólica
- [ ] Esta é a etapa principal do seu projeto (meses 21-24). Para implementar as equações de largura equivalente no L-Galaxies, você precisará:
  1. Identificar no código-fonte onde as propriedades das galáxias (idade ponderada por fluxo, metalicidade ponderada por massa, massa estelar) são calculadas ou armazenadas.
  2. Adicionar novas variáveis de saída para as larguras equivalentes (Hβ, [OIII]λ5007, Hα, [NII]λ6584) no struct de saída em `h_galaxy_output.h`.
  3. Implementar suas equações de regressão simbólica (Operon) em um novo arquivo `.c` ou dentro de `model_misc.c` / `post_process_spec_mags.c`.
  4. Atualizar `save.c` para transferir os valores calculados para o struct de saída.
  5. Atualizar o arquivo de estrutura Python (`LGalaxy_snapshots.py`) para incluir as novas propriedades.
- [ ] Coordenar com o Pablo Araya-Araya, que é especialista no L-Galaxies, conforme previsto no projeto.

---

## PARTE 3 — Obter resultados apresentáveis

### 3.1 Ler os outputs em Python
- [ ] Usar os scripts Python que vêm com o L-Galaxies, localizados em `/AuxCode/Python/`.
- [ ] Executar o script principal:
  ```bash
  python3 AuxCode/Python/main_lgals.py
  ```
- [ ] Este script já gera amostras de galáxias (salvas como `.npy` em `/output/samples/`) e produz figuras padrão (função de massa estelar, relação massa-metalicidade, etc.) na pasta `/figures/`.

### 3.2 Configurar o script de leitura
- [ ] No `main_lgals.py`, ajustar as variáveis em "USER DEFINED INPUTS":
  - `COSMOLOGY`: 'Planck'
  - `SIMULATION`: 'Mil-I' ou 'Mil-II'
  - `FILE_TYPE`: 'snapshots' (default) ou 'galtree'
  - `REDSHIFT`: o redshift do output desejado (ex.: 0.00 para z=0)
  - `SAMPLE_TYPE`: 'All', 'Discs', 'ETGs', ou 'Dwarfs'
  - `STRUCT_TYPE`: apontar para a estrutura de saída correta

### 3.3 Instalar dependências para os plots
- [ ] Instalar LaTeX para labels nos gráficos:
  ```bash
  sudo apt-get install texlive dvipng texlive-latex-extra texlive-fonts-recommended cm-super
  ```
- [ ] Instalar bibliotecas Python necessárias (numpy, matplotlib, scipy, etc.).

### 3.4 Produzir diagramas BPT com os dados do L-Galaxies
- [ ] Após implementar suas equações, extrair as larguras equivalentes preditas dos outputs do L-Galaxies.
- [ ] Calcular as razões de linhas: log([OIII]λ5007/Hβ) vs. log([NII]λ6584/Hα).
- [ ] Plotar o diagrama BPT com as galáxias do mock e comparar com os dados observacionais do SDSS (Werle et al. 2019).
- [ ] Sobrepor as curvas de demarcação de Kauffmann et al. (2003) e Kewley et al. (2006).

### 3.5 Validação e comparação
- [ ] Comparar as distribuições das larguras equivalentes preditas pelo L-Galaxies com as distribuições observadas.
- [ ] Verificar se as galáxias caem nas regiões corretas do diagrama BPT (star-forming, Seyfert, LINER, etc.).
- [ ] Avaliar a dispersão dos modelos usando a abordagem paramétrica (normal multivariada) que você já desenvolveu.
- [ ] Produzir figuras para a dissertação mostrando: histogramas de EW, diagramas BPT, comparação mock vs. observações.

### 3.6 Gerar catálogos mock finais
- [ ] Uma vez validados os modelos, rodar o L-Galaxies no volume completo do Millennium (treefiles 0-511) para gerar um catálogo mock estatisticamente significativo.
- [ ] Salvar os catálogos com as propriedades das galáxias + larguras equivalentes preditas em formato acessível (HDF5 ou FITS).

---

## Resumo do fluxo de trabalho

```
1. Download (código + tabelas + treefiles)
       ↓
2. Instalar pré-requisitos (gcc, make, GSL)
       ↓
3. Configurar (My_Makefile_compilers, My_Makefile_options)
       ↓
4. Compilar (make) e testar (treefile 5)
       ↓
5. Implementar equações de EW no código-fonte
       ↓
6. Recompilar e rodar com as modificações
       ↓
7. Ler outputs com Python e gerar diagramas BPT
       ↓
8. Validar contra dados SDSS e produzir figuras para a dissertação
```

---

*Referências úteis:*
- Site oficial: https://lgalaxiespublicrelease.github.io/
- GitHub (2020): https://github.com/LGalaxiesPublicRelease/LGalaxies2020_PublicRepository
- GitHub (2015): https://github.com/LGalaxiesPublicRelease/LGalaxies_PublicRepository
- Instalação Linux: https://lgalaxiespublicrelease.github.io/codeInstall_linux.html
- Contato do mantenedor: Rob Yates (r.yates3@herts.ac.uk)
- Modelo Henriques2020: https://lgalaxiespublicrelease.github.io/Hen20_index.html
