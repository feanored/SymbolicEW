# Use a imagem base oficial do Miniconda
FROM continuumio/miniconda3:latest

# Atualiza o sistema e instala dependências básicas
RUN apt-get update && apt-get install -y \
    libnss3 \
    libatk-bridge2.0-0 \
    libcups2 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libxkbcommon0 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    build-essential \
    xvfb \
    libgtk-3-0 \
    libxss1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Cria e ativa um ambiente Conda chamado "pyoperon_env"
RUN conda create -n pyoperon_env python=3.12 -y

# Instala o pyoperon e JupyterLab no ambiente conda
RUN conda run -n pyoperon_env pip install numpy matplotlib scikit-learn pandas tqdm pyoperon==0.5.0 pysr==1.5.9 jupyterlab ipywidgets plotly spyder-kernels

# Adiciona o comando para ativar o ambiente ao bashrc
RUN echo "conda activate pyoperon_env" >> ~/.bashrc

# Define o ambiente ativo ao iniciar o container
ENV PATH=/opt/conda/envs/pyoperon_env/bin:$PATH

# Expõe a porta para o JupyterLab
EXPOSE 8888

RUN jupyter server --generate-config

# Configura onetime token
ARG TOKEN=capeta
RUN mkdir -p $HOME/.jupyter/
RUN if [ $TOKEN!=-1 ]; then echo "c.NotebookApp.token='$TOKEN'" >> $HOME/.jupyter/jupyter_notebook_config.py; fi

# Define o diretório de trabalho no container
WORKDIR /workspace

# Comando padrão para iniciar o JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]