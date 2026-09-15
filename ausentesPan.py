# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
#

# %%
from __future__ import annotations

import os
import sys

try:
    import anndata
    import scanpy
except ImportError:
    print("Instalando dependências compatíveis com o ambiente do Colab...")
    # Mantém o pandas travado na versão esperada pelo Colab (2.2.3)
    # !pip install -q "pandas==2.2.3" anndata scanpy

# %%
REPO_NAME = "pipiline_hopifield"
REPO_URL = "https://github.com/letdevx/pipiline_hopifield.git"
DEST_PATH = f"/content/{REPO_NAME}"

# Clona ou atualiza o repositório na VM do Colab
if os.path.exists("/content"):
    if not os.path.exists(DEST_PATH):
        print("Clonando código para a VM...")
        os.system(f"git clone {REPO_URL} {DEST_PATH}")
    else:
        print("Atualizando código na VM...")
        os.system(f"cd {DEST_PATH} && git pull")

    os.system(f"cd {DEST_PATH} && git checkout teste-pipeline_genereico_Pan_F")

# Adiciona a raiz do repo e a pasta 'src' ao sys.path
for _p in (DEST_PATH, os.path.join(DEST_PATH, "src")):
    if os.path.exists(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# %%
try:
    from google.colab import drive  # type: ignore

    if not os.path.exists("/content/drive"):
        print("[Colab] Montando Google Drive em /content/drive...")
        drive.mount("/content/drive")
except (ImportError, Exception):
    pass

# %%
import gc
import os
import sys
from pathlib import Path
import anndata as ad
import numpy as np
import polars as pl
import scipy.io as sio
import scipy.sparse as sp

# %%
# Resolução dinâmica e robusta do diretório raiz e de src/ (Colab e Local)
for _raiz in [
    Path.cwd(),
    Path.cwd().parent,
    Path("/content/pipiline_hopifield"),
    Path("/content/drive/Othercomputers/Meu laptop/Documents/Letworkspace/pipiline_hopifield"),
]:
    if (_raiz / "src").is_dir() and str(_raiz) not in sys.path:
        sys.path.insert(0, str(_raiz))
        sys.path.insert(0, str(_raiz / "src"))
        break

if os.path.exists("/content") and not os.path.exists("/content/pipiline_hopifield"):
    print("Clonando repositório na VM do Colab para carregar src...")
    os.system("git clone -b teste-pipeline_genereico_Pan_F https://github.com/letdevx/pipiline_hopifield.git /content/pipiline_hopifield")
    for _p in ("/content/pipiline_hopifield", "/content/pipiline_hopifield/src"):
        if _p not in sys.path:
            sys.path.insert(0, _p)

from src.config import PATH_REFERENCIA, PATH_BASE, PATH_ORTHBASE_RDS, OUTPUTS
from src.treinamento import ProjetorSWeePR

# 1. Definição e criação do diretório usando Pathlib
DIR_PROJECAO_AUSENTES = Path(OUTPUTS) / "projecao_ausentes_pan"
DIR_PROJECAO_AUSENTES.mkdir(parents=True, exist_ok=True)

# 2. Caminhos dos arquivos de entrada e saída
PATH_MTX_ENTRADA = Path(PATH_BASE) / "imputs" / "matrix_Pan.mtx"
PATH_SAIDA_TXT = DIR_PROJECAO_AUSENTES / "matriz_sweep_ausentes.txt"
PATH_SAIDA_NPY = DIR_PROJECAO_AUSENTES / "matriz_sweep_ausentes.npy"

print(f"Diretório de saída pronto: {DIR_PROJECAO_AUSENTES}")
print(f"OrthBase canônica configurada: {PATH_ORTHBASE_RDS}")
print(f"Caminho entrada MTX: {PATH_MTX_ENTRADA}")
print(f"Caminho saida TXT: {PATH_SAIDA_TXT}")
print(f"Caminho saida NPY: {PATH_SAIDA_NPY}")


# %%
# Resolução dinâmica dos arquivos de entrada (Colab Google Drive com fallback para local PATH_BASE)
caminho_tracking_colab = (
    r"/content/drive/Othercomputers/Meu laptop/Documents/Letworkspace/Teste hop"
    r"/outputsPan-->F/alinhamento/tracking_genes_adicionados_Fujita.csv"
)

tracking_pan_F = caminho_tracking_colab

caminho_pan_colab = (
    r"/content/drive/Othercomputers/Meu laptop/Documents/Letworkspace/Teste hop"
    r"/imputs/pan_anotado.h5ad"
)

matriz_pan = caminho_pan_colab

print(f"Tracking CSV : {tracking_pan_F} (Existe: {os.path.exists(tracking_pan_F)})")
print(f"Matriz Pan   : {matriz_pan} (Existe: {os.path.exists(matriz_pan)})")

# %%
posicao_genes_none_p_f = pl.read_csv(tracking_pan_F)
posicao_genes_none_p_f.head(5)

# %%
posicao_coluna_pan = posicao_genes_none_p_f["posicao_coluna"]
print(f"Total de genes ausentes mapeados: {posicao_coluna_pan.shape[0]}")

# %%
indice_pan = posicao_coluna_pan.to_list()
print(f"Primeiros 10 índices de colunas ausentes: {indice_pan[:10]}")

# %% [markdown]
# ### Injeção de Sentinela Neutro (0.5) nas Colunas Ausentes do Pan
# Injeta o valor sentinela neutro 0.5 (canônico do pipeline Hopfield) nas colunas de genes ausentes.
# A operação preserva o formato esparso da matriz AnnData para economizar memória RAM e evitar OOM.

# %%
print(f"Carregando matriz AnnData: {matriz_pan}...")
adata = ad.read_h5ad(matriz_pan)

# Garante manipulação eficiente de memória com LIL/CSR sem conversão densa
if sp.issparse(adata.X):
    X_mod = adata.X.tolil()
    X_mod[:, indice_pan] = 0.5
    X_mod = X_mod.tocsr()
else:
    X_mod = np.asarray(adata.X, dtype=np.float32).copy()
    X_mod[:, indice_pan] = 0.5
    X_mod = sp.csr_matrix(X_mod)

print(f"Matriz modificada: {X_mod.shape[0]} células × {X_mod.shape[1]} genes (nnz: {X_mod.nnz})")

# %% [markdown]
# ### Exportação Direta no Formato Matrix Market (.mtx)
# Salva a matriz esparsa modificada em disco via `scipy.io.mmwrite` de forma rápida e enxuta.

# %%
print(f"Exportando matriz para formato Matrix Market (.mtx): {PATH_MTX_ENTRADA}...")
sio.mmwrite(str(PATH_MTX_ENTRADA), X_mod)
print(f"Exportação MTX concluída com sucesso")

# Liberação preventiva de memória RAM
del adata, X_mod
gc.collect()

# %% [markdown]
# ### Projeção rSWeeP Canônica com Reuso da Base Congelada Padrão
# Executa a projeção rSWeeP oficial reutilizando a base ortonormal canônica congelada (`PATH_ORTHBASE_RDS`).
# O resultado compactado (600 dimensões) é persistido em `.txt` e `.npy`.

# %%
# Garante que dependências R estejam disponíveis caso executado no Colab
ProjetorSWeePR.verificar_e_instalar_dependencias_r()

# %%
if "imputs" not in PATH_MTX_ENTRADA.parts:
    raise ValueError(f"O caminho de entrada '{PATH_MTX_ENTRADA}' é inválido: a pasta 'imputs' não foi encontrada.")

print(f"[rSWeeP] Inicializando projetor oficial para {PATH_MTX_ENTRADA}...")
projetor = ProjetorSWeePR(
    path_matriz=str(PATH_MTX_ENTRADA),
    path_saida=str(PATH_SAIDA_TXT),
    n_componentes=600,
    seed=42,
    path_orthbase=str(PATH_ORTHBASE_RDS),
)

# Dispara o subprocesso oficial em R (orthBase + SWeeP)
projetor.projetar()

# Validações de integridade pós-projeção
assert projetor.Wswp is not None, "Erro: Matriz projetada retornou nula!"
assert not np.isnan(projetor.Wswp).any(), "Erro: Detectados valores NaN na projeção!"

# Persistência em formato binário NumPy (.npy)
np.save(str(PATH_SAIDA_NPY), projetor.Wswp)

print("\n=======================================================")
print(" [rSWeeP] Projeção de Ausentes Pan Concluída com Sucesso!")
print(f"  Shape final : {projetor.Wswp.shape} (células × 600 dimensões)")
print(f"  TXT salvo em: {PATH_SAIDA_TXT}")
print(f"  NPY salvo em: {PATH_SAIDA_NPY}")
print(f"  OrthBase    : {projetor.path_orthbase}")
print("=======================================================")
