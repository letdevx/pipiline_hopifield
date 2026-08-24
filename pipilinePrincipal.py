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
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Pipeline Principal — Hopifield
# Executa as etapas de pré-processamento e análise da matriz de expressão.

# %% [markdown]
# ## 1. Binarização

# %%
import sys, os, importlib

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath("__file__")), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import config, preprocessing
importlib.reload(config)
importlib.reload(preprocessing)
from config import PATH_ALVO, PATH_REFERENCIA, OUT_BINARIZACAO
from preprocessing import Binarizador

# --- Binarização dos dois datasets ---
binarizador_m = Binarizador(path_h5ad=PATH_ALVO, out_dir=OUT_BINARIZACAO)
binarizador_f = Binarizador(path_h5ad=PATH_REFERENCIA, out_dir=OUT_BINARIZACAO)

binarizador_m.binarizar()
binarizador_f.binarizar()

print("Mathys binarizado em:", binarizador_m.path_binarizada)
print("Fujita binarizado em:", binarizador_f.path_binarizada)

# %% [markdown]
# ## 2. Alinhamento

# %%
import sys, os, importlib
import anndata as ad

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath("__file__")), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import config, alinhamento
importlib.reload(config)
importlib.reload(alinhamento)
from config import PATH_FEATURES_REFERENCIA, PATH_FEATURES_ALVO, PATH_TOP5000, OUT_ALINHAMENTO, OUT_TOP_GENES
from alinhamento import (LeitorFeatures, AnalisadorSobreposicao, Alinhador,
                          ValidadorAlinhamento, AnalisadorCobertura,
                          SelecionadorGenesFrequentes)

# %%
# Passo 1 — Leitura dos arquivos de features
leitor = LeitorFeatures(PATH_FEATURES_REFERENCIA, PATH_FEATURES_ALVO)
leitor.ler()
print(leitor)

# %%
# Passo 2 — Análise de sobreposição
from config import PATH_REFERENCIA

# var_names são idênticos no original e no binarizado — lemos direto do original
_f = ad.read_h5ad(PATH_REFERENCIA, backed='r')
var_names_f_original = _f.var_names.tolist()
_f.file.close()
del _f

analisador = AnalisadorSobreposicao(leitor.map_f, leitor.map_m, var_names_f_original)
analisador.analisar()
print(analisador)

# %%
# Passo 3 — Alinhamento dos dois h5ad binarizados
alinhador = Alinhador(
    path_binarizada_m = binarizador_m.path_binarizada,
    path_binarizada_f = binarizador_f.path_binarizada,
    out_dir           = OUT_ALINHAMENTO,
    map_f             = leitor.map_f,
    map_m             = leitor.map_m,
    gene_alvo_idx     = analisador.gene_alvo_idx,
    genes_ordenados   = analisador.genes_ordenados,
)
alinhador.alinhar()
alinhador.salvar_como_txt()
alinhador.gerar_tracking(analisador.ids_so_f, leitor.map_f)
print(alinhador)

# %%
# Passo 4 — Validação da ordem de genes
validador = ValidadorAlinhamento(
    path_f_alinhado = alinhador.path_f_alinhado,
    path_m_alinhado = alinhador.path_m_alinhado,
    genes_ordenados = analisador.genes_ordenados,
)
validador.validar()

# %%
# Passo 5 — Cobertura dos top-5000 genes frequentes do Fujita no Mathys
cobertura = AnalisadorCobertura(PATH_TOP5000, leitor.map_f, leitor.map_m)
cobertura.analisar(out_csv=os.path.join(OUT_ALINHAMENTO, "top5000_cobertura_mathys.csv"))

# %% [markdown]
# ## 3. Preparação do conjunto de treinamento

# %%
import sys, os, importlib

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath("__file__")), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import config, alinhamento, treinamento

# Recarrega submódulos explicitamente antes dos pacotes
import alinhamento.selecionador_genes_frequentes as _sgf_mod
import treinamento.gerador_conjunto_treinamento  as _gct_mod
import treinamento.projetor_sweep               as _ps_mod
importlib.reload(_sgf_mod)
importlib.reload(_gct_mod)
importlib.reload(_ps_mod)
importlib.reload(config)
importlib.reload(alinhamento)
importlib.reload(treinamento)

from config import OUT_TOP_GENES, OUT_TREINAMENTO, PATH_SWEEP_REFERENCIA
from alinhamento import SelecionadorGenesFrequentes
from treinamento import GeradorConjuntoTreinamento, ProjetorSWeePR

path_top5k    = os.path.join(OUT_TOP_GENES,   "top5000_frequentes.csv")
path_f_top5k  = os.path.join(OUT_TREINAMENTO, "adataF_binarizado_alinhado_top5000.txt")
path_m_top5k  = os.path.join(OUT_TREINAMENTO, "adataM_binarizado_alinhado_top5000.txt")

# — Passo 3a: Top 5000 genes mais frequentes do Fujita —
selecionador = SelecionadorGenesFrequentes(
    path_txt = alinhador.path_f_alinhado.replace('.h5ad', '.txt'),
    n        = 5000,
)
selecionador.calcular(out_csv=path_top5k).salvar(path_top5k)

# — Passo 3b: Conjuntos de treinamento filtrados (Fujita + Mathys) —
gerador = GeradorConjuntoTreinamento(
    path_top_genes_csv = path_top5k,
    out_dir            = OUT_TREINAMENTO,
)
gerador.gerar(alinhador.path_f_alinhado.replace('.h5ad', '.txt'))
gerador.gerar(alinhador.path_m_alinhado.replace('.h5ad', '.txt'))

# — Passo 3c: Projeção SWeeP via R (células × 600 dim) —
projetor_r = ProjetorSWeePR(
    path_matriz   = path_f_top5k,
    path_saida    = PATH_SWEEP_REFERENCIA,
    n_componentes = 600,
    seed          = 42,
)
projetor_r.projetar()

print("\n=== Arquivos de treinamento prontos ===")
for label, path in [
    ("top5000_frequentes.csv            ", path_top5k),
    ("adataF_binarizado_alinhado_top5000", path_f_top5k),
    ("adataM_binarizado_alinhado_top5000", path_m_top5k),
    ("matriz_reduzida_sweepF.csv        ", PATH_SWEEP_REFERENCIA),
]:
    status = "✓" if os.path.exists(path) else "✗ NÃO ENCONTRADO"
    print(f"  {label} : {status}")

# %% [markdown]
# ## 4. Análise Hopfield (rede35)

# %%
import sys, os, importlib
import numpy as np
import pandas as pd

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath("__file__")), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import config, treinamento
import treinamento.carregador_dados_fujita as _cdf_mod
importlib.reload(_cdf_mod)
importlib.reload(config)
importlib.reload(treinamento)

from config import PATH_SWEEP_REFERENCIA, PATH_LABELS_REFERENCIA, PATH_LABELS_ALVO, OUT_TREINAMENTO, OUT_TOP_GENES
from treinamento import CarregadorDadosFujita

# Passo 4a — Carregamento dos dados Fujita (padrões para treino da rede)
path_matriz_f = os.path.join(OUT_TREINAMENTO, "adataF_binarizado_alinhado_top5000.txt")
path_genes    = os.path.join(OUT_TOP_GENES,   "top5000_frequentes.csv")

carregador = CarregadorDadosFujita(
    path_matriz = path_matriz_f,
    path_genes  = path_genes,
    path_labels = PATH_LABELS_REFERENCIA,
    path_sweep  = PATH_SWEEP_REFERENCIA,
    n_genes     = 5000,
)
carregador.carregar()
print(carregador)

# Passo 4a.2 — Carregamento da matriz Mathys (usada na reconstrução)
path_matriz_m = os.path.join(OUT_TREINAMENTO, "adataM_binarizado_alinhado_top5000.txt")

print("[Mathys] Carregando matriz top5000...")
W_mathys = pd.read_csv(path_matriz_m).to_numpy(dtype=np.float32)
print(f"[Mathys] W_mathys shape: {W_mathys.shape}")

print("[Mathys] Carregando rótulos binários...")
labels_mathys = np.loadtxt(PATH_LABELS_ALVO, dtype=int)
print(f"[Mathys] labels shape: {labels_mathys.shape}, tipos: {np.unique(labels_mathys)}")

# %%
from treinamento import ProjetorSWeP

# Passo 4b — Aplicar PCA sem centralização sobre as projeções SWeeP
projetor = ProjetorSWeP(n_features=5000)
projetor.usar_sweep_precomputado(carregador.Wswp).aplicar_pca()
print(projetor)

# %%
from treinamento import ExtratorPadroesSubcluster

# Passo 4c — Extração de padrões por subcluster (6 classes × 10 representantes = 60 padrões)
# Classes 2 (Endothelial) e 8 (Pericytes) excluídas por baixo número de células
extrator = ExtratorPadroesSubcluster(
    W0      = carregador.W0,
    labels  = carregador.labels,
    classes = [1, 3, 4, 5, 6, 7],
    nc      = 210,
    seed    = 42,
)
extrator.extrair(projetor.Wswp)
print(extrator)

# %%
import sys, os, importlib

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath("__file__")), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import config, treinamento
import treinamento.hopfield as _hopfield_mod
importlib.reload(_hopfield_mod)
importlib.reload(config)
importlib.reload(treinamento)

from config import OUT_HOPFIELD
from treinamento import ModernHopfieldNetwork

# Passo 4d — Treino com padrões Fujita + reconstrução com Mathys
# threshold=0.5: genes ausentes no Mathys (preenchidos com 0.5 pelo alinhador)
# tratados como ausentes (0) na binarização da query
rede35 = ModernHopfieldNetwork(beta=10, n_iters=1, binary=True, threshold=0.5)
rede35.store(extrator.padroes)   # treino: padrões extraídos do Fujita

# Salva rede treinada (transferível entre máquinas)
path_rede = os.path.join(OUT_HOPFIELD, "rede35_treinada.pt")
rede35.salvar(path_rede)

# Recuperação usando a matriz do Mathys
Wrecuperado = rede35.retrieve(W_mathys, batch_size=4096)
print(rede35)

# %% [markdown]
# ### Carregamento da rede treinada (segunda máquina)
# Execute esta célula caso esteja continuando a análise a partir do arquivo `.pt` gerado na outra máquina.
# Não é necessário reexecutar o treinamento — basta carregar o arquivo e seguir para o passo 4e.

# %%
import sys, os, importlib
import numpy as np
import pandas as pd

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath("__file__")), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import config, treinamento
import treinamento.hopfield as _hopfield_mod
importlib.reload(_hopfield_mod)
importlib.reload(config)
importlib.reload(treinamento)

from config import OUT_HOPFIELD, OUT_TREINAMENTO, PATH_LABELS_M
from treinamento import ModernHopfieldNetwork

# Carrega rede treinada salva na outra máquina
path_rede = os.path.join(OUT_HOPFIELD, "rede35_treinada.pt")
rede35 = ModernHopfieldNetwork.carregar(path_rede)
print(rede35)

# Carrega matriz Mathys para recuperação
path_matriz_m = os.path.join(OUT_TREINAMENTO, "adataM_binarizado_alinhado_top5000.txt")

W_mathys = pd.read_csv(path_matriz_m).to_numpy(dtype=np.float32)
print(f"[Mathys] W_mathys shape: {W_mathys.shape}")

labels_mathys = np.loadtxt(PATH_LABELS_M, dtype=int)
print(f"[Mathys] labels shape: {labels_mathys.shape}, tipos: {np.unique(labels_mathys)}")

# Recuperação
Wrecuperado = rede35.retrieve(W_mathys, batch_size=4096)

# %%
import sys, os, importlib

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath("__file__")), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import treinamento
import treinamento.avaliador_hopfield as _av_mod
importlib.reload(_av_mod)
importlib.reload(treinamento)

from treinamento import AvaliadorHopfield

# Passo 4e — Avaliação: 6 classes (sem Endothelial=2 e sem Pericytes=8)
avaliador = AvaliadorHopfield(
    padroes = extrator.padroes,
    classes = [1, 3, 4, 5, 6, 7],
    nc      = 210,
)
avaliador.avaliar(Wrecuperado, labels_mathys).plotar()
print(avaliador)

# %% [markdown]
# ## 5. Testes binário puro (sem 0.5)

# %%
import sys, os, importlib
import numpy as np
import pandas as pd

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath("__file__")), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import config
importlib.reload(config)
from config import OUT_TREINAMENTO, PATH_LABELS_M

# Passo 5a — Carregar Mathys top5000 e converter 0.5 → 0 (binário puro)
# O arquivo original não é modificado; a conversão ocorre apenas em memória.
path_matriz_m = os.path.join(OUT_TREINAMENTO, "adataM_binarizado_alinhado_top5000.txt")

W_mathys_bin = pd.read_csv(path_matriz_m).to_numpy(dtype=np.float32)
n_meio = int((W_mathys_bin == 0.5).sum())
W_mathys_bin[W_mathys_bin == 0.5] = 0.0

print(f"[Mathys binário] shape: {W_mathys_bin.shape}")
print(f"[Mathys binário] valores convertidos de 0.5 → 0: {n_meio}")
print(f"[Mathys binário] valores únicos: {np.unique(W_mathys_bin)}")

labels_mathys_bin = np.loadtxt(PATH_LABELS_M, dtype=int)
print(f"[Mathys binário] labels shape: {labels_mathys_bin.shape}, tipos: {np.unique(labels_mathys_bin)}")

# %%
import sys, os, importlib

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath("__file__")), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import treinamento
import treinamento.avaliador_hopfield as _av_mod
importlib.reload(_av_mod)
importlib.reload(treinamento)
from treinamento import AvaliadorHopfield

# Passo 5b — Auto-imputação: rede treinada em Fujita recebe as próprias células Fujita
# Baseline interno — esperamos alta taxa de reconstrução e classificação.
print("=== Auto-imputação: Fujita → Fujita ===")
Wrecuperado_fujita = rede35.retrieve(carregador.W0, batch_size=4096)

avaliador_fujita = AvaliadorHopfield(
    padroes = extrator.padroes,
    classes = [1, 3, 4, 5, 6, 7],
    nc      = 210,
)
avaliador_fujita.avaliar(Wrecuperado_fujita, carregador.labels).plotar()
print(avaliador_fujita)

# %%
# Passo 5c — Imputação Mathys binário puro (0/1 apenas, sem sentinela 0.5)
# Comparar acuracia/f1/taxa_reconstrucao com Seção 4e (Mathys com 0.5).
print("=== Imputação Mathys: binário puro (sem 0.5) ===")
Wrecuperado_mathys_bin = rede35.retrieve(W_mathys_bin, batch_size=4096)

avaliador_mathys_bin = AvaliadorHopfield(
    padroes = extrator.padroes,
    classes = [1, 3, 4, 5, 6, 7],
    nc      = 210
)
avaliador_mathys_bin.avaliar(Wrecuperado_mathys_bin, labels_mathys_bin).plotar()
print(avaliador_mathys_bin)
