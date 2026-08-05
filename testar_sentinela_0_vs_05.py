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
# # Teste Comparativo: Sentinela Neutra (0.5) vs. Absência Estrita (0.0)
#
# Este experimento avalia estatística e biologicamente o impacto de tratar genes ausentes no alinhamento cross-dataset (Mathys $\rightarrow$ Fujita) como:
# 1. **Sentinela Neutra (0.5):** Spin $0.0$ em $\{-1, +1\}$, resultando em peso nulo no produto escalar da atenção Softmax Hopfield.
# 2. **Absência Estrita (0.0):** Spin $-1.0$ em $\{-1, +1\}$, penalizando ativamente protótipos que expressam genes não detectados no Mathys.

# %% [markdown]
# ## 1. Configuração e Carregamento do Ambiente
#

# %%
import sys, os
import gc
import importlib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath('__file__')), 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import config
importlib.reload(config)
from config import (
    PATH_M, PATH_F, PATH_FEATURES_F, PATH_FEATURES_M,
    PATH_SWEEP_F, PATH_SWEEP_M, PATH_LABELS_F, PATH_LABELS_M,
    OUT_TOP_GENES, OUT_HOPFIELD, OUT_RELATORIO, OUT_ALINHAMENTO, OUT_TREINAMENTO
)

import treinamento.hopfield
import treinamento.extrator_padroes
import treinamento.avaliador_hopfield
importlib.reload(treinamento.hopfield)
importlib.reload(treinamento.extrator_padroes)
importlib.reload(treinamento.avaliador_hopfield)

from alinhamento import LeitorFeatures, AnalisadorSobreposicao, Alinhador, SelecionadorGenesDiferenciais
from treinamento import (
    CarregadorDadosFujita, ProjetorSWeePR,
    ExtratorPadroesSubcluster, ModernHopfieldNetwork, AvaliadorHopfield
)
from treinamento.extrator_padroes import EstrategiaKMeansDinamico

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

import polars as pl

print("=== 1. Carregando e Alinhando Datasets (Fujita & Mathys) ===")

path_f_mat = os.path.join(OUT_TREINAMENTO, 'adataF_binarizado_alinhado_expandido.npy')
path_m_mat = os.path.join(OUT_TREINAMENTO, 'adataM_binarizado_alinhado_expandido.npy')
path_top_csv = os.path.join(OUT_TOP_GENES, 'genes_expandidos_diferenciais_chi2.csv')
path_sweep_csv = os.path.join(OUT_TREINAMENTO, 'matriz_reduzida_sweepF_expandido.csv')

if os.path.exists(path_f_mat) and os.path.exists(path_top_csv) and os.path.exists(path_sweep_csv):
    df_genes = pl.read_csv(path_top_csv)
    n_genes = len(df_genes)
    carregador = CarregadorDadosFujita(
        path_matriz = path_f_mat,
        path_genes  = path_top_csv,
        path_labels = PATH_LABELS_F,
        path_sweep  = path_sweep_csv,
        n_genes     = n_genes,
    ).carregar()
    W0_f = carregador.W0
    clo_f = carregador.labels
    Wswp_f = carregador.Wswp
else:
    raise FileNotFoundError("Matrizes alinhadas do Fujita ausentes em outputs/treinamento/")

if os.path.exists(path_m_mat):
    W_mathys = np.load(path_m_mat).astype(np.float32, copy=False)
    clo_m = np.loadtxt(PATH_LABELS_M, dtype=int, skiprows=1)
    print(f"Dataset Mathys carregado: shape {W_mathys.shape}, clo shape {clo_m.shape}")
else:
    raise FileNotFoundError("Matrizes do Mathys não encontradas em outputs/treinamento/")

# %% [markdown]
# ## 2. Extração de Protótipos Consolidados ($k=3$, $nc=30$)
#

# %%
print("\n=== 2. Extraindo 210 Protótipos Consolidados (k=3 vizinhos) ===")
estr_kmeans = EstrategiaKMeansDinamico(k_range=[30], seed=SEED)
extrator = ExtratorPadroesSubcluster(
    estrategia=estr_kmeans,
    W0      = W0_f,
    labels  = clo_f,
    classes = [1, 2, 3, 4, 5, 6, 7],
    seed    = SEED,
    k       = 3,
    nc      = 30
)
extrator.extrair(Wswp_f)
perf35 = extrator.padroes
meta_eval = extrator.meta
print(f"Protótipos extraídos: perf35 shape {perf35.shape}")

# %% [markdown]
# ## 3. Construção da Rede Hopfield com Cosseno Harmonizado
#

# %%
print("\n=== 3. Inicializando ModernHopfieldNetwork (beta=15.0, threshold=0.0, normalize=True) ===")
rede35 = ModernHopfieldNetwork(beta=15.0, n_iters=1, binary=True, threshold=0.0, normalize=True)
rede35.store(perf35)

# %% [markdown]
# ## 4. Execução dos Cenários Comparativos
#

# %%
print("\n=== 4. Teste Cenário A: Sentinela Neutra (0.5) ===")
Wrec_a = rede35.retrieve(W_mathys, batch_size=1024, normalize=True)

avaliador_a = AvaliadorHopfield(
    padroes = perf35,
    classes = [1, 2, 3, 4, 5, 6, 7],
    nc      = 30,
    meta    = meta_eval,
    metrica = "cosseno"
)
avaliador_a.avaliar(Wrec_a, clo_m)

print("\n=== 4b. Teste Cenário B: Absência Estrita (0.5 -> 0.0) ===")
W_mathys_zero = np.where(W_mathys == 0.5, 0.0, W_mathys)
Wrec_b = rede35.retrieve(W_mathys_zero, batch_size=1024, normalize=True)

avaliador_b = AvaliadorHopfield(
    padroes = perf35,
    classes = [1, 2, 3, 4, 5, 6, 7],
    nc      = 30,
    meta    = meta_eval,
    metrica = "cosseno"
)
avaliador_b.avaliar(Wrec_b, clo_m)

# %% [markdown]
# ## 5. Quadro Comparativo e Visualização
#

# %%
df_comp = pd.DataFrame([
    {
        "Cenário": "A: Sentinela Neutra (0.5)",
        "Acurácia (%)": f"{avaliador_a.acuracia * 100:.2f}%",
        "F1 Weighted": f"{avaliador_a.f1_weighted:.4f}",
        "F1 Macro": f"{avaliador_a.f1_macro:.4f}",
        "Semelhança Média": f"{avaliador_a.semelhanca_media:.4f}",
        "Taxa Reconstrução": f"{avaliador_a.taxa_reconstrucao:.4f}"
    },
    {
        "Cenário": "B: Absência Estrita (0.0)",
        "Acurácia (%)": f"{avaliador_b.acuracia * 100:.2f}%",
        "F1 Weighted": f"{avaliador_b.f1_weighted:.4f}",
        "F1 Macro": f"{avaliador_b.f1_macro:.4f}",
        "Semelhança Média": f"{avaliador_b.semelhanca_media:.4f}",
        "Taxa Reconstrução": f"{avaliador_b.taxa_reconstrucao:.4f}"
    }
])

print("\n=======================================================")
print("     RESULTADO COMPARATIVO: SENTINELA 0.5 VS 0.0      ")
print("=======================================================")
print(df_comp.to_string(index=False))

# Plotar Matrizes de Confusão Lado a Lado
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

labels_cat = [1, 2, 3, 4, 5, 6, 7]
sns.heatmap(avaliador_a.matriz_conf, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels_cat, yticklabels=labels_cat, ax=axes[0])
axes[0].set_title(f'Cenário A: Sentinela 0.5\n(F1-Weighted: {avaliador_a.f1_weighted:.4f})', fontweight='bold')
axes[0].set_xlabel('Predito'); axes[0].set_ylabel('Real')

sns.heatmap(avaliador_b.matriz_conf, annot=True, fmt='d', cmap='Greens',
            xticklabels=labels_cat, yticklabels=labels_cat, ax=axes[1])
axes[1].set_title(f'Cenário B: Absência Estrita 0.0\n(F1-Weighted: {avaliador_b.f1_weighted:.4f})', fontweight='bold')
axes[1].set_xlabel('Predito'); axes[1].set_ylabel('Real')

plt.tight_layout()
os.makedirs(OUT_RELATORIO, exist_ok=True)
PATH_PLOT = os.path.join(OUT_RELATORIO, 'comparacao_sentinela_05_vs_00.png')
plt.savefig(PATH_PLOT, dpi=300)
print(f"\nGráfico comparativo salvo em: {PATH_PLOT}")
