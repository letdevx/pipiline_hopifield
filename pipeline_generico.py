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

# %%
from __future__ import annotations

try:
    import anndata
    import scanpy
except ImportError:
    print("Instalando dependências compatíveis com o ambiente do Colab...")
    # Mantém o pandas travado na versão esperada pelo Colab (2.2.3)
    # !pip install -q "pandas==2.2.3" anndata scanpy

# %%
# !git push origin Teste_sem_binarização_dados_brutos

# %%
"""Notebook executável do Pipeline Genérico Hopfield para scRNA-seq.

Executa o fluxo fim a fim:
1. Binarização de matrizes scRNA-seq
2. Alinhamento de espaços gênicos com sentinela neutra (0.5)
3. Projeção dimensional compacta (SWeeP / rSWeeP)
4. Extração de padrões de subclusters por classe biológica
5. Treinamento e avaliação da Rede de Hopfield Moderna
6. Imputação e classificação cross-dataset
"""

import gc
import importlib
import os
import shutil
import sys

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn as sns
import torch
from numpy.typing import NDArray
from sklearn.metrics import classification_report, confusion_matrix

# %%
REPO_NAME = "pipiline_hopifield"  # Nome do seu repo
REPO_URL = "https://github.com/letdevx/pipiline_hopifield.git"
DEST_PATH = f"/content/{REPO_NAME}"

# Clona ou atualiza o código na VM
if not os.path.exists(DEST_PATH):
    print("Clonando código para a VM...")
    # !git clone {REPO_URL} {DEST_PATH}
else:
    print("Atualizando código na VM...")
    # !cd {DEST_PATH} && git pull

# !cd {DEST_PATH} && git checkout teste_pipeline_generico_F

# Adiciona a raiz do repo e a pasta 'src' da VM ao path do Python
for _p in (DEST_PATH, os.path.join(DEST_PATH, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# %%
try:
    from google.colab import drive  # type: ignore

    if not os.path.exists("/content/drive"):
        print("[Colab] Montando Google Drive em /content/drive...")
        drive.mount("/content/drive")
except (ImportError, Exception):
    pass

DRIVE_INPUTS = (
    "/content/drive/Othercomputers/Meu laptop/Documents/Letworkspace/Teste hop/imputs"
)
# !ls "{DRIVE_INPUTS}"

# %%
# !cd /content/pipiline_hopifield && git pull


# %%
import config

importlib.reload(config)


# %%
try:
    pass
    # %load_ext autoreload
    # %autoreload 2
except Exception:
    pass

# Detecção robusta do diretório raiz e de src/ para Jupyter, Scripts e Colab
if "__file__" in globals():
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
else:
    ROOT_DIR = os.path.abspath(os.getcwd())
    if not os.path.exists(os.path.join(ROOT_DIR, "src")) and os.path.exists(
        os.path.join(os.path.dirname(ROOT_DIR), "src")
    ):
        ROOT_DIR = os.path.dirname(ROOT_DIR)

SRC_DIR = os.path.join(ROOT_DIR, "src")
for p in (ROOT_DIR, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import config

importlib.reload(config)
import preprocessing
from config import (
    OUT_ALINHAMENTO,
    OUT_BINARIZACAO,
    OUT_HOPFIELD,
    OUT_IMPUTACAO,
    OUT_MTX_ALVO_IMPUTADO,
    OUT_MTX_ALVO_SENTINELA,
    OUT_MTX_REFERENCIA,
    OUT_TOP_GENES,
    PATH_ALVO,
    PATH_FEATURES_ALVO,
    PATH_FEATURES_REFERENCIA,
    PATH_LABELS_ALVO,
    PATH_LABELS_REFERENCIA,
    PATH_REFERENCIA,
    PATH_SWEEP_REFERENCIA,
)

importlib.reload(preprocessing)
import alinhamento

importlib.reload(alinhamento)
import treinamento

importlib.reload(treinamento)

from alinhamento import (
    AlinhadorEsparso,
    AnalisadorSobreposicao,
    ExportadorMTX,
    LeitorFeatures,
    ValidadorAlinhamento,
    ValidadorFeatures,
    ValidadorOrdemGenes,
)
from preprocessing import Binarizador
from treinamento import (
    AvaliadorHopfield,
    CarregadorDadosFujita,
    ExportadorImputacao,
    ExtratorPadroesSubcluster,
    ModernHopfieldNetwork,
    ProjetorSWeePR,
    ProjetorSWeP,
)
from treinamento.hopfield_utils import wsort

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed_all(SEED)
    # Garante determinismo em operações CUDA (útil para reprodutibilidade)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Dispositivo: {device} ({torch.cuda.get_device_name(0)})")
    print(
        f"VRAM disponível: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
    )
else:
    device = torch.device("cpu")
    print(f"Dispositivo: {device} (GPU não disponível)")


# %% [markdown]
#
# #### 2. Binarização
#  Converte as matrizes de expressão `.h5ad` para formato binário (valores > 0 → 1, zeros → 0). O `Binarizador` detecta automaticamente se o arquivo já existe e pula o processamento nesse caso.

# %%
binarizador_ref = Binarizador(path_h5ad=PATH_REFERENCIA, out_dir=OUT_BINARIZACAO)
binarizador_alvo = Binarizador(path_h5ad=PATH_ALVO, out_dir=OUT_BINARIZACAO)

binarizador_ref.binarizar()
binarizador_alvo.binarizar()

print("Referência binarizada em:", binarizador_ref.path_binarizada)
print("Alvo binarizado em:", binarizador_alvo.path_binarizada)


# %% [markdown]
# #### 3. Alinhamento de espaços gênicos ↔️
#  Os dois datasets têm espaços gênicos distintos (36 591 genes no Fujita, 32 643 no Mathys, ~30 312 em comum).
#  O alinhamento:
#  1. Lê os mapeamentos `gene_name → Ensembl ID` de cada dataset.
#  2. Valida a integridade dos arquivos de features e a compatibilidade com as matrizes AnnData.
#  3. Define a ordem canônica dos genes baseada no Fujita (referência).
#  4. Realinha ambas as matrizes para esse espaço canônico.
#     - Genes ausentes no **Mathys** são preenchidos com `0.5` como sentinela.
#  5. Valida que as duas matrizes resultantes têm genes na mesma ordem.

# %%
# Passo 1 — Leitura dos arquivos de features
leitor = LeitorFeatures(PATH_FEATURES_REFERENCIA, PATH_FEATURES_ALVO)
leitor.ler()
print(leitor)
assert leitor.map_f is not None and leitor.map_m is not None
assert (
    binarizador_ref.path_binarizada is not None
    and binarizador_alvo.path_binarizada is not None
)

# %%
# Passo 1.5 — Validação Prévia de Compatibilidade e Ordem de Colunas (Fail-Fast)
validador_feat = ValidadorFeatures(min_match_pct=50.0, min_genes_comuns=1000)
validador_feat.validar_tudo(
    path_features_ref=PATH_FEATURES_REFERENCIA,
    path_features_alvo=PATH_FEATURES_ALVO,
    path_h5ad_ref=binarizador_ref.path_binarizada,
    path_h5ad_alvo=binarizador_alvo.path_binarizada,
    map_f=leitor.map_f,
    map_m=leitor.map_m,
)

# %%
# Passo 2 — Análise de sobreposição dos espaços gênicos
# var_names idênticos no original e no binarizado — lemos direto do original
_f = ad.read_h5ad(PATH_REFERENCIA, backed="r")
var_names_f_original = _f.var_names.tolist()
_f.file.close()
del _f

analisador = AnalisadorSobreposicao(leitor.map_f, leitor.map_m, var_names_f_original)
analisador.analisar()
print(analisador)
assert (
    analisador.gene_alvo_idx is not None
    and analisador.genes_ordenados is not None
    and analisador.ids_so_f is not None
)

# %%
# Passo 3 — Alinhamento dos dois h5ad binarizados (100% Esparso & OOM-Safe)
alinhador = AlinhadorEsparso(
    path_binarizada_m=binarizador_alvo.path_binarizada,
    path_binarizada_f=binarizador_ref.path_binarizada,
    out_dir=OUT_ALINHAMENTO,
    map_f=leitor.map_f,
    map_m=leitor.map_m,
    gene_alvo_idx=analisador.gene_alvo_idx,
    genes_ordenados=analisador.genes_ordenados,
)
alinhador.alinhar()
alinhador.gerar_tracking(analisador.ids_so_f, leitor.map_f)
print(alinhador)
assert alinhador.path_f_alinhado is not None and alinhador.path_m_alinhado is not None

# %%
# Passo 4 — Validação da ordem de genes
validador = ValidadorAlinhamento(
    path_f_alinhado=alinhador.path_f_alinhado,
    path_m_alinhado=alinhador.path_m_alinhado,
    genes_ordenados=analisador.genes_ordenados,
)
validador.validar()

# Passo 5 — Validação estrita de Ensembl IDs (sem versão) e Exportação MTX da Referência
validador_genes = ValidadorOrdemGenes()
exportador_mtx_ref = ExportadorMTX(
    out_dir=OUT_MTX_REFERENCIA, validador=validador_genes
)
adata_f_exp = ad.read_h5ad(alinhador.path_f_alinhado, backed="r")
exportador_mtx_ref.exportar(
    matriz=adata_f_exp,
    genes_referencia=analisador.genes_ordenados,
    map_features=leitor.map_f,
    nome_etapa="Referência Alinhada (Fujita)",
)
if hasattr(adata_f_exp, "file") and adata_f_exp.file is not None:
    adata_f_exp.file.close()
del adata_f_exp
gc.collect()

# %% [markdown]
# #### 4. Adicionando genes faltantes ao conjunto alvo 🧮
#

# %%
# ==============================================================================
# Uso direto das matrizes completas alinhadas (Zero cópia / 100% Esparso)
# ==============================================================================
path_f_completo = alinhador.path_f_alinhado  # adataF_binarizado_alinhado.h5ad
path_m_completo = alinhador.path_m_alinhado  # adataM_binarizado_alinhado.h5ad

# Salva a lista completa de genes para o Carregador
path_todos_genes = os.path.join(OUT_ALINHAMENTO, "genes_canonicos_completos.csv")
pd.DataFrame({"gene": analisador.genes_ordenados}).to_csv(path_todos_genes, index=False)

# Exportação e Validação MTX do Alvo com Sentinela 0.5 (pré-Hopfield)
exportador_mtx_sentinela = ExportadorMTX(
    out_dir=OUT_MTX_ALVO_SENTINELA, validador=validador_genes
)
adata_m_alin = ad.read_h5ad(alinhador.path_m_alinhado)
mask_ausentes_cap4 = alinhador.obter_mascara_ausentes()
X_sentinela = adata_m_alin.X.copy()
if sp.issparse(X_sentinela):
    X_sentinela = X_sentinela.tolil()
    X_sentinela[:, mask_ausentes_cap4] = 0.5
    X_sentinela = X_sentinela.tocsr()
else:
    X_sentinela = np.asarray(X_sentinela, dtype=np.float32)
    X_sentinela[:, mask_ausentes_cap4] = 0.5

exportador_mtx_sentinela.exportar(
    matriz=X_sentinela,
    genes=analisador.genes_ordenados,
    genes_referencia=analisador.genes_ordenados,
    map_features=leitor.map_m,
    barcodes=adata_m_alin.obs_names.tolist(),
    nome_etapa="Alvo com Sentinela 0.5 (Mathys)",
)
del adata_m_alin, X_sentinela
gc.collect()


# %%

# %% [markdown]
# #### 5. Projeção SWeeP (rSWeeP via R ) 📚

# %%
projetor_r = ProjetorSWeePR(
    path_matriz=path_f_completo,
    path_saida=PATH_SWEEP_REFERENCIA,
    n_componentes=600,
    seed=SEED,
)
projetor_r.projetar()


# %%
assert analisador.genes_ordenados is not None
carregador = CarregadorDadosFujita(
    path_matriz=path_f_completo,  # Aceita .h5ad e .npy nativamente!
    path_genes=analisador.genes_ordenados,  # Pode passar a lista de genes diretamente
    path_labels=PATH_LABELS_REFERENCIA,
    path_sweep=PATH_SWEEP_REFERENCIA,
    n_genes=len(analisador.genes_ordenados),
)
carregador.carregar()
print(carregador)
assert carregador.W0 is not None and carregador.Wswp is not None


# %%
adata_m: ad.AnnData | None = None
W_mathys: NDArray[np.float32] | sp.spmatrix
if path_m_completo.endswith(".npy"):
    W_mathys = np.load(
        path_m_completo, mmap_mode="r"
    )  # mmap evita carregar 11GB de uma vez
else:
    adata_m = ad.read_h5ad(path_m_completo)
    W_mathys = adata_m.X  # Mantém em formato esparso CSR


# %%
# ==============================================================================
# Carregamento e Remapeamento dos Rótulos (Referência e Alvo)
# ==============================================================================
from treinamento import carregar_labels

# 1. Carregamento robusto dos rótulos brutos
labels_referencia = carregar_labels(PATH_LABELS_REFERENCIA)
labels_alvo = carregar_labels(PATH_LABELS_ALVO)
assert labels_referencia is not None and labels_alvo is not None

print(
    f"[Labels] Referência : {len(labels_referencia)} células | Tipos: {np.unique(labels_referencia)}"
)
print(
    f"[Labels] Alvo       : {len(labels_alvo)} células | Tipos: {np.unique(labels_alvo)}"
)

# 2. Remapeamento canônico (clo_ref e clo_alvo)
# Classes não presentes em [1, 3, 4, 5, 6, 7, 0] são remapeadas para a classe 2
clo_ref = labels_referencia.copy()
clo_ref[~np.isin(clo_ref, [1, 3, 4, 5, 6, 7, 0])] = 2

clo_alvo = labels_alvo.copy()
clo_alvo[~np.isin(clo_alvo, [1, 3, 4, 5, 6, 7, 0])] = 2

# 3. Exibição das distribuições
print("\nDistribuição Referência (clo_ref):")
vals_r, counts_r = np.unique(clo_ref, return_counts=True)
for v, c in zip(vals_r, counts_r):
    print(f"  classe {v}: {c:>6d} células")

print("\nDistribuição Alvo (clo_alvo):")
vals_a, counts_a = np.unique(clo_alvo, return_counts=True)
for v, c in zip(vals_a, counts_a):
    print(f"  classe {v}: {c:>6d} células")


# %%
assert analisador.genes_ordenados is not None
assert carregador.Wswp is not None
projetor = ProjetorSWeP(
    n_features=len(analisador.genes_ordenados), n_componentes=600, seed=SEED
)
projetor.usar_sweep_precomputado(carregador.Wswp).aplicar_pca()
print(projetor)
assert projetor.Wswp is not None


# %% [markdown]
# #### 9. Extração de padrões por subcluster (perf35)
# Para cada uma das 7 classes executa KMeans com `nc=30` clusters no espaço SWeeP e seleciona o vetor binário mais próximo de cada centroide como protótipo.
#

# %%
extrator = ExtratorPadroesSubcluster(
    W0=carregador.W0,
    labels=clo_ref,
    classes=[1, 2, 3, 4, 5, 6, 7],
    seed=SEED,
    nc=30,
    k=10,
)
extrator.extrair(projetor.Wswp)
assert extrator.padroes is not None and extrator.meta is not None
perf35 = extrator.padroes
meta_eval = extrator.meta
print(extrator)
print(
    f"perf35 shape: {perf35.shape}  (esperado: (210, {len(analisador.genes_ordenados)}))"
)


# %%
rede35 = ModernHopfieldNetwork(beta=50.0, n_iters=1, binary=True, threshold=0.0)
# A rede agora mapeará e armazenará apenas a versão original W0-Binária
rede35.store(perf35)
meta_eval = extrator.meta  # mapeamento padrao -> classe
print(rede35)

# %%
import os

PATH_PT = os.path.join(OUT_HOPFIELD, "rede35.pt")
PATH_META = os.path.join(OUT_HOPFIELD, "rede35.json")

# 1. Cria e armazena os 210 padrões na rede Hopfield
rede35 = ModernHopfieldNetwork(beta=50.0, n_iters=1, binary=True, threshold=0.0)
rede35.store(perf35)
meta_eval = extrator.meta

# 2. Salva a rede (.pt) e os metadados (.json) no disco
rede35.salvar_com_metadados(
    path_pt=PATH_PT,
    path_meta=PATH_META,
    meta=extrator.meta,
    classes=[1, 2, 3, 4, 5, 6, 7],
    nc=30,
)

print("Rede Hopfield e metadados salvos com sucesso em outputs/hopfield/!")


# %%


NC = 30
CLASSES_ARR = np.array([1, 2, 3, 4, 5, 6, 7])

assert carregador.W0 is not None
assert perf35 is not None

# Agora a query é o espaço W0 Binário Original!
W0_arr: NDArray[np.float32] = (
    sp.csr_matrix(carregador.W0).toarray().astype(np.float32)
    if sp.issparse(carregador.W0)
    else np.asarray(carregador.W0, dtype=np.float32)
)
Wk4_res = wsort(W0_arr[clo_ref == 3])
Wk4: NDArray[np.float32] = np.asarray(Wk4_res, dtype=np.float32)
n_test: int = min(1000, int(Wk4.shape[0]))
Wtes: NDArray[np.float32] = rede35.retrieve(Wk4[:n_test], batch_size=4096)
print(f"hopf_ts(Wswp[:{n_test}], rede35): shape {Wtes.shape}")

perf35_f = perf35.astype(np.float64)
Wtes_f = Wtes.astype(np.float64)
a2 = (Wtes_f**2).sum(axis=1, keepdims=True)
b2 = (perf35_f**2).sum(axis=1, keepdims=True).T
idx_proto = (a2 + b2 - 2 * (Wtes_f @ perf35_f.T)).argmin(axis=1)
pred_sub = CLASSES_ARR[idx_proto // NC]

acc_sub = (pred_sub == 3).mean()
print(f"\nAcurácia subclasse clo_ref==3: {acc_sub * 100:.2f}%")


y_true_sub = np.full(n_test, 3)
labels_sub = sorted(set(y_true_sub) | set(pred_sub))
print(classification_report(y_true_sub, pred_sub, labels=labels_sub, zero_division=0))

cm_sub = confusion_matrix(y_true_sub, pred_sub, labels=labels_sub)
fig, ax = plt.subplots(figsize=(max(6, len(labels_sub)), max(5, len(labels_sub))))
sns.heatmap(
    cm_sub,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels_sub,
    yticklabels=labels_sub,
    ax=ax,
)
ax.set_xlabel("Predito")
ax.set_ylabel("Real")
ax.set_title("Matriz de Confusão — rede35 (subconjunto clo==3)")
plt.tight_layout()
plt.show()


# %% [markdown]
# #### 12. Auto-imputação — Fujita → Fujita
# Baseline interno: a rede treinada em Fujita recebe as próprias células Fujita.Esperamos alta taxa de reconstrução e classificação.

# %%


# %%
# ==============================================================================
# 3. Auto-imputação (Fujita → Fujita)
# ==============================================================================
print("\n=== Auto-imputação: Fujita → Fujita ===")
assert carregador.W0 is not None
Wrecuperado_f = rede35.retrieve(carregador.W0, batch_size=2048)
print(f"Auto-imputação concluída! Shape: {Wrecuperado_f.shape}")


# %%
assert perf35 is not None
avaliador_f = AvaliadorHopfield(
    padroes=perf35,
    classes=[1, 2, 3, 4, 5, 6, 7],
    nc=30,
    nomes_classes=[
        "Excitatory",
        "Inhibitory",
        "Astrocytes",
        "Microglia",
        "Oligodendrocytes",
        "OPC",
        "Pericytes",
    ],
    meta=meta_eval,
)

# 1. Avalia a recuperação contra os rótulos verdadeiros
avaliador_f.avaliar(Wrecuperado_f, clo_ref)
print(avaliador_f)

# 2. Plota a Matriz de Confusão
avaliador_f.plotar(titulo="Confusão — rede35 (Fujita → Fujita)")


# %% [markdown]
# #### 13. Imputação cross-dataset — Mathys com Sentinela Neutra 0.5
# Injeta o valor sentinela 0.5 em todos os genes ausentes no Mathys durante a recuperação na rede Hopfield.
# A rede realiza a atenção contínua (onde 0.5 se torna 0.0 no espaço bipolar) e reconstrói o perfil completo.

# %%
print("=== Imputação cross-dataset: Mathys (Sentinela Neutra 0.5) ===")

# Se rede35 ou metadados não estiverem em memória (ex: reinício de kernel), carrega do checkpoint
if (
    "rede35" not in globals()
    or "perf35" not in globals()
    or "meta_eval" not in globals()
):
    PATH_PT = os.path.join(OUT_HOPFIELD, "rede35.pt")
    PATH_META = os.path.join(OUT_HOPFIELD, "rede35.json")
    if os.path.exists(PATH_PT) and os.path.exists(PATH_META):
        print(f"Carregando checkpoint de rede35 salvo em {PATH_PT}...")
        rede35, meta_eval, meta_json = ModernHopfieldNetwork.carregar_com_metadados(
            PATH_PT, PATH_META
        )
        assert rede35.patterns is not None
        perf35 = ((rede35.patterns.cpu().numpy() + 1.0) / 2.0).astype(np.float32)
    else:
        raise RuntimeError(
            "A variável 'rede35' não está definida e o checkpoint em outputs/hopfield/ não foi encontrado. Execute as células de treino anteriores."
        )

assert perf35 is not None

# 1. Identificação dos genes ausentes no Mathys
if "alinhador" in globals() and hasattr(alinhador, "obter_mascara_ausentes"):
    mask_ausentes = alinhador.obter_mascara_ausentes()
elif adata_m is not None and "presente_no_dataset" in adata_m.var:
    mask_ausentes = ~adata_m.var["presente_no_dataset"].to_numpy()
else:
    path_track = os.path.join(OUT_ALINHAMENTO, "tracking_genes_adicionados_mathys.csv")
    if os.path.exists(path_track):
        df_tr = pd.read_csv(path_track)
        mask_ausentes = np.zeros(W_mathys.shape[1], dtype=bool)
        mask_ausentes[df_tr["posicao_coluna"].to_numpy()] = True
    else:
        mask_ausentes = np.zeros(W_mathys.shape[1], dtype=bool)

n_genes_ausentes = np.sum(mask_ausentes)
print(
    f"Total de genes ausentes no Mathys (Sentinela 0.5): {n_genes_ausentes:,} de {len(mask_ausentes):,} genes canônicos."
)

# 2. Recuperação na rede Hopfield com injeção de 0.5 nos genes ausentes (Lotes OOM-Safe)
print(
    "\nRecuperando padrões na Modern Hopfield Network (batch_size=2048, sentinela=0.5)..."
)
Wrecuperado_m = rede35.retrieve(
    queries=W_mathys,
    batch_size=40000,
    mask_sentinela_ausentes=mask_ausentes,
    fill_value=0.5,
)
print(f"Recuperação concluída! Matriz reconstruída: {Wrecuperado_m.shape}")

# 3. Exportação Estruturada OOM-Safe em AnnData (.h5ad Gzip), .npy e JSON (ADR 017)
assert analisador.genes_ordenados is not None

exportador_imp = ExportadorImputacao(out_dir=OUT_IMPUTACAO)
rel_imp = exportador_imp.exportar(
    w_original=W_mathys,
    w_recuperado=Wrecuperado_m,
    genes_canonica=analisador.genes_ordenados,
    map_features=leitor.map_m,
    adata_alvo_original=alinhador.path_m_alinhado or PATH_ALVO,
    classes_reais=clo_alvo,
    info_modelo={
        "beta": rede35.beta,
        "n_iters": rede35.n_iters,
        "binary": rede35.binary,
        "threshold": rede35.threshold,
        "nc": 30,
        "n_padroes": perf35.shape[0],
    },
    nome_modelo="rede35",
    exportar_npy=True,
    substituir_sentinela=True,
    limiar_sentinela=0.5,
)

PATH_IMPUTADO_H5AD = rel_imp["arquivos_gerados"]["h5ad"]
PATH_IMPUTADO_NPY = rel_imp["arquivos_gerados"]["npy"]

# 4. Retrocompatibilidade: Garante o arquivo no caminho legado esperado em outputs/top_genes/
os.makedirs(OUT_TOP_GENES, exist_ok=True)
PATH_IMPUTADO = os.path.join(OUT_TOP_GENES, "X_mathys_IMPUTADO_rede35.npy")
if PATH_IMPUTADO_NPY and os.path.exists(PATH_IMPUTADO_NPY):
    shutil.copyfile(PATH_IMPUTADO_NPY, PATH_IMPUTADO)

genes_faltantes_qtd = rel_imp["estatisticas_imputacao"]["total_sentinelas_resolvidos"]
genes_resolvidos_um = rel_imp["estatisticas_imputacao"]["valores_resolvidos_para_um"]
genes_resolvidos_zero = rel_imp["estatisticas_imputacao"][
    "valores_resolvidos_para_zero"
]

print("\n--- Estatísticas da Imputação Cross-Dataset (ADR 017) ---")
print(f"  Total de coordenadas sentinelas resolvidas: {genes_faltantes_qtd:,}")
print(
    f"  Posições ativadas (1.0): {genes_resolvidos_um:,} ({genes_resolvidos_um / max(1, genes_faltantes_qtd) * 100:.2f}%)"
)
print(
    f"  Posições inativadas (0.0): {genes_resolvidos_zero:,} ({genes_resolvidos_zero / max(1, genes_faltantes_qtd) * 100:.2f}%)"
)
print(f"\n[Exportação] Matriz AnnData (.h5ad Gzip) : {PATH_IMPUTADO_H5AD}")
print(f"[Exportação] Matriz NumPy (.npy)        : {PATH_IMPUTADO_NPY}")
print(f"[Exportação] Retrocompatibilidade (.npy) : {PATH_IMPUTADO}")

# 6. Exportação e Validação MTX do Alvo Imputado pós-Hopfield
exportador_mtx_imp = ExportadorMTX(
    out_dir=OUT_MTX_ALVO_IMPUTADO, validador=validador_genes
)
adata_imp_loaded = ad.read_h5ad(PATH_IMPUTADO_H5AD, backed="r")
exportador_mtx_imp.exportar(
    matriz=adata_imp_loaded,
    genes_referencia=analisador.genes_ordenados,
    map_features=leitor.map_m,
    nome_etapa="Alvo Imputado pós-Hopfield (Mathys)",
)
if hasattr(adata_imp_loaded, "file") and adata_imp_loaded.file is not None:
    adata_imp_loaded.file.close()
del adata_imp_loaded
gc.collect()

# 5. Avaliação do Tipo Celular Cross-Dataset
avaliador_m = AvaliadorHopfield(
    padroes=perf35,
    classes=[1, 2, 3, 4, 5, 6, 7],
    nc=30,
    meta=meta_eval,
)
avaliador_m.avaliar(Wrecuperado_m, clo_alvo).plotar(
    titulo="Confusão — rede35 (Mathys → Fujita, Sentinela 0.5)"
)
print(avaliador_m)
