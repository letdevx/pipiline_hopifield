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

# %%

# # `pipeline_hopfield_v2` — Pipeline Completo com Classes `src/`
# 
# Reimplementação do fluxo de `script01_analises_preliminares.ipynb` usando as
# classes disponíveis em `src/` em vez de funções inline.
# 
# **Contexto biológico.** O experimento parte de matrizes binárias de expressão
# gênica (~40 000 células Fujita × ~45 000 células Mathys × N genes). São
# selecionados os **5 000 genes mais frequentes** do Fujita e cada célula é
# representada por um vetor SWeeP de **600 dimensões** via projeção `W0 · R5k`
# (rSWeeP, AIBIALab). A memória associativa armazena perfis **binários** por
# tipo celular; o espaço SWeeP é usado para clusterizar e escolher protótipos.
# 
# **Diferenças em relação ao `script01`:**
# - Usa as classes de `src/` para binarização, alinhamento, seleção de genes,
#   projeção SWeeP, extração de padrões e avaliação.
# - Cobre o pipeline completo (binarização → avaliação cross-dataset).
# - Remapeamento de classes seguindo o padrão do script01: classes não presentes
#   em `[1, 3, 4, 5, 6, 7]` são remapeadas para a classe `2`.
# - Configuração: `nc=10` clusters por classe, `k=1` representante por centroide
#   → **70 padrões** (7 classes × 10 subclusters).
# 
# **Rede utilizada:** Modern Hopfield Network (Ramsauer et al., 2020) com
# capacidade de armazenamento exponencial e recuperação equivalente a um passo
# de *attention* (`softmax(β · Ξ · ξ) · Ξᵀ`).

# ## 1. Imports e configuração

# %%
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
import os
import sys

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

# !cd {DEST_PATH} && git checkout Teste_sem_binarização_dados_brutos

# Adiciona a pasta 'src' da VM ao path do Python
SRC_PATH = os.path.join(DEST_PATH, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

print("Módulos prontos para importação!")

# %%
try:
    from google.colab import drive
    if not os.path.exists('/content/drive'):
        print("[Colab] Montando Google Drive em /content/drive...")
        drive.mount('/content/drive')
except (ImportError, Exception):
    pass

DRIVE_INPUTS = '/content/drive/Othercomputers/Meu laptop/Documents/Letworkspace/Teste hop/imputs'
# !ls "{DRIVE_INPUTS}"

# %%
# %load_ext autoreload
# %autoreload 2

import sys, os

import importlib
import numpy as np
import pandas as pd
import torch
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, classification_report

# Detecção robusta do diretório raiz e de src/ para Jupyter, Scripts e Colab
if '__file__' in globals():
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
else:
    ROOT_DIR = os.path.abspath(os.getcwd())
    if not os.path.exists(os.path.join(ROOT_DIR, 'src')) and os.path.exists(os.path.join(os.path.dirname(ROOT_DIR), 'src')):
        ROOT_DIR = os.path.dirname(ROOT_DIR)

SRC_DIR = os.path.join(ROOT_DIR, 'src')
for p in (ROOT_DIR, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import config
importlib.reload(config)
from config import (
    PATH_REFERENCIA, PATH_ALVO, PATH_FEATURES_REFERENCIA, PATH_FEATURES_ALVO,
    PATH_SWEEP_REFERENCIA, PATH_SWEEP_ALVO, PATH_LABELS_REFERENCIA, PATH_LABELS_ALVO,
    OUT_BINARIZACAO, OUT_ALINHAMENTO, OUT_TOP_GENES,
    OUT_TREINAMENTO, OUT_HOPFIELD, OUT_RELATORIO,
)

import alinhamento
importlib.reload(alinhamento)

from preprocessing import Binarizador
from alinhamento import (
    LeitorFeatures, AnalisadorSobreposicao, Alinhador, AlinhadorEsparso,
    ValidadorAlinhamento, SelecionadorGenesFrequentes, AnalisadorCobertura,
)
from treinamento import (
    GeradorConjuntoTreinamento, CarregadorDadosFujita,
    ProjetorSWeP, ProjetorSWeePR,
    ExtratorPadroesSubcluster, ModernHopfieldNetwork, AvaliadorHopfield,
    GeradorRelatorio,
)
from treinamento.hopfield_utils import wsort, closervects


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.manual_seed_all(SEED)
    # Garante determinismo em operações CUDA (útil para reprodutibilidade)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f'Dispositivo: {device} ({torch.cuda.get_device_name(0)})')
    print(f'VRAM disponível: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
else:
    device = torch.device('cpu')
    print(f'Dispositivo: {device} (GPU não disponível)')


# ## 2. Binarização
# 
# Converte as matrizes de expressão `.h5ad` para formato binário (valores > 0 → 1,
# zeros → 0). O `Binarizador` detecta automaticamente se o arquivo já existe e
# pula o processamento nesse caso.

# %%
binarizador_ref = Binarizador(path_h5ad=PATH_REFERENCIA, out_dir=OUT_BINARIZACAO)
binarizador_alvo = Binarizador(path_h5ad=PATH_ALVO, out_dir=OUT_BINARIZACAO)

binarizador_ref.binarizar()
binarizador_alvo.binarizar()

print('Referência binarizada em:', binarizador_ref.path_binarizada)
print('Alvo binarizado em:', binarizador_alvo.path_binarizada)


# ## 3. Alinhamento de espaços gênicos
# 
# Os dois datasets têm espaços gênicos distintos (36 591 genes no Fujita,
# 32 643 no Mathys, ~30 312 em comum). O alinhamento:
# 
# 1. Lê os mapeamentos `gene_name → Ensembl ID` de cada dataset.
# 2. Define a ordem canônica dos genes baseada no Fujita (referência).
# 3. Realinha ambas as matrizes para esse espaço canônico.
#    - Genes ausentes no **Mathys** são preenchidos com `0.5` como sentinela.
# 4. Valida que as duas matrizes resultantes têm genes na mesma ordem.

# %%


# Passo 1 — Leitura dos arquivos de features
leitor = LeitorFeatures(PATH_FEATURES_REFERENCIA, PATH_FEATURES_ALVO)
leitor.ler()
print(leitor)


# %%


# Passo 2 — Análise de sobreposição dos espaços gênicos
# var_names idênticos no original e no binarizado — lemos direto do original
_f = ad.read_h5ad(PATH_REFERENCIA, backed='r')
var_names_f_original = _f.var_names.tolist()
_f.file.close()
del _f

analisador = AnalisadorSobreposicao(leitor.map_f, leitor.map_m, var_names_f_original)
analisador.analisar()
print(analisador)


# %%


# Passo 3 — Alinhamento dos dois h5ad binarizados (100% Esparso & OOM-Safe)
alinhador = AlinhadorEsparso(
    path_binarizada_m = binarizador_alvo.path_binarizada,
    path_binarizada_f = binarizador_ref.path_binarizada,
    out_dir           = OUT_ALINHAMENTO,
    map_f             = leitor.map_f,
    map_m             = leitor.map_m,
    gene_alvo_idx     = analisador.gene_alvo_idx,
    genes_ordenados   = analisador.genes_ordenados,
)
alinhador.alinhar()
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


# ## 4. Seleção dos top-5000 genes frequentes
# 
# Seleciona os 5 000 genes com maior frequência (soma de coluna) no Fujita.
# Em seguida verifica quantos desses genes estão presentes no Mathys e gera
# os conjuntos filtrados para treinamento.

# %%


path_top5k   = os.path.join(OUT_TOP_GENES,   'top5000_frequentes.csv')

# Top 5000 genes mais frequentes do Fujita (calculado direto do .h5ad esparso)
selecionador = SelecionadorGenesFrequentes(path_h5ad=alinhador.path_f_alinhado, n=5000)
selecionador.calcular(out_csv=path_top5k).salvar(path_top5k)
print(selecionador)


# %%


# Cobertura dos top-5000 genes do Fujita no Mathys
cobertura = AnalisadorCobertura(path_top5k, leitor.map_f, leitor.map_m)
cobertura.analisar(out_csv=os.path.join(OUT_ALINHAMENTO, 'top5000_cobertura_mathys.csv'))


# %%


# Conjuntos de treinamento filtrados (Fujita + Mathys com Sentinela 0.5 OOM-Safe)
res_extracao = alinhador.extrair_subconjunto(
    lista_genes_ou_csv = path_top5k,
    out_dir            = OUT_TREINAMENTO,
    fill_value_mathys  = 0.5,
    exportar_npy       = True,
    exportar_h5ad      = True,
)
path_f_top5k = res_extracao['path_f_npy']
path_m_top5k = res_extracao['path_m_npy']


# ## 5. Projeção SWeeP (rSWeeP via R / fallback Python)
# 
# Projeta a matriz binarizada do Fujita (células × 5 000 genes) no espaço
# SWeeP de 600 dimensões usando a base ortonormal rSWeeP.
# Se o R não estiver disponível, `ProjetorSWeePR` usa o fallback Python (QR sintético).
# 
# ```
# Wswp = W0 @ R5k        (células × 600)
# ```

# %%


projetor_r = ProjetorSWeePR(
    path_matriz   = path_f_top5k,
    path_saida    = PATH_SWEEP_REFERENCIA,
    n_componentes = 600,
    seed          = SEED,
)
projetor_r.projetar()


# ## 6. Carregamento dos dados
# 
# Carrega:
# - `W0`: matriz binária Fujita (células × 5 000 genes) — usada como padrões da rede.
# - `labels`: rótulos inteiros de tipo celular por célula.
# - `Wswp`: projeções SWeeP pré-computadas (células × 600) — usadas para K-means.

# %%


# Fujita — padrões de treinamento
carregador = CarregadorDadosFujita(
    path_matriz = path_f_top5k,
    path_genes  = path_top5k,
    path_labels = PATH_LABELS_REFERENCIA,
    path_sweep  = PATH_SWEEP_REFERENCIA,
    n_genes     = 5000,
)
carregador.carregar()
print(carregador)


# %%


# Mathys — dados para imputação cross-dataset
# Genes ausentes no Mathys foram preenchidos com 0.5 (sentinela) pelo Alinhador.
print('[Mathys] Carregando matriz top5000...')
if path_m_top5k.endswith('.npy'):
    W_mathys = np.load(path_m_top5k)
else:
    W_mathys = pd.read_csv(path_m_top5k).to_numpy(dtype=np.float32)
print(f'[Mathys] W_mathys shape: {W_mathys.shape}')

print('[Mathys] Carregando rótulos...')
labels_mathys = np.loadtxt(PATH_LABELS_ALVO, dtype=int)
print(f'[Mathys] labels shape: {labels_mathys.shape}, tipos: {np.unique(labels_mathys)}')


# ## 7. Remapeamento de classes (clo)
# 
# Seguindo o padrão do `script01_analises_preliminares.m` original:
# classes não presentes em `[1, 3, 4, 5, 6, 7]` são remapeadas para `2`,
# resultando em 7 classes: Excitatory (1), Endothelial/remapeadas (2),
# Inhibitory (3), Astrocytes (4), Microglia (5), Oligodendrocytes (6),
# OPCs (7).
# 
# ```matlab
# clo = cl;
# clo(~ismember(clo,[1 3 4 5 6 7 0])) = 2;
# ```

# %%


clo = carregador.labels.copy()
clo[~np.isin(clo, [1, 3, 4, 5, 6, 7, 0])] = 2

clo_m = labels_mathys.copy()
clo_m[~np.isin(clo_m, [1, 3, 4, 5, 6, 7, 0])] = 2

print('Distribuição Fujita (clo):')
vals, counts = np.unique(clo, return_counts=True)
for v, c in zip(vals, counts):
    print(f'  classe {v}: {c:>6d} células')

print('\nDistribuição Mathys (clo_m):')
vals_m, counts_m = np.unique(clo_m, return_counts=True)
for v, c in zip(vals_m, counts_m):
    print(f'  classe {v}: {c:>6d} células')


# ## 8. PCA no espaço SWeeP
# 
# Aplica PCA **sem centralização** sobre as projeções SWeeP — equivalente ao
# `pca(W, 'Centered', false)` do MATLAB. Os scores resultantes `Wpc`
# são usados como espaço auxiliar para visualizações e análises.

# %%


projetor = ProjetorSWeP(n_features=5000, n_componentes=600, seed=SEED)
projetor.usar_sweep_precomputado(carregador.Wswp).aplicar_pca()
print(projetor)


# ## 8b. Visualização PCA — separação das classes no espaço SWeeP
# 
# Scatter plot dos dois primeiros componentes principais do espaço SWeeP do Fujita,
# colorido por tipo celular (`clo`). Permite verificar se os 7 tipos já formam grupos
# separados **antes** do treinamento da rede — separação visual aqui indica que o espaço
# SWeeP captura bem as diferenças biológicas entre os tipos celulares.
# 
# `projetor.Wpc` já está calculado na célula anterior; nenhum novo cálculo é necessário.

# %%


# PCA scatter — PC1 × PC2 colorido por tipo celular (amostra 5 000 células)
rng_pca = np.random.default_rng(SEED)
idx_pca = rng_pca.choice(len(projetor.Wpc), min(5000, len(projetor.Wpc)), replace=False)

CLASSE_NOMES = {0: 'Sem rótulo', 1: 'Excitatory', 2: 'Endothelial',
                3: 'Inhibitory',  4: 'Astrocytes', 5: 'Microglia',
                6: 'Oligodendrocytes', 7: 'OPCs'}
CORES = {0: '#cccccc', 1: '#e6194b', 2: '#3cb44b', 3: '#4363d8',
         4: '#f58231', 5: '#911eb4', 6: '#42d4f4', 7: '#f032e6'}

fig_pca, ax = plt.subplots(figsize=(9, 7))
for cls in sorted(CORES):
    mask = clo[idx_pca] == cls
    if mask.any():
        ax.scatter(projetor.Wpc[idx_pca][mask, 0],
                   projetor.Wpc[idx_pca][mask, 1],
                   c=CORES[cls], label=f'{cls} — {CLASSE_NOMES[cls]}',
                   s=6, alpha=0.6, linewidths=0)
ax.set_xlabel('PC 1'); ax.set_ylabel('PC 2')
ax.set_title(f'PCA — espaço SWeeP Fujita (n={len(idx_pca):,} células)')
ax.legend(markerscale=3, fontsize=9, loc='best')
plt.tight_layout(); plt.show()


# ## 9. Extração de padrões por subcluster (perf35)
# 
# Para cada uma das 7 classes executa KMeans com `nc=10` clusters no espaço
# SWeeP e seleciona o vetor binário mais próximo de cada centroide como
# representante. Resulta em `7 × 10 = 70 padrões`.
# 
# ```matlab
# for ii in classes:
#     km = kmeans(Wswp[clo==ii], nc)
#     for centroide in km.centroids:
#         idx = closervects(Wswp[clo==ii], centroide, k=1)
#         perf35.append(W0[clo==ii][idx])
# ```

# %%


extrator = ExtratorPadroesSubcluster(
    W0      = carregador.W0,
    labels  = clo,
    classes = [1, 2, 3, 4, 5, 6, 7],
    seed    = SEED,
    k       = 1,
)
extrator.extrair(projetor.Wswp)
perf35 = extrator.padroes
print(extrator)
print(f'perf35 shape: {perf35.shape}  (esperado: (70, 5000))')


# ## 10. Treinamento da rede (rede35)
# 
# Armazena os 70 padrões na Modern Hopfield Network.
# 
# **Regra de armazenamento:** simplesmente guardar os padrões — não há
# treinamento iterativo.
# 
# **Parâmetros:**
# - `beta=8.0`: temperatura inversa do softmax (maior → mais winner-takes-all)
# - `n_iters=1`: uma iteração de atualização já é suficiente
# - `threshold=0.5`: genes preenchidos com 0.5 (sentinela Mathys) são tratados
#   como ausentes (< 0.5 → 0) na binarização da query

# %%


rede35 = ModernHopfieldNetwork(beta=30.0, n_iters=1, binary=True, threshold=0.0)
# A rede agora mapeará e armazenará apenas a versão original W0-Binária
rede35.store(perf35)
meta_eval = extrator.meta  # mapeamento padrao -> classe
print(rede35)


# ## 10b. Alternativa: carregar rede pre-treinada
# 
# **Use esta celula em vez das secoes 9 e 10** quando o treinamento foi feito
# noutra maquina. Copie os arquivos rede35_v2.pt e rede35_v2_metadata.json para
# a pasta outputs/hopfield/ desta maquina e execute apenas esta celula.
# 
# Se treinou aqui (secoes 9 e 10), **pule esta celula**.

# %%


import json as _json

PATH_PT   = os.path.join(OUT_HOPFIELD, 'rede35_v2.pt')
PATH_META = os.path.join(OUT_HOPFIELD, 'rede35_v2_metadata.json')

# Carrega rede e metadados
rede35 = ModernHopfieldNetwork.carregar(PATH_PT)

with open(PATH_META) as _f:
    _meta_json = _json.load(_f)

# Reconstrói perf35 em {0,1} a partir dos padrões salvos em {-1,+1}
perf35 = ((rede35.patterns.cpu().numpy() + 1.0) / 2.0).clip(0.0, 1.0).astype('float32')

# Variáveis de avaliação (substituem extrator.meta quando rede vem de fora)
meta_eval = [tuple(x) for x in _meta_json['meta']]

print(rede35)
print(f'perf35 shape: {perf35.shape}')
print(f'Classes: {_meta_json["classes"]}  nc={_meta_json["nc"]}  padroes={_meta_json["n_patterns"]}')


# ## 11. Teste numa subclasse (clo == 3)
# 
# Seguindo o padrão da seção 13 do `script01`: embaralha aleatoriamente as
# células da classe 3 e testa as primeiras 1000 (amostra representativa).
# 
# ```matlab
# Wk4  = wsort(W0(clo==3, :));          % embaralhamento aleatório
# Wtes = hopf_ts(Wk4(1:1000,:), rede35);
# ```

# %%


NC   = 10
CLASSES_ARR = np.array([1, 2, 3, 4, 5, 6, 7])

# Agora a query é o espaço W0 Binário Original!
Wk4    = wsort(carregador.W0[clo == 3])
n_test = min(1000, Wk4.shape[0])
Wtes   = rede35.retrieve(Wk4[:n_test], batch_size=4096)
print(f'hopf_ts(Wswp[:{n_test}], rede35): shape {Wtes.shape}')

perf35_f = perf35.astype(np.float64)
Wtes_f   = Wtes.astype(np.float64)
a2 = (Wtes_f ** 2).sum(axis=1, keepdims=True)
b2 = (perf35_f ** 2).sum(axis=1, keepdims=True).T
idx_proto = (a2 + b2 - 2 * (Wtes_f @ perf35_f.T)).argmin(axis=1)
pred_sub  = CLASSES_ARR[idx_proto // NC]

acc_sub = (pred_sub == 3).mean()
print(f'\nAcurácia subclasse clo==3: {acc_sub * 100:.2f}%')


# %%


y_true_sub = np.full(n_test, 3)
labels_sub = sorted(set(y_true_sub) | set(pred_sub))
print(classification_report(y_true_sub, pred_sub, labels=labels_sub, zero_division=0))

cm_sub = confusion_matrix(y_true_sub, pred_sub, labels=labels_sub)
fig, ax = plt.subplots(figsize=(max(6, len(labels_sub)), max(5, len(labels_sub))))
sns.heatmap(cm_sub, annot=True, fmt='d', cmap='Oranges',
            xticklabels=labels_sub, yticklabels=labels_sub, ax=ax)
ax.set_xlabel('Predito'); ax.set_ylabel('Real')
ax.set_title('Matriz de Confusão — rede35 (subconjunto clo==3)')
plt.tight_layout(); plt.show()


# ## 12. Auto-imputação — Fujita → Fujita
# 
# Baseline interno: a rede treinada em Fujita recebe as próprias células Fujita.
# Esperamos alta taxa de reconstrução e classificação.

# %%


print('=== Auto-imputação: Fujita → Fujita ===')
Wrecuperado_f = rede35.retrieve(carregador.W0, batch_size=4096)

avaliador_f = AvaliadorHopfield(
    padroes = perf35,
    classes = [1, 2, 3, 4, 5, 6, 7],
    nc      = 10,
    meta    = meta_eval,
)
avaliador_f.avaliar(Wrecuperado_f, clo).plotar(titulo='Confusão — rede35 (Fujita → Fujita)')
print(avaliador_f)


# ## 13. Imputação cross-dataset — Mathys com sentinela 0.5
# 
# A rede treinada em Fujita recebe células do Mathys. Os 6 289 genes ausentes
# no Mathys foram preenchidos com `0.5` pelo `Alinhador` — o limiar `threshold=0.5`
# da rede os trata como ausentes (0) na binarização da query.

# %%


print('=== Imputação cross-dataset: Mathys ===')

# =========================================================================
# AUTO-IMPUTAÇÃO DE CROSS-DATASET MATHYS 
print('\\n--- Processo de Imputação ---')
print('Usando template Fujita para preencher buracos do Mathys (np.where(== 0.5))')

# Consulta a Rede com W_m_bin, que possui os valores 0.5 originais
Wrecuperado_m = rede35.retrieve(W_mathys, batch_size=4096)

genes_faltantes_qtd = np.sum(W_mathys == 0.5)
# Preservamos as marcações corretas originais do Mathys onde a expressão existe.
W_mathys_imputado = np.where(W_mathys == 0.5, Wrecuperado_m, W_mathys)
genes_resolvidos_qtd = np.sum(W_mathys_imputado == 0.5)

print(f'Genes faltantes originais Mathys (0.5): {genes_faltantes_qtd}')
print(f'Genes faltantes após Imputação: {genes_resolvidos_qtd}')

os.makedirs(OUT_TOP_GENES, exist_ok=True)
PATH_IMPUTADO = os.path.join(OUT_TOP_GENES, 'X_mathys_IMPUTADO_rede35.npy')
np.save(PATH_IMPUTADO, W_mathys_imputado)
print(f'Matriz Mathys Imputada Exportada para: {PATH_IMPUTADO}')
# =========================================================================

avaliador_m = AvaliadorHopfield(
    padroes = perf35,
    classes = [1, 2, 3, 4, 5, 6, 7],
    nc      = 10,
    meta    = meta_eval,
)
avaliador_m.avaliar(Wrecuperado_m, clo_m).plotar(titulo='Confusão — rede35 (Mathys → Fujita, 0.5)')
print(avaliador_m)# ## 14. Imputação cross-dataset — Mathys binário puro (0.5 → 0)
# 
# Comparação: converte os valores sentinela `0.5 → 0` antes da recuperação,
# equivalente a tratar todos os genes ausentes como definitivamente inativos.
# Permite comparar o impacto do sentinela `0.5` na qualidade da recuperação.

# %%


W_mathys_bin = W_mathys.copy()
n_meio = int((W_mathys_bin == 0.5).sum())
W_mathys_bin[W_mathys_bin == 0.5] = 0.0
print(f'Valores convertidos de 0.5 → 0: {n_meio}')
print(f'Valores únicos após conversão: {np.unique(W_mathys_bin)}')

print('\n=== Imputação cross-dataset: Mathys (binário puro, sem 0.5) ===')
Wrecuperado_m_bin = rede35.retrieve(W_mathys_bin, batch_size=4096)

avaliador_m_bin = AvaliadorHopfield(
    padroes = perf35,
    classes = [1, 2, 3, 4, 5, 6, 7],
    nc      = 10,
    meta    = meta_eval,
)
avaliador_m_bin.avaliar(Wrecuperado_m_bin, clo_m).plotar(titulo='Confusão — rede35 (Mathys binário puro)')
print(avaliador_m_bin)


# %%


# --- Diagnóstico: mapeamento de classes Mathys → protótipos Fujita ---

print("=== Diagnóstico: distribuição de protótipos selecionados por classe Mathys ===\n")

perf_f64 = perf35.astype(np.float64)
Wm_f64   = Wrecuperado_m.astype(np.float64)
a2 = (Wm_f64 ** 2).sum(axis=1, keepdims=True)
b2 = (perf_f64 ** 2).sum(axis=1, keepdims=True).T
idx_proto_m = (a2 + b2 - 2 * (Wm_f64 @ perf_f64.T)).argmin(axis=1)
pred_m = np.array([meta_eval[i][0] for i in idx_proto_m])

classes_eval = [1, 2, 3, 4, 5, 6, 7]
CM_diag = pd.DataFrame(0, index=classes_eval, columns=classes_eval, dtype=int)
CM_diag.index.name   = 'Mathys (verdadeiro)'
CM_diag.columns.name = 'Fujita (predito)'

for true_c, pred_c in zip(clo_m, pred_m):
    if true_c in classes_eval and pred_c in classes_eval:
        CM_diag.loc[true_c, pred_c] += 1

print("Mapeamento Mathys → protótipos Fujita (contagens):")
display(CM_diag)

# Convergência: fração de posições que mudaram após retrieve()
# Comparação limitada às posições não-sentinela (onde W_mathys é 0 ou 1)
mask_nao_sentinela = (W_mathys != 0.5)
diff_frac = float((W_mathys[mask_nao_sentinela] != Wrecuperado_m[mask_nao_sentinela]).mean())
print(f"\nFração de genes não-sentinela alterados após retrieve(): {diff_frac:.4f}")
print("  ≈ 0 → rede não está modificando as células (verificar configuração)")
print("  > 0 → a rede está realizando a recuperação corretamente")


# ## 15. Persistência da rede

# %%


import json as _json

os.makedirs(OUT_HOPFIELD, exist_ok=True)
path_rede_v2 = os.path.join(OUT_HOPFIELD, 'rede35_v2.pt')
path_meta_v2 = os.path.join(OUT_HOPFIELD, 'rede35_v2_metadata.json')

# Salva rede
rede35.salvar(path_rede_v2)

# Salva metadados (necessários para AvaliadorHopfield na máquina de aplicação)
_metadata = {
    'classes'   : [1, 2, 3, 4, 5, 6, 7],
    'nc'        : 10,
    'n_patterns': int(perf35.shape[0]),
    'n_genes'   : int(perf35.shape[1]),
    'meta'      : [[int(c_), int(idx)] for c_, idx in extrator.meta],
}
with open(path_meta_v2, 'w') as _f:
    _json.dump(_metadata, _f, indent=2)

print(f'Rede salva em    : {path_rede_v2}')
print(f'Metadados salvos : {path_meta_v2}')
print('Para carregar: rede = ModernHopfieldNetwork.carregar(path)')


# ## Notas finais
# 
# **Diferenças em relação ao `pipilinePrincipal.ipynb`:**
# - Inclui remapeamento de classes `clo` (padrão do script01): classes raras → 2.
# - Inclui teste em subclasse (seção 11) e auto-imputação Fujita→Fujita (seção 12).
# - Avalia tanto Mathys com sentinela `0.5` quanto Mathys binário puro.
# 
# **Hiperparâmetros:**
# - `beta`: controla a nitidez da recuperação. Para padrões esparsos de alta
#   dimensão, valores entre 4 e 16 costumam funcionar bem.
# - `nc`: número de subclusters por classe. Aumentar aumenta a capacidade de
#   representar variabilidade intraclasse.
# - `k`: número de representantes por centroide. `k=1` usa o indivíduo mais
#   próximo; `k>1` pode ser usado para padrões de consenso.
# 
# **Classes utilizadas (src/):**
# ```
# preprocessing/  → Binarizador
# alinhamento/    → LeitorFeatures, AnalisadorSobreposicao, Alinhador,
#                    ValidadorAlinhamento, SelecionadorGenesFrequentes, AnalisadorCobertura
# treinamento/    → GeradorConjuntoTreinamento, CarregadorDadosFujita,
#                    ProjetorSWeP, ProjetorSWeePR, ExtratorPadroesSubcluster,
#                    ModernHopfieldNetwork, AvaliadorHopfield
# ```

# ## 16. Análise Comparativa — Fujita vs Mathys
# 
# Consolida as matrizes de confusão e as métricas de reconstrução dos três cenários
# avaliados nas seções 12, 13 e 14 para facilitar a comparação direta.

# %%


# Matrizes de confusão — contagens e normalizadas (todos os cenários)
fig, axes = plt.subplots(3, 2, figsize=(14, 18))

avaliador_f.plotar(    titulo='Fujita → Fujita (contagens)',          ax=axes[0, 0])
avaliador_f.plotar(    titulo='Fujita → Fujita (normalizada)',         ax=axes[0, 1], normalizado=True)
avaliador_m.plotar(    titulo='Mathys → Fujita 0.5 (contagens)',       ax=axes[1, 0])
avaliador_m.plotar(    titulo='Mathys → Fujita 0.5 (normalizada)',     ax=axes[1, 1], normalizado=True)
avaliador_m_bin.plotar(titulo='Mathys → Fujita binário (contagens)',   ax=axes[2, 0])
avaliador_m_bin.plotar(titulo='Mathys → Fujita binário (normalizada)', ax=axes[2, 1], normalizado=True)

plt.suptitle('Matrizes de Confusão — rede35', fontsize=14, y=1.01)
plt.tight_layout()
plt.show()


# ## 16b. t-SNE — estrutura global das células
# 
# Visualização não-linear da separação entre tipos celulares em duas dimensões.
# 
# **Passo A** — t-SNE dos 5 000 Fujita no espaço SWeeP (primeiros 50 PCs):
# mostra se os tipos celulares formam ilhas distintas no espaço de baixa dimensionalidade.
# 
# **Passo B** — t-SNE conjunto (Fujita ● + Mathys ▲ reconstruídos, 5 000 de cada):
# células do mesmo tipo das duas espécies devem se sobrepor se a reconstrução
# pela Hopfield Network preservou as identidades celulares. Divergência = sinal de
# perda de informação no alinhamento cross-dataset.

# %%


from sklearn.manifold import TSNE
from sklearn.decomposition import PCA as SkPCA

# ── Passo A: t-SNE de Fujita no espaço SWeeP ──────────────────────────────
rng_tsne = np.random.default_rng(SEED)
N_TSNE = 5000
idx_f_t = rng_tsne.choice(len(projetor.Wpc), N_TSNE, replace=False)

print(f'Rodando t-SNE em {N_TSNE} células Fujita (input: primeiros 50 PCs)...')
tsne_a = TSNE(n_components=2, perplexity=40, random_state=SEED, n_jobs=-1)
Z_f = tsne_a.fit_transform(projetor.Wpc[idx_f_t, :50])
print('  Concluído.')

fig_tsne_a, ax = plt.subplots(figsize=(9, 7))
for cls in sorted(CORES):
    mask = clo[idx_f_t] == cls
    if mask.any():
        ax.scatter(Z_f[mask, 0], Z_f[mask, 1],
                   c=CORES[cls], label=f'{cls} — {CLASSE_NOMES[cls]}',
                   s=8, alpha=0.7, linewidths=0)
ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
ax.set_title(f't-SNE — Fujita (SWeeP, n={N_TSNE:,} células)')
ax.legend(markerscale=3, fontsize=9)
plt.tight_layout(); plt.show()

# ── Passo B: t-SNE Fujita + Mathys (0.5) reconstruído ────────────────────
print('Ajustando PCA em espaço gênico (5000 dims → 50)...')
pca_gene = SkPCA(n_components=50, random_state=SEED)
pca_gene.fit(carregador.W0)

idx_m_t = rng_tsne.choice(len(Wrecuperado_m), N_TSNE, replace=False)
X_f_rec = pca_gene.transform(Wrecuperado_f[idx_f_t])   # Fujita reconstruído
X_m_rec = pca_gene.transform(Wrecuperado_m[idx_m_t])   # Mathys reconstruído
X_joint = np.vstack([X_f_rec, X_m_rec])

print(f'Rodando t-SNE conjunto ({2*N_TSNE:,} células)...')
tsne_b = TSNE(n_components=2, perplexity=40, random_state=SEED, n_jobs=-1)
Z_joint = tsne_b.fit_transform(X_joint)
print('  Concluído.')

Z_fj = Z_joint[:N_TSNE]
Z_mj = Z_joint[N_TSNE:]

fig_tsne_b, ax = plt.subplots(figsize=(10, 8))
for cls in sorted(CORES):
    if cls == 0:
        continue
    mf = clo[idx_f_t]   == cls
    mm = clo_m[idx_m_t] == cls
    if mf.any():
        ax.scatter(Z_fj[mf, 0], Z_fj[mf, 1], c=CORES[cls], s=8,
                   alpha=0.6, linewidths=0, marker='o', label=f'F cls{cls}')
    if mm.any():
        ax.scatter(Z_mj[mm, 0], Z_mj[mm, 1], c=CORES[cls], s=14,
                   alpha=0.8, linewidths=0.4, marker='^',
                   edgecolors='k', label=f'M cls{cls}')
ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
ax.set_title(f't-SNE — Fujita (●) + Mathys 0.5 (▲) reconstruídos (n={N_TSNE:,} cada)')
ax.legend(markerscale=2, fontsize=7, ncol=2, loc='best')
plt.tight_layout(); plt.show()


# ## 16c. DBSCAN — validação de clusters não-supervisionada
# 
# Aplica DBSCAN sobre as coordenadas t-SNE 2D do Fujita (Passo A acima) para verificar
# se os clusters de densidade concordam com os rótulos biológicos `clo`.
# 
# - **Curva k-NN**: ordena as distâncias ao k-ésimo vizinho — o "cotovelo" indica o
#   valor ideal de `eps`. Ajuste `EPS` na célula abaixo se necessário.
# - **ARI (Adjusted Rand Index)**: quantifica o acordo entre DBSCAN e `clo`.
#   ARI ≈ 1 → clusters não-supervisionados capturam os tipos celulares.
#   ARI ≈ 0 → resultado aleatório.

# %%


from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

# ── Escolha visual de eps (curva k-NN) ────────────────────────────────────
k_nn = 30
nbrs = NearestNeighbors(n_neighbors=k_nn).fit(Z_f)
dists, _ = nbrs.kneighbors(Z_f)
knn_dists = np.sort(dists[:, -1])

fig_dbscan_knn, ax = plt.subplots(figsize=(7, 3))
ax.plot(knn_dists, lw=1)
ax.set_xlabel('Pontos ordenados'); ax.set_ylabel(f'Distância ao {k_nn}º vizinho')
ax.set_title('Curva k-NN — escolha de eps para DBSCAN')
plt.tight_layout(); plt.show()

# ── DBSCAN nas coordenadas t-SNE Fujita ───────────────────────────────────
EPS        = 2.0    # ajuste com base na curva acima se necessário
MIN_SAMP   = 30

db = DBSCAN(eps=EPS, min_samples=MIN_SAMP).fit(Z_f)
labels_db  = db.labels_
n_clusters = len(set(labels_db)) - (1 if -1 in labels_db else 0)
n_ruido    = (labels_db == -1).sum()
print(f'Clusters encontrados: {n_clusters}  |  Pontos de ruído: {n_ruido} ({n_ruido/len(labels_db):.1%})')

# Apenas pontos com rótulo clo ≠ 0 para o ARI
mask_clo  = clo[idx_f_t] != 0
ari = adjusted_rand_score(clo[idx_f_t][mask_clo], labels_db[mask_clo])
print(f'ARI (DBSCAN vs clo, excl. classe 0): {ari:.4f}')
print('  ARI = 1 → clusters concordam perfeitamente com os rótulos biológicos')
print('  ARI ≈ 0 → concordância aleatória')

# ── Plot comparativo: clo vs DBSCAN ───────────────────────────────────────
fig_dbscan_comp, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
for cls in sorted(CORES):
    m = clo[idx_f_t] == cls
    if m.any():
        ax.scatter(Z_f[m, 0], Z_f[m, 1], c=CORES[cls],
                   label=f'{cls}—{CLASSE_NOMES[cls]}', s=6, alpha=0.7, linewidths=0)
ax.set_title('t-SNE colorido por tipo celular (clo)')
ax.legend(markerscale=3, fontsize=8)
ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')

ax = axes[1]
uniq_db = sorted(set(labels_db))
cmap_db = plt.get_cmap('tab20', len(uniq_db))
for i, lbl in enumerate(uniq_db):
    m = labels_db == lbl
    nome = 'ruído' if lbl == -1 else f'cluster {lbl}'
    cor  = '#aaaaaa' if lbl == -1 else cmap_db(i)
    ax.scatter(Z_f[m, 0], Z_f[m, 1], c=[cor], label=nome,
               s=6 if lbl != -1 else 3, alpha=0.7, linewidths=0)
ax.set_title(f'DBSCAN (eps={EPS}, min={MIN_SAMP})  ARI={ari:.3f}')
ax.legend(markerscale=3, fontsize=8, ncol=2)
ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')

plt.suptitle('Validação de clusters — DBSCAN vs rótulos biológicos', fontsize=12)
plt.tight_layout(); plt.show()


# %%


# Tabela de métricas globais — comparação entre os três cenários
df_resumo = pd.DataFrame([
    avaliador_f.metricas_resumo('Fujita → Fujita'),
    avaliador_m.metricas_resumo('Mathys → Fujita (0.5)'),
    avaliador_m_bin.metricas_resumo('Mathys → Fujita (bin)'),
]).set_index('dataset')

display(df_resumo.style
    .format({
        'n_celulas':         '{:,}',
        'acuracia':          '{:.2%}',
        'f1_macro':          '{:.4f}',
        'f1_weighted':       '{:.4f}',
        'taxa_reconstrucao': '{:.2%}',
        'semelhanca_media':  '{:.4f}',
    })
    .set_caption('Métricas de Reconstrução — rede35')
    .highlight_max(color='lightgreen', axis=0)
)


# %%


# Métricas por classe — precision, recall e F1 para cada dataset
print('=== Fujita → Fujita ===')
display(avaliador_f.metricas_por_classe())

print('=== Mathys → Fujita (0.5) ===')
display(avaliador_m.metricas_por_classe())

print('=== Mathys → Fujita (binário puro) ===')
display(avaliador_m_bin.metricas_por_classe())


# ## 18. Reconstrução dos Genes Ausentes no Mathys
# 
# Analisa como a rede reconstruiu os genes do top-5000 que estavam **ausentes no Mathys**
# (preenchidos com sentinela `0.5` no alinhamento), comparando os dois cenários:
# 
# - **0.5 (sentinela):** genes ausentes → `0.0` em bipolar → **neutros** no attention (não influenciam o padrão recuperado)
# - **bin (binário puro):** genes ausentes → `−1.0` em bipolar → **negativos** no attention (suprimem padrões que expressam esses genes)
# 
# **Referência:** taxa de ativação desses mesmos genes nas células Fujita (ground truth).

# %%


# --- Identificação dos genes ausentes e métricas por gene ---

# 1. Genes do top-5000 ausentes no Mathys
df_cob    = pd.read_csv(os.path.join(OUT_ALINHAMENTO, 'top5000_cobertura_mathys.csv'))
ausentes  = df_cob[df_cob['presente_mathys'] == False].copy()
print(f'Genes ausentes no Mathys (top-5000): {len(ausentes)}')

# 2. Mapa Ensembl ID → índice de coluna no arquivo top-5000
with open(path_f_top5k, encoding='utf-8') as fh:
    top5k_header = fh.readline().strip().split(',')
gene_to_col         = {g: i for i, g in enumerate(top5k_header)}
ausentes['col_idx'] = ausentes['ensembl_id'].map(gene_to_col)
ausentes            = ausentes.dropna(subset=['col_idx'])
col_idx             = ausentes['col_idx'].astype(int).values
gene_ids            = ausentes['ensembl_id'].values
freqs               = ausentes['frequencia'].values
print(f'Colunas localizadas na matriz: {len(col_idx)}')

# 3. Taxa de ativação por gene em cada condição
#    carregador.W0   (células_F × 5000)  — referência Fujita
#    Wrecuperado_m   (células_M × 5000)  — reconstruído com 0.5
#    Wrecuperado_m_bin (células_M × 5000) — reconstruído com 0.0
rate_ref  = carregador.W0[:, col_idx].mean(axis=0)
rate_m05  = Wrecuperado_m[:, col_idx].mean(axis=0)
rate_mbin = Wrecuperado_m_bin[:, col_idx].mean(axis=0)

# 4. Erro absoluto médio
mae_05  = float(np.abs(rate_m05  - rate_ref).mean())
mae_bin = float(np.abs(rate_mbin - rate_ref).mean())
print(f'\nMAE cenário 0.5 vs Fujita  : {mae_05:.4f}')
print(f'MAE cenário bin vs Fujita  : {mae_bin:.4f}')
vencedor = '0.5 (sentinela)' if mae_05 < mae_bin else 'binário puro'
print(f'Cenário mais próximo da referência: {vencedor}')

# 5. Tabela por gene
df_ausentes = pd.DataFrame({
    'gene'        : gene_ids,
    'frequencia'  : freqs,
    'ref_fujita'  : rate_ref.round(4),
    'rec_05'      : rate_m05.round(4),
    'rec_bin'     : rate_mbin.round(4),
    'diff_05_ref' : (rate_m05  - rate_ref).round(4),
    'diff_bin_ref': (rate_mbin - rate_ref).round(4),
})
display(df_ausentes.sort_values('frequencia', ascending=False).reset_index(drop=True))


# %%


# --- Visualizações: reconstrução dos genes ausentes ---

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f'Reconstrução dos {len(col_idx)} genes ausentes no Mathys', fontsize=13)

# Plot 1: taxa de referência vs taxa reconstruída (ambos cenários)
ax = axes[0]
ax.scatter(rate_ref, rate_m05,  alpha=0.7, label='Mathys 0.5', color='steelblue', s=40)
ax.scatter(rate_ref, rate_mbin, alpha=0.7, label='Mathys bin', color='tomato',    s=40, marker='s')
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='ideal (y=x)')
ax.set_xlabel('Taxa de ativação — Fujita (referência)')
ax.set_ylabel('Taxa de ativação — Mathys (reconstruído)')
ax.set_title('Referência vs Reconstruído')
ax.legend()
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)

# Plot 2: distribuição da diferença (reconstruído − referência) por cenário
ax = axes[1]
ax.hist(rate_m05  - rate_ref, bins=20, alpha=0.6, label='0.5 − ref', color='steelblue')
ax.hist(rate_mbin - rate_ref, bins=20, alpha=0.6, label='bin − ref', color='tomato')
ax.axvline(0, color='k', lw=1, ls='--')
ax.set_xlabel('Diferença (reconstruído − referência)')
ax.set_ylabel('Número de genes')
ax.set_title(f'Distribuição do erro\nMAE 0.5={mae_05:.3f} | MAE bin={mae_bin:.3f}')
ax.legend()

# Plot 3: barras horizontais — top-20 genes por frequência, taxa nas 3 condições
ax = axes[2]
n_show = min(20, len(col_idx))
ordem  = np.argsort(freqs)[::-1][:n_show]
y_pos  = np.arange(n_show)
h      = 0.25
ax.barh(y_pos + h,  rate_ref[ordem],  h, label='Fujita (ref)', color='gray',      alpha=0.8)
ax.barh(y_pos,      rate_m05[ordem],  h, label='Mathys 0.5',   color='steelblue', alpha=0.8)
ax.barh(y_pos - h,  rate_mbin[ordem], h, label='Mathys bin',   color='tomato',    alpha=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels([gene_ids[i][:14] for i in ordem], fontsize=8)
ax.set_xlabel('Taxa de ativação')
ax.set_title(f'Top-{n_show} genes ausentes (por frequência)')
ax.legend(fontsize=8)
ax.set_xlim(0, 1.1)

plt.tight_layout()
plt.show()


# ## 17. Relatório Final do Experimento
# 
# Gera relatório HTML autocontido com:
# - Metadados completos do experimento (topologia da rede, hiperparâmetros, dimensões dos datasets)
# - Métricas globais e por classe para os três cenários de avaliação
# - Matrizes de confusão (contagens e normalizadas)
# - Análise de reconstrução dos genes ausentes no Mathys (seção 18)
# 
# Arquivos gerados em `outputs/relatorio/`:
# - `metricas_globais.csv`
# - `metricas_por_classe.csv`
# - `relatorio_teste_5mil_genes_binario_intermediario_0.5.html`

# %%


import importlib
import treinamento.gerador_relatorio as _gr_mod
importlib.reload(_gr_mod)
from treinamento.gerador_relatorio import GeradorRelatorio

relatorio = GeradorRelatorio(
    out_dir          = OUT_RELATORIO,
    nome_experimento = 'teste_5mil_genes_binario_intermediario_0.5',
)
relatorio.adicionar_metadados(
    titulo           = 'teste 5 mil genes mais frequentes binário e intermediário 0.5',
    modelo           = 'Modern Hopfield Network (Ramsauer et al., 2020)',
    beta             = rede35.beta,
    n_iters          = rede35.n_iters,
    binary           = rede35.binary,
    threshold        = rede35.threshold,
    n_padroes        = int(rede35.patterns.shape[0]),
    n_genes          = int(perf35.shape[1]),
    n_classes        = 7,
    nc_subclusters   = NC,
    n_celulas_fujita = carregador.W0.shape[0],
    n_celulas_mathys = W_mathys.shape[0],
    seed             = SEED,
)
relatorio.adicionar_avaliador('Fujita → Fujita',       avaliador_f)
relatorio.adicionar_avaliador('Mathys → Fujita (0.5)', avaliador_m)
relatorio.adicionar_avaliador('Mathys → Fujita (bin)', avaliador_m_bin)
relatorio.adicionar_genes_ausentes(df_ausentes, mae_05, mae_bin)
relatorio.adicionar_figura('PCA — Espaço SWeeP Fujita',                fig_pca,          secao='Visualizações Exploratórias')
relatorio.adicionar_figura('t-SNE — Fujita (SWeeP)',                   fig_tsne_a,       secao='t-SNE e DBSCAN')
relatorio.adicionar_figura('t-SNE — Fujita (●) + Mathys (▲)',          fig_tsne_b,       secao='t-SNE e DBSCAN')
relatorio.adicionar_figura('DBSCAN — Curva k-NN',                      fig_dbscan_knn,   secao='t-SNE e DBSCAN')
relatorio.adicionar_figura('DBSCAN — Clusters vs Rótulos Biológicos',  fig_dbscan_comp,  secao='t-SNE e DBSCAN')
relatorio.gerar()
print(relatorio)

