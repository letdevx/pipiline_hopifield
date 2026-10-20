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

import importlib
import os
import subprocess
import sys

# %%
try:
    import anndata
    import polars
    import scanpy
except ImportError:
    print("Instalando dependências compatíveis com o ambiente do Colab...")
    # !pip install -q "pandas==2.2.3" anndata scanpy polars

# %% [markdown]
# # Teste: Injeção de Sentinelas (0.5) via MarcadorSentinela e Vetorização rSWeeP 600D
#
# Avalia o impacto da inclusão do ruído de valores sentinela (0.5) nas colunas de genes ausentes
# sobre a acurácia de classificação de um modelo MLP projetado no espaço latente rSWeeP 600D.
#
# Este script utiliza a infraestrutura completa e a resolução dinâmica de caminhos
# configurada no `pipeline_generico_Pan_Mathys.ipynb`, suportando tanto o Google Colab
# quanto o ambiente local Windows/Linux.

# %%
# ==============================================================================
# CONFIGURAÇÃO DO REPOSITÓRIO E VALIDAÇÃO FAIL FAST DE INTEGRIDADE GIT (COLAB)
# ==============================================================================
REPO_NAME = "pipiline_hopifield"
REPO_URL = "https://github.com/letdevx/pipiline_hopifield.git"
REPO_BRANCH = "reconstrucao_Pan_Mathys_Otimizado"
DEST_PATH = f"/content/{REPO_NAME}"

# Commit hash esperado no Colab (ADR 024)
EXPECTED_COMMIT = "be19fe03517afcc8d4313adedfc558d0ef7409cd"


def _is_google_colab() -> bool:
    """Detecta se a execução atual está ocorrendo no Google Colab."""
    return (
        "google.colab" in sys.modules
        or os.path.exists("/content")
        or os.environ.get("COLAB_GPU") is not None
        or os.environ.get("COLAB_RELEASE_TAG") is not None
    )


def sincronizar_e_validar_repo_colab(
    repo_url: str,
    dest_path: str,
    branch: str,
    expected_commit: str | None,
) -> dict[str, str] | None:
    """Sincroniza o repositório na VM do Colab e valida o commit HEAD (Fail Fast)."""
    if not _is_google_colab():
        print(
            "[Git Validador] Execução em ambiente local detectada. "
            "Sincronização remota e validação estrita de commit do Colab ignoradas."
        )
        return None

    if expected_commit is None or not expected_commit.strip():
        raise RuntimeError(
            "No ambiente do Google Colab, a variável EXPECTED_COMMIT é obrigatória!"
        )

    exp_clean = expected_commit.strip().lower()

    # 1. Clona se o repositório ainda não existir na VM
    if not os.path.exists(dest_path):
        print(f"[Colab Git] Clonando '{repo_url}' em '{dest_path}'...")
        res_clone = subprocess.run(
            ["git", "clone", repo_url, dest_path],
            capture_output=True,
            text=True,
            check=False,
        )
        if res_clone.returncode != 0:
            raise RuntimeError(
                f"Falha ao clonar repositório:\n{res_clone.stderr.strip()}"
            )

    # 2. Checkout e pull da branch remota
    print(f"[Colab Git] Atualizando branch '{branch}' a partir de origin...")
    subprocess.run(
        ["git", "-C", dest_path, "checkout", branch],
        capture_output=True,
        text=True,
        check=False,
    )
    subprocess.run(
        ["git", "-C", dest_path, "fetch", "origin", branch],
        capture_output=True,
        text=True,
        check=False,
    )
    res_pull = subprocess.run(
        ["git", "-C", dest_path, "pull", "origin", branch],
        capture_output=True,
        text=True,
        check=False,
    )
    if res_pull.returncode != 0:
        print(f"[Colab Git] Aviso durante git pull:\n{res_pull.stderr.strip()}")

    # 3. Inspeciona o commit HEAD da VM
    def _git_meta(args: list[str]) -> str:
        return subprocess.run(
            ["git", "-C", dest_path, *args],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    head_completo = _git_meta(["rev-parse", "HEAD"]).lower()
    head_curto = _git_meta(["rev-parse", "--short", "HEAD"]).lower()
    autor = _git_meta(["log", "-1", "--format=%an"])
    data = _git_meta(["log", "-1", "--format=%ad", "--date=iso"])
    mensagem = _git_meta(["log", "-1", "--format=%s"])

    coincide = head_completo.startswith(exp_clean) or (exp_clean == head_curto)

    if not coincide:
        res_ancestor = subprocess.run(
            [
                "git",
                "-C",
                dest_path,
                "merge-base",
                "--is-ancestor",
                exp_clean,
                "HEAD",
            ],
            capture_output=True,
            check=False,
        )
        if res_ancestor.returncode == 0:
            coincide = True

    if not coincide:
        msg_erro = (
            "\n"
            + "=" * 80
            + "\n[FALHA DE INTEGRIDADE - FAIL FAST] DIVERGÊNCIA DE COMMIT NO GOOGLE COLAB\n"
            + "=" * 80
            + f"\n• Commit Esperado : {expected_commit}"
            + f"\n• Commit no Colab : {head_curto} ({head_completo})"
            + f"\n• Branch atual    : {branch}"
            + f"\n• Autor do Commit : {autor}"
            + f"\n• Data do Commit  : {data}"
            + f"\n• Mensagem        : {mensagem}"
            + "\n"
            + "-" * 80
            + "\nMOTIVO DA INTERRUPÇÃO:"
            + "\nO repositório clonado na VM do Google Colab não está no commit esperado."
            + "\nVocê provavelmente esqueceu de fazer push dos commits locais do seu laptop!"
            + "\n"
            + "\nCOMO CORRIGIR:"
            + "\n1. No terminal do seu laptop, envie os commits mais recentes:"
            + "\n       git status"
            + f"\n       git push origin {branch}"
            + "\n2. No Google Colab, re-execute esta célula para atualizar a VM e prosseguir."
            + "\n"
            + "=" * 80
        )
        raise RuntimeError(msg_erro)

    print(
        f"[Git Validador] Integridade confirmada! HEAD: {head_curto} - '{mensagem}' (Branch: {branch})"
    )
    return {
        "hash_completo": head_completo,
        "hash_curto": head_curto,
        "autor": autor,
        "data": data,
        "mensagem": mensagem,
        "branch": branch,
    }


# Executa a verificação na inicialização da célula
sincronizar_e_validar_repo_colab(
    repo_url=REPO_URL,
    dest_path=DEST_PATH,
    branch=REPO_BRANCH,
    expected_commit=EXPECTED_COMMIT,
)

# Adiciona caminhos do repositório ao sys.path se estiver no Colab
if _is_google_colab():
    for _p in (DEST_PATH, os.path.join(DEST_PATH, "src")):
        if _p not in sys.path:
            sys.path.insert(0, _p)

# %%
# ==============================================================================
# MONTAGEM DO GOOGLE DRIVE E CONFIGURAÇÃO DE CAMINHOS DINÂMICOS
# ==============================================================================
try:
    from google.colab import drive  # type: ignore

    if not os.path.exists("/content/drive"):
        print("[Colab] Montando Google Drive em /content/drive...")
        drive.mount("/content/drive")
except (ImportError, Exception):
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

from config import (
    OUT_ALINHAMENTO,
    OUT_MTX_ALVO_SENTINELA,
    OUTPUTS,
    PATH_ALVO,
    PATH_FEATURES_ALVO,
    PATH_FEATURES_REFERENCIA,
    PATH_LABELS_ALVO,
    PATH_LABELS_REFERENCIA,
    PATH_ORTHBASE_RDS,
    PATH_REFERENCIA,
    PATH_SWEEP_ALVO_SENTINELA,
    PATH_SWEEP_REFERENCIA,
    configurar_diretorios,
)

print(f"[Config] Diretório de saídas ativo: {OUTPUTS}")
print(
    f"[Config] Referência: '{getattr(config, 'NOME_REF', 'ref')}' | Alvo: '{getattr(config, 'NOME_ALVO', 'alvo')}'"
)

# %%
# ==============================================================================
# IMPORTAÇÃO DOS MÓDULOS CIENTÍFICOS DO PIPELINE
# ==============================================================================
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from alinhamento import LeitorFeatures, MarcadorSentinela
from treinamento import (
    ProjetorSWeePR,
    carregar_labels,
    remapear_labels_canonicos,
)

# %%
# ==============================================================================
# DEFINIÇÃO DOS ARTEFATOS E ESPAÇO GÊNICO DA REFERÊNCIA
# ==============================================================================
SEED = 42

# 1. Identificação do espaço de genes canônicos da referência
path_todos_genes = os.path.join(OUT_ALINHAMENTO, "genes_canonicos_completos.csv")
leitor = LeitorFeatures(PATH_FEATURES_REFERENCIA, PATH_FEATURES_ALVO)
leitor.ler()

genes_canonica: list[str]
if os.path.exists(path_todos_genes):
    print(f"[Genes] Carregando genes canônicos de {path_todos_genes}...")
    df_genes = pd.read_csv(path_todos_genes)
    col_g = "gene" if "gene" in df_genes.columns else df_genes.columns[0]
    genes_canonica = df_genes[col_g].dropna().tolist()
else:
    print(
        f"[Genes] Carregando genes canônicos diretamente da referência: {PATH_REFERENCIA}..."
    )
    _adata_ref = ad.read_h5ad(PATH_REFERENCIA, backed="r")
    genes_canonica = list(_adata_ref.var_names)
    if hasattr(_adata_ref, "file") and _adata_ref.file is not None:
        _adata_ref.file.close()

print(f"[Genes] Total de genes na referência canônica: {len(genes_canonica):,}")

# 2. Definição do diretório de saída do teste
DIR_TESTE_EXP = os.path.join(OUTPUTS, "experimento_sentinela_mlp")
os.makedirs(DIR_TESTE_EXP, exist_ok=True)

PATH_ALVO_SENTINELA_H5AD = os.path.join(DIR_TESTE_EXP, "alvo_mathys_com_sentinela.h5ad")
PATH_TRACKING_CSV = os.path.join(DIR_TESTE_EXP, "tracking_genes_ausentes.csv")
DIR_MTX_SENTINELA = os.path.join(DIR_TESTE_EXP, "mtx_sentinela")
PATH_SWEEP_SENTINELA_TXT = os.path.join(
    DIR_TESTE_EXP, "matriz_sweep600d_com_sentinela.txt"
)

# %%
# ==============================================================================
# ETAPA 1: INJEÇÃO FÍSICA DE SENTINELAS (0.5) VIA MarcadorSentinela (ADR 028)
# ==============================================================================
print("\n" + "=" * 75)
print("  ETAPA 1: INJEÇÃO FÍSICA DE SENTINELAS (0.5) NO DATASET ALVO")
print("=" * 75)

marcador = MarcadorSentinela(
    genes_referencia=genes_canonica,
    map_features_alvo=leitor.map_m,
    fill_value=0.5,
    chunk_size=5000,
)

adata_sentinela = marcador.marcar_dataset(
    adata_alvo=PATH_ALVO,
    path_saida_h5ad=PATH_ALVO_SENTINELA_H5AD,
    path_tracking_csv=PATH_TRACKING_CSV,
    exportar_mtx=True,
    path_saida_mtx=DIR_MTX_SENTINELA,
    forcar=True,
)

mask_ausentes = marcador.obter_mascara_ausentes()
print(
    f"[MarcadorSentinela] Matriz alinhada: {adata_sentinela.shape[0]:,} células × {adata_sentinela.shape[1]:,} genes."
)
print(
    f"[MarcadorSentinela] Genes ausentes preenchidos com 0.5: {int(np.sum(mask_ausentes)):,}"
)

# %%
# ==============================================================================
# ETAPA 2: PROJEÇÃO VETORIAL rSWeeP 600D (VIA PACOTE OFICIAL R DA UFPR)
# ==============================================================================
print("\n" + "=" * 75)
print("  ETAPA 2: PROJEÇÃO VETORIAL rSWeeP 600D (PACOTE OFICIAL UFPR)")
print("=" * 75)

ProjetorSWeePR.verificar_e_instalar_dependencias_r()

projetor_sentinela = ProjetorSWeePR(
    path_matriz=DIR_MTX_SENTINELA,
    path_saida=PATH_SWEEP_SENTINELA_TXT,
    n_componentes=600,
    seed=SEED,
    path_orthbase=PATH_ORTHBASE_RDS,
    forcar_recriacao=False,
)
projetor_sentinela.projetar()

W_sweep_sentinela = projetor_sentinela.Wswp
assert W_sweep_sentinela is not None
print(
    f"[rSWeeP] Projeção 600D concluída com sucesso! Dimensões: {W_sweep_sentinela.shape}"
)

# %%
# ==============================================================================
# ETAPA 3: AVALIAÇÃO DE ACURÁCIA COM CLASSIFICADOR MLP
# ==============================================================================
print("\n" + "=" * 75)
print("  ETAPA 3: AVALIAÇÃO DO IMPACTO DAS SENTINELAS NA ACURÁCIA DO MLP")
print("=" * 75)

labels_alvo_bruto = carregar_labels(PATH_LABELS_ALVO)
assert labels_alvo_bruto is not None

CLASSES_CANONICAS = [1, 2, 3, 4, 5, 6, 7]
clo_alvo, _ = remapear_labels_canonicos(
    labels_alvo_bruto, classes_validas=CLASSES_CANONICAS, label_desconhecido=0
)

# Filtra apenas células com linhagem canônica conhecida (1 a 7)
idx_validos = np.where(clo_alvo > 0)[0]
X_eval = W_sweep_sentinela[idx_validos]
y_eval = clo_alvo[idx_validos]

X_train, X_test, y_train, y_test = train_test_split(
    X_eval, y_eval, test_size=0.30, random_state=SEED, stratify=y_eval
)

print(
    f"[MLP] Treinando com {X_train.shape[0]:,} células e testando em {X_test.shape[0]:,} células..."
)
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    max_iter=60,
    random_state=SEED,
    verbose=True,
)
mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)
acuracia = accuracy_score(y_test, y_pred)

print("\n" + "=" * 75)
print(f"  ACURÁCIA DO MLP COM RUÍDO DE SENTINELA (0.5) = {acuracia * 100:.2f}%")
print("=" * 75)
print("\nRelatório de Classificação Detalhado:")
print(classification_report(y_test, y_pred))
