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
# # Teste: Injeção Direta OOM-Safe de Sentinelas (0.5) em Arquivos Matrix Market (.mtx) e Projeção rSWeeP 600D
#
# Este experimento avalia a abordagem de injeção direta de valores sentinela (0.5)
# nas colunas correspondentes de genes ausentes do Fujita diretamente sobre o arquivo
# `matrix.mtx` em disco através de streaming com buffer (via `InjetorMTXSentinela`).
#
# **Vantagens em Hardware Modesto:**
# - Consumo de memória RAM inferior a 50 MB (imune a falhas de Out-Of-Memory).
# - I/O direto e streaming sequencial em blocos configuráveis de linhas.
# - Atualização automática de metadados e compatibilidade estrita com o pacote R `rSWeeP` da UFPR.

# %%
# ==============================================================================
# VALIDAÇÃO DE INTEGRIDADE DO COMMIT HEAD NO GOOGLE COLAB (ADR 024 - FAIL FAST)
# ==============================================================================
REPO_NAME = "pipiline_hopifield"
REPO_URL = f"https://github.com/letdevx/{REPO_NAME}.git"
REPO_BRANCH = "reconstrução_Fujita_com_mlp_funcionando_corretamente"
DEST_PATH = f"/content/{REPO_NAME}"

# Commit hash esperado no Colab (ADR 024)
EXPECTED_COMMIT = "e8f3f17"


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
        ["git", "-C", dest_path, "fetch", "origin", branch],
        capture_output=True,
        text=True,
        check=False,
    )
    subprocess.run(
        ["git", "-C", dest_path, "checkout", branch],
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

    if not os.path.exists("/content/drive/MyDrive"):
        print("[Colab] Montando Google Drive em /content/drive...")
        drive.mount("/content/drive")
    else:
        print("[Colab] Google Drive já montado em /content/drive.")
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
    OUT_TREINAMENTO,
    OUTPUTS,
    PATH_ORTHBASE_RDS,
    PATH_REFERENCIA,
    configurar_diretorios,
)

# Reconfiguração dinâmica da árvore de diretórios do projeto para:
# output_pan_anotado_pan_com_sentinela_mtx_05
OUTPUTS_EXP = configurar_diretorios(
    nome_ref="pan_anotado",
    nome_alvo="pan_com_sentinela_mtx_05",
    criar_diretorios=True,
)

print(f"[Config] Diretório de saídas ativo: {OUTPUTS_EXP}")
print(
    f"[Config] Referência: '{getattr(config, 'NOME_REF', 'pan_anotado')}' | Alvo: '{getattr(config, 'NOME_ALVO', 'pan_com_sentinela_mtx_05')}'"
)

# %%
# ==============================================================================
# IMPORTAÇÃO DOS MÓDULOS CIENTÍFICOS DO PIPELINE
# ==============================================================================
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from alinhamento import ExportadorMTX, InjetorMTXSentinela, ValidadorOrdemGenes
from treinamento import ProjetorSWeePR

# %%
# ==============================================================================
# RESOLUÇÃO DINÂMICA DE CAMINHOS DOS ARTEFATOS E DADOS DE ENTRADA
# ==============================================================================
SEED = 42

# Variáveis opcionais de override para apontar caminhos específicos manualmente
CAMINHO_PAN_CUSTOMIZADO = "/content/drive/MyDrive/imputs/pan_anotado.h5ad"
CAMINHO_MTX_BASE_CUSTOMIZADO = ""
CAMINHO_TRACKING_CUSTOMIZADO = ""


def resolver_caminho_pan(customizado: str = "") -> str:
    """Localiza o dataset PAN (pan_anotado.h5ad) entre caminhos do Google Drive e locais."""
    if customizado and os.path.exists(customizado):
        return customizado

    candidatos = [
        customizado,
        "/content/drive/MyDrive/imputs/pan_anotado.h5ad",
        "/content/drive/MyDrive/inputs/pan_anotado.h5ad",
        getattr(config, "PATH_REFERENCIA", ""),
        r"C:\Users\Leticia\Documents\Letworkspace\Teste hop\imputs\pan_anotado.h5ad",
        r"C:\Users\Leticia\Downloads\pan_anotado.h5ad",
        r"C:\Users\Leticia\Documents\Letworkspace\Sweep-Harmonization\Meus_testes\Colocando tipo celular nos dados\matriz_anotada_binaria.h5ad",
        "/content/drive/Othercomputers/Meu laptop/Documents/Letworkspace/Teste hop/imputs/pan_anotado.h5ad",
        "/content/drive/Othercomputers/Meu laptop/Documents/Letworkspace/Teste hop/imputs/PAN_Bin/pan_anotado.h5ad",
        "/content/drive/MyDrive/Teste hop/imputs/pan_anotado.h5ad",
        "/content/drive/MyDrive/pan_anotado.h5ad",
        os.path.join(ROOT_DIR, "imputs", "pan_anotado.h5ad"),
    ]

    for cand in candidatos:
        if cand and os.path.exists(cand):
            return cand

    raise FileNotFoundError(
        "[Erro] Dataset 'pan_anotado.h5ad' não encontrado nos caminhos mapeados. "
        "Verifique a montagem do Google Drive ou informe 'CAMINHO_PAN_CUSTOMIZADO'."
    )


def resolver_caminho_tracking_fujita(customizado: str = "") -> str:
    """Localiza o arquivo tracking_genes_adicionados_Fujita.csv entre a pasta de alinhamento e Drive."""
    if customizado:
        if os.path.exists(customizado):
            return customizado
        dir_name = os.path.dirname(customizado)
        if os.path.exists(dir_name):
            base_lower = os.path.basename(customizado).lower()
            try:
                for arq in os.listdir(dir_name):
                    if arq.lower() == base_lower:
                        return os.path.join(dir_name, arq)
            except OSError:
                pass

    pastas_alinhamento = [
        os.path.join(OUTPUTS_EXP, "alinhamento"),
        "/content/drive/MyDrive/output_pan_anotado_pan_com_sentinela_05/alinhamento",
        "/content/drive/MyDrive/output_pan_anotado_pan_com_sentinela_mtx_05/alinhamento",
        "/content/drive/MyDrive/Teste hop/output_pan_anotado_pan_com_sentinela_05/alinhamento",
        "/content/drive/Othercomputers/Meu laptop/Documents/Letworkspace/Teste hop/output_pan_anotado_pan_com_sentinela_05/alinhamento",
        "/content/output_pan_anotado_pan_com_sentinela_05/alinhamento",
        os.path.join(
            ROOT_DIR, "output_pan_anotado_pan_com_sentinela_05", "alinhamento"
        ),
        r"C:\Users\Leticia\Documents\Letworkspace\Teste hop\output_pan_anotado_pan_com_sentinela_05\alinhamento",
    ]

    nomes_tracking = [
        "tracking_genes_adicionados_Fujita.csv",
        "tracking_genes_adicionados_fujita.csv",
        "tracking_genes_adicionados.csv",
        "tracking_genes_ausentes.csv",
    ]

    # 1. Busca prioritária nas pastas de alinhamento do experimento
    for pasta in pastas_alinhamento:
        if pasta and os.path.exists(pasta):
            for nome in nomes_tracking:
                cand = os.path.join(pasta, nome)
                if os.path.exists(cand):
                    return cand
            try:
                for arq in os.listdir(pasta):
                    if arq.lower().endswith(".csv") and "tracking" in arq.lower():
                        return os.path.join(pasta, arq)
            except OSError:
                pass

    # 2. Candidatos secundários (locais e pastas gerais do Drive)
    candidatos = [
        r"C:\Users\Leticia\Documents\Letworkspace\Teste faltantes Pan\Dados\tracking_genes_adicionados_Fujita.csv",
        "/content/tracking_genes_adicionados_Fujita.csv",
        "/content/drive/Othercomputers/Meu laptop/Documents/Letworkspace/Teste faltantes Pan/Dados/tracking_genes_adicionados_Fujita.csv",
        "/content/drive/MyDrive/Teste faltantes Pan/Dados/tracking_genes_adicionados_Fujita.csv",
        "/content/drive/MyDrive/imputs/tracking_genes_adicionados_Fujita.csv",
        "/content/drive/MyDrive/tracking_genes_adicionados_Fujita.csv",
        os.path.join(ROOT_DIR, "tracking_genes_adicionados_Fujita.csv"),
    ]

    for cand in candidatos:
        if cand and os.path.exists(cand):
            return cand

    raise FileNotFoundError(
        "[Erro] Arquivo 'tracking_genes_adicionados_Fujita.csv' não encontrado na pasta "
        "'output_pan_anotado_pan_com_sentinela_05/alinhamento' nem nos caminhos mapeados. "
        "Verifique se a pasta existe ou informe 'CAMINHO_TRACKING_CUSTOMIZADO'."
    )


# Definição e criação das subpastas no diretório configurado
DIR_ALINHAMENTO = config.OUT_ALINHAMENTO
DIR_TREINAMENTO = config.OUT_TREINAMENTO
DIR_MTX_BASE = os.path.join(DIR_ALINHAMENTO, "mtx_pan_base")
DIR_MTX_SENTINELA = os.path.join(DIR_ALINHAMENTO, "mtx_pan_sentinela_direto")

os.makedirs(DIR_MTX_BASE, exist_ok=True)
os.makedirs(DIR_MTX_SENTINELA, exist_ok=True)
os.makedirs(DIR_TREINAMENTO, exist_ok=True)

PATH_TRACKING_RESOLVIDO = resolver_caminho_tracking_fujita(CAMINHO_TRACKING_CUSTOMIZADO)
PATH_SWEEP_SENTINELA_TXT = os.path.join(
    DIR_TREINAMENTO, "matriz_sweep600d_pan_com_sentinela_direto.txt"
)
PATH_SWEEP_SENTINELA_NPY = os.path.join(
    DIR_TREINAMENTO, "matriz_sweep600d_pan_com_sentinela_direto.npy"
)

print(f"[Entrada] Tracking: {PATH_TRACKING_RESOLVIDO}")

# %%
# ==============================================================================
# PREPARAÇÃO DA PASTA MTX BASE DE ENTRADA (SE AINDA NÃO EXISTIR)
# ==============================================================================
# Verifica se já temos uma pasta MTX base pronta ou se devemos exportar a partir do pan_anotado.h5ad
candidatos_mtx_base = [
    CAMINHO_MTX_BASE_CUSTOMIZADO,
    DIR_MTX_BASE,
    "/content/drive/MyDrive/output_pan_anotado_pan_com_sentinela_05/alinhamento/mtx_pan_base",
    "/content/drive/MyDrive/output_pan_anotado_pan_com_sentinela_05/alinhamento/mtx_referencia",
]

dir_mtx_base_encontrado: str | None = None
for cand_dir in candidatos_mtx_base:
    if cand_dir and os.path.exists(os.path.join(cand_dir, "matrix.mtx")):
        dir_mtx_base_encontrado = cand_dir
        break

if dir_mtx_base_encontrado:
    print(f"[MTX Base] Pasta MTX base localizada em: {dir_mtx_base_encontrado}")
else:
    print(
        "[MTX Base] Pasta MTX base não encontrada. Exportando formato base a partir de pan_anotado.h5ad..."
    )
    PATH_PAN_RESOLVIDO = resolver_caminho_pan(CAMINHO_PAN_CUSTOMIZADO)
    print(f"[PAN] Carregando AnnData original: {PATH_PAN_RESOLVIDO}...")
    adata_pan_base = ad.read_h5ad(PATH_PAN_RESOLVIDO)
    print(
        f"[PAN] Dimensões originais: {adata_pan_base.shape[0]:,} células × {adata_pan_base.shape[1]:,} genes"
    )

    exportador_base = ExportadorMTX(out_dir=DIR_MTX_BASE)
    exportador_base.exportar(
        matriz=adata_pan_base,
        nome_etapa="Exportação MTX Base PAN",
        forcar=False,
    )
    dir_mtx_base_encontrado = DIR_MTX_BASE
    del adata_pan_base
    import gc

    gc.collect()

# %%
# ==============================================================================
# ETAPA 1: INJEÇÃO DIRETA OOM-SAFE DE SENTINELAS (0.5) NO ARQUIVO MATRIX MARKET (.MTX)
# ==============================================================================
print("\n" + "=" * 75)
print("  ETAPA 1: INJEÇÃO DIRETA DE SENTINELAS (0.5) EM DISCO VIA InjetorMTXSentinela")
print("=" * 75)

injetor_mtx = InjetorMTXSentinela(
    dir_mtx_origem=dir_mtx_base_encontrado,
    dir_mtx_destino=DIR_MTX_SENTINELA,
    fill_value=0.5,
    buffer_size=100_000,
)

relatorio_injecao = injetor_mtx.injetar_de_tracking(
    path_tracking_csv=PATH_TRACKING_RESOLVIDO,
    coluna_indice="posicao_coluna",
    indexado_em_zero=True,
    forcar=True,
)

print(f"\n[InjetorMTXSentinela] Injeção concluída com sucesso em: {DIR_MTX_SENTINELA}")
print(f"  • Células auditadas   : {relatorio_injecao['n_celulas']:,}")
print(f"  • Genes auditados     : {relatorio_injecao['n_genes']:,}")
print(f"  • Colunas injetadas   : {relatorio_injecao['colunas_injetadas']:,}")
print(f"  • NNZ final           : {relatorio_injecao['nnz_final']:,}")

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
assert W_sweep_sentinela is not None, "Erro: A matriz projetada retornou None!"
assert not np.isnan(W_sweep_sentinela).any(), (
    "Erro: Detectados valores NaN na projeção rSWeeP!"
)

# Salva também a projeção no formato binário NumPy (.npy) para carga ultrarrápida
np.save(PATH_SWEEP_SENTINELA_NPY, W_sweep_sentinela)

print("\n" + "=" * 75)
print(" [rSWeeP] PROJEÇÃO 600D CONCLUÍDA COM SUCESSO A PARTIR DO MTX INJETADO!")
print(f"  • Shape final : {W_sweep_sentinela.shape} (células × 600 dimensões)")
print(f"  • TXT salvo em: {PATH_SWEEP_SENTINELA_TXT}")
print(f"  • NPY salvo em: {PATH_SWEEP_SENTINELA_NPY}")
print(f"  • OrthBase    : {projetor_sentinela.path_orthbase}")
print("=" * 75)
