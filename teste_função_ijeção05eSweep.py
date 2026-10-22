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
# # Teste: Injeção de Sentinelas (0.5) nas Colunas Faltantes do Fujita no Dataset PAN e Projeção rSWeeP 600D
#
# Avalia o comportamento e a representação latente do dataset canônico de referência PAN
# ao injetar valores sentinela (0.5) nas exatas colunas de genes ausentes do dataset Fujita,
# mapeadas a partir de `tracking_genes_adicionados_Fujita.csv`.
#
# Este experimento reconfigura a árvore de diretórios de saída para o padrão dinâmico
# `output_pan_anotado_pan_com_sentinela_05`, persistindo a matriz AnnData modificada,
# a exportação Matrix Market (.mtx) e a projeção vetorial rSWeeP 600D oficial.

# %%
# ==============================================================================
# CONFIGURAÇÃO DO REPOSITÓRIO E VALIDAÇÃO FAIL FAST DE INTEGRIDADE GIT (COLAB)
# ==============================================================================
REPO_NAME = "pipiline_hopifield"
REPO_URL = "https://github.com/letdevx/pipiline_hopifield.git"
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
# output_pan_anotado_pan_com_sentinela_05
OUTPUTS_EXP = configurar_diretorios(
    nome_ref="pan_anotado",
    nome_alvo="pan_com_sentinela_05",
    criar_diretorios=True,
)

print(f"[Config] Diretório de saídas ativo: {OUTPUTS_EXP}")
print(
    f"[Config] Referência: '{getattr(config, 'NOME_REF', 'pan_anotado')}' | Alvo: '{getattr(config, 'NOME_ALVO', 'pan_com_sentinela_05')}'"
)

# %%
# ==============================================================================
# IMPORTAÇÃO DOS MÓDULOS CIENTÍFICOS DO PIPELINE
# ==============================================================================
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from alinhamento import ExportadorMTX
from treinamento import ProjetorSWeePR

# %%
# ==============================================================================
# RESOLUÇÃO DINÂMICA DE CAMINHOS DOS ARTEFATOS E DADOS DE ENTRADA
# ==============================================================================
SEED = 42

# Variáveis opcionais de override para apontar caminhos específicos manualmente
CAMINHO_PAN_CUSTOMIZADO = "/content/drive/MyDrive/imputs/pan_anotado.h5ad"
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

    # 1. Busca prioritária nas pastas de alinhamento do experimento (output_pan_anotado_pan_com_sentinela_05/alinhamento)
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


PATH_PAN_RESOLVIDO = resolver_caminho_pan(CAMINHO_PAN_CUSTOMIZADO)
PATH_TRACKING_RESOLVIDO = resolver_caminho_tracking_fujita(CAMINHO_TRACKING_CUSTOMIZADO)

print(f"[Entrada] Matriz PAN : {PATH_PAN_RESOLVIDO}")
print(f"[Entrada] Tracking   : {PATH_TRACKING_RESOLVIDO}")

# Definição e criação das subpastas no diretório configurado
DIR_ALINHAMENTO = config.OUT_ALINHAMENTO
DIR_TREINAMENTO = config.OUT_TREINAMENTO
DIR_MTX_SENTINELA = os.path.join(DIR_ALINHAMENTO, "mtx_pan_sentinela")
os.makedirs(DIR_MTX_SENTINELA, exist_ok=True)
os.makedirs(DIR_TREINAMENTO, exist_ok=True)

PATH_PAN_SENTINELA_H5AD = os.path.join(OUTPUTS_EXP, "pan_com_sentinela_05.h5ad")
PATH_TRACKING_COPIA = os.path.join(
    DIR_ALINHAMENTO, "tracking_genes_adicionados_fujita.csv"
)
PATH_SWEEP_SENTINELA_TXT = os.path.join(
    DIR_TREINAMENTO, "matriz_sweep600d_pan_com_sentinela.txt"
)
PATH_SWEEP_SENTINELA_NPY = os.path.join(
    DIR_TREINAMENTO, "matriz_sweep600d_pan_com_sentinela.npy"
)

# %%
# ==============================================================================
# ETAPA 1: INJEÇÃO FÍSICA OOM-SAFE DE SENTINELAS (0.5) NO DATASET PAN
# ==============================================================================
print("\n" + "=" * 75)
print("  ETAPA 1: INJEÇÃO FÍSICA DE SENTINELAS (0.5) NO DATASET PAN")
print("=" * 75)

# 1. Carrega o arquivo de tracking e extrai as posições de colunas ausentes
df_tracking = pd.read_csv(PATH_TRACKING_RESOLVIDO)
print(f"[Tracking] Total de registros no arquivo: {len(df_tracking):,}")

if "posicao_coluna" not in df_tracking.columns:
    raise ValueError(
        f"Coluna 'posicao_coluna' ausente no arquivo de tracking: {df_tracking.columns.tolist()}"
    )

indices_colunas_ausentes = df_tracking["posicao_coluna"].dropna().astype(int).to_numpy()
indices_colunas_ausentes = np.unique(indices_colunas_ausentes)
print(
    f"[Tracking] Posições únicas de colunas ausentes do Fujita a injetar 0.5: {len(indices_colunas_ausentes):,}"
)
print(f"  • Menor índice de coluna: {indices_colunas_ausentes.min()}")
print(f"  • Maior índice de coluna: {indices_colunas_ausentes.max()}")

# 2. Carrega a matriz do dataset PAN
print(f"\n[PAN] Carregando AnnData: {PATH_PAN_RESOLVIDO}...")
adata_pan = ad.read_h5ad(PATH_PAN_RESOLVIDO)
n_obs, n_vars = adata_pan.shape
print(f"[PAN] Dimensões originais: {n_obs:,} células × {n_vars:,} genes")

if indices_colunas_ausentes.max() >= n_vars:
    raise IndexError(
        f"Índice de coluna máximo ({indices_colunas_ausentes.max()}) excede o número "
        f"de genes do dataset PAN ({n_vars}). Verifique se a matriz PAN já está alinhada ao espaço canônico."
    )

# 3. Injeção de Sentinelas (0.5) em lotes OOM-Safe preservando formato esparso
assert adata_pan.X is not None, "A matriz adata_pan.X não pode ser nula."
CHUNK_SIZE = 5000
chunks_modificados: list[sp.csr_matrix] = []

print(
    f"[Injeção] Injetando valor 0.5 nas colunas ausentes em lotes de {CHUNK_SIZE:,} células..."
)
for inicio in range(0, n_obs, CHUNK_SIZE):
    fim = min(inicio + CHUNK_SIZE, n_obs)
    fatia = adata_pan.X[inicio:fim]

    # Converte o lote para csr_matrix e lil_matrix sem alocar matriz global densa
    chunk_mat = sp.csr_matrix(fatia, dtype=np.float32)
    chunk_lil = chunk_mat.tolil()
    chunk_lil[:, indices_colunas_ausentes] = 0.5
    chunks_modificados.append(sp.csr_matrix(chunk_lil, dtype=np.float32))

# Monta a matriz final esparsa CSR
adata_pan.X = sp.vstack(chunks_modificados, format="csr")
print(
    f"[Injeção] Injeção concluída com sucesso! Formato esparso CSR: {adata_pan.X.shape} (nnz: {adata_pan.X.nnz:,})"
)

# 4. Atualiza metadados var com máscara de auditoria
mask_sentinela = np.zeros(n_vars, dtype=bool)
mask_sentinela[indices_colunas_ausentes] = True
adata_pan.var["sentinela_fujita_05"] = mask_sentinela

# 5. Persiste o AnnData modificado
print(
    f"\n[Persistência] Gravando AnnData com sentinela em: {PATH_PAN_SENTINELA_H5AD}..."
)
adata_pan.write_h5ad(PATH_PAN_SENTINELA_H5AD, compression="gzip")
tamanho_mb = os.path.getsize(PATH_PAN_SENTINELA_H5AD) / (1024 * 1024)
print(f"[Persistência] Arquivo .h5ad gravado com sucesso! ({tamanho_mb:.2f} MB)")

# Salva cópia do tracking no diretório de alinhamento
df_tracking.to_csv(PATH_TRACKING_COPIA, index=False)
print(f"[Persistência] Cópia do tracking salva em: {PATH_TRACKING_COPIA}")

# 6. Exportação no formato Matrix Market (MTX) para rSWeeP
print(f"\n[MTX] Exportando formato Matrix Market em: {DIR_MTX_SENTINELA}...")
exportador = ExportadorMTX(out_dir=DIR_MTX_SENTINELA)
exportador.exportar(matriz=adata_pan, forcar=True)
print(f"[MTX] Exportação concluída com sucesso em {DIR_MTX_SENTINELA}")

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
print(" [rSWeeP] PROJEÇÃO 600D CONCLUÍDA COM SUCESSO!")
print(f"  • Shape final : {W_sweep_sentinela.shape} (células × 600 dimensões)")
print(f"  • TXT salvo em: {PATH_SWEEP_SENTINELA_TXT}")
print(f"  • NPY salvo em: {PATH_SWEEP_SENTINELA_NPY}")
print(f"  • OrthBase    : {projetor_sentinela.path_orthbase}")
print("=" * 75)
