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

import glob
import importlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# %%
try:
    import anndata
    import polars
    import scanpy
    import scipy
except ImportError:
    print(
        "[Colab Setup] Instalando dependências compatíveis com o ambiente do Colab..."
    )
    # !pip install -q "pandas==2.2.3" anndata scanpy polars scipy

# %% [markdown]
# # Projeção Espectral rSWeeP 600D — Dataset Fujita (`matrizFiltradaeNormalizadaF`)
#
# Este notebook executa **exclusivamente a projeção dimensional espectral via rSWeeP**
# (algoritmo oficial desenvolvido pelo AIBIALab / UFPR, De Pierri et al., 2020) sobre o dataset
# Fujita filtrado e normalizado (`matrizFiltradaeNormalizadaF.h5ad`), localizado na pasta `imputs`
# no ambiente do Google Colab ou Google Drive.
#
# ## Diretrizes Arquiteturais e Científicas
# - **ADR 019:** Obrigatoriedade irrevogável do pacote R oficial `rSWeeP` (`orthBase()` + `SWeeP()`),
#   sem fallbacks sintéticos;
# - **ADR 021:** Congelamento determinístico da base ortonormal em formato `.rds` para reprodutibilidade
#   e invariância geométrica em etapas subsequentes (classificação por MLP);
# - **ADR 024:** Validação *Fail Fast* da integridade do repositório Git no Google Colab via hash de commit;
# - **Destino dos Artefatos:** Todos os resultados gerados (matriz projetada `.txt`, matriz binária `.npy`,
#   base `.rds` e exportação `.mtx`) são persistidos na pasta **`sweeps para mlp`**.

# %%
# ==============================================================================
# CONFIGURAÇÃO DO REPOSITÓRIO E VALIDAÇÃO FAIL FAST DE INTEGRIDADE GIT (COLAB)
# ==============================================================================
REPO_NAME = "pipiline_hopifield"
REPO_URL = "https://github.com/letdevx/pipiline_hopifield.git"
REPO_BRANCH = "reconstrução_Fujita_com_mlp_funcionando_corretamente"
DEST_PATH = f"/content/{REPO_NAME}"

# Commit hash esperado no Colab (ADR 024)
EXPECTED_COMMIT = "a7f1fe8"


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
# MONTAGEM DO GOOGLE DRIVE E DIRETÓRIO DE DESTINO: sweeps para mlp
# ==============================================================================
try:
    from google.colab import drive  # type: ignore

    if not os.path.ismount("/content/drive") and not os.path.exists(
        "/content/drive/MyDrive"
    ):
        print("[Colab] Montando Google Drive em /content/drive...")
        drive.mount("/content/drive")
    else:
        print("[Colab] Google Drive já montado em /content/drive.")
except (ImportError, Exception) as e:
    if "google.colab" in sys.modules:
        print(f"[Colab] Aviso durante montagem do Google Drive: {e}")

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

# ==============================================================================
# DEFINIÇÃO DO DIRETÓRIO DE RESULTADOS: "sweeps para mlp"
# ==============================================================================
NOME_PASTA_SAIDA = "sweeps para mlp"
DIR_SWEEPS_MLP = os.path.join(ROOT_DIR, NOME_PASTA_SAIDA)
os.makedirs(DIR_SWEEPS_MLP, exist_ok=True)

# Diretório espelho no Google Drive (caso o Drive esteja montado no Colab)
DIR_SWEEPS_MLP_DRIVE: str | None = None
if os.path.exists("/content/drive/MyDrive"):
    DIR_SWEEPS_MLP_DRIVE = os.path.join("/content/drive/MyDrive", NOME_PASTA_SAIDA)
    os.makedirs(DIR_SWEEPS_MLP_DRIVE, exist_ok=True)
elif os.path.exists("/content/drive/Othercomputers/Meu laptop/Documents/Letworkspace"):
    DIR_SWEEPS_MLP_DRIVE = os.path.join(
        "/content/drive/Othercomputers/Meu laptop/Documents/Letworkspace",
        REPO_NAME,
        NOME_PASTA_SAIDA,
    )
    os.makedirs(DIR_SWEEPS_MLP_DRIVE, exist_ok=True)

print(f"[Diretório de Saída Local/VM] : {DIR_SWEEPS_MLP}")
if DIR_SWEEPS_MLP_DRIVE:
    print(f"[Diretório de Saída no Drive] : {DIR_SWEEPS_MLP_DRIVE}")

# %%
# ==============================================================================
# IMPORTAÇÃO DOS MÓDULOS CIENTÍFICOS DO PIPELINE (src/)
# ==============================================================================
import anndata as ad
import numpy as np
import scipy.sparse as sp

from alinhamento import ExportadorMTX
from treinamento import ProjetorSWeePR

SEED = 42

# %%
# ==============================================================================
# RESOLUÇÃO DINÂMICA DO CAMINHO DO ARQUIVO: matrizFiltradaeNormalizadaF
# ==============================================================================
# Variável opcional para sobrescrever o caminho da matriz manualmente, se necessário
CAMINHO_MATRIZ_CUSTOMIZADO = ""


def resolver_caminho_matriz_fujita(customizado: str = "") -> str:
    """Localiza o arquivo matrizFiltradaeNormalizadaF (.h5ad) no Colab, Drive ou local."""
    if customizado and os.path.exists(customizado):
        return customizado

    candidatos = [
        # Pasta imputs no ambiente do Colab / Drive
        "/content/imputs/matrizFiltradaeNormalizadaF.h5ad",
        "/content/imputs/matrizFiltradaeNormalizadaF",
        "/content/drive/MyDrive/imputs/matrizFiltradaeNormalizadaF.h5ad",
        "/content/drive/MyDrive/imputs/matrizFiltradaeNormalizadaF",
        "/content/drive/MyDrive/Teste hop/imputs/matrizFiltradaeNormalizadaF.h5ad",
        "/content/drive/MyDrive/matrizFiltradaeNormalizadaF.h5ad",
        "/content/matrizFiltradaeNormalizadaF.h5ad",
        # Drive sincronizado com o laptop da pesquisadora
        "/content/drive/Othercomputers/Meu laptop/Documents/Letworkspace/Teste hop/imputs/matrizFiltradaeNormalizadaF.h5ad",
        "/content/drive/Othercomputers/Meu computador/Documents/Letworkspace/Teste hop/imputs/matrizFiltradaeNormalizadaF.h5ad",
        "/content/drive/Othercomputers/My Laptop/Documents/Letworkspace/Teste hop/imputs/matrizFiltradaeNormalizadaF.h5ad",
        "/content/drive/Othercomputers/Meu laptop/Documents/Letworkspace/Sweep-Harmonization/Meus_testes/Controle_qualidade/dataF/MatrizfiltradaenormalizadaF/matrizFiltradaeNormalizadaF.h5ad",
        # Caminhos relativos à raiz do repositório
        os.path.join(ROOT_DIR, "imputs", "matrizFiltradaeNormalizadaF.h5ad"),
        os.path.join(ROOT_DIR, "imputs", "matrizFiltradaeNormalizadaF"),
        # Caminhos no Windows local
        r"C:\Users\Leticia\Documents\Letworkspace\Sweep-Harmonization\Meus_testes\Controle_qualidade\dataF\MatrizfiltradaenormalizadaF\matrizFiltradaeNormalizadaF.h5ad",
        r"C:\Users\Leticia\Documents\Letworkspace\Teste hop\imputs\matrizFiltradaeNormalizadaF.h5ad",
    ]

    for cand in candidatos:
        if cand and os.path.exists(cand):
            return cand

    # Varredura inteligente com glob no Colab (/content e /content/drive)
    if os.path.exists("/content"):
        achados_content = glob.glob(
            "/content/**/matrizFiltradaeNormalizadaF*.h5ad", recursive=True
        )
        if achados_content:
            return achados_content[0]

        if os.path.exists("/content/drive"):
            achados_drive = glob.glob(
                "/content/drive/**/matrizFiltradaeNormalizadaF*.h5ad", recursive=True
            )
            if achados_drive:
                return achados_drive[0]

    # Diagnóstico didático em caso de arquivo ausente
    is_colab = os.path.exists("/content")
    drive_montado = os.path.exists("/content/drive/MyDrive") or (
        os.path.exists("/content/drive") and os.path.ismount("/content/drive")
    )
    detalhes_drive = ""
    if is_colab:
        if not drive_montado:
            detalhes_drive = (
                "\n\n[DIAGNÓSTICO COLAB] O Google Drive NÃO está montado!\n"
                "Para montar o Google Drive, execute:\n"
                "    from google.colab import drive\n"
                "    drive.mount('/content/drive')\n"
            )
        else:
            pastas_drive = (
                os.listdir("/content/drive") if os.path.exists("/content/drive") else []
            )
            detalhes_drive = (
                f"\n\n[DIAGNÓSTICO COLAB] Pastas em /content/drive: {pastas_drive}"
            )

    raise FileNotFoundError(
        "[Erro] Arquivo 'matrizFiltradaeNormalizadaF' não encontrado nos caminhos mapeados.\n"
        "Verifique se o Google Drive foi montado ou defina 'CAMINHO_MATRIZ_CUSTOMIZADO = \"/caminho/do/arquivo.h5ad\"'."
        f"{detalhes_drive}"
    )


PATH_MATRIZ_RESOLVIDO = resolver_caminho_matriz_fujita(CAMINHO_MATRIZ_CUSTOMIZADO)
print(f"[Entrada Encontrada] : {PATH_MATRIZ_RESOLVIDO}")

# Definição das rotas de saída dentro da pasta 'sweeps para mlp'
DIR_MTX_DESTINO = os.path.join(DIR_SWEEPS_MLP, "mtx_matrizFiltradaeNormalizadaF")
PATH_SWEEP_TXT = os.path.join(
    DIR_SWEEPS_MLP, "matriz_sweep600d_matrizFiltradaeNormalizadaF.txt"
)
PATH_SWEEP_NPY = os.path.join(
    DIR_SWEEPS_MLP, "matriz_sweep600d_matrizFiltradaeNormalizadaF.npy"
)
PATH_ORTHBASE_RDS = os.path.join(
    DIR_SWEEPS_MLP, "orthbase_mproj_600d_matrizFiltradaeNormalizadaF.rds"
)

# %%
# ==============================================================================
# ETAPA 1: CARGA OOM-SAFE E EXPORTAÇÃO PARA MATRIX MARKET (.mtx)
# ==============================================================================
print("\n" + "=" * 75)
print("  ETAPA 1: INSPEÇÃO DO DATASET FUJITA E EXPORTAÇÃO MATRIX MARKET (MTX)")
print("=" * 75)

# Carrega o AnnData em modo backed para inspecionar dimensões sem sobrecarregar a memória
print(f"[AnnData] Inspecionando matriz: {PATH_MATRIZ_RESOLVIDO}...")
adata_info = ad.read_h5ad(PATH_MATRIZ_RESOLVIDO, backed="r")
n_celulas, n_genes = adata_info.shape
print(f"[AnnData] Dimensões: {n_celulas:,} células × {n_genes:,} genes")
adata_info.file.close()
del adata_info

# Exporta para formato Matrix Market (10x Machine Learning) utilizando ExportadorMTX de src/alinhamento
arquivo_mtx_principal = os.path.join(DIR_MTX_DESTINO, "matrix.mtx")
FORCAR_EXPORTACAO_MTX = False

if os.path.exists(arquivo_mtx_principal) and not FORCAR_EXPORTACAO_MTX:
    print(
        f"[MTX] Matriz Matrix Market já existe em: {arquivo_mtx_principal}. Exportação ignorada."
    )
else:
    print("\n[MTX] Carregando matriz completa para exportação esparsa OOM-safe...")
    adata = ad.read_h5ad(PATH_MATRIZ_RESOLVIDO)
    print(f"[MTX] Exportando para: {DIR_MTX_DESTINO}...")
    exportador = ExportadorMTX(out_dir=DIR_MTX_DESTINO)
    exportador.exportar(matriz=adata, forcar=True, validar_regex=False)
    del adata
    print(f"[MTX] Exportação concluída com sucesso em: {DIR_MTX_DESTINO}")

# %%
# ==============================================================================
# ETAPA 2: VERIFICAÇÃO DO AMBIENTE R E INSTALAÇÃO NÃO-INTERATIVA DO rSWeeP
# ==============================================================================
print("\n" + "=" * 75)
print("  ETAPA 2: VERIFICAÇÃO E INSTALAÇÃO DO PACOTE OFICIAL rSWeeP (UFPR)")
print("=" * 75)

# Verifica se o Rscript está presente e instala rSWeeP via BiocManager se necessário
ProjetorSWeePR.verificar_e_instalar_dependencias_r()

# %%
# ==============================================================================
# ETAPA 3: PROJEÇÃO VETORIAL rSWeeP 600D (PACOTE OFICIAL UFPR)
# ==============================================================================
print("\n" + "=" * 75)
print("  ETAPA 3: PROJEÇÃO VETORIAL rSWeeP 600D (VIA Rscript)")
print("=" * 75)

tempo_inicio = time.time()

projetor_fujita = ProjetorSWeePR(
    path_matriz=DIR_MTX_DESTINO,
    path_saida=PATH_SWEEP_TXT,
    n_componentes=600,
    seed=SEED,
    path_orthbase=PATH_ORTHBASE_RDS,
    forcar_recriacao=False,
)

print("[rSWeeP] Iniciando projeção com base ortonormal congelada...")
print(f"  • Matriz de Entrada : {DIR_MTX_DESTINO}")
print(f"  • Arquivo de Saída  : {PATH_SWEEP_TXT}")
print(f"  • Base OrthBase     : {PATH_ORTHBASE_RDS}")

projetor_fujita.projetar()

tempo_decorrido = time.time() - tempo_inicio

W_sweep_fujita = projetor_fujita.Wswp
assert W_sweep_fujita is not None, "Erro: A matriz projetada retornou None!"
assert not np.isnan(W_sweep_fujita).any(), (
    "Erro: Detectados valores NaN na projeção rSWeeP!"
)

# Salva a matriz projetada em formato binário NumPy (.npy) para carga ultrarrápida no treinamento da MLP
np.save(PATH_SWEEP_NPY, W_sweep_fujita)
print(f"[NPY] Matriz binária NumPy salva em: {PATH_SWEEP_NPY}")

# %%
# ==============================================================================
# ETAPA 4: PERSISTÊNCIA SECUNDÁRIA NO GOOGLE DRIVE E AUDITORIA FINAL
# ==============================================================================
print("\n" + "=" * 75)
print("  ETAPA 4: PERSISTÊNCIA DOS ARTEFATOS E RELATÓRIO DE AUDITORIA")
print("=" * 75)

# Se houver diretório no Google Drive conectado, copia os artefatos finais para persistência definitiva
if DIR_SWEEPS_MLP_DRIVE and os.path.exists(DIR_SWEEPS_MLP_DRIVE):
    print("[Drive Sync] Copiando artefatos gerados para o Google Drive...")
    for arq in (PATH_SWEEP_TXT, PATH_SWEEP_NPY, PATH_ORTHBASE_RDS):
        if os.path.exists(arq):
            dest_drive = os.path.join(DIR_SWEEPS_MLP_DRIVE, os.path.basename(arq))
            shutil.copy2(arq, dest_drive)
            print(f"  • Copiado: {os.path.basename(arq)} -> {dest_drive}")
    print("[Drive Sync] Sincronização concluída com sucesso!")

tamanho_txt_mb = os.path.getsize(PATH_SWEEP_TXT) / (1024 * 1024)
tamanho_npy_mb = os.path.getsize(PATH_SWEEP_NPY) / (1024 * 1024)
tamanho_rds_mb = (
    os.path.getsize(PATH_ORTHBASE_RDS) / (1024 * 1024)
    if os.path.exists(PATH_ORTHBASE_RDS)
    else 0.0
)

print("\n" + "=" * 75)
print(" [rSWeeP] PROJEÇÃO 600D CONCLUÍDA COM SUCESSO!")
print("=" * 75)
print(
    f"  • Dimensões Finais : {W_sweep_fujita.shape[0]:,} células × {W_sweep_fujita.shape[1]} componentes"
)
print(
    f"  • Tempo de Execução: {tempo_decorrido:.2f} segundos ({tempo_decorrido / 60:.2f} min)"
)
print(f"  • Arquivo .txt     : {PATH_SWEEP_TXT} ({tamanho_txt_mb:.2f} MB)")
print(f"  • Arquivo .npy     : {PATH_SWEEP_NPY} ({tamanho_npy_mb:.2f} MB)")
print(f"  • Base OrthBase RDS: {PATH_ORTHBASE_RDS} ({tamanho_rds_mb:.2f} MB)")
print(f"  • Pasta de Saída   : {DIR_SWEEPS_MLP}")
if DIR_SWEEPS_MLP_DRIVE:
    print(f"  • Pasta no Drive   : {DIR_SWEEPS_MLP_DRIVE}")
print("=" * 75)
