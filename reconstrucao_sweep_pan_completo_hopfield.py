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
# # Reconstrução Contínua de Vetores SWeeP 600D com Memória Hopfield Completa do Conjunto Pan
#
# Este notebook executa o experimento de reconstrução associativa contínua de vetores **SWeeP (`600` dimensões)** utilizando **100% das células do conjunto referência Pan (`58.326 células × 600 dimensões`)** diretamente na memória da [`ModernHopfieldNetwork`](src/treinamento/hopfield.py) (sem compressão por K-Means):
#
# 1. **Teste 1 (`Pan → Pan faltantes` com colunas do Pan faltantes no Mathys = `0.5`):**
#    - Toma o dataset **Pan completo (`58.326 células × 61.541 genes`, `~61k` genes)**;
#    - Substitui em formato esparso (`LIL → CSR`) as colunas presentes no Pan e faltantes no Mathys (`tracking_genes_adicionados_mathys.csv`) pelo valor sentinela neutro `0.5`;
#    - Projeta a matriz modificada (`58.326 × 61.541`) para `600` dimensões via pacote oficial R `rSWeeP` (`ProjetorSWeePR`) com a base ortonormal canônica congelada (`orthbase_mproj_600d.rds`, `61.541 × 600`);
#    - Calibra `beta` e `n_iters` e reconstrói os vetores SWeeP na Rede de Hopfield (com máscara *Leave-One-Out* `i ≠ j` e sem máscara).
# 2. **Teste 2 (`Pan → Fujita faltantes` com colunas presentes no Pan e ausentes no Fujita = `0.5`):**
#    - Reconstrói a matriz SWeeP do conjunto Fujita (`40.913 células × 600 dimensões`), na qual as `25.065` colunas presentes nos `61.541` genes do Pan e ausentes no Fujita foram preenchidas com `0.5`, consultando a memória completa do SWeeP Pan (`58.326 × 600`).
# 3. **Validação em Dois Níveis (Antes `0.5` vs. Pós-Hopfield):**
#    - **Nível 1:** Fidelidade geométrica no espaço SWeeP 600D (`MSE`, `MAE`, `Similaridade de Cosseno` e `Correlação de Pearson`);
#    - **Nível 2:** Desempenho de classificação no modelo pré-treinado `clasificador_pan.keras` (`scaler_celular_pan.pkl`) nas classes canônicas `1` a `7`, Matrizes de Confusão e projeção PCA do espaço latente (`32D`).

# %% [markdown]
# ## 0. Configuração de Ambiente (Local Windows e Google Colab)

# %%
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _is_google_colab() -> bool:
    """Detecta se a execução atual está ocorrendo no Google Colab."""
    return (
        "google.colab" in sys.modules
        or (os.name != "nt" and os.path.exists("/content"))
        or os.environ.get("COLAB_GPU") is not None
        or os.environ.get("COLAB_RELEASE_TAG") is not None
    )


# 1. Google Colab: Montagem de Google Drive e Instalação de Dependências Faltantes
if _is_google_colab():
    print("[Colab] Execução em ambiente Google Colab detectada.")
    try:
        from google.colab import drive  # type: ignore[import-not-found]

        if not os.path.exists("/content/drive"):
            print("[Colab] Montando Google Drive em /content/drive...")
            drive.mount("/content/drive")
        else:
            print("[Colab] Google Drive já montado.")
    except Exception as err:
        print(f"[Colab] Aviso ao montar Google Drive: {err}")

    # Garante instalação de dependências essenciais ausentes no runtime padrão do Colab
    _deps_colab = [
        ("anndata", "anndata"),
        ("scanpy", "scanpy"),
        ("polars", "polars"),
        ("umap", "umap-learn"),
    ]
    _instalar: list[str] = []
    for _mod, _pkg in _deps_colab:
        try:
            __import__(_mod)
        except ImportError:
            _instalar.append(_pkg)

    if _instalar:
        print(f"[Colab] Instalando dependências ausentes: {', '.join(_instalar)}...")
        res_pip = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", *_instalar],
            capture_output=True,
            text=True,
            check=False,
        )
        if res_pip.returncode != 0:
            print(f"[Colab] Aviso durante pip install:\n{res_pip.stderr.strip()}")
        else:
            print("[Colab] Dependências instaladas com sucesso.")

# 2. Resolução do Diretório Raiz do Repositório (com fail-safe de git clone no Colab)
candidatos_raiz = [
    Path.cwd(),
    Path.cwd().parent,
    Path("/content/pipiline_hopifield"),
    Path("/content/drive/MyDrive/pipiline_hopifield"),
    Path(
        "/content/drive/Othercomputers/Meu laptop/Documents/Letworkspace/pipiline_hopifield"
    ),
    Path("/content/drive/MyDrive/Letworkspace/pipiline_hopifield"),
]

ROOT_DIR: Path | None = None
for _candidato in candidatos_raiz:
    if (_candidato / "src").is_dir():
        ROOT_DIR = _candidato.resolve()
        break

# Se estiver no Colab e o repositório ainda não existir no ambiente da VM, clona automaticamente
dest_repo = Path("/content/pipiline_hopifield")
repo_url = "https://github.com/letdevx/pipiline_hopifield.git"
branch = "teste_reconstrucao_global_sweep_pan"

if ROOT_DIR is None and _is_google_colab():
    print(
        f"[Colab] Repositório não encontrado. Clonando branch '{branch}' de '{repo_url}'..."
    )
    res_clone = subprocess.run(
        ["git", "clone", "--branch", branch, repo_url, str(dest_repo)],
        capture_output=True,
        text=True,
        check=False,
    )
    if res_clone.returncode != 0:
        print(f"[Colab] Aviso ao clonar branch '{branch}': {res_clone.stderr.strip()}")
        print("[Colab] Tentando clonagem padrão do repositório...")
        subprocess.run(["git", "clone", repo_url, str(dest_repo)], check=False)
    if (dest_repo / "src").is_dir():
        ROOT_DIR = dest_repo.resolve()
elif ROOT_DIR is not None and _is_google_colab() and (ROOT_DIR / ".git").is_dir():
    # Atualiza o repositório existente na VM com os commits mais recentes
    print(f"[Colab] Sincronizando repositório via git pull ({branch})...")
    subprocess.run(
        ["git", "-C", str(ROOT_DIR), "pull", "origin", branch],
        capture_output=True,
        text=True,
        check=False,
    )

if ROOT_DIR is None or not (ROOT_DIR / "src").is_dir():
    ROOT_DIR = Path.cwd().resolve()
    if not (ROOT_DIR / "src").is_dir():
        msg_erro = (
            "\n"
            + "=" * 80
            + "\n[ERRO DE AMBIENTE] MÓDULO 'src' NÃO ENCONTRADO\n"
            + "=" * 80
            + "\nO repositório 'pipiline_hopifield' precisa ser clonado na VM do Colab.\n"
            + "Execute em uma célula de código no Colab antes desta:\n"
            + "    !git clone -b teste_reconstrucao_global_sweep_pan https://github.com/letdevx/pipiline_hopifield.git /content/pipiline_hopifield\n"
            + "\nEm seguida, adicione os caminhos ao sys.path:\n"
            + "    import sys\n"
            + "    sys.path.insert(0, '/content/pipiline_hopifield')\n"
            + "    sys.path.insert(0, '/content/pipiline_hopifield/src')\n"
            + "=" * 80
        )
        raise RuntimeError(msg_erro)

SRC_DIR = ROOT_DIR / "src"
for _p in (str(ROOT_DIR), str(SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Importação de dependências científicas
import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402
import tensorflow as tf  # noqa: E402
import torch  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay  # noqa: E402

from src.config import OUTPUTS, PATH_ORTHBASE_RDS  # noqa: E402
from src.treinamento import (  # noqa: E402
    NOMES_CLASSES_PADRAO,
    ModernHopfieldNetwork,
    avaliar_no_classificador_pan,
    calcular_metricas_geometricas_600d,
    calibrar_hopfield_sweep_completo,
    carregar_matriz_sweep_600d,
    gerar_sweep_pan_com_faltantes_mathys,
)

DIR_SAIDA = Path(OUTPUTS) / "reconstrucao_sweep_pan_completo"
DIR_SAIDA.mkdir(parents=True, exist_ok=True)

DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Diretório raiz do projeto : {ROOT_DIR}")
print(f"Diretório de saídas       : {DIR_SAIDA}")
print(f"Base ortonormal rSWeeP    : {PATH_ORTHBASE_RDS}")
print(f"Dispositivo de computação : {DISPOSITIVO}")
if DISPOSITIVO.type == "cuda":
    _nome_gpu = torch.cuda.get_device_name(0)
    _vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU / VRAM disponível     : {_nome_gpu} ({_vram_gb:.2f} GB)")
else:
    print("GPU / VRAM disponível     : Nenhuma (executando em CPU / RAM)")


# %%
def resolver_caminho_existente(candidatos: list[Path], descricao: str) -> Path:
    """Retorna o primeiro caminho existente na lista de candidatos ou o primeiro como padrão."""
    for caminho in candidatos:
        if caminho.exists():
            print(f"[OK] {descricao}: {caminho}")
            return caminho
    candidato_padrao = candidatos[0]
    msg = f"[AVISO] {descricao} não encontrado em nenhum dos {len(candidatos)} locais candidatos."
    if _is_google_colab():
        msg += f" (Padrão: {candidato_padrao} | No Colab, envie para /content/ ou /content/drive/MyDrive/)"
    else:
        msg += f" (Padrão: {candidato_padrao})"
    print(msg)
    return candidato_padrao


DIR_DOWNLOADS = Path.home() / "Downloads"
DIR_LETWORKSPACE = ROOT_DIR.parent
candidatos_teste_hop = [
    Path("/content/drive/Othercomputers/Meu laptop/Documents/Letworkspace/Teste hop"),
    Path("/content/drive/MyDrive/Teste hop"),
    Path("/content/drive/MyDrive/Letworkspace/Teste hop"),
    ROOT_DIR.parent / "Teste hop",
    Path("/content/Teste hop"),
]
DIR_DRIVE_TESTE_HOP = next(
    (p for p in candidatos_teste_hop if p.exists()),
    candidatos_teste_hop[0],
)

# 1. SWeeP Pan Completo de Referência (58.326 × 600) e Rótulos do Pan
PATH_SWEEP_PAN_REF = resolver_caminho_existente(
    [
        DIR_DOWNLOADS / "matriz_reduzida_sweepREF.txt",
        Path(
            "/content/drive/Othercomputers/Meu laptop/Downloads/matriz_reduzida_sweepREF.txt"
        ),
        Path("/content/drive/MyDrive/Downloads/matriz_reduzida_sweepREF.txt"),
        Path("/content/drive/MyDrive/matriz_reduzida_sweepREF.txt"),
        Path("/content/matriz_reduzida_sweepREF.txt"),
        ROOT_DIR / "outputs" / "treinamento" / "matriz_reduzida_sweepREF.txt",
        DIR_DRIVE_TESTE_HOP
        / "outputs"
        / "treinamento"
        / "matriz_reduzida_sweepREF.txt",
        DIR_DRIVE_TESTE_HOP / "imputs" / "matriz_reduzida_sweepREF.txt",
        ROOT_DIR / "imputs" / "matriz_reduzida_sweepREF.txt",
    ],
    "SWeeP Pan Completo (Referência 58.326 × 600)",
)

PATH_ROTULOS_PAN = resolver_caminho_existente(
    [
        ROOT_DIR / "imputs" / "PanNumerico.csv",
        DIR_LETWORKSPACE
        / "Treinamento-MLP-Mathys"
        / "Inputs"
        / "rotulos"
        / "PanNumerico.csv",
        Path(
            "/content/drive/Othercomputers/Meu laptop/Documents/Letworkspace/Treinamento-MLP-Mathys/Inputs/rotulos/PanNumerico.csv"
        ),
        Path(
            "/content/drive/MyDrive/Treinamento-MLP-Mathys/Inputs/rotulos/PanNumerico.csv"
        ),
        Path("/content/drive/MyDrive/PanNumerico.csv"),
        Path("/content/PanNumerico.csv"),
        DIR_DRIVE_TESTE_HOP / "imputs" / "PanNumerico.csv",
    ],
    "Rótulos Numéricos do Conjunto Pan",
)

# 2. Arquivos do Teste 1 (Dataset Pan ~61k genes + Colunas faltantes no Mathys)
PATH_PAN_H5AD = resolver_caminho_existente(
    [
        ROOT_DIR / "imputs" / "pan_anotado.h5ad",
        DIR_DRIVE_TESTE_HOP / "imputs" / "PAN_Bin" / "pan_anotado.h5ad",
        Path(
            "/content/drive/Othercomputers/Meu laptop/Documents/Letworkspace/Teste hop/imputs/PAN_Bin/pan_anotado.h5ad"
        ),
        Path("/content/drive/MyDrive/pan_anotado.h5ad"),
        Path("/content/pan_anotado.h5ad"),
    ],
    "Dataset Pan Completo (~61k genes - pan_anotado.h5ad)",
)

PATH_TRACKING_MATHYS = resolver_caminho_existente(
    [
        ROOT_DIR / "outputs" / "alinhamento" / "tracking_genes_adicionados_mathys.csv",
        DIR_DRIVE_TESTE_HOP
        / "outputs"
        / "alinhamento"
        / "tracking_genes_adicionados_mathys.csv",
        Path("/content/drive/MyDrive/tracking_genes_adicionados_mathys.csv"),
        Path("/content/tracking_genes_adicionados_mathys.csv"),
    ],
    "Tracking de Colunas do Pan Faltantes no Mathys",
)

# 3. Arquivos do Teste 2 (SWeeP Fujita com Faltantes do Pan = 0.5 e Fujita Original)
PATH_SWEEP_FUJITA_FALTANTES = resolver_caminho_existente(
    [
        DIR_DOWNLOADS / "matriz_sweep600d_com_sentinela.txt",
        DIR_DOWNLOADS / "matriz_reduzida_sweepALVO_sentinela (1).txt",
        DIR_DOWNLOADS / "matriz_sweep600d_com_sentinela (1).txt",
        Path(
            "/content/drive/Othercomputers/Meu laptop/Downloads/matriz_sweep600d_com_sentinela.txt"
        ),
        Path(
            "/content/drive/Othercomputers/Meu laptop/Downloads/matriz_reduzida_sweepALVO_sentinela (1).txt"
        ),
        Path(
            "/content/drive/Othercomputers/Meu laptop/Downloads/matriz_sweep600d_com_sentinela (1).txt"
        ),
        DIR_LETWORKSPACE
        / "Teste_MlP_fujita_para_fujitafaltante_fujitapos_imputacao"
        / "matriz_swee_fujita_600d_com_sentinela.txt",
        Path(
            "/content/drive/Othercomputers/Meu laptop/Documents/Letworkspace/Teste_MlP_fujita_para_fujitafaltante_fujitapos_imputacao/matriz_swee_fujita_600d_com_sentinela.txt"
        ),
        Path("/content/drive/MyDrive/matriz_sweep600d_com_sentinela.txt"),
        Path("/content/drive/MyDrive/matriz_reduzida_sweepALVO_sentinela (1).txt"),
        Path("/content/drive/MyDrive/Downloads/matriz_sweep600d_com_sentinela.txt"),
        Path("/content/matriz_sweep600d_com_sentinela.txt"),
        Path("/content/matriz_reduzida_sweepALVO_sentinela (1).txt"),
        Path("/content/matriz_swee_fujita_600d_com_sentinela.txt"),
        ROOT_DIR / "outputs" / "treinamento" / "matriz_sweep600d_com_sentinela.txt",
        ROOT_DIR / "imputs" / "matriz_sweep600d_com_sentinela.txt",
        DIR_DRIVE_TESTE_HOP / "imputs" / "matriz_sweep600d_com_sentinela.txt",
    ],
    "SWeeP Fujita com Colunas Faltantes do Pan = 0.5 (40.913 × 600)",
)

PATH_SWEEP_FUJITA_ORIGINAL = resolver_caminho_existente(
    [
        DIR_DOWNLOADS / "matriz_reduzida_sweepF.csv",
        Path(
            "/content/drive/Othercomputers/Meu laptop/Downloads/matriz_reduzida_sweepF.csv"
        ),
        Path("/content/drive/MyDrive/Downloads/matriz_reduzida_sweepF.csv"),
        Path("/content/drive/MyDrive/matriz_reduzida_sweepF.csv"),
        DIR_LETWORKSPACE
        / "Teste_MlP_fujita_para_fujitafaltante_fujitapos_imputacao"
        / "matriz_sweep600d_matrizFiltradaeNormalizadaF.txt",
        Path(
            "/content/drive/Othercomputers/Meu laptop/Documents/Letworkspace/Teste_MlP_fujita_para_fujitafaltante_fujitapos_imputacao/matriz_sweep600d_matrizFiltradaeNormalizadaF.txt"
        ),
        Path("/content/matriz_reduzida_sweepF.csv"),
        Path("/content/matriz_sweep600d_matrizFiltradaeNormalizadaF.txt"),
        ROOT_DIR / "imputs" / "matriz_reduzida_sweepF.csv",
        DIR_DRIVE_TESTE_HOP / "imputs" / "matriz_reduzida_sweepF.csv",
    ],
    "SWeeP Fujita Original Sem 0.5 (40.913 × 600)",
)

PATH_ROTULOS_FUJITA = resolver_caminho_existente(
    [
        DIR_LETWORKSPACE
        / "Treinamento-MLP-Mathys"
        / "Inputs"
        / "rotulos"
        / "cell_types_numerico_fujita.txt",
        Path(
            "/content/drive/Othercomputers/Meu laptop/Documents/Letworkspace/Treinamento-MLP-Mathys/Inputs/rotulos/cell_types_numerico_fujita.txt"
        ),
        DIR_LETWORKSPACE
        / "Teste_MlP_fujita_para_fujitafaltante_fujitapos_imputacao"
        / "cell_types_numerico_fujita.txt",
        Path(
            "/content/drive/Othercomputers/Meu laptop/Documents/Letworkspace/Teste_MlP_fujita_para_fujitafaltante_fujitapos_imputacao/cell_types_numerico_fujita.txt"
        ),
        Path(
            "/content/drive/MyDrive/Treinamento-MLP-Mathys/Inputs/rotulos/cell_types_numerico_fujita.txt"
        ),
        Path("/content/drive/MyDrive/cell_types_numerico_fujita.txt"),
        Path("/content/cell_types_numerico_fujita.txt"),
        ROOT_DIR / "imputs" / "cell_types_numerico_fujita.txt",
        DIR_DRIVE_TESTE_HOP / "imputs" / "cell_types_numerico_fujita.txt",
    ],
    "Rótulos Numéricos do Conjunto Fujita",
)

# 4. Classificador MLP Pré-Treinado no Conjunto Pan e Scaler
PATH_MODELO_KERAS = resolver_caminho_existente(
    [
        ROOT_DIR / "modelos" / "clasificador_pan.keras",
        DIR_LETWORKSPACE
        / "Treinamento-MLP-Mathys"
        / "teste_pan_ausentes"
        / "clasificador_pan.keras",
        DIR_LETWORKSPACE
        / "Treinamento-MLP-Mathys"
        / "teste_PAN_para_Fujita"
        / "clasificador_pan.keras",
        Path(
            "/content/drive/Othercomputers/Meu laptop/Documents/Letworkspace/Treinamento-MLP-Mathys/teste_pan_ausentes/clasificador_pan.keras"
        ),
        Path(
            "/content/drive/Othercomputers/Meu laptop/Documents/Letworkspace/Treinamento-MLP-Mathys/teste_PAN_para_Fujita/clasificador_pan.keras"
        ),
        Path(
            "/content/drive/MyDrive/Treinamento-MLP-Mathys/teste_pan_ausentes/clasificador_pan.keras"
        ),
        Path("/content/drive/MyDrive/clasificador_pan.keras"),
        DIR_DRIVE_TESTE_HOP / "imputs" / "clasificador_pan.keras",
        ROOT_DIR / "imputs" / "clasificador_pan.keras",
    ],
    "Classificador MLP Pré-Treinado (clasificador_pan.keras)",
)

PATH_SCALER_PKL = resolver_caminho_existente(
    [
        ROOT_DIR / "modelos" / "scaler_celular_pan.pkl",
        DIR_LETWORKSPACE
        / "Treinamento-MLP-Mathys"
        / "teste_pan_ausentes"
        / "scaler_celular_pan.pkl",
        DIR_LETWORKSPACE
        / "Treinamento-MLP-Mathys"
        / "teste_PAN_para_Fujita"
        / "scaler_celular_pan.pkl",
        Path(
            "/content/drive/Othercomputers/Meu laptop/Documents/Letworkspace/Treinamento-MLP-Mathys/teste_pan_ausentes/scaler_celular_pan.pkl"
        ),
        Path(
            "/content/drive/Othercomputers/Meu laptop/Documents/Letworkspace/Treinamento-MLP-Mathys/teste_PAN_para_Fujita/scaler_celular_pan.pkl"
        ),
        Path(
            "/content/drive/MyDrive/Treinamento-MLP-Mathys/teste_pan_ausentes/scaler_celular_pan.pkl"
        ),
        Path("/content/drive/MyDrive/scaler_celular_pan.pkl"),
        DIR_DRIVE_TESTE_HOP / "imputs" / "scaler_celular_pan.pkl",
        ROOT_DIR / "imputs" / "scaler_celular_pan.pkl",
    ],
    "Scaler do Classificador Pan (scaler_celular_pan.pkl)",
)

# %% [markdown]
# ## 1. Carregamento do SWeeP Pan Completo e Memória Hopfield (`58.326 × 600`, Sem K-Means)
#
# Diferentemente da abordagem baseada em subclusters K-Means, armazenamos **todas as `58.326` células** e **todas as `600` dimensões SWeeP** do conjunto Pan diretamente na memória associativa da `ModernHopfieldNetwork` com `binary=False` e `normalize=True`.

# %%
sweep_pan_ref = carregar_matriz_sweep_600d(PATH_SWEEP_PAN_REF)
rotulos_pan = (
    pl.read_csv(str(PATH_ROTULOS_PAN), has_header=False)
    .to_numpy()
    .ravel()
    .astype(np.int32)
)

print(f"SWeeP Pan Referência : {sweep_pan_ref.shape} (dtype={sweep_pan_ref.dtype})")
print(
    f"Rótulos Pan          : {rotulos_pan.shape} | Classes únicas: {np.unique(rotulos_pan)}"
)

if not PATH_MODELO_KERAS.exists() or not PATH_SCALER_PKL.exists():
    raise FileNotFoundError(
        f"Classificador ({PATH_MODELO_KERAS}) ou scaler ({PATH_SCALER_PKL}) não encontrado!\n"
        "Execute no Colab: !git -C /content/pipiline_hopifield pull origin teste_reconstrucao_global_sweep_pan"
    )

modelo_mlp_pan = tf.keras.models.load_model(str(PATH_MODELO_KERAS))
scaler_pan = joblib.load(str(PATH_SCALER_PKL))

aval_pan_ref_teto = avaliar_no_classificador_pan(
    sweep_600d=sweep_pan_ref,
    rotulos=rotulos_pan,
    modelo_keras=modelo_mlp_pan,
    scaler=scaler_pan,
    extrair_latente_32d=False,
)
print(
    f"[Teto de Referência — SWeeP Pan Completo Original] "
    f"Acurácia MLP (Classes 1..7): {aval_pan_ref_teto['acuracia'] * 100:.2f}% | "
    f"F1-Macro: {aval_pan_ref_teto['f1_macro'] * 100:.2f}%"
)

# %% [markdown]
# ## 2. Preparação do Teste 1 (`Pan → Pan faltantes` com Colunas Faltantes do Mathys = `0.5`)
#
# Nesta etapa:
# 1. Carregamos o dataset **Pan completo (`pan_anotado.h5ad`, `58.326 células × 61.541 genes`, `~61k` genes)** e o mapeamento de colunas presentes no Pan e ausentes no Mathys (`tracking_genes_adicionados_mathys.csv`);
# 2. Substituímos no dataset Pan essas colunas faltantes no Mathys por `0.5` em formato esparso (`LIL → CSR`, sem materialização densa);
# 3. Projetamos a matriz modificada (`58.326 × 61.541`) via pacote R oficial `rSWeeP` (`ProjetorSWeePR`) com a base ortonormal congelada `orthbase_mproj_600d.rds` (`61.541 × 600`).
#
# > **Nota de Cache:** Caso `matriz_sweep_pan_faltantes_mathys.npy`/`.txt` já tenha sido gerado em `DIR_SAIDA` (ou em `outputs/projecao_ausentes_pan/`), ele é carregado diretamente sem necessidade de reexecutar o R, a menos que `FORCAR_REPROJECAO_PAN_MATHYS = True`.

# %%
FORCAR_REPROJECAO_PAN_MATHYS = False

# Verifica se já existe projeção salva em outputs/projecao_ausentes_pan ou em DIR_SAIDA
PATH_CACHE_AUSENTES_PAN = (
    ROOT_DIR / "outputs" / "projecao_ausentes_pan" / "matriz_sweep_ausentes.txt"
)

if (
    not FORCAR_REPROJECAO_PAN_MATHYS
    and not (DIR_SAIDA / "matriz_sweep_pan_faltantes_mathys.npy").exists()
    and not (DIR_SAIDA / "matriz_sweep_pan_faltantes_mathys.txt").exists()
    and PATH_CACHE_AUSENTES_PAN.exists()
):
    print(f"[Cache] Carregando projeção prévia de {PATH_CACHE_AUSENTES_PAN}...")
    sweep_pan_faltantes_mathys = carregar_matriz_sweep_600d(PATH_CACHE_AUSENTES_PAN)
elif (
    (PATH_PAN_H5AD.exists() and PATH_TRACKING_MATHYS.exists())
    or (DIR_SAIDA / "matriz_sweep_pan_faltantes_mathys.npy").exists()
    or (DIR_SAIDA / "matriz_sweep_pan_faltantes_mathys.txt").exists()
):
    sweep_pan_faltantes_mathys = gerar_sweep_pan_com_faltantes_mathys(
        path_pan_h5ad=PATH_PAN_H5AD,
        path_tracking_mathys_csv=PATH_TRACKING_MATHYS,
        dir_saida=DIR_SAIDA,
        path_orthbase_rds=PATH_ORTHBASE_RDS,
        n_componentes=600,
        seed=42,
        fill_value=0.5,
        forcar_reprojecao=FORCAR_REPROJECAO_PAN_MATHYS,
    )
else:
    # Caso executado em máquina local sem o .h5ad de 61k genes montado, usa o SWeeP Pan c/ sentinela disponível em Downloads ou Drive para validação do fluxo
    PATH_FALLBACK_SENTINELA_PAN = resolver_caminho_existente(
        [
            DIR_DOWNLOADS / "matriz_reduzida_sweepALVO_sentinela.txt",
            DIR_DOWNLOADS / "matriz_sweep600d_pan_com_sentinela_direto.txt",
            Path(
                "/content/drive/Othercomputers/Meu laptop/Downloads/matriz_reduzida_sweepALVO_sentinela.txt"
            ),
            Path(
                "/content/drive/Othercomputers/Meu laptop/Downloads/matriz_sweep600d_pan_com_sentinela_direto.txt"
            ),
            Path("/content/drive/MyDrive/matriz_reduzida_sweepALVO_sentinela.txt"),
            Path(
                "/content/drive/MyDrive/matriz_sweep600d_pan_com_sentinela_direto.txt"
            ),
            Path("/content/matriz_reduzida_sweepALVO_sentinela.txt"),
            Path("/content/matriz_sweep600d_pan_com_sentinela_direto.txt"),
        ],
        "SWeeP Pan com Sentinela 0.5 pré-projetado",
    )
    sweep_pan_faltantes_mathys = carregar_matriz_sweep_600d(PATH_FALLBACK_SENTINELA_PAN)

print(
    f"SWeeP Pan c/ Faltantes = 0.5 : {sweep_pan_faltantes_mathys.shape} "
    f"(dtype={sweep_pan_faltantes_mathys.dtype})"
)

# %% [markdown]
# ## 3. Calibração de `beta` e `n_iters` + Reconstrução do Teste 1 (`Pan → Pan faltantes`)
#
# Executamos um Grid Search em lotes (`batch_size = 1024`) sobre `beta ∈ [15, 30, 50, 100, 200, 500]` e `n_iters ∈ [1, 2, 3]` aplicando **máscara diagonal *Leave-One-Out* (`mask_self=True`, `i ≠ j`)**, de modo que cada célula `i` do Pan com colunas faltantes preenchidas por `0.5` seja reconstruída exclusivamente a partir das outras `58.325` células da memória.

# %%
BETAS_GRID = (15.0, 30.0, 50.0, 100.0, 200.0, 500.0)
N_ITERS_GRID = (1, 2, 3)
BATCH_SIZE_HOPFIELD = 1024

df_grid_pan, melhor_cfg_pan = calibrar_hopfield_sweep_completo(
    sweep_referencia=sweep_pan_ref,
    sweep_faltante=sweep_pan_faltantes_mathys,
    rotulos_referencia=rotulos_pan,
    betas=BETAS_GRID,
    n_iters_list=N_ITERS_GRID,
    batch_size=BATCH_SIZE_HOPFIELD,
    mask_self=True,
    modelo_mlp=modelo_mlp_pan,
    scaler_mlp=scaler_pan,
    device=DISPOSITIVO,
)

df_grid_pan.to_csv(DIR_SAIDA / "grid_search_calibracao_pan_faltantes.csv", index=False)
print(
    "\n=== Resultado do Grid Search (Teste 1: Pan -> Pan Faltantes c/ Máscara i != j) ==="
)
print(df_grid_pan.to_string(index=False))

BETA_OTIMO = float(melhor_cfg_pan["beta"])
N_ITERS_OTIMO = int(melhor_cfg_pan["n_iters"])
print(
    f"\nMelhor configuração selecionada: beta={BETA_OTIMO}, n_iters={N_ITERS_OTIMO} | "
    f"Cosseno: {melhor_cfg_pan['cosseno_antes']:.4f} -> {melhor_cfg_pan['cosseno_medio']:.4f} | "
    f"Acurácia MLP: {melhor_cfg_pan['acuracia_antes'] * 100:.2f}% -> {melhor_cfg_pan['acuracia_mlp'] * 100:.2f}%"
)

# %% [markdown]
# ### 3.1. Reconstrução Final do Teste 1 (Com Máscara `i ≠ j` e Sem Máscara) e Avaliação Completa

# %%
rede_hopfield_pan = ModernHopfieldNetwork(
    beta=BETA_OTIMO,
    n_iters=N_ITERS_OTIMO,
    binary=False,
    normalize=True,
    device=DISPOSITIVO,
)
rede_hopfield_pan.store(sweep_pan_ref)

# Modo A: Com máscara Leave-One-Out (i != j)
sweep_pan_rec_loo = rede_hopfield_pan.retrieve(
    sweep_pan_faltantes_mathys,
    batch_size=BATCH_SIZE_HOPFIELD,
    mask_self=True,
)
diag_loo = dict(rede_hopfield_pan.ultimo_diagnostico_atencao)

# Modo B: Sem máscara (memória completa incluindo a própria célula i)
sweep_pan_rec_full = rede_hopfield_pan.retrieve(
    sweep_pan_faltantes_mathys,
    batch_size=BATCH_SIZE_HOPFIELD,
    mask_self=False,
)
diag_full = dict(rede_hopfield_pan.ultimo_diagnostico_atencao)

# Métricas Geométricas 600D (Nível 1)
met_pan_antes, df_cls_pan_antes = calcular_metricas_geometricas_600d(
    sweep_pan_ref, sweep_pan_faltantes_mathys, rotulos=rotulos_pan
)
met_pan_loo, df_cls_pan_loo = calcular_metricas_geometricas_600d(
    sweep_pan_ref, sweep_pan_rec_loo, rotulos=rotulos_pan
)
met_pan_full, df_cls_pan_full = calcular_metricas_geometricas_600d(
    sweep_pan_ref, sweep_pan_rec_full, rotulos=rotulos_pan
)

# Avaliação no Classificador MLP (Nível 2)
aval_pan_antes = avaliar_no_classificador_pan(
    sweep_pan_faltantes_mathys,
    rotulos_pan,
    modelo_mlp_pan,
    scaler_pan,
    extrair_latente_32d=False,
)
aval_pan_loo = avaliar_no_classificador_pan(
    sweep_pan_rec_loo,
    rotulos_pan,
    modelo_mlp_pan,
    scaler_pan,
    extrair_latente_32d=False,
)
aval_pan_full = avaliar_no_classificador_pan(
    sweep_pan_rec_full,
    rotulos_pan,
    modelo_mlp_pan,
    scaler_pan,
    extrair_latente_32d=False,
)

df_resumo_teste1 = pd.DataFrame(
    [
        {
            "cenario": "1. Referência Pan Original (Sem 0.5)",
            "mse_600d": 0.0,
            "mae_600d": 0.0,
            "cosseno_medio": 1.0,
            "pearson_medio": 1.0,
            "acuracia_mlp_pct": aval_pan_ref_teto["acuracia"] * 100.0,
            "f1_macro_mlp_pct": aval_pan_ref_teto["f1_macro"] * 100.0,
        },
        {
            "cenario": "2. Antes da Hopfield (Pan c/ Faltantes = 0.5)",
            "mse_600d": met_pan_antes["mse"],
            "mae_600d": met_pan_antes["mae"],
            "cosseno_medio": met_pan_antes["cosseno_medio"],
            "pearson_medio": met_pan_antes["pearson_medio"],
            "acuracia_mlp_pct": aval_pan_antes["acuracia"] * 100.0,
            "f1_macro_mlp_pct": aval_pan_antes["f1_macro"] * 100.0,
        },
        {
            "cenario": f"3. Pós-Hopfield LOO (i != j, beta={BETA_OTIMO:.0f}, it={N_ITERS_OTIMO})",
            "mse_600d": met_pan_loo["mse"],
            "mae_600d": met_pan_loo["mae"],
            "cosseno_medio": met_pan_loo["cosseno_medio"],
            "pearson_medio": met_pan_loo["pearson_medio"],
            "acuracia_mlp_pct": aval_pan_loo["acuracia"] * 100.0,
            "f1_macro_mlp_pct": aval_pan_loo["f1_macro"] * 100.0,
        },
        {
            "cenario": f"4. Pós-Hopfield Sem Máscara (beta={BETA_OTIMO:.0f}, it={N_ITERS_OTIMO})",
            "mse_600d": met_pan_full["mse"],
            "mae_600d": met_pan_full["mae"],
            "cosseno_medio": met_pan_full["cosseno_medio"],
            "pearson_medio": met_pan_full["pearson_medio"],
            "acuracia_mlp_pct": aval_pan_full["acuracia"] * 100.0,
            "f1_macro_mlp_pct": aval_pan_full["f1_macro"] * 100.0,
        },
    ]
)
df_resumo_teste1.to_csv(DIR_SAIDA / "resumo_teste1_pan_faltantes.csv", index=False)
print("\n=== Resumo Consolidado — Teste 1 (Pan -> Pan Faltantes) ===")
print(df_resumo_teste1.to_string(index=False))

# %%
# Plot das Matrizes de Confusão do Teste 1 (Antes 0.5 vs. Pós-Hopfield LOO vs. Pós-Hopfield Full)
rotulos_curtos = [f"C{i}" for i in range(1, 8)]
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for ax, aval, titulo in zip(
    axes,
    [aval_pan_antes, aval_pan_loo, aval_pan_full],
    [
        f"Antes Hopfield (0.5)\nAcc: {aval_pan_antes['acuracia'] * 100:.2f}% | F1: {aval_pan_antes['f1_macro'] * 100:.2f}%",
        f"Pós-Hopfield LOO (i != j)\nAcc: {aval_pan_loo['acuracia'] * 100:.2f}% | F1: {aval_pan_loo['f1_macro'] * 100:.2f}%",
        f"Pós-Hopfield Sem Máscara\nAcc: {aval_pan_full['acuracia'] * 100:.2f}% | F1: {aval_pan_full['f1_macro'] * 100:.2f}%",
    ],
    strict=True,
):
    ConfusionMatrixDisplay(
        confusion_matrix=aval["matriz_confusao"],
        display_labels=rotulos_curtos,
    ).plot(cmap="Blues", ax=ax, values_format="d", colorbar=False)
    ax.set_title(titulo, fontsize=11)
    ax.grid(False)

plt.suptitle(
    "Teste 1 (Pan -> Pan Faltantes): Matrizes de Confusão no Classificador MLP",
    fontsize=14,
)
plt.tight_layout()
plt.savefig(DIR_SAIDA / "matriz_confusao_teste1_pan_faltantes.png", dpi=180)
plt.show()

# %% [markdown]
# ## 4. Teste 2 (`Pan → Fujita faltantes` com Colunas Presentes no Pan e Ausentes no Fujita = `0.5`)
#
# Agora utilizamos a `ModernHopfieldNetwork` abastecida com **100% do SWeeP Pan (`58.326 × 600`)** para reconstruir a matriz SWeeP do conjunto **Fujita (`40.913 × 600`)** projetada com sentinela `0.5` nas `25.065` colunas presentes nos `61.541` genes do Pan e ausentes no Fujita.

# %%
if not PATH_SWEEP_FUJITA_FALTANTES.exists():
    raise FileNotFoundError(
        f"\n[ERRO TESTE 2] Arquivo SWeeP Fujita com sentinela 0.5 não encontrado!\n"
        f"Caminho verificado: {PATH_SWEEP_FUJITA_FALTANTES}\n\n"
        f"Como resolver no Google Colab:\n"
        f"  1. Opção A (Upload rápido na VM): Faça upload de 'matriz_sweep600d_com_sentinela.txt' para '/content/' na barra lateral do Colab; OU\n"
        f"  2. Opção B (Google Drive): Copie 'matriz_sweep600d_com_sentinela.txt' para o Google Drive em '/content/drive/MyDrive/' ou sincronize a pasta Downloads do laptop."
    )

if not PATH_SWEEP_FUJITA_ORIGINAL.exists():
    raise FileNotFoundError(
        f"\n[ERRO TESTE 2] Arquivo SWeeP Fujita Original (sem 0.5) não encontrado!\n"
        f"Caminho verificado: {PATH_SWEEP_FUJITA_ORIGINAL}\n\n"
        f"Como resolver no Google Colab:\n"
        f"  1. Opção A (Upload rápido na VM): Faça upload de 'matriz_reduzida_sweepF.csv' para '/content/' na barra lateral do Colab; OU\n"
        f"  2. Opção B (Google Drive): Copie para '/content/drive/MyDrive/matriz_reduzida_sweepF.csv'."
    )

if not PATH_ROTULOS_FUJITA.exists():
    raise FileNotFoundError(
        f"\n[ERRO TESTE 2] Rótulos celulares do conjunto Fujita não encontrados!\n"
        f"Caminho verificado: {PATH_ROTULOS_FUJITA}\n\n"
        f"Como resolver no Google Colab:\n"
        f"  Faça upload de 'cell_types_numerico_fujita.txt' para '/content/' ou copie para '/content/drive/MyDrive/'."
    )

sweep_fujita_faltantes = carregar_matriz_sweep_600d(PATH_SWEEP_FUJITA_FALTANTES)
sweep_fujita_original = carregar_matriz_sweep_600d(PATH_SWEEP_FUJITA_ORIGINAL)
rotulos_fujita = (
    pl.read_csv(str(PATH_ROTULOS_FUJITA), has_header=False)
    .to_numpy()
    .ravel()
    .astype(np.int32)
)

print(f"SWeeP Fujita c/ Faltantes = 0.5 : {sweep_fujita_faltantes.shape}")
print(f"SWeeP Fujita Original Sem 0.5   : {sweep_fujita_original.shape}")
print(f"Rótulos Fujita                  : {rotulos_fujita.shape}")

# Recuperação associativa cruzada (Pan -> Fujita)
sweep_fujita_reconstruido = rede_hopfield_pan.retrieve(
    sweep_fujita_faltantes,
    batch_size=BATCH_SIZE_HOPFIELD,
    mask_self=False,
)
diag_fujita = dict(rede_hopfield_pan.ultimo_diagnostico_atencao)
print(f"Diagnóstico de Atenção (Pan -> Fujita): {diag_fujita}")

# Avaliação Nível 1: Fidelidade Geométrica 600D (contra SWeeP Fujita Original sem 0.5)
met_fuj_antes, df_cls_fuj_antes = calcular_metricas_geometricas_600d(
    sweep_fujita_original, sweep_fujita_faltantes, rotulos=rotulos_fujita
)
met_fuj_pos, df_cls_fuj_pos = calcular_metricas_geometricas_600d(
    sweep_fujita_original, sweep_fujita_reconstruido, rotulos=rotulos_fujita
)

# Avaliação Nível 2: Classificador MLP do Pan (Classes 1 a 7) + Extração do Espaço Latente 32D
aval_fuj_orig = avaliar_no_classificador_pan(
    sweep_fujita_original,
    rotulos_fujita,
    modelo_mlp_pan,
    scaler_pan,
    extrair_latente_32d=True,
)
aval_fuj_antes = avaliar_no_classificador_pan(
    sweep_fujita_faltantes,
    rotulos_fujita,
    modelo_mlp_pan,
    scaler_pan,
    extrair_latente_32d=True,
)
aval_fuj_pos = avaliar_no_classificador_pan(
    sweep_fujita_reconstruido,
    rotulos_fujita,
    modelo_mlp_pan,
    scaler_pan,
    extrair_latente_32d=True,
)

df_resumo_teste2 = pd.DataFrame(
    [
        {
            "cenario": "1. SWeeP Fujita Original (Sem 0.5)",
            "mse_vs_fujita_orig": 0.0,
            "mae_vs_fujita_orig": 0.0,
            "cosseno_vs_fujita_orig": 1.0,
            "pearson_vs_fujita_orig": 1.0,
            "acuracia_mlp_pct": aval_fuj_orig["acuracia"] * 100.0,
            "f1_macro_mlp_pct": aval_fuj_orig["f1_macro"] * 100.0,
            "f1_weighted_mlp_pct": aval_fuj_orig["f1_weighted"] * 100.0,
        },
        {
            "cenario": "2. Antes da Hopfield (Fujita c/ 25.065 Faltantes do Pan = 0.5)",
            "mse_vs_fujita_orig": met_fuj_antes["mse"],
            "mae_vs_fujita_orig": met_fuj_antes["mae"],
            "cosseno_vs_fujita_orig": met_fuj_antes["cosseno_medio"],
            "pearson_vs_fujita_orig": met_fuj_antes["pearson_medio"],
            "acuracia_mlp_pct": aval_fuj_antes["acuracia"] * 100.0,
            "f1_macro_mlp_pct": aval_fuj_antes["f1_macro"] * 100.0,
            "f1_weighted_mlp_pct": aval_fuj_antes["f1_weighted"] * 100.0,
        },
        {
            "cenario": f"3. Pós-Hopfield (Memória Pan Completa, beta={BETA_OTIMO:.0f}, it={N_ITERS_OTIMO})",
            "mse_vs_fujita_orig": met_fuj_pos["mse"],
            "mae_vs_fujita_orig": met_fuj_pos["mae"],
            "cosseno_vs_fujita_orig": met_fuj_pos["cosseno_medio"],
            "pearson_vs_fujita_orig": met_fuj_pos["pearson_medio"],
            "acuracia_mlp_pct": aval_fuj_pos["acuracia"] * 100.0,
            "f1_macro_mlp_pct": aval_fuj_pos["f1_macro"] * 100.0,
            "f1_weighted_mlp_pct": aval_fuj_pos["f1_weighted"] * 100.0,
        },
    ]
)
df_resumo_teste2.to_csv(DIR_SAIDA / "resumo_teste2_fujita_faltantes.csv", index=False)
print("\n=== Resumo Consolidado — Teste 2 (Pan -> Fujita Faltantes) ===")
print(df_resumo_teste2.to_string(index=False))

print("\n=== Relatório por Classe no MLP — Fujita Antes (0.5) ===")
print(aval_fuj_antes["df_relatorio_classes"].to_string(index=False))

print("\n=== Relatório por Classe no MLP — Fujita Pós-Hopfield ===")
print(aval_fuj_pos["df_relatorio_classes"].to_string(index=False))

# %%
# Plot das Matrizes de Confusão e Espaço Latente 32D do Teste 2 (Fujita Antes 0.5 vs. Pós-Hopfield)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, aval, titulo in zip(
    axes,
    [aval_fuj_antes, aval_fuj_pos],
    [
        f"Fujita Antes Hopfield (c/ 0.5)\nAcc: {aval_fuj_antes['acuracia'] * 100:.2f}% | F1-Macro: {aval_fuj_antes['f1_macro'] * 100:.2f}%",
        f"Fujita Pós-Hopfield (Memória Pan 100%)\nAcc: {aval_fuj_pos['acuracia'] * 100:.2f}% | F1-Macro: {aval_fuj_pos['f1_macro'] * 100:.2f}%",
    ],
    strict=True,
):
    ConfusionMatrixDisplay(
        confusion_matrix=aval["matriz_confusao"],
        display_labels=rotulos_curtos,
    ).plot(cmap="Blues", ax=ax, values_format="d", colorbar=False)
    ax.set_title(titulo, fontsize=11)
    ax.grid(False)

plt.suptitle(
    "Teste 2 (Pan -> Fujita Faltantes): Matrizes de Confusão no Classificador MLP",
    fontsize=14,
)
plt.tight_layout()
plt.savefig(DIR_SAIDA / "matriz_confusao_teste2_fujita_faltantes.png", dpi=180)
plt.show()

# %%
# Visualização PCA 2D da Penúltima Camada Densa (32 Neurônios) — Antes (0.5) vs. Pós-Hopfield
if (
    aval_fuj_antes["latente_32d"] is not None
    and aval_fuj_pos["latente_32d"] is not None
):
    pca_antes = PCA(n_components=2, random_state=42).fit_transform(
        aval_fuj_antes["latente_32d"]
    )
    pca_pos = PCA(n_components=2, random_state=42).fit_transform(
        aval_fuj_pos["latente_32d"]
    )
    y_fuj_1_7 = aval_fuj_pos["y_real"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, proj_2d, titulo in zip(
        axes,
        [pca_antes, pca_pos],
        [
            "Espaço Latente 32D (PCA) — Fujita Antes (0.5)",
            "Espaço Latente 32D (PCA) — Fujita Pós-Hopfield",
        ],
        strict=True,
    ):
        for c in range(1, 8):
            mask_c = y_fuj_1_7 == c
            if np.any(mask_c):
                ax.scatter(
                    proj_2d[mask_c, 0],
                    proj_2d[mask_c, 1],
                    s=4,
                    alpha=0.55,
                    label=NOMES_CLASSES_PADRAO.get(c, f"Classe {c}"),
                )
        ax.set_title(titulo, fontsize=11)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(markerscale=3, fontsize=8, loc="best")

    plt.tight_layout()
    plt.savefig(DIR_SAIDA / "pca_latente_32d_fujita_antes_vs_pos_hopfield.png", dpi=180)
    plt.show()

# %% [markdown]
# ## 5. Exportação das Matrizes SWeeP Reconstruídas

# %%
path_pan_loo_txt = (
    DIR_SAIDA / "sweep_pan_faltantes_mathys_reconstruido_hopfield_loo.txt"
)
path_pan_full_txt = (
    DIR_SAIDA / "sweep_pan_faltantes_mathys_reconstruido_hopfield_full.txt"
)
path_fujita_rec_txt = DIR_SAIDA / "sweep_fujita_faltantes_pan_reconstruido_hopfield.txt"

np.savetxt(path_pan_loo_txt, sweep_pan_rec_loo, delimiter="\t", fmt="%.6f")
np.savetxt(path_pan_full_txt, sweep_pan_rec_full, delimiter="\t", fmt="%.6f")
np.savetxt(path_fujita_rec_txt, sweep_fujita_reconstruido, delimiter="\t", fmt="%.6f")

np.save(path_pan_loo_txt.with_suffix(".npy"), sweep_pan_rec_loo)
np.save(path_pan_full_txt.with_suffix(".npy"), sweep_pan_rec_full)
np.save(path_fujita_rec_txt.with_suffix(".npy"), sweep_fujita_reconstruido)

print("Matrizes SWeeP reconstruídas salvas com sucesso em:")
print(f" - {path_pan_loo_txt}")
print(f" - {path_pan_full_txt}")
print(f" - {path_fujita_rec_txt}")
