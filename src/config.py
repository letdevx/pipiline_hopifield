"""Configuração global de caminhos e parâmetros do pipeline.

Centraliza as variáveis de diretório raiz, caminhos de entrada (locais e Colab)
e estrutura de pastas geradas em outputs/.
"""

import os

# ---------------------------------------------------------------------------
# Entradas — busca dinâmica (Local Windows / Linux / Google Drive Colab)
# ---------------------------------------------------------------------------


def _resolver_path_base() -> str:
    """Resolve dinamicamente o diretório raiz dos dados entre Colab, Windows e repositório local."""
    env_base = os.environ.get("PIPELINE_PATH_BASE") or os.environ.get("PATH_BASE")
    if env_base and os.path.exists(env_base):
        return env_base

    # 1. Caminho Google Colab com Google Drive montado
    colab_path = (
        r"/content/drive/Othercomputers/Meu laptop/Documents/Letworkspace/Teste hop"
    )
    if os.path.exists(colab_path):
        return colab_path

    # 2. Caminho Windows Local da pesquisadora ("Meu laptop")
    windows_path = r"C:\Users\Leticia\Documents\Letworkspace\Teste hop"
    if os.path.exists(windows_path):
        return windows_path

    # 3. Raiz do repositório local do projeto
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return repo_root


PATH_BASE: str = _resolver_path_base()

# Entrada Conjunto de Referência
PATH_REFERENCIA: str = os.path.join(PATH_BASE, "imputs", "pan_anotado.h5ad")

# Entrada Conjunto Alvo
PATH_ALVO: str = os.path.join(PATH_BASE, "imputs", "matrizFiltradaeNormalizadaF.h5ad")

# Features
PATH_FEATURES_REFERENCIA: str = os.path.join(
    PATH_BASE, "imputs", "featuresPANcorrigido.tsv"
)
PATH_FEATURES_ALVO: str = os.path.join(PATH_BASE, "imputs", "featuresF.tsv")

# Matrizes SWeeP, Base Ortonormal Congelada e Rótulos
PATH_SWEEP_REFERENCIA: str = os.path.join(
    PATH_BASE, "outputs", "treinamento", "matriz_reduzida_sweepREF.txt"
)
PATH_SWEEP_ALVO: str = os.path.join(
    PATH_BASE, "outputs", "treinamento", "matriz_reduzida_sweepALVO.txt"
)
PATH_SWEEP_ALVO_SENTINELA: str = os.path.join(
    PATH_BASE, "outputs", "treinamento", "matriz_reduzida_sweepALVO_sentinela.txt"
)
PATH_ORTHBASE_RDS: str = os.environ.get(
    "ORTHBASE_PATH",
    os.path.join(PATH_BASE, "outputs", "treinamento", "orthbase_mproj_600d.rds"),
)
PATH_LABELS_REFERENCIA: str = os.path.join(PATH_BASE, "imputs", "PanNumerico.csv")
PATH_LABELS_ALVO: str = os.path.join(PATH_BASE, "imputs", "cell_types_binarioF.txt")

# ---------------------------------------------------------------------------
# Saídas — geradas automaticamente dentro da raiz do projeto
# ---------------------------------------------------------------------------
OUTPUTS: str = os.path.join(PATH_BASE, "outputs")
OUT_BINARIZACAO: str = os.path.join(OUTPUTS, "binarizacao")
OUT_ALINHAMENTO: str = os.path.join(OUTPUTS, "alinhamento")
OUT_TOP_GENES: str = os.path.join(OUTPUTS, "top_genes")
OUT_TREINAMENTO: str = os.path.join(OUTPUTS, "treinamento")
OUT_HOPFIELD: str = os.path.join(OUTPUTS, "hopfield")
OUT_IMPUTACAO: str = os.path.join(OUTPUTS, "imputacao")
OUT_RELATORIO: str = os.path.join(OUTPUTS, "relatorio")

# Diretórios dedicados para exportação Matrix Market (MTX)
OUT_MTX_REFERENCIA: str = os.path.join(OUT_ALINHAMENTO, "mtx_referencia")
OUT_MTX_ALVO_SENTINELA: str = os.path.join(OUT_ALINHAMENTO, "mtx_alvo_sentinela")
OUT_MTX_ALVO_IMPUTADO: str = os.path.join(OUT_IMPUTACAO, "mtx_alvo_imputado")

# ---------------------------------------------------------------------------
# Retrocompatibilidade (Aliases para nomes legados)
# ---------------------------------------------------------------------------
PATH_F: str = PATH_REFERENCIA
PATH_M: str = PATH_ALVO
PATH_FEATURES_F: str = PATH_FEATURES_REFERENCIA
PATH_FEATURES_M: str = PATH_FEATURES_ALVO
PATH_SWEEP_F: str = PATH_SWEEP_REFERENCIA
PATH_SWEEP_M: str = PATH_SWEEP_ALVO
PATH_SWEEP_M_SENTINELA: str = PATH_SWEEP_ALVO_SENTINELA
PATH_LABELS_F: str = PATH_LABELS_REFERENCIA
PATH_LABELS_M: str = PATH_LABELS_ALVO
PATH_ORTHBASE: str = PATH_ORTHBASE_RDS
