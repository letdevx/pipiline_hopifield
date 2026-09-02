"""Configuração global de caminhos e parâmetros do pipeline.

Centraliza as variáveis de diretório raiz, caminhos de entrada (locais e Colab)
e estrutura de pastas geradas em outputs/.
"""

import os

# ---------------------------------------------------------------------------
# Entradas — busca dinâmica (Local Windows / Linux / Google Drive Colab)
# ---------------------------------------------------------------------------

PATH_BASE: str = (
    r"/content/drive/Othercomputers/Meu laptop/Documents/Letworkspace/Teste hop"
)

# Entrada Conjunto de Referência
PATH_REFERENCIA: str = os.path.join(
    PATH_BASE, "imputs", "pan_anotado.h5ad"
)

# Entrada Conjunto Alvo
PATH_ALVO: str = os.path.join(PATH_BASE, "imputs", "matrizFiltradaeNormalizadaF.h5ad")

# Features
PATH_FEATURES_REFERENCIA: str = os.path.join(PATH_BASE, "imputs", "featuresPANcorrigido.tsv")
PATH_FEATURES_ALVO: str = os.path.join(PATH_BASE, "imputs", "featuresFtsv")

# Matrizes SWeeP, Base Ortonormal Congelada e Rótulos
PATH_SWEEP_REFERENCIA: str = os.path.join(
    PATH_BASE, "outputs", "treinamento", "matriz_reduzida_sweepREF.txt"
)
PATH_SWEEP_ALVO: str = os.path.join(
    PATH_BASE, "outputs", "treinamento", "matriz_reduzida_sweepALVO.txt"
)
PATH_ORTHBASE_RDS: str = os.path.join(
    PATH_BASE, "outputs", "treinamento", "orthbase_mproj_600d.rds"
)
PATH_LABELS_REFERENCIA: str = os.path.join(
    PATH_BASE, "imputs", "PanNumerico.txt"
)
PATH_LABELS_ALVO: str = os.path.join(
    PATH_BASE, "imputs", "cell_types_binarioF.txt"
)

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
PATH_LABELS_F: str = PATH_LABELS_REFERENCIA
PATH_LABELS_M: str = PATH_LABELS_ALVO
PATH_ORTHBASE: str = PATH_ORTHBASE_RDS
