"""Configuração global de caminhos e parâmetros do pipeline.

Centraliza as variáveis de diretório raiz, caminhos de entrada (locais e Colab)
e estrutura de pastas geradas em outputs/.
"""

import os

# Raiz do projeto (pasta pipiline_hopifield/)
ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Entradas — busca dinâmica (Local Windows / Linux / Google Drive Colab)
# ---------------------------------------------------------------------------

PATH_BASE: str = (
    r"/content/drive/Othercomputers/Meu laptop/Documents/Letworkspace/Teste hop/imputs"
)

# Entrada Conjunto de Referência
PATH_REFERENCIA: str = os.path.join(PATH_BASE, "matrizFiltradaeNormalizadaF.h5ad")

# Entrada Conjunto Alvo
PATH_ALVO: str = os.path.join(PATH_BASE, "matriz_anotada_finalM.h5ad")

# Features
PATH_FEATURES_REFERENCIA: str = os.path.join(PATH_BASE, "featuresF.tsv")
PATH_FEATURES_ALVO: str = os.path.join(PATH_BASE, "featuresM.tsv")

# Matrizes SWeeP e Rótulos
PATH_SWEEP_REFERENCIA: str = os.path.join(
    ROOT, "outputs", "treinamento", "matriz_reduzida_sweepF.csv"
)
PATH_SWEEP_ALVO: str = os.path.join(
    ROOT, "outputs", "treinamento", "matriz_reduzida_sweepM.csv"
)
PATH_LABELS_REFERENCIA: str = os.path.join(PATH_BASE, "cell_types_binarioF.txt")
PATH_LABELS_ALVO: str = os.path.join(PATH_BASE, "tipos_celulares_numericoMs.txt")

# ---------------------------------------------------------------------------
# Saídas — geradas automaticamente dentro da raiz do projeto
# ---------------------------------------------------------------------------
OUTPUTS: str = os.path.join(PATH_BASE, "outputs")
OUT_BINARIZACAO: str = os.path.join(OUTPUTS, "binarizacao")
OUT_ALINHAMENTO: str = os.path.join(OUTPUTS, "alinhamento")
OUT_TOP_GENES: str = os.path.join(OUTPUTS, "top_genes")
OUT_TREINAMENTO: str = os.path.join(OUTPUTS, "treinamento")
OUT_HOPFIELD: str = os.path.join(OUTPUTS, "hopfield")
OUT_RELATORIO: str = os.path.join(OUTPUTS, "relatorio")

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
