import os

# Raiz do projeto (pasta pipiline_hopifield/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Entradas — ajuste os caminhos conforme o ambiente
# ---------------------------------------------------------------------------
# Entrada Conjunto de Referência (PATH_REFERENCIA)
PATH_REFERENCIA = os.path.join(ROOT, "imputs", "pan_anotado.h5ad")

# Entrada Conjunto Alvo (PATH_ALVO)
_PATH_ALVO_DEFAULT = os.path.join(ROOT, "imputs", "matriz_anotada_finalM.h5ad")
_PATH_ALVO_SWEEP   = r"/content/drive/Othercomputers/Meu laptop/Documents/Letworkspace/Teste hop/imputs/matrizFiltradaeNormalizadaMParcial.h5ad"
PATH_ALVO = _PATH_ALVO_SWEEP if os.path.exists(_PATH_ALVO_SWEEP) else _PATH_ALVO_DEFAULT

# Features
_PATH_FEAT_ALVO_DEFAULT = os.path.join(ROOT, "imputs", "featuresM.tsv.gz")
_PATH_FEAT_ALVO_SWEEP   = r"/content/drive/Othercomputers/Meu laptop/Documents/Letworkspace/Teste hop/imputs/featuresM.tsv.gz"
PATH_FEATURES_ALVO = _PATH_FEAT_ALVO_SWEEP if os.path.exists(_PATH_FEAT_ALVO_SWEEP) else _PATH_FEAT_ALVO_DEFAULT

_PATH_FEAT_REF_DEFAULT = os.path.join(ROOT, "imputs", "features.tsv")
_PATH_FEAT_REF_ROOT    = os.path.join(ROOT, "featuresPAN.tsv")
PATH_FEATURES_REFERENCIA = _PATH_FEAT_REF_ROOT if os.path.exists(_PATH_FEAT_REF_ROOT) else _PATH_FEAT_REF_DEFAULT

PATH_TOP5000 = r"C:\Users\Leticia\Documents\Letworkspace\Sweep-Harmonization\Meus_testes\Testes Hopifild\top_5000_frequentes.csv"

PATH_SWEEP_REFERENCIA  = os.path.join(ROOT, "outputs", "treinamento", "matriz_reduzida_sweepF.csv")
PATH_SWEEP_ALVO        = os.path.join(ROOT, "outputs", "treinamento", "matriz_reduzida_sweepM.csv")
PATH_LABELS_REFERENCIA = os.path.join(ROOT, "imputs", "PanNumerico.csv")

_PATH_LABELS_ALVO_PARCIAL = os.path.join(ROOT, "imputs", "celltypeBinMparcial.csv")
_PATH_LABELS_ALVO_TXT     = os.path.join(ROOT, "imputs", "tipos_celulares_numericoMs.txt")
PATH_LABELS_ALVO = _PATH_LABELS_ALVO_PARCIAL if os.path.exists(_PATH_LABELS_ALVO_PARCIAL) else _PATH_LABELS_ALVO_TXT

# ---------------------------------------------------------------------------
# Saídas — geradas automaticamente dentro da raiz do projeto
# ---------------------------------------------------------------------------
OUTPUTS         = os.path.join(ROOT, "outputs")
OUT_BINARIZACAO = os.path.join(OUTPUTS, "binarizacao")
OUT_ALINHAMENTO = os.path.join(OUTPUTS, "alinhamento")
OUT_TOP_GENES   = os.path.join(OUTPUTS, "top_genes")
OUT_TREINAMENTO = os.path.join(OUTPUTS, "treinamento")
OUT_HOPFIELD    = os.path.join(OUTPUTS, "hopfield")
OUT_RELATORIO   = os.path.join(OUTPUTS, "relatorio")

# ---------------------------------------------------------------------------
# Retrocompatibilidade (Aliases para nomes legados)
# ---------------------------------------------------------------------------
PATH_F = PATH_REFERENCIA
PATH_M = PATH_ALVO
PATH_FEATURES_F = PATH_FEATURES_REFERENCIA
PATH_FEATURES_M = PATH_FEATURES_ALVO
PATH_SWEEP_F = PATH_SWEEP_REFERENCIA
PATH_SWEEP_M = PATH_SWEEP_ALVO
PATH_LABELS_F = PATH_LABELS_REFERENCIA
PATH_LABELS_M = PATH_LABELS_ALVO


