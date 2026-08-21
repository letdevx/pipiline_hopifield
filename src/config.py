import os

# Raiz do projeto (pasta pipiline_hopifield/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Entradas — ajuste os caminhos conforme o ambiente
# ---------------------------------------------------------------------------
# Entrada Mathys (PATH_M)
_PATH_M_DEFAULT = os.path.join(ROOT, "imputs", "matriz_anotada_finalM.h5ad")
_PATH_M_SWEEP   = r"C:\Users\Leticia\Documents\Letworkspace\Sweep-Harmonization\Meus_testes\Controle_qualidade\dataM\matrizFiltradaeNormalizadaMParcial.h5ad"
PATH_M = _PATH_M_SWEEP if os.path.exists(_PATH_M_SWEEP) else _PATH_M_DEFAULT

# Entrada Fujita (PATH_F)
PATH_F = os.path.join(ROOT, "imputs", "pan_anotado.h5ad")

# Features
_PATH_FEAT_M_DEFAULT = os.path.join(ROOT, "imputs", "featuresM.tsv.gz")
_PATH_FEAT_M_SWEEP   = r"C:\Users\Leticia\Documents\Letworkspace\Sweep-Harmonization\Meus_testes\Controle_qualidade\dataM\featuresM.tsv.gz"
PATH_FEATURES_M = _PATH_FEAT_M_SWEEP if os.path.exists(_PATH_FEAT_M_SWEEP) else _PATH_FEAT_M_DEFAULT

_PATH_FEAT_F_DEFAULT = os.path.join(ROOT, "imputs", "features.tsv")
_PATH_FEAT_F_ROOT    = os.path.join(ROOT, "featuresPAN.tsv")
PATH_FEATURES_F = _PATH_FEAT_F_ROOT if os.path.exists(_PATH_FEAT_F_ROOT) else _PATH_FEAT_F_DEFAULT

PATH_TOP5000 = r"C:\Users\Leticia\Documents\Letworkspace\Sweep-Harmonization\Meus_testes\Testes Hopifild\top_5000_frequentes.csv"

PATH_SWEEP_F  = os.path.join(ROOT, "outputs", "treinamento", "matriz_reduzida_sweepF.csv")
PATH_SWEEP_M  = os.path.join(ROOT, "outputs", "treinamento", "matriz_reduzida_sweepM.csv")
PATH_LABELS_F = os.path.join(ROOT, "imputs", "PanNumerico.csv")

_PATH_LABELS_M_PARCIAL = os.path.join(ROOT, "imputs", "celltypeBinMparcial.csv")
_PATH_LABELS_M_TXT     = os.path.join(ROOT, "imputs", "tipos_celulares_numericoMs.txt")
PATH_LABELS_M = _PATH_LABELS_M_PARCIAL if os.path.exists(_PATH_LABELS_M_PARCIAL) else _PATH_LABELS_M_TXT

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

