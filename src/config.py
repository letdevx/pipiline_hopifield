import os

# Raiz do projeto (pasta pipiline_hopifield/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Entradas — busca dinâmica (Local Windows / Linux / Google Drive Colab)
# ---------------------------------------------------------------------------

PATH_BASE = r"/content/drive/Othercomputers/Meu laptop/Documents/Letworkspace/Teste hop/imputs"

# Entrada Conjunto de Referência
PATH_REFERENCIA = os.path.join(PATH_BASE, "pan_anotado.h5ad")

# Entrada Conjunto Alvo
PATH_ALVO = os.path.join(PATH_BASE, "matriz_anotada_finalM.h5ad")

# Features
_feat_ref = os.path.join(PATH_BASE, "featuresPANcorrigido.tsv")
if not os.path.exists(_feat_ref):
    for _fn in ["featuresPAN.tsv", "features.tsv"]:
        _p = os.path.join(PATH_BASE, _fn)
        if os.path.exists(_p):
            _feat_ref = _p
            break
PATH_FEATURES_REFERENCIA = _feat_ref

_feat_alvo = os.path.join(PATH_BASE, "featuresM.tsv.gz")
if not os.path.exists(_feat_alvo):
    _p = os.path.join(PATH_BASE, "featuresM.tsv")
    if os.path.exists(_p):
        _feat_alvo = _p
PATH_FEATURES_ALVO = _feat_alvo

#

PATH_SWEEP_REFERENCIA  = os.path.join(ROOT, "outputs", "treinamento", "matriz_reduzida_sweepF.csv")
PATH_SWEEP_ALVO        = os.path.join(ROOT, "outputs", "treinamento", "matriz_reduzida_sweepM.csv")
PATH_LABELS_REFERENCIA = os.path.join(PATH_BASE, "PanNumerico.csv")
PATH_LABELS_ALVO       = os.path.join(PATH_BASE, "tipos_celulares_numericoMs.txt")

# ---------------------------------------------------------------------------as
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


