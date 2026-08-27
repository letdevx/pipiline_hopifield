import os

# Raiz do projeto (pasta pipiline_hopifield/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Entradas — busca dinâmica (Local Windows / Linux / Google Drive Colab)
# ---------------------------------------------------------------------------
_POSSIVEIS_DIRETORIOS_INPUT = [
    os.path.join(ROOT, "imputs"),
    r"/content/drive/Othercomputers/Meu laptop/Documents/Letworkspace/pipiline_hopifield/imputs",
]

def _encontrar_arquivo(nomes, default_relativo="imputs"):
    if isinstance(nomes, str):
        nomes = [nomes]
    for d in _POSSIVEIS_DIRETORIOS_INPUT:
        for n in nomes:
            p = os.path.join(d, n)
            if os.path.exists(p):
                return p
    return os.path.join(ROOT, default_relativo, nomes[0])

# Entrada Conjunto de Referência
PATH_REFERENCIA = _encontrar_arquivo(["pan_anotado.h5ad", "MatrizfiltradaenormalizadaF.h5ad", "matrizFiltradaeNormalizadaF.h5ad"])

# Entrada Conjunto Alvo
PATH_ALVO = _encontrar_arquivo(["matriz_anotada_finalM.h5ad", "matrizFiltradaeNormalizadaMParcial.h5ad", "matrizFiltradaeNormalizadaM.h5ad"])

# Features
PATH_FEATURES_REFERENCIA = _encontrar_arquivo(["featuresPANcorrigido.tsv", "featuresPAN.tsv", "features.tsv"], default_relativo="")
PATH_FEATURES_ALVO       = _encontrar_arquivo(["featuresM.tsv.gz", "featuresM.tsv"])

PATH_TOP5000 = _encontrar_arquivo(["top_5000_frequentes.csv", "top5000_frequentes.csv"])

PATH_SWEEP_REFERENCIA  = os.path.join(ROOT, "outputs", "treinamento", "matriz_reduzida_sweepF.csv")
PATH_SWEEP_ALVO        = os.path.join(ROOT, "outputs", "treinamento", "matriz_reduzida_sweepM.csv")
PATH_LABELS_REFERENCIA = _encontrar_arquivo(["PanNumerico.csv", "cell_types_binarioF.txt", "celltypeBinF_eceltypename.csv"])
PATH_LABELS_ALVO       = _encontrar_arquivo(["tipos_celulares_numericoMs.txt", "celltypeBinMparcial.csv"])

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


