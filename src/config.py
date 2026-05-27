import os

# Raiz do projeto (pasta pipiline_hopifield/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Entradas — ajuste os caminhos conforme o ambiente
# ---------------------------------------------------------------------------
PATH_M = r"C:\Users\Leticia\Documents\Letworkspace\Sweep-Harmonization\Meus_testes\Controle_qualidade\dataM\matrizFiltradaeNormalizadaMParcial.h5ad"
PATH_F = r"C:\Users\Leticia\Documents\Letworkspace\Sweep-Harmonization\Meus_testes\Controle_qualidade\dataF\MatrizfiltradaenormalizadaF\matrizFiltradaeNormalizadaF.h5ad"

PATH_FEATURES_M = r"C:\Users\Leticia\Documents\Letworkspace\Sweep-Harmonization\Meus_testes\Controle_qualidade\dataM\featuresM.tsv.gz"
PATH_FEATURES_F = r"C:\Users\Leticia\Documents\Letworkspace\Sweep-Harmonization\Meus_testes\Controle_qualidade\dataF\dados_combinados\features.tsv\features.tsv"

PATH_TOP5000 = r"C:\Users\Leticia\Documents\Letworkspace\Sweep-Harmonization\Meus_testes\Testes Hopifild\top_5000_frequentes.csv"

PATH_SWEEP_F  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "outputs", "treinamento", "matriz_reduzida_sweepF.csv")
PATH_LABELS_F = os.path.join(ROOT, "imputs", "cell_types_binarioF.txt")
PATH_LABELS_M = os.path.join(ROOT, "imputs", "celltypeBinMparcial.csv")

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
