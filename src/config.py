import os

# Raiz do projeto (pasta pipiline_hopifield/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Entradas — ajuste os caminhos conforme o ambiente
# ---------------------------------------------------------------------------
PATH_M = r"C:\Users\Leticia\Documents\Letworkspace\Sweep-Harmonization\Meus_testes\Controle_qualidade\dataM\matrizFiltradaeNormalizadaM.h5ad"
PATH_F = r"C:\Users\Leticia\Documents\Letworkspace\Sweep-Harmonization\Meus_testes\Controle_qualidade\dataF\MatrizfiltradaenormalizadaF\matrizFiltradaeNormalizadaF.h5ad"

PATH_FEATURES_M = r"C:\Users\Leticia\Documents\Letworkspace\Sweep-Harmonization\Meus_testes\Controle_qualidade\dataM\featuresM.tsv.gz"
PATH_FEATURES_F = r"C:\Users\Leticia\Documents\Letworkspace\Sweep-Harmonization\Meus_testes\Controle_qualidade\dataF\dados_combinados\features.tsv\features.tsv"

PATH_TOP5000 = r"C:\Users\Leticia\Documents\Letworkspace\Sweep-Harmonization\Meus_testes\Testes Hopifild\top_5000_frequentes.csv"

# ---------------------------------------------------------------------------
# Saídas — geradas automaticamente dentro da raiz do projeto
# ---------------------------------------------------------------------------
OUTPUTS         = os.path.join(ROOT, "outputs")
OUT_BINARIZACAO = os.path.join(OUTPUTS, "binarizacao")
OUT_ALINHAMENTO = os.path.join(OUTPUTS, "alinhamento")
OUT_TOP_GENES   = os.path.join(OUTPUTS, "top_genes")
OUT_TREINAMENTO = os.path.join(OUTPUTS, "treinamento")
