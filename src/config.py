import os

# Raiz do projeto (pasta pipiline_hopifield/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Resolução Dinâmica de Diretórios de Entradas (Colab, Drive, Local e Fallback)
# ---------------------------------------------------------------------------
_CANDIDATOS_DIR = [
    os.getenv("PIPELINE_DATA_DIR"),
    r"/content/drive/Other computers/Meu laptop/Documents/Letworkspace/Teste hop/imputs",
    r"/content/drive/Othercomputers/Meu laptop/Documents/Letworkspace/Teste hop/imputs",
    r"/content/drive/Other computers/Meu laptop/Documents/Letworkspace/pipiline_hopifield/imputs",
    r"/content/drive/Othercomputers/Meu laptop/Documents/Letworkspace/pipiline_hopifield/imputs",
    r"/content/drive/MyDrive/imputs",
    r"/content/drive/MyDrive/pipiline_hopifield/imputs",
    r"/content/drive/MyDrive/Teste hop/imputs",
    r"/content/pipiline_hopifield/imputs",
    r"C:\Users\Leticia\Documents\Letworkspace\Sweep-Harmonization\Meus_testes\Controle_qualidade\dataM",
    r"C:\Users\Leticia\Documents\Letworkspace\pipiline_hopifield\imputs",
    os.path.join(ROOT, "imputs"),
]

INPUTS_DIR = os.path.join(ROOT, "imputs")
for d in _CANDIDATOS_DIR:
    if d and os.path.exists(d):
        INPUTS_DIR = d
        break

def _buscar_no_drive(nome_arquivo):
    """Busca dinamicamente pelo arquivo no /content/drive (Google Colab)."""
    if not os.path.exists("/content/drive"):
        return None
    try:
        for root, dirs, files in os.walk("/content/drive"):
            if nome_arquivo in files:
                return os.path.join(root, nome_arquivo)
            # Limita a profundidade para manter alta velocidade
            if root.count(os.sep) - "/content/drive".count(os.sep) >= 6:
                dirs.clear()
    except Exception:
        pass
    return None

def _resolver_caminho(nome_arquivo, *caminhos_candidatos):
    """Retorna o primeiro caminho existente entre os candidatos informados ou no INPUTS_DIR."""
    # 1. Checa candidatos diretos específicos
    for path in caminhos_candidatos:
        if path and os.path.exists(path):
            return path

    # 2. Checa no INPUTS_DIR resolvido
    path_inputs = os.path.join(INPUTS_DIR, nome_arquivo)
    if os.path.exists(path_inputs):
        return path_inputs

    # 3. Busca exaustiva nos diretórios candidatos conhecidos
    for d in _CANDIDATOS_DIR:
        if d and os.path.exists(d):
            p = os.path.join(d, nome_arquivo)
            if os.path.exists(p):
                return p

    # 4. Busca dinâmica global em subpastas de /content/drive no Colab
    path_drive = _buscar_no_drive(nome_arquivo)
    if path_drive:
        return path_drive

    # 5. Fallback final para a pasta do projeto com aviso explicativo
    caminho_final = os.path.join(ROOT, "imputs", nome_arquivo)
    if not os.path.exists(caminho_final):
        print(f"[AVISO config.py] Arquivo '{nome_arquivo}' não foi encontrado nos locais verificados.")
        print(f" -> Caminho retornado (fallback): {caminho_final}")
        print(" -> Dica Colab: Certifique-se de montar o Google Drive antes de importar a config:")
        print("    from google.colab import drive")
        print("    drive.mount('/content/drive')")
        print(" -> Ou defina a variável de ambiente com a pasta correta:")
        print("    import os; os.environ['PIPELINE_DATA_DIR'] = '/caminho/para/seus/dados'")
    return caminho_final

# ---------------------------------------------------------------------------
# Entradas — Resolução Dinâmica
# ---------------------------------------------------------------------------
# Entrada Conjunto de Referência (PATH_REFERENCIA)
PATH_REFERENCIA = _resolver_caminho(
    "pan_anotado.h5ad",
    os.path.join(INPUTS_DIR, "pan_anotado.h5ad"),
    r"/content/drive/Other computers/Meu laptop/Documents/Letworkspace/Teste hop/imputs/pan_anotado.h5ad",
    r"/content/drive/Othercomputers/Meu laptop/Documents/Letworkspace/Teste hop/imputs/pan_anotado.h5ad"
)

# Entrada Conjunto Alvo (PATH_ALVO)
PATH_ALVO = _resolver_caminho(
    "matriz_anotada_finalM.h5ad",
    os.path.join(INPUTS_DIR, "matrizFiltradaeNormalizadaMParcial.h5ad"),
    r"/content/drive/Other computers/Meu laptop/Documents/Letworkspace/Teste hop/imputs/matrizFiltradaeNormalizadaMParcial.h5ad",
    r"/content/drive/Othercomputers/Meu laptop/Documents/Letworkspace/Teste hop/imputs/matrizFiltradaeNormalizadaMParcial.h5ad",
    r"C:\Users\Leticia\Documents\Letworkspace\Sweep-Harmonization\Meus_testes\Controle_qualidade\dataM\matrizFiltradaeNormalizadaMParcial.h5ad",
)

# Features
PATH_FEATURES_ALVO = _resolver_caminho(
    "featuresM.tsv.gz",
    os.path.join(INPUTS_DIR, "featuresM.tsv.gz"),
    r"/content/drive/Othercomputers/Meu laptop/Documents/Letworkspace/Teste hop/imputs/featuresM.tsv.gz",
    r"C:\Users\Leticia\Documents\Letworkspace\Sweep-Harmonization\Meus_testes\Controle_qualidade\dataM\featuresM.tsv.gz",
)

PATH_FEATURES_REFERENCIA = _resolver_caminho(
    "featuresPANcorrigido.tsv",
    os.path.join(ROOT, "featuresPANcorrigido.tsv")
)

PATH_TOP5000 = _resolver_caminho(
    "top_5000_frequentes.csv",
    os.path.join(INPUTS_DIR, "top_5000_frequentes.csv"),
    r"C:\Users\Leticia\Documents\Letworkspace\Sweep-Harmonization\Meus_testes\Testes Hopifild\top_5000_frequentes.csv",
)

PATH_SWEEP_REFERENCIA  = os.path.join(ROOT, "outputs", "treinamento", "matriz_reduzida_sweepF.csv")
PATH_SWEEP_ALVO        = os.path.join(ROOT, "outputs", "treinamento", "matriz_reduzida_sweepM.csv")
PATH_LABELS_REFERENCIA = _resolver_caminho("PanNumerico.csv", os.path.join(INPUTS_DIR, "PanNumerico.csv"))

PATH_LABELS_ALVO = _resolver_caminho(
    "celltypeBinMparcial.csv",
    os.path.join(INPUTS_DIR, "celltypeBinMparcial.csv"),
    os.path.join(INPUTS_DIR, "tipos_celulares_numericoMs.txt"),
)

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



