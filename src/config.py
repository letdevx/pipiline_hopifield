"""Configuração global de caminhos e parâmetros do pipeline.

Centraliza as variáveis de diretório raiz, caminhos de entrada (locais e Colab),
extração automática de nomes dos conjuntos e estrutura dinâmica de pastas
no padrão output_<nome_ref>_<nome_alvo>/.
"""

import os
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Entradas — busca dinâmica (Local Windows / Linux / Google Drive Colab)
# ---------------------------------------------------------------------------


def _resolver_path_base() -> str:
    """Resolve dinamicamente o diretório raiz dos dados entre Colab, Windows e repositório local.

    Returns
    -------
    str
        Caminho absoluto do diretório base do projeto ou dos dados montados.
    """
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


def _extrair_nome_dataset(caminho: str) -> str:
    """Extrai e sanitiza o identificador textual do conjunto a partir do nome do arquivo.

    Parameters
    ----------
    caminho : str
        Caminho do arquivo (ex.: '/path/to/pan_anotado.h5ad').

    Returns
    -------
    str
        Nome base do arquivo sem extensão, com caracteres especiais convertidos para sublinhado.
    """
    stem = Path(caminho).stem
    # Remove ou substitui caracteres especiais por sublinhado
    sanitizado = re.sub(r"[^\w\-]", "_", stem)
    return sanitizado or "dados"


PATH_BASE: str = _resolver_path_base()

# Entrada Conjunto de Referência
PATH_REFERENCIA: str = os.environ.get(
    "PATH_REFERENCIA",
    os.path.join(PATH_BASE, "imputs", "pan_anotado.h5ad"),
)

# Entrada Conjunto Alvo
PATH_ALVO: str = os.environ.get(
    "PATH_ALVO",
    os.path.join(PATH_BASE, "imputs", "pan_com_sentinela_05.h5ad"),
)

# Identificadores automáticos dos conjuntos
NOME_REF: str = os.environ.get("NOME_REF") or _extrair_nome_dataset(PATH_REFERENCIA)
NOME_ALVO: str = os.environ.get("NOME_ALVO") or _extrair_nome_dataset(PATH_ALVO)

# Features
PATH_FEATURES_REFERENCIA: str = os.path.join(
    PATH_BASE, "imputs", "featuresPANcorrigido.tsv"
)
PATH_FEATURES_ALVO: str = os.path.join(PATH_BASE, "imputs", "featuresPANcorrigido.tsv")

# Base Ortonormal Congelada e Rótulos
PATH_ORTHBASE_RDS: str = os.environ.get(
    "ORTHBASE_PATH",
    os.path.join(PATH_BASE, "imputs", "orthbase_mproj_600d.rds"),
)
PATH_LABELS_REFERENCIA: str = os.path.join(PATH_BASE, "imputs", "PanNumerico.csv")
PATH_LABELS_ALVO: str = os.path.join(PATH_BASE, "imputs", "PanNumerico.csv")

# ---------------------------------------------------------------------------
# Saídas dinâmicas — inicializadas no padrão output_<nome_ref>_<nome_alvo>
# ---------------------------------------------------------------------------
OUTPUTS: str = os.path.join(PATH_BASE, f"output_{NOME_REF}_{NOME_ALVO}")
OUT_BINARIZACAO: str = os.path.join(OUTPUTS, "binarizacao")
OUT_ALINHAMENTO: str = os.path.join(OUTPUTS, "alinhamento")
OUT_TOP_GENES: str = os.path.join(OUTPUTS, "top_genes")
OUT_TREINAMENTO: str = os.path.join(OUTPUTS, "treinamento")
OUT_HOPFIELD: str = os.path.join(OUTPUTS, "hopfield")
OUT_IMPUTACAO: str = os.path.join(OUTPUTS, "imputacao")
OUT_RELATORIO: str = os.path.join(OUTPUTS, "relatorio")
OUT_SWEEP_ALVO_POS_IMPUTACAO: str = os.path.join(OUTPUTS, "sweep_alvo_pos_imputacao")
OUT_SWEEP_POS_IMPUTACAO: str = OUT_SWEEP_ALVO_POS_IMPUTACAO

# Diretórios dedicados para exportação Matrix Market (MTX)
OUT_MTX_REFERENCIA: str = os.path.join(OUT_ALINHAMENTO, "mtx_referencia")
OUT_MTX_ALVO_SENTINELA: str = os.path.join(OUT_ALINHAMENTO, "mtx_alvo_sentinela")
OUT_MTX_ALVO_IMPUTADO: str = os.path.join(OUT_IMPUTACAO, "mtx_alvo_imputado")

# Matrizes SWeeP — vinculadas dinamicamente a OUT_TREINAMENTO
PATH_SWEEP_REFERENCIA: str = os.path.join(
    OUT_TREINAMENTO, "matriz_reduzida_sweepREF.txt"
)
PATH_SWEEP_ALVO: str = os.path.join(OUT_TREINAMENTO, "matriz_reduzida_sweepALVO.txt")
PATH_SWEEP_ALVO_SENTINELA: str = os.path.join(
    OUT_TREINAMENTO, "matriz_reduzida_sweepALVO_sentinela.txt"
)

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


def configurar_diretorios(
    path_referencia: str | None = None,
    path_alvo: str | None = None,
    nome_ref: str | None = None,
    nome_alvo: str | None = None,
    path_base: str | None = None,
    criar_diretorios: bool = False,
) -> str:
    """Atualiza dinamicamente as entradas, nomes e toda a árvore de diretórios de saída.

    Permite que notebooks ou scripts reconfigurem o pipeline para novos conjuntos
    de dados em tempo de execução, recalculando automaticamente a pasta
    'output_<nome_ref>_<nome_alvo>' e todas as subpastas derivadas.

    Parameters
    ----------
    path_referencia : str | None, default=None
        Novo caminho para o arquivo do conjunto de referência (.h5ad).
    path_alvo : str | None, default=None
        Novo caminho para o arquivo do conjunto alvo (.h5ad).
    nome_ref : str | None, default=None
        Identificador amigável para o conjunto de referência. Se omitido,
        é inferido automaticamente do nome de arquivo em `path_referencia`.
    nome_alvo : str | None, default=None
        Identificador amigável para o conjunto alvo. Se omitido,
        é inferido automaticamente do nome de arquivo em `path_alvo`.
    path_base : str | None, default=None
        Diretório base onde as saídas serão criadas. Se None, mantém o `PATH_BASE` atual.
    criar_diretorios : bool, default=False
        Se True, cria fisicamente as subpastas no disco via `os.makedirs(..., exist_ok=True)`.

    Returns
    -------
    str
        Caminho absoluto do diretório de saídas atualizado (`OUTPUTS`).
    """
    global PATH_BASE, PATH_REFERENCIA, PATH_ALVO, NOME_REF, NOME_ALVO
    global OUTPUTS, OUT_BINARIZACAO, OUT_ALINHAMENTO, OUT_TOP_GENES
    global OUT_TREINAMENTO, OUT_HOPFIELD, OUT_IMPUTACAO, OUT_RELATORIO
    global OUT_SWEEP_ALVO_POS_IMPUTACAO, OUT_SWEEP_POS_IMPUTACAO
    global OUT_MTX_REFERENCIA, OUT_MTX_ALVO_SENTINELA, OUT_MTX_ALVO_IMPUTADO
    global PATH_SWEEP_REFERENCIA, PATH_SWEEP_ALVO, PATH_SWEEP_ALVO_SENTINELA
    global PATH_F, PATH_M, PATH_SWEEP_F, PATH_SWEEP_M, PATH_SWEEP_M_SENTINELA

    if path_base is not None:
        PATH_BASE = str(path_base)

    if path_referencia is not None:
        PATH_REFERENCIA = str(path_referencia)
        PATH_F = PATH_REFERENCIA
        if nome_ref is None:
            NOME_REF = _extrair_nome_dataset(PATH_REFERENCIA)

    if nome_ref is not None:
        NOME_REF = str(nome_ref)

    if path_alvo is not None:
        PATH_ALVO = str(path_alvo)
        PATH_M = PATH_ALVO
        if nome_alvo is None:
            NOME_ALVO = _extrair_nome_dataset(PATH_ALVO)

    if nome_alvo is not None:
        NOME_ALVO = str(nome_alvo)

    # Recalcula a pasta principal de saídas e todas as subpastas
    OUTPUTS = os.path.join(PATH_BASE, f"output_{NOME_REF}_{NOME_ALVO}")
    OUT_BINARIZACAO = os.path.join(OUTPUTS, "binarizacao")
    OUT_ALINHAMENTO = os.path.join(OUTPUTS, "alinhamento")
    OUT_TOP_GENES = os.path.join(OUTPUTS, "top_genes")
    OUT_TREINAMENTO = os.path.join(OUTPUTS, "treinamento")
    OUT_HOPFIELD = os.path.join(OUTPUTS, "hopfield")
    OUT_IMPUTACAO = os.path.join(OUTPUTS, "imputacao")
    OUT_RELATORIO = os.path.join(OUTPUTS, "relatorio")
    OUT_SWEEP_ALVO_POS_IMPUTACAO = os.path.join(OUTPUTS, "sweep_alvo_pos_imputacao")
    OUT_SWEEP_POS_IMPUTACAO = OUT_SWEEP_ALVO_POS_IMPUTACAO

    OUT_MTX_REFERENCIA = os.path.join(OUT_ALINHAMENTO, "mtx_referencia")
    OUT_MTX_ALVO_SENTINELA = os.path.join(OUT_ALINHAMENTO, "mtx_alvo_sentinela")
    OUT_MTX_ALVO_IMPUTADO = os.path.join(OUT_IMPUTACAO, "mtx_alvo_imputado")

    PATH_SWEEP_REFERENCIA = os.path.join(
        OUT_TREINAMENTO, "matriz_reduzida_sweepREF.txt"
    )
    PATH_SWEEP_ALVO = os.path.join(OUT_TREINAMENTO, "matriz_reduzida_sweepALVO.txt")
    PATH_SWEEP_ALVO_SENTINELA = os.path.join(
        OUT_TREINAMENTO, "matriz_reduzida_sweepALVO_sentinela.txt"
    )
    PATH_SWEEP_F = PATH_SWEEP_REFERENCIA
    PATH_SWEEP_M = PATH_SWEEP_ALVO
    PATH_SWEEP_M_SENTINELA = PATH_SWEEP_ALVO_SENTINELA

    if criar_diretorios:
        for d in (
            OUT_BINARIZACAO,
            OUT_ALINHAMENTO,
            OUT_TOP_GENES,
            OUT_TREINAMENTO,
            OUT_HOPFIELD,
            OUT_IMPUTACAO,
            OUT_RELATORIO,
            OUT_SWEEP_ALVO_POS_IMPUTACAO,
            OUT_MTX_REFERENCIA,
            OUT_MTX_ALVO_SENTINELA,
            OUT_MTX_ALVO_IMPUTADO,
        ):
            os.makedirs(d, exist_ok=True)

    return OUTPUTS
