"""Testes unitários para o InjetorMTXSentinela (Injeção OOM-Safe de 0.5 em MTX)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import scipy.io as sio
import scipy.sparse as sp

from alinhamento import ExportadorMTX, InjetorMTXSentinela


def _obter_array(mat: Any) -> np.ndarray:
    toarray_fn = getattr(mat, "toarray", None)
    if callable(toarray_fn):
        return np.asarray(toarray_fn())
    return np.asarray(mat)


@pytest.fixture
def pasta_mtx_sintetica(tmp_path: Path) -> dict[str, Any]:
    """Cria uma pasta MTX sintética de teste com matriz esparsa, genes e barcodes."""
    dir_origem = tmp_path / "mtx_origem"
    dir_origem.mkdir(parents=True, exist_ok=True)

    n_celulas = 25
    n_genes = 20

    # Cria matriz esparsa com valores conhecidos
    rng = np.random.default_rng(42)
    linhas_idx = rng.integers(0, n_celulas, size=35)
    cols_idx = rng.integers(0, n_genes, size=35)
    vals = rng.uniform(0.1, 1.0, size=35).astype(np.float32)
    matriz_esparsa = sp.csr_matrix(
        (vals, (linhas_idx, cols_idx)),
        shape=(n_celulas, n_genes),
        dtype=np.float32,
    )

    genes = [f"ENSG{10000000000 + i}" for i in range(n_genes)]
    barcodes = [f"CELL_{i:03d}-1" for i in range(n_celulas)]

    exportador = ExportadorMTX(out_dir=dir_origem)
    exportador.exportar(
        matriz=matriz_esparsa,
        genes=genes,
        barcodes=barcodes,
        nome_etapa="MTX Sintético Teste",
        forcar=True,
    )

    return {
        "dir_origem": dir_origem,
        "n_celulas": n_celulas,
        "n_genes": n_genes,
        "matriz_original": matriz_esparsa.toarray(),
        "genes": genes,
        "barcodes": barcodes,
    }


def test_injecao_mtx_streaming_consistencia(
    pasta_mtx_sintetica: dict[str, Any], tmp_path: Path
) -> None:
    """Valida a injeção de 0.5 e a equivalência exata dos valores não modificados."""
    dir_destino = tmp_path / "mtx_injetado"
    colunas_injetar_1based = [3, 8, 15]

    injetor = InjetorMTXSentinela(
        dir_mtx_origem=pasta_mtx_sintetica["dir_origem"],
        dir_mtx_destino=dir_destino,
        fill_value=0.5,
        buffer_size=10,
    )

    relatorio = injetor.injetar_colunas(
        colunas_1based=colunas_injetar_1based, forcar=True
    )

    assert relatorio["status"] == "SUCESSO"
    assert relatorio["n_celulas"] == pasta_mtx_sintetica["n_celulas"]
    assert relatorio["n_genes"] == pasta_mtx_sintetica["n_genes"]

    # Lê o arquivo MTX gerado
    matriz_pos_raw = sio.mmread(str(dir_destino / "matrix.mtx"))
    matriz_pos: np.ndarray = _obter_array(matriz_pos_raw)

    matriz_esperada = pasta_mtx_sintetica["matriz_original"].copy()
    for col_1b in colunas_injetar_1based:
        col_0b = col_1b - 1
        matriz_esperada[:, col_0b] = 0.5
        # Verifica se todas as células possuem 0.5 nessa coluna
        np.testing.assert_allclose(matriz_pos[:, col_0b], 0.5, rtol=1e-5)

    # Verifica se os outros genes permaneceram idênticos
    colunas_nao_modificadas = [
        c
        for c in range(pasta_mtx_sintetica["n_genes"])
        if (c + 1) not in colunas_injetar_1based
    ]
    np.testing.assert_allclose(
        matriz_pos[:, colunas_nao_modificadas],
        matriz_esperada[:, colunas_nao_modificadas],
        rtol=1e-5,
    )


def test_injecao_mtx_de_tracking_csv(
    pasta_mtx_sintetica: dict[str, Any], tmp_path: Path
) -> None:
    """Valida a injeção a partir de um arquivo CSV de tracking (0-indexed)."""
    dir_destino = tmp_path / "mtx_tracking_destino"
    path_csv = tmp_path / "tracking_teste.csv"

    # Genes ausentes nas posições 0-indexed: 1, 5, 10
    colunas_0based = [1, 5, 10]
    df_trk = pd.DataFrame(
        {
            "gene_name": ["G1", "G5", "G10"],
            "ensembl_id": ["ENSG1", "ENSG5", "ENSG10"],
            "posicao_coluna": colunas_0based,
            "valor_inserido": [0.5, 0.5, 0.5],
        }
    )
    df_trk.to_csv(path_csv, index=False)

    injetor = InjetorMTXSentinela(
        dir_mtx_origem=pasta_mtx_sintetica["dir_origem"],
        dir_mtx_destino=dir_destino,
        fill_value=0.5,
    )

    relatorio = injetor.injetar_de_tracking(
        path_tracking_csv=path_csv,
        coluna_indice="posicao_coluna",
        indexado_em_zero=True,
        forcar=True,
    )

    assert relatorio["status"] == "SUCESSO"
    assert relatorio["colunas_injetadas"] == 3

    matriz_pos_raw = sio.mmread(str(dir_destino / "matrix.mtx"))
    matriz_pos: np.ndarray = _obter_array(matriz_pos_raw)

    for c_0b in colunas_0based:
        np.testing.assert_allclose(matriz_pos[:, c_0b], 0.5, rtol=1e-5)


def test_injecao_mtx_validacoes_e_erros(
    pasta_mtx_sintetica: dict[str, Any], tmp_path: Path
) -> None:
    """Valida que entradas inconsistentes disparam exceções claras (Fail Fast)."""
    injetor = InjetorMTXSentinela(
        dir_mtx_origem=pasta_mtx_sintetica["dir_origem"],
        dir_mtx_destino=tmp_path / "mtx_erros",
    )

    # 1. Colunas vazias
    with pytest.raises(ValueError, match="não pode ser vazio"):
        injetor.injetar_colunas([])

    # 2. Coluna fora dos limites (> n_cols)
    with pytest.raises(ValueError, match="Índices de coluna inválidos"):
        injetor.injetar_colunas([100])

    # 3. Coluna fora dos limites (< 1)
    with pytest.raises(ValueError, match="Índices de coluna inválidos"):
        injetor.injetar_colunas([0])

    # 4. Arquivo de tracking inexistente
    with pytest.raises(FileNotFoundError, match="não encontrado"):
        injetor.injetar_de_tracking(tmp_path / "inexistente.csv")

    # 5. Coluna de índice inexistente no tracking
    path_csv_invalido = tmp_path / "tracking_invalido.csv"
    pd.DataFrame({"outra_coluna": [1, 2]}).to_csv(path_csv_invalido, index=False)
    with pytest.raises(ValueError, match="ausente no CSV"):
        injetor.injetar_de_tracking(path_csv_invalido, coluna_indice="posicao_coluna")


def test_injecao_mtx_metadados_e_auditoria(
    pasta_mtx_sintetica: dict[str, Any], tmp_path: Path
) -> None:
    """Valida que metadados TSV são copiados e passam na auditoria de conformidade."""
    dir_destino = tmp_path / "mtx_auditoria"

    injetor = InjetorMTXSentinela(
        dir_mtx_origem=pasta_mtx_sintetica["dir_origem"],
        dir_mtx_destino=dir_destino,
        fill_value=0.5,
    )

    relatorio = injetor.injetar_colunas(colunas_1based=[1, 2], forcar=True)

    assert relatorio["status"] == "SUCESSO"
    assert (dir_destino / "genes_referencia.tsv").exists()
    assert (dir_destino / "barcodes.tsv").exists()
    assert relatorio["auditoria"] is not None
    assert relatorio["auditoria"]["status"] in ("APROVADO", "VALIDADO", "OK")


def test_injecao_mtx_com_simbolos_genicos_sem_erro_regex(tmp_path: Path) -> None:
    """Valida que datasets com símbolos gênicos (ex: TNFRSF4, RPL7P7) não falham na auditoria."""
    dir_origem = tmp_path / "mtx_simbolos"
    dir_destino = tmp_path / "mtx_simbolos_injetado"
    dir_origem.mkdir(parents=True, exist_ok=True)

    n_celulas = 10
    n_genes = 5
    matriz_esparsa = sp.csr_matrix(np.zeros((n_celulas, n_genes), dtype=np.float32))
    genes_simbolos = ["TNFRSF4", "TNFRSF18", "ATAD3B", "MTND1P23", "PRDM16"]
    barcodes = [f"CELL_{i}" for i in range(n_celulas)]

    exportador = ExportadorMTX(out_dir=dir_origem)
    exportador.exportar(
        matriz=matriz_esparsa,
        genes=genes_simbolos,
        barcodes=barcodes,
        nome_etapa="MTX Símbolos Teste",
        validar_regex=False,
        forcar=True,
    )

    # 1. Teste com detecção automática (validar_regex=None)
    injetor_auto = InjetorMTXSentinela(
        dir_mtx_origem=dir_origem,
        dir_mtx_destino=dir_destino,
        fill_value=0.5,
    )
    relatorio_auto = injetor_auto.injetar_colunas(colunas_1based=[1, 3], forcar=True)
    assert relatorio_auto["status"] == "SUCESSO"
    assert relatorio_auto["auditoria"] is not None
    assert relatorio_auto["auditoria"]["status"] in ("APROVADO", "VALIDADO", "OK")

    # 2. Teste com desativação explícita (validar_regex=False)
    injetor_explicit = InjetorMTXSentinela(
        dir_mtx_origem=dir_origem,
        dir_mtx_destino=dir_destino,
        fill_value=0.5,
        validar_regex=False,
    )
    relatorio_explicit = injetor_explicit.injetar_colunas(
        colunas_1based=[2, 4], forcar=True
    )
    assert relatorio_explicit["status"] == "SUCESSO"
    assert relatorio_explicit["auditoria"]["status"] in ("APROVADO", "VALIDADO", "OK")
