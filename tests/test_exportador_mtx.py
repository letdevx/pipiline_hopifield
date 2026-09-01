"""Testes unitários para o ExportadorMTX."""

import anndata as ad
import numpy as np
import pytest
import scipy.io as sio
import scipy.sparse as sp

from src.alinhamento.exportador_mtx import ExportadorMTX


def test_exportador_mtx_anndata(tmp_path):
    """Testa exportação completa a partir de objeto AnnData."""
    out_dir = tmp_path / "mtx_anndata"
    n_celulas, n_genes = 20, 5
    genes = [f"ENSG0000000000{i}" for i in range(1, n_genes + 1)]
    barcodes = [f"cell_barcode_{i}" for i in range(n_celulas)]

    rng = np.random.default_rng(42)
    x_dense = (rng.random((n_celulas, n_genes)) > 0.7).astype(np.float32)
    x_csr = sp.csr_matrix(x_dense)

    adata = ad.AnnData(
        X=x_csr,
        obs={"tipo": ["Excitatory"] * n_celulas},
        var={"simbolo": ["GENE_A", "GENE_B", "GENE_C", "GENE_D", "GENE_E"]},
    )
    adata.obs_names = barcodes
    adata.var_names = genes

    map_feat = {g: f"SYM_{i}" for i, g in enumerate(genes)}

    exportador = ExportadorMTX(out_dir=out_dir)
    res = exportador.exportar(
        matriz=adata,
        genes_referencia=genes,
        map_features=map_feat,
        nome_etapa="Teste AnnData",
    )

    assert res["status"] == "APROVADO"
    assert res["n_celulas"] == n_celulas
    assert res["n_genes"] == n_genes

    # Lê de volta o matrix.mtx gravado
    mtx_read = sio.mmread(str(out_dir / "matrix.mtx"))
    assert mtx_read.shape == (n_celulas, n_genes)
    assert np.allclose(mtx_read.toarray(), x_dense)

    # Confere genes_referencia.tsv
    with open(out_dir / "genes_referencia.tsv", encoding="utf-8") as f:
        linhas = [line.strip().split("\t") for line in f if line.strip()]
    assert len(linhas) == n_genes
    assert linhas[0] == [genes[0], map_feat[genes[0]]]

    # Confere barcodes.tsv
    with open(out_dir / "barcodes.tsv", encoding="utf-8") as f:
        barcodes_read = [line.strip() for line in f if line.strip()]
    assert barcodes_read == barcodes


def test_exportador_mtx_matriz_esparsa_direta(tmp_path):
    """Testa exportação a partir de matriz esparsa direta CSR e arrays de nomes."""
    out_dir = tmp_path / "mtx_esparsa"
    genes = ["ENSG00000141510", "ENSG00000133703", "ENSG00000012048"]
    matriz = sp.csr_matrix([[1.0, 0.0, 0.5], [0.0, 1.0, 0.0]])

    exportador = ExportadorMTX(out_dir=out_dir)
    res = exportador.exportar(
        matriz=matriz,
        genes=genes,
        genes_referencia=genes,
        nome_etapa="Teste Esparsa",
    )

    assert res["status"] == "APROVADO"
    assert res["n_celulas"] == 2
    assert res["n_genes"] == 3


def test_exportador_mtx_rejeita_genes_divergentes(tmp_path):
    """Testa se o validador integrado aborta a gravação se houver genes fora de ordem."""
    out_dir = tmp_path / "mtx_erro"
    genes_ref = ["ENSG00000141510", "ENSG00000133703"]
    genes_errados = ["ENSG00000133703", "ENSG00000141510"]  # Trocados
    matriz = sp.csr_matrix([[1.0, 0.0]])

    exportador = ExportadorMTX(out_dir=out_dir)
    with pytest.raises(
        ValueError, match="Divergência na ordem de colunas gênicas detectada"
    ):
        exportador.exportar(
            matriz=matriz,
            genes=genes_errados,
            genes_referencia=genes_ref,
        )
