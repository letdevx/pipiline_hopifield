"""Testes unitários para o módulo ExportadorImputacao."""

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from src.treinamento.exportador_imputacao import ExportadorImputacao


def test_exportador_imputacao_completo_h5ad_layers_e_npy(tmp_path: Path) -> None:
    """Valida a exportação completa de matriz imputada em AnnData (com layers), NPY e JSON."""
    n_celulas = 24
    n_genes = 8
    chunk_size = 6

    rng = np.random.default_rng(42)

    # Matriz original: valores binários com genes sentinela 0.5
    w_orig = rng.choice([0.0, 1.0], size=(n_celulas, n_genes)).astype(np.float32)
    # Coluna 2 e 5 são inteiramente ausentes (sentinelas 0.5 em todas as células)
    w_orig[:, 2] = 0.5
    w_orig[:, 5] = 0.5
    # Algumas posições pontuais também recebem sentinela
    w_orig[0, 0] = 0.5
    w_orig[3, 1] = 0.5

    # Matriz recuperada pela Hopfield: predições binarizadas {0, 1}
    w_rec = rng.choice([0.0, 1.0], size=(n_celulas, n_genes)).astype(np.float32)

    genes_canonica = [f"ENSG0000000{i}" for i in range(n_genes)]
    map_features = {f"ENSG0000000{i}": f"GENE_{i}" for i in range(n_genes)}

    classes_reais = rng.integers(1, 8, size=n_celulas)
    pred_classes = rng.integers(1, 8, size=n_celulas)
    prototipos_idx = rng.integers(0, 35, size=n_celulas)

    info_modelo = {"beta": 8.0, "nc": 10, "n_padroes": 35}

    exportador = ExportadorImputacao(out_dir=tmp_path, chunk_size=chunk_size)
    relatorio = exportador.exportar(
        w_original=w_orig,
        w_recuperado=w_rec,
        genes_canonica=genes_canonica,
        map_features=map_features,
        pred_classes=pred_classes,
        classes_reais=classes_reais,
        prototipos_idx=prototipos_idx,
        info_modelo=info_modelo,
        nome_modelo="teste_rede",
        exportar_npy=True,
    )

    # 1. Verifica integridade do relatório retornado
    assert relatorio["dimensoes"]["n_celulas"] == n_celulas
    assert relatorio["dimensoes"]["n_genes"] == n_genes
    assert relatorio["estatisticas_imputacao"]["total_sentinelas_resolvidos"] == int(
        np.sum(w_orig == 0.5)
    )

    path_h5ad = relatorio["arquivos_gerados"]["h5ad"]
    path_npy = relatorio["arquivos_gerados"]["npy"]
    path_json = relatorio["arquivos_gerados"]["relatorio_json"]

    assert Path(path_h5ad).exists()
    assert Path(path_npy).exists()
    assert Path(path_json).exists()

    # 2. Inspeciona o arquivo JSON em disco
    with open(path_json, encoding="utf-8") as f:
        dados_json = json.load(f)
    assert dados_json["modelo"] == "teste_rede"
    assert "estatisticas_imputacao" in dados_json

    # 3. Inspeciona o arquivo AnnData (.h5ad)
    adata = ad.read_h5ad(path_h5ad)
    assert adata.n_obs == n_celulas
    assert adata.n_vars == n_genes
    assert sp.issparse(adata.X)

    X_denso = sp.csr_matrix(adata.X).toarray()

    # Validação biológica e matemática da substituição:
    # Onde w_orig == 0.5, deve receber exatamente w_rec
    # Onde w_orig != 0.5, deve preservar w_orig original
    esperado = np.where(w_orig == 0.5, w_rec, w_orig)
    np.testing.assert_array_almost_equal(X_denso, esperado)

    # 4. Valida layers
    assert "original" in adata.layers
    assert "mascara_imputada" in adata.layers
    orig_denso = sp.csr_matrix(adata.layers["original"]).toarray()
    mask_denso = sp.csr_matrix(adata.layers["mascara_imputada"]).toarray()

    np.testing.assert_array_almost_equal(orig_denso, w_orig)
    np.testing.assert_array_almost_equal(mask_denso, (w_orig == 0.5).astype(np.float32))

    # 5. Valida obs
    assert "tipo_celular_real" in adata.obs.columns
    assert "tipo_predito_hopfield" in adata.obs.columns
    assert "prototipo_hopfield_idx" in adata.obs.columns
    assert "n_genes_imputados" in adata.obs.columns
    assert "pct_genes_imputados" in adata.obs.columns
    np.testing.assert_array_equal(
        adata.obs["tipo_celular_real"].to_numpy(), classes_reais
    )

    # 6. Valida var
    assert isinstance(adata.var, pd.DataFrame)
    assert "gene_symbol" in adata.var.columns
    assert "gene_imputado" in adata.var.columns
    assert list(adata.var.index) == genes_canonica
    assert (
        adata.var.loc["ENSG00000002", "gene_imputado"] is True
        or adata.var.loc["ENSG00000002", "gene_imputado"] == 1
    )
    assert adata.var.loc["ENSG00000001", "gene_symbol"] == "GENE_1"

    # 7. Valida uns
    assert adata.uns["modelo"] == "teste_rede"
    assert adata.uns["dataset_referencia"] == "Fujita"
    assert adata.uns["dataset_alvo"] == "Mathys"
    assert "parametros_modelo" in adata.uns

    # 8. Valida arquivo .npy
    arr_npy = np.load(path_npy)
    np.testing.assert_array_almost_equal(arr_npy, esperado)


def test_exportador_imputacao_com_adata_orig_obs(tmp_path: Path) -> None:
    """Valida a preservação autêntica de metadados celulares (obs) a partir de um AnnData alvo."""
    n_celulas = 10
    n_genes = 4

    barcodes = [f"AAACCGG_{i}-1" for i in range(n_celulas)]
    obs_custom = pd.DataFrame(
        {
            "doador_id": [f"D_{i % 2}" for i in range(n_celulas)],
            "diagnostico": [
                "AD" if i % 2 == 0 else "Controle" for i in range(n_celulas)
            ],
        },
        index=pd.Index(barcodes, name="barcode"),
    )

    path_src_h5ad = tmp_path / "mathys_alvo_mock.h5ad"
    adata_src = ad.AnnData(
        X=sp.csr_matrix(np.zeros((n_celulas, n_genes), dtype=np.float32)),
        obs=obs_custom,
        var=pd.DataFrame(index=[f"G_{i}" for i in range(n_genes)]),
    )
    adata_src.write_h5ad(path_src_h5ad)

    w_orig = np.zeros((n_celulas, n_genes), dtype=np.float32)
    w_orig[:, 1] = 0.5
    w_rec = np.ones((n_celulas, n_genes), dtype=np.float32)

    exportador = ExportadorImputacao(out_dir=tmp_path)
    relatorio = exportador.exportar(
        w_original=w_orig,
        w_recuperado=w_rec,
        genes_canonica=[f"G_{i}" for i in range(n_genes)],
        adata_alvo_original=path_src_h5ad,
        nome_modelo="modelo_obs_test",
        exportar_npy=False,
    )

    adata_res = ad.read_h5ad(relatorio["arquivos_gerados"]["h5ad"])
    assert list(adata_res.obs_names) == barcodes
    assert "doador_id" in adata_res.obs.columns
    assert "diagnostico" in adata_res.obs.columns
    assert relatorio["arquivos_gerados"]["npy"] is None


def test_exportador_validacoes_dimensoes_incompativeis(tmp_path: Path) -> None:
    """Valida se o exportador levanta exceções claras em caso de dimensões divergentes."""
    exportador = ExportadorImputacao(out_dir=tmp_path)
    w_orig = np.zeros((5, 4), dtype=np.float32)
    w_rec = np.zeros((5, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="Formatos incompatíveis"):
        exportador.exportar(
            w_original=w_orig,
            w_recuperado=w_rec,
            genes_canonica=["G0", "G1", "G2", "G3"],
        )

    with pytest.raises(ValueError, match="Inconsistência de dimensões"):
        exportador.exportar(
            w_original=w_orig,
            w_recuperado=w_orig,
            genes_canonica=["G0", "G1"],
        )
