"""Testes unitários para o ValidadorFeatures (Validação Estrita Pré-Alinhamento)."""

import gzip

import anndata as ad
import numpy as np
import pytest
from scipy import sparse

from src.alinhamento.validador_features import ValidadorFeatures


@pytest.fixture
def tmp_features_dir(tmp_path):
    d = tmp_path / "features"
    d.mkdir()
    return d


def test_validacao_arquivo_features_correto(tmp_features_dir):
    """Testa se um arquivo de features correto (Col 0 = Ensembl, Col 1 = Symbol) passa na validação."""
    tsv_path = tmp_features_dir / "features_correto.tsv"
    data = [
        ("ENSG00000141510", "TP53"),
        ("ENSG00000133703", "KRAS"),
        ("ENSG00000012048", "BRCA1"),
        ("ENSG00000139618", "BRCA2"),
        ("ENSG00000171862", "PTEN"),
    ]
    with open(tsv_path, "w", encoding="utf-8") as f:
        for eid, sym in data:
            f.write(f"{eid}\t{sym}\n")

    validador = ValidadorFeatures()
    assert (
        validador.validar_arquivo_features(str(tsv_path), dataset_name="Teste") is True
    )


def test_validacao_arquivo_features_gzip_correto(tmp_features_dir):
    """Testa se um arquivo comprimido .tsv.gz correto passa na validação."""
    gz_path = tmp_features_dir / "features_correto.tsv.gz"
    data = [
        ("ENSG00000141510", "TP53"),
        ("ENSG00000133703", "KRAS"),
        ("ENSG00000012048", "BRCA1"),
    ]
    with gzip.open(gz_path, "wt", encoding="utf-8") as f:
        for eid, sym in data:
            f.write(f"{eid}\t{sym}\n")

    validador = ValidadorFeatures()
    assert (
        validador.validar_arquivo_features(str(gz_path), dataset_name="TesteGZ") is True
    )


def test_detecta_colunas_invertidas(tmp_features_dir):
    """Testa se a inversão de colunas (Col 0 = Symbol, Col 1 = Ensembl) dispara ValueError com diagnóstico."""
    tsv_path = tmp_features_dir / "features_invertido.tsv"
    data = [
        ("TP53", "ENSG00000141510"),
        ("KRAS", "ENSG00000133703"),
        ("BRCA1", "ENSG00000012048"),
        ("BRCA2", "ENSG00000139618"),
        ("PTEN", "ENSG00000171862"),
    ]
    with open(tsv_path, "w", encoding="utf-8") as f:
        for sym, eid in data:
            f.write(f"{sym}\t{eid}\n")

    validador = ValidadorFeatures()
    with pytest.raises(ValueError, match="COLUNAS INVERTIDAS"):
        validador.validar_arquivo_features(
            str(tsv_path), dataset_name="MathysInvertido"
        )


def test_arquivo_com_uma_coluna_apenas(tmp_features_dir):
    """Testa se arquivo com apenas 1 coluna dispara ValueError."""
    tsv_path = tmp_features_dir / "features_uma_coluna.tsv"
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("TP53\nKRAS\nBRCA1\n")

    validador = ValidadorFeatures()
    with pytest.raises(ValueError, match="possui apenas 1 coluna"):
        validador.validar_arquivo_features(str(tsv_path), dataset_name="Invalido")


def test_validacao_compatibilidade_anndata_sucesso(tmp_path):
    """Testa se var_names compatíveis com as chaves do mapa passam na validação."""
    genes = ["TP53", "KRAS", "BRCA1", "BRCA2", "PTEN"]
    X = sparse.csr_matrix(np.ones((10, len(genes)), dtype=np.float32))
    adata = ad.AnnData(X=X, var=dict(gene_symbols=genes))
    adata.var_names = genes

    h5ad_path = tmp_path / "test_matrix.h5ad"
    adata.write_h5ad(h5ad_path)

    map_features = {
        "TP53": "ENSG00000141510",
        "KRAS": "ENSG00000133703",
        "BRCA1": "ENSG00000012048",
        "BRCA2": "ENSG00000139618",
        "PTEN": "ENSG00000171862",
    }

    validador = ValidadorFeatures(min_match_pct=50.0)
    assert (
        validador.validar_compatibilidade_anndata(
            str(h5ad_path), map_features, dataset_name="FujitaTest"
        )
        is True
    )
    assert (
        validador.validar_compatibilidade_anndata(
            adata, map_features, dataset_name="FujitaTestInMemory"
        )
        is True
    )


def test_detecta_mismatch_var_names_anndata(tmp_path):
    """Testa se discrepância entre var_names do AnnData e o mapa de features dispara ValueError."""
    # Matriz usa códigos de camundongo ou IDs inexistentes
    genes_adata = [f"UNKNOWN_GENE_{i}" for i in range(100)]
    X = sparse.csr_matrix(np.ones((5, 100), dtype=np.float32))
    adata = ad.AnnData(X=X)
    adata.var_names = genes_adata

    h5ad_path = tmp_path / "test_mismatch.h5ad"
    adata.write_h5ad(h5ad_path)

    map_features = {
        "TP53": "ENSG00000141510",
        "KRAS": "ENSG00000133703",
        "BRCA1": "ENSG00000012048",
    }

    validador = ValidadorFeatures(min_match_pct=50.0)
    with pytest.raises(ValueError, match="INCOMPATIBILIDADE DE IDENTIFICADORES"):
        validador.validar_compatibilidade_anndata(
            str(h5ad_path), map_features, dataset_name="MismatchTest"
        )


def test_sobreposicao_inter_dataset(tmp_path):
    """Testa validação de sobreposição compartilhada entre dois mapas de features."""
    map_f = {f"GENE_{i}": f"ENSG{i:08d}" for i in range(2000)}
    map_m = {f"GENE_{i}": f"ENSG{i:08d}" for i in range(500, 2500)}  # 1500 em comum

    validador = ValidadorFeatures(min_genes_comuns=1000)
    assert validador.validar_sobreposicao_inter_dataset(map_f, map_m) is True

    # Caso com sobreposição insuficiente (< 1000)
    map_m_insuficiente = {
        f"GENE_{i}": f"ENSG{i:08d}" for i in range(1900, 3000)
    }  # apenas 100 em comum
    with pytest.raises(ValueError, match="SOBREPOSIÇÃO GENÔMICA ANORMALMENTE BAIXA"):
        validador.validar_sobreposicao_inter_dataset(map_f, map_m_insuficiente)


def test_validar_tudo_fluxo_completo(tmp_path, tmp_features_dir):
    """Testa o método orquestrador validar_tudo."""
    tsv_ref = tmp_features_dir / "features_ref.tsv"
    tsv_alvo = tmp_features_dir / "features_alvo.tsv"

    with open(tsv_ref, "w") as f:
        for i in range(1500):
            f.write(f"ENSG{i:08d}\tGENE_{i}\n")

    with open(tsv_alvo, "w") as f:
        for i in range(500, 2000):
            f.write(f"ENSG{i:08d}\tGENE_{i}\n")

    map_f = {f"GENE_{i}": f"ENSG{i:08d}" for i in range(1500)}
    map_m = {f"GENE_{i}": f"ENSG{i:08d}" for i in range(500, 2000)}

    genes_f = [f"GENE_{i}" for i in range(1500)]
    genes_m = [f"GENE_{i}" for i in range(500, 2000)]

    adata_f = ad.AnnData(X=sparse.csr_matrix((10, len(genes_f)), dtype=np.float32))
    adata_f.var_names = genes_f
    path_f = tmp_path / "fujita.h5ad"
    adata_f.write_h5ad(path_f)

    adata_m = ad.AnnData(X=sparse.csr_matrix((10, len(genes_m)), dtype=np.float32))
    adata_m.var_names = genes_m
    path_m = tmp_path / "mathys.h5ad"
    adata_m.write_h5ad(path_m)

    validador = ValidadorFeatures(min_match_pct=50.0, min_genes_comuns=1000)
    assert (
        validador.validar_tudo(
            path_features_ref=str(tsv_ref),
            path_features_alvo=str(tsv_alvo),
            path_h5ad_ref=str(path_f),
            path_h5ad_alvo=str(path_m),
            map_f=map_f,
            map_m=map_m,
        )
        is True
    )
