"""Testes unitários para o AlinhadorEsparso e fluxo OOM-Safe."""

import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import pytest

from src.alinhamento.alinhador_esparso import AlinhadorEsparso
from src.alinhamento.selecionador_genes_frequentes import SelecionadorGenesFrequentes
from src.treinamento.gerador_conjunto_treinamento import GeradorConjuntoTreinamento


@pytest.fixture
def ambiente_teste():
    temp_dir = tempfile.mkdtemp()
    
    # Cria AnnData simulado Fujita (3 células x 4 genes: G0, G1, G2, G3)
    # G0: expressa em 3 cel, G1: expressa em 2 cel, G2: expressa em 1 cel, G3: expressa em 0 cel
    X_f = sp.csr_matrix(np.array([
        [1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0]
    ], dtype=np.float32))
    adata_f = ad.AnnData(X=X_f, var=pd.DataFrame(index=["G0", "G1", "G2", "G3"]))
    path_f = os.path.join(temp_dir, "adata_f.h5ad")
    adata_f.write_h5ad(path_f)

    # Cria AnnData simulado Mathys (2 células x 3 genes: G0, G1, G4) -> G2 e G3 ausentes em Mathys!
    X_m = sp.csr_matrix(np.array([
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0]
    ], dtype=np.float32))
    adata_m = ad.AnnData(X=X_m, var=pd.DataFrame(index=["G0", "G1", "G4"]))
    path_m = os.path.join(temp_dir, "adata_m.h5ad")
    adata_m.write_h5ad(path_m)

    # Ordem Canônica do Fujita: G0, G1, G2, G3
    genes_ordenados = ["G0", "G1", "G2", "G3"]
    gene_alvo_idx = {"G0": 0, "G1": 1, "G2": 2, "G3": 3}
    map_f = {"G0": "G0", "G1": "G1", "G2": "G2", "G3": "G3"}
    map_m = {"G0": "G0", "G1": "G1", "G4": "G4"} # G4 não está no Fujita

    out_alinhamento = os.path.join(temp_dir, "alinhamento")
    out_treinamento = os.path.join(temp_dir, "treinamento")
    os.makedirs(out_alinhamento, exist_ok=True)
    os.makedirs(out_treinamento, exist_ok=True)

    yield {
        "temp_dir": temp_dir,
        "path_f": path_f,
        "path_m": path_m,
        "genes_ordenados": genes_ordenados,
        "gene_alvo_idx": gene_alvo_idx,
        "map_f": map_f,
        "map_m": map_m,
        "out_alinhamento": out_alinhamento,
        "out_treinamento": out_treinamento,
    }

    shutil.rmtree(temp_dir, ignore_errors=True)


def test_alinhador_esparso_execucao_e_estrutura(ambiente_teste):
    cfg = ambiente_teste
    alinhador = AlinhadorEsparso(
        path_binarizada_m=cfg["path_m"],
        path_binarizada_f=cfg["path_f"],
        out_dir=cfg["out_alinhamento"],
        map_f=cfg["map_f"],
        map_m=cfg["map_m"],
        gene_alvo_idx=cfg["gene_alvo_idx"],
        genes_ordenados=cfg["genes_ordenados"],
    )
    alinhador.alinhar()

    assert os.path.exists(alinhador.path_f_alinhado)
    assert os.path.exists(alinhador.path_m_alinhado)

    adata_f_alin = ad.read_h5ad(alinhador.path_f_alinhado)
    adata_m_alin = ad.read_h5ad(alinhador.path_m_alinhado)

    # Verifica shapes (Fujita: 3x4, Mathys: 2x4)
    assert adata_f_alin.shape == (3, 4)
    assert adata_m_alin.shape == (2, 4)

    # Verifica se foram mantidos em formato esparso
    assert sp.issparse(adata_f_alin.X)
    assert sp.issparse(adata_m_alin.X)

    # Verifica metadados de presença
    assert np.all(adata_f_alin.var['presente_no_dataset'] == [True, True, True, True])
    assert np.array_equal(adata_m_alin.var['presente_no_dataset'].to_numpy(), [True, True, False, False])


def test_selecionador_genes_frequentes_h5ad(ambiente_teste):
    cfg = ambiente_teste
    alinhador = AlinhadorEsparso(
        path_binarizada_m=cfg["path_m"],
        path_binarizada_f=cfg["path_f"],
        out_dir=cfg["out_alinhamento"],
        map_f=cfg["map_f"],
        map_m=cfg["map_m"],
        gene_alvo_idx=cfg["gene_alvo_idx"],
        genes_ordenados=cfg["genes_ordenados"],
    ).alinhar()

    path_top_csv = os.path.join(cfg["out_alinhamento"], "top_genes.csv")
    selecionador = SelecionadorGenesFrequentes(path_h5ad=alinhador.path_f_alinhado, n=2)
    selecionador.calcular(out_csv=path_top_csv).salvar(path_top_csv)

    df_top = pd.read_csv(path_top_csv)
    # G0 (frequência 3) e G1 (frequência 2) devem ser os top 2
    assert list(df_top["gene"]) == ["G0", "G1"]
    assert list(df_top["frequencia"]) == [3, 2]


def test_extrair_subconjunto_com_sentinela(ambiente_teste):
    cfg = ambiente_teste
    alinhador = AlinhadorEsparso(
        path_binarizada_m=cfg["path_m"],
        path_binarizada_f=cfg["path_f"],
        out_dir=cfg["out_alinhamento"],
        map_f=cfg["map_f"],
        map_m=cfg["map_m"],
        gene_alvo_idx=cfg["gene_alvo_idx"],
        genes_ordenados=cfg["genes_ordenados"],
    ).alinhar()

    # Subconjunto com G0, G1 e G2 (G2 está ausente no Mathys)
    genes_sub = ["G0", "G1", "G2"]
    res = alinhador.extrair_subconjunto(
        lista_genes_ou_csv=genes_sub,
        out_dir=cfg["out_treinamento"],
        fill_value_mathys=0.5
    )

    assert os.path.exists(res["path_f_npy"])
    assert os.path.exists(res["path_m_npy"])

    X_f = np.load(res["path_f_npy"])
    X_m = np.load(res["path_m_npy"])

    assert X_f.shape == (3, 3)
    assert X_m.shape == (2, 3)

    # G2 é a 3ª coluna (índice 2). No Mathys, deve ter recebido 0.5 em todas as linhas!
    np.testing.assert_array_equal(X_m[:, 2], np.array([0.5, 0.5], dtype=np.float32))

    # G0 e G1 devem manter seus valores originais de Mathys:
    # Mathys original G0: [1, 0], G1: [0, 1]
    np.testing.assert_array_equal(X_m[:, 0], np.array([1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(X_m[:, 1], np.array([0.0, 1.0], dtype=np.float32))


def test_gerador_conjunto_treinamento_h5ad(ambiente_teste):
    cfg = ambiente_teste
    alinhador = AlinhadorEsparso(
        path_binarizada_m=cfg["path_m"],
        path_binarizada_f=cfg["path_f"],
        out_dir=cfg["out_alinhamento"],
        map_f=cfg["map_f"],
        map_m=cfg["map_m"],
        gene_alvo_idx=cfg["gene_alvo_idx"],
        genes_ordenados=cfg["genes_ordenados"],
    ).alinhar()

    path_top_csv = os.path.join(cfg["out_alinhamento"], "top_genes.csv")
    pd.DataFrame({"gene": ["G0", "G2"], "frequencia": [3, 1]}).to_csv(path_top_csv, index=False)

    gerador = GeradorConjuntoTreinamento(path_top_genes_csv=path_top_csv, out_dir=cfg["out_treinamento"])
    gerador.gerar_de_h5ad(alinhador.path_f_alinhado, is_mathys=False)
    gerador.gerar_de_h5ad(alinhador.path_m_alinhado, is_mathys=True, fill_value=0.5)

    path_f_npy = os.path.join(cfg["out_treinamento"], "adataF_binarizado_alinhado_top2.npy")
    path_m_npy = os.path.join(cfg["out_treinamento"], "adataM_binarizado_alinhado_top2.npy")

    assert os.path.exists(path_f_npy)
    assert os.path.exists(path_m_npy)

    X_m = np.load(path_m_npy)
    # G2 (índice 1 no subset) ausente no Mathys deve ser 0.5
    np.testing.assert_array_equal(X_m[:, 1], np.array([0.5, 0.5], dtype=np.float32))
