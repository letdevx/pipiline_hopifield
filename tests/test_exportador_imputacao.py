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


def test_exportador_com_mask_ausentes_e_probabilidade(tmp_path: Path) -> None:
    """Valida a injeção da sentinela via mask_ausentes e a gravação de camadas prob e original."""
    n_celulas = 10
    n_genes = 5
    # w_orig sem sentinelas 0.5 (apenas 0.0 e 1.0)
    w_orig = np.zeros((n_celulas, n_genes), dtype=np.float32)
    w_orig[:, 0] = 1.0  # gene 0 observado ativo
    # genes 2 e 4 são ausentes na plataforma alvo
    mask_ausentes = np.array([False, False, True, False, True])

    # Hopfield recupera 1.0 no gene 2 e 0.0 no gene 4
    w_rec = np.zeros((n_celulas, n_genes), dtype=np.float32)
    w_rec[:, 0] = 1.0
    w_rec[:, 2] = 1.0

    w_prob = np.zeros((n_celulas, n_genes), dtype=np.float32)
    w_prob[:, 2] = 0.95

    exportador = ExportadorImputacao(out_dir=tmp_path, chunk_size=4)
    rel = exportador.exportar(
        w_original=w_orig,
        w_recuperado=w_rec,
        genes_canonica=[f"G_{i}" for i in range(n_genes)],
        mask_ausentes=mask_ausentes,
        w_probabilidade=w_prob,
        nome_modelo="teste_mask",
    )

    # 1. Deve resolver 10 * 2 = 20 coordenadas sentinelas
    assert rel["estatisticas_imputacao"]["total_sentinelas_resolvidos"] == 20
    assert rel["estatisticas_imputacao"]["valores_resolvidos_para_um"] == 10
    assert rel["estatisticas_imputacao"]["valores_resolvidos_para_zero"] == 10

    # 2. Carrega o AnnData gerado e audita as camadas
    adata = ad.read_h5ad(rel["arquivos_gerados"]["h5ad"])
    assert "original" in adata.layers
    assert "mascara_imputada" in adata.layers
    assert "probabilidade_imputada" in adata.layers

    orig_dense = adata.layers["original"].toarray()
    # Posições de mask_ausentes no layer original devem ser 0.5
    assert np.all(orig_dense[:, 2] == 0.5)
    assert np.all(orig_dense[:, 4] == 0.5)
    assert np.all(orig_dense[:, 0] == 1.0)
    assert np.all(orig_dense[:, 1] == 0.0)

    # Matriz X deve ter os valores imputados {1, 0}
    X_dense = adata.X.toarray()
    assert np.all(X_dense[:, 2] == 1.0)
    assert np.all(X_dense[:, 4] == 0.0)
    assert np.all(X_dense[:, 0] == 1.0)

    # Probabilidade deve conter o valor 0.95
    prob_dense = adata.layers["probabilidade_imputada"].toarray()
    assert np.all(prob_dense[:, 2] == 0.95)


def test_verificar_imputacao_completa_quando_arquivos_existem(tmp_path: Path) -> None:
    """Verifica se verificar_imputacao_completa retorna True quando todos os arquivos existem e não são vazios."""
    out_imp = tmp_path / "imputacao"
    out_imp.mkdir()
    out_mtx = out_imp / "mtx"
    out_mtx.mkdir()
    out_sweep = out_imp / "sweep"
    out_sweep.mkdir()
    out_top = tmp_path / "top_genes"
    out_top.mkdir()

    nome_modelo = "rede_teste"
    n_genes = 100
    base_nome = f"mathys_imputado_fujita_{nome_modelo}_{n_genes}genes"

    # Cria arquivos simulados não-vazios
    (out_imp / f"{base_nome}.h5ad").write_text("dummy h5ad")
    (out_imp / f"{base_nome}.npy").write_text("dummy npy")
    (out_imp / f"relatorio_{base_nome}.json").write_text("{}")
    (out_imp / "metricas_tipo_celular_mathys.csv").write_text("classe,f1\n1,0.9")
    (out_imp / "metricas_tipo_celular_mathys.json").write_text("[]")
    (out_imp / "painel_metricas_tipo_celular_mathys.png").write_text("dummy png")
    (out_mtx / "matrix.mtx").write_text("%%MatrixMarket")
    (out_mtx / "barcodes.tsv").write_text("cell1\n")
    (out_mtx / "features.tsv").write_text("gene1\n")
    (out_sweep / "sweep_alvo_pos_imputacao.txt").write_text("0.1 0.2")
    (out_top / f"X_mathys_IMPUTADO_{nome_modelo}.npy").write_text("dummy")
    (out_top / "X_mathys_IMPUTADO_rede180.npy").write_text("dummy")

    todos_ok, caminhos = ExportadorImputacao.verificar_imputacao_completa(
        out_imputacao=out_imp,
        nome_modelo=nome_modelo,
        n_genes=n_genes,
        out_mtx=out_mtx,
        out_sweep=out_sweep,
        out_top_genes=out_top,
    )

    assert todos_ok is True
    assert "h5ad" in caminhos
    assert "npy" in caminhos
    assert "mtx" in caminhos
    assert "sweep" in caminhos
    assert "metricas_csv" in caminhos
    assert "painel_png" in caminhos


def test_verificar_imputacao_completa_quando_arquivo_ausente_ou_vazio(
    tmp_path: Path,
) -> None:
    """Verifica se verificar_imputacao_completa retorna False se faltar arquivo ou arquivo for 0 bytes."""
    out_imp = tmp_path / "imputacao"
    out_imp.mkdir()

    nome_modelo = "rede_teste"
    n_genes = 50
    base_nome = f"mathys_imputado_fujita_{nome_modelo}_{n_genes}genes"

    # Sem criar arquivos
    todos_ok, _ = ExportadorImputacao.verificar_imputacao_completa(
        out_imputacao=out_imp,
        nome_modelo=nome_modelo,
        n_genes=n_genes,
    )
    assert todos_ok is False

    # Cria alguns arquivos mas deixa o h5ad vazio (0 bytes)
    (out_imp / f"{base_nome}.h5ad").touch()  # 0 bytes
    (out_imp / f"{base_nome}.npy").write_text("conteudo")
    (out_imp / f"relatorio_{base_nome}.json").write_text("{}")
    (out_imp / "metricas_tipo_celular_mathys.csv").write_text("col1,col2\n")
    (out_imp / "metricas_tipo_celular_mathys.json").write_text("[]")
    (out_imp / "painel_metricas_tipo_celular_mathys.png").write_text("img")

    todos_ok_vazio, _ = ExportadorImputacao.verificar_imputacao_completa(
        out_imputacao=out_imp,
        nome_modelo=nome_modelo,
        n_genes=n_genes,
    )
    # Falha porque h5ad tem 0 bytes
    assert todos_ok_vazio is False
