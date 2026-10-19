"""Testes unitários automatizados para o componente MarcadorSentinela."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from src.alinhamento.marcador_sentinela import MarcadorSentinela

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def fixture_dados_sinteticos() -> tuple[
    ad.AnnData, list[str], list[str], dict[str, str]
]:
    """Gera dados sintéticos compactos (50 células × 60 genes) e referência canônica (100 genes)."""
    rng = np.random.default_rng(42)
    n_obs = 50
    n_vars_alvo = 60
    n_vars_ref = 100

    genes_alvo = [f"ENSG{i:011d}" for i in range(1, n_vars_alvo + 1)]
    genes_ref = [f"ENSG{i:011d}" for i in range(1, n_vars_ref + 1)]

    # Matriz binarizada esparsa (80% zeros, 20% uns)
    x_dense = rng.choice([0.0, 1.0], size=(n_obs, n_vars_alvo), p=[0.8, 0.2]).astype(
        np.float32
    )
    x_csr = sp.csr_matrix(x_dense)

    obs_df = pd.DataFrame(
        {
            "tipo_celular": rng.choice(["Astrocito", "Neuronio"], size=n_obs),
            "doador": rng.choice(["D1", "D2"], size=n_obs),
        },
        index=[f"celula_{i:04d}" for i in range(n_obs)],
    )

    var_df = pd.DataFrame(index=genes_alvo)
    adata_alvo = ad.AnnData(X=x_csr, obs=obs_df, var=var_df)

    map_features = {f"Gene_{g}": g for g in genes_alvo}

    return adata_alvo, genes_alvo, genes_ref, map_features


def test_marcador_sentinela_alinhamento_e_injecao_fisica(
    tmp_path: Path, fixture_dados_sinteticos: tuple[Any, ...]
) -> None:
    """Valida a injeção física de 0.5 nas colunas ausentes e preservação dos dados originais."""
    adata_alvo, _, genes_ref, _ = fixture_dados_sinteticos
    path_h5ad = tmp_path / "alvo_sentinela.h5ad"
    path_track = tmp_path / "tracking.csv"

    marcador = MarcadorSentinela(
        genes_referencia=genes_ref, fill_value=0.5, chunk_size=15
    )
    adata_res = marcador.marcar_dataset(
        adata_alvo=adata_alvo,
        path_saida_h5ad=path_h5ad,
        path_tracking_csv=path_track,
        forcar=True,
    )

    # 1. Dimensões e tipo
    assert adata_res.shape == (50, 100)
    assert sp.issparse(adata_res.X)
    assert path_h5ad.exists()
    assert path_track.exists()

    # 2. Máscara de ausentes (60 medidos -> 40 ausentes)
    mask_aus = marcador.obter_mascara_ausentes()
    assert mask_aus.shape == (100,)
    assert int(np.sum(mask_aus)) == 40
    assert int(np.sum(~mask_aus)) == 60

    # 3. Injeção física de 0.5 nas colunas ausentes
    x_mat = (
        adata_res.X.toarray() if sp.issparse(adata_res.X) else np.asarray(adata_res.X)
    )
    colunas_ausentes = np.where(mask_aus)[0]
    colunas_presentes = np.where(~mask_aus)[0]

    for col in colunas_ausentes:
        assert np.allclose(x_mat[:, col], 0.5)

    # 4. Preservação dos valores medidos nas primeiras 60 colunas
    x_orig = adata_alvo.X.toarray()
    for col in colunas_presentes:
        assert np.allclose(x_mat[:, col], x_orig[:, col])

    # 5. Metadados .var e .obs
    assert "presente_no_dataset" in adata_res.var
    assert "ausente" in adata_res.var
    assert "valor_sentinela" in adata_res.var
    assert list(adata_res.obs["tipo_celular"]) == list(adata_alvo.obs["tipo_celular"])


def test_marcador_sentinela_idempotencia(
    tmp_path: Path, fixture_dados_sinteticos: tuple[Any, ...]
) -> None:
    """Verifica se chamadas subsequentes com forcar=False reutilizam o arquivo já existente."""
    adata_alvo, _, genes_ref, _ = fixture_dados_sinteticos
    path_h5ad = tmp_path / "alvo_idempotente.h5ad"

    marcador = MarcadorSentinela(
        genes_referencia=genes_ref, fill_value=0.5, chunk_size=20
    )
    _ = marcador.marcar_dataset(
        adata_alvo=adata_alvo,
        path_saida_h5ad=path_h5ad,
        forcar=True,
    )
    mtime_inicial = path_h5ad.stat().st_mtime

    # Segunda chamada não deve regravar o arquivo
    adata_cached = marcador.marcar_dataset(
        adata_alvo=adata_alvo,
        path_saida_h5ad=path_h5ad,
        forcar=False,
    )
    assert path_h5ad.stat().st_mtime == mtime_inicial
    assert adata_cached.shape == (50, 100)


def test_marcador_sentinela_exportar_mtx_e_tracking(
    tmp_path: Path, fixture_dados_sinteticos: tuple[Any, ...]
) -> None:
    """Valida a exportação Matrix Market e o relatório CSV de rastreamento."""
    adata_alvo, _, genes_ref, _ = fixture_dados_sinteticos
    path_h5ad = tmp_path / "alvo_mtx.h5ad"
    path_mtx_dir = tmp_path / "mtx_saida"
    path_track = tmp_path / "track_mtx.csv"

    marcador = MarcadorSentinela(
        genes_referencia=genes_ref, fill_value=0.5, chunk_size=25
    )
    _ = marcador.marcar_dataset(
        adata_alvo=adata_alvo,
        path_saida_h5ad=path_h5ad,
        path_tracking_csv=path_track,
        exportar_mtx=True,
        path_saida_mtx=path_mtx_dir,
        forcar=True,
    )

    assert (path_mtx_dir / "matrix.mtx").exists()
    assert (path_mtx_dir / "genes_referencia.tsv").exists()
    assert (path_mtx_dir / "barcodes.tsv").exists()

    df_track = pd.read_csv(path_track)
    assert len(df_track) == 40
    assert set(df_track.columns) >= {
        "posicao_coluna",
        "ensembl_id",
        "valor_inserido",
        "presente_referencia",
        "presente_alvo",
    }


def test_marcador_sentinela_entrada_polimorfica(
    tmp_path: Path, fixture_dados_sinteticos: tuple[Any, ...]
) -> None:
    """Valida a carga polimórfica de referência a partir de arquivo CSV."""
    _, _, genes_ref, _ = fixture_dados_sinteticos
    path_csv_ref = tmp_path / "genes_canonicos.csv"
    pd.DataFrame({"gene": genes_ref}).to_csv(path_csv_ref, index=False)

    marcador = MarcadorSentinela(genes_referencia=path_csv_ref)
    assert len(marcador.genes_canonica) == 100
    assert marcador.genes_canonica[0] == "ENSG00000000001"
    assert marcador.genes_canonica[-1] == "ENSG00000000100"


def test_marcador_sentinela_chunk_size_equivalencia(
    tmp_path: Path, fixture_dados_sinteticos: tuple[Any, ...]
) -> None:
    """Valida que processar em micro-chunks ou em chunk único gera matrizes idênticas."""
    adata_alvo, _, genes_ref, _ = fixture_dados_sinteticos

    marcador_micro = MarcadorSentinela(genes_referencia=genes_ref, chunk_size=5)
    res_micro = marcador_micro.marcar_dataset(
        adata_alvo=adata_alvo,
        path_saida_h5ad=tmp_path / "micro.h5ad",
        forcar=True,
    )

    marcador_macro = MarcadorSentinela(genes_referencia=genes_ref, chunk_size=50)
    res_macro = marcador_macro.marcar_dataset(
        adata_alvo=adata_alvo,
        path_saida_h5ad=tmp_path / "macro.h5ad",
        forcar=True,
    )

    diff = np.abs((res_micro.X - res_macro.X).toarray()).max()
    assert diff < 1e-6
