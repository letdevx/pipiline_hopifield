"""Testes unitários para o script de automação sweepPan.py."""

from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio
import scipy.sparse as sp

sweepPan = pytest.importorskip("sweepPan", reason="sweepPan.py não encontrado na raiz")
main = sweepPan.main
projetar_matriz_pan05 = sweepPan.projetar_matriz_pan05
resolver_caminho_entrada = sweepPan.resolver_caminho_entrada
resolver_caminhos = sweepPan.resolver_caminhos


def test_resolver_caminhos_padrao() -> None:
    """Verifica se os caminhos padrão do pipeline são resolvidos adequadamente."""
    c_mat, c_txt, c_npy, c_rds = resolver_caminhos()

    assert any(
        nome in c_mat.lower()
        for nome in ["matrix_pan", "matriz_pan", "pan05", "pan-05"]
    )
    assert c_txt.endswith("matriz_reduzida_sweepPan05.txt")
    assert c_npy.endswith("matriz_reduzida_sweepPan05.npy")
    assert c_rds.endswith("orthbase_mproj_600d.rds")


def test_resolver_caminhos_customizados(tmp_path: Path) -> None:
    """Verifica a resolução com parâmetros customizados."""
    mat_custom = tmp_path / "custom.mtx"
    mat_custom.write_text("%%MatrixMarket matrix coordinate real general\n1 1 0\n")
    txt_custom = tmp_path / "saida.txt"
    npy_custom = tmp_path / "saida.npy"
    rds_custom = tmp_path / "base.rds"

    c_mat, c_txt, c_npy, c_rds = resolver_caminhos(
        path_matriz=mat_custom,
        path_saida_txt=txt_custom,
        path_saida_npy=npy_custom,
        path_orthbase=rds_custom,
        path_base=tmp_path,
    )

    assert Path(c_mat) == mat_custom.resolve()
    assert Path(c_txt) == txt_custom.resolve()
    assert Path(c_npy) == npy_custom.resolve()
    assert Path(c_rds) == rds_custom.resolve()


def test_resolver_caminho_entrada_inexistente_erro() -> None:
    """Dispara FileNotFoundError quando um caminho explícito inexistente é fornecido."""
    with pytest.raises(FileNotFoundError, match="não foi encontrado"):
        resolver_caminho_entrada("caminho_totalmente_inexistente_12345.mtx")


def test_resolver_caminho_entrada_encontra_existente(tmp_path: Path) -> None:
    """Encontra automaticamente arquivo candidato na pasta imputs."""
    dir_imputs = tmp_path / "imputs"
    dir_imputs.mkdir(parents=True, exist_ok=True)
    arquivo_cand = dir_imputs / "matriz_Pan05.mtx"
    arquivo_cand.write_text("dummy content")

    resolvido = resolver_caminho_entrada(path_base=tmp_path)
    assert Path(resolvido) == arquivo_cand.resolve()


def test_sweep_pan_projecao_micro_mtx(tmp_path: Path) -> None:
    """Valida o fluxo completo de projeção com micro-dataset gerando .txt e .npy."""
    n_celulas = 10
    n_genes = 12
    n_comp = 4

    rng = np.random.default_rng(101)
    dados = rng.integers(0, 2, size=(n_celulas, n_genes), dtype=np.int32)
    mat_esparsa = sp.csr_matrix(dados)

    path_matriz = tmp_path / "micro_pan05.mtx"
    sio.mmwrite(str(path_matriz), mat_esparsa)

    path_txt = tmp_path / "matriz_reduzida_sweepPan05.txt"
    path_npy = tmp_path / "matriz_reduzida_sweepPan05.npy"
    path_rds = tmp_path / "orthbase_micro.rds"

    # Executa com verificar_r=False no teste para reutilizar o R já instalado sem overhead de checagem
    wswp = projetar_matriz_pan05(
        path_matriz=path_matriz,
        path_saida_txt=path_txt,
        path_saida_npy=path_npy,
        path_orthbase=path_rds,
        n_componentes=n_comp,
        seed=42,
        verificar_r=False,
    )

    assert isinstance(wswp, np.ndarray)
    assert wswp.shape == (n_celulas, n_comp)
    assert path_txt.exists()
    assert path_npy.exists()
    assert path_rds.exists()

    # Valida consistência entre array em memória e arquivo .npy salvo
    wswp_carregado = np.load(path_npy)
    np.testing.assert_allclose(wswp, wswp_carregado, rtol=1e-5)


def test_main_cli(tmp_path: Path) -> None:
    """Testa a interface de linha de comando main() com argumentos válidos."""
    n_celulas = 8
    n_genes = 10
    n_comp = 4

    rng = np.random.default_rng(202)
    mat_esparsa = sp.csr_matrix(
        rng.integers(0, 2, size=(n_celulas, n_genes), dtype=np.int32)
    )

    path_matriz = tmp_path / "cli_pan05.mtx"
    sio.mmwrite(str(path_matriz), mat_esparsa)

    path_txt = tmp_path / "saida_cli.txt"
    path_npy = tmp_path / "saida_cli.npy"
    path_rds = tmp_path / "base_cli.rds"

    argv = [
        "--input",
        str(path_matriz),
        "--output-txt",
        str(path_txt),
        "--output-npy",
        str(path_npy),
        "--orthbase",
        str(path_rds),
        "--n-componentes",
        str(n_comp),
        "--seed",
        "42",
        "--sem-verificacao-r",
    ]

    codigo_retorno = main(argv)
    assert codigo_retorno == 0
    assert path_txt.exists()
    assert path_npy.exists()
    assert path_rds.exists()


def test_main_cli_falha_retorna_codigo_erro() -> None:
    """Testa se main() retorna código 1 ao ocorrer erro em vez de estourar exceção não capturada."""
    argv = ["--input", "arquivo_inexistente_para_erro.mtx"]
    codigo_retorno = main(argv)
    assert codigo_retorno == 1
