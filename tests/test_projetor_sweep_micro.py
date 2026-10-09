"""Testes unitários para validação de projeção ortonormal no espaço rSWeeP com micro-datasets."""

from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio
import scipy.sparse as sp

from src.config import PATH_ORTHBASE_RDS
from src.synthetic.gerador_ground_truth import GeradorGroundTruthSintetico
from src.treinamento.projetor_sweep import ProjetorSWeePR, ProjetorSWeP


def test_projetor_sweep_dimensao_adaptavel() -> None:
    """Verifica se o projetor SWeeP executa sem falhas com matrizes onde N_genes <= n_componentes,
    ajustando automaticamente a dimensão ou operando via decomposição QR adaptativa."""
    n_celulas = 12
    n_genes = 8
    n_comp_alvo = 4  # Dimensão reduzida compatível com o micro-dataset

    gerador = GeradorGroundTruthSintetico(
        n_celulas=n_celulas, n_genes=n_genes, n_classes=3, seed=77
    )
    matriz = gerador.gerar_matriz_pura(formato="numpy")
    assert isinstance(matriz, np.ndarray)

    projetor = ProjetorSWeP(n_features=n_genes, n_componentes=n_comp_alvo, seed=42)
    projetor.gerar_base()

    # Verifica ortogonalidade da matriz R
    R = projetor.R
    assert R is not None
    assert R.shape == (n_genes, n_comp_alvo), (
        f"Base R esperada {(n_genes, n_comp_alvo)}, obtida {R.shape}"
    )

    erro_ortogonalidade = np.abs(R.T @ R - np.eye(n_comp_alvo)).max()
    assert erro_ortogonalidade < 1e-5, (
        f"Erro de ortogonalidade excede limite aceitável: {erro_ortogonalidade}"
    )

    # Projeta os dados
    projetor.projetar(matriz)
    Wswp = projetor.Wswp
    assert Wswp is not None
    assert Wswp.shape == (n_celulas, n_comp_alvo), (
        "A projeção Wswp deve ter dimensões (células × componentes)."
    )


def test_projetor_sweep_preserva_separabilidade_de_classes() -> None:
    """Comprova que células do mesmo tipo biológico permanecem mais próximas no espaço SWeeP do que de outros tipos."""
    gerador = GeradorGroundTruthSintetico(n_celulas=12, n_genes=8, n_classes=3, seed=55)
    matriz = gerador.gerar_matriz_pura(formato="numpy")
    assert isinstance(matriz, np.ndarray)

    # Células 0..3 são Tipo A, Células 4..7 são Tipo B
    projetor = ProjetorSWeP(n_features=8, n_componentes=6, seed=123)
    projetor.gerar_base().projetar(matriz)
    Wswp = projetor.Wswp
    assert Wswp is not None

    # Distância L2 entre duas células do mesmo tipo (C0 e C1 - Tipo A)
    dist_mesma_classe = np.linalg.norm(Wswp[0] - Wswp[1])

    # Distância L2 entre células de tipos diferentes (C0 Tipo A e C5 Tipo B)
    dist_classes_distintas = np.linalg.norm(Wswp[0] - Wswp[5])

    assert dist_mesma_classe < dist_classes_distintas, (
        "A projeção SWeeP deve preservar a topologia e separabilidade biológica."
    )


def test_projetor_sweepr_execucao_r_canonica_e_congelamento(tmp_path: Path) -> None:
    """Valida a execução oficial do ProjetorSWeePR chamando o script R canônico com congelamento de base."""
    n_celulas = 15
    n_genes = 20
    n_comp = 6

    rng = np.random.default_rng(42)
    mat_esparsa = sp.csr_matrix(
        rng.integers(0, 2, size=(n_celulas, n_genes), dtype=np.int32)
    )

    path_mtx = tmp_path / "micro_teste.mtx"
    path_saida1 = tmp_path / "saida_sweep1.txt"
    path_saida2 = tmp_path / "saida_sweep2.txt"
    path_orthbase = tmp_path / "orthbase_congelada.rds"

    sio.mmwrite(str(path_mtx), mat_esparsa)

    # 1. Primeira projeção — gera e congela base
    projetor1 = ProjetorSWeePR(
        path_matriz=path_mtx,
        path_saida=path_saida1,
        n_componentes=n_comp,
        seed=42,
        path_orthbase=path_orthbase,
    )
    projetor1.projetar()

    assert path_saida1.exists(), "Arquivo de saída .txt deve existir."
    assert path_orthbase.exists(), "Arquivo RDS de base congelada deve existir."
    assert projetor1.Wswp is not None
    assert projetor1.Wswp.shape == (n_celulas, n_comp), (
        f"Shape inesperado: {projetor1.Wswp.shape}"
    )

    # 2. Segunda projeção — reutiliza base congelada e deve dar resultado idêntico
    projetor2 = ProjetorSWeePR(
        path_matriz=path_mtx,
        path_saida=path_saida2,
        n_componentes=n_comp,
        seed=999,  # semente diferente mas base congelada carregada
        path_orthbase=path_orthbase,
    )
    projetor2.projetar()

    assert projetor2.Wswp is not None
    np.testing.assert_allclose(
        projetor1.Wswp,
        projetor2.Wswp,
        rtol=1e-5,
        atol=1e-5,
        err_msg="A reutilização da base congelada deve produzir projeções matematicamente idênticas.",
    )


def test_projetor_sweepr_falha_estrita_sem_fallback(tmp_path: Path) -> None:
    """Verifica que ProjetorSWeePR dispara RuntimeError estrito e não mascara erros com fallbacks."""
    path_invalido = tmp_path / "arquivo_inexistente.mtx"
    path_saida = tmp_path / "saida_invalida.txt"

    projetor = ProjetorSWeePR(
        path_matriz=path_invalido,
        path_saida=path_saida,
        n_componentes=6,
        seed=42,
    )

    with pytest.raises(RuntimeError, match=r"FALHA CRÍTICA no subprocesso R"):
        projetor.projetar()


def test_projetor_sweepr_suporta_diretorio_mtx(tmp_path: Path) -> None:
    """Verifica se ProjetorSWeePR aceita um diretório contendo matrix.mtx (padrão ExportadorMTX)."""
    n_celulas = 10
    n_genes = 15
    n_comp = 5

    rng = np.random.default_rng(42)
    mat_esparsa = sp.csr_matrix(
        rng.integers(0, 2, size=(n_celulas, n_genes), dtype=np.int32)
    )

    pasta_mtx = tmp_path / "mtx_alvo_sentinela"
    pasta_mtx.mkdir()
    path_mtx = pasta_mtx / "matrix.mtx"
    sio.mmwrite(str(path_mtx), mat_esparsa)

    path_saida = tmp_path / "saida_diretorio_sweep.txt"
    path_orthbase = tmp_path / "orthbase_dir.rds"

    # Passa o diretório diretamente como path_matriz
    projetor = ProjetorSWeePR(
        path_matriz=pasta_mtx,
        path_saida=path_saida,
        n_componentes=n_comp,
        seed=42,
        path_orthbase=path_orthbase,
    )
    projetor.projetar()

    assert path_saida.exists()
    assert projetor.Wswp is not None
    assert projetor.Wswp.shape == (n_celulas, n_comp)


def test_projetor_sweepr_padrao_config_path_orthbase(tmp_path: Path) -> None:
    """Valida que ProjetorSWeePR assume automaticamente PATH_ORTHBASE_RDS do config quando path_orthbase é omitido."""
    path_matriz = tmp_path / "teste.mtx"
    path_saida = tmp_path / "saida.txt"

    projetor = ProjetorSWeePR(
        path_matriz=path_matriz,
        path_saida=path_saida,
        n_componentes=600,
        seed=42,
    )

    assert projetor.path_orthbase == str(PATH_ORTHBASE_RDS)
    assert not projetor.forcar_recriacao


def test_projetor_sweepr_forcar_recriacao(tmp_path: Path) -> None:
    """Valida que a flag forcar_recriacao=True força a regeneração e sobrescrita da base congelada."""
    n_celulas = 10
    n_genes = 12
    n_comp = 4

    rng = np.random.default_rng(42)
    mat_esparsa = sp.csr_matrix(
        rng.integers(0, 2, size=(n_celulas, n_genes), dtype=np.int32)
    )

    path_mtx = tmp_path / "matriz_recriacao.mtx"
    path_saida1 = tmp_path / "saida_recriacao1.txt"
    path_saida2 = tmp_path / "saida_recriacao2.txt"
    path_orthbase = tmp_path / "orthbase_recriacao.rds"

    sio.mmwrite(str(path_mtx), mat_esparsa)

    # 1. Primeira geração da base (seed=42)
    projetor1 = ProjetorSWeePR(
        path_matriz=path_mtx,
        path_saida=path_saida1,
        n_componentes=n_comp,
        seed=42,
        path_orthbase=path_orthbase,
    )
    projetor1.projetar()
    assert path_orthbase.exists()
    wswp1 = projetor1.Wswp
    assert wswp1 is not None

    # 2. Forçar recriação com semente diferente (seed=999, forcar_recriacao=True)
    projetor2 = ProjetorSWeePR(
        path_matriz=path_mtx,
        path_saida=path_saida2,
        n_componentes=n_comp,
        seed=999,
        path_orthbase=path_orthbase,
        forcar_recriacao=True,
    )
    projetor2.projetar()
    wswp2 = projetor2.Wswp
    assert wswp2 is not None

    # As projeções devem diferir porque uma nova base foi forçada com semente 999
    assert not np.allclose(wswp1, wswp2), (
        "Com forcar_recriacao=True e seed diferente, a base deve ser regenerada e as projeções devem diferir."
    )


def test_projetor_sweepr_sentinela_decomposta(tmp_path: Path) -> None:
    """Valida a projeção SWeeP canônica com decomposição exata da sentinela 0.5."""
    n_celulas = 10
    n_genes = 16
    n_comp = 4

    rng = np.random.default_rng(42)
    mat_densa = rng.integers(0, 2, size=(n_celulas, n_genes)).astype(np.float32)

    mask_ausentes = np.zeros(n_genes, dtype=bool)
    mask_ausentes[12:16] = True

    mat_com_sentinela = mat_densa.copy()
    mat_com_sentinela[:, mask_ausentes] = 0.5
    mat_com_sentinela_esparsa = sp.csr_matrix(mat_com_sentinela)

    mat_pura = mat_densa.copy()
    mat_pura[:, mask_ausentes] = 0.0
    mat_pura_esparsa = sp.csr_matrix(mat_pura)

    path_mtx_completo = tmp_path / "completo.mtx"
    path_mtx_puro = tmp_path / "puro.mtx"
    path_saida_completo = tmp_path / "saida_completo.txt"
    path_saida_decomposto = tmp_path / "saida_decomposto.txt"
    path_orthbase = tmp_path / "orthbase_sent.rds"
    path_mask_txt = tmp_path / "mask_ausentes.txt"

    sio.mmwrite(str(path_mtx_completo), mat_com_sentinela_esparsa)
    sio.mmwrite(str(path_mtx_puro), mat_pura_esparsa)

    indices_1based = [i + 1 for i in range(n_genes) if mask_ausentes[i]]
    with open(path_mask_txt, "w", encoding="utf-8") as f:
        for idx in indices_1based:
            f.write(f"{idx}\n")

    # 1. Projeção padrão da matriz com 0.5 gravado no MTX
    proj_padrao = ProjetorSWeePR(
        path_matriz=path_mtx_completo,
        path_saida=path_saida_completo,
        n_componentes=n_comp,
        seed=42,
        path_orthbase=path_orthbase,
    )
    proj_padrao.projetar()
    w_padrao = proj_padrao.Wswp
    assert w_padrao is not None

    # 2. Projeção da matriz pura com vetor sentinela decomposto
    proj_dec = ProjetorSWeePR(
        path_matriz=path_mtx_puro,
        path_saida=path_saida_decomposto,
        n_componentes=n_comp,
        seed=42,
        path_orthbase=path_orthbase,
        path_mask_sentinela=path_mask_txt,
    )
    proj_dec.projetar()
    w_dec = proj_dec.Wswp
    assert w_dec is not None

    np.testing.assert_allclose(w_padrao, w_dec, rtol=1e-5, atol=1e-5)
