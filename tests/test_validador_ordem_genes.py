"""Testes unitários para ValidadorOrdemGenes e sanitização de Ensembl IDs."""

import pytest
import scipy.sparse as sp

from src.alinhamento.leitor_features import LeitorFeatures
from src.alinhamento.validador_ordem_genes import ValidadorOrdemGenes


def test_validador_ensembl_regex_sucesso():
    """Testa se IDs Ensembl estáveis de 11 dígitos sem versão passam com sucesso."""
    genes_validos = [
        "ENSG00000141510",
        "ENSG00000133703",
        "ENSG00000012048",
        "ENSMUSG00000012345",
    ]
    validador = ValidadorOrdemGenes()
    res = validador.validar_formato_ensembl(genes_validos)
    assert res["valido"] is True
    assert res["invalidos_qtd"] == 0


def test_validador_ensembl_regex_rejeita_versao():
    """Testa se IDs com versão (.1, .3) são explicitamente rejeitados pela regex canônica."""
    genes_com_versao = [
        "ENSG00000141510.3",
        "ENSG00000133703.14",
    ]
    validador = ValidadorOrdemGenes()
    res = validador.validar_formato_ensembl(genes_com_versao)
    assert res["valido"] is False
    assert res["invalidos_qtd"] == 2
    assert len(res["invalidos_amostra"]) == 2


def test_validador_ensembl_regex_rejeita_simbolos_e_invalidos():
    """Testa se símbolos gênicos e IDs corrompidos são identificados como inválidos."""
    genes_invalidos = [
        "TP53",
        "KRAS",
        "",
        "ENSG123",  # Menos de 11 dígitos
        "ENSG00000141510_extra",
    ]
    validador = ValidadorOrdemGenes()
    res = validador.validar_formato_ensembl(genes_invalidos)
    assert res["valido"] is False
    assert res["invalidos_qtd"] == 5


def test_validador_genes_ordem_identica():
    """Testa se vetores perfeitamente coincidentes são aprovados."""
    genes = ["ENSG00000141510", "ENSG00000133703", "ENSG00000012048"]
    validador = ValidadorOrdemGenes()
    assert validador.validar_genes(genes, genes, validar_regex=True) is True


def test_validador_genes_falha_tamanho_divergente():
    """Testa se tamanhos diferentes disparam ValueError."""
    genes1 = ["ENSG00000141510", "ENSG00000133703"]
    genes2 = ["ENSG00000141510"]
    validador = ValidadorOrdemGenes()
    with pytest.raises(ValueError, match="Incompatibilidade de tamanho"):
        validador.validar_genes(genes1, genes2)


def test_validador_genes_falha_permutacao():
    """Testa se inversão posicional de genes dispara ValueError com diagnóstico."""
    genes_ref = ["ENSG00000141510", "ENSG00000133703", "ENSG00000012048"]
    genes_trocados = ["ENSG00000133703", "ENSG00000141510", "ENSG00000012048"]
    validador = ValidadorOrdemGenes()
    with pytest.raises(
        ValueError, match="Divergência na ordem de colunas gênicas detectada"
    ):
        validador.validar_genes(genes_trocados, genes_ref)


def test_validador_matriz_sucesso_e_falha():
    """Testa validação dimensional e numérica de matriz esparsa."""
    genes_ref = ["ENSG00000141510", "ENSG00000133703"]
    matriz_ok = sp.csr_matrix([[1.0, 0.0], [0.5, 1.0]])

    validador = ValidadorOrdemGenes()
    assert validador.validar_matriz(matriz_ok, genes_ref) is True

    # Colunas insuficientes
    matriz_errada = sp.csr_matrix([[1.0], [0.5]])
    with pytest.raises(ValueError, match="Número de colunas da matriz"):
        validador.validar_matriz(matriz_errada, genes_ref)


def test_leitor_features_sanitizacao_versao(tmp_path):
    """Testa se LeitorFeatures remove automaticamente o sufixo de versão (.1, .3)."""
    f_path = tmp_path / "features_com_versao.tsv"
    with open(f_path, "w", encoding="utf-8") as f:
        f.write("ENSG00000141510.3\tTP53\n")
        f.write("ENSG00000133703.12\tKRAS\n")
        f.write("ENSG00000012048\tBRCA1\n")

    leitor = LeitorFeatures(path_features_referencia=f_path, path_features_alvo=f_path)
    leitor.ler()

    assert leitor.map_referencia is not None
    assert leitor.map_referencia["TP53"] == "ENSG00000141510"
    assert leitor.map_referencia["KRAS"] == "ENSG00000133703"
    assert leitor.map_referencia["BRCA1"] == "ENSG00000012048"
