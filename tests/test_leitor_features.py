"""Testes unitários para o LeitorFeatures e desambiguação de símbolos duplicados."""

from collections.abc import Sequence

from src.alinhamento.analisador_sobreposicao import AnalisadorSobreposicao
from src.alinhamento.exportador_mtx import ExportadorMTX
from src.alinhamento.leitor_features import LeitorFeatures


def test_leitor_features_desambiguacao_duplicatas(tmp_path):
    """Testa se símbolos duplicados recebem sufixos -1, -2 no padrão make_unique."""
    path_feat = tmp_path / "features_test.tsv"
    # Simula 2 linhas com TBCE e 1 com GAPDH
    linhas = [
        "ENSG00000111640\tGAPDH\tGene Expression\n",
        "ENSG00000112499\tTBCE\tGene Expression\n",
        "ENSG00000284733\tTBCE\tGene Expression\n",
    ]
    path_feat.write_text("".join(linhas), encoding="utf-8")

    leitor = LeitorFeatures(
        path_features_referencia=str(path_feat),
        path_features_alvo=str(path_feat),
    )
    leitor.ler()

    assert leitor.map_f is not None
    # Deve mapear o primeiro TBCE tanto no símbolo base quanto em make_unique
    assert leitor.map_f["GAPDH"] == "ENSG00000111640"
    assert leitor.map_f["TBCE"] == "ENSG00000112499"
    # Deve mapear a segunda ocorrência para TBCE-1
    assert leitor.map_f["TBCE-1"] == "ENSG00000284733"


def test_analisador_sobreposicao_com_genes_desambiguados():
    """Testa se AnalisadorSobreposicao traduz TBCE-1 para Ensembl ID e não deixa literal."""
    map_f = {
        "GAPDH": "ENSG00000111640",
        "TBCE": "ENSG00000112499",
        "TBCE-1": "ENSG00000284733",
    }
    map_m = {
        "GAPDH": "ENSG00000111640",
        "TBCE": "ENSG00000112499",
    }
    var_names_f: Sequence[str] = ["GAPDH", "TBCE", "TBCE-1"]

    analisador = AnalisadorSobreposicao(
        map_f=map_f,
        map_m=map_m,
        var_names_f_original=var_names_f,
    )
    analisador.analisar()

    assert analisador.genes_ordenados is not None
    assert len(analisador.genes_ordenados) == 3
    # Nenhum gene ordenado pode ter ficado como o símbolo literal TBCE-1
    assert "TBCE-1" not in analisador.genes_ordenados
    assert analisador.genes_ordenados == [
        "ENSG00000111640",
        "ENSG00000112499",
        "ENSG00000284733",
    ]


def test_exportador_mtx_map_features_bidirecional(tmp_path):
    """Testa se genes_referencia.tsv recebe os símbolos quando map_features é {Symbol: Ensembl}."""
    out_dir = tmp_path / "mtx_bidirecional"
    matriz = [[1.0, 0.0], [0.0, 1.0]]
    genes = ["ENSG00000111640", "ENSG00000112499"]
    # map_features no formato {Symbol: Ensembl}
    map_features = {
        "GAPDH": "ENSG00000111640",
        "TBCE": "ENSG00000112499",
    }

    exportador = ExportadorMTX(out_dir=out_dir)
    res = exportador.exportar(
        matriz=matriz,
        genes=genes,
        genes_referencia=genes,
        map_features=map_features,
        nome_etapa="Teste Bidirecional",
    )

    assert res["status"] == "APROVADO"

    # Confere se genes_referencia.tsv gravou Ensembl_ID e Gene_Symbol corretamente
    with open(out_dir / "genes_referencia.tsv", encoding="utf-8") as f:
        linhas = [line.strip().split("\t") for line in f if line.strip()]

    assert linhas[0] == ["ENSG00000111640", "GAPDH"]
    assert linhas[1] == ["ENSG00000112499", "TBCE"]
