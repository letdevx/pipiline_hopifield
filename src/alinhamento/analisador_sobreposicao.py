"""Módulo de Análise de Sobreposição e Ordenação Canônica de Genes.

Calcula a interseção e diferença simétrica de genes entre conjuntos de dados
e estabelece o vocabulário gênico canônico indexado.
"""

from __future__ import annotations

from typing import Mapping, Sequence


class AnalisadorSobreposicao:
    """Calcula sobreposição de espaços gênicos e define a ordem canônica baseada no Fujita.

    Parameters
    ----------
    map_f : Mapping[str, str]
        Dicionário {Gene Symbol: Ensembl ID} do dataset de referência (Fujita).
    map_m : Mapping[str, str]
        Dicionário {Gene Symbol: Ensembl ID} do dataset alvo (Mathys).
    var_names_f_original : Sequence[str]
        Lista de nomes de genes na ordem original da matriz de referência.

    Attributes
    ----------
    map_f : Mapping[str, str]
        Mapa de features da referência.
    map_m : Mapping[str, str]
        Mapa de features do alvo.
    var_names_f_original : Sequence[str]
        Ordem original das variáveis de referência.
    ids_comuns : set[str] | None
        Conjunto de Ensembl IDs presentes em ambos os conjuntos.
    ids_so_f : set[str] | None
        Conjunto de Ensembl IDs exclusivos da referência.
    ids_so_m : set[str] | None
        Conjunto de Ensembl IDs exclusivos do alvo.
    genes_ordenados : list[str] | None
        Vetor canônico de Ensembl IDs ordenados.
    gene_alvo_idx : dict[str, int] | None
        Mapeamento de cada Ensembl ID para seu índice de coluna canônico [0..N-1].
    """

    def __init__(
        self,
        map_f: Mapping[str, str],
        map_m: Mapping[str, str],
        var_names_f_original: Sequence[str],
    ) -> None:
        self.map_f: Mapping[str, str] = map_f
        self.map_m: Mapping[str, str] = map_m
        self.var_names_f_original: Sequence[str] = var_names_f_original
        self.ids_comuns: set[str] | None = None
        self.ids_so_f: set[str] | None = None
        self.ids_so_m: set[str] | None = None
        self.genes_ordenados: list[str] | None = None
        self.gene_alvo_idx: dict[str, int] | None = None

    def analisar(self) -> AnalisadorSobreposicao:
        """Executa a análise de conjuntos e indexa o espaço gênico canônico.

        Returns
        -------
        AnalisadorSobreposicao
            A própria instância com os atributos calculados.
        """
        ids_f: set[str] = set(self.map_f.values())
        ids_m: set[str] = set(self.map_m.values())
        self.ids_comuns = ids_f & ids_m
        self.ids_so_f = ids_f - ids_m
        self.ids_so_m = ids_m - ids_f

        seen: set[str] = set()
        self.genes_ordenados = []
        for gene_name in self.var_names_f_original:
            eid: str = self.map_f.get(gene_name, gene_name)
            if eid not in seen:
                self.genes_ordenados.append(eid)
                seen.add(eid)

        self.gene_alvo_idx = {g: i for i, g in enumerate(self.genes_ordenados)}

        print(f"[AnalisadorSobreposicao] Em comum  : {len(self.ids_comuns)}")
        print(f"[AnalisadorSobreposicao] Só Fujita : {len(self.ids_so_f)}")
        print(f"[AnalisadorSobreposicao] Só Mathys : {len(self.ids_so_m)}  <- serão excluídos")
        print(f"[AnalisadorSobreposicao] Espaço gênico final: {len(self.genes_ordenados)} genes")
        return self

    def __repr__(self) -> str:
        """Representação textual do analisador de sobreposição."""
        def _n(x: set[str] | list[str] | None) -> str:
            return str(len(x)) if x is not None else "não calculado"

        return (
            f"AnalisadorSobreposicao(\n"
            f"  ids_comuns      = {_n(self.ids_comuns)}\n"
            f"  ids_so_f        = {_n(self.ids_so_f)}\n"
            f"  ids_so_m        = {_n(self.ids_so_m)}\n"
            f"  genes_ordenados = {_n(self.genes_ordenados)}\n"
            f")"
        )
