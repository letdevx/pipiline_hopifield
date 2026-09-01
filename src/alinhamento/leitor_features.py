"""Módulo de Leitura e Mapeamento de Features Genômicas (10x Genomics).

Lê arquivos de anotação de features (TSV/CSV) e extrai dicionários bidirecionais
para conversão entre símbolos de genes (Gene Symbols) e identificadores Ensembl ID.
"""

from __future__ import annotations

import os

import polars as pl

try:
    from src.config import PATH_FEATURES_ALVO, PATH_FEATURES_REFERENCIA
except ImportError:
    from config import (  # type: ignore[import-not-found]
        PATH_FEATURES_ALVO,
        PATH_FEATURES_REFERENCIA,
    )

PathType = str | os.PathLike[str]


class LeitorFeatures:
    """Lê arquivos TSV de features do 10x Genomics e mapeia gene_name → Ensembl ID.

    Parameters
    ----------
    path_features_referencia : str | os.PathLike[str] | None, optional
        Caminho do arquivo de features do conjunto de referência.
    path_features_alvo : str | os.PathLike[str] | None, optional
        Caminho do arquivo de features do conjunto alvo.
    path_features_f : str | os.PathLike[str] | None, optional
        Alias legado para `path_features_referencia`.
    path_features_m : str | os.PathLike[str] | None, optional
        Alias legado para `path_features_alvo`.

    Attributes
    ----------
    path_features_referencia : str
        Caminho normalizado das features de referência.
    path_features_alvo : str
        Caminho normalizado das features do conjunto alvo.
    map_referencia : dict[str, str] | None
        Dicionário mapeando {Gene Symbol: Ensembl ID} da referência.
    map_alvo : dict[str, str] | None
        Dicionário mapeando {Gene Symbol: Ensembl ID} do alvo.
    """

    def __init__(
        self,
        path_features_referencia: PathType | None = None,
        path_features_alvo: PathType | None = None,
        path_features_f: PathType | None = None,
        path_features_m: PathType | None = None,
    ) -> None:
        self.path_features_referencia: str = str(
            path_features_referencia or path_features_f or PATH_FEATURES_REFERENCIA
        )
        self.path_features_alvo: str = str(
            path_features_alvo or path_features_m or PATH_FEATURES_ALVO
        )
        self.map_referencia: dict[str, str] | None = None
        self.map_alvo: dict[str, str] | None = None

    @property
    def path_features_f(self) -> str:
        """Alias de compatibilidade para path_features_referencia."""
        return self.path_features_referencia

    @path_features_f.setter
    def path_features_f(self, val: PathType) -> None:
        self.path_features_referencia = str(val)

    @property
    def path_features_m(self) -> str:
        """Alias de compatibilidade para path_features_alvo."""
        return self.path_features_alvo

    @path_features_m.setter
    def path_features_m(self, val: PathType) -> None:
        self.path_features_alvo = str(val)

    @property
    def map_f(self) -> dict[str, str] | None:
        """Alias de compatibilidade para map_referencia."""
        return self.map_referencia

    @map_f.setter
    def map_f(self, val: dict[str, str] | None) -> None:
        self.map_referencia = val

    @property
    def map_m(self) -> dict[str, str] | None:
        """Alias de compatibilidade para map_alvo."""
        return self.map_alvo

    @map_m.setter
    def map_m(self, val: dict[str, str] | None) -> None:
        self.map_alvo = val

    def ler(self) -> LeitorFeatures:
        """Carrega e indexa os mapeamentos genômicos de ambos os conjuntos de features.

        Returns
        -------
        LeitorFeatures
            A própria instância após a leitura dos mapeamentos.
        """
        self.map_referencia = self._ler_features(self.path_features_referencia)
        self.map_alvo = self._ler_features(self.path_features_alvo)
        print(
            f"[LeitorFeatures] Referência : {len(self.map_referencia)} genes mapeados"
        )
        print(f"[LeitorFeatures] Alvo       : {len(self.map_alvo)} genes mapeados")
        return self

    def _ler_features(self, path: PathType) -> dict[str, str]:
        """Lê o arquivo TSV/CSV de features utilizando Polars em alta velocidade.

        Aplica desambiguação posicional idêntica ao `var_names_make_unique()` do AnnData/Scanpy,
        assegurando que símbolos duplicados (ex: TBCE e TBCE-1) mapeiem para seus respectivos
        Ensembl IDs exclusivos sem perda de registros.
        """
        df: pl.DataFrame = pl.read_csv(
            str(path),
            separator="\t",
            has_header=False,
            new_columns=["ensembl_id", "gene_name"],
            columns=[0, 1],
        ).with_columns(
            [
                pl.col("ensembl_id")
                .cast(pl.Utf8)
                .str.strip_chars()
                .str.replace(r"\.\d+$", ""),
                pl.col("gene_name").cast(pl.Utf8).str.strip_chars(),
            ]
        )

        names: list[str] = df["gene_name"].to_list()
        ensembls: list[str] = df["ensembl_id"].to_list()

        counts: dict[str, int] = {}
        mapping: dict[str, str] = {}

        for gene, eid in zip(names, ensembls, strict=False):
            if gene in counts:
                counts[gene] += 1
                unique_gene = f"{gene}-{counts[gene]}"
            else:
                counts[gene] = 0
                unique_gene = gene
                # Mapeia também a primeira ocorrência sob o símbolo base
                mapping[gene] = eid

            # Mapeia o símbolo único disambiguado (ex: TBCE-1) para seu respectivo Ensembl ID
            mapping[unique_gene] = eid

        return mapping

    def __repr__(self) -> str:
        """Representação textual do leitor de features."""
        n_ref: str = (
            str(len(self.map_referencia))
            if self.map_referencia is not None
            else "não carregado"
        )
        n_alvo: str = (
            str(len(self.map_alvo)) if self.map_alvo is not None else "não carregado"
        )
        return (
            f"LeitorFeatures(\n"
            f"  path_features_referencia = {self.path_features_referencia}\n"
            f"  path_features_alvo       = {self.path_features_alvo}\n"
            f"  map_referencia           = {n_ref} genes\n"
            f"  map_alvo                 = {n_alvo} genes\n"
            f")"
        )
