"""Módulo de Análise de Cobertura e Rastreamento de Genes Inter-Datasets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Union

import polars as pl

PathType = Union[str, os.PathLike[str]]


class AnalisadorCobertura:
    """Verifica quantos dos top-N genes frequentes do conjunto de referência estão presentes no alvo.

    Parameters
    ----------
    path_top_n : str | os.PathLike[str]
        Caminho do CSV contendo a lista dos top-N genes selecionados.
    map_f : Mapping[str, str]
        Mapeamento de features do conjunto de referência.
    map_m : Mapping[str, str]
        Mapeamento de features do conjunto alvo.

    Attributes
    ----------
    path_top_n : str
        Caminho normalizado do CSV de entrada.
    map_f : Mapping[str, str]
        Mapa de features da referência.
    map_m : Mapping[str, str]
        Mapa de features do alvo.
    """

    def __init__(
        self,
        path_top_n: PathType,
        map_f: Mapping[str, str],
        map_m: Mapping[str, str],
    ) -> None:
        self.path_top_n: str = str(path_top_n)
        self.map_f: Mapping[str, str] = map_f
        self.map_m: Mapping[str, str] = map_m

    def analisar(self, out_csv: PathType) -> pl.DataFrame:
        """Calcula as taxas de presença, ausência e falta de anotação e persiste o relatório em CSV.

        Parameters
        ----------
        out_csv : str | os.PathLike[str]
            Caminho do arquivo CSV onde o relatório de cobertura será salvo.

        Returns
        -------
        pl.DataFrame
            DataFrame Polars com a tabela de cobertura detalhada.
        """
        out_csv_str: str = str(out_csv)
        if os.path.exists(out_csv_str):
            print(f"[AnalisadorCobertura] Arquivo já existe, pulando: {out_csv_str}")
            return pl.read_csv(out_csv_str)

        top_n: pl.DataFrame = pl.read_csv(self.path_top_n)
        ids_m_list: list[str] = list(self.map_m.values())  # Ensembl IDs do Mathys

        # top_n['gene'] já contém Ensembl IDs (nomes de coluna do TXT alinhado)
        df: pl.DataFrame = (
            top_n
            .with_columns([
                pl.col("gene")
                    .is_in(ids_m_list)
                    .alias("presente_mathys"),
                pl.lit(False).alias("sem_ensembl_fujita"),
            ])
            .rename({"gene": "ensembl_id"})
        )

        total: int = df.height
        presentes: int = int(df["presente_mathys"].sum() or 0)
        sem_ensembl: int = int(df["sem_ensembl_fujita"].sum() or 0)
        ausentes: int = total - presentes - sem_ensembl

        print(f"\n{'='*50}")
        print(f"  Top {total} genes frequentes do Fujita")
        print(f"{'='*50}")
        print(f"  Presentes no Mathys       : {presentes:>5}  ({presentes/total*100:.1f}%)")
        print(f"  Ausentes no Mathys (zeros): {ausentes:>5}  ({ausentes/total*100:.1f}%)")
        print(f"  Sem Ensembl ID no Fujita  : {sem_ensembl:>5}  ({sem_ensembl/total*100:.1f}%)")
        print(f"{'='*50}")

        os.makedirs(os.path.dirname(os.path.abspath(out_csv_str)), exist_ok=True)
        df.write_csv(out_csv_str)
        print(f"\n[AnalisadorCobertura] Resultado salvo em: {out_csv_str}")
        return df

    def __repr__(self) -> str:
        """Representação textual do analisador de cobertura."""
        return (
            f"AnalisadorCobertura(\n"
            f"  path_top_n = {self.path_top_n}\n"
            f")"
        )
