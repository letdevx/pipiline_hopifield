"""Módulo de Validação Pós-Alinhamento Dimensional.

Garante que duas matrizes scRNA-seq alinhadas compartilhem rigorosamente
a mesma contagem, nomes e ordem posicional exata de genes.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import anndata as ad

PathType = str | os.PathLike[str]


class ValidadorAlinhamento:
    """Valida que dois arquivos AnnData alinhados possuem a mesma lista e ordem exata de genes.

    Parameters
    ----------
    path_f_alinhado : str | os.PathLike[str]
        Caminho para o .h5ad alinhado de referência (Fujita).
    path_m_alinhado : str | os.PathLike[str]
        Caminho para o .h5ad alinhado do conjunto alvo (Mathys).
    genes_ordenados : Sequence[str]
        Lista esperada de genes na ordem canônica.

    Attributes
    ----------
    path_f_alinhado : str
        Caminho do arquivo de referência.
    path_m_alinhado : str
        Caminho do arquivo alvo.
    genes_ordenados : list[str]
        Vetor canônico de genes.
    """

    def __init__(
        self,
        path_f_alinhado: PathType,
        path_m_alinhado: PathType,
        genes_ordenados: Sequence[str],
    ) -> None:
        self.path_f_alinhado: str = str(path_f_alinhado)
        self.path_m_alinhado: str = str(path_m_alinhado)
        self.genes_ordenados: list[str] = list(genes_ordenados)

    def validar(self) -> ValidadorAlinhamento:
        """Verifica a paridade dimensional e posicional exata entre os datasets.

        Returns
        -------
        ValidadorAlinhamento
            A própria instância se validado com sucesso.

        Raises
        ------
        ValueError
            Se houver divergência de contagem ou ordem gênica.
        """
        print("[ValidadorAlinhamento] Carregando metadados...")
        _f: ad.AnnData = ad.read_h5ad(self.path_f_alinhado, backed="r")
        _m: ad.AnnData = ad.read_h5ad(self.path_m_alinhado, backed="r")
        genes_f: list[str] = list(_f.var_names)
        genes_m: list[str] = list(_m.var_names)
        if hasattr(_f, "file") and _f.file is not None:
            _f.file.close()
        if hasattr(_m, "file") and _m.file is not None:
            _m.file.close()
        del _f, _m

        if len(genes_f) != len(self.genes_ordenados):
            raise ValueError(
                f"[VALIDAÇÃO FALHOU] Fujita alinhado tem {len(genes_f)} genes, "
                f"esperado {len(self.genes_ordenados)}."
            )
        if len(genes_m) != len(self.genes_ordenados):
            raise ValueError(
                f"[VALIDAÇÃO FALHOU] Mathys alinhado tem {len(genes_m)} genes, "
                f"esperado {len(self.genes_ordenados)}."
            )

        divs_f = [
            (i, self.genes_ordenados[i], genes_f[i])
            for i in range(len(self.genes_ordenados))
            if genes_f[i] != self.genes_ordenados[i]
        ]
        if divs_f:
            msg = "\n  ".join(
                f"pos {i}: esperado={e!r} encontrado={e2!r}" for i, e, e2 in divs_f[:5]
            )
            raise ValueError(
                f"[VALIDAÇÃO FALHOU] Fujita diverge em {len(divs_f)} posição(ões):\n  {msg}"
            )

        divs_m = [
            (i, self.genes_ordenados[i], genes_m[i])
            for i in range(len(self.genes_ordenados))
            if genes_m[i] != self.genes_ordenados[i]
        ]
        if divs_m:
            msg = "\n  ".join(
                f"pos {i}: esperado={e!r} encontrado={e2!r}" for i, e, e2 in divs_m[:5]
            )
            raise ValueError(
                f"[VALIDAÇÃO FALHOU] Mathys diverge em {len(divs_m)} posição(ões):\n  {msg}"
            )

        divs_fm = [
            (i, genes_f[i], genes_m[i])
            for i in range(len(genes_f))
            if genes_f[i] != genes_m[i]
        ]
        if divs_fm:
            msg = "\n  ".join(
                f"pos {i}: F={gf!r} M={gm!r}" for i, gf, gm in divs_fm[:5]
            )
            raise ValueError(
                f"[VALIDAÇÃO FALHOU] Fujita e Mathys divergem em {len(divs_fm)} posição(ões):\n  {msg}"
            )

        print(f"[OK] Número de genes idêntico: {len(self.genes_ordenados)}")
        print("[OK] Fujita alinhado == ordem de referência")
        print("[OK] Mathys alinhado == ordem de referência")
        print("[OK] Fujita alinhado == Mathys alinhado")
        print("[ValidadorAlinhamento] Validação concluída com sucesso.")
        return self

    def __repr__(self) -> str:
        """Representação textual do validador de alinhamento."""
        return (
            f"ValidadorAlinhamento(\n"
            f"  path_f_alinhado  = {self.path_f_alinhado}\n"
            f"  path_m_alinhado  = {self.path_m_alinhado}\n"
            f"  genes_ordenados  = {len(self.genes_ordenados)} genes\n"
            f")"
        )
