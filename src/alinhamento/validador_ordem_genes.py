"""Módulo de Validação Estrita de Ordem Gênica e Ensembl IDs.

Assegura paridade posicional rigorosa (1-to-1) contra a referência canônica,
validação de sintaxe estável sem versões de transcripto (Regex) e integridade
estrutural de matrizes de expressão gênica e diretórios MTX.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from re import Pattern
from typing import Any

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

PathType = str | os.PathLike[str]
MatrixInput = NDArray[Any] | sp.spmatrix


class ValidadorOrdemGenes:
    r"""Validador posicional e sintático de genes para pipelines scRNA-seq.

    Garante que matrizes de expressão e arquivos associados (TSV/MTX) sigam
    estritamente a ordem canônica da referência e utilizem identificadores
    Ensembl válidos e imutáveis (sem sufixos de versão de release).

    Parameters
    ----------
    padrao_ensembl : str | Pattern[str] | None, optional
        Expressão regular para validação de identificadores Ensembl.
        O padrão default exige o prefixo Ensembl seguido de 11 dígitos numéricos
        sem sufixo de versão: `^ENS[A-Z]*G\\d{11}$`.

    Attributes
    ----------
    regex_ensembl : Pattern[str]
        Objeto compilado de expressão regular utilizado nas validações.
    """

    PADRAO_ENSEMBL_DEFAULT: str = r"^ENS[A-Z]*G\d{11}$"

    def __init__(self, padrao_ensembl: str | Pattern[str] | None = None) -> None:
        if padrao_ensembl is None:
            self.regex_ensembl: Pattern[str] = re.compile(self.PADRAO_ENSEMBL_DEFAULT)
        elif isinstance(padrao_ensembl, str):
            self.regex_ensembl = re.compile(padrao_ensembl)
        else:
            self.regex_ensembl = padrao_ensembl

    def validar_formato_ensembl(
        self,
        genes: Sequence[str],
        regex: str | Pattern[str] | None = None,
    ) -> dict[str, Any]:
        """Realiza validação censitária (100% dos genes) do formato de identificadores Ensembl.

        Parameters
        ----------
        genes : Sequence[str]
            Lista ou vetor de identificadores a serem validados.
        regex : str | Pattern[str] | None, optional
            Expressão regular opcional para sobrepor a padrão da instância.

        Returns
        -------
        dict[str, Any]
            Dicionário com diagnóstico contendo status booleano, total analisado,
            quantidade de inválidos e amostra das primeiras divergências.
        """
        reg: Pattern[str] = (
            re.compile(regex)
            if isinstance(regex, str)
            else (regex if regex is not None else self.regex_ensembl)
        )

        invalidos: list[tuple[int, str]] = []
        for i, gene in enumerate(genes):
            gene_str: str = str(gene).strip()
            if not reg.match(gene_str):
                invalidos.append((i, gene_str))

        return {
            "valido": len(invalidos) == 0,
            "total": len(genes),
            "invalidos_qtd": len(invalidos),
            "invalidos_amostra": invalidos[:10],
            "padrao_utilizado": reg.pattern,
        }

    def validar_genes(
        self,
        genes_teste: Sequence[str],
        genes_referencia: Sequence[str],
        validar_regex: bool = True,
    ) -> bool:
        """Verifica paridade posicional estrita entre dois vetores de genes.

        Parameters
        ----------
        genes_teste : Sequence[str]
            Vetor de genes submetido a teste.
        genes_referencia : Sequence[str]
            Vetor de genes da referência canônica.
        validar_regex : bool, default=True
            Se True, valida também o formato Ensembl sem versão dos genes de teste.

        Returns
        -------
        bool
            True se a paridade posicional e o regex forem 100% conformes.

        Raises
        ------
        ValueError
            Se houver qualquer discrepância de contagem, valor ou ordem.
        """
        n_teste: int = len(genes_teste)
        n_ref: int = len(genes_referencia)

        if n_teste != n_ref:
            raise ValueError(
                f"[ValidadorOrdemGenes] Incompatibilidade de tamanho: "
                f"encontrados {n_teste:,} genes, esperados {n_ref:,}."
            )

        if validar_regex:
            res_regex = self.validar_formato_ensembl(genes_teste)
            if not res_regex["valido"]:
                amostra = res_regex["invalidos_amostra"]
                raise ValueError(
                    f"[ValidadorOrdemGenes] {res_regex['invalidos_qtd']:,} identificadores Ensembl "
                    f"possuem formato inválido conforme o regex '{res_regex['padrao_utilizado']}'.\n"
                    f"Amostra dos primeiros inválidos (índice, valor): {amostra}\n"
                    f"DICA: Verifique se os IDs contêm versão de anotação (ex: '.1', '.2') "
                    f"ou símbolos gênicos em vez de Ensembl IDs."
                )

        divergencias: list[tuple[int, str, str]] = []
        for i in range(n_ref):
            gt: str = str(genes_teste[i]).strip()
            gr: str = str(genes_referencia[i]).strip()
            if gt != gr:
                divergencias.append((i, gr, gt))
                if len(divergencias) >= 5:
                    break

        if divergencias:
            detalhes = "\n  ".join(
                f"Posição {idx}: esperado={esp!r}, encontrado={enc!r}"
                for idx, esp, enc in divergencias
            )
            raise ValueError(
                f"[ValidadorOrdemGenes] Divergência na ordem de colunas gênicas detectada!\n"
                f"Primeiras divergências:\n  {detalhes}"
            )

        return True

    def validar_matriz(
        self,
        matriz: MatrixInput,
        genes_referencia: Sequence[str],
        checar_nans: bool = True,
    ) -> bool:
        """Verifica compatibilidade dimensional e integridade numérica de uma matriz de expressão.

        Parameters
        ----------
        matriz : MatrixInput
            Matriz esparsa ou array denso (células × genes).
        genes_referencia : Sequence[str]
            Vetor canônico de genes.
        checar_nans : bool, default=True
            Se True, verifica ausência de NaNs e Infs.

        Returns
        -------
        bool
            True se aprovado.

        Raises
        ------
        ValueError
            Se a matriz divergir em colunas ou contiver NaNs.
        """
        n_cols: int = int(matriz.shape[1])
        n_ref: int = len(genes_referencia)

        if n_cols != n_ref:
            raise ValueError(
                f"[ValidadorOrdemGenes] Número de colunas da matriz ({n_cols:,}) "
                f"não coincide com a referência ({n_ref:,})."
            )

        if checar_nans:
            if sp.issparse(matriz):
                csr = sp.csr_matrix(matriz)
                if np.isnan(csr.data).any() or np.isinf(csr.data).any():
                    raise ValueError(
                        "[ValidadorOrdemGenes] Matriz esparsa contém valores NaN ou Inf."
                    )
            else:
                arr = np.asarray(matriz)
                if np.isnan(arr).any() or np.isinf(arr).any():
                    raise ValueError(
                        "[ValidadorOrdemGenes] Matriz densa contém valores NaN ou Inf."
                    )

        return True

    def validar_pasta_mtx(
        self,
        pasta_mtx: PathType,
        genes_referencia: Sequence[str],
        validar_regex: bool = True,
    ) -> dict[str, Any]:
        """Audita uma pasta exportada no formato Matrix Market (10x Machine Learning).

        Parameters
        ----------
        pasta_mtx : PathType
            Caminho do diretório contendo `matrix.mtx`, `genes_referencia.tsv` e `barcodes.tsv`.
        genes_referencia : Sequence[str]
            Lista canônica de genes de referência.
        validar_regex : bool, default=True
            Se True, aplica checagem de regex sobre a coluna de Ensembl IDs.

        Returns
        -------
        dict[str, Any]
            Dicionário com o relatório de auditoria e caminhos dos arquivos.

        Raises
        ------
        FileNotFoundError
            Se algum arquivo obrigatório estiver ausente.
        ValueError
            Se houver divergência dimensional ou de ordem gênica.
        """
        pasta: str = str(pasta_mtx)
        path_mtx: str = os.path.join(pasta, "matrix.mtx")
        path_genes: str = os.path.join(pasta, "genes_referencia.tsv")
        path_barcodes: str = os.path.join(pasta, "barcodes.tsv")

        for p, desc in [
            (path_mtx, "matrix.mtx"),
            (path_genes, "genes_referencia.tsv"),
            (path_barcodes, "barcodes.tsv"),
        ]:
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"[ValidadorOrdemGenes] Arquivo obrigatório não encontrado: {p} ({desc})"
                )

        # 1. Leitura de genes_referencia.tsv (Ensembl ID na Coluna 0)
        genes_tsv: list[str] = []
        with open(path_genes, encoding="utf-8") as f:
            for linha in f:
                linha_clean = linha.strip()
                if not linha_clean:
                    continue
                partes = linha_clean.split("\t")
                genes_tsv.append(partes[0].strip())

        self.validar_genes(
            genes_teste=genes_tsv,
            genes_referencia=genes_referencia,
            validar_regex=validar_regex,
        )

        # 2. Leitura de barcodes.tsv
        n_barcodes: int = 0
        with open(path_barcodes, encoding="utf-8") as f:
            for linha in f:
                if linha.strip():
                    n_barcodes += 1

        # 3. Leitura do cabeçalho do arquivo MTX para conferência de dimensões
        nrows: int = 0
        ncols: int = 0
        nnz: int = 0
        with open(path_mtx, encoding="utf-8") as f:
            for linha in f:
                if linha.startswith("%"):
                    continue
                parts = linha.strip().split()
                if len(parts) == 3:
                    nrows = int(parts[0])
                    ncols = int(parts[1])
                    nnz = int(parts[2])
                    break

        if ncols != len(genes_referencia):
            raise ValueError(
                f"[ValidadorOrdemGenes] Dimensão de colunas no cabeçalho MTX ({ncols:,}) "
                f"diverge do esperado ({len(genes_referencia):,})."
            )

        if nrows != n_barcodes:
            raise ValueError(
                f"[ValidadorOrdemGenes] Linhas no MTX ({nrows:,}) não conferem com "
                f"quantidade de barcodes ({n_barcodes:,})."
            )

        return {
            "status": "APROVADO",
            "pasta": pasta,
            "n_celulas": nrows,
            "n_genes": ncols,
            "entradas_nao_nulas": nnz,
            "arquivos": {
                "mtx": path_mtx,
                "genes": path_genes,
                "barcodes": path_barcodes,
            },
        }

    def validar_tudo_ou_falhar(
        self,
        matriz: MatrixInput,
        genes: Sequence[str],
        genes_referencia: Sequence[str],
        nome_etapa: str = "Etapa",
    ) -> bool:
        """Executa validação completa combinada (Matriz + Genes + Regex) com fail-fast.

        Parameters
        ----------
        matriz : MatrixInput
            Matriz de expressão.
        genes : Sequence[str]
            Identificadores da matriz testada.
        genes_referencia : Sequence[str]
            Identificadores da referência canônica.
        nome_etapa : str, default="Etapa"
            Identificador descritivo para mensagens de log.

        Returns
        -------
        bool
            True se todas as validações forem concluídas com sucesso.
        """
        print(f"\n[ValidadorOrdemGenes] Iniciando auditoria para '{nome_etapa}'...")
        self.validar_genes(genes, genes_referencia, validar_regex=True)
        self.validar_matriz(matriz, genes_referencia, checar_nans=True)
        print(
            f"  [OK] {len(genes):,} genes auditados: 100% de conformidade posicional e sintática (Regex)."
        )
        print(f"  [OK] Matriz com shape {matriz.shape} íntegra e livre de NaNs/Infs.")
        print(f"[ValidadorOrdemGenes] '{nome_etapa}': APROVADO com sucesso! [Ok]\n")
        return True

    def __repr__(self) -> str:
        """Representação textual do validador de ordem gênica."""
        return (
            f"ValidadorOrdemGenes(\n  regex_ensembl = {self.regex_ensembl.pattern!r}\n)"
        )
