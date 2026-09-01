"""Módulo de Exportação OOM-Safe de Matrizes no Formato Matrix Market (MTX).

Gera arquivos `matrix.mtx` (orientados a Machine Learning: células nas linhas × genes nas colunas),
`genes_referencia.tsv` (2 colunas: Ensembl ID sem versão e Gene Symbol) e `barcodes.tsv`
com garantia de integridade e auditoria pré e pós-gravação.
"""

from __future__ import annotations

import gc
import os
from collections.abc import Mapping, Sequence
from typing import Any

import anndata as ad
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from numpy.typing import NDArray

from .validador_ordem_genes import ValidadorOrdemGenes

PathType = str | os.PathLike[str]
MatrixInput = NDArray[Any] | sp.spmatrix | sp.sparray | ad.AnnData | Any


class ExportadorMTX:
    """Exportador de matrizes scRNA-seq para formato Matrix Market (10x Machine Learning).

    Escreve a matriz em formato esparso comprimido com verificação de ordem gênica
    e sanitização de Ensembl IDs.

    Parameters
    ----------
    out_dir : PathType
        Diretório onde os arquivos `matrix.mtx`, `genes_referencia.tsv` e `barcodes.tsv`
        serão gravados.
    validador : ValidadorOrdemGenes | None, optional
        Instância do validador de ordem gênica. Se None, cria uma padrão.

    Attributes
    ----------
    out_dir : str
        Caminho do diretório de saída.
    validador : ValidadorOrdemGenes
        Instância do validador.
    """

    def __init__(
        self,
        out_dir: PathType,
        validador: ValidadorOrdemGenes | None = None,
    ) -> None:
        self.out_dir: str = str(out_dir)
        self.validador: ValidadorOrdemGenes = (
            validador if validador is not None else ValidadorOrdemGenes()
        )
        os.makedirs(self.out_dir, exist_ok=True)

    @staticmethod
    def _converter_para_csr(dado: Any) -> sp.csr_matrix:
        """Converte dados em memória ou backed (AnnData / SciPy / NumPy) para sp.csr_matrix.

        Parameters
        ----------
        dado : Any
            Objeto de dados (array NumPy, matriz esparsa SciPy ou dataset backed do AnnData).

        Returns
        -------
        sp.csr_matrix
            Matriz esparsa no formato CSR com dtype float32.
        """
        if hasattr(dado, "to_memory"):
            # Suporte nativo a datasets backed do AnnData (_CSRDataset, _CSCDataset)
            dado_mem = dado.to_memory()
            return sp.csr_matrix(dado_mem, dtype=np.float32)
        if sp.issparse(dado):
            return sp.csr_matrix(dado, dtype=np.float32)
        return sp.csr_matrix(np.asarray(dado, dtype=np.float32), dtype=np.float32)

    def exportar(
        self,
        matriz: MatrixInput,
        genes: Sequence[str] | None = None,
        genes_referencia: Sequence[str] | None = None,
        map_features: Mapping[str, str] | None = None,
        barcodes: Sequence[str] | None = None,
        nome_etapa: str = "Exportação MTX",
        forcar: bool = False,
    ) -> dict[str, Any]:
        """Exporta matriz e metadados para a pasta configurada.

        Parameters
        ----------
        matriz : MatrixInput
            Objeto AnnData, matriz esparsa SciPy ou array NumPy (células × genes).
        genes : Sequence[str] | None, optional
            Vetor de genes correspondente às colunas da matriz (obrigatório se não for AnnData).
        genes_referencia : Sequence[str] | None, optional
            Vetor canônico de genes de referência para validação 1-to-1.
        map_features : Mapping[str, str] | None, optional
            Mapeamento {Ensembl ID: Gene Symbol} ou {Gene Symbol: Ensembl ID}
            para enriquecimento de `genes_referencia.tsv`.
        barcodes : Sequence[str] | None, optional
            Identificadores de célula para `barcodes.tsv`.
        nome_etapa : str, default="Exportação MTX"
            Rótulo descritivo para logs de auditoria.
        forcar : bool, default=False
            Se True, regrava mesmo se os arquivos já existirem.

        Returns
        -------
        dict[str, Any]
            Dicionário com caminhos gerados e relatório de conformidade.

        Raises
        ------
        ValueError
            Se houver divergência dimensional ou inconsistência posicional de genes.
        """
        path_mtx: str = os.path.join(self.out_dir, "matrix.mtx")
        path_genes: str = os.path.join(self.out_dir, "genes_referencia.tsv")
        path_barcodes: str = os.path.join(self.out_dir, "barcodes.tsv")

        if (
            not forcar
            and os.path.exists(path_mtx)
            and os.path.exists(path_genes)
            and os.path.exists(path_barcodes)
        ):
            print(
                f"[ExportadorMTX] Arquivos já existem em {self.out_dir}, validando integridade..."
            )
            if genes_referencia is not None:
                return self.validador.validar_pasta_mtx(self.out_dir, genes_referencia)
            return {
                "status": "EXISTENTE",
                "pasta": self.out_dir,
                "arquivos": {
                    "mtx": path_mtx,
                    "genes": path_genes,
                    "barcodes": path_barcodes,
                },
            }

        # 1. Extração da matriz esparsa, genes e barcodes
        matriz_esparsa: sp.csr_matrix
        genes_lista: list[str]
        barcodes_lista: list[str]

        if isinstance(matriz, ad.AnnData):
            adata: ad.AnnData = matriz
            genes_lista = [str(g).split(".")[0].strip() for g in adata.var_names]
            barcodes_lista = [str(b).strip() for b in adata.obs_names]

            if adata.X is None:
                raise ValueError(
                    "[ExportadorMTX] AnnData fornecido não possui matriz de contagens em '.X'."
                )
            matriz_esparsa = self._converter_para_csr(adata.X)
        else:
            if genes is None:
                raise ValueError(
                    "[ExportadorMTX] Parâmetro 'genes' é obrigatório quando a entrada não é AnnData."
                )
            genes_lista = [str(g).split(".")[0].strip() for g in genes]
            matriz_esparsa = self._converter_para_csr(matriz)

            n_celulas: int = (
                int(matriz_esparsa.shape[0]) if matriz_esparsa.shape is not None else 0
            )
            if barcodes is not None:
                barcodes_lista = [str(b).strip() for b in barcodes]
                if len(barcodes_lista) != n_celulas:
                    raise ValueError(
                        f"[ExportadorMTX] Contagem de barcodes ({len(barcodes_lista)}) "
                        f"diverge do número de células na matriz ({n_celulas})."
                    )
            else:
                barcodes_lista = [f"celula_{i}" for i in range(n_celulas)]

        # 2. Auditoria Pré-Gravação (Validação de Ordem e Regex)
        ref_comparacao = (
            genes_referencia if genes_referencia is not None else genes_lista
        )
        self.validador.validar_tudo_ou_falhar(
            matriz=matriz_esparsa,
            genes=genes_lista,
            genes_referencia=ref_comparacao,
            nome_etapa=f"Pré-Gravação: {nome_etapa}",
        )

        # 3. Gravação de matrix.mtx (OOM-Safe via scipy.io.mmwrite)
        print(f"[ExportadorMTX] Gravando {path_mtx} (CSR: {matriz_esparsa.shape})...")
        sio.mmwrite(path_mtx, matriz_esparsa)

        # 4. Gravação de genes_referencia.tsv (2 colunas: Ensembl_ID\tGene_Symbol)
        print(f"[ExportadorMTX] Gravando {path_genes} ({len(genes_lista)} genes)...")
        map_simbolos: dict[str, str] = {}
        if map_features is not None:
            for k, v in map_features.items():
                if k.startswith("ENS"):
                    map_simbolos[k] = v
                elif v.startswith("ENS"):
                    map_simbolos[v] = k
                else:
                    map_simbolos[k] = v

        with open(path_genes, "w", encoding="utf-8") as f_genes:
            for gene_id in genes_lista:
                simbolo: str = map_simbolos.get(gene_id, gene_id)
                f_genes.write(f"{gene_id}\t{simbolo}\n")

        # 5. Gravação de barcodes.tsv
        print(
            f"[ExportadorMTX] Gravando {path_barcodes} ({len(barcodes_lista)} células)..."
        )
        with open(path_barcodes, "w", encoding="utf-8") as f_barcodes:
            for b in barcodes_lista:
                f_barcodes.write(f"{b}\n")

        gc.collect()

        # 6. Auditoria Pós-Gravação da Pasta MTX
        print(f"[ExportadorMTX] Realizando auditoria pós-gravação em {self.out_dir}...")
        relatorio = self.validador.validar_pasta_mtx(
            self.out_dir,
            genes_referencia=ref_comparacao,
            validar_regex=True,
        )
        print(
            f"[ExportadorMTX] '{nome_etapa}' exportada e validada com sucesso! [Ok]\n"
        )
        return relatorio

    def __repr__(self) -> str:
        """Representação textual do exportador MTX."""
        return f"ExportadorMTX(out_dir={self.out_dir!r})"
