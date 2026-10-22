"""Módulo de Injeção Direta OOM-Safe de Valores Sentinela (0.5) em Arquivos Matrix Market (.mtx).

Permite injetar valores constantes (ex.: 0.5) nas colunas de genes ausentes
diretamente em arquivos `.mtx` em disco através de streaming com buffer,
sem carregar a matriz global na memória RAM, garantindo execução segura em hardware modesto.
"""

from __future__ import annotations

import gc
import os
import shutil
from collections.abc import Sequence
from typing import Any

import pandas as pd

from .validador_ordem_genes import ValidadorOrdemGenes

PathType = str | os.PathLike[str]


class InjetorMTXSentinela:
    """Injetor OOM-Safe de valores sentinela diretamente no formato Matrix Market (.mtx).

    Processa o arquivo `matrix.mtx` em streaming de texto através de lotes com buffer,
    reescrevendo o cabeçalho dimensional e anexando as coordenadas das colunas ausentes
    sem instanciar matrizes densas em memória.

    Parameters
    ----------
    dir_mtx_origem : PathType
        Diretório contendo os arquivos `matrix.mtx`, `genes_referencia.tsv` (ou `features.tsv`)
        e `barcodes.tsv` de entrada.
    dir_mtx_destino : PathType | None, optional
        Diretório onde os arquivos modificados serão persistidos. Se None, utiliza o
        próprio diretório de origem realizando substituição atômica via arquivo temporário.
    fill_value : float, default=0.5
        Valor sentinela a ser injetado nas colunas selecionadas.
    buffer_size : int, default=100_000
        Número de linhas de texto agrupadas por bloco de escrita para otimização de I/O em disco.
    genes_referencia : Sequence[str] | None, optional
        Lista de genes de referência para auditoria. Se omitido, tenta carregar de `genes_referencia.tsv`.
    validador : ValidadorOrdemGenes | None, optional
        Instância do validador de integridade e ordem gênica. Se None, cria uma padrão.
    validar_regex : bool | None, optional
        Se True, valida o regex canônico Ensembl (ENSG...). Se False, ignora validação de regex
        (ideal para datasets que utilizam Gene Symbols oficiais como TNFRSF4, RPL7P7).
        Se None (padrão), auto-detecta a necessidade de validar examinando os identificadores.

    Attributes
    ----------
    dir_mtx_origem : str
        Caminho absoluto do diretório de origem.
    dir_mtx_destino : str
        Caminho absoluto do diretório de destino.
    fill_value : float
        Valor sentinela configurado.
    buffer_size : int
        Tamanho do buffer de streaming.
    genes_referencia : Sequence[str] | None
        Vetor canônico de genes configurado.
    validador : ValidadorOrdemGenes
        Validador de integridade do pipeline.
    validar_regex : bool | None
        Configuração de validação regex de identificadores.
    """

    def __init__(
        self,
        dir_mtx_origem: PathType,
        dir_mtx_destino: PathType | None = None,
        fill_value: float = 0.5,
        buffer_size: int = 100_000,
        genes_referencia: Sequence[str] | None = None,
        validador: ValidadorOrdemGenes | None = None,
        validar_regex: bool | None = None,
    ) -> None:
        self.dir_mtx_origem: str = os.path.abspath(str(dir_mtx_origem))
        self.dir_mtx_destino: str = (
            os.path.abspath(str(dir_mtx_destino))
            if dir_mtx_destino is not None
            else self.dir_mtx_origem
        )
        self.fill_value: float = float(fill_value)
        self.buffer_size: int = int(buffer_size)
        self.genes_referencia: Sequence[str] | None = genes_referencia
        self.validador: ValidadorOrdemGenes = (
            validador if validador is not None else ValidadorOrdemGenes()
        )
        self.validar_regex: bool | None = validar_regex

        os.makedirs(self.dir_mtx_destino, exist_ok=True)

    def injetar_de_tracking(
        self,
        path_tracking_csv: PathType,
        coluna_indice: str = "posicao_coluna",
        indexado_em_zero: bool = True,
        forcar: bool = False,
        validar_regex: bool | None = None,
    ) -> dict[str, Any]:
        """Carrega os índices de colunas a partir do arquivo CSV de tracking e orquestra a injeção.

        Parameters
        ----------
        path_tracking_csv : PathType
            Caminho do arquivo CSV de tracking (ex.: `tracking_genes_adicionados_Fujita.csv`).
        coluna_indice : str, default="posicao_coluna"
            Nome da coluna no CSV contendo os índices posicionais dos genes a preencher.
        indexado_em_zero : bool, default=True
            Se True, considera que os índices no CSV são 0-indexed (padrão Python/AnnData)
            e soma 1 para convertê-los para o formato 1-indexed do Matrix Market.
        forcar : bool, default=False
            Se True, reprocessa mesmo se o arquivo de destino já existir.
        validar_regex : bool | None, optional
            Sobrescreve a checagem regex Ensembl na auditoria pós-injeção.

        Returns
        -------
        dict[str, Any]
            Relatório de auditoria e métricas da injeção concluída.

        Raises
        ------
        FileNotFoundError
            Se o arquivo de tracking não existir no caminho especificado.
        ValueError
            Se a coluna de índice não estiver presente no CSV.
        """
        path_csv = str(path_tracking_csv)
        if not os.path.exists(path_csv):
            raise FileNotFoundError(
                f"[InjetorMTXSentinela] Arquivo de tracking não encontrado: '{path_csv}'."
            )

        df_tracking = pd.read_csv(path_csv)
        if coluna_indice not in df_tracking.columns:
            raise ValueError(
                f"[InjetorMTXSentinela] Coluna '{coluna_indice}' ausente no CSV de tracking: "
                f"{df_tracking.columns.tolist()}"
            )

        indices_raw = df_tracking[coluna_indice].dropna().astype(int).unique()
        if indexado_em_zero:
            colunas_1based = {int(idx) + 1 for idx in indices_raw}
        else:
            colunas_1based = {int(idx) for idx in indices_raw}

        print(
            f"[InjetorMTXSentinela] Tracking carregado: {len(colunas_1based):,} colunas únicas identificadas."
        )
        return self.injetar_colunas(
            colunas_1based=colunas_1based,
            forcar=forcar,
            validar_regex=validar_regex,
        )

    def injetar_colunas(
        self,
        colunas_1based: Sequence[int] | set[int],
        forcar: bool = False,
        validar_regex: bool | None = None,
    ) -> dict[str, Any]:
        """Injeta o valor sentinela nas colunas especificadas diretamente no arquivo `matrix.mtx`.

        Parameters
        ----------
        colunas_1based : Sequence[int] | set[int]
            Conjunto ou sequência de posições de colunas (1-indexed) a serem preenchidas.
        forcar : bool, default=False
            Se True, sobrescreve o arquivo no diretório de destino.
        validar_regex : bool | None, optional
            Sobrescreve a checagem regex Ensembl na auditoria pós-injeção.
            Se True, sobrescreve o arquivo no diretório de destino.

        Returns
        -------
        dict[str, Any]
            Dicionário com status, dimensões finais e contagem de entradas não nulas.

        Raises
        ------
        FileNotFoundError
            Se o arquivo `matrix.mtx` de origem não for encontrado.
        ValueError
            Se o conjunto de colunas estiver vazio ou contiver índices fora dos limites da matriz.
        """
        path_mtx_origem = os.path.join(self.dir_mtx_origem, "matrix.mtx")
        path_mtx_destino = os.path.join(self.dir_mtx_destino, "matrix.mtx")

        if not os.path.exists(path_mtx_origem):
            raise FileNotFoundError(
                f"[InjetorMTXSentinela] Arquivo 'matrix.mtx' não encontrado em '{self.dir_mtx_origem}'."
            )

        colunas_set = {int(c) for c in colunas_1based}
        if not colunas_set:
            raise ValueError(
                "[InjetorMTXSentinela] O conjunto de colunas para injeção não pode ser vazio."
            )

        if (
            not forcar
            and os.path.exists(path_mtx_destino)
            and self.dir_mtx_origem != self.dir_mtx_destino
        ):
            print(
                f"[InjetorMTXSentinela] Arquivo de destino já existe em '{path_mtx_destino}' e forcar=False. Pulando injeção."
            )
            return {
                "status": "EXISTENTE",
                "dir_destino": self.dir_mtx_destino,
                "path_mtx": path_mtx_destino,
            }

        path_tmp = os.path.join(self.dir_mtx_destino, "matrix_sentinela.mtx.tmp")

        # 1. Leitura de cabeçalho e contagem de dimensões originais
        linhas_cabecalho, n_rows, n_cols, nnz_orig = self._ler_cabecalho(
            path_mtx_origem
        )

        # Validação de limites dimensionais
        col_min = min(colunas_set)
        col_max = max(colunas_set)
        if col_min < 1 or col_max > n_cols:
            raise ValueError(
                f"[InjetorMTXSentinela] Índices de coluna inválidos: min={col_min}, max={col_max}. "
                f"A matriz possui {n_cols} colunas (genes)."
            )

        print(
            f"[InjetorMTXSentinela] Matriz base: {n_rows:,} células × {n_cols:,} genes (NNZ original: {nnz_orig:,})"
        )
        print(
            f"[InjetorMTXSentinela] Injetando {self.fill_value} em {len(colunas_set):,} colunas em streaming..."
        )

        # 2. Execução do streaming OOM-Safe determinístico em duas etapas
        entradas_existentes_mantidas, entradas_injetadas = (
            self._processar_stream_seguro(
                path_in=path_mtx_origem,
                path_out=path_tmp,
                linhas_cabecalho=linhas_cabecalho,
                n_rows=n_rows,
                n_cols=n_cols,
                nnz_orig=nnz_orig,
                colunas_set=colunas_set,
            )
        )

        # 3. Substituição atômica no destino
        if os.path.exists(path_mtx_destino):
            os.remove(path_mtx_destino)
        os.replace(path_tmp, path_mtx_destino)

        # 4. Sincronização dos metadados TSV
        self._sincronizar_metadados()

        nnz_final = entradas_existentes_mantidas + entradas_injetadas
        print(
            f"[InjetorMTXSentinela] Injeção concluída com sucesso!"
            f"\n  • NNZ anterior : {nnz_orig:,}"
            f"\n  • NNZ novo     : {nnz_final:,} (+{entradas_injetadas:,} entradas {self.fill_value})"
            f"\n  • Arquivo MTX  : {path_mtx_destino}"
        )

        gc.collect()

        # 5. Auditoria de conformidade pós-gravação
        relatorio_auditoria = self._executar_auditoria_se_possivel(
            validar_regex=validar_regex
        )

        return {
            "status": "SUCESSO",
            "dir_destino": self.dir_mtx_destino,
            "path_mtx": path_mtx_destino,
            "n_celulas": n_rows,
            "n_genes": n_cols,
            "nnz_anterior": nnz_orig,
            "nnz_final": nnz_final,
            "colunas_injetadas": len(colunas_set),
            "fill_value": self.fill_value,
            "auditoria": relatorio_auditoria,
        }

    @staticmethod
    def _ler_cabecalho(path_mtx: str) -> tuple[list[str], int, int, int]:
        """Lê os comentários e extrai as dimensões da linha de dados do MTX."""
        linhas_cabecalho: list[str] = []
        n_rows = 0
        n_cols = 0
        nnz = 0

        with open(path_mtx, encoding="utf-8") as f_in:
            for line in f_in:
                if line.startswith("%"):
                    linhas_cabecalho.append(line)
                    continue
                partes = line.strip().split()
                if len(partes) >= 3:
                    n_rows = int(partes[0])
                    n_cols = int(partes[1])
                    nnz = int(partes[2])
                    break

        if n_rows == 0 or n_cols == 0:
            raise ValueError(
                f"[InjetorMTXSentinela] Falha ao extrair dimensões válidas de '{path_mtx}'."
            )

        return linhas_cabecalho, n_rows, n_cols, nnz

    def _processar_stream_seguro(
        self,
        path_in: str,
        path_out: str,
        linhas_cabecalho: list[str],
        n_rows: int,
        n_cols: int,
        nnz_orig: int,
        colunas_set: set[int],
    ) -> tuple[int, int]:
        """Processa a injeção calculando previamente o NNZ exato para escrita estritamente sequencial."""
        # Pré-passo rápido: conta se há entradas existentes que serão sobrescritas nas colunas_set
        removidos = 0
        with open(path_in, encoding="utf-8") as f_in_check:
            for line in f_in_check:
                if line.startswith("%"):
                    continue
                partes = line.strip().split()
                if len(partes) >= 3 and int(partes[1]) in colunas_set:
                    removidos += 1

        novas_entradas = n_rows * len(colunas_set)
        nnz_exato = (nnz_orig - removidos) + novas_entradas

        entradas_copiadas = 0
        entradas_injetadas = 0
        val_str = f"{self.fill_value:.6g}" if self.fill_value != 0.5 else "0.5"

        with (
            open(path_in, encoding="utf-8") as f_in,
            open(path_out, "w", encoding="utf-8") as f_out,
        ):
            # 1. Escreve comentários originais
            for c_line in linhas_cabecalho:
                f_out.write(c_line)

            # 2. Escreve a linha de dimensões com o NNZ exato
            f_out.write(f"{n_rows} {n_cols} {nnz_exato}\n")

            # 3. Fase 1: Copia entradas originais em streaming, pulando as pertencentes a colunas_set
            passou_dimensao = False
            buffer: list[str] = []

            for line in f_in:
                if not passou_dimensao:
                    if line.startswith("%"):
                        continue
                    passou_dimensao = True
                    continue

                partes = line.strip().split()
                if len(partes) < 3:
                    continue

                col_idx = int(partes[1])
                if col_idx in colunas_set:
                    continue

                buffer.append(line)
                entradas_copiadas += 1

                if len(buffer) >= self.buffer_size:
                    f_out.writelines(buffer)
                    buffer.clear()

            if buffer:
                f_out.writelines(buffer)
                buffer.clear()

            # 4. Fase 2: Injeção dos valores sentinela para todas as células nas colunas_set
            colunas_ordenadas = sorted(colunas_set)

            for row_idx in range(1, n_rows + 1):
                for col_idx in colunas_ordenadas:
                    buffer.append(f"{row_idx} {col_idx} {val_str}\n")
                    entradas_injetadas += 1

                if len(buffer) >= self.buffer_size:
                    f_out.writelines(buffer)
                    buffer.clear()

            if buffer:
                f_out.writelines(buffer)
                buffer.clear()

        return entradas_copiadas, entradas_injetadas

    def _sincronizar_metadados(self) -> None:
        """Sincroniza os arquivos de metadados TSV da pasta de origem para o destino."""
        if self.dir_mtx_origem == self.dir_mtx_destino:
            return

        for nome_arq in (
            "genes_referencia.tsv",
            "features.tsv",
            "barcodes.tsv",
            "tracking_genes_adicionados_fujita.csv",
            "tracking_genes_adicionados_Fujita.csv",
        ):
            src = os.path.join(self.dir_mtx_origem, nome_arq)
            dst = os.path.join(self.dir_mtx_destino, nome_arq)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)

    def _executar_auditoria_se_possivel(
        self, validar_regex: bool | None = None
    ) -> dict[str, Any] | None:
        """Executa auditoria via ValidadorOrdemGenes caso a lista de genes esteja disponível."""
        genes_ref = (
            list(self.genes_referencia) if self.genes_referencia is not None else None
        )

        if genes_ref is None:
            # Tenta carregar do próprio genes_referencia.tsv ou features.tsv
            for cand_nome in ("genes_referencia.tsv", "features.tsv"):
                cand_path = os.path.join(self.dir_mtx_destino, cand_nome)
                if os.path.exists(cand_path):
                    df_g = pd.read_csv(cand_path, sep="\t", header=None)
                    genes_ref = df_g[0].astype(str).tolist()
                    break

        if genes_ref is not None:
            vr = validar_regex if validar_regex is not None else self.validar_regex
            if vr is None:
                total_amostra = min(len(genes_ref), 100)
                ens_count = sum(
                    1
                    for g in genes_ref[:total_amostra]
                    if str(g).strip().startswith("ENS")
                )
                vr = (ens_count / total_amostra) > 0.8 if total_amostra > 0 else False
                if not vr:
                    print(
                        "[InjetorMTXSentinela] Identificadores de genes detectados como símbolos gênicos ou formato misto. "
                        "Validação sintática de Ensembl regex desativada automaticamente."
                    )
            return self.validador.validar_pasta_mtx(
                self.dir_mtx_destino,
                genes_referencia=genes_ref,
                validar_regex=vr,
            )

        return None

    def __repr__(self) -> str:
        """Representação textual do injetor MTX."""
        return (
            f"InjetorMTXSentinela(origem={self.dir_mtx_origem!r}, "
            f"destino={self.dir_mtx_destino!r}, fill_value={self.fill_value})"
        )
