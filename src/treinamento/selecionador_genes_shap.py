"""Módulo de Seleção de Features Celulares via SHAP sobre Modern Hopfield Networks.

Calcula valores Shapley através do algoritmo GradientExplainer (Expected Gradients)
aplicado diretamente sobre a função de atenção Softmax da Rede Hopfield Moderna,
identificando marcadores informativos por linhagem e conjuntos consolidados.
"""

from __future__ import annotations

import gc
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import scipy.sparse as sp
import shap
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray

from treinamento.hopfield import ModernHopfieldNetwork

PathType = str | os.PathLike[str]


class HopfieldClassifierWrapper(nn.Module):
    """Wrapper PyTorch diferenciável para a Modern Hopfield Network com Softmax Class Pooling.

    Permite que bibliotecas de interpretabilidade baseadas em gradiente (como o SHAP
    GradientExplainer) calculem gradientes automáticos em relação aos genes de entrada.

    Parameters
    ----------
    hopfield : ModernHopfieldNetwork
        Instância treinada da rede Hopfield contendo os padrões memorizados.
    classes : Sequence[int]
        Identificadores inteiros das classes biológicas analisadas.
    meta_padroes : Sequence[tuple[int, int]] | None, optional
        Metadados `(classe, idx)` associados a cada protótipo memorizado.
    nc : int, default=30
        Número de subclusters ou protótipos por classe (quando meta_padroes for None).

    Attributes
    ----------
    beta : float
        Temperatura inversa da atenção Softmax.
    normalize : bool
        Indicador de normalização esférica unitária (cosseno L2).
    binary : bool
        Indicador de mapeamento bipolar {-1, +1}.
    patterns : torch.Tensor
        Buffer de tensores contendo os protótipos memorizados.
    class_indicator : torch.Tensor
        Matriz binária indicadora mapeando protótipos para classes (n_padroes × n_classes).
    classes : list[int]
        Lista ordenada de classes de saída.
    """

    patterns: torch.Tensor
    class_indicator: torch.Tensor

    def __init__(
        self,
        hopfield: ModernHopfieldNetwork,
        classes: Sequence[int],
        meta_padroes: Sequence[tuple[int, int]] | None = None,
        nc: int = 30,
    ) -> None:
        super().__init__()
        if hopfield.patterns.numel() == 0:
            raise RuntimeError(
                "[HopfieldClassifierWrapper] A rede Hopfield não possui padrões armazenados."
            )

        self.beta: float = float(hopfield.beta)
        self.normalize: bool = bool(hopfield.normalize)
        self.binary: bool = bool(hopfield.binary)
        self.classes: list[int] = [int(c) for c in classes]

        n_classes: int = len(self.classes)
        padroes_tensor: torch.Tensor = (
            hopfield.patterns.detach().clone().to(torch.float32)
        )
        n_padroes: int = int(padroes_tensor.shape[0])

        self.register_buffer("patterns", padroes_tensor)

        # Mapeamento de protótipos para as classes
        indicador: torch.Tensor = torch.zeros(
            (n_padroes, n_classes), dtype=torch.float32
        )
        if meta_padroes is not None:
            classes_dos_padroes = [m[0] for m in meta_padroes]
        else:
            classes_dos_padroes = [self.classes[j // nc] for j in range(n_padroes)]

        for p_idx, c_val in enumerate(classes_dos_padroes):
            if c_val in self.classes:
                col_idx = self.classes.index(c_val)
                indicador[p_idx, col_idx] = 1.0

        self.register_buffer("class_indicator", indicador)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executa a passagem direta calculando as probabilidades de cada classe celular.

        Parameters
        ----------
        x : torch.Tensor
            Tensor de queries celulares com shape `(batch_size, n_genes)`.

        Returns
        -------
        torch.Tensor
            Matriz de probabilidades a posteriori com shape `(batch_size, n_classes)`.
        """
        # Mapeamento linear diferenciável de {0, 1} para {-1, +1}
        x_in: torch.Tensor = 2.0 * x - 1.0 if self.binary else x

        if self.normalize:
            x_norm: torch.Tensor = F.normalize(x_in, p=2, dim=-1)
            p_norm: torch.Tensor = F.normalize(self.patterns, p=2, dim=-1)
            logits: torch.Tensor = self.beta * torch.matmul(
                x_norm, p_norm.transpose(0, 1)
            )
        else:
            logits = self.beta * torch.matmul(x_in, self.patterns.transpose(0, 1))

        # Softmax Attention contínua da Hopfield
        atencao: torch.Tensor = F.softmax(logits, dim=-1)

        # Softmax Class Pooling: agregação de probabilidades por classe
        probabilidades: torch.Tensor = torch.matmul(atencao, self.class_indicator)
        return probabilidades


class SelecionadorGenesSHAPHopfield:
    """Motor de seleção e explicabilidade de features gênicas via SHAP na Hopfield.

    Aplica o algoritmo GradientExplainer sobre o wrapper diferenciável da Modern Hopfield Network,
    permitindo identificar marcadores biológicos por linhagem celular e consolidar rankings.

    Parameters
    ----------
    hopfield_net : ModernHopfieldNetwork
        Rede Hopfield treinada.
    classes : Sequence[int]
        Lista de classes canônicas.
    meta_padroes : Sequence[tuple[int, int]] | None, optional
        Metadados dos protótipos memorizados.
    nomes_classes : Sequence[str] | None, optional
        Rótulos legíveis das classes.
    nomes_genes : Sequence[str] | None, optional
        Nomes ou símbolos dos genes analisados.
    batch_size : int, default=32
        Tamanho de lote para processamento OOM-Safe.
    """

    def __init__(
        self,
        hopfield_net: ModernHopfieldNetwork | None = None,
        classes: Sequence[int] | None = None,
        meta_padroes: Sequence[tuple[int, int]] | None = None,
        nomes_classes: Sequence[str] | None = None,
        nomes_genes: Sequence[str] | None = None,
        batch_size: int = 32,
        *,
        modelo_hopfield: ModernHopfieldNetwork | None = None,
        padroes_memoria: NDArray[Any] | None = None,
        classes_padroes: Sequence[tuple[int, int]] | None = None,
        n_background_por_classe: int | None = None,
        seed: int = 42,
    ) -> None:
        net = hopfield_net if hopfield_net is not None else modelo_hopfield
        if net is None:
            raise ValueError(
                "[SelecionadorGenesSHAP] hopfield_net (ou modelo_hopfield) deve ser fornecido."
            )
        self.hopfield: ModernHopfieldNetwork = net

        if padroes_memoria is not None and (
            self.hopfield.patterns is None or self.hopfield.patterns.numel() == 0
        ):
            self.hopfield.store(padroes_memoria)

        meta = meta_padroes if meta_padroes is not None else classes_padroes
        self.meta_padroes: list[tuple[int, int]] | None = (
            list(meta) if meta is not None else None
        )

        if classes is not None:
            self.classes: list[int] = [int(c) for c in classes]
        elif self.meta_padroes is not None:
            self.classes = sorted(list({int(m[0]) for m in self.meta_padroes}))
        else:
            self.classes = list(range(1, 8))

        if nomes_classes is not None:
            self.nomes_classes: list[str] = list(nomes_classes)
        else:
            self.nomes_classes = [f"Classe_{c}" for c in self.classes]

        self.nomes_genes: list[str] | None = (
            list(nomes_genes) if nomes_genes is not None else None
        )
        self.batch_size: int = int(batch_size)
        self.seed: int = int(seed)
        self.n_background_por_classe: int | None = n_background_por_classe

        self.wrapper: HopfieldClassifierWrapper = HopfieldClassifierWrapper(
            hopfield=self.hopfield,
            classes=self.classes,
            meta_padroes=self.meta_padroes,
        )
        self.valores_shap: list[NDArray[np.float32]] | None = None
        self.amostras_explicadas: NDArray[np.float32] | None = None
        self.labels_explicados: NDArray[np.int_] | None = None

    def preparar_background(
        self,
        matriz_expressao: NDArray[Any] | sp.spmatrix,
        labels: Sequence[int] | NDArray[Any] | None = None,
        n_amostras: int = 70,
        seed: int = 42,
    ) -> torch.Tensor:
        """Seleciona uma amostra estratificada representativa para o baseline de fundo do SHAP.

        Parameters
        ----------
        matriz_expressao : NDArray | sp.spmatrix
            Matriz de contagens/expressão gênica.
        labels : Sequence[int] | NDArray | None, optional
            Rótulos das células para amostragem estratificada balanceada.
        n_amostras : int, default=70
            Quantidade de células no baseline.
        seed : int, default=42
            Semente aleatória reprodutível.

        Returns
        -------
        torch.Tensor
            Tensor PyTorch com a matriz de background baseline.
        """
        rng = np.random.RandomState(seed)
        n_total: int = int(matriz_expressao.shape[0])
        n_sel: int = min(n_amostras, n_total)

        indices: list[int] = []
        if labels is not None:
            labels_arr: NDArray[np.int_] = np.asarray(labels, dtype=int)
            amostras_por_classe: int = max(1, n_sel // len(self.classes))
            for c in self.classes:
                idx_c = np.where(labels_arr == c)[0]
                if len(idx_c) > 0:
                    escolhidos = rng.choice(
                        idx_c, size=min(len(idx_c), amostras_por_classe), replace=False
                    )
                    indices.extend(escolhidos.tolist())

            if len(indices) < n_sel:
                faltantes = n_sel - len(indices)
                restantes = np.setdiff1d(np.arange(n_total), indices)
                if len(restantes) > 0:
                    indices.extend(
                        rng.choice(
                            restantes,
                            size=min(len(restantes), faltantes),
                            replace=False,
                        ).tolist()
                    )
        else:
            indices = rng.choice(n_total, size=n_sel, replace=False).tolist()

        if sp.issparse(matriz_expressao):
            mat_csr: sp.csr_matrix = sp.csr_matrix(matriz_expressao)
            bg_np = np.asarray(mat_csr[indices].toarray(), dtype=np.float32)
        else:
            mat_dense: NDArray[np.float32] = np.asarray(
                matriz_expressao, dtype=np.float32
            )
            bg_np = mat_dense[indices]

        return torch.tensor(bg_np, dtype=torch.float32)

    def explicar(
        self,
        matriz_expressao: NDArray[Any] | sp.spmatrix,
        labels: Sequence[int] | NDArray[Any] | None = None,
        n_background: int = 70,
        n_amostras_explicar: int | None = 1000,
        batch_size: int | None = None,
        seed: int = 42,
    ) -> SelecionadorGenesSHAPHopfield:
        """Executa a interpolação GradientExplainer em mini-lotes OOM-Safe.

        Parameters
        ----------
        matriz_expressao : NDArray | sp.spmatrix
            Matriz de entrada (AnnData, NumPy ou SciPy CSR).
        labels : Sequence[int] | NDArray | None, optional
            Rótulos das células analisadas.
        n_background : int, default=70
            Tamanho da população de referência baseline.
        n_amostras_explicar : int | None, default=1000
            Número máximo de amostras estratificadas a explicar (None processa tudo).
        batch_size : int | None, optional
            Tamanho de mini-lote para avaliação de autograd.
        seed : int, default=42
            Semente aleatória para amostragem determinística.

        Returns
        -------
        SelecionadorGenesSHAPHopfield
            A própria instância com os valores SHAP computados.
        """
        b_size: int = int(batch_size if batch_size is not None else self.batch_size)
        rng = np.random.RandomState(seed)
        n_total: int = int(matriz_expressao.shape[0])

        # 1. Seleciona subamostra para explicação se n_amostras_explicar for fornecido
        idx_alvo: NDArray[np.int_]
        if n_amostras_explicar is not None and n_amostras_explicar < n_total:
            if labels is not None:
                labels_total: NDArray[np.int_] = np.asarray(labels, dtype=int)
                idx_list: list[int] = []
                por_classe: int = max(1, n_amostras_explicar // len(self.classes))
                for c in self.classes:
                    c_indices = np.where(labels_total == c)[0]
                    if len(c_indices) > 0:
                        qtd = min(len(c_indices), por_classe)
                        idx_list.extend(
                            rng.choice(c_indices, size=qtd, replace=False).tolist()
                        )
                if len(idx_list) < n_amostras_explicar:
                    sobra = n_amostras_explicar - len(idx_list)
                    resto = np.setdiff1d(np.arange(n_total), idx_list)
                    if len(resto) > 0:
                        idx_list.extend(
                            rng.choice(
                                resto, size=min(len(resto), sobra), replace=False
                            ).tolist()
                        )
                idx_alvo = np.array(idx_list, dtype=int)
            else:
                idx_alvo = rng.choice(n_total, size=n_amostras_explicar, replace=False)
        else:
            idx_alvo = np.arange(n_total, dtype=int)

        if sp.issparse(matriz_expressao):
            mat_csr_alvo: sp.csr_matrix = sp.csr_matrix(matriz_expressao)
            X_alvo: NDArray[np.float32] = np.asarray(
                mat_csr_alvo[idx_alvo].toarray(), dtype=np.float32
            )
        else:
            mat_dense_alvo: NDArray[np.float32] = np.asarray(
                matriz_expressao, dtype=np.float32
            )
            X_alvo = mat_dense_alvo[idx_alvo]

        y_alvo: NDArray[np.int_] | None = (
            np.asarray(labels, dtype=int)[idx_alvo] if labels is not None else None
        )

        self.amostras_explicadas = X_alvo
        self.labels_explicados = y_alvo

        # 2. Prepara baseline de fundo
        bg_tensor: torch.Tensor = self.preparar_background(
            matriz_expressao=matriz_expressao,
            labels=labels,
            n_amostras=n_background,
            seed=seed,
        )

        # 3. Inicializa o GradientExplainer
        self.wrapper.eval()
        explainer = shap.GradientExplainer(self.wrapper, cast(Any, bg_tensor))

        # 4. Avaliação em mini-lotes OOM-Safe
        n_queries: int = int(X_alvo.shape[0])
        shap_acumulados: list[list[NDArray[np.float32]]] = [[] for _ in self.classes]

        print(
            f"[SelecionadorGenesSHAP] Iniciando explicação de {n_queries} células "
            f"({X_alvo.shape[1]} genes) em lotes de {b_size}..."
        )

        for i in range(0, n_queries, b_size):
            i_fim = min(i + b_size, n_queries)
            lote_np = X_alvo[i:i_fim]
            lote_tensor = torch.tensor(lote_np, dtype=torch.float32)

            shap_lote = explainer.shap_values(lote_tensor)

            # Suporte ao formato 3D (n_samples, n_features, n_classes) do SHAP recente
            # e ao formato de lista de arrays legada
            if isinstance(shap_lote, np.ndarray) and shap_lote.ndim == 3:
                for c_idx in range(len(self.classes)):
                    val_c = shap_lote[:, :, c_idx].astype(np.float32)
                    shap_acumulados[c_idx].append(val_c)
            elif isinstance(shap_lote, list):
                for c_idx in range(len(self.classes)):
                    val_c = np.asarray(shap_lote[c_idx], dtype=np.float32)
                    shap_acumulados[c_idx].append(val_c)
            elif (
                isinstance(shap_lote, np.ndarray)
                and shap_lote.ndim == 2
                and len(self.classes) == 1
            ):
                shap_acumulados[0].append(shap_lote.astype(np.float32))

            del lote_tensor, shap_lote
            if i % (b_size * 10) == 0:
                gc.collect()

        # Consolida tensores SHAP por classe
        self.valores_shap = [np.vstack(lista_c) for lista_c in shap_acumulados]

        print(
            f"[SelecionadorGenesSHAP] Concluído. Matrizes SHAP calculadas para "
            f"{len(self.classes)} classes celulares."
        )
        return self

    def ajustar(
        self,
        X: NDArray[Any] | sp.spmatrix,
        y: Sequence[int] | NDArray[Any] | None = None,
        n_background: int | None = None,
    ) -> SelecionadorGenesSHAPHopfield:
        """Executa a calibração de background e o cálculo de explicabilidade SHAP.

        Parameters
        ----------
        X : NDArray | sp.spmatrix
            Matriz de expressão celular.
        y : Sequence[int] | NDArray | None, optional
            Rótulos celulares.
        n_background : int | None, optional
            Tamanho da população de referência baseline.

        Returns
        -------
        SelecionadorGenesSHAPHopfield
            A própria instância calculada.
        """
        bg_samples = (
            n_background
            if n_background is not None
            else (
                self.n_background_por_classe * len(self.classes)
                if self.n_background_por_classe is not None
                else 70
            )
        )
        return self.explicar(
            matriz_expressao=X,
            labels=y,
            n_background=bg_samples,
            n_amostras_explicar=None,
            seed=self.seed,
        )

    def obter_ranking_por_classe(self, top_n: int = 50) -> pl.DataFrame:
        """Gera a tabela estruturada com os genes marcadores com maior impacto SHAP por classe.

        Parameters
        ----------
        top_n : int, default=50
            Quantidade de principais genes a selecionar por tipo celular.

        Returns
        -------
        pl.DataFrame
            DataFrame Polars com as colunas:
            `gene`, `classe`, `nome_classe`, `shap_medio_positivo`, `frequencia_expressao`.
        """
        if self.valores_shap is None or self.amostras_explicadas is None:
            raise RuntimeError(
                "[SelecionadorGenesSHAP] Execute .explicar() antes de extrair os rankings."
            )

        n_genes: int = int(self.amostras_explicadas.shape[1])
        gene_names: list[str] = (
            self.nomes_genes
            if self.nomes_genes is not None and len(self.nomes_genes) == n_genes
            else [f"Gene_{j}" for j in range(n_genes)]
        )

        linhas: list[dict[str, Any]] = []

        for c_idx, c_val in enumerate(self.classes):
            c_nome = self.nomes_classes[c_idx]
            shap_c: NDArray[np.float32] = self.valores_shap[c_idx]

            # Foco em células da própria classe se houver rótulos disponíveis
            if self.labels_explicados is not None:
                mask_c = self.labels_explicados == c_val
                if mask_c.sum() > 0:
                    shap_alvo = shap_c[mask_c]
                    x_alvo = self.amostras_explicadas[mask_c]
                else:
                    shap_alvo = shap_c
                    x_alvo = self.amostras_explicadas
            else:
                shap_alvo = shap_c
                x_alvo = self.amostras_explicadas

            # Média de contribuição positiva do gene para a classe: mean(max(0, SHAP))
            shap_positivo = np.maximum(0.0, shap_alvo)
            score_genes = shap_positivo.mean(axis=0)
            freq_genes = (x_alvo > 0.0).mean(axis=0)

            n_top_real = min(top_n, n_genes)
            top_indices = np.argsort(score_genes)[-n_top_real:][::-1]

            for idx in top_indices:
                linhas.append(
                    {
                        "gene": gene_names[idx],
                        "classe": int(c_val),
                        "nome_classe": str(c_nome),
                        "shap_medio_positivo": float(score_genes[idx]),
                        "frequencia_expressao": float(freq_genes[idx]),
                    }
                )

        return pl.DataFrame(linhas)

    def obter_genes_consolidados(self, top_n_por_classe: int = 50) -> pl.DataFrame:
        """Gera a lista unificada e consolidada de features ordenadas por relevância máxima.

        Parameters
        ----------
        top_n_por_classe : int, default=50
            Número de marcadores prioritários por linhagem a incluir na união.

        Returns
        -------
        pl.DataFrame
            DataFrame Polars com as colunas:
            `gene`, `max_shap`, `classes_associadas`, `ranking_consolidado`.
        """
        df_linhagens = self.obter_ranking_por_classe(top_n=top_n_por_classe)

        # Agrupa genes unificando classes e calculando o maior escore de impacto
        df_agrupado = (
            df_linhagens.group_by("gene")
            .agg(
                [
                    pl.col("shap_medio_positivo").max().alias("max_shap"),
                    pl.col("nome_classe")
                    .str.join(delimiter="; ")
                    .alias("classes_associadas"),
                ]
            )
            .sort("max_shap", descending=True)
            .with_columns(pl.int_range(1, pl.len() + 1).alias("ranking_consolidado"))
        )
        return df_agrupado

    def obter_rankings(
        self,
        top_n_por_classe: int = 15,
        out_dir_csv: PathType | None = None,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Gera e retorna simultaneamente os rankings por linhagem e consolidado global.

        Parameters
        ----------
        top_n_por_classe : int, default=15
            Quantidade de marcadores de topo por linhagem celular.
        out_dir_csv : str | Path | None, optional
            Diretório opcional para exportação automática dos relatórios CSV.

        Returns
        -------
        tuple[pl.DataFrame, pl.DataFrame]
            Tupla contendo (df_marcadores_classe, df_genes_globais).
        """
        df_classes = self.obter_ranking_por_classe(top_n=top_n_por_classe)
        df_global = self.obter_genes_consolidados(top_n_por_classe=top_n_por_classe)
        if out_dir_csv is not None:
            self.salvar_relatorio(
                out_dir=out_dir_csv, top_n_por_classe=top_n_por_classe
            )
        return df_classes, df_global

    def filtrar_matriz(
        self,
        in_path: PathType,
        out_path: PathType,
        genes_selecionados: Sequence[str],
    ) -> SelecionadorGenesSHAPHopfield:
        """Gera uma nova matriz persistida contendo apenas as colunas dos genes selecionados.

        Parameters
        ----------
        in_path : str | Path
            Caminho do arquivo de entrada (.npy ou .csv).
        out_path : str | Path
            Caminho de gravação da matriz filtrada (.npy ou .csv).
        genes_selecionados : Sequence[str]
            Lista de nomes/identificadores dos genes a manter.

        Returns
        -------
        SelecionadorGenesSHAPHopfield
            A própria instância.
        """
        in_str: str = str(in_path)
        out_str: str = str(out_path)
        os.makedirs(os.path.dirname(os.path.abspath(out_str)), exist_ok=True)

        genes_alvo: list[str] = list(genes_selecionados)

        if in_str.endswith(".npy"):
            mat: NDArray[Any] = np.load(in_str)
            if self.nomes_genes is None or len(self.nomes_genes) != mat.shape[1]:
                raise RuntimeError(
                    "[SelecionadorGenesSHAP] Lista 'nomes_genes' incompatível com as colunas da matriz .npy."
                )

            indices = [
                self.nomes_genes.index(g) for g in genes_alvo if g in self.nomes_genes
            ]
            mat_filtrada = mat[:, indices]
            np.save(out_str, mat_filtrada)
            print(
                f"[SelecionadorGenesSHAP] Matriz .npy filtrada salva em: {out_str} ({mat_filtrada.shape})"
            )
        else:
            with open(in_str, encoding="utf-8") as fh:
                header = fh.readline().strip("\n").strip("\r").split(",")

            coluna_id = header[0]
            colunas_validas = [coluna_id] + [c for c in genes_alvo if c in header]

            if out_str.endswith(".npy"):
                df_lazy = pl.scan_csv(in_str).select(colunas_validas).collect()
                arr = df_lazy.select(colunas_validas[1:]).to_numpy().astype(np.float32)
                np.save(out_str, arr)
                print(
                    f"[SelecionadorGenesSHAP] Matriz .npy gerada a partir de CSV: {out_str} ({arr.shape})"
                )
            else:
                pl.scan_csv(in_str).select(colunas_validas).sink_csv(out_str)
                print(
                    f"[SelecionadorGenesSHAP] Matriz CSV filtrada salva em: {out_str}"
                )

        return self

    def salvar_relatorio(
        self,
        out_dir: PathType = "outputs/shap",
        top_n_por_classe: int = 50,
    ) -> SelecionadorGenesSHAPHopfield:
        """Exporta os relatórios estruturados de marcadores e genes consolidados para CSV.

        Parameters
        ----------
        out_dir : str | Path, default="outputs/shap"
            Diretório de destino dos arquivos.
        top_n_por_classe : int, default=50
            Quantidade de marcadores por classe no relatório.

        Returns
        -------
        SelecionadorGenesSHAPHopfield
            A própria instância.
        """
        pasta = Path(out_dir)
        pasta.mkdir(parents=True, exist_ok=True)

        df_classes = self.obter_ranking_por_classe(top_n=top_n_por_classe)
        path_classes = pasta / "top_marcadores_por_classe.csv"
        df_classes.write_csv(str(path_classes))

        df_consol = self.obter_genes_consolidados(top_n_por_classe=top_n_por_classe)
        path_consol = pasta / "genes_consolidados_shap.csv"
        df_consol.write_csv(str(path_consol))

        print(f"[SelecionadorGenesSHAP] Relatórios persistidos em: {pasta}")
        return self

    def plotar_sumario(
        self,
        top_n: int = 15,
        out_png: PathType | None = None,
    ) -> None:
        """Gera visualização gráfica dos biomarcadores mais fortes para cada linhagem celular.

        Parameters
        ----------
        top_n : int, default=15
            Quantidade de genes a exibir por subclasse.
        out_png : str | Path | None, optional
            Caminho do arquivo PNG de destino (se None, exibe na tela).
        """
        df_rank = self.obter_ranking_por_classe(top_n=top_n)
        n_classes = len(self.classes)

        fig, axes = plt.subplots(
            nrows=n_classes,
            ncols=1,
            figsize=(10, 3.0 * n_classes),
            sharex=False,
            constrained_layout=True,
        )
        if n_classes == 1:
            axes = [axes]

        try:
            for idx, c_val in enumerate(self.classes):
                ax = axes[idx]
                df_c = df_rank.filter(pl.col("classe") == c_val).sort(
                    "shap_medio_positivo", descending=True
                )

                genes_c = df_c["gene"].to_list()[::-1]
                scores_c = df_c["shap_medio_positivo"].to_list()[::-1]
                nome_c = self.nomes_classes[idx]

                ax.barh(genes_c, scores_c, color="teal", alpha=0.85)
                ax.set_title(
                    f"Top Biomarcadores SHAP: {nome_c} (Classe {c_val})",
                    fontsize=11,
                    fontweight="bold",
                )
                ax.set_xlabel("Impacto SHAP Positivo Médio", fontsize=9)
                ax.grid(axis="x", linestyle="--", alpha=0.5)

            if out_png is not None:
                caminho_fig = Path(out_png)
                caminho_fig.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(str(caminho_fig), dpi=200, bbox_inches="tight")
                print(f"[SelecionadorGenesSHAP] Gráfico salvo em: {caminho_fig}")
            else:
                plt.show()
        finally:
            plt.close(fig)

    def plotar_heatmap_marcadores(
        self,
        top_n_por_classe: int = 10,
        out_png: PathType | None = None,
        normalizar_linhas: bool = True,
        cmap: str = "YlGnBu",
        figsize: tuple[float, float] | None = None,
    ) -> None:
        """Gera um mapa de calor (heatmap) dos genes com maior impacto SHAP por tipo celular.

        Parameters
        ----------
        top_n_por_classe : int, default=10
            Quantidade de genes mais informativos por linhagem a incluir no mapa.
        out_png : str | Path | None, optional
            Caminho para gravação da figura PNG em alta resolução. Se None, exibe na tela.
        normalizar_linhas : bool, default=True
            Se True, normaliza a intensidade de cada gene entre [0, 1] destacando a especificidade.
        cmap : str, default="YlGnBu"
            Paleta de cores para o heatmap (ex: "YlGnBu", "magma", "viridis").
        figsize : tuple[float, float] | None, optional
            Dimensões personalizadas da figura em polegadas (largura, altura).
        """
        if self.valores_shap is None or self.amostras_explicadas is None:
            raise RuntimeError(
                "[SelecionadorGenesSHAP] Execute .explicar() antes de plotar o heatmap."
            )

        # 1. Obtém os top genes de cada classe
        df_rank = self.obter_ranking_por_classe(top_n=top_n_por_classe)

        # Preserva a ordem agrupada por linhagem eliminando duplicatas
        genes_ordenados: list[str] = []
        for c_val in self.classes:
            genes_c = (
                df_rank.filter(pl.col("classe") == c_val)
                .sort("shap_medio_positivo", descending=True)["gene"]
                .to_list()
            )
            for g in genes_c:
                if g not in genes_ordenados:
                    genes_ordenados.append(g)

        if not genes_ordenados:
            return

        n_genes_sel: int = len(genes_ordenados)
        n_classes: int = len(self.classes)
        n_genes_total: int = int(self.amostras_explicadas.shape[1])

        nomes_referencia: list[str] = (
            self.nomes_genes
            if self.nomes_genes is not None and len(self.nomes_genes) == n_genes_total
            else [f"Gene_{j}" for j in range(n_genes_total)]
        )
        idx_map: dict[str, int] = {
            g: nomes_referencia.index(g)
            for g in genes_ordenados
            if g in nomes_referencia
        }

        # 2. Constrói a matriz M (n_genes_sel × n_classes)
        matriz_m: NDArray[np.float32] = np.zeros(
            (n_genes_sel, n_classes), dtype=np.float32
        )

        for c_idx, c_val in enumerate(self.classes):
            shap_c: NDArray[np.float32] = self.valores_shap[c_idx]
            if self.labels_explicados is not None:
                mask_c = self.labels_explicados == c_val
                shap_alvo = shap_c[mask_c] if mask_c.sum() > 0 else shap_c
            else:
                shap_alvo = shap_c

            shap_pos = np.maximum(0.0, shap_alvo)
            mean_pos: NDArray[np.float32] = np.asarray(
                shap_pos.mean(axis=0), dtype=np.float32
            )

            for g_row, g_nome in enumerate(genes_ordenados):
                if g_nome in idx_map:
                    g_col = idx_map[g_nome]
                    matriz_m[g_row, c_idx] = float(mean_pos[g_col])

        # 3. Normalização Min-Max por linha com proteção contra divisão por zero
        matriz_plot: NDArray[np.float32]
        label_cbar: str
        if normalizar_linhas:
            min_l = matriz_m.min(axis=1, keepdims=True)
            max_l = matriz_m.max(axis=1, keepdims=True)
            den = max_l - min_l
            den[den == 0.0] = 1.0
            matriz_plot = (matriz_m - min_l) / den
            label_cbar = "Impacto Relativo (Normalizado [0, 1])"
        else:
            matriz_plot = matriz_m
            label_cbar = "Impacto SHAP Positivo Médio"

        # 4. Renderização do Heatmap
        dim_fig: tuple[float, float] = (
            figsize
            if figsize is not None
            else (max(7.5, 1.2 * n_classes), max(6.0, 0.35 * n_genes_sel))
        )
        fig, ax = plt.subplots(figsize=dim_fig, constrained_layout=True)

        try:
            im = ax.imshow(
                matriz_plot, cmap=cmap, aspect="auto", interpolation="nearest"
            )
            cbar = fig.colorbar(im, ax=ax, shrink=0.7)
            cbar.set_label(label_cbar, fontsize=10)

            # Rótulos dos eixos
            ax.set_xticks(np.arange(n_classes))
            ax.set_xticklabels(
                self.nomes_classes,
                rotation=30,
                ha="right",
                fontsize=10,
                fontweight="bold",
            )

            ax.set_yticks(np.arange(n_genes_sel))
            ax.set_yticklabels(genes_ordenados, fontsize=8)

            ax.set_title(
                f"Mapa de Calor de Biomarcadores Celulares (Top {top_n_por_classe} por Linhagem)",
                fontsize=12,
                fontweight="bold",
                pad=12,
            )
            ax.set_xlabel(
                "Tipos Celulares Canônicos",
                fontsize=11,
                fontweight="bold",
                labelpad=8,
            )
            ax.set_ylabel("Genes Marcadores", fontsize=11, fontweight="bold")

            if out_png is not None:
                caminho_fig = Path(out_png)
                caminho_fig.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(str(caminho_fig), dpi=200, bbox_inches="tight")
                print(f"[SelecionadorGenesSHAP] Heatmap salvo em: {caminho_fig}")
            else:
                plt.show()
        finally:
            plt.close(fig)
