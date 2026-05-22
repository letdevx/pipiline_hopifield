import os
import numpy as np
import pandas as pd

from .hopfield_utils import sorti


class CarregadorDadosFujita:
    """Carrega os dados do dataset Fujita para análise com rede Hopfield.

    Atributos
    ---------
    path_matriz  : caminho para a matriz binária (.txt, células × genes)
    path_genes   : caminho para o CSV com os top genes de referência
    path_labels  : caminho para o arquivo de rótulos de tipo celular (.txt)
    path_sweep   : (opcional) caminho para o CSV com projeções SWeeP pré-computadas
    n_genes      : número de genes mais frequentes a selecionar (padrão: 5000)
    X            : matriz binária completa carregada (células × genes_totais)
    W0           : submatriz com os top n_genes mais frequentes (células × n_genes)
    ids_top      : índices dos top genes selecionados em X
    genes        : DataFrame com a lista de genes de referência
    labels       : array de rótulos inteiros de tipo celular
    Wswp         : matriz de projeções SWeeP pré-computadas (ou None)
    """

    def __init__(self, path_matriz, path_genes, path_labels, path_sweep=None, n_genes=5000):
        self.path_matriz = path_matriz
        self.path_genes = path_genes
        self.path_labels = path_labels
        self.path_sweep = path_sweep
        self.n_genes = n_genes
        self.X = None
        self.W0 = None
        self.ids_top = None
        self.genes = None
        self.labels = None
        self.Wswp = None

    def carregar(self):
        """Carrega todos os arquivos de entrada e seleciona os top genes.

        Retorna o próprio objeto para permitir encadeamento de chamadas.
        """
        self._carregar_matriz()
        self._selecionar_top_genes()
        self._carregar_genes()
        self._carregar_labels()
        if self.path_sweep is not None:
            self._carregar_sweep()
        print(f"[CarregadorDadosFujita] Carregamento concluído: "
              f"{self.X.shape[0]} células, {self.n_genes} genes selecionados")
        return self

    def _carregar_matriz(self):
        print(f"[CarregadorDadosFujita] Carregando matriz: {self.path_matriz}")
        self.X = pd.read_csv(self.path_matriz).to_numpy(dtype=np.float32)
        print(f"[CarregadorDadosFujita] Matriz carregada: {self.X.shape}")

    def _selecionar_top_genes(self):
        # O arquivo já contém apenas os top N genes na ordem correta (frequência Fujita).
        # Re-ordenar aqui causaria mismatch com W_mathys carregado diretamente do CSV.
        self.ids_top = np.arange(self.X.shape[1])
        self.W0 = self.X
        print(f"[CarregadorDadosFujita] W0 shape: {self.W0.shape}")

    def _carregar_genes(self):
        print(f"[CarregadorDadosFujita] Carregando genes: {self.path_genes}")
        self.genes = pd.read_csv(self.path_genes)
        print(f"[CarregadorDadosFujita] {len(self.genes)} genes carregados")

    def _carregar_labels(self):
        print(f"[CarregadorDadosFujita] Carregando rótulos: {self.path_labels}")
        self.labels = np.loadtxt(self.path_labels, dtype=int).ravel()
        tipos = np.unique(self.labels)
        print(f"[CarregadorDadosFujita] Rótulos shape: {self.labels.shape}, tipos: {tipos}")

    def _carregar_sweep(self):
        print(f"[CarregadorDadosFujita] Carregando SWeeP pré-computado: {self.path_sweep}")
        self.Wswp = pd.read_csv(self.path_sweep).to_numpy(dtype=np.float32)
        print(f"[CarregadorDadosFujita] Wswp shape: {self.Wswp.shape}")

    def __repr__(self):
        x_shape = self.X.shape if self.X is not None else "não carregado"
        w0_shape = self.W0.shape if self.W0 is not None else "não gerado"
        wswp = self.Wswp.shape if self.Wswp is not None else "não carregado"
        labels = self.labels.shape if self.labels is not None else "não carregados"
        return (
            f"CarregadorDadosFujita(\n"
            f"  path_matriz  = {self.path_matriz}\n"
            f"  path_genes   = {self.path_genes}\n"
            f"  path_labels  = {self.path_labels}\n"
            f"  path_sweep   = {self.path_sweep}\n"
            f"  X            = {x_shape}\n"
            f"  W0           = {w0_shape}\n"
            f"  labels       = {labels}\n"
            f"  Wswp         = {wswp}\n"
            f")"
        )
