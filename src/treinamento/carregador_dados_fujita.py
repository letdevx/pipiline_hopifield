import os
import numpy as np
import pandas as pd

from .hopfield_utils import sorti


def carregar_labels(path_ou_array):
    """Carrega array 1D de rótulos de tipo celular a partir de arquivo (.txt, .csv, .tsv) ou array.
    
    Suporta arquivos com ou sem cabeçalho e diferentes delimitadores.
    """
    if path_ou_array is None:
        return None
    if isinstance(path_ou_array, (np.ndarray, list, pd.Series)):
        return np.asarray(path_ou_array, dtype=int).ravel()
    
    path_str = str(path_ou_array)
    try:
        labels = np.loadtxt(path_str, dtype=int).ravel()
    except Exception:
        try:
            labels = np.loadtxt(path_str, dtype=int, skiprows=1).ravel()
        except Exception:
            df = pd.read_csv(path_str, header=None)
            try:
                labels = df.iloc[:, 0].to_numpy(dtype=int).ravel()
            except ValueError:
                labels = pd.read_csv(path_str).iloc[:, 0].to_numpy(dtype=int).ravel()
    return labels


class CarregadorDados:
    """Carregador unificado de dados scRNA-seq para análise com rede Hopfield.

    Atributos
    ---------
    path_matriz  : caminho para a matriz binária (.h5ad, .npy, .csv)
    path_genes   : caminho para arquivo de genes ou lista/DataFrame com genes
    path_labels  : caminho para o arquivo de rótulos ou array
    path_sweep   : (opcional) caminho para o CSV com projeções SWeeP pré-computadas
    n_genes      : número de genes a considerar
    X            : matriz binária carregada (células × genes_totais)
    W0           : matriz de trabalho (células × n_genes)
    ids_top      : índices dos genes selecionados
    genes        : DataFrame com a lista de genes
    labels       : array de rótulos inteiros de tipo celular
    Wswp         : matriz de projeções SWeeP pré-computadas (ou None)
    """

    def __init__(self, path_matriz, path_genes=None, path_labels=None, path_sweep=None, n_genes=None):
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
        """Carrega todos os arquivos de entrada.

        Retorna o próprio objeto para permitir encadeamento de chamadas.
        """
        self._carregar_matriz()
        self._selecionar_top_genes()
        if self.path_genes is not None:
            self._carregar_genes()
        if self.path_labels is not None:
            self._carregar_labels()
        if self.path_sweep is not None:
            self._carregar_sweep()
        print(f"[{self.__class__.__name__}] Carregamento concluído: "
              f"{self.X.shape[0]} células, {self.n_genes} genes selecionados")
        return self

    def _carregar_matriz(self):
        print(f"[{self.__class__.__name__}] Carregando matriz: {self.path_matriz}")
        if str(self.path_matriz).endswith('.npy'):
            self.X = np.load(self.path_matriz, mmap_mode='r')
        elif str(self.path_matriz).endswith('.h5ad'):
            import anndata as ad
            adata = ad.read_h5ad(self.path_matriz)
            self.X = adata.X
        else:
            self.X = pd.read_csv(self.path_matriz).to_numpy(dtype=np.float32)
        print(f"[{self.__class__.__name__}] Matriz carregada: {self.X.shape}")

    def _selecionar_top_genes(self):
        self.ids_top = np.arange(self.X.shape[1])
        self.W0 = self.X
        self.n_genes = self.X.shape[1]
        print(f"[{self.__class__.__name__}] W0 shape: {self.W0.shape} ({self.n_genes} genes)")

    def _carregar_genes(self):
        print(f"[{self.__class__.__name__}] Carregando genes: {self.path_genes}")
        if isinstance(self.path_genes, pd.DataFrame):
            self.genes = self.path_genes
        elif isinstance(self.path_genes, (list, np.ndarray, tuple)):
            self.genes = pd.DataFrame({'gene': list(self.path_genes)})
        else:
            self.genes = pd.read_csv(self.path_genes)
        print(f"[{self.__class__.__name__}] {len(self.genes)} genes carregados")

    def _carregar_labels(self):
        print(f"[{self.__class__.__name__}] Carregando rótulos: {self.path_labels}")
        self.labels = carregar_labels(self.path_labels)
        tipos = np.unique(self.labels)
        print(f"[{self.__class__.__name__}] Rótulos shape: {self.labels.shape}, tipos: {tipos}")

    def _carregar_sweep(self):
        if os.path.exists(self.path_sweep):
            print(f"[{self.__class__.__name__}] Carregando SWeeP pré-computado: {self.path_sweep}")
            self.Wswp = pd.read_csv(self.path_sweep).to_numpy(dtype=np.float32)
            print(f"[{self.__class__.__name__}] Wswp shape: {self.Wswp.shape}")

    def __repr__(self):
        x_shape = self.X.shape if self.X is not None else "não carregado"
        w0_shape = self.W0.shape if self.W0 is not None else "não gerado"
        wswp = self.Wswp.shape if self.Wswp is not None else "não carregado"
        labels = self.labels.shape if self.labels is not None else "não carregados"
        return (
            f"{self.__class__.__name__}(\n"
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


# Alias para retrocompatibilidade
CarregadorDadosFujita = CarregadorDados
