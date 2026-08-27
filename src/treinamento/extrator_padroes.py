import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans

from .hopfield_utils import closervects
from .estrategias_clusterizacao import EstrategiaKMeansDinamico, EstrategiaKMeansFixo

class ExtratorPadroesSubcluster:
    """Extrai perfis binários representativos por subcluster (perf35).

    Utiliza uma Estratégia de Clusterização injetada (KMeans Dinâmico, HDBSCAN, etc)
    no espaço SWeeP e seleciona o vetor binário mais próximo de cada centroide
    como representante. Resulta em um número dinâmico de padrões otimizados.

    Atributos
    ---------
    W0         : matriz binária completa (células × genes)
    labels     : array de rótulos de classe por célula
    classes    : lista de classes a processar
    estrategia : Instância de uma EstrategiaClusterizacao
    seed       : semente para reprodutibilidade
    padroes    : array de padrões binários (n_dinamico × genes)
    meta       : lista de tuplas (classe, idx_global) por padrão
    """

    def __init__(self, W0, labels, classes=None, estrategia=None, seed=42, k=1, nc=None):
        if sp.issparse(W0):
            self.W0 = W0.tocsr()
        else:
            self.W0 = np.asarray(W0, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=int)
        self.classes = classes if classes is not None else [1, 3, 4, 5, 6, 7, 2]
        
        # Fallback para retrocompatibilidade
        if estrategia is None:
            if nc is not None:
                self.estrategia = EstrategiaKMeansFixo(n_clusters=nc, seed=seed)
            else:
                self.estrategia = EstrategiaKMeansDinamico(k_range=[10], seed=seed)
        else:
            self.estrategia = estrategia
            
        self.seed = seed
        self.k = k
        self.padroes = None
        self.meta = None

    def extrair(self, Wswp):
        """Extrai padrões representativos de subclusters no espaço SWeeP.

        Para cada classe, aplica a Estratégia de Clusterização em Wswp e seleciona
        o vetor de W0 mais próximo de cada centroide biológico como protótipo.

        Retorna o próprio objeto para permitir encadeamento de chamadas.
        """
        Wswp = np.asarray(Wswp, dtype=np.float32)
        classes_validas = [c for c in self.classes if (self.labels == c).any()]
        padroes_list = []
        meta_list = []

        for cls in classes_validas:
            ids_cls = np.where(self.labels == cls)[0]
            Wswp_cls = Wswp[ids_cls]

            print(f"[ExtratorPadroesSubcluster] Classe {cls}: n={len(ids_cls)} "
                  f"Aplicando estratégia de clusterização...")
                  
            # Aplica o padrão Strategy
            centroides = self.estrategia.clusterizar(Wswp_cls)

            for centroide in centroides:
                if self.k == 1:
                    idx_local = closervects(Wswp_cls, centroide, k=1)
                    idx_global = ids_cls[idx_local]
                    row = self.W0[idx_global]
                    if sp.issparse(row):
                        row = row.toarray().ravel()
                    padroes_list.append(row.astype(np.float32, copy=False))
                    meta_list.append((cls, int(idx_global)))
                else:
                    idxs = closervects(Wswp_cls, centroide, k=self.k)
                    idxs_global = ids_cls[idxs]
                    sub = self.W0[idxs_global]
                    if sp.issparse(sub):
                        sub = sub.toarray()
                    padrao = (sub.astype(np.float32, copy=False).mean(axis=0) >= 0.5).astype(np.float32)
                    padroes_list.append(padrao)
                    meta_list.append((cls, int(ids_cls[idxs[0]])))

        self.padroes = np.vstack(padroes_list).astype(np.float32)
        self.meta = meta_list
        print(f"[ExtratorPadroesSubcluster] Extração concluída: "
              f"{self.padroes.shape[0]} padrões ({len(classes_validas)} classes, amostragem k={self.k})", flush=True)
        return self

    def __repr__(self):
        padroes = self.padroes.shape if self.padroes is not None else "não extraídos"
        return (
            f"ExtratorPadroesSubcluster(\n"
            f"  W0         = {self.W0.shape}\n"
            f"  labels     = {self.labels.shape}\n"
            f"  classes    = {self.classes}\n"
            f"  estrategia = {self.estrategia.__class__.__name__}\n"
            f"  k          = {self.k}\n"
            f"  seed       = {self.seed}\n"
            f"  padroes    = {padroes}\n"
            f")"
        )
