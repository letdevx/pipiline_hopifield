import numpy as np
from sklearn.cluster import KMeans

from .hopfield_utils import closervects


class ExtratorPadroesSubcluster:
    """Extrai perfis binários representativos por subcluster (perf35).

    Para cada classe em `classes`, executa KMeans com `nc` clusters no
    espaço SWeeP e seleciona o vetor binário mais próximo de cada centroide
    como representante. Resulta em len(classes) × nc padrões.

    Atributos
    ---------
    W0       : matriz binária completa (células × genes)
    labels   : array de rótulos de classe por célula
    classes  : lista de classes a processar
    nc       : número de subclusters por classe
    seed     : semente para reprodutibilidade do KMeans
    padroes  : array de padrões extraídos (len(classes)*nc × genes)
    meta     : lista de tuplas (classe, idx_global) por padrão
    """

    def __init__(self, W0, labels, classes=None, nc=10, seed=42, k=1):
        self.W0 = np.asarray(W0, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=int)
        self.classes = classes if classes is not None else [1, 3, 4, 5, 6, 7, 2]
        # nc pode ser int (mesmo valor para todas as classes) ou dict {classe: nc}
        self.nc = nc
        self.seed = seed
        self.k = k
        self.padroes = None
        self.meta = None

    def extrair(self, Wswp):
        """Extrai padrões representativos de subclusters no espaço SWeeP.

        Para cada classe, aplica KMeans(nc) em Wswp e seleciona o vetor
        de W0 mais próximo de cada centroide como protótipo.

        Retorna o próprio objeto para permitir encadeamento de chamadas.
        """
        Wswp = np.asarray(Wswp, dtype=np.float32)
        classes_validas = [c for c in self.classes if (self.labels == c).any()]
        padroes_list = []
        meta_list = []

        for cls in classes_validas:
            nc_cls = self.nc.get(cls, self.nc.get("default", 10)) if isinstance(self.nc, dict) else self.nc
            ids_cls = np.where(self.labels == cls)[0]
            Wk4 = self.W0[ids_cls]
            Wswp_cls = Wswp[ids_cls]

            print(f"[ExtratorPadroesSubcluster] Classe {cls}: n={len(ids_cls)}, "
                  f"KMeans(nc={nc_cls})...")
            km = KMeans(n_clusters=nc_cls, n_init=5, random_state=self.seed).fit(Wswp_cls)

            for centroide in km.cluster_centers_:
                if self.k == 1:
                    idx_local = closervects(Wswp_cls, centroide, k=1)
                    padroes_list.append(Wk4[idx_local])
                    meta_list.append((cls, int(ids_cls[idx_local])))
                else:
                    idxs = closervects(Wswp_cls, centroide, k=self.k)
                    padrao = (Wk4[idxs].mean(axis=0) > 0.5).astype(np.float32)
                    padroes_list.append(padrao)
                    meta_list.append((cls, int(ids_cls[idxs[0]])))

        self.padroes = np.vstack(padroes_list).astype(np.float32)
        self.meta = meta_list
        nc_resumo = self.nc if not isinstance(self.nc, dict) else self.nc
        print(f"[ExtratorPadroesSubcluster] Extração concluída: "
              f"{self.padroes.shape[0]} padrões ({len(classes_validas)} classes)")
        return self

    def __repr__(self):
        padroes = self.padroes.shape if self.padroes is not None else "não extraídos"
        nc_str = str(self.nc) if not isinstance(self.nc, dict) else "{" + ", ".join(
            f"{k}: {v}" for k, v in sorted(self.nc.items())) + "}"
        return (
            f"ExtratorPadroesSubcluster(\n"
            f"  W0       = {self.W0.shape}\n"
            f"  labels   = {self.labels.shape}\n"
            f"  classes  = {self.classes}\n"
            f"  nc       = {nc_str}\n"
            f"  k        = {self.k}\n"
            f"  seed     = {self.seed}\n"
            f"  padroes  = {padroes}\n"
            f")"
        )
