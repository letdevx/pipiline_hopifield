import numpy as np
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
import warnings

class EstrategiaClusterizacao(ABC):
    """Interface base para as estratégias de clusterização."""
    
    @abstractmethod
    def clusterizar(self, Wswp_cls):
        """
        Recebe os vetores SWeeP da classe atual e retorna os centróides dos clusters.
        
        Args:
            Wswp_cls (np.ndarray): Matriz de vetores SWeeP (células x features)
            
        Returns:
            np.ndarray: Array de centróides (clusters x features)
        """
        pass


class EstrategiaKMeansDinamico(EstrategiaClusterizacao):
    """
    Estratégia que testa vários valores de k (clusters) e escolhe o melhor
    usando o Calinski-Harabasz Index (Variance Ratio Criterion).
    """
    def __init__(self, k_range=range(2, 16), seed=42):
        self.k_range = k_range
        self.seed = seed
        
    def clusterizar(self, Wswp_cls):
        n_samples = len(Wswp_cls)
        
        # Se houver menos células do que o k mínimo, retorna a média de todas como centróide
        if n_samples < min(self.k_range):
            print(f"[EstrategiaKMeansDinamico] Poucas amostras ({n_samples}). Usando 1 cluster.")
            return np.array([Wswp_cls.mean(axis=0)])
            
        best_score = -1
        best_k = min(self.k_range)
        best_centroids = None
        
        # Filtra os k possíveis (não pode ter k >= n_samples)
        valid_ks = [k for k in self.k_range if k < n_samples]
        
        if not valid_ks:
            return np.array([Wswp_cls.mean(axis=0)])

        for k in valid_ks:
            km = KMeans(n_clusters=k, n_init=5, random_state=self.seed)
            labels = km.fit_predict(Wswp_cls)
            
            # Se apenas um cluster foi encontrado de alguma forma (ou todas as células são muito iguais)
            if len(set(labels)) < 2:
                continue
                
            score = calinski_harabasz_score(Wswp_cls, labels)
            if score > best_score:
                best_score = score
                best_k = k
                best_centroids = km.cluster_centers_
                
        if best_centroids is None:
            # Fallback
            best_centroids = np.array([Wswp_cls.mean(axis=0)])
            best_k = 1
            
        print(f"[EstrategiaKMeansDinamico] Melhor k encontrado: {best_k} (Calinski-Harabasz: {best_score:.2f})")
        return best_centroids


class EstrategiaKMeansFixo(EstrategiaClusterizacao):
    """
    Estratégia que utiliza o K-Means com um número fixo de clusters (k estático).
    É mais rápida pois não calcula métricas de validação de clusters.
    """
    def __init__(self, n_clusters=10, seed=42):
        self.n_clusters = n_clusters
        self.seed = seed
        
    def clusterizar(self, Wswp_cls):
        n_samples = len(Wswp_cls)
        
        # Se houver menos células do que o k solicitado, 
        # retorna as próprias células como centróides (Opção A escolhida por padrão).
        if n_samples <= self.n_clusters:
            print(f"[EstrategiaKMeansFixo] Células insuficientes ({n_samples}) para {self.n_clusters} clusters. Retornando todas como centróides.")
            return Wswp_cls
            
        km = KMeans(n_clusters=self.n_clusters, n_init=5, random_state=self.seed)
        km.fit(Wswp_cls)
        
        return km.cluster_centers_


class EstrategiaHDBSCAN(EstrategiaClusterizacao):
    """
    Estratégia que utiliza HDBSCAN para descobrir agrupamentos densos sem necessitar
    de um k fixo e ignorando ruídos biológicos/outliers.
    """
    def __init__(self, min_cluster_size=15):
        self.min_cluster_size = min_cluster_size
        
    def clusterizar(self, Wswp_cls):
        try:
            import hdbscan
        except ImportError:
            raise ImportError("O pacote 'hdbscan' não está instalado. Rode 'pip install hdbscan' no seu terminal.")
            
        clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size)
        labels = clusterer.fit_predict(Wswp_cls)
        
        unique_labels = set(labels)
        centroides = []
        
        for c_id in unique_labels:
            if c_id == -1: # Ignora o ruído
                continue
                
            cluster_points = Wswp_cls[labels == c_id]
            centroid = cluster_points.mean(axis=0)
            centroides.append(centroid)
            
        if not centroides:
            # Fallback: Se não encontrar clusters (tudo foi ruído ou pequeno demais), retorna 1 centróide global
            print("[EstrategiaHDBSCAN] Nenhum cluster denso encontrado. Usando a média global.")
            return np.array([Wswp_cls.mean(axis=0)])
            
        centroides_arr = np.vstack(centroides)
        print(f"[EstrategiaHDBSCAN] Encontrados {len(centroides)} clusters densos válidos.")
        return centroides_arr
