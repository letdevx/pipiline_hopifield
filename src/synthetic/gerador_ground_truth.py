"""Gerador de Synthetic Ground Truth para Auditoria de scRNA-Seq e Redes Hopfield.

Gera micro-datasets humano-verificáveis (padrão 12 células × 8 genes com 3 tipos celulares)
e matrizes de alta escala (até 36.591 genes) com injeção controlada de dropouts estocásticos,
genes ausentes e ruído sequencial, viabilizando provas reais sem suposições ad-hoc.
"""

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp


class GeradorGroundTruthSintetico:
    """Gerador parametrizável de matrizes de controle para benchmarking de bioinformática e IA.

    Atributos
    ---------
    n_celulas : int
        Número total de células na matriz gerada (padrão: 12).
    n_genes : int
        Número total de genes (padrão: 8).
    n_classes : int
        Número de tipos celulares distintos (padrão: 3).
    seed : int
        Semente para reprodutibilidade aleatória.
    gene_names : list[str]
        Lista com os rótulos de cada gene (ex: ['G0', 'G1', ..., 'G7']).
    cell_names : list[str]
        Lista com os rótulos de cada célula (ex: ['C0_TipoA', ...]).
    labels : np.ndarray
        Array com o tipo celular numérico ou string correspondente a cada célula.
    matriz_pura : np.ndarray
        Matriz (n_celulas × n_genes) contendo o Ground Truth exato sem ruído.
    """

    def __init__(self, n_celulas=12, n_genes=8, n_classes=3, seed=42):
        self.n_celulas = n_celulas
        self.n_genes = n_genes
        self.n_classes = min(n_classes, n_celulas)
        self.seed = seed
        
        self.rng = np.random.default_rng(seed)
        self.gene_names = [f"G{j}" for j in range(n_genes)]
        self.cell_names = []
        self.labels = []
        self.matriz_pura = None
        
        self._construir_ground_truth()

    def _construir_ground_truth(self):
        """Constrói internamente a matriz pura em blocos bem definidos por classe biológica."""
        matriz = np.zeros((self.n_celulas, self.n_genes), dtype=np.float32)
        
        # Alocação balanceada de células para as classes
        celulas_por_classe = int(np.ceil(self.n_celulas / self.n_classes))
        genes_por_classe = int(np.ceil(self.n_genes / self.n_classes))
        
        nomes_classes = ["TipoA", "TipoB", "TipoC", "TipoD", "TipoE", "TipoF", "TipoG"]
        
        for idx in range(self.n_celulas):
            cls_idx = min(idx // celulas_por_classe, self.n_classes - 1)
            nome_cls = nomes_classes[cls_idx] if cls_idx < len(nomes_classes) else f"Tipo{cls_idx}"
            
            self.cell_names.append(f"C{idx}_{nome_cls}")
            self.labels.append(cls_idx + 1)  # Classes 1, 2, 3... (compatível com clo do pipeline)
            
            # Ativa um bloco específico de genes para cada classe celular (Assinatura Transcricional)
            g_start = cls_idx * genes_por_classe
            g_end = min(g_start + genes_por_classe, self.n_genes)
            if g_start < self.n_genes:
                matriz[idx, g_start:g_end] = 1.0
            else:
                # Se houver mais classes que blocos de genes, ativa genes de forma cíclica
                matriz[idx, cls_idx % self.n_genes] = 1.0

        self.labels = np.array(self.labels, dtype=int)
        self.matriz_pura = matriz

    def gerar_matriz_pura(self, formato="numpy", contagem_continua=False):
        """Retorna o Ground Truth real em estado perfeito sem dropouts.
        
        Parâmetros
        ----------
        formato : {'numpy', 'dataframe', 'anndata', 'esparso'}
        contagem_continua : bool
            Se True, simula contagens de leitura scRNA-Seq ao invés de binário {0, 1}
            multiplicando os ativos por valores entre 5.0 e 25.0.
        """
        data = self.matriz_pura.copy()
        if contagem_continua:
            mask = data > 0
            # Simula contagens variáveis preservando o sinal > 0
            contagens = self.rng.uniform(5.0, 25.0, size=data.shape).astype(np.float32)
            data[mask] = contagens[mask]
            
        return self._formatar_saida(data, formato)

    def gerar_matriz_perturbada(self, taxa_dropout=0.15, genes_remover=None, 
                                dropouts_deterministicos=None, formato="numpy"):
        """Gera uma matriz com perturbações bio-realistas (dropouts e genes ausentes).
        
        Parâmetros
        ----------
        taxa_dropout : float
            Probabilidade estocástica de um gene ativo (1.0) sofrer dropout e ir para 0.0.
        genes_remover : list[str] ou None
            Lista de nomes de genes para remover completamente (ex: simular genes ausentes em Mathys).
        dropouts_deterministicos : list[tuple[int, int]] ou None
            Lista explícita de coordenadas (idx_celula, idx_gene) para forçar dropout (para testes unitários).
        formato : {'numpy', 'dataframe', 'anndata'}
        """
        data = self.matriz_pura.copy()
        genes_atuais = list(self.gene_names)
        
        if dropouts_deterministicos is not None:
            for c_idx, g_idx in dropouts_deterministicos:
                if 0 <= c_idx < self.n_celulas and 0 <= g_idx < len(genes_atuais):
                    data[c_idx, g_idx] = 0.0
        elif taxa_dropout > 0:
            mask_ativos = (data > 0)
            sorteio = self.rng.random(size=data.shape)
            drop_mask = mask_ativos & (sorteio < taxa_dropout)
            data[drop_mask] = 0.0
            
        if genes_remover is not None:
            colunas_manter_idx = [j for j, g in enumerate(genes_atuais) if g not in genes_remover]
            data = data[:, colunas_manter_idx]
            genes_atuais = [genes_atuais[j] for j in colunas_manter_idx]
            
        return self._formatar_saida(data, formato, genes_customizados=genes_atuais)

    def _formatar_saida(self, data, formato, genes_customizados=None):
        cols = genes_customizados if genes_customizados is not None else self.gene_names
        if formato == "numpy":
            return data
        elif formato == "dataframe":
            return pd.DataFrame(data, index=self.cell_names, columns=cols)
        elif formato == "anndata":
            adata = ad.AnnData(X=data.copy(), 
                               obs=pd.DataFrame({"clo": self.labels, "cell_name": self.cell_names}, index=self.cell_names),
                               var=pd.DataFrame({"gene_name": cols}, index=cols))
            return adata
        elif formato == "esparso":
            return sp.csr_matrix(data)
        else:
            raise ValueError(f"[GeradorGroundTruth] Formato desconhecido: {formato}")

    def gerar_tabela_markdown(self, matriz, genes_customizados=None, titulo=None):
        """Retorna uma string em formato Markdown com a tabela legível para inspeção humana."""
        cols = genes_customizados if genes_customizados is not None else self.gene_names
        
        if isinstance(matriz, ad.AnnData):
            data_arr = matriz.X.toarray() if sp.issparse(matriz.X) else matriz.X
            cols = matriz.var_names.tolist()
            linhas_nomes = matriz.obs_names.tolist()
        elif isinstance(matriz, pd.DataFrame):
            data_arr = matriz.values
            cols = matriz.columns.tolist()
            linhas_nomes = matriz.index.tolist()
        else:
            data_arr = np.asarray(matriz)
            linhas_nomes = self.cell_names[:data_arr.shape[0]]

        lines = []
        if titulo:
            lines.append(f"### {titulo}\n")
        
        header = "| Célula / Rótulo | " + " | ".join(cols) + " |"
        divisor = "| :--- | " + " | ".join([":---:" for _ in cols]) + " |"
        lines.append(header)
        lines.append(divisor)
        
        for i, row in enumerate(data_arr):
            nome_linha = linhas_nomes[i] if i < len(linhas_nomes) else f"C{i}"
            vals_str = []
            for val in row:
                if val == int(val):
                    vals_str.append(f"{int(val)}")
                else:
                    vals_str.append(f"{val:.1f}")
            line = f"| **{nome_linha}** | " + " | ".join(vals_str) + " |"
            lines.append(line)
            
        return "\n".join(lines) + "\n"

    def __repr__(self):
        return (f"GeradorGroundTruthSintetico(celulas={self.n_celulas}, "
                f"genes={self.n_genes}, classes={self.n_classes}, seed={self.seed})")
