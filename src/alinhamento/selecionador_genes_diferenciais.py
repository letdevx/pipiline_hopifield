import os
import numpy as np
import pandas as pd
import polars as pl
import anndata as ad
import scipy.sparse as sp
from sklearn.feature_selection import chi2


class SelecionadorGenesDiferenciais:
    """Calcula os N genes com maior valor de Chi-Square (qui-quadrado) em relação aos rótulos de tipo celular.

    Filtra a expressão de expressão gênica binarizada medindo o Ganho de Informação Biológica
    entre a presença/ausência do gene (0 ou 1) e o tipo celular (clo), eliminando o ruído
    de genes constitutivos (housekeeping genes) que são ativos em todas as células.
    """

    def __init__(self, path_txt_or_h5ad, path_labels, n=5000):
        self.path_input   = path_txt_or_h5ad
        self.path_labels  = path_labels
        self.n            = n
        self.df_resultado = None

    def calcular(self, out_csv=None):
        if out_csv and os.path.exists(out_csv):
            print(f"[SelecionadorGenesDiferenciais] Arquivo já existe, pulando: {out_csv}")
            self.df_resultado = pl.read_csv(out_csv)
            return self

        print(f"[SelecionadorGenesDiferenciais] Lendo matriz e rótulos para o cálculo de Chi2...")
        
        # Carrega rótulos das células
        if isinstance(self.path_labels, (np.ndarray, list)):
            labels = np.asarray(self.path_labels, dtype=int).ravel()
        else:
            labels = np.loadtxt(self.path_labels, dtype=int).ravel()

        # Remapeia classes seguindo a regra clo (classes raras -> 2)
        clo = labels.copy()
        clo[~np.isin(clo, [1, 3, 4, 5, 6, 7, 0])] = 2
        mask_valid = (clo != 0)

        # Leitura da matriz
        if str(self.path_input).endswith('.h5ad'):
            adata = ad.read_h5ad(self.path_input)
            gene_names = adata.var_names.tolist()
            X = adata.X[mask_valid]
            if sp.issparse(X):
                X = X.toarray()
        else:
            # Leitura de CSV/TXT binarizado
            df = pd.read_csv(self.path_input, dtype=np.float32)
            gene_names = df.columns.tolist()
            X = df.values[mask_valid]
            del df

        y = clo[mask_valid]

        print(f"  Executando Chi-Square em {X.shape[1]} genes para {X.shape[0]} células...")
        scores, pvalues = chi2(X, y)
        scores = np.nan_to_num(scores, nan=0.0)

        n_real = min(self.n, len(gene_names))
        idx_top = np.argsort(scores)[-n_real:][::-1]

        top_genes = [gene_names[i] for i in idx_top]
        top_scores = [float(scores[i]) for i in idx_top]

        self.df_resultado = pl.DataFrame({
            'gene': top_genes,
            'chi2_score': top_scores
        })

        print(f"[SelecionadorGenesDiferenciais] Concluído. Top {n_real} genes discriminativos selecionados por Chi2.")
        print(f"  Maior escore Chi2: {top_scores[0]:.2f} (Gene: {top_genes[0]})")
        print(f"  Menor escore Chi2 do Top: {top_scores[-1]:.2f} (Gene: {top_genes[-1]})")

        if out_csv:
            self.salvar(out_csv)

        return self

    def salvar(self, out_csv):
        if self.df_resultado is None:
            raise RuntimeError("Execute .calcular() antes de salvar.")
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        self.df_resultado.write_csv(out_csv)
        print(f"[SelecionadorGenesDiferenciais] Salvo em: {out_csv}")
        return self

    def filtrar_matriz(self, in_csv_or_npy, out_csv_or_npy):
        """Salva nova matriz contendo apenas as colunas dos genes selecionados por Chi2."""
        if self.df_resultado is None:
            raise RuntimeError("Execute .calcular() antes de filtrar.")

        lista_genes = self.df_resultado["gene"].to_list()
        os.makedirs(os.path.dirname(out_csv_or_npy), exist_ok=True)

        if str(in_csv_or_npy).endswith('.npy'):
            # Se for array npy, assume que as colunas já estão ordenadas ou lê via dataframe
            arr = np.load(in_csv_or_npy)
            np.save(out_csv_or_npy, arr)
            print(f"[SelecionadorGenesDiferenciais] Matriz salva: {out_csv_or_npy}")
        else:
            with open(in_csv_or_npy, encoding='utf-8') as fh:
                header = fh.readline().strip('\n').strip('\r').split(',')

            coluna_celulas = header[0]
            colunas_validas = [coluna_celulas] + [c for c in lista_genes if c in header]

            if str(out_csv_or_npy).endswith('.npy'):
                df_filtered = pl.scan_csv(in_csv_or_npy).select(colunas_validas).collect()
                if df_filtered.columns[0] == coluna_celulas and not df_filtered.dtypes[0].is_numeric():
                    arr = df_filtered.select(colunas_validas[1:]).to_numpy().astype(np.float32)
                else:
                    arr = df_filtered.to_numpy().astype(np.float32)
                np.save(out_csv_or_npy, arr)
                print(f"[SelecionadorGenesDiferenciais] Matriz filtrada salva em binário (lazy): {out_csv_or_npy} ({arr.shape})")
            else:
                pl.scan_csv(in_csv_or_npy).select(colunas_validas).sink_csv(out_csv_or_npy)
                print(f"[SelecionadorGenesDiferenciais] Matriz filtrada salva em: {out_csv_or_npy}")
        return self

    def __repr__(self):
        n = len(self.df_resultado) if self.df_resultado is not None else 'não calculado'
        return f"SelecionadorGenesDiferenciais(input={self.path_input}, n={self.n}, resultado={n} genes)"
