import os
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp


class Binarizador:
    """Responsável pelas etapas de pré-processamento da matriz de expressão.

    Atributos
    ---------
    path_h5ad       : caminho para o arquivo .h5ad de entrada
    out_dir         : diretório onde os arquivos gerados serão salvos
    chunk           : número de células processadas por vez (padrão 3000)
    path_binarizada : preenchido automaticamente após chamar .binarizar()
    """

    def __init__(self, path_h5ad, out_dir, chunk=3000):
        self.path_h5ad = path_h5ad
        self.out_dir = out_dir
        self.chunk = chunk
        self.path_binarizada = None  # definido após binarizar()

    # ------------------------------------------------------------------
    # Métodos públicos (etapas do pipeline)
    # ------------------------------------------------------------------

    def binarizar(self, nome_arquivo="matrizM_binarizada.txt"):
        """Binariza a matriz de expressão e salva no diretório de saída.

        Valores > 0 viram 1.0, zeros permanecem 0.0.
        Retorna o próprio objeto para permitir encadeamento de chamadas.
        """
        self.path_binarizada = os.path.join(self.out_dir, nome_arquivo)

        adata = ad.read_h5ad(self.path_h5ad, backed='r')
        n_celulas, n_genes = adata.shape
        gene_names = list(adata.var_names)
        print(f"[Preprocessador] Shape da matriz: {adata.shape}")

        os.makedirs(self.out_dir, exist_ok=True)

        with open(self.path_binarizada, 'w', buffering=64 * 1024 * 1024) as fout:
            fout.write(','.join(gene_names) + '\n')

            for start in range(0, n_celulas, self.chunk):
                end = min(start + self.chunk, n_celulas)
                X = adata.X[start:end]
                if sp.issparse(X):
                    X = X.toarray()

                bin_chunk = (X > 0).astype(np.float32)
                pd.DataFrame(bin_chunk).to_csv(
                    fout, header=False, index=False, float_format='%.1f'
                )

                if start == 0:
                    print(f"[Preprocessador] Primeiro chunk escrito ({end} células). Continuando...")

        adata.file.close()
        print(f"[Preprocessador] Arquivo salvo: {self.path_binarizada}")
        return self  # permite encadear: prep.binarizar().proxima_etapa()

    def carregar_binarizada(self):
        """Carrega a matriz binarizada já gerada como DataFrame do pandas."""
        if self.path_binarizada is None:
            raise RuntimeError("Execute .binarizar() antes de carregar.")
        print(f"[Preprocessador] Carregando: {self.path_binarizada}")
        return pd.read_csv(self.path_binarizada)

    # ------------------------------------------------------------------
    # Representação do objeto (útil para debug no notebook)
    # ------------------------------------------------------------------

    def __repr__(self):
        binarizada = self.path_binarizada or "ainda não gerada"
        return (
            f"Preprocessador(\n"
            f"  path_h5ad       = {self.path_h5ad}\n"
            f"  out_dir         = {self.out_dir}\n"
            f"  chunk           = {self.chunk}\n"
            f"  path_binarizada = {binarizada}\n"
            f")"
        )
