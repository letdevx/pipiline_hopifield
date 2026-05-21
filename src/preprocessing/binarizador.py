import os
import numpy as np
import anndata as ad
import scipy.sparse as sp


class Binarizador:
    """Responsável pelas etapas de pré-processamento da matriz de expressão.

    Atributos
    ---------
    path_h5ad          : caminho para o arquivo .h5ad de entrada
    out_dir            : diretório geral de saída do pipeline
    out_dir_binarizada : pasta específica onde o .h5ad binarizado será salvo
                         (padrão: mesmo que out_dir)
    path_binarizada    : preenchido automaticamente após chamar .binarizar()
    """

    def __init__(self, path_h5ad, out_dir=None, out_dir_binarizada=None):
        self.path_h5ad = path_h5ad
        self.out_dir = out_dir or os.path.join(os.getcwd(), "outputs")
        self.out_dir_binarizada = out_dir_binarizada or self.out_dir
        self.path_binarizada = None

    def binarizar(self, nome_arquivo="matrizBinarizadaM.h5ad"):
        """Binariza a matriz de expressão e salva como .h5ad no diretório de saída.

        Valores > 0 viram 1, zeros permanecem 0 (dtype int8).
        Retorna o próprio objeto para permitir encadeamento de chamadas.
        """
        nome_entrada = os.path.splitext(os.path.basename(self.path_h5ad))[0]
        pasta_saida = os.path.join(self.out_dir_binarizada, nome_entrada)
        self.path_binarizada = os.path.join(pasta_saida, nome_arquivo)

        if os.path.exists(self.path_binarizada):
            print(f"[Binarizador] Arquivo já existe, pulando: {self.path_binarizada}")
            return self

        print("[Binarizador] Carregando arquivo h5ad...")
        adata = ad.read_h5ad(self.path_h5ad)
        print(f"[Binarizador] Shape da matriz: {adata.shape}")

        print("[Binarizador] Binarizando a matriz...")
        if sp.issparse(adata.X):
            adata.X.data = np.where(adata.X.data > 0, 1, 0)
            adata.X = adata.X.astype(np.int8)
        else:
            adata.X = np.where(adata.X > 0, 1, 0).astype(np.int8)

        os.makedirs(pasta_saida, exist_ok=True)

        print("[Binarizador] Salvando arquivo h5ad binarizado...")
        adata.write_h5ad(self.path_binarizada)
        print(f"[Binarizador] Arquivo salvo: {self.path_binarizada}")
        return self

    def carregar_binarizada(self):
        """Carrega a matriz binarizada já gerada como AnnData."""
        if self.path_binarizada is None:
            raise RuntimeError("Execute .binarizar() antes de carregar.")
        print(f"[Binarizador] Carregando: {self.path_binarizada}")
        return ad.read_h5ad(self.path_binarizada)

    def __repr__(self):
        binarizada = self.path_binarizada or "ainda não gerada"
        return (
            f"Binarizador(\n"
            f"  path_h5ad          = {self.path_h5ad}\n"
            f"  out_dir            = {self.out_dir}\n"
            f"  out_dir_binarizada = {self.out_dir_binarizada}\n"
            f"  path_binarizada    = {binarizada}\n"
            f")"
        )
