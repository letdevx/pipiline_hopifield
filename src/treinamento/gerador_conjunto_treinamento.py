import os

import polars as pl


class GeradorConjuntoTreinamento:
    """Filtra arquivos TXT alinhados para manter somente os genes do conjunto de treinamento.

    Usa Polars streaming (sink_csv) para suportar arquivos massivos sem carregá-los na memória.
    Mantém a ordem original das colunas do arquivo alinhado.
    """

    def __init__(self, path_top_genes_csv, out_dir, chunk=3000):
        self.path_top_genes_csv = path_top_genes_csv
        self.out_dir            = out_dir
        self.chunk              = chunk  # mantido por compatibilidade
        self.genes_selecionados = None
        self._carregar_genes()

    def _carregar_genes(self):
        df = pl.read_csv(self.path_top_genes_csv)
        self.genes_selecionados = set(df['gene'].to_list())
        print(f"[GeradorConjuntoTreinamento] {len(self.genes_selecionados)} genes carregados de: {self.path_top_genes_csv}")

    def gerar(self, path_txt):
        with open(path_txt, encoding='utf-8') as f:
            todos_genes = f.readline().strip().split(',')

        genes_filtrados = [g for g in todos_genes if g in self.genes_selecionados]
        n = len(genes_filtrados)
        nome       = os.path.splitext(os.path.basename(path_txt))[0]
        path_saida = os.path.join(self.out_dir, f"{nome}_top{n}.txt")
        os.makedirs(self.out_dir, exist_ok=True)

        if os.path.exists(path_saida):
            print(f"[GeradorConjuntoTreinamento] Arquivo já existe, pulando: {path_saida}")
            self.path_saida = path_saida
            return self

        path_tmp = path_saida + ".tmp"
        if os.path.exists(path_tmp):
            os.remove(path_tmp)

        print(f"\n[GeradorConjuntoTreinamento] Processando: {path_txt}")
        print(f"  Genes encontrados no arquivo: {n} de {len(self.genes_selecionados)} selecionados")
        print(f"  Escrevendo via Polars streaming...")

        (
            pl.scan_csv(path_txt, infer_schema_length=1)
            .select(genes_filtrados)
            .sink_csv(path_tmp)
        )

        os.rename(path_tmp, path_saida)
        print(f"[GeradorConjuntoTreinamento] Salvo: {path_saida}  ({n} genes)")
        self.path_saida = path_saida
        return self

    def gerar_de_h5ad(self, path_h5ad, is_mathys=False, fill_value=0.5, exportar_npy=True, exportar_h5ad=True):
        """Filtra AnnData .h5ad para os top genes e salva diretamente em .npy e .h5ad de forma OOM-Safe."""
        import gc
        import anndata as ad
        import numpy as np
        import pandas as pd
        import scipy.sparse as sp

        adata = ad.read_h5ad(path_h5ad)
        todos_genes = adata.var_names.tolist()
        genes_filtrados = [g for g in todos_genes if g in self.genes_selecionados]
        n = len(genes_filtrados)
        nome = os.path.splitext(os.path.basename(path_h5ad))[0]

        path_npy = os.path.join(self.out_dir, f"{nome}_top{n}.npy")
        path_saida_h5ad = os.path.join(self.out_dir, f"{nome}_top{n}.h5ad")
        os.makedirs(self.out_dir, exist_ok=True)

        if os.path.exists(path_npy) and (not exportar_h5ad or os.path.exists(path_saida_h5ad)):
            print(f"[GeradorConjuntoTreinamento] Arquivos já existem, pulando: {nome}_top{n}")
            self.path_saida = path_npy
            return self

        print(f"\n[GeradorConjuntoTreinamento] Processando .h5ad: {path_h5ad}")
        print(f"  Genes selecionados encontrados: {n} de {len(self.genes_selecionados)}")

        gene_to_idx = {g: i for i, g in enumerate(todos_genes)}
        col_indices = [gene_to_idx[g] for g in genes_filtrados]

        X_sub = adata.X[:, col_indices]
        if sp.issparse(X_sub):
            X_dense = X_sub.toarray().astype(np.float32)
        else:
            X_dense = np.asarray(X_sub, dtype=np.float32)

        # Injeção de sentinela se for Mathys e houver anotação de presença
        if is_mathys and 'presente_no_dataset' in adata.var:
            presente_sub = adata.var['presente_no_dataset'].iloc[col_indices].to_numpy()
            ausentes_mask = ~presente_sub
            n_ausentes = np.sum(ausentes_mask)
            if n_ausentes > 0:
                print(f"  Injetando sentinela {fill_value} em {n_ausentes} genes ausentes no Mathys...")
                X_dense[:, ausentes_mask] = fill_value

        if exportar_npy:
            np.save(path_npy, X_dense)
            print(f"[GeradorConjuntoTreinamento] Salvo .npy: {path_npy} ({X_dense.shape})")

        if exportar_h5ad:
            var_df = pd.DataFrame(index=pd.Index(genes_filtrados, name='ensembl_id'))
            adata_sub = ad.AnnData(X=sp.csr_matrix(X_dense), obs=adata.obs.copy(), var=var_df)
            adata_sub.write_h5ad(path_saida_h5ad, compression="gzip")
            print(f"[GeradorConjuntoTreinamento] Salvo .h5ad: {path_saida_h5ad}")

        del adata, X_sub, X_dense
        gc.collect()

        self.path_saida = path_npy
        return self


    def __repr__(self):
        n = len(self.genes_selecionados) if self.genes_selecionados else 'não carregado'
        return (
            f"GeradorConjuntoTreinamento(\n"
            f"  path_top_genes_csv = {self.path_top_genes_csv}\n"
            f"  out_dir            = {self.out_dir}\n"
            f"  genes_selecionados = {n}\n"
            f")"
        )
