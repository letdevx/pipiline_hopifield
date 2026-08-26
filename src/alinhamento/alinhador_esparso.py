"""Módulo de alinhamento esparso e OOM-Safe para matrizes scRNA-seq."""

import gc
import os
import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as sp


class AlinhadorEsparso:
    """Alinha dois h5ad binarizados ao mesmo espaço gênico de referência (Fujita) de forma 100% esparsa.
    
    Projetado para ambientes com restrição de memória RAM (<= 16GB). Evita a materialização densa
    de colunas ausentes no espaço de genoma completo (~36k genes), mantendo a representação
    estritamente em matrizes esparsas CSR comprimidas e permitindo a injeção do valor sentinela (0.5)
    sob demanda ou nos subconjuntos filtrados (Top 5k / 11k).
    """

    def __init__(self, path_binarizada_m, path_binarizada_f, out_dir,
                 map_f, map_m, gene_alvo_idx, genes_ordenados):
        self.path_binarizada_m = str(path_binarizada_m)
        self.path_binarizada_f = str(path_binarizada_f)
        self.out_dir = str(out_dir)
        self.map_f = map_f
        self.map_m = map_m
        self.gene_alvo_idx = gene_alvo_idx
        self.genes_ordenados = list(genes_ordenados)
        
        nome_f = "adataF_binarizado_alinhado"
        pasta_f = os.path.join(self.out_dir, nome_f)
        self.path_f_alinhado = os.path.join(pasta_f, f"{nome_f}.h5ad")

        nome_m = "adataM_binarizado_alinhado"
        pasta_m = os.path.join(self.out_dir, nome_m)
        self.path_m_alinhado = os.path.join(pasta_m, f"{nome_m}.h5ad")

    def _projetar_esparso(self, adata, ensembl_map, dataset_name="Dataset"):
        """Projeta matriz AnnData para o espaço canônico de genes_ordenados usando projeção CSR."""
        n_celulas = adata.n_obs
        n_genes_alvo = len(self.genes_ordenados)

        old_idx = []
        new_idx = []
        present_new_cols = set()

        for old_i, gene_name in enumerate(adata.var_names):
            eid = ensembl_map.get(gene_name, gene_name)
            if eid in self.gene_alvo_idx:
                new_col = self.gene_alvo_idx[eid]
                old_idx.append(old_i)
                new_idx.append(new_col)
                present_new_cols.add(new_col)

        print(f"[{dataset_name}] {len(old_idx)} genes mapeados para o espaço canônico ({n_genes_alvo} genes totais).")

        # Matriz de projeção P (n_vars_original -> n_genes_alvo)
        P_data = np.ones(len(old_idx), dtype=np.float32)
        P = sp.csr_matrix((P_data, (old_idx, new_idx)), shape=(adata.n_vars, n_genes_alvo), dtype=np.float32)

        print(f"[{dataset_name}] Projetando matriz esparsa...")
        if sp.issparse(adata.X):
            X_novo = adata.X.dot(P)
        else:
            X_novo = sp.csr_matrix(adata.X).dot(P)

        if not sp.isspmatrix_csr(X_novo):
            X_novo = X_novo.tocsr()

        del P
        gc.collect()

        # Anota metadados de presença por gene
        presente_arr = np.zeros(n_genes_alvo, dtype=bool)
        if len(present_new_cols) > 0:
            presente_arr[list(present_new_cols)] = True

        var_novo = pd.DataFrame(
            {
                'presente_no_dataset': presente_arr,
            },
            index=pd.Index(self.genes_ordenados, name='ensembl_id')
        )

        return ad.AnnData(X=X_novo, obs=adata.obs.copy(), var=var_novo)

    def alinhar(self, forcar=False):
        """Executa o alinhamento esparso para ambos os datasets (Fujita e Mathys)."""
        pasta_f = os.path.dirname(self.path_f_alinhado)
        pasta_m = os.path.dirname(self.path_m_alinhado)

        # 1. Alinhamento Fujita (Referência)
        if os.path.exists(self.path_f_alinhado) and not forcar:
            print(f"[AlinhadorEsparso] Fujita já alinhado, pulando: {self.path_f_alinhado}")
        else:
            print("[AlinhadorEsparso] Carregando Fujita binarizado...")
            adataf = ad.read_h5ad(self.path_binarizada_f)
            print(f"  Shape original Fujita: {adataf.shape}")
            adataf_alinhado = self._projetar_esparso(adataf, self.map_f, dataset_name="Fujita")
            del adataf
            gc.collect()

            os.makedirs(pasta_f, exist_ok=True)
            print(f"  Salvando Fujita alinhado (.h5ad CSR)...")
            adataf_alinhado.write_h5ad(self.path_f_alinhado, compression="gzip")
            print(f"  Salvo em {self.path_f_alinhado} (shape: {adataf_alinhado.shape})  [OK]\n")
            del adataf_alinhado
            gc.collect()

        # 2. Alinhamento Mathys (Alvo)
        if os.path.exists(self.path_m_alinhado) and not forcar:
            print(f"[AlinhadorEsparso] Mathys já alinhado, pulando: {self.path_m_alinhado}")
        else:
            print("[AlinhadorEsparso] Carregando Mathys binarizado...")
            adatam = ad.read_h5ad(self.path_binarizada_m)
            print(f"  Shape original Mathys: {adatam.shape}")
            adatam_alinhado = self._projetar_esparso(adatam, self.map_m, dataset_name="Mathys")
            del adatam
            gc.collect()

            os.makedirs(pasta_m, exist_ok=True)
            print(f"  Salvando Mathys alinhado (.h5ad CSR)...")
            adatam_alinhado.write_h5ad(self.path_m_alinhado, compression="gzip")
            print(f"  Salvo em {self.path_m_alinhado} (shape: {adatam_alinhado.shape})  [OK]\n")
            del adatam_alinhado
            gc.collect()

        print("[AlinhadorEsparso] Alinhamento genômico canônico concluído com sucesso.")
        return self

    def gerar_tracking(self, ids_so_f, map_f):
        """Gera relatório de genes exclusivos do Fujita (ausentes no Mathys)."""
        if self.path_m_alinhado is None:
            raise RuntimeError("Execute .alinhar() antes de gerar o tracking.")
        
        out_tracking = os.path.join(self.out_dir, "tracking_genes_adicionados_mathys.csv")
        if os.path.exists(out_tracking):
            print(f"[AlinhadorEsparso] Tracking já existe, pulando: {out_tracking}")
            return pl.read_csv(out_tracking)

        inv_map_f = {v: k for k, v in map_f.items()}
        tracking_rows = []
        for eid in sorted(ids_so_f):
            if eid in self.gene_alvo_idx:
                tracking_rows.append({
                    'gene_name': inv_map_f.get(eid, eid),
                    'ensembl_id': eid,
                    'posicao_coluna': self.gene_alvo_idx[eid],
                    'valor_inserido': 0.5,
                    'presente_fujita': True,
                    'presente_mathys': False,
                })
        df_tracking = pl.DataFrame(tracking_rows).sort('posicao_coluna')
        df_tracking.write_csv(out_tracking)
        print(f"[AlinhadorEsparso] Tracking salvo em: {out_tracking} ({len(df_tracking)} genes)")
        return df_tracking

    def extrair_subconjunto(self, lista_genes_ou_csv, out_dir=None, exportar_npy=True, exportar_h5ad=True, fill_value_mathys=0.5):
        """Extrai um subconjunto ordenado de genes (ex: Top 5000) e injeta sentinela 0.5 no Mathys de forma OOM-Safe.
        
        Parâmetros
        ----------
        lista_genes_ou_csv : list ou str
            Lista de nomes/IDs de genes ou caminho para CSV contendo coluna 'gene'.
        out_dir : str, opcional
            Diretório de saída. Padrão: self.out_dir.
        exportar_npy : bool
            Se True, salva matrizes binárias float32 .npy prontas para consumo imediato.
        exportar_h5ad : bool
            Se True, salva AnnData .h5ad.
        fill_value_mathys : float
            Valor sentinela a ser atribuído às colunas ausentes no Mathys (padrão 0.5).
        """
        if isinstance(lista_genes_ou_csv, (str, os.PathLike)):
            df_g = pl.read_csv(lista_genes_ou_csv)
            col_nome = 'gene' if 'gene' in df_g.columns else df_g.columns[0]
            genes_desejados = df_g[col_nome].to_list()
        else:
            genes_desejados = list(lista_genes_ou_csv)

        n_genes = len(genes_desejados)
        destino_dir = out_dir if out_dir is not None else self.out_dir
        os.makedirs(destino_dir, exist_ok=True)

        print(f"\n[AlinhadorEsparso] Extraindo subconjunto de {n_genes} genes...")
        
        # Mapeia posições dos genes desejados no espaço canônico
        gene_to_pos = {g: i for i, g in enumerate(self.genes_ordenados)}
        col_indices = []
        genes_encontrados = []
        for g in genes_desejados:
            if g in gene_to_pos:
                col_indices.append(gene_to_pos[g])
                genes_encontrados.append(g)

        col_indices = np.array(col_indices, dtype=np.int32)
        n_encontrados = len(col_indices)
        print(f"  {n_encontrados} de {n_genes} genes localizados no espaço canônico.")

        # --- 1. Extração Fujita ---
        path_f_npy = os.path.join(destino_dir, f"adataF_binarizado_alinhado_top{n_encontrados}.npy")
        path_f_h5ad = os.path.join(destino_dir, f"adataF_binarizado_alinhado_top{n_encontrados}.h5ad")

        print(f"  Processando Fujita (Top {n_encontrados})...")
        adata_f = ad.read_h5ad(self.path_f_alinhado)
        X_f_sub = adata_f.X[:, col_indices]
        if sp.issparse(X_f_sub):
            X_f_dense = X_f_sub.toarray().astype(np.float32)
        else:
            X_f_dense = np.asarray(X_f_sub, dtype=np.float32)

        if exportar_npy:
            np.save(path_f_npy, X_f_dense)
            print(f"  Salvo Fujita .npy: {path_f_npy} ({X_f_dense.shape})")

        if exportar_h5ad:
            var_f = pd.DataFrame(index=pd.Index(genes_encontrados, name='ensembl_id'))
            adata_f_sub = ad.AnnData(X=sp.csr_matrix(X_f_dense), obs=adata_f.obs.copy(), var=var_f)
            adata_f_sub.write_h5ad(path_f_h5ad, compression="gzip")
            print(f"  Salvo Fujita .h5ad: {path_f_h5ad}")

        del adata_f, X_f_sub, X_f_dense
        gc.collect()

        # --- 2. Extração Mathys (com Sentinela 0.5) ---
        path_m_npy = os.path.join(destino_dir, f"adataM_binarizado_alinhado_top{n_encontrados}.npy")
        path_m_h5ad = os.path.join(destino_dir, f"adataM_binarizado_alinhado_top{n_encontrados}.h5ad")

        print(f"  Processando Mathys (Top {n_encontrados} com sentinela={fill_value_mathys})...")
        adata_m = ad.read_h5ad(self.path_m_alinhado)
        presente_mathys_all = adata_m.var['presente_no_dataset'].to_numpy()
        presente_mathys_sub = presente_mathys_all[col_indices]

        X_m_sub = adata_m.X[:, col_indices]
        if sp.issparse(X_m_sub):
            X_m_dense = X_m_sub.toarray().astype(np.float32)
        else:
            X_m_dense = np.asarray(X_m_sub, dtype=np.float32)

        # Injeta sentinela apenas nas colunas ausentes no Mathys
        ausentes_mask = ~presente_mathys_sub
        n_ausentes = np.sum(ausentes_mask)
        if n_ausentes > 0:
            print(f"  Injetando sentinela {fill_value_mathys} em {n_ausentes} colunas ausentes no Mathys...")
            X_m_dense[:, ausentes_mask] = fill_value_mathys

        if exportar_npy:
            np.save(path_m_npy, X_m_dense)
            print(f"  Salvo Mathys .npy: {path_m_npy} ({X_m_dense.shape})")

        if exportar_h5ad:
            var_m = pd.DataFrame(
                {'presente_no_dataset': presente_mathys_sub},
                index=pd.Index(genes_encontrados, name='ensembl_id')
            )
            adata_m_sub = ad.AnnData(X=sp.csr_matrix(X_m_dense), obs=adata_m.obs.copy(), var=var_m)
            adata_m_sub.write_h5ad(path_m_h5ad, compression="gzip")
            print(f"  Salvo Mathys .h5ad: {path_m_h5ad}")

        del adata_m, X_m_sub, X_m_dense
        gc.collect()

        return {
            'path_f_npy': path_f_npy if exportar_npy else None,
            'path_f_h5ad': path_f_h5ad if exportar_h5ad else None,
            'path_m_npy': path_m_npy if exportar_npy else None,
            'path_m_h5ad': path_m_h5ad if exportar_h5ad else None,
        }

    def salvar_como_txt(self, chunk=500):
        """Método de retrocompatibilidade com aviso sobre descontinuação de TXT de 36k genes."""
        print("[AlinhadorEsparso] AVISO: A geração de arquivos .txt com 36k colunas foi descontinuada "
              "para evitar estouro de memória (>20GB). Use extrair_subconjunto() para exportar .npy / .h5ad.")
        return self

    def __repr__(self):
        return (
            f"AlinhadorEsparso(\n"
            f"  path_binarizada_m = {self.path_binarizada_m}\n"
            f"  path_binarizada_f = {self.path_binarizada_f}\n"
            f"  out_dir           = {self.out_dir}\n"
            f"  path_f_alinhado   = {self.path_f_alinhado}\n"
            f"  path_m_alinhado   = {self.path_m_alinhado}\n"
            f")"
        )
