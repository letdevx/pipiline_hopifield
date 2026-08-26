import gc
import os

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as sp


class Alinhador:
    """Alinha dois h5ad binarizados ao mesmo espaço gênico de referência (Fujita)."""

    def __init__(self, path_binarizada_m, path_binarizada_f, out_dir,
                 map_f, map_m, gene_alvo_idx, genes_ordenados):
        self.path_binarizada_m = path_binarizada_m
        self.path_binarizada_f = path_binarizada_f
        self.out_dir = out_dir
        self.map_f = map_f
        self.map_m = map_m
        self.gene_alvo_idx = gene_alvo_idx
        self.genes_ordenados = genes_ordenados
        self.path_f_alinhado = None
        self.path_m_alinhado = None

    def alinhar(self):
        nome_f = "adataF_binarizado_alinhado"
        pasta_f = os.path.join(self.out_dir, nome_f)
        self.path_f_alinhado = os.path.join(pasta_f, f"{nome_f}.h5ad")

        nome_m = "adataM_binarizado_alinhado"
        pasta_m = os.path.join(self.out_dir, nome_m)
        self.path_m_alinhado = os.path.join(pasta_m, f"{nome_m}.h5ad")

        if os.path.exists(self.path_f_alinhado):
            print(f"[Alinhador] Fujita já alinhado, pulando: {self.path_f_alinhado}")
        else:
            print("[Alinhador] Carregando Fujita binarizado...")
            adataf = ad.read_h5ad(self.path_binarizada_f)
            print(f"  shape original: {adataf.shape}")
            print("[Alinhador] Alinhando Fujita (fill=0.0)...")
            adataf_alinhado = self._alinhar_direto(adataf, self.map_f, fill_value=0.0)
            del adataf
            gc.collect()
            os.makedirs(pasta_f, exist_ok=True)
            adataf_alinhado.write_h5ad(self.path_f_alinhado)
            print(f"  shape final: {adataf_alinhado.shape}")
            del adataf_alinhado
            gc.collect()
            print(f"  salvo em {self.path_f_alinhado}  [OK]\n")

        if os.path.exists(self.path_m_alinhado):
            print(f"[Alinhador] Mathys já alinhado, pulando: {self.path_m_alinhado}")
        else:
            print("[Alinhador] Carregando Mathys binarizado...")
            adatam = ad.read_h5ad(self.path_binarizada_m)
            print(f"  shape original: {adatam.shape}")
            print("[Alinhador] Alinhando Mathys (genes ausentes → 0.5)...")
            adatam_alinhado = self._alinhar_direto(adatam, self.map_m, fill_value=0.5)
            del adatam
            gc.collect()
            os.makedirs(pasta_m, exist_ok=True)
            adatam_alinhado.write_h5ad(self.path_m_alinhado)
            print(f"  shape final: {adatam_alinhado.shape}")
            del adatam_alinhado
            gc.collect()
            print(f"  salvo em {self.path_m_alinhado}  [OK]")

        print("\n[Alinhador] Concluído.")
        return self

    def _alinhar_direto(self, adata, ensembl_map, fill_value=0.0):
        n_celulas = adata.n_obs
        n_genes   = len(self.genes_ordenados)

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

        # Matriz de projeção P (old_vars -> new_genes)
        P_data = np.ones(len(old_idx), dtype=np.float32)
        P = sp.csr_matrix((P_data, (old_idx, new_idx)), shape=(adata.n_vars, n_genes))

        print("  Multiplicando matrizes (projeção)...")
        # Multiplicação é extremamente rápida e consome pouca memória
        if sp.issparse(adata.X):
            X_novo = adata.X.dot(P).astype(np.float32)
        else:
            X_novo = sp.csr_matrix(adata.X).dot(P).astype(np.float32)
        if sp.issparse(X_novo):
            X_novo = X_novo.tocsr()
        else:
            X_novo = sp.csr_matrix(X_novo)
        del P
        gc.collect()


        if fill_value != 0.0:
            missing_cols = np.array(
                sorted(set(range(n_genes)) - present_new_cols), dtype=np.int32
            )
            if len(missing_cols) > 0:
                print(f"  Preenchendo {len(missing_cols)} colunas ausentes com {fill_value}...")
                print("  Mesclando colunas ausentes (otimizado para baixa memória)...")
                n_miss = len(missing_cols)
                n_novo = X_novo.nnz
                n_total = n_novo + n_celulas * n_miss
                
                final_indices = np.empty(n_total, dtype=np.int32)
                final_data = np.empty(n_total, dtype=np.float32)
                final_indptr = np.empty(n_celulas + 1, dtype=np.int64)
                final_indptr[0] = 0
                
                novo_indptr = X_novo.indptr
                novo_indices = X_novo.indices
                novo_data = X_novo.data
                
                idx_ptr = 0
                fill_dat_arr = np.full(n_miss, fill_value, dtype=np.float32)
                
                for i in range(n_celulas):
                    start = novo_indptr[i]
                    end = novo_indptr[i+1]
                    ex_idx = novo_indices[start:end]
                    ex_dat = novo_data[start:end]
                    
                    if len(ex_idx) == 0:
                        row_idx = missing_cols
                        row_dat = fill_dat_arr
                    else:
                        row_idx = np.concatenate([ex_idx, missing_cols])
                        row_dat = np.concatenate([ex_dat, fill_dat_arr])
                        sort_mask = np.argsort(row_idx)
                        row_idx = row_idx[sort_mask]
                        row_dat = row_dat[sort_mask]
                        
                    n_elem = len(row_idx)
                    end_ptr = idx_ptr + n_elem
                    
                    final_indices[idx_ptr:end_ptr] = row_idx
                    final_data[idx_ptr:end_ptr] = row_dat
                    
                    idx_ptr = end_ptr
                    final_indptr[i+1] = idx_ptr

                X_novo = sp.csr_matrix((final_data, final_indices, final_indptr), shape=(n_celulas, n_genes))
                del final_data, final_indices, final_indptr
                gc.collect()
                print(f"  Preenchimento concluído.")

        # AnnData exige pd.DataFrame para .var
        var_novo = pd.DataFrame(index=pd.Index(self.genes_ordenados, name='ensembl_id'))
        return ad.AnnData(X=X_novo, obs=adata.obs.copy(), var=var_novo)

    def salvar_como_txt(self, chunk=500):
        """Salva os arquivos alinhados em formato TXT (CSV) dentro de suas respectivas pastas.

        Usa Polars (escrita Rust) para velocidade máxima. Processa em chunks de memória.
        Requer que .alinhar() tenha sido executado antes.
        Usa escrita atômica (arquivo .tmp → renomeação) para evitar arquivos incompletos.
        """
        if self.path_f_alinhado is None or self.path_m_alinhado is None:
            raise RuntimeError("Execute .alinhar() antes de salvar como TXT.")

        for path_h5ad in (self.path_f_alinhado, self.path_m_alinhado):
            path_txt = os.path.splitext(path_h5ad)[0] + ".txt"
            path_tmp = path_txt + ".tmp"

            if os.path.exists(path_txt):
                print(f"[Alinhador] TXT já existe, pulando: {path_txt}")
                continue

            if os.path.exists(path_tmp):
                os.remove(path_tmp)

            print(f"[Alinhador] Salvando TXT: {path_txt}")
            adata = ad.read_h5ad(path_h5ad, backed='r')
            n_celulas = adata.n_obs
            gene_names = adata.var_names.tolist()
            total = 0

            with open(path_tmp, 'w', buffering=128 * 1024 * 1024) as fout:
                fout.write(','.join(gene_names) + '\n')
                for start in range(0, n_celulas, chunk):
                    end = min(start + chunk, n_celulas)
                    X = adata.X[start:end]
                    if sp.issparse(X):
                        X = X.toarray()
                    fout.write(pl.from_numpy(np.asfortranarray(X.astype(np.float32))).write_csv(include_header=False))
                    total += end - start
                    if total % (chunk * 5) == 0:
                        print(f"  {total} células processadas...")

            adata.file.close()
            os.rename(path_tmp, path_txt)
            print(f"  Salvo: {path_txt}  ({total} células x {len(gene_names)} genes)  [Ok]")
        return self

    def gerar_tracking(self, ids_so_f, map_f):
        if self.path_m_alinhado is None:
            raise RuntimeError("Execute .alinhar() antes de gerar o tracking.")
        out_tracking = os.path.join(self.out_dir, "tracking_genes_adicionados_mathys.csv")
        if os.path.exists(out_tracking):
            print(f"[Alinhador] Tracking já existe, pulando: {out_tracking}")
            return pl.read_csv(out_tracking)
        inv_map_f = {v: k for k, v in map_f.items()}
        tracking_rows = []
        for eid in sorted(ids_so_f):
            if eid in self.gene_alvo_idx:
                tracking_rows.append({
                    'gene_name'      : inv_map_f.get(eid, eid),
                    'ensembl_id'     : eid,
                    'posicao_coluna' : self.gene_alvo_idx[eid],
                    'valor_inserido' : 0.5,
                    'presente_fujita': True,
                    'presente_mathys': False,
                })
        df_tracking = pl.DataFrame(tracking_rows).sort('posicao_coluna')
        df_tracking.write_csv(out_tracking)
        print(f"[Alinhador] Tracking salvo em: {out_tracking} ({len(df_tracking)} genes)")
        return df_tracking

    def __repr__(self):
        return (
            f"Alinhador(\n"
            f"  path_binarizada_m = {self.path_binarizada_m}\n"
            f"  path_binarizada_f = {self.path_binarizada_f}\n"
            f"  out_dir           = {self.out_dir}\n"
            f"  path_f_alinhado   = {self.path_f_alinhado or 'ainda não gerado'}\n"
            f"  path_m_alinhado   = {self.path_m_alinhado or 'ainda não gerado'}\n"
            f")"
        )
