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
            print(f"  salvo em {self.path_f_alinhado}  ✓\n")

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
            print(f"  salvo em {self.path_m_alinhado}  ✓")

        print("\n[Alinhador] Concluído.")
        return self

    def _alinhar_direto(self, adata, ensembl_map, fill_value=0.0):
        n_celulas = adata.n_obs
        n_genes   = len(self.genes_ordenados)

        old_col_map      = np.full(adata.n_vars, -1, dtype=np.intp)
        present_new_cols = set()
        for old_i, gene_name in enumerate(adata.var_names):
            eid = ensembl_map.get(gene_name, gene_name)
            if eid in self.gene_alvo_idx:
                new_col = self.gene_alvo_idx[eid]
                old_col_map[old_i] = new_col
                present_new_cols.add(new_col)

        X_coo        = adata.X.tocoo() if sp.issparse(adata.X) else sp.coo_matrix(adata.X)
        new_cols_arr = old_col_map[X_coo.col]
        mask         = new_cols_arr >= 0

        X_novo = sp.csr_matrix(
            (X_coo.data[mask].astype(np.float32),
             (X_coo.row[mask], new_cols_arr[mask])),
            shape=(n_celulas, n_genes),
            dtype=np.float32,
        )
        del X_coo, new_cols_arr, mask

        if fill_value != 0.0:
            missing_cols = np.array(
                sorted(set(range(n_genes)) - present_new_cols), dtype=np.intp
            )
            if len(missing_cols) > 0:
                print(f"  Preenchendo {len(missing_cols)} colunas ausentes com {fill_value}...")
                n_fill        = len(missing_cols) * n_celulas
                fill_rows     = np.tile(np.arange(n_celulas, dtype=np.int32), len(missing_cols))
                fill_cols_arr = np.repeat(missing_cols.astype(np.int32), n_celulas)
                fill_data     = np.full(n_fill, fill_value, dtype=np.float32)
                X_fill = sp.csr_matrix(
                    (fill_data, (fill_rows, fill_cols_arr)),
                    shape=(n_celulas, n_genes),
                    dtype=np.float32,
                )
                del fill_rows, fill_cols_arr, fill_data
                X_novo = X_novo + X_fill
                del X_fill
                gc.collect()
                print(f"  Preenchimento concluído.")

        # AnnData exige pd.DataFrame para .var
        var_novo = pd.DataFrame(index=pd.Index(self.genes_ordenados, name='ensembl_id'))
        return ad.AnnData(X=X_novo, obs=adata.obs.copy(), var=var_novo)

    def salvar_como_txt(self, chunk=5000):
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
                    fout.write(pl.from_numpy(X.astype(np.float32)).write_csv(include_header=False))
                    total += end - start
                    if total % (chunk * 5) == 0:
                        print(f"  {total} células processadas...")

            adata.file.close()
            os.rename(path_tmp, path_txt)
            print(f"  Salvo: {path_txt}  ({total} células × {len(gene_names)} genes)  ✓")
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
