import os
import gc
import numpy as np
import pandas as pd
import polars as pl
import anndata as ad
import scipy.sparse as sp


class LeitorFeatures:
    """Lê arquivos TSV de features do 10x Genomics e mapeia gene_name → Ensembl ID."""

    def __init__(self, path_features_f, path_features_m):
        self.path_features_f = path_features_f
        self.path_features_m = path_features_m
        self.map_f = None
        self.map_m = None

    def ler(self):
        self.map_f = self._ler_features(self.path_features_f)
        self.map_m = self._ler_features(self.path_features_m)
        print(f"[LeitorFeatures] Fujita : {len(self.map_f)} genes mapeados")
        print(f"[LeitorFeatures] Mathys : {len(self.map_m)} genes mapeados")
        return self

    def _ler_features(self, path):
        df = pd.read_csv(path, sep='\t', header=None, usecols=[0, 1],
                         names=['ensembl_id', 'gene_name'])
        df = df.drop_duplicates(subset='gene_name', keep='first')
        return dict(zip(df['gene_name'], df['ensembl_id']))

    def __repr__(self):
        n_f = len(self.map_f) if self.map_f else 'não carregado'
        n_m = len(self.map_m) if self.map_m else 'não carregado'
        return (
            f"LeitorFeatures(\n"
            f"  path_features_f = {self.path_features_f}\n"
            f"  path_features_m = {self.path_features_m}\n"
            f"  map_f           = {n_f} genes\n"
            f"  map_m           = {n_m} genes\n"
            f")"
        )


class AnalisadorSobreposicao:
    """Calcula sobreposição de espaços gênicos e define a ordem canônica baseada no Fujita."""

    def __init__(self, map_f, map_m, var_names_f_original):
        self.map_f = map_f
        self.map_m = map_m
        self.var_names_f_original = var_names_f_original
        self.ids_comuns = None
        self.ids_so_f = None
        self.ids_so_m = None
        self.genes_ordenados = None
        self.gene_alvo_idx = None

    def analisar(self):
        ids_f = set(self.map_f.values())
        ids_m = set(self.map_m.values())
        self.ids_comuns = ids_f & ids_m
        self.ids_so_f   = ids_f - ids_m
        self.ids_so_m   = ids_m - ids_f

        seen = set()
        self.genes_ordenados = []
        for gene_name in self.var_names_f_original:
            eid = self.map_f.get(gene_name, gene_name)
            if eid not in seen:
                self.genes_ordenados.append(eid)
                seen.add(eid)

        self.gene_alvo_idx = {g: i for i, g in enumerate(self.genes_ordenados)}

        print(f"[AnalisadorSobreposicao] Em comum  : {len(self.ids_comuns)}")
        print(f"[AnalisadorSobreposicao] Só Fujita : {len(self.ids_so_f)}")
        print(f"[AnalisadorSobreposicao] Só Mathys : {len(self.ids_so_m)}  ← serão excluídos")
        print(f"[AnalisadorSobreposicao] Espaço gênico final: {len(self.genes_ordenados)} genes")
        return self

    def __repr__(self):
        def _n(x): return len(x) if x is not None else 'não calculado'
        return (
            f"AnalisadorSobreposicao(\n"
            f"  ids_comuns      = {_n(self.ids_comuns)}\n"
            f"  ids_so_f        = {_n(self.ids_so_f)}\n"
            f"  ids_so_m        = {_n(self.ids_so_m)}\n"
            f"  genes_ordenados = {_n(self.genes_ordenados)}\n"
            f")"
        )


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
        # Caminhos computados antes de qualquer I/O para permitir skip
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

            # Remove .tmp de execução anterior interrompida, se existir
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
            return pd.read_csv(out_tracking)
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
        df_tracking = (
            pd.DataFrame(tracking_rows)
            .sort_values('posicao_coluna')
            .reset_index(drop=True)
        )
        df_tracking.to_csv(out_tracking, index=False)
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


class ValidadorAlinhamento:
    """Valida que dois AnnData alinhados possuem exatamente a mesma lista de genes na mesma ordem."""

    def __init__(self, path_f_alinhado, path_m_alinhado, genes_ordenados):
        self.path_f_alinhado = path_f_alinhado
        self.path_m_alinhado = path_m_alinhado
        self.genes_ordenados = genes_ordenados

    def validar(self):
        print("[ValidadorAlinhamento] Carregando metadados...")
        _f = ad.read_h5ad(self.path_f_alinhado, backed='r')
        _m = ad.read_h5ad(self.path_m_alinhado, backed='r')
        genes_f = _f.var_names.tolist()
        genes_m = _m.var_names.tolist()
        _f.file.close()
        _m.file.close()
        del _f, _m

        if len(genes_f) != len(self.genes_ordenados):
            raise ValueError(
                f"[VALIDAÇÃO FALHOU] Fujita alinhado tem {len(genes_f)} genes, "
                f"esperado {len(self.genes_ordenados)}."
            )
        if len(genes_m) != len(self.genes_ordenados):
            raise ValueError(
                f"[VALIDAÇÃO FALHOU] Mathys alinhado tem {len(genes_m)} genes, "
                f"esperado {len(self.genes_ordenados)}."
            )

        divs_f = [(i, self.genes_ordenados[i], genes_f[i])
                  for i in range(len(self.genes_ordenados))
                  if genes_f[i] != self.genes_ordenados[i]]
        if divs_f:
            msg = "\n  ".join(
                f"pos {i}: esperado={e!r} encontrado={e2!r}" for i, e, e2 in divs_f[:5]
            )
            raise ValueError(
                f"[VALIDAÇÃO FALHOU] Fujita diverge em {len(divs_f)} posição(ões):\n  {msg}"
            )

        divs_m = [(i, self.genes_ordenados[i], genes_m[i])
                  for i in range(len(self.genes_ordenados))
                  if genes_m[i] != self.genes_ordenados[i]]
        if divs_m:
            msg = "\n  ".join(
                f"pos {i}: esperado={e!r} encontrado={e2!r}" for i, e, e2 in divs_m[:5]
            )
            raise ValueError(
                f"[VALIDAÇÃO FALHOU] Mathys diverge em {len(divs_m)} posição(ões):\n  {msg}"
            )

        divs_fm = [(i, genes_f[i], genes_m[i])
                   for i in range(len(genes_f))
                   if genes_f[i] != genes_m[i]]
        if divs_fm:
            msg = "\n  ".join(
                f"pos {i}: F={gf!r} M={gm!r}" for i, gf, gm in divs_fm[:5]
            )
            raise ValueError(
                f"[VALIDAÇÃO FALHOU] Fujita e Mathys divergem em {len(divs_fm)} posição(ões):\n  {msg}"
            )

        print(f"✓ Número de genes idêntico: {len(self.genes_ordenados)}")
        print(f"✓ Fujita alinhado == ordem de referência")
        print(f"✓ Mathys alinhado == ordem de referência")
        print(f"✓ Fujita alinhado == Mathys alinhado")
        print("[ValidadorAlinhamento] Validação concluída com sucesso.")
        return self

    def __repr__(self):
        return (
            f"ValidadorAlinhamento(\n"
            f"  path_f_alinhado  = {self.path_f_alinhado}\n"
            f"  path_m_alinhado  = {self.path_m_alinhado}\n"
            f"  genes_ordenados  = {len(self.genes_ordenados)} genes\n"
            f")"
        )


class SelecionadorGenesFrequentes:
    """Calcula os N genes mais frequentes a partir de um arquivo TXT alinhado.

    Frequência = número de células em que o gene tem valor > 0.
    Leitura em chunks para suportar arquivos grandes.
    """

    def __init__(self, path_txt, n=5000, chunk=3000):
        self.path_txt     = path_txt
        self.n            = n
        self.chunk        = chunk
        self.df_resultado = None

    def calcular(self):
        print(f"[SelecionadorGenesFrequentes] Lendo: {self.path_txt}")
        with open(self.path_txt, encoding='utf-8') as f:
            genes = f.readline().strip().split(',')

        frequencias = np.zeros(len(genes), dtype=np.int64)
        total_celulas = 0

        for chunk_df in pd.read_csv(self.path_txt, chunksize=self.chunk, header=0):
            frequencias  += (chunk_df.values > 0).sum(axis=0)
            total_celulas += len(chunk_df)
            if total_celulas % (self.chunk * 10) == 0:
                print(f"  {total_celulas} células processadas...")

        n_real = min(self.n, len(genes))
        idx_top = np.argsort(frequencias)[::-1][:n_real]
        self.df_resultado = pd.DataFrame({
            'gene'      : [genes[i] for i in idx_top],
            'frequencia': frequencias[idx_top],
        })
        print(f"[SelecionadorGenesFrequentes] Concluído. {total_celulas} células, top {n_real} genes selecionados.")
        return self

    def salvar(self, out_csv):
        if self.df_resultado is None:
            raise RuntimeError("Execute .calcular() antes de salvar.")
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        self.df_resultado.to_csv(out_csv, index=False)
        print(f"[SelecionadorGenesFrequentes] Salvo em: {out_csv}")
        return self

    def __repr__(self):
        n = len(self.df_resultado) if self.df_resultado is not None else 'não calculado'
        return (
            f"SelecionadorGenesFrequentes(\n"
            f"  path_txt     = {self.path_txt}\n"
            f"  n            = {self.n}\n"
            f"  df_resultado = {n} genes\n"
            f")"
        )


class AnalisadorCobertura:
    """Verifica quantos dos top-N genes frequentes do Fujita estão presentes no Mathys."""

    def __init__(self, path_top_n, map_f, map_m):
        self.path_top_n = path_top_n
        self.map_f = map_f
        self.map_m = map_m

    def analisar(self, out_csv):
        if os.path.exists(out_csv):
            print(f"[AnalisadorCobertura] Arquivo já existe, pulando: {out_csv}")
            return pd.read_csv(out_csv)
        top_n     = pd.read_csv(self.path_top_n)
        ids_m_set = set(self.map_m.values())
        resultados = []
        for _, row in top_n.iterrows():
            gene_name   = row['gene']
            frequencia  = row['frequencia']
            ensembl_id  = self.map_f.get(gene_name, None)
            no_mathys   = (ensembl_id in ids_m_set) if ensembl_id else False
            sem_ensembl = ensembl_id is None
            resultados.append({
                'gene_name'         : gene_name,
                'frequencia'        : frequencia,
                'ensembl_id'        : ensembl_id if ensembl_id else 'N/A',
                'presente_mathys'   : no_mathys,
                'sem_ensembl_fujita': sem_ensembl,
            })

        df          = pd.DataFrame(resultados)
        total       = len(df)
        presentes   = df['presente_mathys'].sum()
        sem_ensembl = df['sem_ensembl_fujita'].sum()
        ausentes    = total - presentes - sem_ensembl

        print(f"\n{'='*50}")
        print(f"  Top {total} genes frequentes do Fujita")
        print(f"{'='*50}")
        print(f"  Presentes no Mathys       : {presentes:>5}  ({presentes/total*100:.1f}%)")
        print(f"  Ausentes no Mathys (zeros): {ausentes:>5}  ({ausentes/total*100:.1f}%)")
        print(f"  Sem Ensembl ID no Fujita  : {sem_ensembl:>5}  ({sem_ensembl/total*100:.1f}%)")
        print(f"{'='*50}")

        df.to_csv(out_csv, index=False)
        print(f"\n[AnalisadorCobertura] Resultado salvo em: {out_csv}")
        return df

    def __repr__(self):
        return (
            f"AnalisadorCobertura(\n"
            f"  path_top_n = {self.path_top_n}\n"
            f")"
        )
