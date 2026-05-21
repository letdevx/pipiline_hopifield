import os

import polars as pl


class AnalisadorCobertura:
    """Verifica quantos dos top-N genes frequentes do Fujita estão presentes no Mathys."""

    def __init__(self, path_top_n, map_f, map_m):
        self.path_top_n = path_top_n
        self.map_f = map_f
        self.map_m = map_m

    def analisar(self, out_csv):
        if os.path.exists(out_csv):
            print(f"[AnalisadorCobertura] Arquivo já existe, pulando: {out_csv}")
            return pl.read_csv(out_csv)

        top_n = pl.read_csv(self.path_top_n)
        map_f_df = pl.DataFrame({
            'gene'      : list(self.map_f.keys()),
            'ensembl_id': list(self.map_f.values()),
        })
        ids_m_list = list(self.map_m.values())

        df = (
            top_n
            .join(map_f_df, on='gene', how='left')
            .with_columns([
                pl.col('ensembl_id')
                    .is_in(ids_m_list)
                    .fill_null(False)
                    .alias('presente_mathys'),
                pl.col('ensembl_id')
                    .is_null()
                    .alias('sem_ensembl_fujita'),
            ])
            .rename({'gene': 'gene_name'})
        )

        total       = df.height
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

        df.write_csv(out_csv)
        print(f"\n[AnalisadorCobertura] Resultado salvo em: {out_csv}")
        return df

    def __repr__(self):
        return (
            f"AnalisadorCobertura(\n"
            f"  path_top_n = {self.path_top_n}\n"
            f")"
        )
