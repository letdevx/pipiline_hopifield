import polars as pl


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
        df = (
            pl.read_csv(
                path,
                separator='\t',
                has_header=False,
                new_columns=['ensembl_id', 'gene_name'],
                columns=[0, 1],
            )
            .unique(subset=['gene_name'], keep='first')
        )
        return dict(zip(df['gene_name'].to_list(), df['ensembl_id'].to_list()))

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
