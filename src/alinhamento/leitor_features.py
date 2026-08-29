import polars as pl

try:
    from src.config import PATH_FEATURES_REFERENCIA, PATH_FEATURES_ALVO
except ImportError:
    from config import PATH_FEATURES_REFERENCIA, PATH_FEATURES_ALVO


class LeitorFeatures:
    """Lê arquivos TSV de features do 10x Genomics e mapeia gene_name → Ensembl ID."""

    def __init__(
        self,
        path_features_referencia=None,
        path_features_alvo=None,
        path_features_f=None,
        path_features_m=None,
    ):
        self.path_features_referencia = (
            path_features_referencia
            or path_features_f
            or PATH_FEATURES_REFERENCIA
        )
        self.path_features_alvo = (
            path_features_alvo
            or path_features_m
            or PATH_FEATURES_ALVO
        )
        self.map_referencia = None
        self.map_alvo = None

    @property
    def path_features_f(self):
        return self.path_features_referencia

    @path_features_f.setter
    def path_features_f(self, val):
        self.path_features_referencia = val

    @property
    def path_features_m(self):
        return self.path_features_alvo

    @path_features_m.setter
    def path_features_m(self, val):
        self.path_features_alvo = val

    @property
    def map_f(self):
        return self.map_referencia

    @map_f.setter
    def map_f(self, val):
        self.map_referencia = val

    @property
    def map_m(self):
        return self.map_alvo

    @map_m.setter
    def map_m(self, val):
        self.map_alvo = val

    def ler(self):
        self.map_referencia = self._ler_features(self.path_features_referencia)
        self.map_alvo = self._ler_features(self.path_features_alvo)
        print(f"[LeitorFeatures] Referência : {len(self.map_referencia)} genes mapeados")
        print(f"[LeitorFeatures] Alvo       : {len(self.map_alvo)} genes mapeados")
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
            .with_columns([
                pl.col('ensembl_id').cast(pl.Utf8).str.strip_chars(),
                pl.col('gene_name').cast(pl.Utf8).str.strip_chars(),
            ])
            .unique(subset=['gene_name'], keep='first')
        )
        return dict(zip(df['gene_name'].to_list(), df['ensembl_id'].to_list()))

    def __repr__(self):
        n_ref = len(self.map_referencia) if self.map_referencia is not None else 'não carregado'
        n_alvo = len(self.map_alvo) if self.map_alvo is not None else 'não carregado'
        return (
            f"LeitorFeatures(\n"
            f"  path_features_referencia = {self.path_features_referencia}\n"
            f"  path_features_alvo       = {self.path_features_alvo}\n"
            f"  map_referencia           = {n_ref} genes\n"
            f"  map_alvo                 = {n_alvo} genes\n"
            f")"
        )

