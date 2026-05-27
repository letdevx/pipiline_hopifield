import os

import polars as pl


class SelecionadorGenesFrequentes:
    """Calcula os N genes mais frequentes a partir de um arquivo CSV/TXT binarizado.

    Frequência = soma da coluna (equivalente a células com valor > 0 em dados binarizados).
    A primeira coluna é detectada automaticamente como identificador de célula.
    """

    def __init__(self, path_txt, n=5000):
        self.path_txt     = path_txt
        self.n            = n
        self.df_resultado = None
        self._data        = None

    def calcular(self, out_csv=None):
        if out_csv and os.path.exists(out_csv):
            print(f"[SelecionadorGenesFrequentes] Arquivo já existe, pulando: {out_csv}")
            self.df_resultado = pl.read_csv(out_csv)
            return self
        print(f"[SelecionadorGenesFrequentes] Lendo: {self.path_txt}")

        with open(self.path_txt, encoding='utf-8') as fh:
            header = fh.readline().strip().split(',')
        coluna_celulas = header[0]
        total_genes    = len(header) - 1

        print(f"  Calculando frequências para {total_genes} genes (streaming)...")

        somas = (
            pl.scan_csv(self.path_txt, infer_schema_length=1)
            .select(pl.all().exclude(coluna_celulas).sum())
            .collect()
        )
        df_frequencias = somas.unpivot(variable_name="gene", value_name="frequencia")

        n_real = min(self.n, len(df_frequencias))
        self.df_resultado = (
            df_frequencias
            .sort("frequencia", descending=True)
            .head(n_real)
        )

        print(f"[SelecionadorGenesFrequentes] Concluído. Top {n_real} genes selecionados.")
        return self

    def salvar(self, out_csv):
        if self.df_resultado is None:
            raise RuntimeError("Execute .calcular() antes de salvar.")
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        self.df_resultado.write_csv(out_csv)
        print(f"[SelecionadorGenesFrequentes] Salvo em: {out_csv}")
        return self

    def filtrar_matriz(self, out_csv):
        """Salva nova matriz contendo apenas as colunas dos top N genes selecionados."""
        if self.df_resultado is None or self._data is None:
            raise RuntimeError("Execute .calcular() antes de filtrar.")

        coluna_celulas = self._data.columns[0]
        lista_genes    = self.df_resultado["gene"].to_list()
        colunas_validas = [coluna_celulas] + [c for c in lista_genes if c in self._data.columns]

        df_filtrado = self._data.select(colunas_validas)
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df_filtrado.write_csv(out_csv)
        print(f"[SelecionadorGenesFrequentes] Matriz filtrada salva em: {out_csv} ({len(df_filtrado.columns)} colunas)")
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
