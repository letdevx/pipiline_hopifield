import base64
import io
import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class GeradorRelatorio:
    """Persiste resultados de avaliação da rede Hopfield em HTML + CSV.

    Gera três arquivos em out_dir:
    - metricas_globais.csv        — uma linha por avaliador, métricas globais
    - metricas_por_classe.csv     — precision/recall/F1 por classe e avaliador
    - relatorio_{experimento}.html — HTML autocontido com tabelas e imagens base64
    """

    def __init__(self, out_dir, nome_experimento="experimento"):
        self.out_dir = out_dir
        self.nome_experimento = nome_experimento
        self._avaliadores = {}   # {nome: AvaliadorHopfield}, ordem de inserção
        self._metadados = {}
        self._genes_ausentes = None
        self._mae_05  = None
        self._mae_bin = None

    def adicionar_metadados(self, **kwargs):
        self._metadados.update(kwargs)
        return self

    def adicionar_avaliador(self, nome, avaliador):
        self._avaliadores[nome] = avaliador
        return self

    def adicionar_genes_ausentes(self, df_ausentes, mae_05, mae_bin):
        self._genes_ausentes = df_ausentes
        self._mae_05  = mae_05
        self._mae_bin = mae_bin
        return self

    def gerar(self):
        os.makedirs(self.out_dir, exist_ok=True)
        self._salvar_csv_resumo()
        self._salvar_csv_por_classe()
        self._gerar_html()
        print("[GeradorRelatorio] Concluído.")
        return self

    # ------------------------------------------------------------------
    # CSVs
    # ------------------------------------------------------------------

    def _salvar_csv_resumo(self):
        rows = [av.metricas_resumo(nome) for nome, av in self._avaliadores.items()]
        path = os.path.join(self.out_dir, "metricas_globais.csv")
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"[GeradorRelatorio] Salvo: {path}")

    def _salvar_csv_por_classe(self):
        frames = []
        for nome, av in self._avaliadores.items():
            df = av.metricas_por_classe().copy()
            df.insert(0, "dataset", nome)
            frames.append(df)
        path = os.path.join(self.out_dir, "metricas_por_classe.csv")
        pd.concat(frames, ignore_index=True).to_csv(path, index=False)
        print(f"[GeradorRelatorio] Salvo: {path}")

    # ------------------------------------------------------------------
    # HTML
    # ------------------------------------------------------------------

    @staticmethod
    def _fig_para_base64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def _gerar_html(self):
        css = """
        body{font-family:Arial,sans-serif;max-width:1400px;margin:0 auto;padding:24px;background:#f5f5f5}
        h1{color:#2c3e50;border-bottom:3px solid #2980b9;padding-bottom:8px}
        h2{color:#34495e;margin-top:40px}
        h3{color:#555;margin:20px 0 6px}
        table{border-collapse:collapse;width:100%;margin-bottom:16px;background:#fff;
              box-shadow:0 1px 3px rgba(0,0,0,.12)}
        th{background:#2980b9;color:#fff;padding:10px 14px;text-align:left}
        td{padding:8px 12px;border-bottom:1px solid #e0e0e0;text-align:left}
        tr:hover td{background:#eef6ff}
        .best{background:#c6efce!important;font-weight:bold}
        .meta td:first-child{font-weight:bold;color:#555;width:220px}
        .grid{display:flex;flex-wrap:wrap;gap:16px;margin-bottom:24px}
        .card{background:#fff;padding:12px;box-shadow:0 1px 3px rgba(0,0,0,.12)}
        .card h3{margin:0 0 8px;font-size:13px}
        img{max-width:100%}
        """

        # Metadados
        meta_rows = "\n".join(
            f"<tr><td>{k}</td><td>{v}</td></tr>"
            for k, v in self._metadados.items()
        )
        meta_html = (
            f"<table class='meta'>"
            f"<tr><th>Parâmetro</th><th>Valor</th></tr>"
            f"{meta_rows}</table>"
        )

        # Tabela de métricas globais com destaque no melhor por coluna
        colunas_num = ["acuracia", "f1_macro", "f1_weighted", "taxa_reconstrucao", "semelhanca_media"]
        rows_resumo = [av.metricas_resumo(nome) for nome, av in self._avaliadores.items()]
        df_res = pd.DataFrame(rows_resumo)
        max_idx = {c: df_res[c].idxmax() for c in colunas_num}

        cab_res = (
            "<tr><th>Dataset</th><th>N Células</th>"
            "<th>Acurácia</th><th>F1 Macro</th><th>F1 Weighted</th>"
            "<th>Taxa Reconstrução</th><th>Semelhança Média</th></tr>"
        )
        lins_res = []
        for i, r in df_res.iterrows():
            tds = [f"<td>{r['dataset']}</td>", f"<td>{r['n_celulas']:,}</td>"]
            for col in colunas_num:
                cls = ' class="best"' if max_idx[col] == i else ""
                tds.append(f"<td{cls}>{r[col]:.4f}</td>")
            lins_res.append(f"<tr>{''.join(tds)}</tr>")
        tabela_res = f"<table>{cab_res}{''.join(lins_res)}</table>"

        # Matrizes de confusão embutidas como base64
        cards = []
        for nome, av in self._avaliadores.items():
            for normalizado, sufixo in [(False, "contagens"), (True, "normalizada")]:
                fig, ax = plt.subplots(figsize=(6, 5))
                av.plotar(titulo=f"{nome} ({sufixo})", normalizado=normalizado, ax=ax)
                img_b64 = self._fig_para_base64(fig)
                plt.close(fig)
                cards.append(
                    f'<div class="card">'
                    f'<h3>{nome} ({sufixo})</h3>'
                    f'<img src="data:image/png;base64,{img_b64}">'
                    f'</div>'
                )
        confusion_html = f'<div class="grid">{"".join(cards)}</div>'

        # Tabelas por classe
        per_class_html = []
        for nome, av in self._avaliadores.items():
            df_pc = av.metricas_por_classe()
            cab = "<tr>" + "".join(f"<th>{c}</th>" for c in df_pc.columns) + "</tr>"
            lins = []
            for _, row in df_pc.iterrows():
                lins.append("<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>")
            per_class_html.append(f"<h3>{nome}</h3><table>{cab}{''.join(lins)}</table>")

        # Seção: genes ausentes no Mathys
        genes_html = ""
        if self._genes_ausentes is not None:
            df_ga  = self._genes_ausentes
            df_top = df_ga.sort_values("frequencia", ascending=False).head(20)
            cab_ga = "<tr>" + "".join(f"<th>{c}</th>" for c in df_top.columns) + "</tr>"
            lins_ga = []
            for _, row in df_top.iterrows():
                lins_ga.append("<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>")
            tabela_ga = f"<table>{cab_ga}{''.join(lins_ga)}</table>"

            import numpy as _np
            ref  = df_ga["ref_fujita"].values
            r05  = df_ga["rec_05"].values
            rbin = df_ga["rec_bin"].values

            fig_ga, axes_ga = plt.subplots(1, 2, figsize=(12, 5))
            ax = axes_ga[0]
            ax.scatter(ref, r05,  alpha=0.7, label="Mathys 0.5", color="steelblue", s=30)
            ax.scatter(ref, rbin, alpha=0.7, label="Mathys bin", color="tomato", s=30, marker="s")
            ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
            ax.set_xlabel("Fujita (ref)"); ax.set_ylabel("Mathys (rec)")
            ax.set_title("Ref vs Reconstruído"); ax.legend()
            ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)

            ax = axes_ga[1]
            ax.hist(r05  - ref, bins=20, alpha=0.6, label=f"0.5−ref (MAE={self._mae_05:.3f})", color="steelblue")
            ax.hist(rbin - ref, bins=20, alpha=0.6, label=f"bin−ref (MAE={self._mae_bin:.3f})", color="tomato")
            ax.axvline(0, color="k", lw=1, ls="--")
            ax.set_xlabel("Erro (reconstruído − referência)"); ax.set_ylabel("Número de genes")
            ax.set_title("Distribuição do erro"); ax.legend()

            img_ga = self._fig_para_base64(fig_ga)
            plt.close(fig_ga)

            genes_html = f"""
  <h2>Genes Ausentes no Mathys (top-5000)</h2>
  <p>Genes do top-5000 Fujita ausentes no Mathys (preenchidos com sentinela 0.5).</p>
  <p>MAE cenário 0.5 vs Fujita: <strong>{self._mae_05:.4f}</strong> &nbsp;|&nbsp;
     MAE cenário bin vs Fujita: <strong>{self._mae_bin:.4f}</strong></p>
  <img src="data:image/png;base64,{img_ga}" style="max-width:960px">
  <h3>Top-20 genes por frequência</h3>
  {tabela_ga}"""

        # Monta HTML final
        data_geracao = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Relatório — {self.nome_experimento}</title>
  <style>{css}</style>
</head>
<body>
  <h1>Relatório: {self.nome_experimento}</h1>
  <p>Gerado em: <strong>{data_geracao}</strong></p>

  <h2>Metadados do Experimento</h2>
  {meta_html}

  <h2>Métricas Globais</h2>
  {tabela_res}

  <h2>Matrizes de Confusão</h2>
  {confusion_html}

  <h2>Métricas por Classe</h2>
  {"".join(per_class_html)}
  {genes_html}
</body>
</html>"""

        path_html = os.path.join(self.out_dir, f"relatorio_{self.nome_experimento}.html")
        with open(path_html, "w", encoding="utf-8") as fh:
            fh.write(html)
        print(f"[GeradorRelatorio] HTML salvo: {path_html}")

    def __repr__(self):
        return (
            f"GeradorRelatorio(\n"
            f"  out_dir          = {self.out_dir}\n"
            f"  nome_experimento = {self.nome_experimento}\n"
            f"  avaliadores      = {list(self._avaliadores.keys())}\n"
            f")"
        )
