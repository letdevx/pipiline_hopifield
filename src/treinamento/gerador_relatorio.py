"""Módulo de Geração e Exportação de Relatórios Consolidados (HTML e CSV).

Compila métricas de classificação, matrizes de confusão, acurácias globais
e gráficos comparativos em relatórios autocontidos.
"""

from __future__ import annotations

import base64
import io
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from numpy.typing import NDArray

from .avaliador_hopfield import AvaliadorHopfield

PathType = str | os.PathLike[str]


class GeradorRelatorio:
    """Persiste resultados de avaliação da rede Hopfield em relatórios HTML e arquivos CSV.

    Gera três arquivos em out_dir:
    - metricas_globais.csv        — uma linha por avaliador, métricas globais
    - metricas_por_classe.csv     — precision/recall/F1 por classe e avaliador
    - relatorio_{experimento}.html — HTML autocontido com tabelas e imagens base64

    Parameters
    ----------
    out_dir : str | os.PathLike[str]
        Diretório onde os relatórios serão gravados.
    nome_experimento : str, default="experimento"
        Identificador textual do experimento.

    Attributes
    ----------
    out_dir : str
        Caminho do diretório de saída.
    nome_experimento : str
        Nome do relatório.
    """

    def __init__(
        self, out_dir: PathType, nome_experimento: str = "experimento"
    ) -> None:
        self.out_dir: str = str(out_dir)
        self.nome_experimento: str = str(nome_experimento)
        self._avaliadores: dict[str, AvaliadorHopfield] = {}
        self._metadados: dict[str, Any] = {}
        self._genes_ausentes: pd.DataFrame | None = None
        self._mae_05: float | None = None
        self._mae_bin: float | None = None
        self._figuras: list[tuple[str, str, str]] = []

    def adicionar_metadados(self, **kwargs: Any) -> GeradorRelatorio:
        """Adiciona metadados contextuais (parâmetros de treino) ao relatório.

        Parameters
        ----------
        **kwargs : Any
            Pares chave-valor de metadados.

        Returns
        -------
        GeradorRelatorio
            A própria instância.
        """
        self._metadados.update(kwargs)
        return self

    def adicionar_avaliador(
        self, nome: str, avaliador: AvaliadorHopfield
    ) -> GeradorRelatorio:
        """Adiciona uma instância de avaliador executado ao conjunto do relatório.

        Parameters
        ----------
        nome : str
            Identificador do dataset ou cenário.
        avaliador : AvaliadorHopfield
            Instância com métricas já avaliadas.

        Returns
        -------
        GeradorRelatorio
            A própria instância.
        """
        self._avaliadores[nome] = avaliador
        return self

    def adicionar_genes_ausentes(
        self,
        df_ausentes: pd.DataFrame,
        mae_05: float,
        mae_bin: float,
    ) -> GeradorRelatorio:
        """Registra métricas específicas da reconstituição de genes ausentes.

        Parameters
        ----------
        df_ausentes : pd.DataFrame
            Tabela comparativa com colunas ref_fujita, rec_05, rec_bin.
        mae_05 : float
            Erro absoluto médio do cenário 0.5.
        mae_bin : float
            Erro absoluto médio do cenário binarizado.

        Returns
        -------
        GeradorRelatorio
            A própria instância.
        """
        self._genes_ausentes = df_ausentes
        self._mae_05 = float(mae_05)
        self._mae_bin = float(mae_bin)
        return self

    def adicionar_figura(
        self, titulo: str, fig: Figure, secao: str = "Visualizações"
    ) -> GeradorRelatorio:
        """Converte e anexa uma figura Matplotlib em formato Base64 ao relatório.

        Parameters
        ----------
        titulo : str
            Título do painel gráfico.
        fig : Figure
            Figura Matplotlib.
        secao : str, default="Visualizações"
            Nome da seção agrupada no HTML.

        Returns
        -------
        GeradorRelatorio
            A própria instância.
        """
        img_b64: str = self._fig_para_base64(fig)
        self._figuras.append((secao, titulo, img_b64))
        return self

    def _gerar_figuras_html(self) -> str:
        if not self._figuras:
            return ""
        secoes: dict[str, list[tuple[str, str]]] = {}
        for sec, titulo, img in self._figuras:
            secoes.setdefault(sec, []).append((titulo, img))
        blocos: list[str] = []
        for sec, items in secoes.items():
            cards = "".join(
                f'<div class="card"><h3>{t}</h3>'
                f'<img src="data:image/png;base64,{i}"></div>'
                for t, i in items
            )
            blocos.append(f"<h2>{sec}</h2><div class='grid'>{cards}</div>")
        return "\n".join(blocos)

    def gerar(self) -> GeradorRelatorio:
        """Executa a persistência dos arquivos CSV e do documento HTML completo.

        Returns
        -------
        GeradorRelatorio
            A própria instância.
        """
        os.makedirs(self.out_dir, exist_ok=True)
        self._salvar_csv_resumo()
        self._salvar_csv_por_classe()
        self._gerar_html()
        print("[GeradorRelatorio] Concluído.")
        return self

    def _salvar_csv_resumo(self) -> None:
        rows: list[dict[str, Any]] = [
            av.metricas_resumo(nome) for nome, av in self._avaliadores.items()
        ]
        path: str = os.path.join(self.out_dir, "metricas_globais.csv")
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"[GeradorRelatorio] Salvo: {path}")

    def _salvar_csv_por_classe(self) -> None:
        frames: list[pd.DataFrame] = []
        for nome, av in self._avaliadores.items():
            df: pd.DataFrame = av.metricas_por_classe().copy()
            df.insert(0, "dataset", nome)
            frames.append(df)
        path: str = os.path.join(self.out_dir, "metricas_por_classe.csv")
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(path, index=False)
        else:
            pd.DataFrame().to_csv(path, index=False)
        print(f"[GeradorRelatorio] Salvo: {path}")

    @staticmethod
    def _fig_para_base64(fig: Figure) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def _gerar_html(self) -> None:
        css: str = """
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
        meta_rows: str = "\n".join(
            f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in self._metadados.items()
        )
        meta_html: str = (
            f"<table class='meta'>"
            f"<tr><th>Parâmetro</th><th>Valor</th></tr>"
            f"{meta_rows}</table>"
        )

        # Tabela de métricas globais com destaque no melhor por coluna
        colunas_num: list[str] = [
            "acuracia",
            "f1_macro",
            "f1_weighted",
            "taxa_reconstrucao",
            "semelhanca_media",
        ]
        rows_resumo: list[dict[str, Any]] = [
            av.metricas_resumo(nome) for nome, av in self._avaliadores.items()
        ]
        df_res: pd.DataFrame = pd.DataFrame(rows_resumo)
        max_idx: dict[str, Any] = {}
        if not df_res.empty:
            for c in colunas_num:
                if c in df_res.columns:
                    max_idx[c] = df_res[c].idxmax()

        cab_res: str = (
            "<tr><th>Dataset</th><th>N Células</th>"
            "<th>Acurácia</th><th>F1 Macro</th><th>F1 Weighted</th>"
            "<th>Taxa Reconstrução</th><th>Semelhança Média</th></tr>"
        )
        lins_res: list[str] = []
        for i, r in df_res.iterrows():
            tds = [f"<td>{r['dataset']}</td>", f"<td>{r['n_celulas']:,}</td>"]
            for col in colunas_num:
                cls_attr = ' class="best"' if max_idx.get(col) == i else ""
                tds.append(f"<td{cls_attr}>{r[col]:.4f}</td>")
            lins_res.append(f"<tr>{''.join(tds)}</tr>")
        tabela_res: str = f"<table>{cab_res}{''.join(lins_res)}</table>"

        # Matrizes de confusão embutidas como base64
        cards: list[str] = []
        for nome, av in self._avaliadores.items():
            for normalizado, sufixo in [(False, "contagens"), (True, "normalizada")]:
                fig, ax = plt.subplots(figsize=(6, 5))
                av.plotar(titulo=f"{nome} ({sufixo})", normalizado=normalizado, ax=ax)
                img_b64: str = self._fig_para_base64(fig)
                plt.close(fig)
                cards.append(
                    f'<div class="card">'
                    f"<h3>{nome} ({sufixo})</h3>"
                    f'<img src="data:image/png;base64,{img_b64}">'
                    f"</div>"
                )
        confusion_html: str = f'<div class="grid">{"".join(cards)}</div>'

        # Tabelas por classe
        per_class_html: list[str] = []
        for nome, av in self._avaliadores.items():
            df_pc: pd.DataFrame = av.metricas_por_classe()
            cab = "<tr>" + "".join(f"<th>{c}</th>" for c in df_pc.columns) + "</tr>"
            lins = []
            for _, row in df_pc.iterrows():
                lins.append("<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>")
            per_class_html.append(f"<h3>{nome}</h3><table>{cab}{''.join(lins)}</table>")

        # Seção: genes ausentes no Mathys
        genes_html: str = ""
        if self._genes_ausentes is not None:
            df_ga: pd.DataFrame = self._genes_ausentes
            df_top: pd.DataFrame = df_ga.sort_values(
                "frequencia", ascending=False
            ).head(20)
            cab_ga = "<tr>" + "".join(f"<th>{c}</th>" for c in df_top.columns) + "</tr>"
            lins_ga = []
            for _, row in df_top.iterrows():
                lins_ga.append("<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>")
            f"<table>{cab_ga}{''.join(lins_ga)}</table>"

            ref: NDArray[np.float32] = df_ga["ref_fujita"].to_numpy().astype(np.float32)
            r05: NDArray[np.float32] = df_ga["rec_05"].to_numpy().astype(np.float32)
            rbin: NDArray[np.float32] = df_ga["rec_bin"].to_numpy().astype(np.float32)
            freqs: NDArray[np.float32] = (
                df_ga["frequencia"].to_numpy().astype(np.float32)
            )

            fig_ga, axes_ga = plt.subplots(1, 3, figsize=(18, 5))
            ax0 = axes_ga[0]
            ax0.scatter(
                ref, r05, alpha=0.7, label="Mathys 0.5", color="steelblue", s=30
            )
            ax0.scatter(
                ref,
                rbin,
                alpha=0.7,
                label="Mathys bin",
                color="tomato",
                s=30,
                marker="s",
            )
            ax0.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
            ax0.set_xlabel("Fujita (ref)")
            ax0.set_ylabel("Mathys (rec)")
            ax0.set_title("Ref vs Reconstruído")
            ax0.legend()
            ax0.set_xlim(-0.05, 1.05)
            ax0.set_ylim(-0.05, 1.05)

            ax1 = axes_ga[1]
            mae05_str = f"{self._mae_05:.3f}" if self._mae_05 is not None else "0"
            maebin_str = f"{self._mae_bin:.3f}" if self._mae_bin is not None else "0"
            ax1.hist(
                r05 - ref,
                bins=20,
                alpha=0.6,
                label=f"0.5−ref (MAE={mae05_str})",
                color="steelblue",
            )
            ax1.hist(
                rbin - ref,
                bins=20,
                alpha=0.6,
                label=f"bin−ref (MAE={maebin_str})",
                color="tomato",
            )
            ax1.axvline(0, color="k", lw=1, ls="--")
            ax1.set_xlabel("Erro (reconstruído − referência)")
            ax1.set_ylabel("Número de genes")
            ax1.set_title("Distribuição do erro")
            ax1.legend()

            ax2 = axes_ga[2]
            n_show: int = min(20, len(ref))
            ordem: NDArray[np.intp] = np.argsort(freqs)[::-1][:n_show]
            y_pos: NDArray[np.intp] = np.arange(n_show)
            h_bar: float = 0.25
            gene_names: NDArray[Any] = (
                df_ga["gene"].to_numpy()
                if "gene" in df_ga.columns
                else np.arange(len(ref)).astype(str)
            )
            ax2.barh(
                y_pos + h_bar,
                ref[ordem],
                h_bar,
                label="Fujita (ref)",
                color="gray",
                alpha=0.8,
            )
            ax2.barh(
                y_pos,
                r05[ordem],
                h_bar,
                label="Mathys 0.5",
                color="steelblue",
                alpha=0.8,
            )
            ax2.barh(
                y_pos - h_bar,
                rbin[ordem],
                h_bar,
                label="Mathys bin",
                color="tomato",
                alpha=0.8,
            )
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels([str(gene_names[i])[:14] for i in ordem], fontsize=8)
            ax2.set_xlabel("Taxa de ativação")
            ax2.set_title(f"Top-{n_show} genes ausentes (por frequência)")
            ax2.legend(fontsize=8)
            ax2.set_xlim(0, 1.1)

            img_ga: str = self._fig_para_base64(fig_ga)
            plt.close(fig_ga)

            genes_html = f"""
  <h2>Genes Ausentes no Mathys (top-5000)</h2>
  <p>Genes do top-5000 Fujita ausentes no Mathys (preenchidos com sentinela 0.5).</p>
  <p>MAE cenário 0.5 vs Fujita: <strong>{mae05_str}</strong> &nbsp;|&nbsp;
     MAE cenário bin vs Fujita: <strong>{maebin_str}</strong></p>
  <img src="data:image/png;base64,{img_ga}" style="max-width:1200px">"""

        figuras_html: str = self._gerar_figuras_html()

        # Monta HTML final
        html: str = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Relatório — {self.nome_experimento}</title>
  <style>{css}</style>
</head>
<body>
  <h1>Relatório: {self.nome_experimento}</h1>

  <h2>Metadados do Experimento</h2>
  {meta_html}

  <h2>Métricas Globais</h2>
  {tabela_res}

  <h2>Matrizes de Confusão</h2>
  {confusion_html}

  <h2>Métricas por Classe</h2>
  {"".join(per_class_html)}
  {figuras_html}
  {genes_html}
</body>
</html>"""

        path_html: str = os.path.join(
            self.out_dir, f"relatorio_{self.nome_experimento}.html"
        )
        with open(path_html, "w", encoding="utf-8") as fh:
            fh.write(html)
        print(f"[GeradorRelatorio] HTML salvo: {path_html}")

    def __repr__(self) -> str:
        """Representação textual do gerador de relatórios."""
        return (
            f"GeradorRelatorio(\n"
            f"  out_dir          = {self.out_dir}\n"
            f"  nome_experimento = {self.nome_experimento}\n"
            f"  avaliadores      = {list(self._avaliadores.keys())}\n"
            f")"
        )
