---
tipo: indice
tags: [caixa-de-entrada, triagem, hipoteses, lembretes]
criado: 2026-07-30
atualizado: 2026-08-04
resumo: "Ponto de captura e triagem rápida de hipóteses biológicas, leituras e lembretes de documentação."
---

# 📥 Índice da Caixa de Entrada (Triagem Científica)

Esta pasta funciona como a área de triagem de hipóteses, pré-prints de artigos e lembretes de documentação antes de serem organizados no grafo.

---

## 📌 Lembretes & Itens Pendentes de Triagem

* 🧪 **[[00_Caixa-de-Entrada/lembrete_revisao_conceitual_tecnica_pipeline|Lembrete: Plano de Revisão Conceitual e Técnica do Pipeline (Synthetic Ground Truth)]]**
  * **Pendente:** Criar e executar plano rigoroso de avaliação conceitual (biológica) e técnica (desempenho/IA) para cada etapa do pipeline com datasets sintéticos (*ground truth* vs *dropouts*) e testes unitários.

---

## 🔄 Fluxo de Triagem Científica

1. **É um projeto de pesquisa com entregáveis e código?** $\rightarrow$ Mova para `01_Projetos/` (tipo: `projeto`).
2. **É um domínio científico de acompanhamento contínuo?** $\rightarrow$ Mova para `02_Areas/` (tipo: `area`).
3. **É um conceito teórico, modelo matemático ou aprendizado atômico?** $\rightarrow$ Mova para `03_Conhecimento/` (tipo: `conceito`) e vincule em `03_Conhecimento/index.md`.
4. **É um artigo, documentação externa ou decisão formal?** $\rightarrow$ Mova para `04_Recursos/` (tipo: `recurso` ou `adr`).
5. **Não possui utilidade atual ou concluído?** $\rightarrow$ Mova para `05_Arquivos/`.
