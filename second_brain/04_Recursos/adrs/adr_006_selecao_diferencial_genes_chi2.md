---
tipo: adr
tags: [adr, chi2, selecao-features, informacao-mutua, bioinformatica, decisao-arquitetura]
criado: 2026-07-30
atualizado: 2026-07-30
resumo: "ADR 006: Decisão de adotar a Seleção Diferencial por Qui-Quadrado (Chi2 / Ganho de Informação) em substituição à frequência simples."
---

# ADR 006: Seleção Diferencial por Qui-Quadrado ($\chi^2$) para Filtragem de Genes Informáticos

> **Status:** Aceito  
> **Data:** 30/07/2026  
> **Decisores:** Equipe de Bioinformática & Agente AI  

---

## 1. Contexto

A seleção inicial de genes por frequência simples (`SelecionadorGenesFrequentes`) priorizou genes constitutivos (*housekeeping genes*), ativados em quase todas as células. Como esses genes não possuem variabilidade discriminatória entre tipos celulares, a distância L2 e o produto escalar da rede Hopfield foram dominados por ruído de fundo, resultando em perda de acurácia na imputação cross-dataset Fujita $\rightarrow$ Mathys.

## 2. Decisão

Substituir o critério de frequência simples pelo teste de **Qui-Quadrado ($\chi^2$)** (`SelecionadorGenesDiferenciais` via `sklearn.feature_selection.chi2`):
- Avalia-se a dependência estatística entre a presença/ausência do gene ($0$ ou $1$) e a classe biológica (`clo`).
- Selecionam-se os **Top 5.000 genes com maior escore $\chi^2$** (maior Ganho de Informação Biológica).
- Unificam-se os Top 5.000 discriminativos aos genes exclusivos do Fujita para manter a compatibilidade cross-dataset.

Veja o **[[03_Conhecimento/binarizacao_expressao_genica|Conceito Atômico de Binarização]]**.

## 3. Consequências Biológicas

- **Vantagens:** Elimina genes constitutivos inespecíficos (*housekeeping*) e prioriza marcadores biológicos reais de linhagem celular (ex: *CX3CR1*, *GFAP*, *GAD1/2*), elevando dramaticamente a acurácia da classificação de subtipos e reduzindo falsos positivos na matriz de confusão.

## 4. Consequências Técnicas

- **Vantagens:** Melhora o F1-Score macro e a separabilidade dos centroides K-Means no espaço reduzido rSWeeP.
- **Desempenho:** O cálculo do $\chi^2$ em 36.591 genes leva menos de 5 segundos.
