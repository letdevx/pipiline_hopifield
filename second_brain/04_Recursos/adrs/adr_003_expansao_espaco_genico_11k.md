---
tipo: adr
tags: [adr, expansao-genica, top-genes, dimensao, decisao-arquitetura]
criado: 2026-07-30
atualizado: 2026-07-30
resumo: "ADR 003: Decisão de expandir o espaço gênico para ~11.000 genes (Top 5k Fujita + Exclusivos do Fujita)."
---

# ADR 003: Expansão do Espaço Gênico para ~11.000 Genes (Top 5k Fujita + Exclusivos do Fujita)

> **Status:** Aceito  
> **Data:** 30/07/2026  
> **Decisores:** Equipe de Bioinformática & Agente AI  

---

## 1. Contexto

O pipeline original utilizava apenas 5.000 genes mais frequentes. Contudo, essa restrição descartava genes altamente específicos de classes celulares raras ou exclusivos do tecido de referência Fujita, impedindo a imputação cross-dataset de regiões transcricionais cruciais.

## 2. Decisão

Expandir a dimensionalidade de trabalho da Hopfield para aproximadamente **11.000 genes**, unificando:
1. Os **5.000 genes mais frequentes** do Fujita.
2. Todos os **~6.000 genes nativos do Fujita ausentes no Mathys** (catalogados no rastreamento de alinhamento).

$$\text{Espaço Final} = \text{Top 5.000} \cup \text{Exclusivos Fujita} \approx 11.279 \text{ genes}$$

Veja o **[[03_Conhecimento/alinhamento_ensembl_cross_dataset|Conceito Atômico de Alinhamento]]**.

## 3. Consequências Biológicas

- **Vantagens:** Restaura a capacidade de reconstrução de marcadores biológicos ausentes no dataset Mathys, ampliando o poder de identificação de subtipos neuronais e gliais.

## 4. Consequências Técnicas

- **Impacto em Memória:** Aumenta o tamanho das matrizes binárias de ~5k para ~11k colunas, elevando a matriz $W_0$ do Fujita para ~1.8 GB e exigindo amostragem em lotes (`batch_size=1024/2048`) no `retrieve` para evitar *Out of Memory (OOM)*.
