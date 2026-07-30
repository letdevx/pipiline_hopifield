---
tipo: conceito
tags: [conceito, sentinela, imputacao, incerteza, hopfield]
criado: 2026-07-30
atualizado: 2026-07-30
resumo: "Conceito atômico do uso do valor sentinela neutro 0.5 para representar incerteza biológica em genes não observados."
---

# Sentinela Neutra 0.5

## Definição
A **sentinela neutra 0.5** é a atribuição do valor constante $0.5$ para posições de genes em uma matriz binária $\{0, 1\}$ onde a informação transcricional original é desconhecida ou ausente no sequenciamento.

## Propriedade Matemática em Redes Hopfield
Na computação do produto interno $\xi \cdot \Xi^T$ entre o vetor de busca $\xi$ e os protótipos de memória $\Xi$:
- Se um gene possui valor $0.5$, seu produto com o protótipo ($1$ ou $0$) resulta em $0.5 \times 1 = 0.5$ ou $0.5 \times 0 = 0$.
- O valor $0.5$ atua como uma contribuição de peso constante e idêntica para todas as memórias salvas, cancelando seu efeito na diferenciação de similaridade.
- A decisão de recuperar a classe depende exclusivamente dos genes de fato observados ($0$ ou $1$).

## Conexões
- Área: [[02_Areas/rnaseq_single_cell/index|RNA-Seq Single Cell]]
- Decisão Formada: [[04_Recursos/adrs/adr_002_sentinela_meio_genes_ausentes|ADR 002: Sentinela Neutra 0.5]]
- Conceito Relacionado: [[03_Conhecimento/atencao_softmax_hopfield|Atenção Softmax Hopfield]]
