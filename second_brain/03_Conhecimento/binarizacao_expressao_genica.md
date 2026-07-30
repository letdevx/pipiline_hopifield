---
tipo: conceito
tags: [conceito, binarizacao, scrnaseq, pre-processamento]
criado: 2026-07-30
atualizado: 2026-07-30
resumo: "Conceito atômico da binarização de matrizes de expressão gênica para isolamento de programas transcricionais ON/OFF."
---

# Binarização de Expressão Gênica

## Definição
A **binarização de expressão gênica** é o processo de conversão de uma matriz de contagens contínuas de transcriptômica single-cell (scRNA-seq) em um domínio booleano $\{0, 1\}$:

$$x_{i,j} = \begin{cases} 1 & \text{se } \text{expressão}_{i,j} > 0 \\ 0 & \text{se } \text{expressão}_{i,j} \le 0 \end{cases}$$

## Motivação Biológica e Computacional
1. **Remoção do Ruído de Dropout:** Dados de scRNA-seq contêm altíssima taxa de zeros estocásticos. A binarização remove a dependência da magnitude contínua e foca na ativação de redes gênicas.
2. **Estabilidade em Memória Associativa:** Em redes Hopfield, produtos escalares sobre vetores binários evitam desequilíbrios causados por poucos genes super-expressos.

## Conexões
- Área: [[02_Areas/rnaseq_single_cell/index|RNA-Seq Single Cell]]
- Decisão Formada: [[04_Recursos/adrs/adr_001_binarizacao_expressao_genica|ADR 001: Binarização]]
- Aplicação: [[01_Projetos/pipeline_hopfield_expandido/documentacao_pipeline_hopfield|Pipeline Hopfield Expandido]]
