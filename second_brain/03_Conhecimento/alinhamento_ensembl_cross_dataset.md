---
tipo: conceito
tags: [conceito, alinhamento, ensembl, cross-dataset, bioinformatica]
criado: 2026-07-30
atualizado: 2026-07-30
resumo: "Conceito atômico de alinhamento canônico de genomas entre diferentes coortes scRNA-seq usando Ensembl ID."
---

# Alinhamento Canônico Ensembl

## Definição
O **alinhamento canônico Ensembl** é o procedimento de harmonização de coordenadas de colunas em matrizes de expressão vindas de estudos independentes, utilizando os identificadores estáveis do Ensembl (*ENSG...*) como chave primária de ordenação.

## Princípio de Funcionamento
1. Tradução de nomes de genes (*gene symbols*) para IDs Ensembl.
2. Definição da ordem de genes do dataset de referência (Fujita) como o índice global canônico.
3. Re-ordenação da matriz do dataset secundário (Mathys) para coincidir célula a célula na mesma posição de coluna.

## Conexões
- Área: [[02_Areas/rnaseq_single_cell/index|RNA-Seq Single Cell]]
- Conceito Relacionado: [[03_Conhecimento/sentinela_meio_genes_ausentes|Sentinela Neutra 0.5]]
- Projeto: [[01_Projetos/pipeline_hopfield_expandido/documentacao_pipeline_hopfield|Pipeline Hopfield Expandido]]
