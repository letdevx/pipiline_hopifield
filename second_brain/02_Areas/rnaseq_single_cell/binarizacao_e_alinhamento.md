---
tipo: area
tags: [area, scrnaseq, binarizacao, alinhamento]
criado: 2026-07-30
atualizado: 2026-07-30
resumo: "Nota técnica sobre binarização (>0 -> 1), alinhamento Ensembl e sentinelas cross-dataset em scRNA-seq."
---

# Nota Técnica: Binarização e Alinhamento Canônico em scRNA-Seq

> **Área:** RNA-Seq Single Cell (scRNA-Seq)  
> **Idioma:** PT-BR  

---

## 1. Binarização de Expressão Gênica ($> 0 \rightarrow 1$)

No processamento de dados de RNA-Seq de célula única, as matrizes de contagens puras ou valores normalizados (ex: CPM, TPM, log-normalize) enfrentam severas limitações em algoritmos de memória associativa devido ao ruído estocástico de *dropout* e variações de profundidade de sequenciamento. Veja o **[[03_Conhecimento/binarizacao_expressao_genica|Conceito Atômico de Binarização]]** e a justificativa em **[[04_Recursos/adrs/adr_001_binarizacao_expressao_genica|ADR 001]]**.

Ao transformar a matriz para um estado discreto $\{0, 1\}$ ($x > 0 \Rightarrow 1$), focamos o modelo no **estado computacional ON/OFF dos circuitos gênicos**, eliminando artefatos quantitativos.

---

## 2. Alinhamento de Espaços Gênicos Cross-Dataset (Ensembl ID)

Diferentes estudos de scRNA-seq costumam utilizar versões distintas da anotação do genoma humano (ex: GRCh37/hg19 vs GRCh38/hg38) ou aplicar filtros de qualidade de genes de forma independente. Veja o **[[03_Conhecimento/alinhamento_ensembl_cross_dataset|Conceito Atômico de Alinhamento]]**.

### Fluxo de Alinhamento Canônico:
1. **Padronização de IDs:** Mapeamento de `gene_name` (ex: *CD3D*, *GFAP*) para `ensembl_id` estável (ex: *ENSG00000167286*).
2. **Ordem Canônica do Dataset de Referência:** O dataset Fujita (com 36.591 genes) é estabelecido como a ordem canônica dos índices.
3. **Estratégia de Inserção de Sentinela ($0.5$):**
   - Para o dataset Mathys, genes nativos presentes no Fujita mas ausentes no Mathys são preenchidos com o **[[03_Conhecimento/sentinela_meio_genes_ausentes|valor sentinela neutro 0.5]]**. Veja o **[[04_Recursos/adrs/adr_002_sentinela_meio_genes_ausentes|ADR 002]]**.
   - O valor $0.5$ atua como o ponto médio neutro no produto interno da atenção Softmax Hopfield, permitindo que os genes conhecidos da célula definam a classe e recuperem o padrão correto para os genes ausentes.
