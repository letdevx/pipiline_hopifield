---
tipo: indice
tags: [adrs, arquitetura, decisoes, indice]
criado: 2026-07-30
atualizado: 2026-07-30
resumo: "Índice dos registros formais de decisões de arquitetura (ADRs) do projeto."
---

# 🏛️ Índice de Architecture Decision Records (ADRs)

Esta pasta armazena o repositório de decisões formais de arquitetura para garantir a rastreabilidade do raciocínio técnico e biológico do projeto.

---

## 📋 Catálogo de ADRs Registrados

* 📜 **[[04_Recursos/adrs/adr_001_binarizacao_expressao_genica|ADR 001: Binarização de Expressão Gênica Single-Cell ($>0 \rightarrow 1$)]]**
* 📜 **[[04_Recursos/adrs/adr_002_sentinela_meio_genes_ausentes|ADR 002: Sentinela Neutra 0.5 para Genes Ausentes no Alinhamento]]**
* 📜 **[[04_Recursos/adrs/adr_003_expansao_espaco_genico_11k|ADR 003: Expansão do Espaço Gênico para ~11.000 Genes]]**
* 📜 **[[04_Recursos/adrs/adr_004_projecao_rsweep_600d_kmeans|ADR 004: Projeção rSWeeP 600D para Clusterização K-Means]]**
* 📜 **[[04_Recursos/adrs/adr_005_rede_hopfield_moderna_parametros|ADR 005: Hiperparâmetros da Modern Hopfield Network ($\beta=50.0$, $nc=30$)]]**

---

## 📋 Estrutura Padrão de um ADR

Todo arquivo ADR deve ser registrado como `tipo: adr` no YAML OKF e conter as seguintes seções:
1. **Status** (Aceito / Proposto / Rejeitado / Substituído).
2. **Contexto** (Problema enfrentado).
3. **Decisão** (Escolha efetuada).
4. **Consequências Biológicas** (Impacto no sinal e interpretação biológica).
5. **Consequências Técnicas** (Impacto em desempenho, RAM e código).
