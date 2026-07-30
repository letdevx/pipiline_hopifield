---
tipo: adr
tags: [adr, rsweep, kmeans, prototipos, decisao-arquitetura]
criado: 2026-07-30
atualizado: 2026-07-30
resumo: "ADR 004: Decisão de usar a redução rSWeeP 600D para viabilizar a clusterização K-Means e amostragem de protótipos."
---

# ADR 004: Projeção rSWeeP 600D para Clusterização K-Means e Extração de Protótipos

> **Status:** Aceito  
> **Data:** 30/07/2026  
> **Decisores:** Equipe de Bioinformática & Agente AI  

---

## 1. Contexto

Extrair protótipos representativos executando K-Means diretamente na matriz binária de 40.000 células por ~11.000 genes seria proibitivamente lento e computacionalmente inviável para iterações de teste.

## 2. Decisão

Projetar a matriz binária expandida no espaço vetorial compacto de **600 dimensões** usando a transformação de base ortonormal **rSWeeP** ($W_0 \times R_{\text{expandido}}$). A clusterização K-Means é realizada no espaço SWeeP 600D, e os centroides são convertidos de volta para o vetor binário real mais próximo no espaço original de 11.000 genes via distância euclidiana (`closervects`).

Veja os conceitos **[[03_Conhecimento/projecao_rsweep_600d|Projeção rSWeeP 600D]]** e **[[03_Conhecimento/amostragem_prototipos_kmeans|Amostragem de Protótipos via K-Means]]**.

## 3. Consequências Biológicas

- **Vantagens:** O espaço rSWeeP preserva a topologia e a distância entre os perfis transcriptômicos biológicos, permitindo identificar agrupamentos biológicos coerentes mesmo em menor dimensão.

## 4. Consequências Técnicas

- **Desempenho:** Reduz o custo computacional do K-Means em mais de 18 vezes em relação ao espaço de 11k genes, consumindo apenas ~96 MB de RAM para o armazenamento das projeções $W_{\text{swp}}$.
