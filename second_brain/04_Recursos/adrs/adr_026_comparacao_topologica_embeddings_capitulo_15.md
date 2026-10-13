---
tipo: adr
tags: [adr, pca, umap, tsne, capitulo-15, topologia, rsweep, decisao-arquitetura]
criado: 2026-10-13
atualizado: 2026-10-13
resumo: "ADR 026: Decisão de arquitetura para a comparação dimensional e topológica do Capítulo 15 via Grid 3x3 (PCA, UMAP, t-SNE) no espaço rSWeeP 600D com ajuste padronizado na referência."
---

# ADR 026: Comparação Topológica e Visualização Dimensional Cross-Dataset (Capítulo 15)

> **Status:** Aceito  
> **Data:** 13/10/2026  
> **Decisores:** Pesquisadora e Agente AI Antigravity  

---

## 1. Contexto

Após a etapa de imputação de genes ausentes no conjunto alvo (Mathys) pela Modern Hopfield Network (Capítulo 13) e a caracterização explicável dos biomarcadores celulares via SHAP (Capítulo 14), torna-se imperativo diagnosticar visual e geometricamente se a recuperação associativa foi capaz de restaurar a estrutura topológica das 7 linhagens celulares canônicas do cérebro humano.

O desafio central residia em confrontar três condições experimentais:
1. **Referência Pan (Fujita):** 40.913 células com o genoma canônico completo;
2. **Alvo Original Mathys:** 47.523 células onde ~6.305 genes ausentes foram inicializados com a sentinela neutra 0.5;
3. **Alvo Pós-Imputação Mathys:** 47.523 células com a sentinela neutra substituída pelos valores preditos pela rede Hopfield.

Projetar diretamente as matrizes brutas de expressão gênica densificadas (88.436 células × 36.591 genes) em algoritmos não lineares como t-SNE e UMAP exigiria dezenas de gigabytes de RAM e causaria saturação em hardware modesto (*Out Of Memory*).

---

## 2. Decisão

Adotar a arquitetura implementada em `ComparadorEmbeddings` no **Capítulo 15** do pipeline:

1. **Espaço Vetorial rSWeeP 600D:**
   Todas as reduções dimensionais (PCA, UMAP e t-SNE) operam estritamente sobre as projeções geradas a partir da mesma base ortonormal congelada (`orthBase()`), assegurando que as três matrizes compartilhem o mesmo espaço dimensional compacto de 600 variáveis contínuas.

2. **Formato Visual em Grid 3 × 3:**
   * **Linhas:** PCA (PC1 vs PC2), UMAP (UMAP1 vs UMAP2) e t-SNE (t-SNE1 vs t-SNE2).
   * **Colunas:** Referência Pan, Alvo Original (com Sentinela 0.5) e Alvo Pós-Imputação Hopfield.
   * **Coloração:** Rótulos das 7 linhagens celulares canônicas com a mesma paleta em todos os subplots.

3. **Padronização Geométrica dos Eixos:**
   * **PCA e UMAP:** Ajustados (*fit*) sobre a matriz de Referência e aplicados (*transform*) tanto no Alvo Original quanto no Alvo Pós-Imputação, garantindo identidade estrita de autovetores e vizinhanças.
   * **t-SNE:** Ajuste conjunto (*co-fit*) cobrindo integralmente 100% das células (~135k amostras empilhadas) com pré-redução linear via PCA 50D, desmembrando as coordenadas 2D nas três colunas.

4. **Métricas Quantitativas de Agrupamento Biológico:**
   Cálculo do *Silhouette Score* com subamostragem determinística (`sample_size=5000`) para cada método e condição, mensurando matematicamente o ganho de coerência pós-imputação.

5. **Idempotência e Cache de Saída:**
   Persistência em `outputs/07_embeddings_comparativos/` dos arquivos `coordenadas_grid_pca_umap_tsne.csv`, `metricas_separacao_clusters_cap15.csv` e `grid_3x3_pca_umap_tsne_referencia_vs_alvos.png`, viabilizando carregamento instantâneo caso `FORCAR_RECALCULO_CAPITULO_15 = False`.

---

## 3. Consequências Biológicas

* **Validação de Fidelidade Fenotípica:** Permite verificar diretamente se a imputação aproximou os perfis de astrócitos, neurônios e micróglia do Mathys das regiões biológicas homólogas da Referência Pan.
* **Detecção de Artefatos de Sentinela:** Evidencia o impacto da remoção do ruído artificial de 0.5 que dispersava as células do Mathys no espaço pré-Hopfield.

---

## 4. Consequências Técnicas

* **Consumo de Memória Reduzido:** As três matrizes 600D somadas ocupam apenas ~326 MB em RAM (float32), rodando perfeitamente sem risco de OOM.
* **Convergência Otimizada:** O pré-PCA de 50 dimensões reduz em mais de 10 vezes o tempo de cálculo das probabilidades afins no t-SNE.
