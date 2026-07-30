---
tipo: indice
tags: [conhecimento, conceitos, teorias, indice]
criado: 2026-07-30
atualizado: 2026-07-30
resumo: "Catálogo central de conceitos atômicos, modelos mentais e princípios de bioinformática e ML."
---

# 📚 Índice de Conceitos Atômicos (Conhecimento)

Esta pasta abriga os **conceitos atômicos do conhecimento científico** — princípios teóricos, métodos matemáticos e modelos fundamentais sintetizados de forma independente e reutilizável em múltiplos projetos.

---

## 🧬 Mapeamento por Grandes Temas

### 🔬 Processamento de RNA-Seq Single Cell (scRNA-Seq)
* 🟢 **[[03_Conhecimento/binarizacao_expressao_genica|Binarização de Expressão Gênica]]** — Filtragem de ruído de amostragem e estado ON/OFF ($>0 \Rightarrow 1$).
* 🧬 **[[03_Conhecimento/alinhamento_ensembl_cross_dataset|Alinhamento Canônico Ensembl]]** — Harmonização de genomas e ordenação canônica cross-dataset.
* 🎯 **[[03_Conhecimento/sentinela_meio_genes_ausentes|Sentinela Neutra 0.5]]** — Representação de incerteza biológica para genes ausentes.

### 🧠 Redes Neurais & Memória Associativa
* ⚡ **[[03_Conhecimento/atencao_softmax_hopfield|Atenção Softmax Hopfield]]** — Atualização contínua de estados em Redes Hopfield Modernas.
* 🎯 **[[03_Conhecimento/amostragem_prototipos_kmeans|Amostragem de Protótipos via K-Means]]** — Seleção de células representativas no espaço reduzido.

### 📊 Redução de Dimensionalidade & Embeddings
* 📐 **[[03_Conhecimento/projecao_rsweep_600d|Projeção rSWeeP 600D]]** — Compressão vetorial via matriz de base ortonormal aleatória.

---

## 📋 Diretrizes para Criar Notas de Conceito

1. **Princípio da Atomicidade:** Cada nota deve tratar de **uma única ideia principal**.
2. **Cabeçalho YAML:** Todo conceito deve usar `tipo: conceito`.
3. **Rede de Links:** Vincule a nota aos projetos em `01_Projetos/`, áreas em `02_Areas/` e decisões em `04_Recursos/adrs/`.
