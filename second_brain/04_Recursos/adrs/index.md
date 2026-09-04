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
* 📜 **[[04_Recursos/adrs/adr_006_selecao_diferencial_genes_chi2|ADR 006: Seleção Diferencial por Qui-Quadrado (Chi2 / Ganho de Informação)]]**
* 📜 **[[04_Recursos/adrs/adr_007_pipeline_genoma_completo_36k_esparso|ADR 007: Pipeline com Genoma Completo (36.591 genes) e Serialização Esparsa H5AD]]**
* 📜 **[[04_Recursos/adrs/adr_008_calibracao_temperatura_consenso_hopfield|ADR 008: Calibração da Temperatura ($\beta$) e Suavização de Consenso em Redes Hopfield Modernas]]**
* 📜 **[[04_Recursos/adrs/adr_009_otimizacao_granularidade_subclusters_nc|ADR 009: Otimização Empírica da Granularidade de Protótipos ($nc$) e Capacidade Associativa]]**
* 📜 **[[04_Recursos/adrs/adr_010_harmonizacao_cosseno_e_prototipos_consolidados|ADR 010: Harmonização por Similaridade de Cosseno, Protótipos Consolidados ($k>1$) e Otimização Trans-Dataset]]**
* 📜 **[[04_Recursos/adrs/adr_011_desacoplamento_atencao_subespaco_e_imputacao_expandida|ADR 011: Desacoplamento entre Atenção em Subespaço Compartilhado e Imputação no Espaço Expandido]]**
* 📜 **[[04_Recursos/adrs/adr_012_experimento_amostragem_consenso_k5_210padroes_v3|ADR 012: Experimento com Amostragem por Consenso ($k=5$) e 210 Padrões no Pipeline v3]]**
* 📜 **[[04_Recursos/adrs/adr_013_validacao_estrita_features_pre_alinhamento|ADR 013: Validação Estrita de Features e Identificadores Genômicos Pré-Alinhamento]]**
* 📜 **[[04_Recursos/adrs/adr_014_adocao_estrita_type_hints_docstrings_pyrefly|ADR 014: Adoção Estrita de Type Hints e Docstrings no Padrão NumPy via Pyrefly]]**
* 📜 **[[04_Recursos/adrs/adr_015_adocao_ruff_linter_formatter_automacao_ide|ADR 015: Adoção do Ruff como Linter/Formatador Unificado e Automação de Salvamento no IDE]]**
* 📜 **[[04_Recursos/adrs/adr_016_migracao_gerenciador_pacotes_uv|ADR 016: Migração do Gerenciador de Pacotes para uv e Travamento Determinístico com uv.lock]]**
* 📜 **[[04_Recursos/adrs/adr_017_exportador_anndata_imputacao_cross_dataset|ADR 017: Exportação Estruturada de Imputação Cross-Dataset em AnnData OOM-Safe com Camadas de Rastreabilidade]]**
* 📜 **[[04_Recursos/adrs/adr_018_validacao_ordem_genes_exportacao_mtx|ADR 018: Validação Estrita de Ordem Gênica, Exportação Padronizada Matrix Market (.mtx) e Congelamento da Base Ortonormal SWeeP]]**
* 📜 **[[04_Recursos/adrs/adr_019_obrigatoriedade_rsweep_r_e_congelamento_orthbase|ADR 019: Obrigatoriedade Irrevogável do Algoritmo rSWeeP em R, Execução Canônica via orthBase() + SWeeP() e Eliminação de Fallbacks]]**
* 📜 **[[04_Recursos/adrs/adr_020_resolucao_sentinela_e_validacao_multinivel_imputacao|ADR 020: Resolução da Máscara Sentinela, Camada de Confiança Contínua e Validação Multinível da Imputação]]**
* 📜 **[[04_Recursos/adrs/adr_021_centralizacao_orthbase_config_e_reuso_canonico|ADR 021: Centralização da OrthBase via config.py, Resolução Dinâmica de Ambiente e Reuso Canônico Padrão no Projetor SWeeP]]**

---

## 📋 Estrutura Padrão de um ADR

Todo arquivo ADR deve ser registrado como `tipo: adr` no YAML OKF e conter as seguintes seções:
1. **Status** (Aceito / Proposto / Rejeitado / Substituído).
2. **Contexto** (Problema enfrentado).
3. **Decisão** (Escolha efetuada).
4. **Consequências Biológicas** (Impacto no sinal e interpretação biológica).
5. **Consequências Técnicas** (Impacto em desempenho, RAM e código).
