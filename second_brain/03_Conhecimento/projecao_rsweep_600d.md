---
tipo: conceito
tags: [conceito, rsweep, sweep, embeddings, reducao-dimensionalidade, aibialab, ufpr]
criado: 2026-07-30
atualizado: 2026-09-01
resumo: "Conceito atômico da técnica de redução de dimensionalidade SWeeP/rSWeeP via matriz de projeção ortonormal periódica gerada pelo pacote R oficial da UFPR."
---

# Projeção rSWeeP 600D

## Definição
A **Projeção rSWeeP 600D** (*Spaced Words Projection*) é uma técnica de redução dimensional desenvolvida pelo laboratório AIBIALab/UFPR (De Pierri et al., 2020) e disponibilizada no pacote R `rSWeeP`. O algoritmo mapeia a matriz de expressão gênica (composta por milhares de genes) para um espaço latente ortogonal e compacto de 600 dimensões:

`Wswp = W0 × R_base (matriz de células × 600 dimensões)`

Ao contrário de projeções Gaussianas genéricas, a base ortonormal canônica do `rSWeeP` é gerada pela função `orthBase(lin = n_genes, col = 600, seed = seed)`, que emprega dispersão quase-periódica sobre 50 números primos (`idx %% pslist`) combinada com uma matriz de projeção latente e normalização no intervalo `[-1, 1]`.

## Propriedades-Chave
1. **Preservação Geométrica e Topológica:** Fundamentada no Lema de Johnson-Lindenstrauss, a projeção mantém com alta fidelidade as distâncias relativas e a separabilidade biológica entre tipos celulares.
2. **Execução Canônica R:** Executada estritamente via o pacote R oficial `rSWeeP` através do par `orthBase()` e `SWeeP()`, sem o uso de fallbacks ou aproximações em Python (conforme diretriz mandatória).
3. **Congelamento da Base Ortonormal Canônica:** A base gerada é salva em formato binário R (`.rds`), centralizada pelo caminho `src.config.PATH_ORTHBASE_RDS` (`orthbase_mproj_600d.rds`), assegurando que conjuntos de dados de referência (Fujita), alvos mascarados (Mathys com sentinela) e imputados compartilhem exatamente o mesmo subespaço ortogonal (ADR 018, ADR 019 e ADR 021).
4. **Resolução e Consumo Padrão em Python:** O `ProjetorSWeePR` consome automaticamente a base canônica do `config.py` como padrão, dispensando parametrizações manuais redundantes nos notebooks e suportando `forcar_recriacao=True` quando necessário.
5. **Eficiência Computacional em Espaço Esparso:** Reduz o custo da clusterização K-Means em mais de 18 vezes comparado à execução no espaço gênico bruto de 36.591 genes.

## Conexões
- Recursos: [[04_Recursos/projecao_sweep/documentacao_oficial_sweep|Documentação Oficial rSWeeP]], [[04_Recursos/projecao_sweep/projeao_rsweep|Recurso Projeção rSWeeP]] e [[04_Recursos/projecao_sweep/guia_orthbase_canonica_config|Guia Técnico: OrthBase Canônica SWeeP]]
- Decisões de Arquitetura: [[04_Recursos/adrs/adr_004_projecao_rsweep_600d_kmeans|ADR 004: Projeção rSWeeP 600D e K-Means]], [[04_Recursos/adrs/adr_019_obrigatoriedade_rsweep_r_e_congelamento_orthbase|ADR 019: Obrigatoriedade Irrevogável do rSWeeP e Eliminação de Fallbacks]] e [[04_Recursos/adrs/adr_021_centralizacao_orthbase_config_e_reuso_canonico|ADR 021: Centralização da OrthBase via config.py e Reuso Canônico Padrão]]
- Conceito Relacionado: [[03_Conhecimento/amostragem_prototipos_kmeans|Amostragem de Protótipos via K-Means]]
