---
tipo: indice
tags: [projeto, hopfield, scrnaseq, indice]
criado: 2026-07-30
atualizado: 2026-07-30
resumo: "Índice do projeto Pipeline Hopfield Expandido (~11.000 genes)."
---

# 🧬 Projeto: Pipeline Hopfield Expandido (~11.000 Genes)

Central de documentação, arquitetura e notas do projeto de alinhamento e imputação cross-dataset scRNA-seq.

---

## 📑 Documentação e Arquitetura do Projeto

* 📘 **[[01_Projetos/pipeline_hopfield_expandido/documentacao_pipeline_hopfield|Documentação Mestre do Pipeline]]**
  * Guia de ponta a ponta com diagrama Mermaid, explicação detalhada das 18 seções do notebook, intuição biológica, técnica e gargalos de RAM/tempo.
* 📐 **[[01_Projetos/pipeline_hopfield_expandido/arquitetura_do_sistema|Documento de Arquitetura do Sistema]]**
  * Especificação dos módulos em `src/`, fluxo de dados, transformações de matrizes e perfil de consumo de memória.
* 🧪 **[[01_Projetos/pipeline_hopfield_expandido/plano_auditoria_conceitual_tecnica|Plano de Auditoria Conceitual e Técnica (Synthetic Ground Truth)]]**
  * Metodologia rigorosa de testes unitários (`pytest`), micro-datasets humano-verificáveis e provas reais de invariância matemática em cada etapa.

---

## 🔗 Conceitos Atômicos e ADRs Relacionados

* 🧠 **[[03_Conhecimento/atencao_softmax_hopfield|Atenção Softmax Hopfield]]**
* 🧬 **[[03_Conhecimento/binarizacao_expressao_genica|Binarização de Expressão Gênica]]**
* 📊 **[[03_Conhecimento/projecao_rsweep_600d|Projeção rSWeeP 600D]]**
* 🎯 **[[03_Conhecimento/sentinela_meio_genes_ausentes|Sentinela Neutra 0.5]]**
* 🏛️ **[[04_Recursos/adrs/index|Decisões formais de Arquitetura (ADRs 001 a 005)]]**
