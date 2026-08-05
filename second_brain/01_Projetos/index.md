---
tipo: indice
tags: [projetos, indice, bioinformatica, mestrado]
criado: 2026-07-30
atualizado: 2026-08-04
resumo: "Central de controle para projetos de pesquisa ativos com código e entregáveis."
---

# 🚀 Índice de Projetos Científicos

Esta pasta abriga todos os projetos de pesquisa com metas ativas, entregáveis e código-fonte no repositório.

---

## 🔬 Projetos Ativos

* 🎓 **[[01_Projetos/proposta_mestrado/index|Projeto de Mestrado UFPR]]**
  * Proposta oficial da pesquisa de mestrado (Leticia Astolpho Silvano, UFPR): integração de dados de scRNA-seq, mitigação de efeito de lote e recuperação de perfis via Projeções SWeeP e Redes Hopfield Modernas.
* 🧬 **[[01_Projetos/pipeline_hopfield_expandido/index|Pipeline Hopfield Expandido (~11.000 Genes)]]**
  * Projeto principal de alinhamento cross-dataset (Fujita vs Mathys), projeção rSWeeP 600D, imputação via Modern Hopfield Network com sentinela $0.5$ e validação em 7 classes celulares.

---

## 📋 Diretrizes para Gerenciamento de Projetos

1. Toda pasta de projeto deve conter um `index.md` interno com `tipo: indice` ou `tipo: projeto`.
2. As notas internas do projeto devem apontar para os conceitos atômicos em `03_Conhecimento/` e para os ADRs em `04_Recursos/adrs/`.
3. Ao concluir um projeto, mova a pasta para `05_Arquivos/` e atualize a tag para `[arquivado]`.
