---
tipo: indice
tags: [projeto, indice, mlp, sweep, ruido, benchmark]
criado: 2026-10-24
atualizado: 2026-10-24
resumo: "Índice do projeto de validação de sensibilidade ao ruído (sentinela 0.5) em classificadores MLP com projeções SWeeP 600D."
---

# 🧬 Projeto: Validação de Sensibilidade ao Ruído (MLP + SWeeP 600D)

Central de documentação do experimento de comprovação de sensibilidade ao ruído transcricional (sentinela 0.5) em classificadores celulares baseados em Multi-Layer Perceptron (MLP) utilizando representações vetoriais rSWeeP 600D.

---

## 📑 Documentação e Planos

* 📊 **[[01_Projetos/validacao_ruido_mlp_sweep/relatorio_resultados_ruido_mlp|Relatório de Resultados Empíricos: Impacto do Ruído no MLP]]**
  * Síntese quantitativa consolidada dos 6 cenários experimentais, demonstrando quedas de até **-21,40 p.p. na acurácia** e **-44,53 p.p. no Macro F1**, fundamentando a necessidade da Rede Hopfield Moderna para reconstrução gênica.
* 📘 **[[01_Projetos/validacao_ruido_mlp_sweep/plano_experimentos_ruido_mlp_sweep|Plano Experimental Completo: Validação de Ruído MLP + SWeeP]]**
  * Especificação detalhada dos 6 experimentos (Fujita->Fujita, Fujita->Mathys, Mathys->Mathys, Pan->Pan, Pan->Fujita, Pan->Mathys), dimensões exatas de genes, alinhamento Ensembl, travas de segurança de ortobase e topologia de execução paralela em Git Worktrees.

---

## 🔗 Conexões no Grafo de Conhecimento

* 🎓 **[[01_Projetos/proposta_mestrado/index|Projeto de Mestrado UFPR]]**
* 🧬 **[[01_Projetos/pipeline_hopfield_expandido/index|Pipeline Hopfield Expandido]]**
* 🎯 **[[03_Conhecimento/sentinela_meio_genes_ausentes|Sentinela Neutra 0.5]]**
* 📊 **[[03_Conhecimento/projecao_rsweep_600d|Projeção rSWeeP 600D]]**
* 🧠 **[[03_Conhecimento/atencao_softmax_hopfield|Atenção Softmax Hopfield]]**
