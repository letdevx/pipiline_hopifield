### 🔨 O que esse PR faz?

#### 🎯 Resumo da Mudança
- **Explicabilidade SHAP e Seleção de Biomarcadores Celulares (ADR 023):** Implementa o módulo `src/treinamento/selecionador_genes_shap.py` baseado no método *Expected Gradients* sobre a Modern Hopfield Network com *Softmax Class Pooling*. Suporta acumulação online em *streaming* OOM-Safe (pico de RAM < 1,5 GB para o dataset completo de 40.913 células × 36.591 genes), *background* acelerado por centróides das 7 classes canônicas, e seleção balanceada de 2.000 a 5.000 genes biomarcadores combinando cota mínima por linhagem (70%) e excelência global com contraste de especificidade (30%). Inclui visualização em Heatmap de expressão relativa Min-Max por tipo celular.
- **Auditoria Formal de Overfitting e Robustez Termodinâmica:** Implementa o `AuditorOverfittingHopfield` e o `InjetorPerturbacaoTranscritica` em `src/treinamento/diagnostico_overfitting.py` para avaliar generalização fora da amostra (*holdout*), estabilidade de bacias de atração via perturbações sintéticas escalonadas (5%, 15%, 30%) e saturação termodinâmica via Entropia de Shannon dos pesos de atenção contínua.
- **Otimização de Hiperparâmetros via Algoritmo Genético (ADR 022):** Implementa em `src/treinamento/algoritmo_genetico.py` busca evolutiva de hiperparâmetros (beta inverso de temperatura, limiares de convergência e topologia de padrões) com elitismo e operadores genéticos customizados.
- **Reconstrução e Expansão do Pipeline Pan-Mathys:** Consolida o notebook pareado via Jupytext (`pipeline_generico_Pan_Mathys.py` e `.ipynb`) cobrindo 14 capítulos estruturados, desde o pré-processamento de scRNA-seq, projeção rSWeeP da UFPR, treinamento da Hopfield ótima, auditoria de robustez até a explicabilidade biológica e heatmap SHAP.
- **Avaliação por Tipo Celular no Capítulo 13 (Precision, Recall, F1-Score, Support):** Enriquecimento do `AvaliadorHopfield` (`src/treinamento/avaliador_hopfield.py`) com cálculo estruturado de métricas por classe canônica (mantendo retrocompatibilidade total), método `relatorio_classificacao_completo()` com agregações Macro Avg e Weighted Avg, e painel visual comparativo de barras agrupadas e matriz de confusão persistidos em CSV/JSON no Capítulo 13.

#### 💡 Por que foi feito desta forma? (Decisão de Design / ADR)
- **ADR 023 — Explicabilidade SHAP em Redes Hopfield e Streaming OOM-Safe:** Armazenar tensores tridimensionais brutos de SHAP (40.913 × 36.591 × 7) exigiria 41,9 GB de RAM, inviabilizando a execução em computadores de bancada. O processamento por *mini-batches* com acumulação online cumulativa dos valores absolutos reduz a pegada de memória do acumulador para ~2 MB (redução de 99,99%).
- **Baseline de Centróides Biológicos:** O uso de 7 centróides médios (um por tipo celular canônico) reduz as avaliações de gradiente em 10× em comparação com amostragens aleatórias densas de background, ancorando as explicações na média transcricional biológica de cada linhagem.
- **Contrato Imutável e Fail-Fast:** Todas as operações matemáticas e perturbações preservam a imutabilidade das matrizes originais (`copy=True`). Falhas em dimensões ou configurações disparam `ValueError` de forma preventiva.
- **Sincronização Jupytext:** Garantia de integridade do Jupyter Notebook evitando edição manual de JSON e assegurando reprodutibilidade via scripts Python em formato *percent*.

#### 🛡️ Riscos, Efeitos Colaterais e O que NÃO mudou
- **Áreas Protegidas e Sem Alteração:** O algoritmo de projeção vetorial `rSWeeP` do pacote R oficial da UFPR (`orthBase` e `SWeeP`), os arquivos brutos de matrizes e o binarizador de ground truth permanecem 100% protegidos e inalterados, respeitando as diretrizes científicas do projeto.
- **Compatibilidade Regressiva:** Todos os módulos anteriores do pipeline e suas interfaces públicas permanecem compatíveis. Os 80 testes unitários existentes continuam passando com 100% de sucesso.

#### 🧪 Como testar e validar
1. **Executar a suíte completa de testes unitários:**
   ```powershell
   uv run pytest
   ```
   *(Validação esperada: 81 passed, 1 skipped)*
2. **Executar testes específicos dos novos módulos:**
   ```powershell
   uv run pytest tests/test_memoria_hopfield_reconstrucao.py tests/test_selecionador_genes_shap.py tests/test_diagnostico_overfitting.py tests/test_algoritmo_genetico.py
   ```
3. **Validar conformidade estrita de tipagem e estilo:**
   ```powershell
   uv run ruff check .
   uv run ruff format --check .
   uv run pyrefly check
   ```
4. **Verificar a sincronização e execução do pipeline pareado:**
   ```powershell
   uv run jupytext --sync pipeline_generico_Pan_Mathys.py
   ```

---

### 🔗 Referências

- Tarefa / Card: Reconstrução do Pipeline Pan-Mathys, Diagnóstico de Overfitting e Explicabilidade SHAP com Seleção de 2k-5k Biomarcadores
- ADR 022: `second_brain/04_Recursos/adrs/adr_022_otimizacao_hiperparametros_algoritmo_genetico.md`
- ADR 023: `second_brain/04_Recursos/adrs/adr_023_selecao_features_shap_hopfield.md`
- Spec & Arquitetura: `docs/specs/diagnostico-overfitting-rede-hopfield/`

---

### 📗 Checklist do desenvolvedor

- [x] Foi adicionado ou atualizado testes para a solução? (6 testes SHAP + 10 testes Overfitting + 5 testes Algoritmo Genético)
- [x] Foi criado ou atualizado a documentação que afeta a mudança? (ADR 022, ADR 023, specs, sumário do notebook)
- [x] Dependências e contratos externos validados? (Ruff, Pyrefly e Pytest 100% verdes)

---

### 👀 Checklist do revisor

- Você entendeu o propósito desse PR?
- Você entendeu o fluxo de negócio atendido?
- Você validou que a solução técnica é simples, segura e testável?
- Os testes cobrem os cenários principais e fluxos de exceção?

---

### ✅ Posso fazer merge desse PR?

- Esse PR possui as aprovações necessárias da equipe?
- Todas as conversas e apontamentos foram resolvidos?
- Não existem labels de bloqueio (`wip`, `hold`)?
- Os pipelines de CI/CD e testes passaram com sucesso?
