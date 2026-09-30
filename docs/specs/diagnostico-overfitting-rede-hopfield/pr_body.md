### 🔨 O que esse PR faz?

#### 🎯 Resumo da Mudança
- **Diagnóstico Formal de Overfitting:** Implementa o protocolo automatizado de avaliação de generalização fora da amostra (*holdout*), estabilidade de bacias de atração e detecção de saturação termodinâmica para a Rede Hopfield Moderna em scRNA-seq.
- **Injeção de Perturbações e Bacias de Atração:** Cria o utilitário `InjetorPerturbacaoTranscritica` para testes de estresse com perturbações sintéticas escalonadas (5%, 15%, 30% de *dropout* e *bit-flip*) sem mutação *in-place*.
- **Auditoria de Atenção Softmax e Quimeras:** Avalia a Entropia de Shannon normalizada dos pesos de atenção contínua para prevenir degeneração para tabela de busca 1-NN rígida e detecta estados espúrios com coativação de marcadores celulares antagônicos.

#### 💡 Por que foi feito desta forma? (Decisão de Design / ADR)
- **Isolamento de Responsabilidades (SRP/OCP):** Conforme ADR-01, os componentes foram alocados em `src/treinamento/diagnostico_overfitting.py`, mantendo as classes consolidadas `ModernHopfieldNetwork` e `AvaliadorHopfield` inalteradas e consumidas como clientes.
- **Diagnóstico Termodinâmico Analítico:** Conforme ADR-02, utiliza a distribuição Softmax já computada na inferência para medir entropia O(1), dispensando buscas exaustivas e custosas em grade (*grid search*).
- **Imutabilidade Estrita:** Conforme ADR-03, todas as perturbações operam sobre cópias explícitas (`np.array(..., copy=True)`), evitando corrupção de matrizes biológicas no pipeline.

#### 🛡️ Riscos, Efeitos Colaterais e O que NÃO mudou
- **Áreas Protegidas e Sem Alteração:** As rotinas de projeção vetorial `rSWeeP`, o binarizador de expressão e a dinâmica central da Hopfield continuam 100% inalterados.
- **Resiliência e Compatibilidade:** O novo módulo é puramente consultivo/diagnóstico, sem alterar pesos ou dados de produção. Validações de entrada utilizam política *Fail-Fast* com `ValueError`.

#### 🧪 Como testar e validar
1. Executar os 10 testes unitários específicos da feature:
   ```bash
   uv run pytest tests/test_diagnostico_overfitting.py
   ```
2. Executar toda a suíte de regressão do repositório:
   ```bash
   uv run pytest
   ```
3. Validar formatação e tipagem estática:
   ```bash
   uv run ruff check src/ tests/
   uv run ruff format --check src/ tests/
   uv run pyright src/treinamento/diagnostico_overfitting.py tests/test_diagnostico_overfitting.py
   ```

---

### 🔗 Referências

- Tarefa / Card: Auditoria e Diagnóstico de Overfitting em Rede Hopfield
- Spec / Arquitetura: `docs/specs/diagnostico-overfitting-rede-hopfield/`

---

### 📗 Checklist do desenvolvedor

- [x] Foi adicionado ou atualizado testes para a solução?
- [x] Foi criado ou atualizado a documentação que afeta a mudança?
- [x] Dependências e contratos externos validados?

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
