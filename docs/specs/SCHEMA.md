# Schema de Especificações Técnicas e Arquiteturais (docs/specs/)

Este diretório armazena as especificações de negócio e arquitetura técnica geradas pelo fluxo estruturado do agente de IA e validadas pelo desenvolvedor/pesquisador.

## Estrutura de Diretórios

```
docs/specs/
├── SCHEMA.md                 # Definição das regras de governança e estrutura
├── index.md                  # Índice unificado das especificações do repositório
├── log.md                    # Histórico cronológico de criações e alterações
└── <slug>/                   # Diretório dedicado a uma especificação (kebab-case)
    ├── spec.md               # Especificação funcional e regras de negócio (fase: spec)
    └── arquitetura.md        # Desenho técnico, contratos e responsabilidades (fase: arquitetura)
```

## Frontmatter de `spec.md`

Todo arquivo `spec.md` deve conter frontmatter YAML:

```yaml
---
tipo: especificacao
slug: <slug>
generated.by: antigravity/implementar
data: YYYY-MM-DDTHH:MM:SSZ
resumo: "Resumo executivo do comportamento sob especificação."
---
```

## Seções Obrigatórias de `spec.md`

1. **Repo e Slug**: Identificação do repositório dono e do slug kebab-case.
2. **Problema e Atores**: Descrição do problema de negócio e papéis dos atores.
3. **Contexto do Sistema (Diagrama C1)**: Diagrama C1 em Mermaid.
4. **Fluxo de Negócio**: Diagrama Mermaid de fluxo ou estados anotado com IDs (`RN-XX`, `RF-XX`).
5. **Regras de Negócio e Exceções**: Regras indexadas estritamente aos IDs.
6. **Critérios de Aceite Observáveis**: Cenários no formato Dado/Quando/Então vinculados aos IDs.
7. **Fora de Escopo**: Limites explícitos da funcionalidade.
8. **Glossário do Domínio**: Termos biológicos e computacionais acordados.
