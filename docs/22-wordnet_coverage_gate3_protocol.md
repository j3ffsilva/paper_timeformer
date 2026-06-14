# Protocolo de cobertura WordNet para a Porta 3

**Data de congelamento:** 2026-06-13
**Estado:** auditoria anterior a qualquer nova inferência do ConSeC.

## Motivação

A Porta 2 mostrou dois fatos simultâneos:

1. o ConSeC discrimina muito bem sentidos históricos quando o candidato
   correto existe;
2. o inventário nominal de `graft` não representa adequadamente o objeto
   botânico enxertado.

Portanto, contar sensekeys não basta. Antes de escalar para os 37 alvos,
precisamos verificar:

```text
uso histórico observado -> existe candidato WordNet semanticamente adequado?
```

## Unidade de decisão

A auditoria terá uma decisão por palavra, não por ocorrência.

Para cada um dos 37 alvos serão apresentados:

- todos os sensekeys WordNet 3.0 compatíveis com lema e POS;
- definição e exemplos de cada synset;
- frequência total em D0 e D1;
- quatro contextos determinísticos de D0;
- quatro contextos determinísticos de D1.

A amostra usa seed `20260613` e janela de 41 tokens.

## O que a triagem automática pode concluir

Ela pode medir:

- ausência de inventário;
- monossêmia;
- número de sentidos e carga de revisão;
- disponibilidade de ocorrências nos dois períodos.

Ela não pode concluir cobertura semântica. Um sensekey próximo não substitui
um sentido ausente.

## Rótulos humanos

`coverage_status`:

```text
sufficient
partial
missing
monosemous_covered
monosemous_mismatch
unclear
```

`gate3_decision`:

```text
eligible
diagnostic_only
exclude
needs_more_context
```

Palavras monossêmicas não entram no score de desambiguação. Palavras com
cobertura parcial só podem entrar por subconjuntos pré-definidos que excluam
explicitamente a lacuna.

## Proteções contra seleção

- a ficha não contém `binary` ou `graded` do SemEval;
- a ficha não contém previsões, probabilidades ou erros do ConSeC;
- as amostras são sorteadas antes de qualquer inferência;
- a decisão de elegibilidade deve ser congelada antes da Porta 3;
- casos `unclear` exigem mais contexto e não podem ser promovidos
  automaticamente.

## Saídas

```text
outputs/external_wsd/wordnet_coverage_gate3/target_summary.csv
outputs/external_wsd/wordnet_coverage_gate3/sense_inventory.csv
outputs/external_wsd/wordnet_coverage_gate3/context_samples.csv
annotations/wordnet_coverage_gate3/coverage_review.csv
annotations/wordnet_coverage_gate3/README_REVIEWER.md
```

## Regra para abrir a Porta 3

A Porta 3 só será especificada depois da revisão das 37 linhas.

O conjunto confirmatório poderá conter apenas alvos marcados `eligible`.
`diagnostic_only`, `exclude` e `needs_more_context` serão reportados
separadamente e não poderão melhorar o resultado principal.

Quatro decisões anteriores são transportadas explicitamente para a ficha:

```text
plane_nn     sufficient / eligible
tree_nn      sufficient / eligible
graft_nn     partial / eligible apenas nos subconjuntos cobertos
chairman_nn  monosemous_covered / diagnostic_only
```

Elas foram estabelecidas nas Portas 1 e 2, antes desta auditoria. As outras 33
linhas começam vazias.

## Comando congelado

```bash
outputs/external_wsd/consec_env/bin/python \
  scripts/prepare_wordnet_coverage_gate3.py \
  --targets data/processed/semeval2020_task1/eng_lemma/targets.txt \
  --corpus-dir data/processed/semeval2020_task1/eng_lemma/corpus \
  --output-dir outputs/external_wsd/wordnet_coverage_gate3 \
  --annotation-dir annotations/wordnet_coverage_gate3 \
  --samples-per-period 4 \
  --context-radius 20 \
  --seed 20260613
```
