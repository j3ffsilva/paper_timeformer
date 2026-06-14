# Auditoria de cobertura WordNet para a Porta 3

## Objetivo

Decidir, antes de executar novas previsões do ConSeC, se o inventário WordNet
3.0 contém sentidos adequados para representar os usos históricos de cada uma
das 37 palavras-alvo.

Esta não é uma tarefa de adivinhar se a palavra mudou semanticamente. Não
consulte o gold do SemEval nem previsões de modelos.

## Arquivos

Preencha somente:

```text
coverage_review.csv
```

Use como apoio:

```text
sense_inventory.csv
context_samples.csv
target_summary.csv
```

`sense_inventory.csv` contém todos os sensekeys e definições WordNet do alvo.
`context_samples.csv` contém quatro ocorrências determinísticas de cada período
por palavra, sempre que disponíveis.

Quatro linhas já vêm preenchidas porque foram decididas nas Portas 1 e 2:
`plane_nn`, `tree_nn`, `graft_nn` e `chairman_nn`. Elas permanecem visíveis
para deixar explícito que a decisão foi transportada de evidência anterior.
Revise as outras 33 linhas.

## Como revisar cada palavra

1. Leia todos os sentidos WordNet da palavra.
2. Leia os contextos de 1810-1860 e 1960-2010.
3. Pergunte se cada uso observado possui um sentido WordNet semanticamente
   adequado, sem forçá-lo ao candidato apenas "mais próximo".
4. Preencha os quatro campos de decisão.

Pode usar Google Translate ou dicionário para compreender o contexto. Não use
chatbot ou sistema WSD para decidir o sentido.

## `coverage_status`

Use exatamente um destes valores:

- `sufficient`: os usos amostrados possuem candidatos WordNet adequados;
- `partial`: pelo menos um uso observado não possui candidato exato, mas
  outros usos estão cobertos;
- `missing`: o uso principal observado não é representado adequadamente;
- `monosemous_covered`: há um único sentido e os contextos são compatíveis;
- `monosemous_mismatch`: há um único sentido, mas algum contexto usa outro
  significado;
- `unclear`: a amostra não permite decidir.

Uma palavra com muitos sentidos pode ser `sufficient`. Uma palavra com um
único sentido pode ser `monosemous_mismatch`. O número de sentidos mede carga
de revisão, não qualidade da cobertura.

## `missing_or_mismatched_uses`

Se a cobertura for `partial`, `missing` ou `monosemous_mismatch`, descreva
brevemente o uso ausente. Exemplo:

```text
objeto botânico enxertado (scion), distinto do ato de enxertar
```

Nos demais casos, pode deixar vazio.

## `gate3_decision`

Use:

- `eligible`: pode entrar na análise discriminativa da Porta 3;
- `diagnostic_only`: útil como controle ou para estudar lacuna, mas não deve
  entrar no score confirmatório;
- `exclude`: cobertura inadequada para esta linha experimental;
- `needs_more_context`: é necessário ampliar a amostra antes de decidir.

Palavras monossêmicas normalmente são `diagnostic_only`, pois não testam
desambiguação. Uma palavra com cobertura parcial pode ser `eligible` apenas se
os sentidos/subconjuntos cobertos puderem ser separados antes das previsões e
os usos ausentes forem mantidos fora do score.

## `confidence`

Use `high`, `medium` ou `low`.

## Critério de término

Ao final, todas as 37 linhas devem ter:

```text
coverage_status
gate3_decision
confidence
```

Não altere `target`, `pos`, `n_wordnet_senses` ou `automatic_priority`.
