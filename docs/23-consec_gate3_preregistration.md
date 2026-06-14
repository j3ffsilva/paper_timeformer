# Pré-registração da Porta 3 com ConSeC

**Data de congelamento:** 2026-06-13
**Estado:** anterior à inferência do ConSeC e à consulta do gold nesta etapa.

## Pergunta

A mudança entre distribuições de sentidos previstas por um WSD externo
congelado acompanha a mudança semântica graduada do SemEval?

Para cada palavra:

```text
P0(s | w) = distribuição ConSeC em 1810-1860
P1(s | w) = distribuição ConSeC em 1960-2010
score(w)   = JSD(P0, P1)
```

## Conjuntos congelados

### Confirmatório

Vinte e cinco palavras polissêmicas foram marcadas `sufficient` na auditoria
de cobertura WordNet.

```text
attack bag ball bit circle contemplation donkey fiction gas head land
multitude part pin plane player prop record relationship risk savage stab
thump tree word
```

### Sensibilidade

Vinte e uma das 25 palavras confirmatórias possuem confiança `high`.
`circle`, `contemplation`, `gas` e `record` permanecem na análise principal,
mas saem desta análise de sensibilidade.

### Diagnósticos

Cobertura parcial:

```text
edge face graft lane ounce rag stroke tip twist
```

Controles monossêmicos:

```text
chairman lass quilt
```

Esses 12 alvos não entram no score principal.

## Amostragem

Para cada palavra, usa-se o mesmo número de ocorrências em D0 e D1:

```text
n(w) = min(25, frequência_D0(w), frequência_D1(w))
```

A seleção é determinística, seed `20260613`, antes da inferência. Cada contexto
possui raio de 20 tokens.

### Emenda computacional anterior aos resultados

A preparação inicial havia congelado limite 100. Duas execuções foram
interrompidas sem produzir arquivos de previsão: `token_batch_size=1024` após
1.965 ocorrências e `4096` após 1.137 ocorrências. Nenhuma probabilidade,
distribuição por período ou métrica foi gravada ou observada.

O custo cresce fortemente com inventários de até 33 glosas. Por viabilidade,
o limite foi reduzido para 25 antes de qualquer resultado. A seed, o algoritmo
de amostragem, os alvos, os papéis, o modelo, a JSD e a regra de decisão não
mudaram.

## Modelo

```text
ConSeC SemCor+WNGT oficial
commit 9602b5fd69f57be08a186988d1df34fe4152b63f
checkpoint SHA256
92421ed245723964db09ce396f19a0d1e55fe4d6e10d5ecb83278d9bc70ce8af
```

O modelo, tokenizer, definições WordNet e parâmetros permanecem congelados.

## Score

As probabilidades do ConSeC serão somadas por sentido e normalizadas dentro de
cada período. A divergência Jensen-Shannon usa log natural e fica em
`[0, log(2)]`.

Métrica principal:

```text
Spearman(score_WSD, gold_graded)
```

Incerteza e significância:

- bootstrap pareado por palavra, 20.000 réplicas;
- permutação dos gold scores entre palavras, 20.000 réplicas;
- teste bilateral.

Regra:

```text
GO: Spearman > 0 e p_permutação < 0,05
NO-GO: caso contrário
```

ROC-AUC e average precision para o gold binário são secundários e não alteram
a decisão.

## Análises obrigatórias

1. conjunto confirmatório de 25 palavras;
2. sensibilidade de alta confiança com 21 palavras;
3. distribuição dos controles monossêmicos;
4. resultados dos nove alvos parciais, claramente diagnósticos;
5. frequência versus score WSD como diagnóstico de confundimento;
6. tabela completa por palavra e por período.

## Proibições

- não remover palavra confirmatória depois de observar sua previsão;
- não redefinir sentidos ou glosas;
- não promover alvos parciais ao score principal;
- não escolher camada, threshold ou checkpoint;
- não substituir JSD por outra métrica após ver os resultados.

## Artefato congelado

```text
outputs/external_wsd/consec_gate3_preregistered/occurrences.csv
outputs/external_wsd/consec_gate3_preregistered/frozen_design.json
```
