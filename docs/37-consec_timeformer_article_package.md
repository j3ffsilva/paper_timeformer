# Pacote secundário para o artigo: validação por sentidos

## Objetivo desta fase

Esta fase encerra a exploração aberta da linha ConSeC/WordNet e transforma
essa cadeia confirmatória em material auditável para o artigo. O pacote reúne
resultados da validação semântica externa, incerteza por palavra, exemplos
textuais e figuras reproduzíveis em:

```text
outputs/paper_assets/consec_timeformer/
```

O gerador único é:

```text
scripts/build_consec_timeformer_paper_assets.py
```

## Hierarquia das alegações

### Relação com a alegação principal

A alegação principal do projeto é que o TimeFormer oferece consultas
`token@time` e vizinhanças relacionais específicas de cada período. Essa
alegação não depende de converter automaticamente vizinhos em sentidos
WordNet.

### Alegação desta análise

Distribuições explícitas de sentido estimadas pelo ConSeC capturam mudança
lexical diacrônica no benchmark e possuem correspondência local com a
geometria contextual do TimeFormer.

### Alegação de decomposição

A mudança da mistura de sentidos explica uma parcela pequena, mas positiva e
replicada, do deslocamento geométrico entre períodos. Na `layer_2`, a média
observada da parcela excedente foi `0,048`, com IC 95% `[0,024; 0,081]`.

### Limite essencial

A natureza da componente complementar, numericamente próxima de `0,952`, não
é identificada pela decomposição. Mudança contextual, diferenças de domínio,
estrutura semântica fora da WordNet e outros fatores são hipóteses possíveis,
não conclusões.

## Cadeia de resultados

| Etapa | Resultado |
|---|---|
| Replicação da Porta 3 | Spearman médio `0,585`; permutação conjunta `p=0,0011` |
| Alinhamento por ocorrência, `layer_1` | rho médio `0,062`; IC `[0,047; 0,078]` |
| Alinhamento por ocorrência, `layer_2` | rho médio `0,187`; IC `[0,146; 0,231]` |
| Decomposição suave, `layer_2` | excesso médio de composição `0,048`; IC `[0,024; 0,081]` |
| Bootstrap estratificado | mediana agregada `0,042`; IC `[0,032; 0,053]` |

Dez palavras tiveram intervalo individual inteiramente positivo:
`plane`, `multitude`, `gas`, `record`, `land`, `attack`, `bit`, `thump`,
`risk` e `fiction`. Nenhuma foi robustamente negativa.

## Auditoria qualitativa

Os exemplos abaixo foram escolhidos entre os casos robustos, usando contextos
reais com posterior alto para sentidos ilustrativos definidos antes da
seleção da ocorrência.

| Palavra | 1810-1860 | 1960-2010 | Veredito |
|---|---|---|---|
| `plane` | plano geométrico | avião | substituição dominante clara |
| `multitude` | reunião numerosa de pessoas | grande número indefinido | mudança plausível, mas a nomenclatura do synset deve ser omitida em favor da definição |
| `gas` | substância no estado gasoso | gasolina | diversificação lexical clara |
| `record` | registro como evidência documental | disco fonográfico | emergência clara de um sentido material |

Os oito contextos e seus identificadores estão em
`context_audit_examples.csv`. Eles servem como evidência ilustrativa, não como
uma nova amostra confirmatória.

## Texto-base para a seção de resultados

> ConSeC sense-distribution divergence replicated across three independent
> occurrence samples (mean Spearman rho = 0.585; joint permutation p =
> 0.0011). On the same occurrences, posterior sense dissimilarity was
> positively associated with TimeFormer contextual distance within target
> words, both in layer 1 (mean partial rho = 0.062, 95% CI [0.047, 0.078])
> and, exploratorily, in layer 2 (0.187, [0.146, 0.231]). A soft
> decomposition of temporal centroid displacement showed that changes in
> sense composition accounted for a small but positive excess share in layer
> 2 (mean = 0.048, 95% CI [0.024, 0.081]). Stratified occurrence bootstrap
> preserved the aggregate conclusion (median = 0.042, [0.032, 0.053]) and
> identified ten targets with individually positive intervals.

## Figuras e tabelas

1. `figure1_bootstrap_forest`: resultado individual das 25 palavras e sua
   incerteza. Deve ser a figura principal da decomposição.
2. `figure2_jsd_vs_composition`: mostra que maior mudança explícita de mistura
   tende a acompanhar maior parcela geométrica de composição.
3. `figure3_selected_sense_shifts`: material interpretativo para quatro casos
   robustos.
4. `table1_core_results`: resume toda a cadeia confirmatória.
5. `table2_robust_targets`: lista os dez resultados individuais robustos.
6. `table3_selected_examples`: liga os exemplos qualitativos aos valores
   quantitativos.

As figuras existem em PNG para inspeção e PDF vetorial para o manuscrito. As
tabelas existem em CSV e em fragmentos LaTeX compatíveis com `booktabs`.

## O que não afirmar

- Não afirmar que o sucesso do TimeFormer depende da inferência automática de
  sentidos WordNet.
- Não afirmar que APD e divergência de sentidos são métricas equivalentes.
- Não atribuir automaticamente qualquer natureza à componente complementar.
- Não apresentar a `layer_2` como detector temporal globalmente superior.
- Não generalizar os dez intervalos individuais para as 15 palavras
  classificadas como incertas.
- Não tratar exemplos escolhidos para interpretação como validação
  independente.
- Não afirmar causalidade entre mudança de mistura e deslocamento geométrico.

## Próxima ação

A próxima fase deve consolidar primeiro o resultado principal `token@time`. A
ordem mais segura é:

1. congelar esta versão da validação por sentidos;
2. gerar vizinhos de `w@D0` e `w@D1`, ganhos e perdas, com estabilidade entre
   seeds;
3. usar essas vizinhanças como resultado principal;
4. inserir este pacote como validação externa e análise de interpretabilidade;
5. deixar a componente não atribuída aberta à investigação especializada;
6. somente então montar a narrativa completa do manuscrito.

O posicionamento canônico está em
`docs/38-scientific_positioning_token_time_and_sense_analysis.md`.
