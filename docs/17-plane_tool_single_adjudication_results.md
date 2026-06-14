# Resultado da adjudicação cega de `plane` com um anotador

## Desenho

As 19 ocorrências de D0 previamente selecionadas pela heurística como
`tool` foram avaliadas sem acesso às previsões do LMMS. O anotador usou
Google Translate para compreender trechos de inglês histórico, mas não pediu
ao tradutor que classificasse o sentido de `plane`.

Este é um diagnóstico com **um único anotador**. Portanto, não há estimativa
de concordância ou kappa, e o resultado não substitui uma anotação
independente caso ela se torne necessária para a versão final do artigo.

## Rótulos humanos

| Rótulo | N | Proporção |
|---|---:|---:|
| `tool` | 16 | 84,2% |
| `geometry` | 1 | 5,3% |
| `botanical` | 1 | 5,3% |
| `unclear` | 1 | 5,3% |

Foram atribuídas 18 confianças `high` e uma `medium`. A precisão diagnóstica
da heurística original para selecionar ferramentas foi, assim, `16/19 =
84,2%`.

## LMMS após a anotação

Excluindo o único exemplo `unclear`, o LMMS acertou `2/18 = 11,1%`.
Considerando somente os 16 exemplos humanos de ferramenta, acertou
`2/16 = 12,5%`.

| Rótulo humano | Predição LMMS | N |
|---|---|---:|
| `tool` | `geometry` | 12 |
| `tool` | `aircraft` | 2 |
| `tool` | `tool` | 2 |
| `geometry` | `aircraft` | 1 |
| `botanical` | `tool` | 1 |
| `unclear` | `tool` | 1 |

O erro dominante não decorre de poucos casos limítrofes. O LMMS converteu
12 contextos humanos de ferramenta em `geometry`, inclusive construções
lexicalmente explícitas como `bench plane`, `mould plane` e listas de
ferramentas de carpintaria.

## Conclusão

A adjudicação não resgata o LMMS. Ela mostra simultaneamente que:

1. a heurística tinha algum ruído, pois 3 das 19 ocorrências não foram
   confirmadas como ferramenta;
2. esse ruído não explica a baixa acurácia;
3. o LMMS apresenta dificuldade real com o sentido histórico raro de
   ferramenta;
4. o NO-GO original permanece válido e fica mais bem localizado.

O resultado original de Gate 1 continua congelado. Os números corrigidos são
reportados separadamente como análise pós-adjudicação.

## Próximo teste

O próximo passo é executar o ConSeC oficial, congelado e sem calibração neste
conjunto. A regra de parada permanece:

- se ConSeC também falhar nos exemplos adjudicados de ferramenta, não escalar
  um atlas WordNet geral para os 37 alvos;
- se ConSeC funcionar, investigar se a falha é específica da representação
  por vetores de sentido do LMMS.

Esse teste foi concluído. O ConSeC acertou 14 dos 16 exemplos humanos de
ferramenta. Ver
[resultado do ConSeC](18-consec_plane_adjudicated_results.md).

## Artefatos

```text
annotations/plane_tool_gate1/annotator_a.csv
outputs/external_wsd/lmms_plane_gate1/single_adjudication/
```
