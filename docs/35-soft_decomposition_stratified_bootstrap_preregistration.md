# Pré-registração do bootstrap estratificado da decomposição

**Data:** 2026-06-14
**Estado:** anterior à reamostragem das ocorrências.

## Pergunta

Quais contribuições de composição estimadas por palavra permanecem estáveis
quando variamos as ocorrências observadas em D0 e D1?

## Dados congelados

Serão reutilizados:

```text
25 palavras confirmatórias
3 amostras ConSeC
2 seeds TimeFormer
layer_2 como análise principal
layer_1 como sensibilidade
2.000 nulos por combinação já calculados
```

Nenhuma nova inferência, seleção de palavra ou consulta ao gold será feita.

## Reamostragem

Para cada palavra e amostra ConSeC:

1. reamostrar com reposição 25 ocorrências de D0;
2. reamostrar com reposição 25 ocorrências de D1;
3. usar os mesmos índices nas duas seeds TimeFormer;
4. recalcular a decomposição simétrica;
5. repetir 2.000 vezes.

O pareamento das seeds TimeFormer evita contar como independentes duas
representações quase idênticas das mesmas frases.

Em cada réplica, a estimativa por palavra será a média das:

```text
3 amostras ConSeC × 2 seeds TimeFormer
```

## Score

O score principal será:

```text
excesso de composição bootstrap
= share_C da réplica
- média nula previamente estimada para a combinação
```

O nulo não será recalculado dentro de cada bootstrap. Suas 2.000 permutações
por combinação já estimam a expectativa sob independência sentido-vetor; o
bootstrap atual mede uma fonte diferente de incerteza: seleção de ocorrências.

## Intervalos

Serão reportados:

```text
estimativa observada
mediana bootstrap
IC percentil 95%
probabilidade bootstrap de excesso > 0
```

## Classificação por palavra

```text
robust_positive:
  limite inferior do IC 95% > 0

robust_negative:
  limite superior do IC 95% < 0

uncertain:
  IC 95% inclui zero
```

Essa classificação será aplicada individualmente a `layer_2`. Não haverá
correção por múltiplos testes porque os intervalos são usados para
caracterização de estabilidade, não para declarar 25 descobertas
independentes. O número de palavras robustas será descritivo.

## Critério de estabilidade global

A conclusão agregada da decomposição será considerada estável se:

1. a média dos excessos por palavra tiver IC bootstrap 95% acima de zero;
2. pelo menos 15 das 25 palavras tiverem probabilidade bootstrap
   `P(excesso > 0) > 0,5`;
3. `plane_nn` permanecer `robust_positive`;
4. a correlação entre estimativas observadas e medianas bootstrap for
   `Spearman > 0,8`.

## Auditoria pré-especificada

Serão destacados:

```text
altos observados: plane, multitude, gas, record
baixos observados: player, donkey, stab
```

Também serão listadas todas as palavras cuja classificação seja
`robust_positive` ou `robust_negative`.

## Limitações

O bootstrap mede a incerteza causada pelas ocorrências disponíveis. Ele não
mede:

- incerteza do modelo ConSeC;
- incerteza do inventário WordNet;
- variação entre arquiteturas;
- generalização para outros corpora ou períodos.

## Saídas

```text
outputs/consec_timeformer_soft_decomposition_bootstrap/per_target.csv
outputs/consec_timeformer_soft_decomposition_bootstrap/bootstrap_samples.npz
outputs/consec_timeformer_soft_decomposition_bootstrap/summary.json
```
