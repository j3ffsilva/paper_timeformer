# Pré-registração da decomposição vetorial por sentidos

**Data:** 2026-06-14
**Estado:** anterior ao cálculo da decomposição.

## Pergunta

Quanto do deslocamento temporal do centróide contextual de uma palavra se
alinha à mudança da composição de sentidos, e quanto permanece como deriva
dentro dos sentidos?

## Dados congelados

Serão reutilizados:

```text
3 amostras ConSeC da Porta 3
2 seeds TimeFormer cronológicas
cache de 3.383 ocorrências × layer_1/layer_2
25 palavras confirmatórias
```

Não haverá nova inferência neural nem uso do gold SemEval.

`layer_2` será a análise principal porque a etapa anterior, concluída antes
desta pré-registração, mostrou alinhamento local mais forte entre sua geometria
e os posteriores ConSeC. `layer_1` será sensibilidade.

## Representação

Cada vetor de ocorrência será normalizado para norma 1 antes de calcular
centróides. Para período `t` e sentido `s`:

```text
p_t,s  = média da probabilidade posterior do sentido s
mu_t,s = centróide vetorial ponderado pela probabilidade do sentido s
m_t    = centróide global das ocorrências do período
```

Com posteriores normalizados:

```text
m_t = soma_s p_t,s * mu_t,s
```

## Decomposição simétrica

O deslocamento total é:

```text
T = m_1 - m_0
```

Ele será decomposto exatamente em:

```text
C = soma_s (p_1,s - p_0,s) * (mu_1,s + mu_0,s) / 2
D = soma_s (p_1,s + p_0,s) / 2 * (mu_1,s - mu_0,s)

T = C + D
```

`C` é a componente de composição: mudança das proporções dos sentidos.
`D` é a componente de deriva: deslocamento dos centróides dentro dos sentidos.

A decomposição é simétrica: nenhum período é escolhido como referência.

## Métrica principal

A contribuição alinhada de composição será medida pela projeção:

```text
share_C = dot(C, T) / ||T||²
share_D = dot(D, T) / ||T||²
```

Por construção:

```text
share_C + share_D = 1
```

As shares podem ser negativas ou maiores que 1 quando as componentes se
opõem. Portanto, elas são contribuições direcionais, não percentuais limitados.

## Nulo

Para cada palavra, período, seed ConSeC, seed TimeFormer e camada:

1. preservar os vetores;
2. preservar os posteriores e a mistura média de cada período;
3. embaralhar a correspondência entre posteriores e vetores dentro de cada
   período;
4. repetir 2.000 vezes.

Esse nulo mantém `T` e `p_t,s` fixos, mas destrói a associação entre sentido e
geometria contextual.

```text
excess_share_C = share_C_observada - média(share_C_nula)
```

## Regra de decisão

A contribuição geométrica da composição será considerada estabelecida em
`layer_2` se:

1. `excess_share_C` tiver média positiva nas seis combinações;
2. a média agregada por palavra for positiva;
3. um teste de inversão de sinais nas 25 palavras, com 20.000 permutações,
   produzir `p < 0,05`;
4. pelo menos 15 palavras tiverem excesso agregado positivo;
5. o erro relativo de reconstrução `||T-(C+D)|| / ||T||` for menor que
   `1e-6`.

## Análises secundárias

1. repetir em `layer_1`;
2. reportar normas de `T`, `C` e `D`;
3. reportar cossenos `cos(C,T)` e `cos(D,T)`;
4. correlacionar `excess_share_C` com a JSD temporal do ConSeC;
5. listar palavras em que composição e deriva se reforçam ou se cancelam.

Nenhuma análise secundária substitui a decisão principal.

## Interpretação

```text
excess_share_C > 0:
  a mudança de mistura de sentidos aponta na direção do deslocamento global
  mais do que seria esperado por uma associação aleatória sentido-vetor.

excess_share_C ~ 0:
  a decomposição algébrica existe, mas a atribuição semântica da componente
  não excede o nulo.

share_D dominante:
  a maior parte do deslocamento está associada a mudanças contextuais dentro
  dos sentidos ou a estrutura não capturada pelo inventário.
```

## Saídas

```text
outputs/consec_timeformer_soft_decomposition/per_combination_target.csv
outputs/consec_timeformer_soft_decomposition/per_target.csv
outputs/consec_timeformer_soft_decomposition/summary.json
```
