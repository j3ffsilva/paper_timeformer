# Pré-registração do nulo dentro de palavra

**Data:** 2026-06-14
**Estado:** anterior ao cálculo das permutações dentro de palavra.

## Pergunta

A JSD observada entre D0 e D1 excede a divergência produzida apenas ao dividir
aleatoriamente as mesmas ocorrências em dois grupos?

## Dados congelados

Serão reutilizadas as probabilidades já gravadas nas três seeds da Porta 3.
Nenhuma nova inferência ou seleção de ocorrência será realizada.

Para cada palavra e seed:

```text
50 ocorrências = 25 de D0 + 25 de D1
```

## Nulo

Os rótulos de período serão permutados 20.000 vezes dentro de cada palavra,
preservando grupos de tamanho 25/25.

Para cada palavra:

```text
JSD_excedente = JSD_observada - média(JSD_nula)
z_nulo = (JSD_observada - média_nula) / DP_nulo
p_palavra = proporção(JSD_nula >= JSD_observada)
```

Os p-valores individuais receberão correção Benjamini-Hochberg dentro de cada
seed.

## Métricas agregadas

Nos 25 alvos confirmatórios:

1. Spearman entre `JSD_excedente` e gold graduado;
2. Spearman entre `z_nulo` e gold graduado;
3. média dessas correlações nas três seeds;
4. estabilidade entre seeds;
5. associação dos scores corrigidos com número de sentidos.

## Regra

O nulo será considerado útil se:

```text
1. a média de Spearman(JSD_excedente, gold) for positiva;
2. a correlação for positiva nas três seeds;
3. JSD_excedente reduzir a associação com número de sentidos em média,
   comparada à JSD bruta.
```

O número de palavras significativas após FDR é descritivo e não altera essa
regra.
