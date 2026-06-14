# Pré-registração da replicação da Porta 3

**Data de congelamento:** 2026-06-14
**Estado:** anterior às inferências das novas amostras.

## Objetivo

Testar se o `GO` da Porta 3 é estável quando as ocorrências são reamostradas e
se a associação com o gold permanece depois de controlar o número de sentidos
WordNet disponíveis.

## Configuração congelada

Tudo permanece igual à Porta 3:

- mesmos 25 alvos confirmatórios;
- mesmos 21 alvos de alta confiança;
- mesmos nove diagnósticos parciais;
- mesmos três controles monossêmicos;
- ConSeC e WordNet congelados;
- 25 ocorrências por palavra e período;
- JSD sobre probabilidades médias;
- bootstrap e permutação com 20.000 réplicas.

Novas seeds de amostragem:

```text
20260614
20260615
```

A seed original `20260613` será incluída apenas na consolidação.

## Métricas por seed

1. Spearman entre JSD e gold graduado;
2. IC bootstrap;
3. p por permutação;
4. análise de alta confiança;
5. `rho(JSD, número de sentidos)`;
6. correlação parcial por ranks entre JSD e gold, controlando número de
   sentidos, com p por permutação.

## Regra de replicação

Sucesso se:

```text
1. Spearman bruto for positivo nas duas novas seeds;
2. a média dos três Spearman brutos for positiva;
3. a média dos três Spearman parciais for positiva;
4. a permutação conjunta da média bruta produzir p < 0,05.
```

A significância individual de cada nova seed será reportada, mas não é
requisito isolado por causa de `n=25`.

## Inferência retomável

O avaliador grava cada ocorrência imediatamente em
`occurrence_predictions.csv`. Com `--resume`, execuções interrompidas validam
o cache e processam somente os itens ausentes. O cache não altera a ordem, a
amostra ou as probabilidades.

Previsões de uma seed anterior podem ser reutilizadas apenas quando o
`sample_id` completo coincide. Isso significa mesmo período, palavra,
documento, posição e contexto sob o mesmo modelo e inventário. Itens novos
continuam sendo inferidos normalmente.

## Artefatos

```text
outputs/external_wsd/consec_gate3_rep_seed20260614/
outputs/external_wsd/consec_gate3_rep_seed20260615/
outputs/external_wsd/consec_gate3_replication/
```
