# 19. O resultado sobrevive a novas amostras

A Porta 3 podia depender das 25 ocorrências sorteadas por período. Duas novas
amostras foram executadas, mantendo modelo, palavras, inventários e métrica.

| Seed | Spearman bruto | Controlando nº sentidos |
|---|---:|---:|
| 20260613 | 0,586 | 0,454 |
| 20260614 | 0,549 | 0,402 |
| 20260615 | 0,621 | 0,522 |

A média bruta foi `0,585`, com desvio padrão `0,036`. A permutação conjunta
deu `p=0,00115`. Os rankings de mudança correlacionaram de `0,808` a `0,885`.

## O que foi confirmado

A associação entre divergência de sentidos e o gold:

- não depende de uma única amostra;
- continua positiva após controlar o tamanho do inventário;
- mantém `plane` entre os maiores deslocamentos;
- preserva os controles estáveis em posições baixas.

## O que ainda falta

Scores individuais variam. `plane`, por exemplo, ficou entre `0,334` e
`0,513`. O próximo passo é construir um nulo dentro de cada palavra,
permutando período entre ocorrências.

Relatório:
[Replicação da Porta 3](../docs/26-consec_gate3_replication_results.md).
