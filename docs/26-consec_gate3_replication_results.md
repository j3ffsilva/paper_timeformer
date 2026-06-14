# Replicação da Porta 3

**Data:** 2026-06-14
**Decisão pré-registrada:** replicação bem-sucedida.

## Resultados por seed

| Seed | Spearman | p permutação | Spearman parcial | p parcial | JSD vs. nº sentidos |
|---|---:|---:|---:|---:|---:|
| 20260613 | 0,586 | 0,0028 | 0,454 | 0,0238 | 0,533 |
| 20260614 | 0,549 | 0,0053 | 0,402 | 0,0529 | 0,543 |
| 20260615 | 0,621 | 0,0010 | 0,522 | 0,0102 | 0,453 |

```text
Spearman bruto médio   = 0,585 ± 0,036
Spearman parcial médio = 0,459 ± 0,060
p da permutação conjunta = 0,00115
```

Todos os critérios pré-registrados passaram.

## Estabilidade do ranking

| Seeds | Spearman dos scores |
|---|---:|
| 20260613 × 20260614 | 0,808 |
| 20260613 × 20260615 | 0,885 |
| 20260614 × 20260615 | 0,823 |

O sinal entre palavras é estável, embora a magnitude absoluta varie.

## Maior variação amostral

| Palavra | JSD média | DP | Faixa |
|---|---:|---:|---:|
| `plane_nn` | 0,428 | 0,090 | 0,334–0,513 |
| `record_nn` | 0,178 | 0,083 | 0,103–0,266 |
| `prop_nn` | 0,125 | 0,054 | 0,066–0,173 |
| `ball_nn` | 0,157 | 0,050 | 0,115–0,212 |
| `bit_nn` | 0,059 | 0,046 | 0,028–0,111 |
| `stab_nn` | 0,116 | 0,043 | 0,077–0,162 |

Um score individual deve carregar incerteza de amostragem, mesmo quando o
ranking agregado é estável.

## Tamanho do inventário

O número de candidatos continua associado à JSD. Contudo, a correlação parcial
entre JSD e gold permaneceu positiva nas três seeds:

```text
0,402 a 0,522
média = 0,459
```

Uma seed ficou limítrofe isoladamente (`p=0,0529`), mas as outras duas foram
significativas. O tamanho do inventário explica parte do score, mas não explica
sozinho sua associação com o gold.

## Infraestrutura

O avaliador agora grava cada previsão imediatamente, retoma apenas itens
ausentes, valida os IDs e reaproveita ocorrências idênticas entre seeds. As
novas seeds reutilizaram 230 e 383 previsões, respectivamente.

## Decisão

```text
Replicação da Porta 3 = SUCESSO
```

## Próximos passos

1. Construir um nulo por palavra permutando período entre suas 50 ocorrências.
2. Produzir JSD excedente e p por palavra.
3. Comparar JSD bruta, JSD excedente e correlação parcial.
4. Corrigir múltiplos testes nas análises por palavra.
5. Integrar as distribuições de sentido às explicações do TimeFormer.

## Artefatos

```text
outputs/external_wsd/consec_gate3_rep_seed20260614/
outputs/external_wsd/consec_gate3_rep_seed20260615/
outputs/external_wsd/consec_gate3_replication/
scripts/consolidate_consec_gate3_replication.py
```
