# Resultados da Porta 3: distribuições temporais de sentido

**Data:** 2026-06-13
**Decisão pré-registrada:** `GO`

## Resultado principal

O ConSeC congelado foi aplicado a 1.850 ocorrências: 25 por palavra e período
para os 37 alvos. O score de cada palavra é a Jensen-Shannon entre as
distribuições médias de probabilidade dos sentidos em D0 e D1.

Nos 25 alvos polissêmicos com cobertura WordNet suficiente:

| Métrica | Resultado |
|---|---:|
| Spearman com gold graduado | **0,586** |
| IC bootstrap 95% | **[0,231; 0,818]** |
| p por permutação bilateral | **0,0028** |
| ROC-AUC | 0,714 |
| Average precision | 0,658 |

A regra exigia Spearman positivo e `p < 0,05`.

```text
Porta 3 = GO
```

Na análise de sensibilidade pré-definida, restrita aos 21 alvos de confiança
alta:

```text
Spearman = 0,600
IC 95% = [0,182; 0,840]
p = 0,0055
ROC-AUC = 0,750
AP = 0,719
```

## Ranking confirmatório

| Palavra | JSD | Gold graduado |
|---|---:|---:|
| `plane_nn` | 0,334 | 0,882 |
| `gas_nn` | 0,174 | 0,160 |
| `record_nn` | 0,163 | 0,427 |
| `stab_nn` | 0,162 | 0,401 |
| `ball_nn` | 0,142 | 0,409 |
| `prop_nn` | 0,135 | 0,625 |
| `bag_nn` | 0,084 | 0,100 |
| `head_nn` | 0,081 | 0,295 |
| `attack_nn` | 0,074 | 0,144 |
| `multitude_nn` | 0,072 | 0,100 |
| `bit_nn` | 0,038 | 0,307 |
| `part_nn` | 0,037 | 0,161 |
| `word_nn` | 0,036 | 0,179 |
| `risk_nn` | 0,029 | 0,000 |
| `land_nn` | 0,024 | 0,223 |
| `player_nn` | 0,020 | 0,274 |
| `fiction_nn` | 0,018 | 0,021 |
| `thump_nn` | 0,017 | 0,143 |
| `circle_vb` | 0,014 | 0,171 |
| `tree_nn` | 0,014 | 0,071 |
| `savage_nn` | 0,012 | 0,097 |
| `relationship_nn` | 0,004 | 0,056 |
| `pin_vb` | 0,003 | 0,207 |
| `donkey_nn` | 0,002 | 0,160 |
| `contemplation_nn` | 0,000 | 0,071 |

## Exemplos concretos

### `plane_nn`

```text
D0: geometria 90,9%; avião 0,9%
D1: geometria 20,9%; avião 67,0%
JSD = 0,334
```

### `tree_nn`

```text
D0: planta 95,6%; diagrama 4,2%
D1: planta 86,4%; diagrama 12,7%
JSD = 0,0135
```

### Controles monossêmicos

`chairman`, `lass` e `quilt` têm uma única saída possível e, por construção,
JSD igual a zero. Isso confirma a execução, mas não constitui evidência de
estabilidade lexical.

### `graft_nn`

`graft` permaneceu diagnóstico por cobertura parcial:

```text
D0: ato de enxertar 45,5%; médico 53,7%; corrupção 0,8%
D1: ato de enxertar 9,9%; médico 56,1%; corrupção 34,0%
JSD = 0,164
```

A direção é semanticamente plausível, mas a lacuna do objeto botânico impede
seu uso no score confirmatório.

## Confundimentos

A amostra foi balanceada por palavra e período. O score não apresentou
associação relevante com desequilíbrio de frequência:

```text
rho(JSD, |log frequência D1/D0|) = 0,032
```

Entretanto, palavras com mais candidatos possuem mais graus de liberdade:

```text
rho(JSD, número de sentidos) = 0,533
p = 0,006
```

Uma análise exploratória de correlação parcial por ranks, controlando número
de sentidos, ainda produziu:

```text
rho parcial = 0,454
p = 0,023
```

Essa análise não fazia parte da regra decisória. Deve ser pré-registrada e
replicada antes de ser tratada como resultado confirmatório.

## Emenda computacional

O limite inicial de 100 ocorrências foi reduzido para 25 antes de qualquer
resultado, após execuções interrompidas sem arquivos de previsão. A razão foi
o custo de palavras com até 33 glosas. A emenda está registrada no protocolo.

Uma execução com lote 4096 completou 1.797 itens, mas foi encerrada por memória
antes de gravar saídas. A execução final usou lote 1024, sem alterar dados ou
probabilidades.

## Interpretação

Este é o primeiro resultado do projeto em que uma medida explicitamente
baseada em distribuições de sentidos:

- usa uma régua externa congelada;
- passa por auditoria prévia de cobertura;
- separa alvos confirmatórios, parciais e monossêmicos;
- correlaciona significativamente com o gold graduado.

O resultado sustenta a formulação:

```text
mudança semântica temporal
≈ divergência entre distribuições de sentidos contextuais por período
```

Ele não elimina o problema de inventário nem demonstra estabilidade entre
amostras de ocorrência.

## Próximos passos

1. Pré-registrar uma replicação com duas novas seeds de amostragem.
2. Incluir como métrica obrigatória a associação com número de sentidos.
3. Testar score corrigido pelo nulo condicionado ao tamanho do inventário.
4. Estimar incerteza por reamostragem de ocorrências dentro de cada palavra.
5. Manter os nove alvos parciais fora do resultado confirmatório.
6. Usar as distribuições de sentido para explicar casos individuais e
   compará-las aos perfis relacionais do TimeFormer.

## Artefatos

```text
docs/23-consec_gate3_preregistration.md
scripts/prepare_consec_gate3.py
scripts/evaluate_consec_gate3.py
outputs/external_wsd/consec_gate3_preregistered/
outputs/external_wsd/consec_gate3/occurrence_predictions.csv
outputs/external_wsd/consec_gate3/target_scores.csv
outputs/external_wsd/consec_gate3/summary.json
```
