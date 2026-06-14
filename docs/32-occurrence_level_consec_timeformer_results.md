# Alinhamento ConSeC-TimeFormer por ocorrência

**Data:** 2026-06-14
**Decisão pré-registrada:** alinhamento estabelecido.

## Pergunta

Dentro de uma mesma palavra, ocorrências com posteriores ConSeC mais
diferentes também ocupam posições mais distantes na geometria contextual do
TimeFormer?

Foram codificadas 3.383 ocorrências confirmatórias únicas. Cada ocorrência
recebeu:

```text
posterior ConSeC sobre sentidos WordNet
vetor layer_1 e layer_2 em dois checkpoints TimeFormer
```

A análise foi replicada em três amostras ConSeC e duas seeds TimeFormer.

## Resultado principal

A métrica principal foi a correlação parcial, por palavra, entre:

```text
JSD entre posteriores
distância cosseno entre vetores layer_1
```

controlando se o par ligava períodos diferentes.

```text
rho médio       = 0,062
rho mediano     = 0,059
IC bootstrap 95% da média = [0,047; 0,078]
palavras positivas = 23/25
p por inversão de sinais = 0,00005
```

Todos os critérios pré-registrados passaram:

```text
mediana positiva nas seis combinações = sim
média agregada positiva               = sim
p < 0,05                              = sim
pelo menos 15 palavras positivas      = sim
```

Portanto:

```text
Alinhamento por ocorrência = GO
```

O efeito é pequeno, mas consistente.

## Replicação

| ConSeC | TimeFormer | média dos 25 rhos | mediana | positivas |
|---|---|---:|---:|---:|
| 20260613 | 1000 | 0,048 | 0,032 | 18 |
| 20260613 | 1001 | 0,048 | 0,033 | 20 |
| 20260614 | 1000 | 0,075 | 0,078 | 22 |
| 20260614 | 1001 | 0,075 | 0,082 | 22 |
| 20260615 | 1000 | 0,064 | 0,032 | 22 |
| 20260615 | 1001 | 0,064 | 0,032 | 21 |

As duas seeds TimeFormer produzem números quase idênticos dentro de cada
amostra ConSeC. A maior variação vem da seleção das ocorrências.

## Controle do período

| Pares usados em `layer_1` | rho médio | IC 95% | positivas |
|---|---:|---:|---:|
| somente dentro do período | 0,074 | [0,052; 0,100] | 24 |
| somente entre períodos | 0,047 | [0,030; 0,064] | 21 |
| todos, controlando período | 0,062 | [0,047; 0,078] | 23 |

O alinhamento é mais forte dentro do mesmo período. Logo, não surge apenas
porque ConSeC e TimeFormer distinguem D0 de D1.

## Resultado exploratório de `layer_2`

| Camada | rho parcial médio | IC 95% | positivas |
|---|---:|---:|---:|
| `layer_1` | 0,062 | [0,047; 0,078] | 23/25 |
| `layer_2` | 0,187 | [0,146; 0,231] | 25/25 |

`layer_2` organiza diferenças locais de sentido melhor que `layer_1`, apesar
de sua APD temporal agregada ter sido fraca.

Isso resolve uma aparente contradição:

```text
estrutura local de sentidos em layer_2 = presente
ranking temporal por APD de layer_2    = fraco
```

Um espaço pode separar usos semanticamente distintos sem que a média de todas
as distâncias entre períodos seja uma boa medida da mudança de mistura.

Como `layer_2` era secundária na pré-registração, esse resultado deve ser
tratado como hipótese forte para a próxima análise, não como substituição
retroativa da métrica principal.

## Deriva além da mistura de sentidos

Após controlar a distância entre posteriores, pares entre períodos continuaram
um pouco mais distantes:

| Camada | associação residual período-geometria |
|---|---:|
| `layer_1` | 0,036 |
| `layer_2` | 0,051 |

Usando rótulos duros, a distância entre períodos dentro do mesmo sentido
excedeu a distância intraperíodo:

| Camada | excesso médio de distância cosseno |
|---|---:|
| `layer_1` | 0,0054 |
| `layer_2` | 0,0099 |

Esses efeitos são pequenos e consistentes entre palavras, mas sua natureza não
é identificada por esta análise. Deriva contextual, tópico, gênero e
composição documental são hipóteses possíveis, não atribuições estabelecidas.

## Relação com o NO-GO anterior

A comparação escalar entre palavras perguntou:

> palavras com maior APD também têm maior JSD?

A resposta foi não.

A análise atual perguntou:

> dentro de cada palavra, usos semanticamente diferentes ficam mais distantes?

A resposta foi sim.

As conclusões são compatíveis:

```text
geometria contém estrutura de sentido local
APD global mistura sentido com outras fontes de contexto
```

## Limitações

1. Os milhares de pares não são independentes; por isso a inferência foi feita
   sobre 25 coeficientes por palavra.
2. O ConSeC fornece um inventário WordNet, não uma verdade absoluta.
3. O efeito principal é pequeno.
4. A análise de mesmo sentido usa rótulos duros e perde incerteza.
5. `layer_2` foi uma análise secundária.

## Próximo passo

Fazer uma decomposição suave e vetorial da mudança temporal. Para cada sentido
e período, serão estimados centróides ponderados pelo posterior ConSeC.

A mudança do centróide global será decomposta exatamente em:

```text
composição = mudança das proporções dos sentidos
deriva interna = deslocamento dos centróides dentro dos sentidos
```

Uma decomposição simétrica evita escolher D0 ou D1 como referência. Ela
permitirá dizer, por palavra, quanto da resposta do encoder corresponde à
troca de mistura e quanto permanece como deriva contextual dentro do sentido.

## Artefatos

```text
outputs/consec_timeformer_occurrence_alignment/
scripts/evaluate_consec_timeformer_occurrences.py
docs/31-occurrence_level_consec_timeformer_preregistration.md
```
