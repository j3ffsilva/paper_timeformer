# Decomposição vetorial suave por sentidos

**Data:** 2026-06-14
**Decisão pré-registrada:** contribuição geométrica de composição estabelecida.

## Pergunta

Quanto do deslocamento temporal do centróide contextual aponta na direção
explicada pela mudança da mistura de sentidos ConSeC?

A decomposição simétrica foi aplicada às 25 palavras, em três amostras ConSeC,
duas seeds TimeFormer e duas camadas. Foram executados 2.000 embaralhamentos
por alvo e combinação.

## Validação algébrica

Para todas as 300 decomposições:

```text
T = C + D
```

O maior erro relativo de reconstrução foi:

```text
3,21 × 10^-7
```

Isso passa o limite pré-registrado de `10^-6`.

## Resultado principal: `layer_2`

```text
share de composição observada média = 0,04835
share de deriva média                = 0,95165
share de composição nula média       ≈ -0,00006
excesso médio de composição          = 0,04841
mediana do excesso                   = 0,02584
IC bootstrap 95%                     = [0,02379; 0,08080]
palavras com excesso positivo        = 23/25
p por inversão de sinais             = 0,00005
```

Todos os critérios pré-registrados passaram:

```text
média positiva nas seis combinações = sim
média agregada positiva              = sim
p < 0,05                             = sim
pelo menos 15 palavras positivas     = sim
erro de reconstrução < 10^-6         = sim
```

Portanto:

```text
Contribuição geométrica da composição = GO
```

## Replicação

| ConSeC | TimeFormer | excesso médio | mediana | positivas |
|---|---|---:|---:|---:|
| 20260613 | 1000 | 0,060 | 0,043 | 23 |
| 20260613 | 1001 | 0,061 | 0,042 | 22 |
| 20260614 | 1000 | 0,033 | 0,005 | 16 |
| 20260614 | 1001 | 0,032 | 0,005 | 16 |
| 20260615 | 1000 | 0,053 | 0,029 | 19 |
| 20260615 | 1001 | 0,052 | 0,028 | 19 |

As duas seeds TimeFormer são quase idênticas. A variação principal continua
vindo da amostragem das ocorrências.

## Comparação entre camadas

| Camada | composição média | excesso médio | IC 95% | positivas |
|---|---:|---:|---:|---:|
| `layer_1` | 0,031 | 0,031 | [0,009; 0,061] | 18/25 |
| `layer_2` | 0,048 | 0,048 | [0,024; 0,081] | 23/25 |

A diferença pareada `layer_2 - layer_1` foi:

```text
média = 0,017
IC 95% = [0,004; 0,032]
p por inversão de sinais = 0,009
```

Essa comparação é secundária, mas é coerente com o alinhamento local mais
forte de `layer_2` encontrado na etapa anterior.

## Relação com a mudança de sentidos

Em `layer_2`:

```text
Spearman(excesso de composição, JSD ConSeC) = 0,615
```

Palavras com maior mudança explícita da mistura tendem a apresentar maior
contribuição vetorial de composição.

Exemplos:

| Palavra | excesso de composição | JSD média |
|---|---:|---:|
| `plane_nn` | 0,341 | 0,428 |
| `multitude_nn` | 0,177 | 0,040 |
| `gas_nn` | 0,147 | 0,150 |
| `record_nn` | 0,069 | 0,178 |
| `player_nn` | -0,007 | 0,029 |
| `donkey_nn` | -0,002 | 0,002 |

`plane_nn` é o caso mais limpo: cerca de 34% do deslocamento direcional está
alinhado à troca da composição de sentidos. `multitude_nn` mostra que a
relação não é determinística e requer inspeção qualitativa.

## O significado da componente complementar

A share média de deriva em `layer_2` é aproximadamente `0,952`. Isso **não**
significa que 95% da mudança seja mudança semântica intrassentido.

Esta análise não identifica a natureza dessa componente. Entre as hipóteses
possíveis estão:

```text
tópico
gênero textual
sintaxe
domínio
mudança contextual dentro do sentido
erros ou lacunas do inventário WordNet
ruído amostral
```

A conclusão defensável é:

> A mudança da mistura de sentidos contribui de forma pequena, consistente e
> acima do nulo para a direção do deslocamento vetorial. A natureza da
> componente complementar permanece indeterminada pelo método.

## Relação com os resultados anteriores

As três etapas agora formam uma sequência coerente:

1. scores agregados APD e JSD não ordenam palavras da mesma forma;
2. dentro da palavra, distância de sentido acompanha distância vetorial;
3. a mudança da mistura de sentidos explica uma parte pequena, mas mensurável,
   do deslocamento temporal do centróide.

Isso esclarece por que APD e JSD não são substitutos:

```text
APD mede deslocamento contextual amplo;
JSD mede recomposição em um inventário de sentidos
```

O TimeFormer não falha por medir uma mudança mais ampla. Essa é sua saída
principal. A decomposição apenas testa quanto dela recebe uma atribuição
automática mais estrita segundo ConSeC/WordNet.

## Próximo passo

Antes de ampliar o modelo, a decomposição deve receber incerteza por
reamostragem de ocorrências:

1. bootstrap estratificado dentro de palavra e período;
2. intervalos para `share_C`, `share_D` e excesso contra nulo;
3. estabilidade dos casos extremos, especialmente `plane`, `multitude`,
   `gas`, `record`, `player` e `donkey`;
4. inspeção qualitativa das contribuições por sentido.

Depois disso, os resultados já sustentam uma seção metodológica integrada do
artigo.

## Artefatos

```text
outputs/consec_timeformer_soft_decomposition/
scripts/decompose_consec_timeformer_change.py
docs/33-soft_sense_vector_decomposition_preregistration.md
```
