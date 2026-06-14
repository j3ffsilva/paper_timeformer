# Nulo intrapalavra da Porta 3

**Data:** 2026-06-14
**Decisão pré-registrada:** controle útil.

## Pergunta

A divergência entre D0 e D1 permanece associada ao gold depois de descontar
a JSD esperada ao dividir aleatoriamente as mesmas ocorrências?

Foram reutilizadas as três amostras da Porta 3. Para cada palavra e seed, os
rótulos de período das 50 ocorrências foram permutados 20.000 vezes,
preservando grupos de tamanho 25/25.

## Resultado agregado

| Seed | JSD bruta × gold | JSD excedente × gold | z nulo × gold |
|---|---:|---:|---:|
| 20260613 | 0,586 | 0,319 | 0,330 |
| 20260614 | 0,549 | 0,444 | 0,269 |
| 20260615 | 0,621 | 0,466 | 0,357 |
| **Média** | **0,585** | **0,410** | **0,319** |

Todos os critérios pré-registrados passaram:

```text
JSD excedente positiva nas três seeds = sim
média da correlação positiva          = sim
associação com nº de sentidos reduzida = sim
```

Portanto:

```text
Nulo intrapalavra = ÚTIL
```

## Controle do tamanho do inventário

Correlação absoluta média com o número de sentidos WordNet:

| Score | |Spearman| médio |
|---|---:|
| JSD bruta | 0,509 |
| JSD excedente | 0,165 |
| z nulo | 0,034 |

O nulo remove a maior parte da associação com o tamanho do inventário. Essa
correção tem custo: a correlação média com o gold cai de `0,585` para `0,410`
no excedente e para `0,319` no z.

Isso não invalida a JSD bruta. Mostra que ela combina pelo menos dois
componentes:

```text
divergência temporal observada
+ divergência basal de amostragem/inventário
```

## Significância por palavra

Após Benjamini-Hochberg, houve 9, 8 e 10 alvos confirmatórios significativos
nas três seeds. Seis passaram em todas:

```text
ball_nn
gas_nn
plane_nn
prop_nn
record_nn
stab_nn
```

Treze palavras passaram em pelo menos uma seed. Logo, o teste individual é
mais sensível à amostra do que a correlação agregada e não deve ser usado como
classificador binário definitivo de palavra mudada.

`tree_nn`, por exemplo, passou em apenas uma seed. Esse caso ilustra por que
uma descoberta isolada não substitui estabilidade entre amostras.

## Estabilidade do score corrigido

| Seeds | Spearman da JSD excedente |
|---|---:|
| 20260613 × 20260614 | 0,527 |
| 20260613 × 20260615 | 0,718 |
| 20260614 × 20260615 | 0,713 |

A correção reduz a estabilidade em relação à JSD bruta (`0,808–0,885`), pois
subtrai duas quantidades estimadas com apenas 50 ocorrências. Ainda assim, o
ranking corrigido mantém concordância positiva substancial.

## Decisão metodológica

Usaremos três papéis diferentes:

1. **JSD bruta:** score principal de ranking, por ser mais estável e mais
   associado ao gold;
2. **JSD excedente:** controle confirmatório contra divergência basal e tamanho
   do inventário;
3. **z nulo:** diagnóstico conservador de excepcionalidade por palavra, não
   substituto automático do ranking principal.

Nenhum p-valor individual será interpretado sem replicação entre amostras.

## Próximo experimento

O próximo passo é integrar o instrumento externo de sentido ao TimeFormer.
Sem ajustar modelos ou escolher readouts pelo gold, compararemos por palavra:

```text
ConSeC: JSD bruta e JSD excedente
TimeFormer: layer 1 e resposta adaptativa layer 2 - layer 1
```

A análise deve responder se a reorganização da camada superior acompanha
mudanças explícitas de distribuição de sentidos ou apenas deriva contextual.

## Artefatos

```text
outputs/external_wsd/consec_gate3_within_word_null/
scripts/evaluate_consec_within_word_null.py
docs/27-consec_within_word_null_preregistration.md
```
