# Gate 1 completo de `plane` com ConSeC

**Data:** 2026-06-13
**Decisão:** `GO` para um piloto pequeno da Porta 2.

## Objetivo

Depois de o ConSeC reconhecer 14 das 16 ocorrências humanas de ferramenta,
executamos o modelo em todos os três estratos confirmatórios originalmente
definidos para `plane`:

| Estrato | N |
|---|---:|
| D0 geometria | 182 |
| D0 ferramenta | 19 |
| D1 aviação | 208 |
| **Total** | **409** |

O resultado LMMS original permanece congelado como `NO-GO`. Esta execução é
uma replicação com um segundo WSD externo, não uma substituição retrospectiva.

## Método

```text
modelo: ConSeC SemCor+WNGT congelado
commit: 9602b5fd69f57be08a186988d1df34fe4152b63f
checkpoint SHA-256:
92421ed245723964db09ce396f19a0d1e55fe4d6e10d5ecb83278d9bc70ce8af
inventário: cinco sentidos nominais de plane no WordNet
inferência: extração target-only, sem feedback loop
bootstrap: 20.000 réplicas
seed: 20260613
```

Nenhum parâmetro ou limiar foi escolhido usando essas ocorrências.

## Resultado com rótulos originais

| Estrato | Acurácia | IC 95% | Corte | Passou |
|---|---:|---:|---:|---|
| D0 geometria | **1,000** | [1,000; 1,000] | >= 0,75 | sim |
| D0 ferramenta | **0,789** | [0,579; 0,947] | IC acima de 1/3 | sim |
| D1 aviação | **0,990** | [0,976; 1,000] | >= 0,80 | sim |

A macro acurácia foi `0,927`, IC 95% `[0,858; 0,981]`.

Todos os cinco checks da regra original passaram.

## Resultado pós-adjudicação

Nas 19 ocorrências originalmente marcadas como ferramenta, a anotação humana
produziu 16 ferramentas, uma geometria, uma botânica e uma `unclear`. O caso
botânico fica fora do inventário do lema simples `plane`; o caso `unclear`
não é avaliado.

| Estrato | N | Acurácia | IC 95% | Passou |
|---|---:|---:|---:|---|
| D0 geometria | 183 | **0,995** | [0,984; 1,000] | sim |
| D0 ferramenta | 16 | **0,875** | [0,688; 1,000] | sim |
| D1 aviação | 208 | **0,990** | [0,976; 1,000] | sim |

A macro acurácia pós-adjudicação foi:

```text
0,953
IC 95% [0,891; 0,997]
baseline = 1/3
```

## Frase-âncora

```text
plate figure represent an inclined plane

esperado: geometry
predito:  geometry
probabilidade: 0,776
margem: 0,692
```

## Erros

Foram observados cinco erros nos três estratos pós-adjudicação:

- um plano inclinado de transporte foi classificado como `aircraft`;
- duas posições numa enumeração de ferramentas foram classificadas como
  `geometry`;
- duas ocorrências modernas de avião foram classificadas como `geometry`.

Os dois erros modernos têm margens pequenas (`0,077` e `0,052`). Um dos
erros de ferramenta, porém, tem margem alta, mostrando que ainda existem
falhas sistemáticas locais.

## Decisão

O ConSeC passa o Gate 1 completo:

```text
macro IC acima de 1/3: sim
D0 geometria >= 0,75: sim
D0 ferramenta com IC acima de 1/3: sim
D1 aviação >= 0,80: sim
frase-âncora geométrica: sim

Gate 1 ConSeC = GO
```

Isso sustenta três conclusões:

1. um WSD externo congelado pode funcionar no corpus histórico lematizado;
2. a falha do LMMS era principalmente específica do método;
3. ainda não há evidência suficiente para assumir cobertura robusta nos 37
   alvos.

## Próximo passo

Abrir uma Porta 2 pequena, sem engenharia por palavra, com:

```text
graft_nn
chairman_nn
tree_nn
```

Antes da execução, devem ser congelados para cada palavra:

- os sentidos WordNet candidatos;
- regras heurísticas de subconjuntos de alta confiança;
- tamanhos mínimos;
- cortes de acurácia;
- tratamento de sentidos ausentes ou lexicalizados como multiword.

Somente se esse piloto passar devemos discutir expansão para mais palavras.

## Artefatos

```text
scripts/evaluate_consec_plane_adjudicated.py
outputs/external_wsd/consec_plane_gate1_full/predictions.csv
outputs/external_wsd/consec_plane_gate1_full/summary.json
```
