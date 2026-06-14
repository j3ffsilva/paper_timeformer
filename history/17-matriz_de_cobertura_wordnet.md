# 17. Antes da Porta 3: o inventário precisa cobrir o uso

A Porta 2 mostrou que o ConSeC sabe escolher entre sentidos disponíveis. Ela
também mostrou, com `graft` botânico, que o candidato correto pode simplesmente
não existir.

## A nova unidade de análise

Foi criada uma matriz para os 37 alvos do SemEval contendo:

```text
229 sensekeys WordNet 3.0
296 contextos históricos
4 contextos por palavra em cada período
```

As amostras foram sorteadas deterministicamente antes de novas previsões.
Nenhum gold do SemEval ou resultado do ConSeC aparece na ficha.

## Carga de revisão

| Faixa automática | Palavras |
|---|---:|
| monossêmicas | 3 |
| 2-3 sentidos | 13 |
| 4-8 sentidos | 10 |
| 9 ou mais sentidos | 11 |

Essas faixas medem apenas a dificuldade de inspecionar o inventário. Elas não
determinam cobertura. `graft`, com apenas três sentidos, já demonstrou uma
lacuna histórica.

## Evidência transportada

Quatro palavras já possuem decisão anterior:

| Palavra | Cobertura | Papel |
|---|---|---|
| `plane_nn` | suficiente | elegível |
| `tree_nn` | suficiente | elegível |
| `graft_nn` | parcial | elegível apenas nos sentidos cobertos |
| `chairman_nn` | monossêmica e coberta | diagnóstico |

As outras 33 aguardam revisão em nível de palavra.

## Próximo marco

Depois da ficha preenchida, a Porta 3 poderá ser pré-registrada somente para
os alvos `eligible`. Lacunas, controles monossêmicos e casos incertos serão
reportados separadamente.

Protocolo:
[Cobertura WordNet para a Porta 3](../docs/22-wordnet_coverage_gate3_protocol.md).
