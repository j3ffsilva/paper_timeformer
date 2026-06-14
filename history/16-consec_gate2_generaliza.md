# 16. A Porta 2 mostra que o ConSeC generaliza

A Porta 1 havia mostrado que o ConSeC reconhece os sentidos históricos de
`plane`. A Porta 2 perguntou se o resultado sobreviveria à troca de palavra.

## O teste

Antes das previsões, foram congelados quatro contrastes:

```text
graft: corrupção versus enxerto médico
tree:  planta versus diagrama
```

`chairman` entrou apenas como controle monossêmico. O uso botânico de `graft`
foi registrado como lacuna do inventário, sem ser forçado a um sentido
WordNet inadequado.

A auditoria humana concordou com 84 de 87 rótulos heurísticos. As três
discordâncias foram preservadas na análise.

## Resultado

| Sentido pós-auditoria | N | Acurácia |
|---|---:|---:|
| `graft/corruption` | 22 | 100,0% |
| `graft/medical` | 17 | 100,0% |
| `tree/diagram` | 16 | 87,5% |
| `tree/plant` | 138 | 98,6% |

A macro foi `96,5%`, com IC 95% `[91,8%; 99,8%]`. Todos os cortes
pré-registrados passaram.

```text
Porta 2 ConSeC = GO
```

## O limite que ficou concreto

Nos 90 exemplos botânicos de `graft`, o sentido desejado não existia
adequadamente no inventário. O modelo distribuiu as previsões entre ato de
enxertar, enxerto médico e corrupção.

Isso separa duas perguntas:

```text
1. o modelo escolhe bem entre sentidos disponíveis?
2. o inventário contém o sentido histórico necessário?
```

A Porta 2 respondeu “sim” à primeira nos contrastes avaliados. `graft`
botânico respondeu “nem sempre” à segunda.

## Próxima decisão

O projeto não deve saltar diretamente para os 37 alvos. O próximo passo é
mapear a cobertura WordNet de cada palavra e período. Apenas o subconjunto com
inventário adequado e ocorrências auditáveis deve entrar numa Porta 3
pré-registrada para estimar distribuições temporais de sentido.

Relatório completo:
[Resultados da Porta 2](../docs/21-consec_gate2_results.md).
