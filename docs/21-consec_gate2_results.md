# Resultados da Porta 2 com ConSeC

**Data:** 2026-06-13
**Decisão pré-registrada:** `GO`

## Pergunta

O desempenho do ConSeC observado em `plane_nn` generaliza para outras
palavras, sem ajuste do modelo ou das regras depois das previsões?

A Porta 2 foi congelada em
[`20-consec_gate2_preregistration.md`](20-consec_gate2_preregistration.md)
antes da inferência.

## Auditoria humana

A ficha cega continha 87 contextos. Todas as linhas foram preenchidas antes da
execução do ConSeC.

```text
acordos com a heurística:       84/87 = 96,6%
graft/corruption:               22/23
graft/medical:                  17/17
tree/diagram:                   16/17
tree/plant auditado:            29/30
```

As três discordâncias foram preservadas:

| Item | Heurística | Humano | Tratamento |
|---|---|---|---|
| `G2-009` | corrupção | `unclear` | excluído da análise confirmatória |
| `G2-019` | planta | `other`, confiança baixa | excluído da análise confirmatória |
| `G2-072` | diagrama | planta, confiança alta | mantido como planta |

O contexto de `G2-072` contém `draw genealogical tree`, o que sugere um
provável erro humano. Mesmo assim, ele não foi corrigido depois da anotação. O
resultado pós-auditoria usa literalmente a ficha preenchida.

Nos itens não auditados, o rótulo heurístico congelado foi mantido.

## Execução

Foi usado o checkpoint oficial:

```text
ConSeC SemCor+WNGT
commit 9602b5fd69f57be08a186988d1df34fe4152b63f
SHA256 92421ed245723964db09ce396f19a0d1e55fe4d6e10d5ecb83278d9bc70ce8af
```

O modelo permaneceu congelado. A inferência usou o inventário WordNet
pré-especificado e a extração oficial apenas no alvo, sem feedback loop.

## Resultado confirmatório

### Rótulos originais congelados

| Estrato | N | Acurácia | IC 95% | Corte |
|---|---:|---:|---:|---:|
| `graft/corruption` | 23 | 1,000 | [1,000; 1,000] | 0,75 |
| `graft/medical` | 17 | 1,000 | [1,000; 1,000] | 0,75 |
| `tree/diagram` | 17 | 0,882 | [0,706; 1,000] | 0,75 |
| `tree/plant` | 138 | 0,993 | [0,978; 1,000] | 0,90 |

Macro:

```text
0,969
IC 95% = [0,925; 1,000]
```

### Depois da auditoria

| Estrato | N | Acurácia | IC 95% | Corte | Passou |
|---|---:|---:|---:|---:|---|
| `graft/corruption` | 22 | 1,000 | [1,000; 1,000] | 0,75 | sim |
| `graft/medical` | 17 | 1,000 | [1,000; 1,000] | 0,75 | sim |
| `tree/diagram` | 16 | 0,875 | [0,688; 1,000] | 0,75 | sim |
| `tree/plant` | 138 | 0,986 | [0,964; 1,000] | 0,90 | sim |

Macro:

```text
0,965
IC 95% = [0,918; 0,998]
limite inferior exigido > 1/3
```

Todos os cortes passaram. Portanto:

```text
Porta 2 ConSeC = GO
```

## Erros de `tree`

Nos rótulos pós-auditoria, o ConSeC errou dois diagramas como planta:

- `Pompey’s family tree`;
- `the old trunk of the family tree ... bear ... bough`.

São metáforas genealógicas com vocabulário botânico forte. O modelo também
predisse `G2-072` como diagrama, enquanto a ficha humana o marcou como planta.
Esse caso conta como erro do modelo na análise pós-auditoria.

O outro erro de planta ocorreu em uma metáfora bíblica de uma árvore de
monarquia, que o modelo classificou como diagrama. Esses casos mostram que a
fronteira planta/estrutura abstrata permanece difícil quando o texto explora
simultaneamente os dois campos lexicais.

## Controles e lacuna de inventário

### `chairman`

As 100 ocorrências receberam o único sentido candidato:

```text
presiding_officer: 100/100
```

Isso confirma a execução e a cobertura, mas é tautológico e não entra no
`GO`.

### `graft` botânico

Os 90 usos botânicos foram mantidos fora da decisão porque o objeto botânico
enxertado não está representado adequadamente pelo inventário nominal simples.

| Previsão disponível | N |
|---|---:|
| ato de enxertar | 54 |
| enxerto médico | 35 |
| corrupção | 1 |

Esse resultado não é uma acurácia de 60%. Não há gold candidato correto para
o sentido pretendido. Ele demonstra que **cobertura do inventário é condição
anterior à desambiguação**.

## Interpretação

A Porta 2 responde positivamente à pergunta estreita:

> Quando o sentido histórico relevante existe no inventário WordNet e há um
> subconjunto contextual auditável, o ConSeC congelado generaliza de
> `plane_nn` para novas palavras e novos contrastes de sentido.

Ela não demonstra ainda que:

- os 37 alvos possuem cobertura adequada;
- heurísticas confiáveis podem ser construídas para todos os sentidos;
- distribuições completas de sentido podem ser estimadas sem viés de seleção;
- a divergência entre distribuições WSD correlaciona com o gold graduado.

## Próximos passos

1. Construir uma matriz de cobertura para os 37 alvos:
   `palavra × sensekey × descrição × ocorrências plausíveis em D0/D1`.
2. Classificar cada alvo como:
   `cobertura suficiente`, `monossêmico`, `lacuna de inventário` ou
   `sem subconjunto auditável`.
3. Pré-registrar uma Porta 3 apenas no subconjunto com cobertura suficiente,
   sem criar regras depois de observar previsões.
4. Estimar por período as distribuições de sentidos e uma divergência
   temporal, com bootstrap por documento/ocorrência.
5. Manter separadamente os casos de lacuna, como `graft` botânico, em vez de
   forçá-los ao sensekey mais próximo.
6. Só depois comparar a medida WSD temporal com os 37 gold scores do SemEval.

## Artefatos

```text
scripts/evaluate_consec_gate2.py
tests/test_evaluate_consec_gate2.py
annotations/consec_gate2_audit/
outputs/external_wsd/consec_gate2/predictions.csv
outputs/external_wsd/consec_gate2/summary.json
```
