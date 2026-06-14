# Pré-registração da Porta 2 com ConSeC

**Data de congelamento:** 2026-06-13
**Estado:** especificação anterior à inferência do ConSeC.

## Pergunta

O desempenho do ConSeC em `plane_nn` generaliza para outras palavras, sem
engenharia posterior por palavra?

O piloto usa três personagens já centrais no projeto:

```text
graft_nn
chairman_nn
tree_nn
```

Eles não desempenham o mesmo papel estatístico.

## Inventários e limitações

### `graft_nn`

O WordNet oferece:

| Sentido | Sensekey |
|---|---|
| enxerto médico | `graft%1:08:00::` |
| corrupção/suborno | `graft%1:04:01::` |
| ato de enxertar | `graft%1:04:00::` |

O corpus histórico usa frequentemente `graft` para a muda ou scion inserida
numa planta. Esse objeto botânico não corresponde exatamente ao “ato de
enxertar”. Esses exemplos serão reportados como **lacuna de inventário** e
não serão convertidos artificialmente em gold do terceiro sensekey.

O teste discriminativo usa apenas:

```text
corruption
medical
```

### `chairman_nn`

O WordNet possui apenas:

```text
chairman%1:18:01:: = pessoa que preside uma organização
```

Logo, `chairman_nn` é controle de cobertura e execução. Com um único
candidato, sua acurácia é tautológica e não entra na decisão GO/NO-GO.

### `tree_nn`

O WordNet oferece:

```text
tree%1:20:00:: = planta
tree%1:25:00:: = figura ramificada a partir de uma raiz
tree%1:18:00:: = Sir Herbert Beerbohm Tree
```

Não foi encontrado subconjunto confiável suficiente para a pessoa. O teste
usa apenas planta e diagrama.

## Seleção congelada

As heurísticas estão implementadas em:

```text
scripts/prepare_consec_gate2.py
```

Regras:

- `graft/corruption`: maior sobreposição exclusiva com vocabulário de
  corrupção em janela de 41 tokens;
- `graft/medical`: maior sobreposição exclusiva com vocabulário médico e
  pelo menos uma pista médica a até quatro tokens do alvo;
- `graft/botanical_inventory_gap`: registrado, mas não avaliado;
- `tree/diagram`: modificador imediatamente anterior entre `family`,
  `genealogical`, `decision`, `parse`, `syntax`, `phylogenetic`,
  `classification`, `binary`, `evolutionary`, `taxonomic`, ou `diagram`
  imediatamente posterior;
- `tree/plant`: pelo menos duas pistas botânicas numa janela de 25 tokens,
  excluindo contextos com `family`;
- `chairman`: amostra determinística de controle.

Para limitar custo sem escolher exemplos semanticamente, `tree/plant` é
limitado a 100 ocorrências por período e `chairman` a 50 por período, com
amostragem determinística pela seed `20260613`.

## Tamanhos mínimos

O piloto só pode ser executado se houver:

| Estrato | Mínimo |
|---|---:|
| `graft/corruption` | 15 |
| `graft/medical` | 15 |
| `tree/diagram` | 10 |
| `tree/plant` | 100 |
| `chairman` controle | 100 |

Os conjuntos gerados antes da inferência possuem:

| Estrato | N |
|---|---:|
| `graft/corruption` | 23 |
| `graft/medical` | 17 |
| `tree/diagram` | 17 |
| `tree/plant` | 138 |
| `chairman` controle | 100 |
| `graft` botânico, lacuna de inventário | 90 |

Todos os mínimos foram satisfeitos.

## Métricas e decisão

Cada sentido discriminativo será avaliado por acurácia e bootstrap de 20.000
réplicas.

Critérios:

```text
graft/corruption >= 0,75
graft/medical >= 0,75
tree/diagram >= 0,75
tree/plant >= 0,90
macro dos quatro sentidos: limite inferior do IC 95% > 1/3
```

Além disso:

- nenhuma mudança de regra é permitida após ver previsões;
- `chairman` e lacunas de inventário são reportados, mas não podem salvar nem
  reprovar a porta;
- os contextos selecionados devem passar por uma auditoria cega humana antes
  de serem tratados como avaliação confirmatória;
- erros evidentes da heurística serão reportados em análise separada, como em
  `plane`, sem apagar o resultado original.

## Regra de parada

- `GO`: todos os quatro cortes e o IC macro passam;
- `NO-GO`: qualquer corte falha;
- `INCONCLUSIVO`: auditoria humana mostra precisão insuficiente dos golds ou
  cobertura menor que os mínimos pré-especificados.

Mesmo em caso de `GO`, o próximo passo não será automaticamente executar os
37 alvos. Primeiro será necessário avaliar cobertura WordNet e disponibilidade
de controles para cada palavra.

## Comando de preparação

```bash
PYTHONPATH=src venv/bin/python scripts/prepare_consec_gate2.py \
  --corpus-dir data/processed/semeval2020_task1/eng_lemma/corpus \
  --output-dir outputs/external_wsd/consec_gate2_preregistered \
  --annotation-dir annotations/consec_gate2_audit \
  --seed 20260613
```

## Auditoria humana

Para reduzir carga sem selecionar exemplos depois das previsões, a ficha cega
contém:

```text
todos os graft/corruption
todos os graft/medical
todos os tree/diagram
15 tree/plant de cada período
```

O rótulo heurístico não aparece em nenhum arquivo entregue ao anotador. Ele
permanece apenas no conjunto técnico congelado sob `outputs/`. A ficha contém
87 itens e a auditoria ocorre antes da execução do ConSeC.
