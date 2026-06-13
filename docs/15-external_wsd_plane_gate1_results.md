# Porta 1 de WSD externo: LMMS-SP congelado em `plane`

**Data:** 2026-06-13  
**Decisão:** `NO-GO` estrito na Porta 1; não escalar ainda para 37 alvos.

## Pergunta

Testamos se uma régua externa fixa, treinada para WSD e sem qualquer ajuste no
SemEval, consegue ler os sentidos de `plane_nn` no corpus lematizado:

```text
D0 = 1810-1860
D1 = 1960-2010
```

Os subconjuntos e cortes foram definidos antes da execução:

| Subconjunto | N | Corte |
|---|---:|---:|
| D0 geometria | 182 | acurácia >= 0,75 |
| D0 ferramenta | 19 | desempenho substantivamente acima do acaso, com IC |
| D1 avião | 208 | acurácia >= 0,80 |

Também exigimos acurácia macro com IC 95% acima de `1/3` e classificação
geométrica da frase histórica:

```text
plate figure represent an inclined plane
```

## Método

Usamos LMMS-SP WSD com:

```text
encoder: bert-large-cased congelado
inventário: WordNet 3.0
vetores: lmms-sp-wsd.bert-large-cased.vectors
pooling: média dos WordPieces do alvo
camadas: soma ponderada pelo perfil WSD publicado
decisão: vizinho mais próximo por cosseno entre os cinco sentidos nominais
bootstrap: 20.000 réplicas, seed 20260613
```

Os cinco sensekeys de `plane` foram agregados apenas para relatório:

```text
aircraft: plane%1:06:01::
geometry: plane%1:25:00::
tool:     plane%1:06:00:: e plane%1:06:02::
other:    plane%1:26:00::
```

Os rótulos heurísticos serviram somente para avaliação. Não treinaram,
selecionaram checkpoint ou calibraram o LMMS.

Implementação:

```text
scripts/evaluate_external_wsd_plane.py
tests/test_external_wsd_plane.py
```

Artefatos gerados:

```text
outputs/external_wsd/lmms_plane_gate1/summary.json
outputs/external_wsd/lmms_plane_gate1/metrics.csv
outputs/external_wsd/lmms_plane_gate1/occurrence_predictions.csv
```

## Resultados

| Subconjunto | Acurácia | IC 95% | Corte | Passou |
|---|---:|---:|---:|---|
| D0 geometria | **0,984** | [0,962; 1,000] | >= 0,75 | sim |
| D0 ferramenta | **0,211** | [0,053; 0,421] | acima de 1/3 | **não** |
| D1 avião | **1,000** | [1,000; 1,000] | >= 0,80 | sim |

A acurácia macro estratificada foi:

```text
0,731
IC 95% [0,677; 0,800]
baseline = 1/3
```

A frase-âncora foi corretamente classificada:

```text
esperado: geometry
predito:  geometry
sensekey: plane%1:25:00::
margem:   0,041
```

Portanto, quatro verificações passaram e uma falhou. Pela regra conjuntiva
pré-definida:

```text
Gate 1 = NO-GO
```

## Auditoria da falha em ferramenta

Nas 19 ocorrências heurísticas de D0:

| Predição | N |
|---|---:|
| geometry | 12 |
| tool | 4 |
| aircraft | 3 |

Há ruído visível no subconjunto. Um exemplo contém `plane tree`, mas recebeu
rótulo heurístico de ferramenta por palavras como `timber`. Outros são
inequivocamente ferramentas históricas, incluindo `mould plane`, `bench
planes`, `jack plane` e `smooth plane`, e ainda assim foram frequentemente
classificados como geometria.

Logo, o resultado não pode ser explicado apenas por erro dos rótulos:

1. a heurística de avaliação tem falsos positivos;
2. LMMS também apresenta dificuldade real com o sentido raro e histórico de
   ferramenta;
3. remover exemplos depois de ver as predições invalidaria a Porta 1
   confirmatória.

## Interpretação

O resultado rejeita duas narrativas extremas.

Não é verdade que WSD externo seja incompatível com o corpus lematizado:
geometria e avião foram lidos quase perfeitamente, a oposição temporal
principal foi recuperada e a régua não reinterpretou a frase histórica.

Também não é verdade que o LMMS atual seja suficiente como atlas lexical geral:
ele falhou justamente no sentido raro, que é o caso em que cobertura de cauda
longa e robustez histórica mais importam.

Assim, este experimento demonstra viabilidade parcial da régua fixa, mas não
autoriza avançar diretamente para a Porta 2 ou para os 37 alvos.

## Próximos passos

1. **Congelar este resultado como confirmatório.** Não mudar palavras-chave,
   cortes ou exemplos da Porta 1 original.
2. **Fazer adjudicação cega das 19 ocorrências de ferramenta.** Dois
   anotadores devem rotular `tool`, `geometry`, `aircraft`, `botanical` ou
   `unclear` sem ver as predições LMMS. Isso mede quanto da falha pertence à
   heurística, mas não substitui o `NO-GO` já observado.
3. **Testar um segundo WSD externo congelado no mesmo conjunto adjudicado.**
   A melhor opção é ConSeC, sem ajuste no SemEval. BEM não é atualmente um
   baseline operacional porque o checkpoint oficial deixou de ser
   publicamente recuperável.
4. **Aplicar uma regra de parada.** Se LMMS e ConSeC falharem nos exemplos
   adjudicados de ferramenta, não escalar o atlas WordNet para o benchmark.
   A conclusão será que sentidos históricos raros exigem adaptação de domínio,
   inventário histórico ou supervisão adicional.
5. **Só abrir a Porta 2 se um modelo externo passar.** Nesse caso, manter o
   encoder e o inventário fixos e testar `plane`, `graft`, `chairman` e `tree`
   sem engenharia por palavra.

O experimento de maior valor informacional agora não é mais treino temporal:
é separar, de forma cega, **erro da heurística** de **falha real de WSD na
cauda histórica**, e então verificar se o efeito se replica em um segundo
modelo externo.

## Reprodutibilidade e fontes

Vetores oficiais LMMS-SP BERT-LARGE-CASED:

```text
https://figshare.com/articles/dataset/LMMS-SP_BERT-LARGE-CASED/21975734
arquivo 38999351
MD5 73c920dce40e7839a5e77d69a3047963
licença dos vetores: CC BY 4.0
```

Código e pesos de camadas publicados pelo LMMS:

```text
https://github.com/danlou/LMMS
licença do código: GPLv3
```

Nossa implementação foi escrita independentemente com `transformers`; não
incorpora código do repositório GPL. Os pesos numéricos públicos do perfil WSD
estão registrados como constante para tornar a execução autocontida.

Comando:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
venv/bin/python scripts/evaluate_external_wsd_plane.py \
  --corpus-dir data/processed/semeval2020_task1/eng_lemma/corpus \
  --vectors outputs/external_wsd/lmms/lmms-sp-wsd.bert-large-cased.vectors.zip \
  --output-dir outputs/external_wsd/lmms_plane_gate1 \
  --batch-size 4 \
  --max-length 128 \
  --n-bootstrap 20000 \
  --seed 20260613 \
  --device cuda
```
