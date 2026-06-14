# Protocolo de adjudicação cega para `plane`

## Objetivo

O Gate 1 do WSD externo encontrou 19 ocorrências de `plane` classificadas
heuristicamente como o sentido de ferramenta no período D0. A baixa acurácia
do LMMS nesse subconjunto pode refletir falha do modelo, ruído nos rótulos
heurísticos ou ambos.

Este protocolo mede a qualidade desses rótulos sem alterar retrospectivamente
o resultado original do Gate 1. O NO-GO permanece congelado como resultado do
teste pré-especificado.

## Procedimento

1. Dois anotadores recebem apenas `item_id`, contexto e campos de resposta.
2. A ordem dos itens difere entre as duas fichas.
3. Nenhuma ficha contém rótulo heurístico, previsão, margem ou score do LMMS.
4. Os anotadores trabalham independentemente e usam apenas:
   `tool`, `geometry`, `aircraft`, `botanical` ou `unclear`.
5. Depois das duas anotações, calculam-se acordo bruto e kappa de Cohen.
6. Discordâncias são resolvidas numa terceira ficha de adjudicação.
7. Apenas depois do consenso os rótulos são comparados às previsões do LMMS.

O material entregue aos anotadores está em
`annotations/plane_tool_gate1/`. Eles devem receber somente
`README_ANNOTATORS.md` e sua respectiva planilha.

## Geração reproduzível

```bash
venv/bin/python scripts/prepare_plane_tool_adjudication.py prepare \
  --predictions outputs/external_wsd/lmms_plane_gate1/occurrence_predictions.csv \
  --output-dir annotations/plane_tool_gate1
```

Após o preenchimento:

```bash
venv/bin/python scripts/prepare_plane_tool_adjudication.py summarize \
  --annotator-a annotations/plane_tool_gate1/annotator_a.csv \
  --annotator-b annotations/plane_tool_gate1/annotator_b.csv \
  --output-dir outputs/external_wsd/lmms_plane_gate1/blind_adjudication
```

Se houver apenas um anotador, a análise diagnóstica deve ser identificada
explicitamente como tal:

```bash
venv/bin/python scripts/prepare_plane_tool_adjudication.py single-summary \
  --annotations annotations/plane_tool_gate1/annotator_a.csv \
  --manifest annotations/plane_tool_gate1/manifest.csv \
  --predictions outputs/external_wsd/lmms_plane_gate1/occurrence_predictions.csv \
  --output-dir outputs/external_wsd/lmms_plane_gate1/single_adjudication
```

Nesse desenho não se calcula concordância nem kappa. O uso de tradutor para
compreender o inglês histórico deve ser declarado como parte do protocolo,
desde que a anotação tenha sido concluída sem acesso às previsões do modelo.

## Interpretação

- Alta proporção de consenso `tool` com erro persistente do LMMS confirma uma
  limitação externa de WSD nesse sentido.
- Muitos rótulos `geometry` ou `unclear` indicam que a heurística superestimou
  o sentido de ferramenta.
- A adjudicação diagnostica o Gate 1, mas não deve ser usada para redefinir
  seu limiar depois de observar os resultados.
- O passo experimental seguinte continua sendo testar ConSeC sobre o conjunto
  adjudicado, reportando separadamente o resultado original e o corrigido.

O teste foi executado e está documentado em
[ConSeC no subconjunto adjudicado](18-consec_plane_adjudicated_results.md).
