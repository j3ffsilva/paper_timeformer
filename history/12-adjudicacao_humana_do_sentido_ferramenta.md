# 12. Adjudicação humana do sentido de ferramenta

O Gate 1 havia produzido um resultado assimétrico para `plane_nn`: o LMMS
reconheceu quase perfeitamente geometria em D0 e aviação em D1, mas acertou
apenas 21,1% do subconjunto heurístico de ferramenta.

Ainda havia uma dúvida importante: o modelo realmente falhava ou a heurística
que criou o rótulo de referência estava errada?

## Verificação cega

As 19 ocorrências foram embaralhadas e apresentadas sem previsões, scores ou
rótulos heurísticos. Um anotador classificou cada contexto como:

```text
tool
geometry
aircraft
botanical
unclear
```

O anotador usou tradução automática para compreender o inglês histórico, mas
não para decidir o sentido da palavra-alvo.

## Resultado

Dezenove ocorrências produziram:

```text
tool       16
geometry    1
botanical   1
unclear     1
```

A heurística estava correta em 84,2% dos casos. Entretanto, depois de usar os
rótulos humanos, o LMMS acertou apenas 2 dos 16 exemplos de ferramenta
(12,5%). Doze foram confundidos com geometria.

## O que mudou no diagnóstico

Antes da adjudicação, a falha podia ser dividida de modo desconhecido entre
ruído da heurística e limitação do modelo. Depois dela:

- existe algum ruído de referência;
- esse ruído é pequeno demais para explicar o resultado;
- o problema central é a representação do sentido raro de ferramenta pelo
  LMMS em contextos históricos.

Como houve apenas um anotador, o resultado é diagnóstico e não possui medida
de concordância. Isso deve ser declarado no artigo.

## Decisão

O NO-GO do Gate 1 permanece. O próximo e último controle desta linha é aplicar
um segundo WSD externo congelado, ConSeC, aos mesmos exemplos. Se ele também
falhar, encerra-se a tentativa de usar um atlas WordNet geral como arquitetura
principal para os 37 alvos.

Relatório completo:
[resultado da adjudicação](../docs/17-plane_tool_single_adjudication_results.md).
