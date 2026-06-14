# Guia de anotação da Porta 2

## O que preencher

Preencha somente:

```text
annotator_a.csv
```

Em cada linha, complete `label`, `confidence` e, quando necessário, `notes`.
Não altere `item_id`, `target` ou `context`.

O alvo aparece entre colchetes: `[graft]` ou `[tree]`.

## Regras gerais

Classifique o significado que a palavra possui naquele contexto, não o tema
geral do texto. Use exatamente os rótulos indicados abaixo, em inglês e
minúsculas.

Pode usar Google Translate ou dicionário para compreender o inglês histórico.
Para reduzir a influência do tradutor, substitua `[graft]` ou `[tree]` por
`[PALAVRA]` antes de traduzir. Não pergunte ao tradutor, chatbot ou mecanismo
de busca qual é o sentido correto.

Não consulte previsões do ConSeC, rótulos heurísticos ou resultados anteriores
durante a anotação.

## Como classificar `[graft]`

### `corruption`

Use quando significa vantagem ilícita, suborno, propina ou dinheiro obtido por
abuso de poder.

Pistas comuns:

```text
bribe, corruption, police, government, payment, scandal, political
```

Exemplo inventado:

```text
the official receive money through [graft] and bribery
```

### `medical`

Use quando significa tecido, pele, osso, órgão, vaso ou material transplantado
ou implantado no corpo.

Pistas comuns:

```text
skin, bone, tissue, transplant, surgery, patient, artery, kidney
```

Exemplo inventado:

```text
the surgeon place a skin [graft] over the wound
```

### `botanical`

Use quando se refere a enxertia de plantas, à muda ou parte vegetal inserida
em outra planta, ao scion, stock, broto ou ramo enxertado.

Pistas comuns:

```text
tree, stock, scion, bud, branch, bark, orchard, fruit
```

Exemplo inventado:

```text
the gardener insert the [graft] into the stock
```

### `other`

Use quando o sentido está compreensível, mas não pertence às três categorias
anteriores. Por exemplo, uma junção ou acréscimo não médico e não botânico.

### `unclear`

Use quando o trecho não fornece informação suficiente ou permanece ambíguo
mesmo após tradução. Não use `unclear` apenas porque o inglês está estranho;
use-o quando realmente não for possível decidir o sentido.

## Como classificar `[tree]`

### `plant`

Use para uma árvore literal: planta lenhosa, tronco, galhos, folhas, raízes,
frutos, floresta ou partes físicas de uma árvore.

Exemplo inventado:

```text
the bird build its nest in the [tree]
```

Expressões como `tree stump`, `tree trunk` e `fruit tree` também são `plant`.

### `diagram`

Use para uma estrutura abstrata que se ramifica a partir de uma origem. Inclui
árvore genealógica, árvore familiar, árvore evolutiva, árvore sintática,
árvore de decisão e diagramas de diretórios.

Exemplo inventado:

```text
her ancestors appear in the family [tree]
```

Mesmo sem um desenho visível, `family tree` é `diagram`, não `plant`.

### `person`

Use somente quando `Tree` for sobrenome ou nome de uma pessoa, especialmente
o ator e produtor teatral Sir Herbert Beerbohm Tree. Não use `person` apenas
porque há uma pessoa perto de uma árvore literal.

### `other`

Use quando o sentido está claro, mas não corresponde a planta, estrutura
ramificada ou pessoa.

### `unclear`

Use quando não houver contexto suficiente para decidir ou quando duas
interpretações continuarem igualmente plausíveis.

## Confiança

Preencha `confidence` com:

- `high`: o contexto torna o sentido praticamente inequívoco;
- `medium`: há uma interpretação claramente mais provável, mas sobra dúvida;
- `low`: decisão frágil; outra interpretação ainda parece possível.

Se usar `unclear`, normalmente a confiança deve ser `medium` ou `high` na
avaliação de que o contexto é insuficiente. `low` significa que você escolheu
um rótulo, mas com pouca segurança.

## Notas

O campo `notes` é opcional. Use-o para registrar:

- dúvida entre dois rótulos;
- palavra ou construção difícil;
- necessidade de tradução;
- motivo para `other` ou `unclear`.

Não é necessário registrar notas nos casos evidentes.

## Exemplo de preenchimento

```csv
item_id,target,context,label,confidence,notes
G2-999,graft_nn,the surgeon use a skin [graft],medical,high,
G2-998,tree_nn,the names appear on the family [tree],diagram,high,
```

## Antes de terminar

Confira se:

1. todas as 87 linhas têm `label`;
2. todos os rótulos usam a grafia exata deste guia;
3. toda linha possui `confidence`;
4. nenhuma célula de contexto ou identificação foi modificada.
