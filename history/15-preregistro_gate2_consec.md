# 15. Pré-registro da Porta 2

Depois do `GO` de `plane`, o projeto não executou imediatamente o ConSeC nas
outras palavras. Primeiro foram examinados os inventários e congeladas regras
de avaliação.

## Papéis diferentes

`graft_nn` possui sentidos WordNet de corrupção, enxerto médico e ato de
enxertar. Entretanto, muitos contextos históricos usam `graft` para a muda ou
scion, objeto não representado exatamente no inventário. Esses casos foram
marcados como lacuna de inventário.

`chairman_nn` é monossêmico no WordNet. Ele serve como controle de cobertura,
mas não como evidência de desambiguação.

`tree_nn` permite distinguir planta de figura ramificada. O sentido de Sir
Herbert Beerbohm Tree não possui amostra confiável no corpus.

## Conjuntos congelados

Antes de qualquer previsão foram selecionados:

```text
graft/corruption    23
graft/medical       17
tree/diagram        17
tree/plant         138
chairman controle  100
```

Também foram registrados 90 casos botânicos de `graft` como lacuna de
inventário.

## Auditoria

Foi gerada uma ficha cega com 87 itens: todos os casos raros e uma amostra de
30 plantas. O ConSeC não deve ser executado antes dessa auditoria.

Pré-registração completa:
[Porta 2](../docs/20-consec_gate2_preregistration.md).
