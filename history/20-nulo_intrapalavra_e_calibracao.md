# 20. Quanto da divergência surgiria por acaso?

A Porta 3 mostrou que distribuições de sentido previstas pelo ConSeC
acompanham o gold do SemEval. A replicação mostrou que o resultado não dependia
de uma única amostra. Restava uma pergunta concreta:

> Se dividirmos aleatoriamente as ocorrências da mesma palavra em dois grupos,
> quanta JSD aparece mesmo sem respeitar o tempo?

## O nulo intrapalavra

Cada palavra tinha 50 ocorrências, 25 de D0 e 25 de D1. Em cada uma das três
amostras, os rótulos temporais foram embaralhados 20.000 vezes, mantendo dois
grupos de 25.

Isso produziu uma divergência basal específica para cada palavra:

```text
JSD excedente = JSD temporal observada - média da JSD embaralhada
```

Uma palavra com muitos sentidos candidatos pode apresentar divergência
amostral maior mesmo sem mudança temporal. O nulo mede diretamente esse piso.

## O que aconteceu

| Medida | Spearman médio com gold | Associação média com nº sentidos |
|---|---:|---:|
| JSD bruta | 0,585 | 0,509 |
| JSD excedente | 0,410 | 0,165 |
| z do nulo | 0,319 | 0,034 |

A JSD excedente permaneceu positivamente associada ao gold nas três amostras.
Ao mesmo tempo, sua dependência do tamanho do inventário caiu muito.

O resultado revela uma troca:

- a JSD bruta oferece o ranking mais estável e informativo;
- a JSD excedente é mais limpa como controle de amostragem;
- o z quase elimina o efeito do inventário, mas também remove parte do sinal.

## Palavras individuais

Seis alvos passaram a correção de múltiplos testes nas três amostras:

```text
ball, gas, plane, prop, record, stab
```

Outros passaram apenas em uma ou duas. Isso ensina que a significância de uma
palavra isolada é instável com 25 ocorrências por período. O resultado central
continua sendo o ranking agregado e replicado, acompanhado da incerteza de
cada alvo.

## A nova posição do projeto

O projeto agora dispõe de duas réguas complementares:

```text
TimeFormer -> como as representações internas respondem ao tempo
ConSeC     -> como distribuições explícitas de sentido mudam
```

O passo seguinte não é criar outra métrica de WSD. É verificar se a resposta
adaptativa entre as camadas do TimeFormer se alinha à mudança explícita de
sentidos medida pelo ConSeC.

Detalhes: `docs/28-consec_within_word_null_results.md`.
