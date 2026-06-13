# Conceitos 7 — O que está sendo medido?

Este é o conceito central para compreender por que o projeto mudou tantas
vezes de direção. Vários experimentos encontraram **mudança real**, mas não
necessariamente a mudança que a pergunta científica queria.

## Quatro níveis que não devem ser confundidos

Considere `plane` em dois períodos. Há pelo menos quatro objetos diferentes:

```text
1. corpus:         quais documentos e gêneros aparecem?
2. contexto/uso:   quais palavras, construções e tópicos cercam "plane"?
3. representação: que vetor um modelo atribui à ocorrência?
4. sentido:        geometria, avião, ferramenta, nível de existência etc.
```

Eles se influenciam, mas não são equivalentes.

### Mudança de corpus

É uma alteração na composição dos dados:

```text
D0: mais tratados técnicos e livros de geometria
D1: mais notícias, ficção e textos sobre transporte aéreo
```

Isso pode ser historicamente interessante, mas não prova sozinho que uma
palavra mudou de sentido. `chairman` pode continuar significando "pessoa que
preside", embora o corpus passe de parlamentos para empresas.

### Mudança contextual ou de uso

É uma alteração em:

```text
P_t(contexto | palavra)
```

Ela pode envolver tópico, gênero, sintaxe, entidades, colocados ou sentido.
APD, clustering de ocorrências e perfis relacionais medem principalmente esse
nível. Eles detectam que **algo no uso mudou**, mas não identificam
automaticamente a causa.

### Mudança de representação

É uma alteração no que o encoder produz:

```text
h_theta0(frase, palavra) != h_theta1(frase, palavra)
```

Ela pode acontecer mesmo para a **mesma frase**, apenas porque os pesos do
modelo mudaram. Foi o que o projeto chamou de "régua móvel" ou drift de
checkpoint.

### Mudança de sentido lexical

É a alteração desejada:

```text
P_0(sentido | palavra) != P_1(sentido | palavra)
```

Para `plane`, isso pode ser:

```text
D0: geometria=0,80; avião=0,01; ferramenta=0,19
D1: geometria=0,05; avião=0,93; ferramenta=0,02
```

Aqui mudou a distribuição de sentidos, não apenas o vocabulário ao redor.

## O estimando

Um **estimando** é a quantidade científica que se quer conhecer, antes de
escolher como medi-la. O estimando ideal do projeto é:

```text
Delta_sem(w) = D(P_0(s | w), P_1(s | w))
```

O problema é que `s` não vem observado no corpus. Por isso surgiram vários
**estimadores** ou proxies:

| Estimador | O que observa diretamente | Risco |
|---|---|---|
| JSD de perfis MLM | mudança nas associações previstas | entropia/head/checkpoint |
| APD contextual | distância entre nuvens de ocorrências | frequência, tópico, registro |
| clusters + período | separabilidade dos usos | cluster não é necessariamente sentido |
| campos manuais | massa em eixos definidos por humanos | engenharia por palavra |
| WSD externo | sentidos de inventário previstos | cobertura, cauda histórica, inventário |

Uma métrica pode correlacionar com o gold e ainda não estimar exatamente
`Delta_sem`. Inversamente, uma métrica bem motivada pode ter baixo Spearman
por falta de capacidade, cobertura ou amostra.

## Identificabilidade em uma equação

Os contextos observados são uma mistura:

```text
P_t(c | w) = sum_s P_t(s | w) P_t(c | w, s)
```

Uma mudança em `P_t(c | w)` pode vir de:

```text
P_t(s | w)       mudou  -> mudança na distribuição de sentidos
P_t(c | w, s)    mudou  -> o mesmo sentido passou a ocorrer em outros contextos
ambos mudaram
```

Sem supervisão, inventário ou hipóteses adicionais, essas explicações podem
produzir os mesmos dados observados. Esse é um problema de
**identificabilidade**, não apenas de escolher um algoritmo melhor.

`chairman` é o exemplo canônico: os contextos institucionais mudam, mas o
sentido "pessoa que preside" pode permanecer. `plane` é o caso mais favorável:
contexto e sentido mudam juntos, de geometria para aviação.

## Formas diferentes de mudança

### Substituição dominante

Um sentido perde massa e outro o substitui:

```text
plane: geometria -> avião
```

Uma distância simples entre distribuições costuma capturar bem esse caso.

### Diversificação

O sentido antigo permanece, mas sentidos adicionais ganham massa:

```text
graft: botânico -> botânico + médico + corrupção
```

O centróide pode se mover pouco, apesar de a distribuição ficar multimodal.
Por isso `graft` foi difícil para métodos baseados em média.

### Deriva contextual dentro do mesmo sentido

O sentido permanece, mas sua realização muda:

```text
chairman: presidir parlamento -> presidir empresa/comissão
```

É mudança de uso, possivelmente importante, mas não necessariamente mudança
de identidade lexical.

## Duas contribuições científicas possíveis

O histórico acabou separando duas reivindicações legítimas:

1. **instrumento temporal relacional**: descrever como o modelo reorganiza
   associações e vizinhanças ao aprender cronologicamente;
2. **medidor de mudança de sentido**: estimar alterações em
   `P_t(sentido | palavra)`.

O TimeFormer demonstrou melhor a primeira. A linha de WSD externo tenta tornar
a segunda identificável. Elas se relacionam, mas uma não deve ser apresentada
como se provasse automaticamente a outra.

## Perguntas de controle para qualquer resultado

Ao encontrar "mudança", pergunte:

```text
1. O corpus mudou?
2. As ocorrências/contextos mudaram?
3. O checkpoint/encoder mudou?
4. A métrica responde a frequência ou dispersão?
5. Há evidência de que os grupos correspondem a sentidos?
6. O inventário contém o sentido histórico relevante?
7. Qual desses níveis é o estimando declarado?
```

Essa lista resume a lógica dos capítulos 05 a 10.
