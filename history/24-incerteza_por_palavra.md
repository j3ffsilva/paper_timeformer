# 24. Quais palavras sobrevivem à reamostragem?

A decomposição mostrou uma contribuição média de composição de sentidos. Mas
cada palavra tinha apenas 25 ocorrências por período em cada amostra. Era
necessário verificar quais resultados individuais dependiam da seleção dessas
frases.

## Bootstrap estratificado

D0 e D1 foram reamostrados separadamente, com reposição. As duas seeds do
TimeFormer receberam exatamente os mesmos índices, preservando o pareamento.
O processo foi repetido 2.000 vezes.

## A conclusão global

Na `layer_2`:

```text
mediana bootstrap da contribuição média = 0,042
IC 95% = [0,032; 0,053]
```

O intervalo permanece inteiramente acima de zero. O ranking entre palavras
também é estável (`rho=0,919` entre observado e mediana bootstrap).

## A conclusão individual

Apenas dez das 25 palavras tiveram intervalos inteiramente positivos:

```text
plane, multitude, gas, record, land,
attack, bit, thump, risk, fiction
```

Nenhuma foi robustamente negativa.

Isso estabelece uma distinção importante:

```text
23/25 estimativas pontualmente positivas
10/25 evidências individuais robustas
```

O primeiro número descreve direção. O segundo sustenta afirmações sobre
palavras específicas.

## Casos ilustrativos

`plane` permanece o caso mais forte:

```text
mediana = 0,246
IC 95% = [0,110; 0,363]
```

`multitude`, `gas` e `record` também sobrevivem. `player`, `donkey` e `stab`
ficam incertos; seus pequenos sinais não devem aparecer como conclusões.

## O que esta análise acrescenta

A cadeia de validação semântica externa está madura:

1. o ConSeC foi validado;
2. a mudança de mistura foi replicada;
3. a geometria contém estrutura local de sentido;
4. a composição de sentidos reconhecida pelo inventário se associa a uma
   parcela pequena do deslocamento;
5. essa parcela é globalmente estável e individualmente robusta em dez casos.

Isso não encerra o objetivo principal do projeto. O eixo `token@time` ainda
deve ser apresentado diretamente por vizinhanças temporais, mostrando quais
relações cada palavra ganha e perde. ConSeC acrescenta uma interpretação mais
estrita a parte dessas mudanças, sem definir sozinho o que os vizinhos
significam.

Detalhes: `docs/36-soft_decomposition_bootstrap_results.md`.
