# 21. Duas réguas que não são equivalentes

Depois de validar e calibrar o ConSeC, o projeto tinha duas medidas por
palavra:

```text
TimeFormer layer_1 -> APD entre contextos de D0 e D1
ConSeC             -> JSD entre distribuições de sentidos
```

Parecia natural esperar que palavras altas numa régua também fossem altas na
outra. Essa hipótese foi congelada antes da comparação.

## O teste

Foram usados os 25 alvos confirmatórios da Porta 3. Para reduzir ruído:

- ConSeC foi agregado sobre três amostras;
- TimeFormer foi agregado sobre duas seeds de treino cronológico;
- JSD bruta e JSD excedente foram testadas separadamente;
- gold e número de sentidos foram usados como controles.

## O resultado

| Comparação | Spearman |
|---|---:|
| `layer_1` × JSD bruta | -0,031 |
| `layer_1` × JSD excedente | -0,024 |

As correlações são essencialmente zero. Controlar gold ou tamanho do
inventário não mudou a conclusão.

No mesmo subconjunto, o ConSeC continuou acompanhando o gold, enquanto a APD
de `layer_1` ficou fraca:

```text
ConSeC JSD bruta × gold = 0,604
layer_1 APD × gold      = 0,076
```

## O que aprendemos

APD e JSD não são medidas intercambiáveis.

A APD responde à mudança da nuvem contextual inteira. Tópico, sintaxe,
gênero textual e composição documental podem mover essa nuvem. A JSD responde
à redistribuição de probabilidade entre sentidos de um inventário externo.

Uma palavra pode mudar muito de contexto e pouco de sentido. Também pode trocar
o sentido dominante sem produzir a maior APD do conjunto.

O NO-GO evita um erro conceitual importante:

```text
APD alta não equivale automaticamente a mudança de sentido alta
```

## A próxima escala de análise

Comparar um único número por palavra descartou informação demais. O próximo
passo usa exatamente as mesmas ocorrências:

```text
frase -> posterior ConSeC
      -> vetor TimeFormer
```

Dentro de cada palavra, será possível verificar se a geometria contextual
separa sentidos, medir deriva dentro do mesmo sentido e decompor mudança de
mistura versus mudança contextual.

Detalhes: `docs/30-consec_timeformer_integration_results.md`.
