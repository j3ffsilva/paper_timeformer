# Conceitos 9 — Dos dados históricos ao vetor de uma ocorrência

Este arquivo acompanha concretamente o caminho de uma linha do corpus até a
representação usada nas métricas.

## O corpus SemEval usado

O projeto trabalha com dois arquivos principais:

```text
1810-1860.txt
1960-2010.txt
```

Cada linha representa um documento ou unidade textual. O corpus está:

```text
em inglês
lematizado
com parte do discurso anexada a alguns alvos
sem a morfologia e pontuação completas do texto original
```

Exemplo:

```text
plate 16 figure 4 represent an inclined plane_nn
```

`plane_nn` significa o lema `plane` marcado como substantivo. Formas como
`planes`, `plane's` e flexões foram reduzidas ao lema.

## Lematização: ganho e perda

A lematização agrupa variantes:

```text
flies, flew, flying -> fly
planes             -> plane
```

Isso ajuda a contar ocorrências e reduz sparsidade. Mas perde informação:

```text
tempo verbal
número
capitalização
parte da sintaxe superficial
naturalidade da frase para modelos pré-treinados
```

Um BERT treinado em texto natural recebe aqui frases telegráficas como:

```text
the pilot fly the plane
```

em vez de:

```text
the pilot flew the plane
```

O Gate 1 mostrou que isso não inviabiliza geometria/aviação, mas pode agravar
casos raros e históricos.

## Token do projeto vs. WordPiece

No modelo próprio, `plane_nn` é um único item de vocabulário:

```text
token inteiro -> um embedding
```

No BERT, o texto passa por um tokenizer WordPiece:

```text
unfamiliarword -> un ##fam ##iliar ##word
```

Palavras frequentes podem permanecer inteiras; palavras raras são quebradas.
Para avaliar o alvo, o projeto:

1. remove `_nn` ou `_vb`;
2. tokeniza a lista de palavras com o tokenizer original;
3. localiza todos os WordPieces pertencentes à palavra-alvo;
4. tira a média dos hidden states desses WordPieces.

```text
vetor("plane") = media(vetor(piece_1), ..., vetor(piece_k))
```

Isso é diferente da adaptação inicial do capítulo 09, que transformava a média
estática de WordPieces em um novo embedding de token inteiro e depois
abandonava a tokenização original.

## Embeddings que entram no BERT

Antes da primeira camada Transformer, cada posição combina:

```text
embedding do token/WordPiece
+ embedding posicional
+ embedding de tipo de segmento
-> LayerNorm + dropout
```

Por isso copiar apenas embeddings lexicais e camadas Transformer não reproduz
o BERT. A Option D preservou tokenizer, embeddings posicionais, embedding
LayerNorm, camadas e MLM head originais.

## Janela contextual

Uma ocorrência não é codificada com o corpus inteiro. Extrai-se uma janela
centrada no alvo:

```text
... palavras à esquerda [plane] palavras à direita ...
```

O tamanho da janela define quanta evidência está disponível:

- pequena demais: pode perder `airport`, `carpenter` ou `inclined`;
- grande demais: inclui tópicos distantes e custa mais memória;
- truncada de um lado: pode eliminar justamente a pista relevante.

O código mantém a posição do alvo alinhada depois da tokenização. Isso é
essencial: o vetor usado deve ser o hidden state de `plane`, não de `[CLS]`,
de uma posição fixa ou de um WordPiece vizinho.

## Frase, documento e fronteira

Janelas nunca devem atravessar documentos. Caso contrário:

```text
fim de um documento + começo do próximo
```

vira um contexto artificial que jamais existiu. O bug de fronteiras do
capítulo 05 criou exatamente esse tipo de amostra, contaminando treino e
avaliação.

Por isso a unidade documental participa de:

```text
construção das janelas
split treino/validação
holdout
amostragem de ocorrências
```

Dividir janelas aleatoriamente, em vez de documentos, também pode vazar trechos
quase idênticos entre treino e validação.

## Ocorrência, tipo e centróide

É útil distinguir:

- **tipo lexical**: a palavra abstrata `plane`;
- **ocorrência/token contextual**: uma aparição específica numa frase;
- **vetor contextual**: hidden state dessa ocorrência;
- **centróide**: média de várias ocorrências.

Exemplo:

```text
plane em "inclined plane" -> vetor A
plane em "board the plane" -> vetor B
plane em "jack plane"      -> vetor C
```

O centróide de D0 mistura todos os vetores daquele período. APD compara as
nuvens sem exigir que essa média represente todos os modos.

## Amostragem e frequência

Palavras têm números muito diferentes de ocorrências. Usar todas pode:

```text
dar precisão desigual entre palavras
aumentar custo quadraticamente no APD
fazer palavras raras parecerem mais dispersas
```

Por isso avaliações frequentemente limitam o número de ocorrências por alvo e
mantêm a mesma amostra entre condições. Se cada encoder recebe ocorrências
diferentes, a diferença de resultado pode vir da amostra, não do modelo.

Controles pareados por frequência ajudam, mas não substituem uma amostragem
alinhada.

## Do texto à métrica

O pipeline completo pode ser visualizado assim:

```text
linha do corpus
  -> tokens lematizados
  -> remover sufixo POS para BERT
  -> selecionar janela sem cruzar documento
  -> WordPieces + posições
  -> hidden states por camada
  -> média dos WordPieces do alvo
  -> vetor da ocorrência
  -> nuvem por palavra e período
  -> APD / clustering / WSD / perfil relacional
  -> score por palavra
  -> Spearman contra o gold
```

Cada seta é uma decisão experimental. Entender o projeto plenamente exige
saber em qual seta uma hipótese ou bug atua.
