# Conceitos 2 — Encoders, camadas e checkpoints

Este arquivo explica os blocos de construção de um Transformer/MLM no
nível necessário para acompanhar os capítulos do histórico, sem assumir
conhecimento prévio de deep learning.

<a id="pipeline-encoder"></a>
## Do token ao hidden state

Um encoder não recebe palavras diretamente. O caminho simplificado é:

```text
texto
  -> tokenizer
  -> ids de tokens/WordPieces
  -> embeddings lexicais + posicionais
  -> LayerNorm
  -> camada Transformer 1
  -> camada Transformer 2
  -> ...
  -> hidden state contextual de cada posição
```

O mesmo token pode produzir vetores diferentes em frases diferentes:

```text
plane em "inclined plane" != plane em "board the plane"
```

Isso é o que torna o hidden state **contextual**. O arquivo
[`09-dados_tokenizacao_e_contexto.md`](09-dados_tokenizacao_e_contexto.md)
acompanha esse fluxo com os dados reais do projeto.

**LayerNorm** normaliza ativações dentro do modelo. Não é apenas um detalhe:
as camadas pré-treinadas esperam uma distribuição específica de entrada. Na
primeira adaptação de `bert-tiny`, o modelo próprio não preservava a embedding
LayerNorm original, uma das incompatibilidades que motivaram a Option D.

<a id="mlm"></a>
## MLM (Masked Language Modeling)

MLM é a tarefa de treino usada em todo o projeto (e em BERT): pega-se uma
frase, **esconde-se** (mascara-se) uma ou mais palavras, e treina-se o
modelo para **prever a palavra escondida** a partir do resto da frase.

```text
entrada:  "the [MASK] of projection meets the horizon"
alvo:     "plane"
```

A ideia é que, para acertar essa previsão, o modelo precisa aprender algo
sobre o significado das palavras e como elas se relacionam com o contexto
— "para prever que a palavra escondida é `plane` e não, digamos, `line`, o
modelo precisa ter associado `plane` a contextos geométricos como `the ...
of projection`".

O MLM é o que permite ao modelo, depois de treinado, responder duas
perguntas centrais do projeto:

1. **"Que palavra completaria esta posição?"** — usado no capítulo 04/05
   (perfis log-PMI via `q_t(w)`).
2. **Como subproduto do treino, o modelo também produz representações
   internas (hidden states/[camadas](#mlm-head)) que capturam informação
   sobre o contexto**, mesmo sem mascarar nada — usado no capítulo 06 em
   diante (perfis relacionais sobre hidden states).

<a id="mascaramento"></a>
## Mascaramento

O "mascaramento" é a política de **quais tokens substituir por `[MASK]`,
em quais posições, e com que frequência** durante o treino MLM. Parece
um detalhe de implementação, mas o capítulo 05 mostrou que pode ser a
diferença entre um modelo funcionar e não funcionar.

A política padrão do BERT (usada a partir do capítulo 06) é: a cada época,
escolher 15% dos tokens de cada janela; desses, 80% são substituídos por
`[MASK]`, 10% por um token aleatório, e 10% mantidos inalterados (mas
ainda usados como alvo de previsão). As máscaras mudam por época, mas são
reproduzíveis por uma semente (`seed`).

**O problema descoberto no capítulo 05** foi um mascaramento *diferente*
e mais simples, usado antes do capítulo 06: sempre mascarar a posição
**central** de cada janela de 30 tokens. Como 70% das sentenças do corpus
têm menos de 30 tokens, isso concentrava as máscaras de treino em torno da
posição 12 (desvio 4,1, faixa [2,16]). Mas quando o modelo era avaliado em
ocorrências reais de `graft_nn`, a posição da palavra variava muito mais
(média 14,2, desvio 8,0, faixa [1,30]) — **24,4% das ocorrências caíam em
posições nunca vistas durante o treino**. Com *embeddings posicionais
absolutos* (cada posição 1-30 tem seu próprio vetor aprendido), uma
posição nunca treinada produz uma representação essencialmente aleatória.

**Embedding posicional**: cada posição na sequência (1ª palavra, 2ª
palavra, ...) recebe um vetor próprio, somado ao embedding da palavra, para
que o modelo saiba "onde" cada palavra está na frase. Se uma posição
específica nunca aparece mascarada/avaliada durante o treino, seu
embedding nunca recebe gradiente útil relacionado àquela tarefa.

<a id="mlm-head"></a>
## MLM head

O "MLM head" é a última camada do modelo: ela recebe o vetor oculto
(hidden state) da posição mascarada e produz um **logit para cada token do
vocabulário** — que o [softmax](01-correlacao_e_similaridade.md#softmax)
converte na distribuição `q_t(w)`.

```text
hidden_state (vetor, ex: 128 dimensões)
   -> MLM head (camada linear, ou linear+ativação+linear)
   -> logits (vetor com |V| dimensões, ex: 27.311)
   -> softmax
   -> q_t(w) (distribuição de probabilidade sobre V)
```

**Probe neutro**: para calcular `p_t` (capítulo 04) — "o que o modelo
prevê em geral, sem nenhuma informação sobre `w`" — usa-se uma sequência
mínima como `[CLS] [MASK] [SEP]`, sem nenhuma palavra de conteúdo. A
limitação levantada no capítulo 05 (H4) é que essa sequência de 3 tokens é
muito diferente das janelas de 30 tokens vistas no treino — então `p_t`
pode não ser representativo.

**Weight tying**: no BERT original, a matriz de embeddings de entrada
(que converte tokens em vetores) e a matriz de saída do MLM head (que
converte vetores de volta em logits sobre tokens) **compartilham os
mesmos pesos** — é a mesma matriz, usada nos dois sentidos. Isso é uma
restrição estrutural forte: o espaço onde as palavras "vivem" como entrada
é, por construção, o mesmo espaço usado para prevê-las como saída. O
`RealStaticMLM` do projeto (capítulos 06-09) **não tinha** esse weight
tying — embedding de entrada e MLM head eram matrizes independentes — algo
apontado no capítulo 09 como uma possível causa de instabilidade durante o
fine-tuning contínuo.

<a id="checkpoints"></a>
## Checkpoints

Um "checkpoint" é uma cópia congelada de todos os pesos do modelo num
ponto específico do treino. O projeto usa, centralmente, dois:

```text
theta_0 = checkpoint depois de treinar em D0 (1810-1860)
theta_1 = checkpoint depois de continuar o treino de theta_0 em D1 (1960-2010)
```

A pergunta de pesquisa original é, essencialmente, "o que mudou entre
`theta_0` e `theta_1`?" — mas o capítulo 08 mostrou que essa pergunta tem
uma armadilha: **`theta_0` e `theta_1` são, eles mesmos, dois pontos
diferentes no espaço de parâmetros**, e essa diferença ("drift de
checkpoint") pode ser maior do que qualquer diferença entre os corpora D0
e D1. Ver [encoder fixo](#encoder-fixo) para a solução adotada.

Um terceiro checkpoint importante, introduzido no capítulo 08, é
`theta_init` — o estado do modelo **antes** de qualquer treino nos corpora
do projeto (por exemplo, `bert-tiny` recém-carregado do Hugging Face, ou
adaptado para o vocabulário próprio mas ainda não treinado). `theta_init`
serve como referência de "o que o modelo sabia antes de ver qualquer dado
temporal" — usado tanto para medir quanto sinal o pré-treino já trazia
(capítulo 08) quanto como "professor" em propostas de regularização
(capítulo 09, L2-SP).

<a id="encoder-fixo"></a>
## Encoder fixo

"Encoder fixo" significa: usar **o mesmo checkpoint** (os mesmos pesos)
para processar tanto as frases de D0 quanto as de D1, em vez de processar
cada período com seu checkpoint "nativo".

A configuração **"diagonal"** (usada implicitamente até o capítulo 08) é:

```text
ocorrências de D0  ->  processadas por theta_0
ocorrências de D1  ->  processadas por theta_1
```

O problema: qualquer diferença encontrada nessa configuração mistura
**duas** fontes de variação — diferença de corpus (D0 vs D1) **e**
diferença de checkpoint (`theta_0` vs `theta_1`, que tiveram passos de
treino, otimizador, e exposição a dados diferentes). O capítulo 08 mostrou
que a segunda fonte é, isoladamente, enorme (NMI~0,8-1,0 entre
`theta0_d0` e `theta1_d0`, ou seja: as mesmas frases de D0, processadas
pelos dois checkpoints, já são quase perfeitamente separáveis).

A correção é processar **ambos** os períodos com o **mesmo** checkpoint:

```text
ocorrências de D0  ->  processadas por theta_1
ocorrências de D1  ->  processadas por theta_1
```

Com `theta_0` fixo, o sinal cai para perto do acaso (Spearman ~0,02-0,05) —
porque `theta_0` nunca viu os textos de D1 e mede-os mal. Com `theta_1`
fixo (que viu ambos os corpora durante o treino contínuo), o sinal sobe
para Spearman ~0,20 — o melhor resultado obtido com o encoder próprio do
projeto, e o padrão adotado a partir do capítulo 08.

<a id="lr-discriminativa"></a>
## LR discriminativa

"LR" é a *learning rate* (taxa de aprendizado) — o tamanho do passo que o
otimizador dá em direção ao gradiente a cada atualização. "LR
discriminativa" significa usar **taxas diferentes para partes diferentes
do modelo**, em vez de uma única taxa global.

No capítulo 09, a configuração testada foi:

```text
embeddings + layer 1: LR 1e-5  (menor — muda pouco)
layer 2 + MLM head:   LR 3e-5  (maior — muda mais)
```

A intuição é que camadas mais próximas da entrada (embeddings, primeiras
camadas) tendem a capturar regularidades mais gerais da língua, que não
deveriam mudar muito durante um fine-tuning relativamente curto; camadas
mais próximas da saída (últimas camadas, head) precisam se adaptar mais à
tarefa/domínio específico. No experimento do capítulo 09, a LR
discriminativa produziu um ganho pontual pequeno em `layer_1` (0,325 ->
0,340) — mas, como em quase tudo nesse capítulo, essa diferença não foi
estatisticamente distinguível de zero no [bootstrap](05-estatistica_experimental.md#bootstrap).

Relacionado: **congelamento** é o caso extremo de LR discriminativa, onde
a LR de uma parte do modelo é fixada em zero — essa parte não muda durante
o treino. O capítulo 09 testou congelar embeddings+`layer_1`, e descobriu
que isso não protegia `layer_2` (que continuou perdendo sinal) e impedia o
pequeno ganho que `layer_1` teria com fine-tuning completo.

<a id="fine-tuning"></a>
## Fine-tuning, warmup e scheduler

**Pré-treino** ensina regularidades gerais em um corpus grande. **Fine-tuning**
continua o treino em dados menores ou mais específicos. No projeto:

```text
bert-tiny pré-treinado
  -> fine-tuning MLM em D0
  -> continuação MLM em D1
```

Uma LR alta e constante pode reorganizar rapidamente um modelo pré-treinado.
Por isso a Option D usou:

```text
warmup: LR cresce gradualmente nos primeiros 10% dos passos
linear decay: LR diminui até o fim
gradient clipping: limita atualizações extremas
holdout por documento: mede loss em textos não usados no treino
```

O **scheduler** define como a LR muda ao longo dos passos. Warmup evita um
choque inicial; decay reduz oscilações perto do fim. Essas escolhas não
garantem preservação semântica, mas tornam o fine-tuning mais controlado.

**Weight decay** e L2-SP também são diferentes: weight decay puxa pesos em
direção a zero; L2-SP puxa pesos em direção ao checkpoint inicial.
