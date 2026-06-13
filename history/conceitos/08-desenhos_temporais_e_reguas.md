# Conceitos 8 — Desenhos temporais, checkpoints e réguas

Este arquivo explica como separar duas coisas que acontecem simultaneamente no
projeto: **os dados mudam de período** e **o modelo muda durante o treino**.

## A grade 2 x 2

Com dois corpora (`D0`, `D1`) e dois checkpoints (`theta0`, `theta1`), existem
quatro combinações:

| | Corpus D0 | Corpus D1 |
|---|---|---|
| Encoder `theta0` | `theta0_d0` | `theta0_d1` |
| Encoder `theta1` | `theta1_d0` | `theta1_d1` |

Essa tabela é o experimento conceitual mais importante dos capítulos 07-09.

### Comparação diagonal

```text
theta0_d0  vs.  theta1_d1
```

Muda o corpus **e** o encoder. Se as representações diferem, não sabemos qual
mudança causou o efeito.

### Corpus fixo

```text
theta0_d0  vs.  theta1_d0
theta0_d1  vs.  theta1_d1
```

As frases são as mesmas; só o checkpoint muda. Isso mede drift funcional do
modelo. No projeto, essas comparações já separavam quase perfeitamente os
checkpoints, revelando a régua móvel.

### Encoder fixo

```text
theta1_d0  vs.  theta1_d1
```

O modelo é o mesmo; só o corpus muda. Esse é o desenho adequado quando se quer
comparar ocorrências em um espaço comum.

`theta1` foi preferido a `theta0` porque viu D0 e D1 durante o treino.
`theta0_d1` usa uma régua que nunca aprendeu o domínio moderno.

## A analogia da régua

Imagine medir dois objetos:

```text
objeto antigo com uma régua de madeira
objeto moderno com uma régua de borracha deformada
```

Comparar os números mistura diferença entre objetos e diferença entre réguas.
Um encoder fixo equivale a usar a mesma régua nos dois.

Isso não torna a régua perfeita. Ela ainda pode favorecer sentidos modernos,
ter cobertura ruim ou ser pouco semântica. Apenas remove um confundidor
fundamental.

## Regimes de treino temporal

### Treino contínuo

```text
theta0 = treinar(theta_init, D0)
theta1 = continuar(theta0, D1)
```

Há um único modelo com memória nos pesos. A ordem D0→D1 faz parte do método.

### Modelos independentes

```text
theta0 = treinar(inicializacao_A, D0)
theta1 = treinar(inicializacao_B, D1)
```

Cada período tem seu próprio modelo. Isso evita herança direta, mas cria dois
espaços de coordenadas que precisam ser alinhados, por exemplo com Procrustes.

### Retreino acumulativo

```text
theta0 = treinar_do_zero(D0)
theta1 = treinar_do_zero(D0 + D1)
```

O modelo posterior vê todo o passado, mas não herda os pesos do checkpoint
anterior. Esse baseline testa se a continuidade dos parâmetros acrescenta algo
além da simples exposição aos dados acumulados.

### Encoder externo congelado

```text
h0 = BERT_fixo(D0)
h1 = BERT_fixo(D1)
```

Não existe aprendizado temporal. É uma régua comum para verificar se a tarefa
é solucionável e se o encoder próprio é o gargalo.

<a id="pseudo-periodos"></a>
## Pseudo-períodos

Em um pseudo-período, os documentos de D0+D1 são embaralhados e repartidos com
os mesmos tamanhos:

```text
pseudo-D0 e pseudo-D1 têm tamanho igual aos originais,
mas não representam cronologia real.
```

Se um efeito aparece tanto em D0→D1 quanto em pseudo-D0→pseudo-D1, ele pode ser
efeito geral de fine-tuning ou amostragem, não da ordem histórica.

O controle só é justo se os checkpoints comparados tiverem o mesmo orçamento:

```text
mesmo número de passos
mesmo scheduler
mesma fração de época
mesma regra de seleção
```

O capítulo 09 mostrou que comparar `D1@2` com `pseudo-D1@0,5` confundia
cronologia com quantidade de treino.

## Placebo, nulo e controle não são sinônimos

- **Nulo ressampleado**: não há mudança plantada; estima o ruído esperado.
- **Placebo temporal**: repete-se um período ou usa-se uma sequência sem a
  mudança de interesse.
- **Pseudo-período**: preservam-se tamanhos e pipeline, mas destrói-se a
  cronologia.
- **Oráculo**: usa-se um componente mais forte para verificar se a tarefa é
  possível.
- **Controle pareado**: selecionam-se palavras comparáveis em frequência ou
  outra variável de confusão.

Cada controle elimina uma explicação alternativa diferente.

## Seleção de checkpoint sem olhar o gold

Salvar vários checkpoints cria outra forma de flexibilidade: escolher depois o
que teve melhor Spearman nos 37 alvos seria usar o conjunto de teste como
validação.

No Option D, a seleção usou:

```text
1. loss MLM em documentos retidos;
2. entre checkpoints dentro de 1% da melhor loss,
   maior estabilidade em frases-âncora;
3. desempate por menor distância à inicialização.
```

Somente depois o checkpoint selecionado foi avaliado no SemEval.

## O estado do otimizador também faz parte da história

Continuar treino não significa apenas recarregar pesos. Adam mantém momentos
acumulados dos gradientes. Se o checkpoint salva o modelo mas não o
`optimizer.state_dict`, retomar o treino altera a trajetória de otimização.

Assim, um checkpoint completo para continuidade contém:

```text
pesos do modelo
estado do otimizador
estado do scheduler
época/passo
sementes ou estado aleatório quando necessário
configuração e vocabulário/tokenizer
```

Esse detalhe causou um placebo enganoso no capítulo 03.

## Como ler qualquer comparação do projeto

Antes de interpretar um número, escreva explicitamente:

```text
representacao(checkpoint, corpus, readout, amostra)
```

Por exemplo:

```text
APD(theta1, D0, D1, layer_1, mesmas_ocorrencias)
```

Se duas condições diferem em mais de uma dessas dimensões, a comparação
precisa de uma ablação ou deve ser tratada como descritiva.
