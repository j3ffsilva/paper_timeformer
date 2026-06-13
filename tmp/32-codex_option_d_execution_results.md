# Resposta experimental à segunda opinião: Option D com `bert-tiny` integral

## Resumo

Implementamos e executamos a recomendação principal da segunda opinião:
`prajjwal1/bert-tiny` integral, com tokenizer WordPiece, arquitetura,
embedding LayerNorm, positional embeddings, MLM head e weight tying
originais.

O resultado muda o diagnóstico anterior:

- não houve colapso global da representação;
- a primeira camada preservou um ponto estimado semelhante ou levemente
  superior, sem diferença estatisticamente estabelecida;
- a segunda camada perdeu correlação com o gold sob treino cronológico;
- o controle com pseudo-períodos preservou mais sinal na segunda camada;
- congelar embeddings + camada 1 não ajudou;
- os pontos estimados sugerem uma redistribuição entre camadas, mas
  bootstrap com `n=37` não distingue esse efeito de ruído;
- não há evidência de catastrophic forgetting uniforme do backbone.

## Implementação

Novos arquivos:

```text
src/timeformers/bert_continual.py
scripts/run_bert_tiny_continual_option_d.py
scripts/evaluate_bert_checkpoint_apd.py
scripts/select_frequency_matched_controls.py
tests/test_bert_continual.py
```

O pipeline oferece:

- `AutoModelForMaskedLM` e tokenizer originais;
- verificação empírica de weight tying;
- remoção de `_nn`/`_vb` antes da tokenização;
- janelas WordPiece com mascaramento dinâmico determinístico;
- holdout por documento em cada período;
- AdamW, warmup e linear decay;
- checkpoints em 0,25, 0,5, 1 época e depois por época;
- loss MLM nos dois períodos;
- distância L2 relativa ao checkpoint inicial;
- cosseno de representações em frases-âncora;
- cosseno médio entre embeddings de tokens aleatórios;
- seleção sem gold, separada por período e sem consultar dados futuros;
- pseudo-períodos aleatórios;
- congelamento de embeddings + camada 1;
- avaliador alinhado para embedding, cada camada e médias;
- controles lexicais fora do benchmark pareados por frequência em D0/D1.

Verificações:

```text
weight tying antes e depois do save/load: confirmado
89 testes: passaram
py_compile: passou
git diff --check: passou
```

## Regra de seleção sem gold

Para `theta0`, são usadas apenas losses de validação de D0.
Para `theta1`, usa-se a média das losses de D0 e D1.

Entre checkpoints dentro de 1% da melhor loss:

1. maximiza-se o cosseno com as representações-âncora iniciais;
2. desempata-se pela menor distância L2 ao checkpoint inicial.

O checkpoint final deve pertencer ao último período; `theta1` nunca pode
ser substituído por um checkpoint de D0.

## Experimentos

### Full fine-tuning cronológico

Configuração:

```text
D0=3 épocas
D1=2 épocas
LR=3e-5
warmup=10%
linear decay
batch=192
seq_len=32
2 seeds: 1000 e 1001
```

Cada seed executou 10.349 passos.

Checkpoints selecionados:

| Seed | theta0 | theta1 |
|---|---|---|
| 1000 | D0@3 | D1@2 |
| 1001 | D0@2 | D1@2 |

Em ambas as seeds, a similaridade das representações-âncora em `theta1`
permaneceu aproximadamente `0,95`. O cosseno médio dos embeddings mudou
gradualmente, sem salto de colapso.

### Controle com pseudo-períodos

Os documentos de D0+D1 foram embaralhados e repartidos em dois conjuntos
com os tamanhos originais. Configuração de treino idêntica, seed 1000.

Seleção:

```text
theta0 = pseudo-D0@2
theta1 = pseudo-D1@0,5
```

### Variante com camada inferior congelada

Embeddings e camada 1 foram congelados. Apenas camada 2 e MLM head foram
treinados, com a mesma configuração.

Seleção:

```text
theta0 = D0@1
theta1 = D1@2
```

## Comparação alinhada de readouts

O avaliador antigo produzia `bert-tiny mean_last_4=0,399`, mas, num modelo
com duas camadas, esse valor misturava embedding output e camadas, além de
usar pré-processamento e comprimento diferentes.

O novo avaliador mantém:

```text
mesmas frases normalizadas
max_length=32
mesma amostra de ocorrências
mesmo tokenizer
readouts explicitamente separados
```

Baseline `bert-tiny` original:

| Readout | Spearman |
|---|---:|
| embedding | -0,018 |
| layer 1 | 0,298 |
| layer 2 | 0,136 |
| média layers 1+2 | 0,241 |
| média embedding+layers | 0,160 |

Portanto, `0,399` não era um teto alinhado. O baseline relevante para
`layer_1` é aproximadamente `0,298`.

## Resultados SemEval dos checkpoints pré-selecionados

### Full fine-tuning cronológico

| Readout | Seed 1000 | Seed 1001 |
|---|---:|---:|
| embedding | 0,038 | 0,032 |
| **layer 1** | **0,325** | **0,322** |
| layer 2 | 0,030 | 0,038 |
| média layers 1+2 | 0,189 | 0,194 |
| média embedding+layers | 0,120 | 0,142 |

`layer_1` é estável entre seeds e melhora levemente sobre o baseline
`0,298`. `layer_2` perde quase todo o sinal ranqueado.

Na seed 1000, `theta0` já apresentava:

```text
layer 1 = 0,324
layer 2 = -0,019
```

Assim, a reorganização ocorre em D0; D1 preserva layer 1 e recupera apenas
uma pequena parte de layer 2.

### Pseudo-períodos

| Readout | Spearman |
|---|---:|
| embedding | 0,037 |
| **layer 1** | **0,332** |
| layer 2 | **0,153** |
| média layers 1+2 | 0,270 |
| média embedding+layers | 0,184 |

O ganho em layer 1 também aparece sem cronologia real, sugerindo adaptação
geral ao domínio/corpus. Porém, layer 2 é muito melhor no pseudo-controle
do que no treino cronológico (`0,153` contra `0,030–0,038`), indicando que
a sequência temporal pode afetar a camada superior. Essa comparação
original confundia `0,5` época pseudo com `2` épocas cronológicas; o
controle corrigido é apresentado abaixo.

### Camada inferior congelada

| Readout | Spearman |
|---|---:|
| embedding | -0,018 |
| layer 1 | 0,298 |
| layer 2 | 0,017 |
| média layers 1+2 | 0,149 |
| média embedding+layers | 0,135 |

Como esperado, layer 1 permanece idêntica ao baseline, mas layer 2 ainda
perde o sinal. O full fine-tuning é melhor porque permite que layer 1 se
adapte de `0,298` para aproximadamente `0,323`.

## Controles lexicais fora do benchmark

Foram selecionadas 37 palavras externas com frequências pareadas
simultaneamente em D0 e D1.

Mediana da APD:

| Condição | Grupo | layer 1 | layer 2 |
|---|---|---:|---:|
| cronológica | alvos | 0,318 | 0,339 |
| cronológica | controles | 0,304 | 0,343 |
| pseudo-períodos | alvos | 0,323 | 0,383 |
| pseudo-períodos | controles | 0,309 | 0,372 |

A magnitude absoluta da APD é semelhante entre alvos e controles. O sinal
do SemEval vem do ranking relativo entre palavras, não de uma separação
global de magnitudes. Isso reforça a necessidade de controles semânticos
ou calibração por campo/frequência.

## Ablação de LR discriminativa e L2-SP

Implementamos L2-SP normalizado apenas na camada 2:

```text
loss = loss_MLM + lambda * ||theta_layer2 - theta_init||^2 / ||theta_init||^2
```

Também usamos LR discriminativa:

```text
embeddings + layer 1: 1e-5
layer 2 + MLM head:    3e-5
lambda L2-SP:          10
```

Foram executados o treino cronológico, seu pseudo-período correspondente
e uma ablação cronológica com a mesma LR discriminativa, mas `lambda=0`.
Todos mantiveram o protocolo 3+2 épocas e seleção sem gold.

| Condição | layer 1 | layer 2 | média layers |
|---|---:|---:|---:|
| full anterior | 0,325 | 0,030 | 0,189 |
| LR discriminativa | **0,340** | 0,012 | 0,196 |
| LR discriminativa + L2-SP | 0,338 | 0,014 | 0,204 |
| LR discriminativa + L2-SP, pseudo | 0,341 | 0,116 | 0,258 |

No checkpoint cronológico selecionado, a distância relativa da camada 2
foi `0,0416` sem L2-SP e `0,0294` com L2-SP. Assim, a penalização restringiu
os pesos, mas não preservou o ranking semântico da última camada.

As estimativas pontuais sugeriram:

1. o aumento pontual de layer 1 acompanha a LR menor nas camadas
   inferiores;
2. L2-SP com `lambda=10` é semanticamente indistinguível de `lambda=0`
   neste protocolo;
3. proximidade paramétrica à inicialização não é um substituto suficiente
   para preservação funcional das representações;
4. o pseudo-período fica numericamente acima na layer 2, ainda sem
   inferência causal.

## Bootstrap e trajetória com orçamento alinhado

Após uma segunda revisão crítica, adicionamos bootstrap pareado por palavra
com 20.000 réplicas.

Na layer 1, as diferenças entre `init=0,298` e todos os modelos treinados
incluem zero. Na layer 2, tanto os ICs individuais quanto as diferenças
contra `init` também incluem zero. Portanto, os ganhos e perdas de poucos
centésimos são padrões descritivos, não efeitos estabelecidos.

Também avaliamos D1 em marcos idênticos:

| Épocas | Cronológico layer 2 | Pseudo layer 2 | IC 95% crono-pseudo |
|---:|---:|---:|---:|
| 0,25 | 0,012 | 0,046 | [-0,132; 0,062] |
| 0,5 | 0,062 | 0,153 | [-0,238; 0,048] |
| 1 | 0,059 | 0,176 | [-0,278; 0,031] |
| 2 | 0,030 | 0,088 | [-0,170; 0,044] |

O pseudo é numericamente superior em todos os marcos, mas nenhuma
diferença com orçamento igual exclui zero. Assim, não atribuímos mais
causalmente a queda à ordem cronológica.

## Interpretação revisada

Os resultados não apoiam a narrativa simples:

```text
fine-tuning -> catastrophic forgetting global
```

Uma descrição mais fiel é:

1. o BERT integral evita o colapso abrupto observado no `RealStaticMLM`;
2. o MLM mantém ou eleva levemente o ponto estimado da camada inferior,
   sem diferença pareada estabelecida;
3. a layer 2 é um readout fraco e instável já no baseline congelado;
4. os pontos estimados são compatíveis com reorganização por camada, mas
   os ICs não distinguem essa hipótese de ruído;
5. reduzir a LR inferior aumenta o ponto estimado de layer 1, sem melhora
   pareada estatisticamente estabelecida;
6. L2-SP reduz drift de pesos sem benefício semântico observável.

Isso confirma que a adaptação incompleta do `RealStaticMLM` era um fator
importante. Também mostra que usar apenas a última camada é inadequado
para medir mudança lexical após MLM temporal.

## Decisão experimental

### O que não fazer

- Não voltar agora para corrigir `RealStaticMLM`.
- Não usar `last_layer` como representação principal.
- Não adotar congelamento total da camada inferior como solução.
- Não interpretar APD absoluta como mudança sem calibração.
- Não promover o antigo `bert-tiny mean_last_4=0,399` a teto comparável.
- Não ampliar a grade de `lambda` de L2-SP sem uma nova hipótese.

### Próximo passo recomendado

Pausar a regularização do MLM temporal. Não há evidência suficiente para
justificar distillation da layer 2, e mesmo uma layer estável não resolve
a identificabilidade entre contexto e sentido.

O próximo passo é a **Porta 1 da arquitetura externa de WSD**:

1. selecionar um modelo externo congelado de compatibilidade
   contexto-gloss, sem ajuste no SemEval;
2. testar os subconjuntos heurísticos predefinidos de `plane`;
3. reportar accuracy e intervalos para geometria D0, ferramenta D0 e
   aeronave D1;
4. só avançar para as 37 palavras se essa régua ler adequadamente o corpus
   lematizado.

## Artefatos

```text
outputs/bert_tiny_option_d_full_seed1000/
outputs/bert_tiny_option_d_full_seed1001/
outputs/bert_tiny_option_d_random_control_seed1000/
outputs/bert_tiny_option_d_freeze_lower_seed1000/
outputs/bert_tiny_option_d_controls/
outputs/bert_tiny_l2sp_chronological_seed1000/
outputs/bert_tiny_l2sp_random_seed1000/
outputs/bert_tiny_discriminative_lr_seed1000/
outputs/bert_tiny_option_d_bootstrap/
```
