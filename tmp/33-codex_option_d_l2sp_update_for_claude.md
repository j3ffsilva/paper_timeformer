# Atualização para segunda opinião: BERT integral, dinâmica por camada e falha do L2-SP

## Objetivo desta atualização

Desde a última revisão, substituímos a adaptação parcial de `bert-tiny`
para o `RealStaticMLM` por um pipeline que preserva integralmente:

```text
tokenizer WordPiece
embeddings lexicais e posicionais
embedding LayerNorm
arquitetura BERT
MLM transform e decoder
weight tying
```

Também executamos controles cronológicos, pseudo-períodos, congelamento,
LR discriminativa e L2-SP. Queremos uma nova avaliação crítica sobre:

1. a interpretação de redistribuição do sinal entre camadas;
2. o valor científico de preservar a última camada;
3. se distillation funcional é realmente o próximo teste correto;
4. se deveríamos parar de otimizar o encoder MLM e avançar para uma régua
   externa de WSD/open set.

Não assuma que nossa interpretação está correta. Em particular, procure
efeitos de readout, seleção, baixa potência (`n=37`) e diferenças entre o
controle pseudo-período e o experimento cronológico.

## Protocolo atual

Modelo:

```text
prajjwal1/bert-tiny
2 camadas
d_model=128
tokenizer e MLM head originais
seq_len=32
```

Dados:

```text
D0 = 1810-1860
D1 = 1960-2010
benchmark = SemEval-2020 Task 1, inglês lematizado, 37 alvos
```

Treino principal:

```text
D0: 3 épocas
D1: 2 épocas
batch: 192
LR: 3e-5
warmup: 10%
linear decay
holdout por documento
mascaramento dinâmico determinístico
```

Os checkpoints são salvos em `0,25`, `0,5`, `1` época e depois por época.
Não usamos `truth.tsv` para selecionar checkpoints.

Para `theta0`, a seleção usa somente a loss de validação de D0. Para
`theta1`, usa a média das losses de D0 e D1. Entre checkpoints dentro de
1% da melhor loss:

1. maximiza-se o cosseno com representações-âncora iniciais;
2. desempata-se pela menor distância L2 à inicialização.

O checkpoint final precisa pertencer a D1.

## Comparação alinhada

Corrigimos uma comparação anterior em que `mean_last_4=0,399` misturava
embedding output e as duas camadas, além de usar outro pré-processamento.
O novo avaliador usa as mesmas frases, amostras, tokenizer e
`max_length=32` em todas as condições.

Baseline congelado:

| Readout | Spearman |
|---|---:|
| embedding | -0,018 |
| layer 1 | 0,298 |
| layer 2 | 0,136 |
| média layers 1+2 | 0,241 |
| média embedding+layers | 0,160 |

## Resultado do fine-tuning integral

Duas seeds cronológicas:

| Readout | Seed 1000 | Seed 1001 |
|---|---:|---:|
| embedding | 0,038 | 0,032 |
| layer 1 | **0,325** | **0,322** |
| layer 2 | 0,030 | 0,038 |
| média layers | 0,189 | 0,194 |

Na seed 1000, a reorganização já estava presente em `theta0`:

```text
layer 1 = 0,324
layer 2 = -0,019
```

As representações-âncora finais mantiveram cosseno próximo de `0,95` com
a inicialização. Não houve salto global ou evidência de colapso numérico.

## Pseudo-períodos e congelamento

No controle pseudo-período, D0+D1 foram embaralhados e repartidos com os
tamanhos originais:

| Readout | Spearman |
|---|---:|
| layer 1 | **0,332** |
| layer 2 | **0,153** |
| média layers | 0,270 |

O ganho em layer 1 também ocorre sem ordem temporal, sugerindo adaptação
geral ao domínio. A layer 2, porém, preserva muito mais sinal no
pseudo-controle que no cronológico.

Congelar embeddings + layer 1 produziu:

```text
layer 1 = 0,298
layer 2 = 0,017
```

Logo, congelar a parte inferior não protege a camada superior e impede a
melhora observada em layer 1.

## Controles externos pareados por frequência

Selecionamos 37 palavras fora do benchmark, pareadas em frequência em D0
e D1.

| Condição | Grupo | layer 1 | layer 2 |
|---|---|---:|---:|
| cronológica | alvos | 0,318 | 0,339 |
| cronológica | controles | 0,304 | 0,343 |
| pseudo | alvos | 0,323 | 0,383 |
| pseudo | controles | 0,309 | 0,372 |

A APD absoluta é semelhante entre alvos e controles. O resultado SemEval
depende do ranking relativo, não de uma separação global de magnitudes.
Ainda não temos controles semanticamente pareados tão fortes quanto os
controles de frequência.

## LR discriminativa e L2-SP

Implementamos:

```text
embeddings + layer 1: LR 1e-5
layer 2 + MLM head:    LR 3e-5

loss = loss_MLM
     + lambda * ||theta_layer2 - theta_init||^2 / ||theta_init||^2

lambda = 10
```

Executamos:

1. cronológico com LR discriminativa e `lambda=0`;
2. cronológico com LR discriminativa e `lambda=10`;
3. pseudo-período com LR discriminativa e `lambda=10`.

| Condição | layer 1 | layer 2 | média layers |
|---|---:|---:|---:|
| full anterior | 0,325 | 0,030 | 0,189 |
| LR discriminativa | **0,340** | 0,012 | 0,196 |
| LR discriminativa + L2-SP | 0,338 | 0,014 | 0,204 |
| L2-SP, pseudo-período | 0,341 | 0,116 | 0,258 |

No checkpoint cronológico selecionado:

```text
layer2 relative L2, lambda=0:  0,0416
layer2 relative L2, lambda=10: 0,0294
```

Portanto, L2-SP restringiu efetivamente os pesos, mas não preservou o
ranking semântico da layer 2. A melhora de layer 1 é explicada pela LR
inferior menor, pois aparece igualmente com `lambda=0`.

Nossa interpretação provisória é:

```text
proximidade paramétrica != preservação funcional
```

Não pretendemos abrir uma grade de `lambda` selecionada pelos 37 alvos.

## O que mudou no diagnóstico

A narrativa anterior:

```text
fine-tuning causa catastrophic forgetting global
```

não é sustentada pelo BERT integral. A descrição atual é:

1. a adaptação incompleta do `RealStaticMLM` contribuía para a perda;
2. o BERT integral preserva e melhora levemente o sinal na layer 1;
3. a layer 2 é reorganizada já durante D0;
4. parte da reorganização é adaptação geral ao domínio;
5. há perda adicional associada à ordem cronológica;
6. L2-SP reduz drift de pesos sem recuperar a geometria semântica útil;
7. usar apenas a última camada após MLM temporal é uma escolha ruim de
   readout.

Ainda não está demonstrado que a diferença cronológico versus pseudo seja
um efeito temporal semanticamente desejável. Ela também pode refletir
assimetria lexical, dificuldade MLM ou composição distinta dos períodos.

## Relação com a proposta de WSD/open set

Esses experimentos melhoram o diagnóstico do encoder, mas não resolvem as
paredes de identificabilidade discutidas anteriormente:

```text
mudança em P(contexto | palavra)
não identifica necessariamente
mudança em P(sentido | palavra)
```

Uma layer 1 mais estável melhora a régua contextual, mas não transforma
clusters ou APD em sentidos lexicais identificáveis. Assim, vemos dois
eixos distintos:

1. **eixo de engenharia/diagnóstico**: entender a plasticidade do MLM
   temporal e preservar uma geometria contextual útil;
2. **eixo do estimando científico**: medir distribuições de sentidos por
   uma régua externa, possivelmente WSD probabilístico com open set.

Não queremos confundir sucesso no primeiro eixo com solução do segundo.

## Nossa proposta de próximo experimento mínimo

Testar distillation funcional apenas na layer 2, mantendo:

```text
embeddings + layer 1: LR 1e-5
layer 2 + MLM head:    LR 3e-5
teacher: theta_init congelado
student: modelo temporal
```

Em vez de aproximar pesos, preservaríamos a geometria das representações
em frases-âncora. O desenho preferido é:

1. amostrar âncoras por documento, sem usar alvos ou `truth.tsv`;
2. em D0, usar somente âncoras disponíveis em D0;
3. em D1, repetir âncoras de D0 e adicionar âncoras de D1;
4. guardar hidden states do teacher para evitar um segundo forward;
5. aplicar a loss somente a tokens não especiais e não padding;
6. comparar cosseno ponto a ponto e, idealmente, preservação da matriz de
   similaridades dentro do lote;
7. calibrar o peso da distillation por razão de normas de gradiente em um
   piloto, sem consultar o SemEval;
8. executar uma seed cronológica primeiro;
9. só executar pseudo-período e segunda seed se houver melhora da layer 2
   sem degradação material da validação MLM.

Uma formulação possível:

```text
L = L_MLM
  + alpha * mean(1 - cos(h_student, h_teacher))
  + beta  * ||C_student - C_teacher||_F^2
```

onde `C` é uma matriz de similaridades entre representações-âncora. Para
evitar uma grade oportunista, escolheríamos `alpha` e `beta` por uma regra
prévia de contribuição/gradiente, não pelo gold.

## Melhores passos, em nossa avaliação

### Passo 1: executar uma distillation funcional mínima

Este é o controle causal mais direto para a falha do L2-SP. Ele testa se
preservar a função da layer 2, e não seus parâmetros, mantém o ranking.
Deve ser um experimento pequeno e pré-especificado, não uma busca extensa.

### Passo 2: não condicionar o projeto ao resgate da layer 2

Se a distillation não ajudar, usar layer 1 como readout contextual estável
e encerrar a linha de regularização do backbone. A layer 2 pode continuar
como objeto de análise de plasticidade, não como régua principal.

### Passo 3: reforçar os controles antes de interpretar temporalidade

O contraste cronológico/pseudo precisa de:

```text
segunda seed pseudo-período
controle de ordem invertida D1 -> D0
controles lexicais semanticamente pareados
análise por frequência, polissemia, POS e dificuldade MLM
```

O controle de ordem invertida tem alto valor: separa efeito de ordem de
uma propriedade específica do corpus moderno como segundo estágio.

### Passo 4: avançar em paralelo para uma régua externa de sentidos

Mesmo que a distillation preserve layer 2, APD continua medindo geometria
contextual, não diretamente `P(sentido | palavra)`. Para a reivindicação
principal de mudança semântica, recomendamos manter o plano de WSD
probabilístico/open set como baseline metodológico separado.

### Passo 5: definir uma regra de parada

Sugerimos encerrar a otimização do MLM temporal se:

```text
distillation funcional não superar a LR discriminativa;
o efeito não replicar em pseudo/segunda seed;
ou a melhora depender da escolha do readout usando truth.tsv.
```

Nesse caso, a contribuição mais defensável seria:

1. mostrar empiricamente a redistribuição por camada no MLM temporal;
2. recomendar layer 1/encoder externo como régua;
3. separar plasticidade do encoder de mudança lexical identificável.

## Perguntas para a segunda opinião

1. A evidência sustenta redistribuição entre camadas ou há uma explicação
   alternativa mais provável?
2. O contraste cronológico versus pseudo-período é interpretável com uma
   única seed pseudo?
3. Distillation ponto a ponto, relacional ou ambas: qual controle mínimo
   tem maior valor informacional?
4. O teacher deve ser `theta_init`, o checkpoint anterior ao período ou
   uma média móvel?
5. Usar somente âncoras D0 durante D0 e replay D0+D1 durante D1 evita
   vazamento e ainda testa a hipótese correta?
6. Como calibrar `alpha`/`beta` sem usar os 37 alvos?
7. O controle de ordem invertida deve preceder a distillation?
8. Há razão científica para resgatar a layer 2, dado que layer 1 já é mais
   estável e melhor correlacionada?
9. Esses resultados fortalecem ou enfraquecem a motivação para a
   arquitetura externa de WSD/open set?
10. Qual experimento único você executaria antes de encerrar a linha de
    regularização do MLM temporal?

## Arquivos e artefatos

Código:

```text
src/timeformers/bert_continual.py
scripts/run_bert_tiny_continual_option_d.py
scripts/evaluate_bert_checkpoint_apd.py
scripts/select_frequency_matched_controls.py
tests/test_bert_continual.py
```

Resultados:

```text
outputs/bert_tiny_option_d_full_seed1000/
outputs/bert_tiny_option_d_full_seed1001/
outputs/bert_tiny_option_d_random_control_seed1000/
outputs/bert_tiny_option_d_freeze_lower_seed1000/
outputs/bert_tiny_option_d_controls/
outputs/bert_tiny_l2sp_chronological_seed1000/
outputs/bert_tiny_l2sp_random_seed1000/
outputs/bert_tiny_discriminative_lr_seed1000/
```

Relatórios:

```text
tmp/32-codex_option_d_execution_results.md
tmp/31-codex_bert_tiny_continual_finetuning_second_opinion_report.md
tmp/23-claude_scalable_temporal_semantic_architecture_review.md
docs/05-relational_change_current_plan.md
docs/11-next_machine_handoff_v2.md
```

Verificação atual:

```text
89 testes passaram
py_compile passou
git diff --check passou
```
