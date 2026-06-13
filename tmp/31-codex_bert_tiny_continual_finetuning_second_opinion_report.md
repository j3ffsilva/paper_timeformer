# Pedido de segunda opinião: inicialização `bert-tiny`, treino temporal e perda de sinal semântico

## O que queremos avaliar

Implementamos uma tentativa de inicializar o Timeformer com
`prajjwal1/bert-tiny` antes do treino MLM cronológico:

```text
theta_init = adaptação de bert-tiny para o vocabulário próprio
theta_0    = treino(theta_init, D0)
theta_1    = continua_treino(theta_0, D1)
```

Os corpora são:

```text
D0 = 1810-1860
D1 = 1960-2010
```

O benchmark é SemEval-2020 Task 1, inglês lematizado, com 37 palavras-alvo.
O resultado principal foi que `theta_init` preserva sinal semântico útil,
mas o regime completo de treino reduz esse sinal ao patamar do modelo
anterior treinado do zero.

Queremos uma segunda opinião sobre:

1. se a interpretação de esquecimento durante fine-tuning é sustentada;
2. quais incompatibilidades arquiteturais podem estar causando a perda;
3. qual deve ser a próxima ablação mínima e metodologicamente limpa;
4. se ainda vale adaptar BERT ao modelo próprio ou se devemos preservar
   integralmente a arquitetura/tokenização pré-treinada.

Não assuma que nossa conclusão está correta. Procure explicações
alternativas, controles ausentes e riscos de seleção pelo benchmark.

## Contexto anterior

O modelo anterior era um Transformer próprio:

```text
d_model=128
3 camadas
4 heads
d_ff=384
vocabulário próprio lematizado, cerca de 27 mil tokens
treinado do zero com MLM contínuo
```

Ao comparar ocorrências de D0 e D1 com o mesmo checkpoint `theta1`
(`theta1_d0` contra `theta1_d1`), a melhor APD ficou em Spearman
aproximadamente `0,20`, sem significância estatística.

Como controle, aplicamos `bert-base-uncased` congelado às mesmas frases:

| Representação | APD Spearman | p |
|---|---:|---:|
| última camada | 0,594 | 0,0001 |
| média das últimas 4 | 0,591 | 0,0001 |

Isso mostrou que a tarefa é solucionável e sugeriu que o gargalo principal
era o encoder próprio.

## O que foi implementado

### Arquitetura-alvo

O `RealStaticMLM` foi configurado para acompanhar o `bert-tiny`:

```text
d_model=128
2 camadas
2 heads
d_ff=512
post-norm
GELU
LayerNorm epsilon=1e-12
máscara de padding
seq_len=32
```

Os defaults antigos de pré-norm/ReLU foram preservados para permitir a
leitura dos checkpoints anteriores.

### Transferência das camadas

Para cada camada BERT:

```text
query/key/value
    -> self_attn.in_proj_weight e in_proj_bias

attention.output.dense
    -> self_attn.out_proj

intermediate.dense
    -> linear1

output.dense
    -> linear2

attention.output.LayerNorm
    -> norm1

output.LayerNorm
    -> norm2
```

O carregamento valida dimensões, número de camadas/heads, feed-forward,
ordem de normalização e ativação.

### Transferência dos embeddings lexicais

O vocabulário do projeto contém 27.311 tokens inteiros, enquanto o BERT
usa WordPieces. Para cada token próprio:

1. removemos `_nn` ou `_vb`, quando presente;
2. tokenizamos com o tokenizer do `bert-tiny`;
3. calculamos a média dos embeddings dos WordPieces;
4. usamos essa média para inicializar o embedding do token inteiro.

Todos os 27.311 tokens receberam algum embedding. Tokens especiais foram
mapeados para os especiais correspondentes.

### Componentes não transferidos

Permaneceram aleatórios:

```text
pos_emb
mlm_head
```

O modelo próprio também não possui:

```text
embedding LayerNorm do BERT
token-type embeddings
MLM transform completo do BERT
weight tying entre embedding e decoder
```

Essas diferenças são importantes para interpretar o experimento.

### Integração no pipeline

Foram criados:

```text
src/timeformers/pretrained_init.py
scripts/init_pretrained_encoder.py
tests/test_pretrained_init.py
```

O script principal recebeu:

```text
--init-from-pretrained
--encoder-norm-order
--activation
--layer-norm-eps
--mask-padding / --no-mask-padding
```

O checkpoint é carregado com `strict=True`.

## Verificação da implementação

Foram adicionados testes sintéticos para:

- concatenação correta de Q/K/V;
- cópia de atenção, FFN e LayerNorm;
- média dos WordPieces;
- remoção do sufixo POS;
- rejeição de arquitetura pré-norm incompatível.

Resultado:

```text
83 testes passaram
py_compile passou
git diff --check passou
checkpoint carregado com todas as chaves correspondentes
forward produziu tensores finitos
```

O checkpoint inicial possui 7.419.823 parâmetros e aproximadamente 29 MB.

## Protocolo de treino executado

Configuração:

```text
base_epochs=12
epochs_per_period=8
batch_size=192
lr=1e-4 constante
AdamW, weight_decay=1e-2
gradient clipping=1.0
sem validação
sem early stopping
sem scheduler
seed=1000
```

Dados:

```text
D0: 370.216 janelas
D1: 408.949 janelas
```

Treino:

| Período | Épocas | Passos | Loss inicial | Loss final |
|---|---:|---:|---:|---:|
| D0 | 12 | 23.148 | 6,9455 | 4,7883 |
| D1 | 8 | 17.040 | 5,4283 | 4,9589 |

Total:

```text
40.188 passos
aproximadamente 1h59 em RTX 3060 Laptop 6 GB
```

O otimizador foi mantido entre D0 e D1, como no pipeline contínuo anterior.

## Avaliações realizadas

### Métrica principal

Para evitar confundir mudança do corpus com mudança do checkpoint, usamos
um encoder fixo:

```text
APD(theta, D0, D1)
```

Para cada palavra, calculamos a distância cosseno média entre todas as
ocorrências amostradas de D0 e D1 sob o mesmo encoder. Depois correlacionamos
com o gold graded do SemEval usando Spearman.

Também foram medidos:

- `Delta` do perfil relacional;
- NMI entre clusters e período;
- perfis ocultos relacionais;
- protótipo "modos primeiro";
- oráculo congelado do próprio `bert-tiny`.

### Resultado do `bert-tiny` congelado

O script de oráculo usa a arquitetura e tokenização originais do Hugging
Face, sem adaptação nem treino nos corpora temporais.

| Representação | APD Spearman | p | ROC-AUC | AP |
|---|---:|---:|---:|---:|
| última camada | 0,093 | 0,584 | 0,583 | 0,608 |
| `mean_last_4` | **0,399** | **0,014** | 0,723 | 0,629 |

Como `bert-tiny` tem só duas camadas, `mean_last_4` agrega:

```text
embedding output + layer 1 + layer 2
```

Portanto, essa representação não é diretamente equivalente ao nosso
`mean_last_2`, que agrega apenas as duas camadas Transformer.

### Checkpoint adaptado antes do treino

Usamos o mesmo `theta_init` como encoder fixo para D0 e D1:

| Layer | APD Spearman | p | ROC-AUC | AP |
|---|---:|---:|---:|---:|
| layer_2 | **0,277** | 0,097 | 0,628 | 0,625 |
| mean_last_2 | 0,224 | 0,183 | 0,640 | 0,592 |

No avaliador de perfis ocultos, o melhor método foi:

```text
centered_relational_apd
mean_last_2
Spearman=0,337
p=0,041
ROC-AUC=0,649
AP=0,566
```

Assim, a adaptação preservou sinal mensurável, embora inferior ao
`bert-tiny` original.

### Após 12 épocas em D0 (`theta0`)

| Layer | APD Spearman | p | ROC-AUC | AP |
|---|---:|---:|---:|---:|
| layer_2 | -0,055 | 0,747 | 0,497 | 0,498 |
| mean_last_2 | -0,016 | 0,926 | 0,515 | 0,521 |

O sinal praticamente desapareceu já em `theta0`.

### Após mais 8 épocas em D1 (`theta1`)

| Layer | APD Spearman | p | ROC-AUC | AP |
|---|---:|---:|---:|---:|
| layer_2 | 0,123 | 0,467 | 0,551 | 0,592 |
| mean_last_2 | **0,204** | 0,225 | 0,548 | 0,575 |

Há recuperação parcial em D1, mas o resultado volta ao mesmo patamar
aproximado do modelo antigo treinado do zero.

O melhor perfil oculto diagonal depois do treino foi:

```text
centered_relational_apd
layer_1
Spearman=0,200
p=0,236
```

### Modos primeiro

O clustering das ocorrências sob `theta1` continuou sem distinguir
consistentemente palavras mudadas de controles:

| Palavra | Classe esperada | k | JSD entre períodos |
|---|---|---:|---:|
| graft_nn | mudou | 5 | 0,506 |
| multitude_nn | estável | 4 | 0,317 |
| chairman_nn | estável | 5 | 0,180 |
| plane_nn | mudou | 5 | 0,168 |
| tree_nn | estável | 5 | 0,102 |
| ball_nn | estável | 5 | 0,100 |
| face_nn | estável | 4 | 0,098 |
| lane_nn | estável | 5 | 0,087 |

Mantemos isso apenas como diagnóstico qualitativo.

## O que os resultados parecem mostrar

Nossa leitura atual é:

1. `bert-tiny` contém sinal semântico temporal útil antes do treino;
2. converter WordPieces em tokens inteiros preserva parte desse sinal;
3. o regime de 12+8 épocas com LR `1e-4` destrói a maior parte do sinal;
4. a queda já aparece depois de D0, antes da continuação em D1;
5. o problema imediato não parece ser falta absoluta de capacidade, mas
   adaptação arquitetural incompleta e/ou fine-tuning excessivamente
   agressivo.

Ainda assim, "catastrophic forgetting" é apenas uma hipótese operacional.
O experimento não localiza quando a queda ocorre e não separa vários
mecanismos possíveis.

## Limitações e explicações alternativas

### 1. Head MLM aleatório transmite gradientes grandes ao encoder

O decoder do vocabulário próprio começa aleatório. Todo o encoder é
treinado desde o primeiro passo para sustentar esse head. A queda pode ser
uma fase de reconstrução do espaço para o novo decoder, não apenas
esquecimento causado por muitos dados.

### 2. Não há weight tying

No BERT, embeddings de entrada e decoder MLM compartilham pesos. Aqui:

```text
token_emb != mlm_head.proj
```

Isso remove uma restrição estrutural importante e pode permitir que o
encoder se deforme para acomodar um decoder independente.

### 3. A camada de embeddings não foi transferida integralmente

Transferimos apenas embeddings lexicais médios. Permaneceram ausentes ou
aleatórios:

```text
positional embeddings
embedding LayerNorm
token-type embeddings
dropout/normalização exatamente equivalentes
```

Embora o handoff original recomendasse `pos_emb` aleatório, seria possível
copiar as primeiras 32 posições do BERT. A ausência da embedding LayerNorm
pode ser ainda mais relevante porque as camadas BERT foram treinadas
esperando uma distribuição de entrada normalizada.

### 4. Média de WordPieces não reproduz a distribuição original

Uma palavra de vários WordPieces vira um único token cujo embedding é a
média estática das peças. O encoder passa a receber sequências mais curtas,
sem a estrutura sublexical vista no pré-treino. Isso pode explicar a perda
entre o oráculo congelado (`0,399`) e `theta_init` (`0,277` em APD).

### 5. Comparação de camadas não é perfeitamente alinhada

O melhor resultado do `bert-tiny` original usa embedding output + duas
camadas. O modelo adaptado não expõe uma representação equivalente à
embedding output normalizada do BERT. Comparar `mean_last_4` com
`mean_last_2` mistura diferença de modelo e diferença de readout.

### 6. LR constante e número elevado de passos

Foram 23.148 passos apenas em D0, todos com LR `1e-4`, sem warmup ou
decay. Isso é agressivo para fine-tuning de BERT pequeno.

### 7. Não salvamos checkpoints por época

Temos apenas:

```text
theta_init
theta0 após 12 épocas
theta1 após mais 8 épocas
```

Não sabemos se o sinal caiu na primeira época, degradou gradualmente ou
teve um ponto ótimo intermediário.

### 8. Não há validação independente do gold

O treino não usa `truth.tsv`, o que é correto. Porém, escolher LR, épocas
ou congelamento pelo Spearman dos mesmos 37 alvos causaria overfitting de
configuração. Precisamos de um critério de parada sem gold, um conjunto de
desenvolvimento externo, ou uma ablação pré-registrada.

### 9. Uma única seed

Todos os resultados de treino vêm de `seed=1000`. Não conhecemos a
variabilidade do efeito.

### 10. O objetivo científico pode conflitar com preservar uma régua fixa

Treino temporal contínuo quer adaptar o encoder a cada período. Medição com
encoder fixo quer uma régua estável. Talvez seja estruturalmente inadequado
usar o mesmo conjunto de parâmetros para:

```text
aprender o período
e
servir de instrumento comparável entre períodos
```

Pode ser necessário separar backbone semântico estável e adaptadores
temporais.

## Próximos passos que estamos considerando

### Opção A: ablação conservadora no pipeline atual

Rodar uma grade pequena, definida antes de olhar o gold:

```text
LR: 3e-5 e 5e-5
épocas: D0=3, D1=2
salvar checkpoint a cada época
warmup + decay linear ou cosine
```

Critérios sem gold possíveis:

- perda MLM de validação em documentos retidos;
- distância paramétrica ao checkpoint inicial;
- estabilidade em controles sintéticos;
- retenção em uma tarefa lexical externa não temporal.

Depois, aplicar uma única vez o protocolo SemEval fixo.

### Opção B: corrigir a compatibilidade BERT antes de novo treino

Antes de gastar outra rodada longa:

1. copiar as primeiras 32 positional embeddings;
2. adicionar e copiar embedding LayerNorm;
3. copiar/adaptar o transform do MLM head;
4. usar weight tying entre token embeddings e decoder;
5. comparar `theta_init` com o BERT original usando readouts equivalentes.

Esta opção testa se a queda inicial `0,399 -> 0,277` vem da adaptação
incompleta.

### Opção C: congelamento progressivo

Possibilidades:

- congelar embeddings e camada 1 durante D0;
- treinar primeiro apenas o head MLM;
- descongelar camada 2 e depois camada 1;
- usar LLRD, com LR menor nas camadas inferiores;
- regularizar contra `theta_init` com L2-SP ou distillation.

### Opção D: preservar BERT integralmente

Abandonar o vocabulário/tokenizador próprios para o encoder:

```text
texto lematizado -> tokenizer WordPiece original
bert-tiny completo
MLM head original
fine-tuning temporal conservador
```

As palavras-alvo podem ser representadas pela média de seus subtokens.
Isso preserva exatamente o regime de pré-treino e elimina grande parte da
engenharia de transferência. A desvantagem é alterar contratos existentes
do pipeline relacional.

### Opção E: backbone congelado + parâmetros temporais pequenos

Manter o encoder pré-treinado congelado e aprender por período:

- adapters;
- LoRA;
- prompts;
- pequenos heads temporais;
- residual temporal regularizado.

Isso separa a régua semântica fixa da adaptação temporal. A mudança pode
ser medida na resposta dos módulos temporais ou em distribuições de
ocorrências sob o backbone comum.

## Nossa recomendação provisória

Não recomendamos repetir diretamente 12+8 épocas.

A sequência que parece mais informativa é:

1. **corrigir a compatibilidade de entrada e MLM**: positional embeddings,
   embedding LayerNorm, transform MLM e weight tying;
2. medir novamente `theta_init` sem treino;
3. salvar checkpoints por época;
4. treinar primeiro 3+2 épocas com LR `3e-5`, warmup/decay e pelo menos
   duas seeds;
5. incluir uma variante com camada inferior congelada;
6. escolher parada/configuração sem consultar `truth.tsv`;
7. só então executar a avaliação SemEval;
8. se ainda houver queda forte, migrar para BERT integral ou backbone
   congelado com adapters.

Minha preferência técnica é testar primeiro a **Opção D** como baseline
simples: preservar tokenizer, embeddings, arquitetura e head originais do
`bert-tiny`. Ela responde se o fracasso vem da adaptação para o modelo
próprio. Em paralelo, uma variante da **Opção E** parece mais coerente com
a necessidade de uma régua semântica estável.

## Perguntas para a segunda opinião

1. A evidência disponível sustenta "esquecimento no fine-tuning", ou há
   uma explicação mais provável?
2. Qual incompatibilidade é mais crítica: embedding LayerNorm, positional
   embeddings, MLM head aleatório, ausência de weight tying ou colapso de
   WordPieces?
3. Vale corrigir o `RealStaticMLM`, ou preservar BERT integralmente é um
   baseline obrigatório antes de qualquer outra ablação?
4. Qual estratégia mínima de fine-tuning recomendaria para 780 mil
   janelas: LR, warmup, scheduler, épocas e congelamento?
5. Como escolher checkpoint sem usar os 37 rótulos do SemEval?
6. L2-SP, distillation, EWC, adapters ou LoRA fazem sentido neste regime?
7. Há um desenho melhor para separar encoder semântico estável e dinâmica
   temporal sem perder a contribuição científica do Timeformer?
8. Que controles adicionais são necessários para distinguir:

```text
esquecimento semântico
adaptação ao vocabulário
aprendizado legítimo do domínio histórico
artefato do readout/métrica
```

9. O resultado de `bert-tiny mean_last_4=0,399` é uma comparação justa,
   dado que inclui a embedding output?
10. Qual experimento único teria maior valor informacional antes de
    investir em uma nova rodada longa?

## Arquivos relevantes

Código:

```text
src/timeformers/real_models.py
src/timeformers/pretrained_init.py
src/timeformers/train.py
scripts/init_pretrained_encoder.py
scripts/run_diachronic_relational_experiment.py
scripts/evaluate_hidden_relational_profiles.py
scripts/evaluate_fixed_encoder_v2.py
scripts/evaluate_pretrained_oracle_v2.py
scripts/prototype_modes_first_v2.py
tests/test_pretrained_init.py
```

Resultados:

```text
outputs/semeval2020_pmi_pretrained_init_d128/
outputs/semeval2020_pmi_pretrained_init_d128/init_encoder_fixed_eval/
outputs/semeval2020_pmi_pretrained_init_d128/fixed_encoder_eval/
outputs/semeval2020_pmi_pretrained_init_d128/pretrained_oracle_bert_tiny/
outputs/semeval2020_pmi_pretrained_init_d128/modes_first_v2/
```

Documentação:

```text
docs/11-next_machine_handoff_v2.md
docs/14-perfil_relacional_v2_resultados_fase1.md
docs/05-relational_change_current_plan.md
```
