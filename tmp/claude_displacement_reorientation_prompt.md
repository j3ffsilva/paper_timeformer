# Prompt para revisão externa: reorientação do Paper 2 para deslocamentos temporais

Você deve realizar uma revisão técnica e científica independente do projeto:

`/Users/jeff/Documents/trabalhos/papers/paper-timeformers`

Não altere nenhum arquivo de código, documentação ou resultado existente. Sua tarefa é somente analisar a nova proposta, confrontá-la com o planejamento e a implementação atuais e escrever um parecer.

Escreva obrigatoriamente o parecer final em:

`./tmp/timeformer_displacement_reorientation_review.md`

O parecer deve ser autocontido e detalhado o suficiente para orientar a reescrita do planejamento e a posterior refatoração do código.

---

## 1. Contexto e motivo desta revisão

O Paper 1, submetido ao IBERAMIA, comparou mecanismos de condicionamento temporal, incluindo Token-Time, Memory-Augmented e Additive. Os resultados indicaram que esses mecanismos não se diferenciaram de forma relevante no cenário estudado.

O Paper 2 foi inicialmente planejado em torno da seguinte formulação:

```text
token@time(S,t) = (h_s(t), m_s(t))
```

Nela:

- `h_s(t)` é produzido por um Transformer Token-Time treinado conjuntamente em todos os períodos;
- um agregador transforma ocorrências contextuais em `R_s(t)`;
- teacher/student temporal aprende `m_s(t)`, interpretado como estado de trajetória;
- a representação final concatena posição semântica e estado de trajetória;
- a própria trajetória é aprendida por masked trajectory distillation.

Esse planejamento está descrito principalmente em:

- `docs/novo_planejamento.md`
- `tmp/timeformer_external_review.md`
- `docs/synthetic_results_current.md`, caso exista

O código correspondente está principalmente em:

- `src/timeformers/models.py`
- `src/timeformers/representations.py`
- `src/timeformers/aggregators.py`
- `src/timeformers/aggregator_ssl.py`
- `src/timeformers/trajectory_models.py`
- `src/timeformers/trajectory_train.py`
- `src/timeformers/trajectory_metrics.py`
- `src/timeformers/experiment.py`
- `scripts/run_synthetic_pipeline.py`
- `scripts/run_d5a_student_ablation.py`
- `scripts/run_ssl_aggregator_sanity.py`

Após revisar a intenção científica, concluímos que esse pipeline se desviou da proposta que realmente queremos investigar.

---

## 2. Desvio identificado

O pipeline atual permite que o condicionamento temporal altere internamente toda a representação contextual produzida pelo Transformer. Além disso, aprende uma representação separada de trajetória e a concatena à representação semântica.

Isso significa que atualmente temos aproximadamente:

```text
h(w,c,t) = Transformer_condicionado_por_tempo(w,c,t)
m(w,t)   = encoder_temporal(R(w,t_0), ..., R(w,t_n))

token_time(w,t) = concat(R(w,t), m(w,t))
```

Esse desenho apresenta três diferenças em relação à intenção original:

1. O Transformer semântico não permanece como um modelo padrão treinado sem condicionamento temporal.
2. O significado-base e o efeito temporal não são componentes claramente separáveis.
3. A trajetória é aprendida como representação pelo modelo, embora pudesse ser derivada posteriormente da sequência de deslocamentos.

A preocupação é que inserir tempo dentro do Transformer ou ajustar conjuntamente seus pesos altera o próprio sistema de coordenadas semântico. Isso dificulta interpretar diferenças entre períodos como deslocamentos explícitos e comparáveis.

---

## 3. Nova proposta pretendida

Queremos estudar deslocamentos semânticos temporais explícitos sobre um espaço-base congelado.

### 3.1 Espaço-base

Treinar um Transformer padrão, sem qualquer informação ou condicionamento temporal, usando somente um período-base `t0`.

Exemplo:

```text
t0 = 1950
b(w,c) = Transformer_base(w,c)
```

Depois do treinamento, o Transformer-base é congelado permanentemente. Ele define o sistema de coordenadas semântico de referência.

Por definição:

```text
delta(w,t0) = 0
```

### 3.2 Deslocamento temporal

Para períodos posteriores, aprender um módulo externo ao Transformer-base:

```text
delta(w,t)
```

Esse deslocamento deve ser específico para palavra e período, e provavelmente deve ser um vetor no mesmo espaço de `b`, não apenas o identificador global do período.

A representação consultável seria:

```text
e(w@t) = b(w) + delta(w,t)
```

ou, no caso contextual:

```text
e(w,c@t) = b(w,c) + delta(w,t)
```

Precisamos decidir se `delta` também deve depender do contexto/sentido:

```text
delta(w,c,t)
```

Essa decisão é especialmente importante para polissemia e sentidos coexistentes.

### 3.3 Consultas desejadas

O sistema deve permitir:

```text
deslocamento(gay@1950, gay@1980)
```

produzindo um vetor ou magnitude de deslocamento semanticamente interpretável:

```text
e(gay@1980) - e(gay@1950)
```

Também deve permitir:

```text
similares(gay@1950)
similares(gay@1980)
```

As consultas de vizinhança devem ser realizadas em um espaço compartilhado e retornar relações coerentes com cada período.

É importante avaliar rigorosamente contra qual conjunto os vizinhos devem ser calculados:

- apenas palavras posicionadas no mesmo período;
- todas as palavras em todos os períodos;
- embeddings-base mais seus respectivos deslocamentos no período consultado.

### 3.4 Trajetória como análise posterior

Não queremos obrigatoriamente aprender uma representação de trajetória.

A trajetória de uma palavra deve ser derivada posteriormente da sequência:

```text
delta(w,t0), delta(w,t1), ..., delta(w,tn)
```

Algoritmos posteriores podem então medir:

- direção e magnitude acumuladas;
- velocidade;
- estabilidade;
- mudança gradual;
- mudança abrupta;
- bifurcação ou coexistência de sentidos;
- curvatura e pontos de mudança.

A contribuição principal pretendida passa a ser:

> aprender deslocamentos semânticos temporais explícitos e interpretáveis sobre um espaço semântico-base congelado.

A trajetória seria uma propriedade analisável desses deslocamentos, não uma representação que precisa ser aprendida por teacher/student.

---

## 4. Questões que o parecer deve avaliar

Analise criticamente a proposta. Não assuma que nossas premissas estão corretas. Identifique ambiguidades, impossibilidades, riscos e alternativas melhores.

### 4.1 Validade conceitual

1. A formulação `e(w@t) = b(w) + delta(w,t)` é adequada para o objetivo descrito?
2. Congelar um Transformer treinado somente em `t0` realmente preserva um sistema de coordenadas comparável?
3. Usar o Transformer-base congelado sobre textos de períodos posteriores é metodologicamente válido, considerando mudanças lexicais, sintáticas e de domínio?
4. O deslocamento deve ser:
   - vetorial no mesmo espaço de `b`;
   - uma dimensão concatenada;
   - um subespaço ortogonal;
   - uma transformação, como matriz ou adapter residual?
5. Quais restrições são necessárias para tornar `delta(w,t)` identificável e impedir soluções arbitrárias?
6. Como evitar que o módulo de deslocamento simplesmente reaprenda todo o embedding semântico?
7. O que significa preservar distâncias do Transformer-base, considerando que o deslocamento deve intencionalmente mudar algumas distâncias?

### 4.2 Objetivo de treinamento

Proponha objetivos de treinamento concretos para aprender `delta(w,t)` sem supervisão explícita de sentidos.

Avalie possibilidades como:

- MLM com Transformer-base congelado e apenas o módulo temporal treinável;
- adapter residual aplicado somente à representação-alvo;
- reconstrução/predição de contexto;
- objetivos contrastivos baseados em vizinhança local observável;
- regularização de pequena norma do deslocamento;
- suavidade temporal, sem impedir mudanças abruptas;
- ancoragem de palavras presumidamente estáveis;
- restrições de identidade em `t0`;
- esparsidade ou low-rank;
- alinhamento com o espaço-base.

Explique quais perdas seriam essenciais, opcionais ou perigosas.

### 4.3 Contextualidade e polissemia

O Transformer produz embeddings contextuais por ocorrência, mas as consultas desejadas usam `gay@1950` como objeto palavra-período.

Analise:

1. Como passar de ocorrências contextuais `b(w,c)` para uma representação consultável `e(w@t)`?
2. Um único `delta(w,t)` consegue representar sentidos coexistentes?
3. Devemos aprender múltiplos deslocamentos por palavra-período, por exemplo `delta_k(w,t)`?
4. Como induzir sentidos sem usar labels sintéticos como supervisão?
5. Como definir `similares(w@t)` quando uma palavra é polissêmica?
6. O uso de agregadores `mean`, `set` e `set_slots` ainda é útil, ou pertence à direção anterior?

### 4.4 Relação com o Paper 1 e contribuição científica

Avalie se a nova proposta:

- decorre logicamente do achado de equivalência entre mecanismos de condicionamento;
- constitui contribuição suficientemente nova e defensável;
- evita repetir técnicas clássicas de alinhamento de embeddings diacrônicos;
- deve ser posicionada como residual temporal, campo de deslocamento, adapter temporal ou outra formulação;
- possui risco de ser apenas uma versão de temporal word embeddings/dynamic embeddings já conhecidos.

Indique quais famílias de trabalhos relacionados precisam ser confrontadas. Não invente citações; se não tiver certeza, descreva a família de literatura sem atribuição específica.

### 4.5 Avaliação experimental

Proponha um novo benchmark e métricas alinhados à proposta.

As métricas devem avaliar diretamente:

- qualidade de vizinhança em cada período;
- magnitude e direção do deslocamento;
- estabilidade de palavras sem mudança;
- recuperação de mudanças graduais e abruptas;
- preservação do espaço-base;
- capacidade de distinguir deslocamento real de ruído contextual;
- polissemia e sentidos coexistentes;
- generalização para períodos e palavras não observados, se aplicável.

Avalie quais partes do corpus sintético atual ainda são aproveitáveis e quais precisam mudar.

### 4.6 Mudanças necessárias no código

Inspecione o código atual e produza uma classificação concreta:

1. **Manter sem alteração**
2. **Reaproveitar com adaptação**
3. **Descontinuar da configuração principal, mantendo apenas como baseline/ablação**
4. **Remover futuramente**
5. **Criar**

Considere explicitamente:

- `src/timeformers/models.py`
- `src/timeformers/train.py`
- `src/timeformers/representations.py`
- `src/timeformers/aggregators.py`
- `src/timeformers/aggregator_ssl.py`
- `src/timeformers/trajectory_models.py`
- `src/timeformers/trajectory_train.py`
- `src/timeformers/trajectory_losses.py`
- `src/timeformers/trajectory_metrics.py`
- `src/timeformers/experiment.py`
- scripts experimentais atuais

Proponha uma arquitetura modular para a nova implementação. Por exemplo, avalie a necessidade de componentes como:

```text
BaseSemanticEncoder
TemporalDisplacementModule
TemporalEmbeddingIndex
DisplacementTrainer
DisplacementMetrics
TrajectoryAnalyzer
```

Não se limite a esses nomes ou componentes se houver uma estrutura melhor.

### 4.7 Plano de migração

Proponha uma sequência segura e cientificamente informativa para migrar do pipeline atual:

1. primeiro teste mínimo capaz de falsificar a nova ideia;
2. baselines necessários;
3. ablações essenciais;
4. critérios para decidir se a nova direção funciona;
5. momento adequado para reescrever integralmente `docs/novo_planejamento.md`;
6. quais resultados atuais ainda podem ser aproveitados como diagnóstico ou baseline.

---

## 5. Formato obrigatório do parecer

Escreva o parecer em:

`./tmp/timeformer_displacement_reorientation_review.md`

Use esta estrutura:

1. **Resumo executivo**
2. **Sua compreensão da nova proposta**
3. **A proposta resolve o problema pretendido?**
4. **Principais riscos e ambiguidades**
5. **Formulação matemática recomendada**
6. **Arquitetura e objetivo de treinamento recomendados**
7. **Tratamento de contextualidade e polissemia**
8. **Avaliação experimental recomendada**
9. **Auditoria do código atual e classificação por arquivo**
10. **Plano de migração em etapas**
11. **Veredito: devemos seguir nessa direção?**

No veredito, responda diretamente:

- Estamos corrigindo um desvio real ou abandonando uma direção válida cedo demais?
- A nova proposta é cientificamente coerente?
- Qual é o menor experimento que deve ser realizado antes de uma grande reescrita?
- Quais suposições nossas estão provavelmente erradas?

Novamente: não altere outros arquivos. Escreva apenas o parecer solicitado em `./tmp/timeformer_displacement_reorientation_review.md`.
