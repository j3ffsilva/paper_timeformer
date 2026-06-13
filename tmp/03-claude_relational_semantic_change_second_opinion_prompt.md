# Pedido de segunda opinião independente: mudança semântica relacional entre checkpoints temporais

Você deve realizar uma nova revisão técnica e científica independente do projeto:

`/Users/jeff/Documents/trabalhos/papers/paper-timeformers`

Este pedido substitui, para fins desta revisão, as formulações anteriores
baseadas em:

- Transformer congelado após `t0`;
- módulo externo aprendido `delta(w,t)`;
- dimensão temporal concatenada;
- alinhamento obrigatório entre espaços;
- teacher/student para aprender trajetória.

Essas formulações foram discutidas anteriormente, mas não descrevem com
precisão a proposta que queremos investigar agora.

Não altere código, documentação ou resultados existentes. Analise a proposta,
identifique riscos à validade, proponha alternativas e escreva seu parecer em:

`./tmp/07-timeformer_relational_change_second_opinion.md`

O parecer deve ser autocontido. Não suponha que o leitor tenha acompanhado as
discussões anteriores.

---

## 1. Ideia central que queremos investigar

Queremos treinar **um único Transformer de maneira contínua e cronológica**,
usando os textos de cada período em sequência.

```text
theta_0 = Transformer treinado com textos de t0
theta_1 = theta_0 continuado com textos de t1
theta_2 = theta_1 continuado com textos de t2
...
theta_n = theta_(n-1) continuado com textos de tn
```

Após cada período, salvamos um checkpoint e extraímos representações
semânticas das palavras naquele estado do modelo.

Não pretendemos necessariamente comparar coordenadas absolutas entre
checkpoints:

```text
vetor_theta1(w) - vetor_theta0(w)
```

Sabemos que o sistema de coordenadas interno pode rotacionar, refletir,
redimensionar ou reorganizar-se durante o treinamento contínuo.

Nosso interesse principal é medir **como as relações semânticas de uma palavra
com as demais palavras mudam entre checkpoints**.

Se as relações e distâncias relevantes permanecem iguais, consideramos que o
deslocamento semântico é zero, mesmo que as coordenadas absolutas tenham
mudado.

Em resumo:

> mudança semântica é mudança da posição relacional da palavra, não
> necessariamente mudança de suas coordenadas absolutas.

---

## 2. Exemplo trivial: rotação sem mudança semântica

Considere um espaço bidimensional em `t0`:

```text
gato@t0     = [1, 0]
cachorro@t0 = [0, 1]
rato@t0     = [-1, 0]
```

Após continuar o treinamento em `t1`, o espaço inteiro rotaciona 90 graus:

```text
gato@t1     = [0, 1]
cachorro@t1 = [-1, 0]
rato@t1     = [0, -1]
```

O deslocamento absoluto de `gato` seria:

```text
gato@t1 - gato@t0 = [-1, 1]
```

Mas nenhuma relação semântica mudou. As distâncias e similaridades relativas
entre `gato`, `cachorro` e `rato` foram preservadas.

Portanto, queremos que nossa medida produza:

```text
mudanca_semantica(gato, t0 -> t1) = 0
```

sem exigir que os dois espaços sejam previamente alinhados.

---

## 3. Exemplo trivial: mudança real de vizinhança

Considere as vizinhanças de uma palavra hipotética `gay`.

Em `t0`:

```text
sim(gay, happy)     = 0.95
sim(gay, joyful)    = 0.90
sim(gay, cheerful)  = 0.85
sim(gay, identity)  = 0.20
sim(gay, rights)    = 0.15
sim(gay, community) = 0.10
```

Em `t1`:

```text
sim(gay, happy)     = 0.25
sim(gay, joyful)    = 0.20
sim(gay, cheerful)  = 0.15
sim(gay, identity)  = 0.94
sim(gay, rights)    = 0.91
sim(gay, community) = 0.87
```

Mesmo que não saibamos ou não usemos as coordenadas absolutas de `gay`, a
mudança semântica é clara:

```text
afastamento: happy, joyful, cheerful
aproximacao: identity, rights, community
```

Queremos representar:

- a **magnitude** dessa mudança relacional;
- a **direção relacional**, isto é, de quais conceitos a palavra se afastou e
  de quais se aproximou;
- a **trajetória relacional**, observando essa mudança ao longo de vários
  períodos.

---

## 4. O que não queremos usar como medida principal

### 4.1 Diferença bruta de coordenadas

Não queremos depender principalmente de:

```text
delta_absoluto(w,t0,t1) = vetor_t1(w) - vetor_t0(w)
```

porque esse valor mistura:

- mudança semântica específica da palavra;
- rotação/reflexão/reorganização global do espaço;
- ruído e deriva causada pela otimização.

### 4.2 Média da distância aos k vizinhos

Também reconhecemos que apenas comparar a distância média aos vizinhos não é
suficiente.

Exemplo:

```text
t0: distancia media 0.1 para {happy, joyful, cheerful}
t1: distancia media 0.1 para {identity, rights, community}
```

A densidade local não mudou, mas a semântica mudou completamente porque os
vizinhos são outros.

### 4.3 Alinhamento como requisito obrigatório

Não afirmamos que alinhamento seja inútil. Ele pode ser um baseline ou uma
ferramenta auxiliar.

Porém, nossa hipótese é que podemos estudar mudança semântica por meio de
relações internas de cada checkpoint, sem precisar projetar todos os
checkpoints em um mesmo sistema de coordenadas.

Solicitamos que você avalie criticamente se essa hipótese é válida.

---

## 5. Representação relacional proposta

Para cada checkpoint `theta_t`, extraímos uma representação palavra-período:

```text
e_t(w)
```

Como o Transformer produz embeddings contextuais por ocorrência, inicialmente
podemos definir:

```text
e_t(w) = agregacao das representacoes contextuais de w no corpus do periodo t
```

A agregação mais simples seria a média. Alternativas podem incluir múltiplos
centroides, distribuições de ocorrências ou representações baseadas em
vizinhança.

Dentro de cada checkpoint, definimos o perfil relacional da palavra:

```text
r_t(w)[v] = sim_t(e_t(w), e_t(v))
```

para palavras `v` de um conjunto de referência comparável entre períodos.

O perfil pode ser:

- denso: similaridade com todas as palavras de referência;
- esparso: apenas os `k` vizinhos mais próximos;
- uma distribuição:

```text
P_t(v | w) = softmax(sim_t(e_t(w), e_t(v)) / temperatura)
```

O deslocamento relacional entre períodos seria:

```text
delta_rel(w, t0, t1) = r_t1(w) - r_t0(w)
```

Suas dimensões não representam coordenadas internas do Transformer. Cada
dimensão representa uma relação com uma palavra ou conceito de referência.

Exemplo:

```text
delta_rel(gay,t0,t1)[happy]    = -0.70
delta_rel(gay,t0,t1)[identity] = +0.74
delta_rel(gay,t0,t1)[rights]   = +0.76
```

Esse vetor descreve diretamente a direção relacional da mudança.

---

## 6. Possíveis medidas de magnitude

Queremos avaliar quais medidas são adequadas, robustas e cientificamente
defensáveis.

### 6.1 Sobreposição de vizinhos

```text
N_k(w,t) = conjunto dos k vizinhos mais próximos de w no checkpoint t

change_jaccard(w,t0,t1)
    = 1 - Jaccard(N_k(w,t0), N_k(w,t1))
```

Vantagem: simples e invariável a rotações.

Riscos conhecidos:

- sensível à escolha de `k`;
- ignora intensidade e ordem;
- pequenas perturbações na fronteira de `k` podem parecer grandes mudanças.

### 6.2 Comparação de rankings

Comparar o ranking das palavras mais relacionadas usando, por exemplo:

- weighted Jaccard;
- rank-biased overlap;
- Kendall tau;
- Spearman;
- medidas top-k ponderadas.

### 6.3 Divergência entre distribuições relacionais

```text
change_js(w,t0,t1)
    = JensenShannon(P_t0(.|w), P_t1(.|w))
```

Vantagem: considera identidade e intensidade das relações.

Questões:

- escolha de temperatura;
- efeito do tamanho do vocabulário;
- comparabilidade quando palavras entram ou saem do vocabulário.

### 6.4 Mudança da geometria local

Comparar diretamente a geometria do subgrafo local de `w`:

- distâncias para vizinhos compartilhados;
- estrutura do grafo k-NN;
- centralidade;
- clustering local;
- métodos de comparação de grafos;
- assinaturas espectrais locais.

Solicitamos avaliação sobre qual família de medidas melhor representa mudança
semântica e quais devem aparecer apenas como ablação.

---

## 7. Direção e trajetória sem coordenadas absolutas

Sem alinhamento, não teremos necessariamente um vetor de direção no espaço
interno do Transformer.

Mas podemos definir direção relacional:

```text
direcao_relacional(w,t0,t1)
    = relacoes que aumentaram - relacoes que diminuiram
```

E trajetória relacional:

```text
r_t0(w) -> r_t1(w) -> r_t2(w) -> ... -> r_tn(w)
```

Podemos estudar:

- velocidade: distância entre perfis consecutivos;
- mudança acumulada: distância entre perfil inicial e final;
- aceleração: variação da velocidade;
- mudança abrupta: pico de distância entre períodos consecutivos;
- deriva gradual: pequenas mudanças consistentes;
- retorno: afastamento e posterior aproximação do perfil inicial;
- estabilidade: baixa variação relacional.

Queremos saber se essa formulação é matematicamente e cientificamente sólida.

---

## 8. Processo de treinamento pretendido

O processo que imaginamos é:

```text
1. Inicializar e treinar o Transformer normalmente em t0.
2. Salvar checkpoint theta_0.
3. Extrair e registrar os perfis relacionais r_t0(w).
4. Continuar o treinamento do mesmo Transformer em t1.
5. Salvar checkpoint theta_1.
6. Extrair e registrar os perfis relacionais r_t1(w).
7. Calcular mudanças relacionais entre t0 e t1.
8. Repetir para t2, ..., tn.
```

Nenhuma informação temporal precisa entrar explicitamente no Transformer.
O modelo aprende normalmente com os textos disponíveis em cada período.

Não queremos necessariamente adicionar ou aprender um módulo temporal.

O objeto temporal é produzido pela análise posterior:

```text
perfil_relacional(w,t)
mudanca_relacional(w,t_i,t_j)
trajetoria_relacional(w)
```

---

## 9. Questões centrais para sua avaliação

Analise criticamente, procurando riscos que ainda não percebemos.

### 9.1 Validade da invariância relacional

1. Perfis de similaridade interna realmente eliminam a necessidade de
   alinhamento?
2. Quais transformações globais preservam esses perfis?
3. Rotações e reflexões são invariantes, mas escalas, deformações não
   isométricas e anisotropia também são?
4. Como distinguir reorganização global do espaço de mudança semântica
   específica?
5. Precisamos normalizar ou calibrar similaridades entre checkpoints?

### 9.2 Treinamento contínuo e causalidade temporal

1. Continuar o treinamento de `theta_(t-1)` em `t` é adequado?
2. Catastrophic forgetting pode produzir falsas mudanças semânticas?
3. A ordem cronológica introduz dependência indesejada dos períodos anteriores?
4. Devemos usar replay de textos anteriores, regularização ou algum mecanismo
   para preservar relações estáveis?
5. Como separar mudança causada pelo corpus de t de mudança causada pelo
   número de passos de otimização?
6. Como controlar desigualdade de tamanho entre corpora dos períodos?

### 9.3 Extração de representações palavra-período

1. Devemos extrair `e_t(w)` usando:
   - apenas os contextos do próprio período t;
   - um conjunto fixo de sentenças-sonda compartilhado entre checkpoints;
   - ambos?
2. Usar contextos diferentes em cada período confunde mudança do modelo com
   mudança da distribuição de contextos?
3. Um conjunto fixo de sondas mede o que queremos ou remove justamente o sinal
   semântico temporal?
4. A média das ocorrências é adequada?
5. Como lidar com polissemia e sentidos coexistentes?

### 9.4 Definição do conjunto de referência

1. O perfil relacional deve usar:
   - todas as palavras disponíveis em cada período;
   - apenas a intersecção de vocabulário entre todos os períodos;
   - palavras-âncora de alta frequência;
   - conceitos ou protótipos;
   - uma combinação?
2. O que fazer quando palavras entram ou saem do vocabulário?
3. Como frequência e qualidade variável dos embeddings afetam o perfil?

### 9.5 Métricas

1. Quais métricas relacionais recomenda como principais?
2. Jaccard top-k, weighted Jaccard, RBO, Jensen-Shannon e comparação de grafos
   medem aspectos diferentes ou são redundantes?
3. Como escolher `k` e temperatura sem otimizar no conjunto de teste?
4. Como estimar incerteza e significância da mudança de uma palavra?
5. Como definir um limiar para dizer que houve mudança real?

### 9.6 Relação com literatura e novidade

1. Essa proposta já corresponde a alguma família conhecida de métodos de
   mudança semântica baseada em vizinhança, segunda ordem ou grafos?
2. Qual seria a contribuição nova e defensável?
3. O treinamento contínuo de Transformer mais análise relacional constitui
   contribuição metodológica suficiente?
4. Devemos posicionar a contribuição como:
   - representação relacional invariável a coordenadas;
   - protocolo causal de checkpoints temporais;
   - análise de trajetórias relacionais;
   - diagnóstico de estabilidade de espaços contextuais;
   - combinação desses elementos?

### 9.7 Benchmark e falsificação

1. O corpus sintético atual é adequado para testar essa proposta?
2. Quais classes atuais são úteis: Stable, Drift, Bifurcating, Abrupt?
3. Precisamos criar classes adicionais?
4. Quais baselines são obrigatórios?
5. Qual é o menor experimento capaz de falsificar a hipótese?
6. Qual resultado indicaria que a análise relacional confunde ruído de treino
   com mudança semântica?

### 9.8 Auditoria do código atual

Inspecione o repositório e classifique:

1. o que pode ser reutilizado diretamente;
2. o que pode ser reutilizado com adaptação;
3. o que deve permanecer apenas como baseline;
4. o que deve ser criado.

Considere especialmente:

- `src/timeformers/corpus.py`
- `src/timeformers/dataset.py`
- `src/timeformers/models.py`
- `src/timeformers/train.py`
- `src/timeformers/representations.py`
- `src/timeformers/metrics.py`
- `src/timeformers/trajectory_metrics.py`
- `src/timeformers/experiment.py`
- scripts de execução e sumarização

Avalie se precisamos criar componentes como:

```text
ContinualPeriodTrainer
CheckpointRepresentationExtractor
RelationalProfileIndex
RelationalChangeMetrics
RelationalTrajectoryAnalyzer
```

---

## 10. Alternativas que você deve comparar

Não avalie apenas a proposta principal. Compare-a com:

### Alternativa A — Modelos independentes por período + alinhamento

```text
theta_t treinado do zero em cada período
espaços alinhados posteriormente
```

### Alternativa B — Treinamento contínuo + alinhamento

```text
theta_t continua de theta_(t-1)
espaços alinhados posteriormente
```

### Alternativa C — Treinamento contínuo + análise relacional sem alinhamento

```text
theta_t continua de theta_(t-1)
mudança medida por perfis de vizinhança
```

### Alternativa D — Transformer único condicionado por tempo

```text
um único modelo recebe informação temporal explicitamente
```

### Alternativa E — Transformer congelado aplicado aos corpora de cada período

```text
um modelo treinado em t0 é congelado
mudança medida pela distribuição de representações contextuais
```

Para cada alternativa, discuta:

- o que ela mede;
- seus vieses;
- sua necessidade de alinhamento;
- interpretabilidade;
- adequação às consultas `similares(w@t)` e à trajetória relacional;
- custo computacional;
- utilidade como baseline.

---

## 11. Formato obrigatório do parecer

Escreva o parecer exclusivamente em:

`./tmp/07-timeformer_relational_change_second_opinion.md`

Não modifique outros arquivos.

Use esta estrutura:

1. **Resumo executivo**
2. **Sua compreensão precisa da proposta**
3. **A proposta mede mudança semântica ou apenas instabilidade?**
4. **Invariâncias e limites matemáticos**
5. **Riscos do treinamento contínuo**
6. **Extração e agregação das representações**
7. **Métricas relacionais recomendadas**
8. **Polissemia e sentidos coexistentes**
9. **Comparação das alternativas A–E**
10. **Benchmark mínimo e critérios de falsificação**
11. **Auditoria do código e plano de implementação**
12. **Relação com literatura e contribuição potencial**
13. **Veredito e próximos passos**

No veredito, responda diretamente:

- A proposta está agora formulada de maneira coerente?
- Perfis relacionais realmente permitem evitar alinhamento?
- Qual é o maior risco à validade?
- Qual é o menor experimento que devemos executar?
- Quais controles são indispensáveis?
- Há uma alternativa mais simples ou mais forte que deveríamos preferir?
