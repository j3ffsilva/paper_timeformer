# Planejamento Paper 2 — Timeformer
## Representações de Trajetória para Traceabilidade Temporal
### Documento autocontido — versão 7 (final para implementação)

---

## 1. Contexto e motivação

### O problema

Palavras mudam de significado ao longo do tempo. *Broadcast* em 1900
referia-se a semear sementes em um campo; hoje descreve uma postagem nas
redes sociais alcançando milhões. Modelos de linguagem baseados em
transformers representam palavras como vetores em um espaço geométrico
compartilhado, mas não garantem que *broadcast@1900* e *broadcast@2020*
ocupem vizinhanças diferentes nesse espaço. Chamamos essa propriedade de
**traceabilidade temporal**: a capacidade de consultar token@tempo como
um objeto diretamente inspecionável, cuja posição geométrica reflete o
significado da palavra naquele período específico.

A maior parte do trabalho computacional sobre mudança semântica (Lexical
Semantic Change Detection, LSCD) treina representações distribucionais em
corpora fatiados por tempo e as compara entre períodos. Isso resume a
mudança como um escalar entre dois modelos treinados separadamente —
broadcast@1900 e broadcast@2020 existem implícitos nesses modelos, não
como objetos diretamente inspecionáveis em um espaço compartilhado.

### O que um trabalho anterior estabeleceu (Paper 1)

Um paper anterior (submetido ao IBERAMIA 2025) diagnosticou o problema
em ambiente controlado, usando um corpus sintético com trajetórias
semânticas plantadas. Quatro achados são relevantes aqui:

- Qualquer condicionamento temporal mais que dobra o deslocamento de
  vizinhança vs. baseline sem período (Δ≈−0.56 vs. −0.221 em drift score)
- Um sinal global de período e uma projeção conjunta token-tempo são
  estatisticamente equivalentes em regime de marcadores confiáveis
  (teste de equivalência TOST, pTOST<0.001); a self-attention absorve a
  diferença de condicionamento de input ao longo das camadas
  (CKA sobe de 0.726 para 0.887 entre input e saída)
- Um pilot noise sweep (3 seeds, slope=+1.07, p=0.002) é consistente com
  condicionamento conjunto retendo vantagem sob marcadores degradados —
  replicação necessária antes de afirmação mais forte
- Mean-prototype falha para palavras com dois sentidos coexistentes:
  colapsa ambos em um centroide que não representa nenhum

**Conclusão do Paper 1:** o gargalo não é a arquitetura de condicionamento
— é o objetivo de treinamento. O Masked Language Modeling (MLM) recompensa
predição de token a partir de contexto local; não recompensa consistência
geométrica ao longo do tempo.

**Estratégia de submissão:** este paper é escrito como autocontido.
Depende do Paper 1 apenas como repositório estendido de experimentos
de diagnóstico, e pode ser submetido independentemente de o Paper 1
estar publicado. As Seções 1–3 reconstroem todo o background necessário.

### A pergunta central deste paper

> Representações explícitas de trajetória melhoram traceabilidade
> temporal além de snapshots temporais?

Todo o resto do paper existe para avaliar essa pergunta.
As contribuições são formuladas como hipóteses a serem testadas,
não como afirmações.

### A distinção conceitual central

Propomos que token@tempo é composto por duas dimensões distintas:

```
token@time(S, t)  =  ( h_s(t),   m_s(t) )
                         ↑            ↑
                    onde S está   como S chegou
                       em t            a t
```

**h_s(t)** — representação de posição semântica: onde o token S está
no espaço semântico no período t, dado o contexto local da sentença.
Produzida por transformer com condicionamento temporal (MLM padrão).
Captura desambiguação contextual mas não trajetória.

**m_s(t)** — representação de estado de trajetória: o estado agregado
da trajetória de S no período t, aprendido a partir da sequência
completa de representações de S ao longo dos períodos. Produzida por
encoder temporal treinado com masked trajectory distillation.

Nota de precisão: m_s(t) deriva de h_s^i(t), que vêm de contextos locais
— então m_s(t) carrega contexto local de forma agregada. A formulação
correta é: m_s(t) não representa uma ocorrência contextual específica;
representa o estado agregado da trajetória do token no período. A
distinção com h_s(t) é entre representar uma instância contextual
(h_s) e representar o estado da trajetória (m_s), não entre ter e não
ter informação de contexto.

Esta distinção é a contribuição conceitual central. O encoder temporal,
o objetivo de distillation, e os diagnósticos são mecanismos para
operacionalizar e avaliar essa distinção.

---

## 2. Arquitetura

### 2.1 Visão geral do pipeline

```
Corpus por período
      │
      ▼
┌─────────────────────────────────────┐
│  PASSO 1: Encoder_semântico         │
│  Token-Time Transformer + MLM       │
│  Produz: h_s^i(t) por ocorrência   │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Agregador por período              │
│  Entrada: {h_s^1(t),...,h_s^n(t)} │
│  Etapa interna: {u_s^i(t)}          │
│  Saída: R_s(t)                      │
│  (variante ablada — ver 2.3)        │
└──────────────┬──────────────────────┘
               │  sem extrapolação:
               │  trajetória só existe
               │  onde S existe
               ▼
┌─────────────────────────────────────┐
│  Sequência de representações        │
│  Seq_s = (R_s(t_a),...,R_s(t_b))  │
│  apenas períodos com ocorrência     │
│  + interpolação interna             │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  PASSO 2: Masked Trajectory         │
│  Distillation                       │
│  Teacher (congelado) → alvo fixo   │
│  Student (treinado) → m_s(t)        │
└──────────────┬──────────────────────┘
               │
               ▼
        token@time(S, t)
        = concat(h_s(t), m_s(t))
```

### 2.2 Passo 1: Encoder_semântico

Idêntico ao Token-Time do Paper 1:

```
z_i    = W[e(w_i); τ(t)] + p_i
h_s(t) = TransformerEncoder(z_1,...,z_n)[posição_sujeito]
```

onde e(w_i) é o embedding do token, τ(t) é um vetor de período aprendido
a partir de features sinusoidais de t/T, p_i é o encoding posicional,
e W ∈ R^{d×2d} é uma projeção linear conjunta.

Treinado com MLM padrão sobre o corpus por período. Produz h_s(t) ∈ R^d
para cada ocorrência de S no período t. Congelado durante o Passo 2 na
configuração Independente (ver ablação 3.3).

### 2.3 Agregador por período

**Por que não usar centroide diretamente:**
O Paper 1 mostrou que mean-prototype falha para palavras com sentidos
coexistentes — a média de ocorrências de dois sentidos produz um centroide
no meio, descartando a estrutura bimodal. Se o agregador colapsar a
distribuição de sentidos antes do encoder temporal, m_s(t) não terá
material para capturar multimodalidade.

O agregador é um **mecanismo necessário** para que a tese principal
(trajetória melhora traceabilidade) seja testável sem o gargalo do
centroide — não é contribuição independente do paper.

**Variantes abladas:**

```
Mean pooling:        R_s(t) = (1/n) Σ_i h_s^i(t)
                     baseline; equivalente ao centroide do Paper 1
                     não possui etapa intermediária u_s^i

Attention pooling:   α_i = softmax(w^T h_s^i(t))
                     R_s(t) = Σ_i α_i h_s^i(t)
                     pesos aprendidos; mais expressivo que mean

Set Transformer:     {u_s^i(t)} = SetTransformer({h_s^1,...,h_s^n})
                     R_s(t) = pooling({u_s^i(t)})
                     permutation-invariant; expõe embeddings
                     por-ocorrência u_s^i antes do pooling final
```

Nota de design sobre o Set Transformer: ele produz embeddings
contextualizados por-ocorrência {u_s^i(t)} (cada ocorrência informada
pelas outras via atenção sobre o conjunto), e só então aplica pooling
para obter o vetor agregado R_s(t). A etapa intermediária {u_s^i(t)}
é necessária para o diagnóstico D6 (ver Seção 5).

Nota de linguagem: dizemos "capacidade arquitetural para representar
estrutura multimodal", não "preserva multimodalidade por design".
Permutation-invariance não implica preservação de modos — implica
apenas independência da ordem das ocorrências. Se o Set Transformer
de fato preserva bimodalidade é questão empírica, avaliada em D6.

Nota de robustez: Set Transformer pode sofrer instabilidade com conjuntos
pequenos de ocorrências por período (palavras raras no corpus natural).
Nesse regime, attention pooling pode superá-lo. A ablação dos três
agregadores captura esse cenário se ocorrer.

### 2.4 Construção da sequência de representações

**Princípio: a trajetória de S só existe onde S existe.**

Não se extrapola além dos períodos com ocorrência. Palavras com menos
de 3 períodos cobertos são excluídas das palavras-alvo.

**Interpolação interna (períodos intermediários ausentes):**
Se S não aparece em t_k mas aparece em t_{k-1} e t_{k+1}:

```
R_s(t_k) = (1−α)·R_s(t_{k-1}) + α·R_s(t_{k+1})
α = (t_k − t_{k-1}) / (t_{k+1} − t_{k-1})
```

Interpolação linear é o prior mais conservador: assume mudança gradual
na ausência de evidência. Mudanças abruptas que coincidem com períodos
sem ocorrência serão suavizadas. Esta é uma limitação do corpus, não
da arquitetura: a capacidade de detectar uma mudança abrupta é limitada
pela resolução temporal dos dados. Documentado explicitamente no paper.

**Sem extrapolação nas extremidades.**
Extrapolar palavras que emergem ou desaparecem criaria trajetórias
sintéticas que o encoder temporal tentaria modelar como reais. Por
exemplo: extrapolar "internet" para décadas anteriores a 1980 com base
na trajetória dos anos 90–2000 produziria deslocamentos artificiais sem
correspondência semântica. A trajetória começa no primeiro período com
ocorrência e termina no último.

**A sequência resultante:**
```
Seq_s = (R_s(t_a), R_s(t_a+1), ..., R_s(t_b))
```
onde t_a e t_b são o primeiro e último período com ocorrência de S,
com interpolação nos intermediários ausentes.

**Por que sequência de representações e não deltas:**
Uma abordagem considerada anteriormente usaria deltas
δ_s(t) = R_s(t) − R_s(t−1) como entrada para o encoder temporal. Isso
é bem definido para mean pooling (subtração de centroides), mas
problemático para Set Transformer: a subtração vetorial de duas
representações de conjuntos no espaço latente do Set Transformer não
mapeia de forma garantida para o "deslocamento" desses conjuntos. Redes
que processam conjuntos codificam estatísticas de ordem superior de
formas potencialmente não-comparáveis entre períodos, tornando o delta
semanticamente ambíguo.

A solução é fornecer a sequência de representações diretamente ao encoder
temporal e deixá-lo aprender o que constitui "mudança" — sem assumir que
a subtração euclidiana captura isso. Isso é mais geral, mais defensável,
e remove um viés indutivo (a linearidade da subtração).

### 2.5 Passo 2: Masked Trajectory Distillation

**Configuração principal: auto-supervisionada, sem rótulos externos.**

A configuração principal do teacher é treinada apenas com reconstrução
mascarada e uma regularização contra colapso para identidade — sem
qualquer sinal de rótulo externo. Isso mantém o método genuinamente
self-supervised: m_s(t) é aprendido da estrutura interna da trajetória,
não de anotações de mudança semântica.

```
Objetivo principal do teacher (self-supervised):
L_teacher = L_recon + β · L_anti-identidade

L_recon:          reconstrução da sequência de entrada
L_anti-identidade: regularização que impede o teacher de aprender
                   a função identidade ou uma suavização trivial
```

**Regularização anti-identidade:**
Um autoencoder sobre sequências pode aprender identidade (se capacidade
alta) ou suavização (se houver gargalo). Para evitar isso sem rótulos
externos, penaliza-se diretamente a similaridade entre a representação
produzida e a entrada, e força-se ocupação do espaço (estilo VICReg):

```
L_anti-identidade = max(0, CKA(M_s, Seq_s) − τ_cka)   (penaliza cópia)
                  + termo de variância sobre as dimensões de M_s
                    (impede colapso para subespaço degenerado)
```

onde τ_cka é um limiar (ex: 0.7) acima do qual o teacher está copiando
demais a entrada.

**Supervisão fraca como ablação de teto (NÃO configuração principal):**
Onde sinais externos de mudança estão disponíveis, podemos adicionar
um termo L_sup como ablação para medir quanto desempenho adicional seria
possível com supervisão. Isso é explicitamente um **upper bound
supervisionado**, não a configuração que o paper defende.

```
Ablação de teto:  L_teacher = L_recon + β·L_anti-id + λ·L_sup
```

Fontes de L_sup, usadas apenas em ablação:
- Sintético: o ground truth P(N1|s,t) está disponível. L_sup é uma
  ranking loss — para t com P(N1|s,t) alta, M_s(t) deve estar mais
  próximo do protótipo de N1 do que do de N2. (Formulação por ranking
  evita calcular similaridades brutas e é mais estável.)
- COHA: anotações de mudança do SemEval-2020 para palavras FORA do
  conjunto de avaliação.

**Por que L_sup não é a configuração principal:**
Usar anotações do SemEval como supervisão e depois avaliar no SemEval
seria circular mesmo com separação de palavras — o que se aprenderia é
a noção de mudança que o SemEval codifica, e avaliar nessa mesma noção
inflaria o resultado. Mantendo L_sup como ablação de teto, o paper não
depende conceitualmente de rótulos externos para aprender m_s(t), e a
ablação ainda informa a margem entre auto-supervisão e supervisão.

**Verificação de Sanidade 0 — qualidade do teacher:**
Antes de congelar o teacher, verificar (na configuração principal,
self-supervised):
- CKA(M_s, Seq_s) << 1 (teacher não aprendeu identidade)
- Probe linear de P(N1|s,t) sobre M_s(t) no sintético > probe sobre
  R_s(t) diretamente (teacher extraiu informação de trajetória)

Se o teacher self-supervised falhar nessas verificações, o Passo 2
não tem fundamento. Esta verificação é anterior a todo o resto.

**Student (treinado com trajetória mascarada):**
O student recebe a sequência com um período mascarado e aprende a
reconstruir a representação do teacher naquele período:

```
Entrada: (R_s(t_a),...,R_s(t_{k-1}), [MASK], R_s(t_{k+1}),...,R_s(t_b))
Alvo:    M_s^teacher(t_k)    (teacher congelado — alvo fixo)
Loss:    L_student = ||M_s^student(t_k) − M_s^teacher(t_k)||^2_2
```

O student usa contexto bidirecional: vê toda a trajetória exceto t_k.
Porque o teacher está congelado, não há loop de feedback — o alvo é
fixo e externo ao que está sendo otimizado. Estruturalmente diferente
de data2vec (onde teacher e student se atualizam simultaneamente, com
risco de colapso trivial).

A saída do student é m_s(t), a representação de estado de trajetória.

**Por que bidirecional:**
Uma mudança abrupta em t_k é indistinguível de uma continuação gradual
se você só vê t_{k-1}. O contexto futuro (t_{k+1},...) é necessário para
identificar rupturas. A escolha bidirecional é motivada pelo requisito
explícito de capturar mudanças abruptas (classe Abrupt).

**Por que masked distillation e não next-step prediction ou loss de
trajetória local:**
Next-step prediction pode ser satisfeito por extrapolação local — não
requer internalizar a forma global da trajetória. Uma loss de deslocamento
pode ser satisfeita por movimento global uniforme. A masked distillation
não pode: o student precisa reconstruir qualquer ponto a partir de
qualquer subconjunto do resto, o que requer que m_s(t) dependa da
estrutura global da trajetória.

**Encoder_temporal (variável experimental):**
Teacher e student usam a mesma família de arquiteturas para o encoder
temporal. A escolha é variável experimental (ver Seção 3.1).

### 2.6 Representação final

```
token@time(S, t) = concat(h_s(t), m_s(t))  ∈ R^{d + d_t}
```

Para traceabilidade (onde S estava em t, como chegou lá?), m_s(t) é a
dimensão relevante. Para desambiguação contextual (qual sentido de S
nesta sentença?), h_s(t) é suficiente. O par completo é avaliado em
todas as tarefas para medir a contribuição incremental de cada componente.

---

## 3. Ablações

### 3.1 Encoder_temporal: família de variantes

| Variante | Arquitetura | Causal? | Captura abruptas? |
|---|---|---|---|
| A: Transformer bidirecional | Atenção completa sobre períodos | Não | Sim |
| B: Transformer causal | Atenção mascarada sobre períodos | Sim | Parcialmente |
| C: Projeção linear | W·R_s(t) + b, sem memória | — | Não |

Predição registrada:
- A > B > C em classe Abrupt (D5a — masked reconstruction)
- A ≈ B > C em classe Drift (D2)
- Se A ≈ C: o encoder temporal não contribui além da representação bruta

### 3.2 Agregador por período

| Variante | Capacidade para multimodalidade | Complexidade |
|---|---|---|
| Set Transformer | Alta (capacidade arquitetural) | Alta |
| Attention pooling | Média | Média |
| Mean pooling (centroide) | Nenhuma | Mínima |

Predição registrada:
- Set Transformer > mean pooling em D6 (bimodalidade) para Bifurcating
- Set Transformer ≈ mean pooling em Stable e Drift
- Se Set Transformer ≈ mean pooling em Bifurcating: o centroide não é
  o gargalo — resultado que revisa a conclusão do Paper 1
- Possibilidade contra-intuitiva: attention pooling pode superar Set
  Transformer em palavras com poucas ocorrências por período

### 3.3 Interação entre encoders

| Configuração | Encoder_semântico no Passo 2 | Interpretação |
|---|---|---|
| Independente | Congelado | Passo 2 extrai de representações fixas |
| Fine-tuning leve | lr = 0.1 × lr_passo1 | Encoder adapta para trajetória |
| Simultâneo | Co-treinado desde início | Objetivos co-otimizados |

Se Independente ≈ Simultâneo: h_s e m_s são naturalmente ortogonais;
pipeline sequencial é suficiente e mais interpretável.
Se Independente << Simultâneo: o encoder precisa co-adaptar; pipeline
sequencial perde informação e co-treino é necessário.

### 3.4 Objetivo do teacher

| Configuração | Composição | Papel |
|---|---|---|
| Principal | L_recon + β·L_anti-identidade | self-supervised, defendida |
| Ablação de teto | + λ·L_sup (rótulos externos) | upper bound supervisionado |
| Sem regularização | L_recon apenas | baseline de colapso (controle) |

Mede: (a) a regularização anti-identidade é suficiente para evitar
colapso sem rótulos? (b) quanta margem existe entre auto-supervisão
e supervisão de teto?

### 3.5 Tamanho de d_t

Varredura: d_t ∈ {8, 16, 32, 64} com d = 64 fixo.
Produz curva traceabilidade vs. custo de representação.
Predição: d_t ótimo menor no sintético (trajetória simples) do que
no COHA (trajetórias complexas e ruidosas).

---

## 4. Benchmark sintético

### 4.1 Setup

Corpus SVO (sujeito-verbo-objeto) artificial com vocabulário controlado.
Duas vizinhanças semânticas N1 e N2, cada uma com 4 verbos e 4 objetos.
40 sujeitos, 10 períodos (t0–t9). Ruído: 25% dos marcadores locais
(verbo, objeto) sorteados da vizinhança oposta, de modo que o modelo
não pode inferir o período só pelos co-ocorrentes. A trajetória de cada
sujeito é definida por P(N1|s,t): a probabilidade de aparecer em sentença
da vizinhança N1 no período t.

### 4.2 Classes de trajetória

**Do Paper 1 (mantidas para comparabilidade):**
- **Stable:** P(N1|s,t) ≈ constante. Sem mudança de vizinhança.
- **Drift:** P(N1|s,t) decresce monotonicamente de ≈1 em t0 para ≈0
  em t9. Mudança gradual.
- **Bifurcating:** P(N1|s,t) decresce de ≈1 para ≈0.5. Dois sentidos
  coexistentes nos períodos tardios. Expõe o failure mode do centroide.

**Nova neste paper:**
- **Abrupt:** P(N1|s,t) = 1 até t_k, depois 0 (step function).
  t_k sorteado uniformemente em {t3,...,t7} por sujeito. Mudança em um
  único período — teste discriminativo para a arquitetura bidirecional.

10 sujeitos por classe, 40 no total.

### 4.3 Splits de avaliação

- **Standard (75%):** marcadores 75% fiéis ao período. Condição principal.
- **Ambiguous (50%):** marcadores 50% fiéis — sem sinal confiável nos
  co-ocorrentes; só sujeito e período restam. Replica e expande o split
  que expôs vantagem do Token-Time no Paper 1.
- **Continuation (t8–t9):** períodos excluídos do treinamento MLM.

---

## 5. Diagnósticos de avaliação

Cada diagnóstico roda sobre h_s(t), m_s(t) e token@time separadamente,
para atribuir ganhos a cada componente.

### D1: Probe linear de contexto
Classificador linear para prever o contexto plantado (N1 vs. N2).
Diagnóstico auxiliar — recoverabilidade ≠ traceabilidade geométrica
(demonstrado no Paper 1).

### D2: Context drift score
Recupera os k=10 vizinhos mais próximos por similaridade coseno;
registra a proporção atribuída a N1. Drift score: Δ = valor_em_t9 − valor_em_t0.
Métrica corrigida (contínua, sem discretização do kNN):
```
score(t) = sim(rep(S,t), centroide_N1) − sim(rep(S,t), centroide_N2)
```

### D3: Contrastive sign-flip rate
Mesma sentença com dois rótulos de período; taxa em que trocar o período
inverte a preferência de vizinhança. Baseline sem período produz 0.000.

### D4: Continuation diagnostic
Probe linear em t8–t9, períodos excluídos do treinamento MLM.
Expectativa: m_s(t) generaliza melhor que h_s(t) para períodos não vistos.

### D5a: Masked trajectory reconstruction
Avalia o student na sua própria tarefa: reconstruir M_s(t_k) dado
contexto bidirecional da trajetória.

Este diagnóstico usa contexto bidirecional por design — é avaliação
in-distribution do objetivo de distillation, NÃO previsão de futuro.
Não deve ser confundido com extrapolação causal.

Predição: variante A > B > C. Se A ≈ C: encoder temporal não acrescenta.

### D6: Bimodalidade em Bifurcating
Mede se as representações por-ocorrência formam dois modos (sentidos
coexistentes) ou colapsam.

Definição precisa (corrige inconsistência de versões anteriores):
- Para Set Transformer: silhouette sobre os embeddings por-ocorrência
  {u_s^i(t)} produzidos pela etapa intermediária (antes do pooling).
  Testa se o agregador mantém os modos separados antes de comprimir.
- Para mean pooling (sem etapa intermediária): silhouette sobre as
  ocorrências de entrada {h_s^i(t)}, servindo de baseline.

```
bimodalidade(S, t) = silhouette({u_s^i(t)}, labels={N1_ctx, N2_ctx})
                     (ou {h_s^i(t)} para mean pooling)
```

Alto silhouette = dois modos distintos preservados.
Baixo/negativo = colapso. Predição: Set Transformer > mean pooling
para Bifurcating.

### Varredura de fidelidade
Replica e expande o pilot do Paper 1 (3 seeds → 31 seeds × 7 níveis).
Fidelidades: {0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45}.
Modelos: h_s sozinho, m_s sozinho, token@time.
Predição: gap token@time vs. h_s em D2 cresce monotonicamente com
degradação. Teste: regressão do gap sobre nível + Jonckheere-Terpstra.

### Análise qualitativa de trajetórias (salvaguarda)
Para 4–5 palavras com mudança semântica conhecida (ex: gay, broadcast,
mouse no COHA; sujeitos de cada classe no sintético), plotar inspeção
visual (PCA / t-SNE) das trajetórias de h_s(t) vs. m_s(t).

Justificativa: métricas agregadas de ranking (Spearman no SemEval)
podem ser insensíveis a melhorias arquiteturais reais quando a tarefa
é ruidosa ou o conjunto de palavras é pequeno. Se m_s(t) mostrar
trajetórias mais suaves e semanticamente coerentes que h_s(t), isso é
um resultado qualitativo forte que sustenta o paper mesmo se o ganho
quantitativo for marginal. É uma salvaguarda contra o risco de a métrica
de alto nível não capturar o ganho.

---

## 6. Corpus natural: COHA

### Setup
Corpus of Historical American English: cobertura decadal 1810–2000
(19 décadas), balanceado por gênero. Mesmo corpus usado por Hamilton
et al. — comparação direta com a literatura LSCD.

Palavras-alvo: subconjunto do SemEval-2020 Task 1 (inglês) com ≥ 3
décadas de ocorrência (critério mínimo para trajetória).

**Esparsidade e interpolação:**
- Documentar a distribuição de cobertura decadal das palavras-alvo
- Sem extrapolação além dos extremos de ocorrência
- Interpolação linear apenas para décadas intermediárias ausentes
- Analisar impacto da interpolação comparando palavras com cobertura
  completa vs. parcial

### Diagnósticos no COHA
D2, D3, D4, D6 sobre h_s, m_s e token@time.
Análise qualitativa de trajetórias para palavras conhecidas.

Ortogonalidade: CKA(h_s(t), m_s(t)) para cada período.
Se alto: as duas representações são redundantes — token@time desperdiça
parâmetros. Se baixo: são complementares — o par é justificado.
Predição: CKA(h_s, m_s) < CKA(h_s(t), h_s(t')) para t ≠ t' — h_s e m_s
são mais ortogonais entre si do que h_s é entre períodos diferentes.

---

## 7. Avaliação extrínseca: SemEval-2020 Task 1

Ranking de palavras por magnitude de mudança semântica entre dois
períodos. Correlação de Spearman com o ground truth anotado humano.

Métrica de mudança sobre m_s(t):
```
mudança(S) = distância média entre M_s(t_inicial) e M_s(t_final)
```

Comparação com Hamilton et al. (word2vec diacrônico + alinhamento)
como baseline da literatura LSCD.

Hipótese (não afirmação): m_s(t) melhora o ranking sobre h_s(t) sozinho
porque captura magnitude de deslocamento ao longo da trajetória.

Nota: a análise qualitativa de trajetórias (Seção 5) é a salvaguarda
caso o Spearman seja insensível a um ganho arquitetural real.

---

## 8. Estrutura do paper

```
1. Introduction
   - token@time: posição semântica + estado de trajetória
   - Pergunta central: trajetória melhora traceabilidade?
   - Background do Paper 1; estratégia autocontida
   - Contribuições (como hipóteses)

2. Background
   - LSCD diacrônico: Hamilton et al. como motivação e baseline
   - Temporal LMs: condicionamento sem representação de trajetória
   - Masked objectives, knowledge distillation, set encoders
   - Paper 1 (IBERAMIA) como trabalho anterior direto

3. Posição Semântica vs. Estado de Trajetória
   - Definição formal de traceabilidade
   - Por que h_s(t) sozinho não é suficiente
   - Failure mode do centroide: motivação para agregador expressivo
   (reconstrói o background do Paper 1 — autocontido)

4. Arquitetura
   4.1 Encoder_semântico: Token-Time + MLM
   4.2 Agregador por período: variantes e etapa por-ocorrência
   4.3 Sequência de representações: construção e interpolação
   4.4 Encoder_temporal: família de variantes (A, B, C)
   4.5 Masked Trajectory Distillation: teacher self-supervised
       (regularização anti-identidade), student bidirecional, alvo fixo;
       L_sup como ablação de teto
   4.6 Representação final token@time = (h_s, m_s)

5. Suíte de Diagnósticos
   5.1 Benchmark sintético: classes e splits
   5.2 D1–D6, varredura de fidelidade, análise qualitativa
   5.3 COHA e SemEval-2020

6. Resultados: Benchmark Sintético
   6.1 Verificação de qualidade do teacher (self-supervised vs. teto)
   6.2 Ablação do agregador (D6 bimodalidade)
   6.3 Ablação do Encoder_temporal (D5a e D2)
   6.4 Ablação da interação entre encoders
   6.5 Ablação de d_t
   6.6 Varredura de fidelidade completa
   6.7 Trajetórias qualitativas por classe

7. Resultados: COHA e SemEval-2020
   7.1 D2, D3, D4, D6 no COHA
   7.2 CKA(h_s, m_s): ortogonalidade empírica
   7.3 Trajetórias qualitativas (gay, broadcast, mouse)
   7.4 Ranking SemEval-2020 vs. Hamilton et al.
   7.5 Análise de esparsidade e impacto da interpolação

8. Discussão
   8.1 Quando agregador expressivo importa vs. centroide suficiente
   8.2 Quando bidirecional importa vs. causal suficiente (D5a)
   8.3 Ortogonalidade de h_s e m_s: implicações para co-treinamento
   8.4 Auto-supervisão vs. teto supervisionado: margem do L_sup
   8.5 Causalidade retroativa de m_s(t): nuance ontológica
   8.6 Limitações: esparsidade, resolução temporal, custo, vocabulário
   8.7 Previsão causal de trajetória futura: direção não avaliada

9. Timeformer como Framework: Agenda
   - Previsão causal (variante B como candidato natural)
   - Versões light do encoder temporal
   - Extensão multilíngue
   - Aplicações: QA temporal, análise histórica, detecção de viés

10. Conclusion
```

### Contribuições (formuladas como hipóteses)

1. Introduzimos a distinção entre h_s(t) (posição semântica) e m_s(t)
   (estado de trajetória), e propomos token@time = (h_s, m_s) como
   objeto completo para consultas temporais. Avaliamos se essa distinção
   melhora traceabilidade.

2. Propomos masked trajectory distillation, um objetivo self-supervised:
   um teacher com regularização anti-identidade aprende representações
   de trajetória sem rótulos externos; um student bidirecional reconstrói
   períodos mascarados. Avaliamos se esse objetivo produz representações
   de trajetória com melhor traceabilidade do que objetivos locais.

3. Identificamos que a subtração vetorial sobre representações de conjunto
   não é semanticamente bem definida como "deslocamento", e propomos
   fornecer a sequência de representações diretamente ao encoder temporal,
   deixando o modelo aprender o que constitui mudança.

4. Introduzimos a classe Abrupt no benchmark sintético e D5a (masked
   trajectory reconstruction) como avaliação discriminativa para mudanças
   não-monotônicas. Avaliamos se o encoder bidirecional supera variantes
   causais e lineares nessa condição.

5. Avaliamos se token@time melhora traceabilidade em COHA e SemEval-2020,
   e se h_s e m_s capturam informação ortogonal (CKA).

---

## 9. Cronograma

### Semana 1: Verificação de Sanidade 0 — qualidade do teacher

**Implementar (mínimo viável):**
- Sequência de representações com mean pooling
- Encoder temporal variante C (projeção linear)
- Teacher self-supervised: L_recon + regularização anti-identidade
  (sweep de β; comparar com L_recon puro como controle de colapso)

**Verificação de Sanidade 0:**
- CKA(M_s, Seq_s) << 1? (teacher não é identidade, sem rótulos)
- Probe linear de P(N1|s,t) sobre M_s(t) > probe sobre R_s(t)?
  (teacher extraiu informação de trajetória)

Falha aqui invalida o Passo 2 inteiro. Uma semana para diagnosticar.
Não avançar sem resolução.

### Semana 2: Verificação de Sanidade 1 — sinal nas representações

**Implementar:**
- Student bidirecional (variante A) sobre sequência de representações
- Classe Abrupt no benchmark sintético
- D5a (masked reconstruction) e D6 (bimodalidade com mean pooling)

**Verificação de Sanidade 1:**
- Variante C melhora D2 sobre h_s(t) em classe Drift?
  (sequência de representações carrega sinal temporal)
- Em classe Stable, variante C é estatisticamente indistinguível de
  h_s(t)? (o encoder não inventa trajetória onde não existe — proteção
  contra o global displacement observado no Paper 1)
- D5a: variante A > variante C na classe Abrupt?
  (bidirecional captura rupturas melhor que linear)

Falha em Drift: sequência sem agregador expressivo não carrega sinal —
rever encoder temporal. Falha em Stable: o encoder injeta ruído —
rever regularização. Falha em Abrupt: bidirecional sem vantagem —
rever design da classe ou objetivo.

### Semana 3: Set Transformer e Verificação de Sanidade 2

**Implementar:**
- Agregador Set Transformer com etapa por-ocorrência {u_s^i(t)}
- D6 com Set Transformer (silhouette sobre {u_s^i(t)})

**Verificação de Sanidade 2:**
- Set Transformer > mean pooling em D6 para Bifurcating?

Falha: o centroide não é o gargalo — a tese principal (trajetória
melhora traceabilidade) pode sobreviver com mean pooling, mas a
hipótese sobre multimodalidade não sobrevive.

### Semana 4: Ablações completas no sintético

Somente se semanas 1–3 passaram.
- Ablação agregador × encoder temporal (A,B,C) × d_t
- Ablação interação entre encoders
- Ablação do objetivo do teacher (principal vs. teto vs. sem regularização)
- Varredura de fidelidade (31 seeds × 7 níveis)
- Trajetórias qualitativas por classe

**Ponto de decisão (fim da semana 4):**
Escolher configuração principal para corpus natural. Não levar múltiplas
configurações para o COHA — decidir aqui.

### Semanas 5–8: COHA
- Preprocessamento por década; verificar cobertura; excluir < 3 décadas
- Construção de sequências com configuração escolhida na semana 4
- D2, D3, D4, D6; CKA(h_s, m_s); trajetórias qualitativas
- Análise de esparsidade e impacto da interpolação

### Semana 9: SemEval-2020
- Ranking de mudança semântica; comparação com Hamilton et al.

### Semanas 10–13: Escrita
- Rascunho completo na semana 10; revisão 11–12; polimento 13

---

## 10. Pré-registro

Após semanas 1–3 (verificações de sanidade) e antes de rodar COHA,
registrar em commit datado ou OSF:

1. Set Transformer > mean pooling em D6 (bimodalidade) no COHA para
   palavras com dois sentidos coexistentes
2. token@time > h_s(t) em D2 no COHA; ganho concentrado em m_s(t)
3. Gap token@time vs. h_s(t) cresce monotonicamente com degradação na
   varredura de fidelidade
4. Encoder temporal variante A > variante B em D5a para classe Abrupt
5. CKA(h_s(t), m_s(t)) < CKA(h_s(t), h_s(t')) para t ≠ t'

---

## 11. Sobre o que este paper não avalia

**Previsão causal de trajetória futura:** este paper avalia reconstrução
mascarada bidirecional (D5a), avaliação in-distribution do objetivo de
distillation. Previsão de futuro em contexto causal (onde t_{k+1} não
está disponível) é caso de uso diferente, fora do escopo. Variante B
(encoder causal) é o candidato natural para trabalho futuro.

**Nuance de causalidade retroativa (a discutir na Seção 8.5):**
porque o student é bidirecional, m_s(t) em um período inicial depende
do contexto de períodos posteriores — m_s@1900 computado com dados até
1950 difere de m_s@1900 computado com dados até 2000. Isto é uma
propriedade de medição retrospectiva (descrever uma trajetória já
observada), não de previsão. Não invalida o método, mas é uma nuance
ontológica que o paper reconhece: o "estado de trajetória" em um ponto
é definido em relação à trajetória completa observada, não causalmente.

**Versões light:** a prioridade é estabelecer o teto da abordagem.
A variante C (projeção linear) já está no paper como lower bound.
Versões eficientes são o paper seguinte natural se o teto for alto.

**Corpus não-inglês:** COHA e SemEval-2020 (inglês) são o escopo.
Extensão multilíngue é direção futura.