# Planejamento Paper 2 — Timeformer
## Mudança semântica relacional entre checkpoints cronológicos

### Documento autocontido — versão reorientada

---

## 1. Tese central

Este paper deixa de tratar `token@tempo` como uma representação temporal
explícita aprendida pelo modelo. A nova formulação é:

> mudança semântica temporal é mudança nas relações de uma palavra com o
> restante do vocabulário dentro de checkpoints treinados cronologicamente.

O Transformer não recebe identificador de período, embedding temporal ou
condicionamento token-time. Ele é treinado como um Transformer padrão:

```text
theta_0 = treino(D_0)
theta_1 = continua_treino(theta_0, D_1)
...
theta_t = continua_treino(theta_{t-1}, D_t)
```

A mudança é medida depois do treino, comparando perfis relacionais:

```text
r_t(w)[v] = similaridade_t(w, v)
delta_rel(w, a, b) = r_b(w) - r_a(w)
```

Assim, não perguntamos se o vetor absoluto de uma palavra está no mesmo sistema
de coordenadas em todos os períodos. Perguntamos se o círculo relacional da
palavra mudou. Se uma transformação global preserva as relações internas, ela
deve produzir deslocamento semântico relacional zero.

---

## 2. Motivação

O Paper 1 comparou mecanismos de condicionamento temporal, incluindo
Token-Time, Memory-Augmented e Additive. A conclusão experimental foi que esses
mecanismos não se diferenciaram de forma substantiva no cenário estudado.

Isso sugere que inserir tempo no input do Transformer não é, por si só, a
contribuição científica mais promissora. Mais importante é definir **como
medir mudança semântica** sem confundir:

- mudança semântica real;
- deriva de otimização;
- ruído de amostragem;
- transformação global do espaço vetorial;
- pequenas perturbações irrelevantes.

A hipótese atual é que a unidade de análise deve ser relacional:

```text
palavra em t = perfil de relações dessa palavra no checkpoint theta_t
```

Essa formulação preserva a intuição geométrica de deslocamento semântico, mas
evita depender diretamente de coordenadas absolutas de embeddings.

---

## 3. Definição operacional

### 3.1 Treinamento cronológico

O modelo principal é `continual_real`:

```text
theta_0 = Transformer treinado em D_0
theta_t = theta_(t-1) continuado em D_t
```

Nenhum sinal temporal é fornecido ao modelo. A temporalidade entra apenas pela
ordem dos dados.

### 3.2 Consulta pós-Transformer

A representação principal não é o estado oculto direto do sujeito. O experimento
mostrou que consultar diretamente `h_subj` não recupera de forma confiável a
direção semântica plantada.

A consulta principal passa a ser um probe preditivo pós-Transformer:

```text
[CLS] palavra [MASK] [MASK] [SEP]
```

Extraímos a distribuição prevista sobre pares válidos de contexto sintético:

```text
q_t(w) = P_t(contextos | [CLS] w [MASK] [MASK] [SEP])
```

Duas palavras são consideradas semanticamente próximas quando o checkpoint
prevê distribuições de contexto semelhantes para elas:

```text
r_t(w)[v] = 1 - JS(q_t(w), q_t(v)) / log(2)
```

Onde `JS` é a divergência de Jensen-Shannon.

### 3.3 Deslocamento relacional

O vetor de deslocamento de uma palavra entre dois períodos é:

```text
delta_rel(w, a, b) = r_b(w) - r_a(w)
```

Esse vetor não é uma nova representação aprendida. Ele é uma medida derivada
dos checkpoints.

### 3.4 Trajetória

A trajetória também não é aprendida por teacher/student. Ela é derivada da
sequência:

```text
r_0(w), r_1(w), ..., r_T(w)
```

ou, equivalentemente, da sequência de deslocamentos:

```text
delta_rel(w, 0, 1), delta_rel(w, 1, 2), ..., delta_rel(w, T-1, T)
```

Depois disso, algoritmos de análise podem medir:

- magnitude final;
- caminho total;
- direção;
- recuperação;
- reversão;
- oscilação;
- concentração de evento;
- deriva pré- e pós-evento.

---

## 4. Controles experimentais

### 4.1 Regime principal

`continual_real`:

```text
D_0 -> D_1 -> ... -> D_t
```

Treino cronológico real. Este é o objeto científico principal.

### 4.2 Nulo ressampleado

`resampled_null`:

```text
D'_0, D'_1, ..., D'_t
```

Cada período recebe novos textos, mas a distribuição semântica plantada fica
constante. Este é o controle principal contra falsos positivos semânticos.

Pergunta que responde:

> quanta mudança relacional aparece apenas porque vemos novas amostras sem
> mudança semântica?

### 4.3 Placebo repetido

`continual_placebo`:

```text
D_0 -> D_0 -> ... -> D_0
```

O modelo continua treinando repetidamente no mesmo corpus inicial.

Pergunta que responde:

> quanta mudança relacional aparece apenas por continuar otimizando o modelo?

### 4.4 Modelo congelado

`frozen`:

```text
theta_0 aplicado aos corpora de todos os períodos
```

Pergunta que responde:

> quanta variação aparece sem atualizar pesos?

### 4.5 Modelos independentes por período

`independent_period`:

```text
theta_t = modelo treinado do zero somente em D_t
```

Pergunta que responde:

> a continuidade cronológica dos pesos é necessária, ou snapshots
> independentes bastam?

### 4.6 Treino acumulativo do zero

`cumulative_retrain`:

```text
theta_t = modelo treinado do zero em D_0 + ... + D_t
```

Pergunta que responde:

> basta observar todos os dados anteriores, sem herdar a história de
> otimização?

---

## 5. Benchmark sintético estrutural

O benchmark sintético atual foi redesenhado para testar **forma temporal** da
mudança, não apenas diferença entre início e fim.

### 5.1 Estrutura

- `10` períodos;
- `40` sujeitos;
- `16` âncoras estáveis;
- `24` palavras-alvo;
- `6` quartetos pareados;
- cada quarteto compartilha os mesmos estados inicial e alternativo;
- dentro de cada quarteto, cada palavra recebe uma forma temporal diferente.

Os perfis confirmatórios das palavras-alvo são calculados contra as mesmas
âncoras estáveis. Isso evita que a trajetória de uma palavra seja contaminada
por outras palavras também em movimento.

### 5.2 Condições

#### Gradual persistente

Pequenos passos acumulados até um novo estado:

```text
[0.90, 0.81, 0.72, 0.63, 0.54, 0.46, 0.37, 0.28, 0.19, 0.10]
```

#### Abrupta persistente

Salto localizado seguido de permanência no novo estado:

```text
[0.90, 0.90, 0.90, 0.90, 0.90, 0.10, 0.10, 0.10, 0.10, 0.10]
```

#### Transitória reversível

Mudança forte seguida de retorno:

```text
[0.90, 0.90, 0.90, 0.90, 0.10, 0.10, 0.90, 0.90, 0.90, 0.90]
```

#### Oscilatória

Movimento intenso, mas sem deslocamento final persistente:

```text
[0.90, 0.10, 0.90, 0.10, 0.90, 0.10, 0.90, 0.10, 0.90, 0.90]
```

---

## 6. Métricas principais

### 6.1 Magnitude final

```text
M_final(w) = mean(abs(r_T(w) - r_0(w)))
```

Mede diferença entre início e fim.

### 6.2 Caminho total

```text
L(w) = sum_t mean(abs(r_t(w) - r_(t-1)(w)))
```

Mede atividade total, incluindo oscilação e reversão.

### 6.3 Eficiência de deslocamento

```text
eff(w) = M_final(w) / L(w)
```

Alta quando o caminho percorrido vira deslocamento final; baixa quando há
ida-e-volta ou oscilação.

### 6.4 Recuperação

```text
recovery(w) = 1 - M_final(w) / peak_magnitude(w)
```

Alta quando houve pico intermediário seguido de retorno.

### 6.5 Fidelidade direcional

Compara a direção observada com o oráculo sintético:

```text
cos(delta_observado, delta_oraculo)
```

É sempre interpretada junto com magnitude. Direção positiva isolada não basta
se a magnitude não supera o nulo.

### 6.6 Erro global de forma

Compara a curva normalizada de magnitude acumulada com a curva oracular. Útil
como resumo, mas insuficiente para rupturas abruptas, porque penaliza qualquer
atividade fora do salto ideal.

### 6.7 Métricas locais de evento

Para mudanças abruptas, medimos explicitamente:

- `event_period`: período do maior passo no oráculo;
- `observed_peak_period`: período do maior passo observado;
- `event_period_error`: erro de localização temporal;
- `event_step_magnitude`: magnitude observada no evento;
- `event_concentration`: fração do caminho total concentrada no evento;
- `pre_event_drift`: magnitude acumulada antes do evento;
- `post_event_drift`: caminho percorrido depois do evento;
- `event_fidelity`: direção local do evento contra o oráculo.

Essas métricas permitem separar:

- localizar o evento;
- acertar sua direção;
- concentrar o deslocamento no salto;
- distinguir deriva de fundo de sinal semântico.

---

## 7. Resultados atuais

### 7.1 Experimento estrutural principal

Configuração:

- seeds `1000`, `1001`, `1002`;
- `100` exemplos por sujeito/período;
- `10` períodos;
- orçamento fixo de `8250` atualizações por seed nos regimes principais;
- representação principal: distribuição preditiva + Jensen-Shannon.

O percentil 95 do nulo ressampleado para `M_final` foi:

```text
0.058231
```

Resultados de `continual_real`:

| Condição | `M_final` mediano | Acima do p95 nulo | Caminho | Recuperação | `F_acc` | Shape error |
|---|---:|---:|---:|---:|---:|---:|
| gradual | 0.071929 | 77.8% | 0.196031 | 0.000 | 0.653 | 0.117 |
| abrupt_persistent | 0.072157 | 94.4% | 0.292038 | 0.209 | 0.911 | 0.194 |
| transient | 0.027328 | 0.0% | 0.352545 | 0.660 | 0.914 | 0.228 |
| oscillating | 0.024511 | 11.1% | 0.683296 | 0.703 | 0.925 | 0.204 |

Leitura:

- gradual: deslocamento persistente detectado;
- abrupta: deslocamento final detectado e direção correta;
- transitória: não parece mudança final, mas mostra pico e recuperação;
- oscilatória: grande atividade, mas baixo deslocamento final.

### 7.2 Continuidade cronológica importa

`independent_period` teve desempenho pior que `continual_real` em forma,
recuperação e detecção acima do nulo.

`cumulative_retrain` observou `D_0 + ... + D_t`, mas sem herdar pesos. Ele
reduziu magnitude e comprimiu trajetórias, especialmente em transitória e
oscilatória.

Conclusão:

> a trajetória relacional não é apenas uma propriedade dos dados vistos em cada
> checkpoint; ela também depende da história de otimização.

### 7.3 Rupturas abruptas

Inicialmente, `shape_error` sugeria que o modelo suavizava rupturas. A ablação
de posição da ruptura mostrou uma leitura mais precisa.

Executamos `abrupt_switch_period` em `t3`, `t5` e `t7`.

Para `abrupt_persistent`:

| Ruptura | `M_final` mediano | Acima do p95 nulo | Caminho | Recuperação | `F_acc` | Shape error |
|---:|---:|---:|---:|---:|---:|---:|
| `t3` | 0.072142 | 94.4% | 0.291862 | 0.217 | 0.919 | 0.194 |
| `t5` | 0.072157 | 94.4% | 0.292038 | 0.209 | 0.911 | 0.194 |
| `t7` | 0.072673 | 88.9% | 0.290901 | 0.071 | 0.910 | 0.218 |

As métricas locais mostram que o pico observado ocorre no período correto:

| Ruptura | Evento esperado | Pico observado | Erro | Concentração | Fidelidade local |
|---:|---:|---:|---:|---:|---:|
| `t3` | 3 | 3 | 0 | 0.3025 | 0.9533 |
| `t5` | 5 | 5 | 0 | 0.2898 | 0.9468 |
| `t7` | 7 | 7 | 0 | 0.2868 | 0.9507 |

Interpretação refinada:

> Timeformer recupera o momento e a direção local de rupturas abruptas, mas
> distribui parte do caminho relacional em deriva pré- e pós-evento.

Ou seja, a limitação não é "não ver a ruptura". A limitação é que apenas cerca
de 30% do caminho total fica concentrado no salto abrupto.

### 7.4 Controles locais de evento

Contra controles, a concentração do salto no real fica acima do nulo e do
placebo:

| Ruptura | Concentração real | Real - nulo | Real - placebo | Fidelidade local |
|---:|---:|---:|---:|---:|
| `t3` | 0.3025 | +0.1844 | +0.1765 | 0.9533 |
| `t5` | 0.2898 | +0.1942 | +0.2288 | 0.9468 |
| `t7` | 0.2868 | +0.1714 | +0.2335 | 0.9507 |

A deriva pré-evento contra o nulo ficou próxima de zero. Isso sugere que a
deriva anterior ao evento é, em boa parte, instabilidade de fundo. Já o evento
em si é localizado, direcional e mais concentrado que os controles.

---

## 8. Contribuições pretendidas

### C1. Formulação relacional de mudança semântica temporal

Propomos medir mudança semântica como alteração no perfil relacional de uma
palavra entre checkpoints cronológicos.

### C2. Timeformer como treinamento contínuo, não como input temporal

O modelo não recebe tempo como feature. A temporalidade é induzida pela ordem
cronológica de treinamento.

### C3. Deslocamento vetorial relacional

O deslocamento permanece vetorial:

```text
delta_rel(w, a, b)
```

mas seus eixos são relações com outras palavras, não dimensões ocultas de um
embedding absoluto.

### C4. Métricas para forma temporal

O paper separa:

- deslocamento final;
- atividade total;
- reversão;
- oscilação;
- localização de evento;
- concentração de evento;
- deriva pré/pós-evento.

### C5. Evidência experimental controlada

O benchmark sintético mostra que o regime contínuo registra:

- deslocamento persistente;
- acumulação gradual;
- atividade sem deslocamento final;
- reversão;
- momento e direção de rupturas abruptas.

Também mostra limitações: pequenas mudanças não são sempre detectáveis, e
rupturas abruptas têm caminho parcialmente distribuído fora do evento.

---

## 9. O que não afirmamos ainda

Não afirmamos que o método já resolve LSCD em corpus real.

Não afirmamos que toda variação relacional pequena é mudança semântica
relevante.

Não afirmamos que `shape_error` sozinho mede corretamente rupturas abruptas.

Não afirmamos que o nulo global basta. O nulo mostrou heterogeneidade associada
a propriedades da distribuição base, então calibração por propriedades
observáveis ainda é necessária.

Não afirmamos que polissemia e sentidos coexistentes estejam resolvidos. O caso
bifurcante permanece mais difícil e deve ser tratado como extensão ou estudo
posterior.

---

## 10. Arquitetura de código atual

Componentes principais:

- `src/timeformers/structural_corpus.py`: gera benchmark estrutural com
  âncoras, quartetos e formas temporais;
- `src/timeformers/structural_metrics.py`: métricas de trajetória, direção,
  forma e evento local;
- `src/timeformers/structural_experiment.py`: exporta linhas por sujeito e
  séries temporais;
- `scripts/run_structural_relational_experiment.py`: executa regimes principal
  e controles;
- `scripts/summarize_structural_event_metrics.py`: agrega métricas locais e
  deltas contra controles;
- `docs/structural_relational_experiment_preregistration.md`: pré-registro do
  experimento estrutural;
- `docs/relational_change_current_plan.md`: diário técnico detalhado da
  evolução dos resultados.

Pipeline antigo de teacher/student e trajectory encoder:

- deixa de ser configuração principal;
- pode permanecer como baseline histórico ou ablação;
- não deve orientar a narrativa central do paper.

---

## 11. Próximos passos

### 11.1 Consolidar seção de método

Escrever em formato de paper:

1. treinamento contínuo cronológico;
2. probe preditivo pós-Transformer;
3. perfil relacional;
4. deslocamento relacional;
5. métricas de forma temporal;
6. controles.

### 11.2 Consolidar seção experimental

Transformar os resultados atuais em tabelas finais:

- experimento estrutural principal;
- comparação arquitetural;
- ablação de posição da ruptura;
- métricas locais contra controles.

### 11.3 Calibração do nulo

Investigar limiares condicionados por propriedades observáveis:

- frequência;
- entropia preditiva;
- distribuição base;
- incerteza entre seeds;
- magnitude nula por palavra.

### 11.4 Estrutura semântica multidimensional

O benchmark atual usa uma dimensão sintética dominante. O próximo benchmark
deve incluir múltiplas dimensões semânticas independentes para testar se o
método continua funcionando quando a mudança não está alinhada a um único eixo
latente.

### 11.5 Corpus real

O avanço para corpus real começa por um dataset gratuito e reprodutível:
SemEval-2020 Task 1, inicialmente no recorte em inglês. O objetivo do primeiro
teste não é ainda declarar mudança semântica, mas validar se o pipeline
relacional funciona fora do vocabulário sintético e contra um benchmark
conhecido de lexical semantic change detection.

Infraestrutura inicial criada:

- `src/timeformers/real_corpus.py`: leitura de corpora por período,
  tokenização simples, vocabulário dinâmico, dataset MLM real e probes por
  palavra;
- `src/timeformers/real_models.py`: Transformer MLM com vocabulário e
  comprimento de sequência configuráveis;
- `scripts/run_diachronic_relational_experiment.py`: treino contínuo em textos
  reais por período e exportação de deltas relacionais por palavra.

Formato de entrada esperado:

```text
diachronic_dir/
  1950.txt
  1960.txt
  ...
```

ou:

```text
diachronic_dir/
  1950/*.txt
  1960/*.txt
  ...
```

O probe real inicial usa:

```text
[CLS] palavra [MASK] [SEP]
```

e mede a distribuição prevista sobre uma lista de palavras-âncora. A relação
entre alvos é Jensen-Shannon entre essas distribuições preditivas sobre
âncoras:

```text
q_t(w) = P_t(anchor | [CLS] w [MASK] [SEP])
r_t(w)[v] = 1 - JS(q_t(w), q_t(v)) / log(2)
```

Esse é o análogo real-corpus do probe sintético, mas com uma diferença
importante: as âncoras precisam ser escolhidas de modo estável e interpretável.

O primeiro piloto SemEval deve:

- calibrar nulo;
- controlar número de atualizações por período;
- filtrar palavras-alvo com cobertura suficiente;
- escolher âncoras frequentes e distribuídas no tempo;
- incluir um controle de ordem permutada ou placebo repetido;
- relatar frequência, entropia preditiva e cobertura por palavra;
- comparar o ranking produzido contra os rótulos gold de mudança binária e
  graduada do SemEval;
- evitar afirmações semânticas fortes antes de validação manual/externa.

---

## 12. Frase-guia do paper

> We model semantic change not as movement in an absolute embedding coordinate
> system, but as a change in a word's relational profile across chronologically
> trained Transformer checkpoints.

Em português:

> Modelamos mudança semântica não como movimento em um sistema absoluto de
> coordenadas de embeddings, mas como alteração no perfil relacional de uma
> palavra ao longo de checkpoints de Transformer treinados cronologicamente.
