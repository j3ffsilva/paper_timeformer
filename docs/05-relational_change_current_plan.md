# Plano atual: mudança semântica relacional entre checkpoints

## Status

Este documento descreve a direção experimental principal atual. O pipeline de
teacher/student e trajetória aprendida descrito em `02-novo_planejamento.md` deixa
de ser a configuração principal e permanece somente como baseline histórico.

## Hipótese

Um Transformer padrão é treinado cronologicamente:

```text
theta_0 = treino(D_0)
theta_1 = continua_treino(theta_0, D_1)
...
theta_t = continua_treino(theta_{t-1}, D_t)
```

Nenhum identificador de período é fornecido ao modelo. A mudança semântica não
é a diferença entre coordenadas absolutas de embeddings. Ela é a mudança das
relações internas de uma palavra com as demais palavras no mesmo checkpoint.

```text
r_t(w)[v] = similaridade_t(w, v)
delta_rel(w, a, b) = r_b(w) - r_a(w)
```

Uma transformação global que preserve todas as relações deve produzir mudança
relacional zero. A trajetória é derivada posteriormente da sequência de perfis
e deltas relacionais; ela não é aprendida por teacher/student.

## Representação principal em avaliação

O experimento sintético mostrou que consultar diretamente `h_subj` não recupera
de forma confiável a direção semântica conhecida. A representação principal
passa a ser a distribuição de contextos produzida por um **probe preditivo
pós-Transformer**:

```text
[CLS] palavra [MASK] [MASK] [SEP]
```

Extraímos as distribuições previstas nas posições de verbo e objeto, restritas
aos 16 contextos sintéticos válidos. A relação entre duas palavras é calculada
pela similaridade de Jensen-Shannon entre suas distribuições previstas.

```text
q_t(w) = P_t(contextos | [CLS] w [MASK] [MASK] [SEP])
r_t(w)[v] = 1 - JS(q_t(w), q_t(v)) / log(2)
```

Essa formulação não depende das coordenadas ocultas do Transformer e possui
interpretação direta: duas palavras são semelhantes quando o checkpoint prevê
distribuições de contexto semelhantes para elas.

Isto não é um sinal temporal anterior ao Transformer. É uma consulta posterior
ao treinamento de cada checkpoint.

As geometrias ocultas permanecem como ablações:

- cosseno dos estados mascarados;
- cosseno centralizado, invariante a translação, rotação/reflexão e escala
  positiva global;
- distância euclidiana normalizada, com as mesmas invariâncias globais;
- estado direto do sujeito e centroides contextuais.

## Controles obrigatórios

- `continual_real`: `D_0 -> D_1 -> ... -> D_t`.
- `continual_placebo`: repete `D_0` para estimar deriva causada apenas por
  continuar a otimização.
- `resampled_null`: usa novos textos em cada período, mas mantém constante a
  distribuição semântica plantada (`trajectory_scale=0`). Este é o controle
  principal para falsos positivos semânticos.
- `frozen`: aplica `theta_0` aos diferentes corpora sem atualizar pesos.
- validação e parada antecipada por período, restaurando o melhor checkpoint;
- probes fixos e probes preditivos, separados das ocorrências do próprio corpus.
- registro de passos computados e do passo selecionado para cada checkpoint;
- estado retomável contendo modelo e otimizador.

Resultados brutos e comparações com placebo devem ser mantidos. Calculamos:

```text
vantagem_direcional = direção_real_vs_oráculo - direção_placebo_vs_oráculo
```

A antiga subtração vetorial `delta_real - delta_placebo` permanece somente como
diagnóstico exploratório. Não deve ser interpretada como efeito causal, pois
real e placebo podem percorrer quantidades diferentes de passos de otimização.

## Relações e métricas

- mudança de vizinhos por Jaccard;
- mudança de ranking relacional por Spearman normalizada em `[0,1]`;
- média da mudança absoluta de similaridade;
- CKA como diagnóstico global;
- cosseno entre a direção relacional aprendida e o oráculo sintético;
- direção observada, direção placebo e vantagem direcional pareada;
- Jensen-Shannon como relação principal;
- cosseno oculto, cosseno centralizado e euclidiana normalizada como ablações.

## Resultado atual em três seeds

Configuração: seeds `1000`, `1001` e `1002`; 100 ocorrências por
palavra/período; `t0` com até 60 épocas; períodos posteriores com até 30 épocas
e parada antecipada.

Para Jensen-Shannon sobre distribuições previstas, na mudança acumulada
`t0 -> t9`:

| Classe | Direção observada | DP | Direção placebo | Vantagem pareada | DP |
|---|---:|---:|---:|---:|---:|
| abrupt | +0.968 | 0.012 | +0.554 | +0.414 | 0.030 |
| bifurcating | +0.881 | 0.047 | +0.540 | +0.341 | 0.080 |
| drift | +0.955 | 0.014 | +0.546 | +0.409 | 0.035 |
| stable | +0.951 | 0.012 | +0.617 | +0.334 | 0.043 |

A vantagem pareada foi positiva nas três seeds para todas as classes. O placebo
também apresenta direção positiva substancial, confirmando que deriva de
otimização é uma ameaça real à validade.

O placebo repetido e o nulo ressampleado respondem perguntas diferentes:

- placebo repetido: deriva ao continuar ajustando-se ao mesmo corpus finito;
- nulo ressampleado: variação ao receber novos textos sem mudança na
  distribuição semântica.

O nulo ressampleado deve calibrar a detecção de mudança. O placebo repetido
permanece como diagnóstico de otimização.

Comparação média entre relações em `t0 -> t9`:

| Relação | Direção observada | Placebo | Vantagem |
|---|---:|---:|---:|
| Distribuição prevista + Jensen-Shannon | +0.939 | +0.564 | +0.375 |
| Estado oculto + cosseno | +0.933 | +0.632 | +0.300 |
| Estado oculto + cosseno centralizado | +0.537 | -0.024 | +0.561 |
| Estado oculto + euclidiana normalizada | +0.463 | +0.181 | +0.282 |

Jensen-Shannon é a configuração principal por combinar direção alta,
interpretação semântica direta e menor dependência da geometria interna. O
cosseno centralizado é a ablação geométrica mais forte em vantagem sobre
placebo.

O sinal é muito mais forte para mudança acumulada. Nos passos consecutivos
iniciais (`t1`, `t2`) e no passo final `t8 -> t9`, a direção é fraca ou
instável. Não devemos afirmar ainda que o método detecta mudanças pequenas.

### Controle com orçamento fixo em três seeds

Uma ablação adicional executou as seeds `1000`, `1001` e `1002` sem parada
antecipada e selecionou o checkpoint final de cada período. Real e placebo
receberam exatamente `8250` atualizações em cada seed.

| Relação | Direção observada | Placebo | Vantagem |
|---|---:|---:|---:|
| Distribuição prevista + Jensen-Shannon | +0.913 | +0.603 | +0.310 |
| Estado oculto + cosseno | +0.918 | +0.658 | +0.260 |
| Estado oculto + cosseno centralizado | +0.513 | +0.001 | +0.512 |
| Estado oculto + euclidiana normalizada | +0.493 | +0.293 | +0.200 |

A vantagem Jensen-Shannon diminui em relação ao regime escolhido por validação
(`+0.375`), mas permanece positiva. Portanto, seleção desigual de checkpoints
explica parte, mas não todo, o sinal observado. Para `t0 -> t9`, a vantagem
Jensen-Shannon foi positiva nas três seeds para todas as classes:

| Classe | Direção observada | Placebo | Vantagem | DP da vantagem |
|---|---:|---:|---:|---:|
| abrupt | +0.951 | +0.588 | +0.363 | 0.036 |
| bifurcating | +0.830 | +0.602 | +0.228 | 0.086 |
| drift | +0.934 | +0.577 | +0.357 | 0.034 |
| stable | +0.937 | +0.647 | +0.290 | 0.013 |

O controle fixo também confirma a limitação de resolução temporal. A vantagem
direcional média nos passos consecutivos é positiva de `t1 -> t2` até
`t7 -> t8`, mas fica negativa em `t8 -> t9` (`-0.056`). Para mudanças
acumuladas desde `t0`, a vantagem cresce até aproximadamente `+0.37` e termina
em `+0.31`.

### Sensibilidade à magnitude da mudança

Escalamos cada trajetória em torno de seu valor inicial:

```text
p_t(alpha) = p_0 + alpha * (p_t - p_0)
```

O nulo ressampleado foi executado em três seeds com `alpha=0`. Em `t0 -> t9`,
a magnitude Jensen-Shannon nula apresentou média `0.0062`, desvio padrão
`0.0034` e percentil 95 `0.0135`.

Uma palavra é considerada detectada quando sua magnitude observada supera o
percentil 95 do nulo ressampleado e sua direção possui cosseno positivo com o
oráculo.

As escalas `0.50` e `0.75` foram replicadas nas seeds `1000`, `1001` e
`1002`; `0.25` permanece com uma seed.

| Escala | Magnitude média | Direção média | Detectadas acima do p95 nulo |
|---:|---:|---:|---:|
| 0.25 | 0.0060 | +0.210 | 5.0% |
| 0.50 | 0.0112 | +0.577 | 22.5% |
| 0.75 | 0.0232 | +0.784 | 81.7% |
| 1.00 | 0.0416 | +0.893 | 100.0% |

O cosseno direcional isolado pode parecer positivo mesmo quando a magnitude não
se distingue do nulo. Direção e magnitude devem ser avaliadas conjuntamente.
O limiar prático atual está próximo de `alpha=0.75` e foi replicado em três
seeds. Na escala `0.75`, a taxa de detecção por classe foi:

| Classe | Taxa de detecção | Direção média |
|---|---:|---:|
| abrupt | 90.0% | +0.885 |
| drift | 90.0% | +0.848 |
| stable | 83.3% | +0.670 |
| bifurcating | 63.3% | +0.733 |

`Bifurcating` permanece como a classe mais difícil, consistente com a
necessidade de representar sentidos coexistentes.

### Mais ocorrências versus mais atualizações

Testamos `300` exemplos por sujeito e período, em vez de `100`, sob dois
regimes:

1. mesmo número de épocas, que triplica aproximadamente o número de
   atualizações;
2. mesmo orçamento de `8250` atualizações, reduzindo o número de épocas para
   compensar o corpus maior.

O segundo regime foi replicado nas seeds `1000`, `1001` e `1002`. A comparação
principal é:

| Exemplos | Atualizações | Escala | Magnitude | Direção | Detectadas acima do p95 |
|---:|---:|---:|---:|---:|---:|
| 100 | 8250 | 0.50 | 0.0112 | +0.577 | 22.5% |
| 300 | 8250 | 0.50 | 0.0092 | +0.612 | 11.7% |
| 100 | 8250 | 0.75 | 0.0232 | +0.784 | 81.7% |
| 300 | 8250 | 0.75 | 0.0213 | +0.837 | 78.3% |

Com orçamento fixo, mais ocorrências melhoraram a direção média, mas não
reduziram o limiar nulo nem aumentaram a taxa de detecção. O percentil 95 do
nulo com `300` exemplos foi `0.013526`, praticamente idêntico ao valor de
`0.013509` obtido com `100` exemplos.

Com `300` exemplos e o mesmo número de épocas, a seed `1000` apresentou taxas
de detecção de `52.5%` em `alpha=0.50` e `100%` em `alpha=0.75`, mas esse
regime usa aproximadamente três vezes mais atualizações e ainda não foi
replicado. O ganho não pode ser atribuído somente à diversidade textual.

### Heterogeneidade do nulo

O nulo de alta quantidade de dados revelou que o ruído não é homogêneo entre
sujeitos. A correlação entre o valor semântico inicial plantado `p0` e a
magnitude nula foi `-0.769`:

| Faixa de `p0` | N | Magnitude nula média | p95 |
|---|---:|---:|---:|
| `[0.50, 0.75)` | 12 | 0.012286 | 0.019929 |
| `[0.75, 1.00]` | 108 | 0.002375 | 0.003895 |

Assim, um único limiar global é conservador para a maioria dos sujeitos e
insuficientemente descritivo para os sujeitos próximos à região de maior
incerteza. Essa heterogeneidade não pode ser corrigida com limiares por classe
sintética, pois tais classes não existirão em corpus real. Precisamos estudar
calibração condicionada por propriedades observáveis, como distribuição-base,
entropia preditiva, frequência e incerteza entre réplicas.

Uma palavra classificada como estável pelo gerador pode apresentar mudança
relacional: mesmo que sua propriedade própria permaneça constante, suas
relações mudam quando outras palavras se movem. Essa distinção deve ser
explicitada no paper.

## Riscos ainda abertos

- Na configuração principal, real e placebo executam números diferentes de
  passos devido à parada antecipada. Nas seeds `1001` e `1002`, o regime real
  executou respectivamente `6175` e `5825` passos, enquanto o placebo executou
  `4050` e `4525`. O controle de orçamento fixo reduz, mas não elimina, a
  vantagem observada.
- O placebo repetido positivo confirma deriva de otimização, mas não deve ser
  usado sozinho como distribuição nula semântica.
- O benchmark sintético alinha estruturalmente tarefa, probe e oráculo. Isso é
  apropriado para validação controlada, mas não demonstra ainda validade em
  corpus real.
- Uma palavra `stable` pode apresentar mudança relacional porque outras palavras
  se movem.
- Um limiar nulo global oculta forte heterogeneidade associada à distribuição
  semântica inicial.

## Próximo critério para prosseguir

O experimento pré-registrado em
`docs/06-structural_relational_experiment_preregistration.md` foi executado nas
seeds `1000`, `1001` e `1002`. Ele testou se os checkpoints registram a forma
temporal da mudança, distinguindo acumulação gradual, mudança abrupta
persistente, reversão e oscilação.

O desenho confirmatório usou `16` âncoras estáveis e `24` palavras-alvo. Essa
alteração foi feita antes da execução confirmatória porque o primeiro smoke
test, sem âncoras, misturava a trajetória própria de cada palavra com o
movimento das demais palavras no perfil relacional.

Todos os regimes (`continual_real`, `resampled_null`, `continual_placebo`)
executaram exatamente `8250` passos em cada seed.

### Resultado do experimento estrutural

O percentil 95 do nulo ressampleado para `M_final` foi `0.058231`.

| Condição | `M_final` mediano | Acima do p95 nulo | Caminho mediano | Eficiência | Recuperação | `F_acc` | `F_acc_adv` | Shape error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gradual | 0.071929 | 77.8% | 0.196031 | 0.358 | 0.000 | 0.653 | 1.437 | 0.117 |
| abrupt_persistent | 0.072157 | 94.4% | 0.292038 | 0.240 | 0.209 | 0.911 | 1.182 | 0.194 |
| transient | 0.027328 | 0.0% | 0.352545 | 0.078 | 0.660 | 0.914 | 1.275 | 0.228 |
| oscillating | 0.024511 | 11.1% | 0.683296 | 0.036 | 0.703 | 0.925 | 1.323 | 0.204 |

Leitura pelos critérios pré-registrados:

- H1, acumulação gradual: sustentada. A condição gradual possui direção
  acumulada positiva, vantagem positiva sobre placebo, `M_final` acima do nulo
  e forma mais próxima do oráculo gradual do que do oráculo abrupto.
- H2, caminhos com mesmo destino: parcialmente sustentada. Gradual e abrupta
  chegam a magnitudes finais quase iguais, mas a forma abrupta não foi
  recuperada como abrupta; ela ficou mais próxima do oráculo gradual trocado
  do que do próprio oráculo abrupto. O treinamento contínuo parece suavizar
  rupturas.
- H3, reversão: sustentada. A condição transitória tem pico intermediário acima
  do nulo e recuperação maior que a abrupta.
- H4, atividade versus deslocamento persistente: sustentada. A condição
  oscilatória percorre o maior caminho, mas termina com deslocamento final baixo
  e recuperação alta.

Conclusão atual: o Timeformer registra deslocamento relacional persistente,
acumulação gradual, atividade temporal e reversão. Porém, a recuperação da
forma abrupta ainda é fraca: a arquitetura tende a suavizar ou espalhar a
ruptura temporal.

### Comparação arquitetural: modelos independentes por período

O primeiro baseline do Experimento A comparou o regime contínuo com
`independent_period`: um modelo separado por período, treinado somente em `D_t`,
sem herdar pesos anteriores. O orçamento por checkpoint foi mantido comparável:
`1500` passos em `t0` e `750` passos em cada período posterior, totalizando
`8250` passos por seed também no baseline independente.

| Regime | Condição | `M_final` mediano | Acima do p95 nulo | Caminho mediano | Eficiência | Recuperação | `F_acc` | Shape error |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| continual_real | gradual | 0.071929 | 77.8% | 0.196031 | 0.358 | 0.000 | 0.653 | 0.117 |
| independent_period | gradual | 0.042599 | 0.0% | 0.075852 | 0.542 | 0.000 | 0.889 | 0.160 |
| continual_real | abrupt_persistent | 0.072157 | 94.4% | 0.292038 | 0.240 | 0.209 | 0.911 | 0.194 |
| independent_period | abrupt_persistent | 0.042714 | 5.6% | 0.080660 | 0.496 | 0.025 | 0.731 | 0.350 |
| continual_real | transient | 0.027328 | 0.0% | 0.352545 | 0.078 | 0.660 | 0.914 | 0.228 |
| independent_period | transient | 0.027529 | 0.0% | 0.086491 | 0.305 | 0.280 | 0.649 | 0.570 |
| continual_real | oscillating | 0.024511 | 11.1% | 0.683296 | 0.036 | 0.703 | 0.925 | 0.204 |
| independent_period | oscillating | 0.031528 | 0.0% | 0.123822 | 0.246 | 0.321 | 0.709 | 0.412 |

Pelos critérios do adendo pré-registrado:

- o contínuo teve menor `shape_error` em `gradual`;
- o contínuo teve menor `shape_error` em `abrupt_persistent`;
- o contínuo manteve recuperação maior em `transient` e `oscillating`;
- o contínuo manteve taxa acima do p95 nulo muito superior em `gradual` e
  `abrupt_persistent`.

Conclusão: o baseline independente não basta para recuperar a forma temporal.
A continuidade cronológica dos pesos melhora a recuperação de trajetória e a
distinção entre deslocamento persistente e atividade com retorno.

Esse resultado também enfraquece a hipótese de que a suavização de rupturas
abruptas seja causada apenas pela continuidade dos pesos: o baseline
independente foi ainda pior em `abrupt_persistent`. A suavização pode estar
ligada ao probe, à métrica de forma, à quantidade de evidência por período ou à
própria dificuldade de estimar rupturas a partir de checkpoints discretos.

### Comparação arquitetural: treino acumulativo do zero

O segundo baseline do Experimento A comparou o regime contínuo com
`cumulative_retrain`: para cada período `t`, um modelo é treinado do zero em
`D0 + ... + Dt`. Esse controle testa se basta observar todos os dados
anteriores, sem continuidade dos pesos.

O baseline acumulativo executou exatamente os passos acumulados correspondentes
ao regime contínuo em cada checkpoint:

```text
[1500, 2250, 3000, 3750, 4500, 5250, 6000, 6750, 7500, 8250]
```

Assim, cada seed executou `48750` atualizações no baseline acumulativo.

| Regime | Condição | `M_final` mediano | Acima do p95 nulo | Caminho mediano | Eficiência | Recuperação | `F_acc` | Shape error |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| continual_real | gradual | 0.071929 | 77.8% | 0.196031 | 0.358 | 0.000 | 0.653 | 0.117 |
| cumulative_retrain | gradual | 0.034202 | 5.6% | 0.060609 | 0.608 | 0.000 | 0.568 | 0.098 |
| continual_real | abrupt_persistent | 0.072157 | 94.4% | 0.292038 | 0.240 | 0.209 | 0.911 | 0.194 |
| cumulative_retrain | abrupt_persistent | 0.034485 | 0.0% | 0.082131 | 0.447 | 0.000 | 0.495 | 0.308 |
| continual_real | transient | 0.027328 | 0.0% | 0.352545 | 0.078 | 0.660 | 0.914 | 0.228 |
| cumulative_retrain | transient | 0.017158 | 0.0% | 0.079701 | 0.210 | 0.372 | 0.396 | 0.510 |
| continual_real | oscillating | 0.024511 | 11.1% | 0.683296 | 0.036 | 0.703 | 0.925 | 0.204 |
| cumulative_retrain | oscillating | 0.030637 | 0.0% | 0.099177 | 0.308 | 0.207 | 0.711 | 0.411 |

O acumulativo teve `shape_error` menor em `gradual`, mas com magnitude muito
baixa e quase nenhuma detecção acima do p95 nulo. Nas demais condições, ele
perdeu para o contínuo em forma, caminho, recuperação ou deslocamento
persistente. Em especial, `transient` e `oscillating` foram comprimidos em
trajetórias curtas, com baixa recuperação.

Conclusão: observar `D0 + ... + Dt` não substitui continuidade cronológica dos
pesos para este objetivo. O regime contínuo é o único dos três que registra
simultaneamente deslocamento persistente, atividade temporal e retorno. O
acumulativo sugere que a trajetória não é apenas uma propriedade dos dados
vistos em cada checkpoint; ela também depende da história de otimização.

Antes de corpus real, executar:

1. investigar se a suavização de rupturas abruptas vem da arquitetura
   contínua, do orçamento de treinamento, do probe preditivo ou da métrica de
   forma;
2. diagnosticar e calibrar a heterogeneidade do nulo usando apenas propriedades
   observáveis, sem classes ou parâmetros sintéticos ocultos;
3. separar curvas de aprendizagem por número de atualizações e quantidade de
   exemplos, pois mais épocas e mais dados respondem perguntas diferentes;
4. criar benchmark sintético com estrutura semântica multidimensional, além de
   `p_n1`.

### Próxima ablação: posição da ruptura

O primeiro diagnóstico da suavização abrupta será variar o período de ruptura
sem alterar o restante do desenho estrutural. O gerador agora aceita:

```text
--abrupt-switch-period
--transient-onset-period
--transient-width
```

Os defaults preservam o experimento já executado. A próxima execução deve
comparar ao menos `abrupt_switch_period` em `3`, `5` e `7`, mantendo seeds,
orçamento e métrica principal. Se a ruptura cedo/tarde também for suavizada, a
causa provavelmente não é apenas o ponto discreto da mudança. Se a recuperação
melhorar em alguma posição, precisamos estudar resolução temporal e exposição
pós-ruptura antes de alterar probe ou métrica.

### Resultado da ablação de posição da ruptura

A ablação foi executada nas seeds `1000`, `1001` e `1002` para
`abrupt_switch_period` em `3`, `5` e `7`. O caso `5` corresponde ao experimento
confirmatório original.

Para a condição `abrupt_persistent`, a magnitude final e o caminho total foram
muito semelhantes entre as três posições:

| Ruptura | `M_final` mediano | Acima do p95 nulo | Caminho mediano | Recuperação | `F_acc` | Shape error |
|---:|---:|---:|---:|---:|---:|---:|
| `t3` | 0.072142 | 94.4% | 0.291862 | 0.217 | 0.919 | 0.194 |
| `t5` | 0.072157 | 94.4% | 0.292038 | 0.209 | 0.911 | 0.194 |
| `t7` | 0.072673 | 88.9% | 0.290901 | 0.071 | 0.910 | 0.218 |

A curva por período mostra um pico de `step_magnitude` exatamente no período da
ruptura plantada:

| Ruptura | Pico mediano no período da ruptura | `accumulated_magnitude` mediano antes da ruptura | `accumulated_magnitude` mediano final |
|---:|---:|---:|---:|
| `t3` | 0.0878 | 0.0322 em `t2` | 0.0721 |
| `t5` | 0.0893 | 0.0271 em `t4` | 0.0722 |
| `t7` | 0.0868 | 0.0267 em `t6` | 0.0727 |

Essa ablação muda a interpretação anterior. O modelo registra um salto local no
período correto da ruptura; a suavização aparente vem em parte de deriva
pré-ruptura e de métricas globais que penalizam qualquer atividade fora do
passo abrupto ideal. No caso `t7`, a recuperação é menor e o `shape_error`
maior provavelmente porque há menos períodos pós-ruptura para consolidar o novo
estado.

Próxima consequência metodológica: criar métricas locais de evento para
rupturas, separando:

- deriva pré-evento;
- concentração do salto no período correto;
- persistência pós-evento;
- deslocamento final.

Com isso, `shape_error` continua útil como resumo global, mas não deve ser a
única evidência sobre recuperação de rupturas abruptas.

Essas métricas locais foram implementadas e adicionadas aos arquivos
`structural_metrics.*`:

- `event_period`: período do maior passo no oráculo;
- `observed_peak_period`: período do maior passo observado;
- `event_period_error`: distância absoluta entre os dois períodos;
- `event_step_magnitude`: magnitude observada no passo do evento;
- `event_concentration`: fração do caminho total concentrada no evento;
- `pre_event_drift`: magnitude acumulada antes do evento;
- `pre_event_drift_ratio`: deriva pré-evento normalizada pelo caminho total;
- `post_event_drift`: caminho percorrido depois do evento;
- `post_event_drift_ratio`: deriva pós-evento normalizada pelo caminho total;
- `event_fidelity`: cosseno entre direção observada e oráculo no passo do
  evento.

Na condição `abrupt_persistent`, agregando três seeds:

| Ruptura | Evento esperado | Pico observado | Erro | Concentração | Deriva pré | Deriva pós | Fidelidade local |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `t3` | 3 | 3 | 0 | 0.3025 | 0.0322 | 0.1539 | 0.9533 |
| `t5` | 5 | 5 | 0 | 0.2898 | 0.0271 | 0.1080 | 0.9468 |
| `t7` | 7 | 7 | 0 | 0.2868 | 0.0267 | 0.0502 | 0.9507 |

Conclusão refinada: o modelo localiza a ruptura corretamente e na direção
correta. A limitação não é localização temporal do evento, mas **concentração
do caminho**: apenas cerca de 30% do caminho total fica no salto abrupto; o
restante aparece como deriva antes e depois do evento. Isso explica por que
`shape_error` ainda parece alto mesmo quando o pico local está correto.

Para o paper, a formulação mais precisa é:

> Timeformer recupera o momento e a direção local de rupturas abruptas, mas
> distribui parte do caminho relacional em deriva pré- e pós-evento.

### Sumarização local contra controles

Foi criado o script:

```text
scripts/summarize_structural_event_metrics.py
```

Ele lê múltiplos grupos experimentais no formato `nome=caminho`, agrega as
métricas locais de evento e calcula deltas pareados entre `continual_real` e
controles disponíveis (`resampled_null`, `continual_placebo`,
`independent_period`, `cumulative_retrain`).

Para a grade `switch_03`, `switch_05` e `switch_07`, a saída foi gravada em:

```text
outputs/structural_event_metric_summary/
```

Arquivos principais:

- `structural_event_metric_summary.csv`;
- `structural_event_metric_control_deltas.csv`;
- `structural_event_metric_control_delta_summary.csv`.

Na condição `abrupt_persistent`, o real apresentou erro de localização mediano
zero nas três posições, enquanto os controles tiveram pico local em períodos
menos alinhados ao evento. A concentração do salto no real ficou cerca de
`0.17` a `0.19` acima do nulo ressampleado e cerca de `0.18` a `0.23` acima do
placebo repetido:

| Ruptura | Concentração real | Real - nulo | Real - placebo | Fidelidade local real |
|---:|---:|---:|---:|---:|
| `t3` | 0.3025 | +0.1844 | +0.1765 | 0.9533 |
| `t5` | 0.2898 | +0.1942 | +0.2288 | 0.9468 |
| `t7` | 0.2868 | +0.1714 | +0.2335 | 0.9507 |

O delta de `pre_event_drift` contra o nulo ficou próximo de zero
(`-0.0011`, `+0.0007`, `-0.0027`). Isso sugere que parte da deriva pré-evento
é ruído/instabilidade de fundo, não sinal específico da ruptura. Já o evento
em si apresenta concentração e direção muito superiores aos controles.

Conclusão operacional: para rupturas, a métrica confirmatória deve combinar
ao menos:

1. `event_period_error` baixo;
2. `event_fidelity` alto;
3. `event_concentration` acima dos controles;
4. `pre_event_drift` calibrado contra nulo;
5. `post_event_drift` interpretado junto com a quantidade de períodos
   pós-evento disponíveis.

---

## Pilotos em corpus real (SemEval-2020 Task 1)

### Formulação atual

A formulação matemática do perfil relacional foi formalizada em
`docs/09-relational_profile_formalization.md`. O perfil é definido como log-PMI
sobre o vocabulário completo:

```text
R_t(w)[v] = log( q_t(w)[v] / p_t[v] )
```

Onde `q_t(w)` é a média das distribuições do MLM head sobre ocorrências reais
de `w` mascarada, e `p_t` é a distribuição do probe neutro `[CLS][MASK][SEP]`.
O deslocamento é `pmi_cosine = 1 - cos(R_t0(w), R_t1(w))`.

### Experimentos executados (2026-06-05/06)

| Experimento | d_model | Épocas (t0+t1) | Windows | Spearman graded | AUC binário |
|---|---:|---:|---:|---:|---:|
| `semeval2020_pmi_pilot` | 96 | 3+2 | 409k+421k | -0.057 | 0.482 |
| `semeval2020_pmi_line_documents_3_2` | 96 | 3+2 | 300k+366k | -0.025 | 0.494 |
| `semeval2020_pmi_long_epochs_12_8` | 96 | 12+8 | 409k+421k | **+0.114** | **0.560** |
| `semeval2020_pmi_dynamic_mlm_12_8_d128` | 128 | 12+8 | 370k+409k | -0.070 | 0.509 |

O melhor resultado foi `long_epochs` com `pmi_cosine`: Spearman=+0.114, AUC=0.560.

Top-5 por `pmi_cosine` no `long_epochs`:
`graft_nn` (changed), `record_nn` (changed), `head_nn` (changed),
`relationship_nn` (stable), `prop_nn` (changed).

### Diagnóstico: sinal dominado por mudança de entropia

Em todos os experimentos, `predicted_vs_entropy_abs_delta` apresentou
correlação rho ≈ 0.92–0.95 (p < 0.001). O score `pmi_cosine` está
quase inteiramente determinado por quanto a entropia da distribuição preditiva
de cada palavra mudou entre os dois checkpoints — não pela mudança semântica.

**Interpretação:** palavras cuja distribuição preditiva ficou mais concentrada
(entropia caiu) em `t1` aparecem com alto deslocamento mesmo sem mudança
semântica. Palavras que o modelo nunca aprendeu bem (alta entropia em ambos)
aparecem com baixo deslocamento mesmo que tenham mudado.

Esse é o sinal de convergência do modelo confundido com mudança semântica.
O controle placebo (D_0 repetido) é obrigatório para separar os dois.

### Correções adicionais implementadas na GPU (2026-06-06)

**Bug de fronteiras de documento (crítico):** o corpus SemEval tem uma sentença
por linha, embaralhadas aleatoriamente. O leitor antigo concatenava o arquivo
inteiro como um único documento, criando janelas MLM que atravessavam fronteiras
entre sentenças não relacionadas. Todos os checkpoints anteriores aos pilotos
com o sufixo `_line_documents` são **inválidos** para avaliar o método.

**MLM dinâmico:** o dataset original mascarava deterministicamente o token
central de cada janela, sempre na mesma posição. O novo dataset aplica a
política BERT canônica: 15% dos tokens por época, com 80% `[MASK]`, 10% token
aleatório, 10% mantido. As máscaras variam por época mas são reproduzíveis.
Para `graft_nn`, isso aumentou as apresentações positivas de 4 para 269 em D0.

**Modelo maior:** o experimento `semeval2020_pmi_dynamic_mlm_12_8_d128` usou
d_model=128, 3 camadas, 40.188 passos de gradiente, 1h53m em GPU. Perdas:
D0: 6.94→4.88, D1: 5.54→5.02.

### Formulação Cloze-PMI descartada

A formulação log-PMI falha porque o MLM responde "qual token completa
sintaticamente esta posição", não "quais palavras são semanticamente próximas
de w". O PMI não converte substituibilidade posicional em proximidade semântica
ampla. Evidência empírica: mesmo com MLM dinâmico e modelo maior, a correlação
com variação de entropia permanece rho≈0.94.

**Cloze-PMI é encerrada como abordagem principal.**

---

## Resultados atuais — Perfis relacionais com APD de estados ocultos

A abordagem que produziu sinal positivo usa diretamente os estados ocultos:

```
r_t(w, ocorrência)[v] = cos(h_t(w, ocorrência), centroide_t(v))
APD(w) = distância média entre r_0(w, o_i) e r_1(w, o_j)
         para amostras aleatórias de ocorrências entre os dois períodos
```

Experimento `balanced_apd_layer2` (camada 2, centroides centrados, 3.216
referências compartilhadas, 100 ocorrências por período por palavra):

```
Spearman graded: 0.210
ROC-AUC:        0.542
```

### Problema central do APD: inversão plane_nn / chairman_nn

`chairman_nn` ocupa o rank 1 (falso positivo alto): o campo semântico de
liderança organizacional permanece estável, mas D1 tem representações mais
concentradas (menor variância), inflando o APD mesmo sem mudança de sentido.

`plane_nn` ocupa o rank 35 (falso negativo): a transição geométrico → transporte
é clara nos vizinhos, mas o APD absoluto é pequeno em relação à deriva de campo.

**As vizinhanças são semanticamente corretas; a escala do APD não discrimina.**

### Resultado qualitativo — vizinhanças temporalmente coerentes

Usando `r_t(w) = {v: cos(centroide_t(w), centroide_t(v))}` para as 3.216
referências compartilhadas entre checkpoints:

**`plane_nn` (transição forte):**
- D0: `line, angle, plate, column, stock, canal, building, coast, border, ridge`
- D1: `boat, ship, fence, rail, route, pole, building, road, flag, trail, machine`

**`chairman_nn` (campo estável):**
- D0: `secretary, editor, commander, director, president, committee, jury`
- D1: `secretary, director, commander, president, commissioner, governor, publisher`

**`graft_nn` (transição forte, vizinhança D0 heterogênea):**
- D0: campo heterogêneo (botânico + outros)
- D1: `compound, machinery, currency, commodity, mechanic, utility, acid, organ`

**`tree_nn` (reorganização interna, sem troca de campo):**
- D0 e D1 permanecem no campo natural (plantas, paisagem)

**Controle de campo:** subtraindo a mediana do JSD do campo semântico observado,
`chairman_nn` cai para resíduo mínimo (0.019) e `graft_nn` sobe para resíduo
alto (0.187). O controle de campo é promissor mas ainda não foi aplicado
sistematicamente a todos os 37 alvos.

### Realinhamento da contribuição (2026-06-06)

**O objetivo não é maximizar o Spearman de 37 palavras.**

O alvo é demonstrar que TimeFormer produz **vizinhanças semânticas temporais
coerentes** sem alinhamento geométrico post-hoc e sem anotação externa:

| Propriedade | Hamilton 2016 | APD+BERT (SemEval) | TimeFormer |
|---|---|---|---|
| Embeddings | estáticos | contextuais | contextuais |
| Modelos | 2 independentes | 1 fixo externo | 1 contínuo |
| Domínio | in-domain | out-of-domain | in-domain |
| Alinhamento | Procrustes | não necessário | não necessário |
| Resolução | 2 pontos | 2 pontos | N checkpoints |

A contribuição específica: um único modelo que *aprendeu a transição* — não
dois snapshots alinhados depois. A continuidade cronológica dos pesos produz
uma representação da transição que independe de Procrustes ou encoder externo.

### Experimentos encerrados (decisões finais)

- **Atlas WSD externo como arquitetura principal:** exigiria BEM/ConSeC como
  componente central. A contribuição passaria para o encoder externo.
- **Clustering como estimador de sentido:** muro de identificabilidade formal.
  O algoritmo encontra variância de tópico/registro, não de sentido lexical.
  Mais épocas ou algoritmos melhores não atravessam essa parede.

### Próximos experimentos necessários

1. **Comparação com Hamilton 2016 (prioritária):** word2vec por período +
   Procrustes, mesmo protocolo de relatório de vizinhança. Se word2vec produzir
   vizinhanças tão coerentes com menor custo, a novidade do treinamento contínuo
   precisa ser redefinida.

2. **Field-controlled APD para todos os 37 alvos:** definir campos semânticos
   automáticos por agrupamento de palavras de referência (sem usar clustering
   como estimador de sentido), calcular `APD_adj(w) = APD(w) - mediana(campo)`.

3. **Modelo maior (d=256/512) como ablação:** verificar se a limitação atual
   é de capacidade ou de arquitetura.

## Fase 1 / 1.5 do Perfil Relacional v2 (2026-06-11)

Após `docs/12-novo_perfil_relacional.md` (v2), rodamos a Fase 0A/1 (ablação de
centralização) e a Fase 1.5 (go/no-go espectral, §7-9) sobre os caches
existentes (`outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/hidden_relational_profiles/cache`),
sem reextração.

**Fase 1 (centralização):** quatro variantes de `mu_t` foram comparadas,
todas avaliadas sobre o mesmo `V_ativo` (n_min=10, ~5700-5900 tokens em
`mean_last_2`/`layer_2`). `D_type_uniform_mu` (média não ponderada dos
centróides por tipo sobre `V_ativo`) teve o melhor resultado:
spearman=0.124, AUC=0.601 (mean_last_2), marginalmente acima da abordagem
v1 (`A_reference_mean`, spearman=0.108). `B_global_mu` e
`C_global_mu_active_support` (médias ponderadas por ocorrência) tiveram
desempenho muito pior (spearman ~0, AUC ~0.5) -- dominadas por palavras de
função de alta frequência. Diferença A vs D não é significativa (n=37).
Decisão: não revisar §4.1 do documento canônico por agora; adotar D como
centralização de trabalho.

**Fase 1.5 (NO-GO espectral, §7-9):** testamos a decomposição em modos
semânticos (critério de gap + SVD da matriz de coesão) para
plane_nn/graft_nn (mudança esperada) vs. chairman_nn/tree_nn/ball_nn/
face_nn/lane_nn/multitude_nn (controles estáveis), em três formulações
sucessivas:

1. `filter_support` puro sobre `P_t(w)` (componentes positivos): tau~1e-4,
   `|V_w|` ~ 5000-5900 (quase todo o V_ativo positivo), k=1 quase sempre,
   com k=2 espúrio aparecendo em CONTROLES estáveis (face_nn, multitude_nn),
   não nos alvos de mudança.
2. `filter_support_topn` (top-N por `|P_t(w)[v]|`, N=100/500): tau=None,
   k=None para todas as 8 palavras -- nenhum gap relativo > gamma=0.3 entre
   os candidatos mais correlacionados.
3. Top-N **positivo** fixo (N=50/100/200), gap só sobre autovalores de
   M_t(w) (formulação recomendada por segunda opinião do codex): k=1 para
   TODAS as 8 palavras, ambos os períodos, todos os N -- lambda_1 domina
   lambda_2 por 10-30x sempre.

Diagnóstico: `P_t(w)[v]` decai suavemente e quase monotonicamente de ~0.95
a ~-0.76 ao longo de `V_ativo` (gaps relativos consecutivos ~0.003-0.05),
sem clusters discretos. A matriz de coesão M_t(w) é dominada por um único
modo (a "direção média" de V_w), igualmente para palavras com mudança
conhecida e para controles estáveis. Auditoria de implementação (codex)
confirmou fidelidade ao §7.5/§8.2/§8.3; o padrão não é atribuível a erro de
sinal/eixo/normalização.

**Conclusão:** §7-9 (modos semânticos via SVD da matriz de coesão) é um
NO-GO empírico para este regime (d_model=128, 3 camadas, |V_ativo|~11600,
SemEval eng_lemma). Não construir a infraestrutura de modos/persistência
(Fase 2-3-5 originais do plano v2). O deslocamento relacional
`Delta(w) = 1 - cos(P_t0(w), P_t1(w))` (Fase 1, variante D) permanece como
a métrica de trabalho, com a ressalva de que seu sinal já era fraco
(spearman~0.124, nenhum "changed" target acima do p95 dos estáveis).
Próximo passo proposto: ablação de modelo maior (d=256/512, ver item 3
acima) para avaliar se o colapso em modo único é um efeito de capacidade do
encoder, antes de investir em variações alternativas de V_w/perfil.

## Adendos (2026-06-12) -- diagnóstico do eixo de época e teto de oráculo

Quatro rodadas adicionais de testes baratos sobre o cache existente (sem
reextração), documentadas em detalhe em
`docs/14-perfil_relacional_v2_resultados_fase1.md` §7.9-7.25:

1. **Passo 0 (APD + bimodalidade)**: `APD` (distância par-a-par entre
   ocorrências, sem centróide) performa igual a `Delta` (spearman~0.13) --
   refuta a hipótese de que o problema era a agregação por centróide.
2. **APD_ratio + cluster x período (NMI)**: ambos no acaso contra
   `truth.tsv`. Mas o NMI por palavra revelou que, na configuração
   "diagonal" (`theta0_d0` vs `theta1_d1`), quase TODA palavra (mudada ou
   estável) é quase perfeitamente separável por período em `mean_last_2`.
3. **Grade 2x2 (checkpoint x corpus)**: identificou que essa separação é
   quase toda **drift de checkpoint** (theta0 -> theta1 via treino
   contínuo), não conteúdo do corpus -- com encoder fixo, NMI(corpus)~0.03;
   com dados fixos, NMI(checkpoint)~0.86. Recentralização aditiva remove a
   maior parte, mas não move o spearman de `APD`/`Delta`.
4. **Encoder fixo (Tarefa 1) + modos primeiro (Tarefa 3) + BERT congelado
   (Tarefa 2)**: medir com `theta1` fixo sobre os dois corpora sobe `APD`
   de ~0.13 para ~0.20 (ainda não significativo, n=37). Agrupar a nuvem de
   ocorrências antes do perfil (em vez de depois) produz resultados
   interpretáveis para `graft_nn`/`tree_nn`, mas não para `plane_nn`.
   **O resultado decisivo**: o mesmo `APD`/`NMI`, computado com
   `bert-base-uncased` PRÉ-TREINADO (congelado, mesmas frases), chega a
   spearman~0.59 (p=0.0001) -- estatisticamente significativo e muito
   acima de qualquer resultado com o encoder próprio. Para `plane_nn`, o
   BERT separa cleanly o sentido geométrico (1850) do sentido aviação
   (2000) (`NMI`=0.487 vs `NMI`~0 para `tree_nn`, estável).

**Conclusão revisada**: o gargalo principal é a **qualidade/capacidade do
encoder** (d_model=128, 3 camadas, treinado do zero só com MLM contínuo),
não o desenho do perfil relacional, a centralização, ou a configuração
diagonal -- embora essas três correções (encoder fixo, agrupar antes de
medir) ajudem (~0.13 -> ~0.20) e devam ser mantidas. A ablação de
capacidade "treinar maior do zero" (item 3 acima) é substituída por:
**inicializar o Timeformer a partir de um checkpoint pré-treinado antes do
treino contínuo temporal**, mantendo o resto do pipeline (perfil relacional
v2 + encoder fixo + agrupamento de ocorrências) como infraestrutura de
medição.

## Adendo (2026-06-12) -- resultado da inicialização `bert-tiny`

A inicialização foi implementada e o run completo 12+8 épocas foi
executado em `outputs/semeval2020_pmi_pretrained_init_d128/`.

O controle congelado mostrou que `bert-tiny` contém sinal: APD
`mean_last_4` spearman=0.399 (p=0.014). O checkpoint adaptado para o
vocabulário próprio, antes de qualquer MLM contínuo, ainda alcançou
spearman=0.337 (p=0.041) no melhor perfil oculto e APD fixo=0.277 em
`layer_2`. Porém, depois de 12 épocas em D0 e 8 em D1 a avaliação de
encoder fixo voltou ao teto anterior: APD=0.123 (`layer_2`) e 0.204
(`mean_last_2`), ambos não significativos.

Decisão: a transferência funciona parcialmente, mas o regime 12+8 com LR
`1e-4` causa esquecimento do sinal pré-treinado. O próximo experimento é
uma ablação conservadora de fine-tuning (LR `3e-5`/`5e-5`, menos épocas
e/ou congelamento inicial da camada inferior), com configuração escolhida
sem consultar `truth.tsv`. Não promover o run 12+8 atual a referência.

## Adendo (2026-06-13) -- Option D com BERT integral

A segunda opinião recomendou preservar integralmente tokenizer,
arquitetura, embedding LayerNorm, positional embeddings, MLM head e weight
tying do `prajjwal1/bert-tiny`. O pipeline foi implementado e executado com
2 seeds, 3 épocas em D0 + 2 em D1, LR `3e-5`, warmup/linear decay,
validação por documento e seleção de checkpoint sem gold.

Resultado alinhado:

| Readout | init | theta1 seed 1000 | theta1 seed 1001 |
|---|---:|---:|---:|
| layer 1 | 0.298 | 0.325 | 0.322 |
| layer 2 | 0.136 | 0.030 | 0.038 |
| média layers | 0.241 | 0.189 | 0.194 |

O pseudo-período aleatório produziu layer 1=0.332 e layer 2=0.153.
Congelar embeddings + layer 1 preservou layer 1=0.298, mas layer 2 caiu
para 0.017. Portanto, não há esquecimento global: o sinal é preservado e
ligeiramente melhorado na primeira camada, enquanto a camada superior é
reorganizada pelo MLM, com perda adicional específica da cronologia.

Decisão: BERT integral substitui a adaptação `RealStaticMLM` como base de
trabalho. Não usar apenas a última camada. O próximo teste prioritário é
regularizar a camada 2 contra `theta_init` (L2-SP ou distillation), mantendo
layer 1 treinável com LR menor, e repetir o controle de pseudo-período.
Detalhes em `tmp/32-codex_option_d_execution_results.md`.

## Adendo (2026-06-13) -- LR discriminativa e L2-SP

O teste prioritário foi executado com LR `1e-5` para embeddings/layer 1,
LR `3e-5` para layer 2/head e L2-SP normalizado (`lambda=10`) apenas na
layer 2. Também foi executada a ablação idêntica com `lambda=0`.

| Condição | layer 1 | layer 2 | média layers |
|---|---:|---:|---:|
| full anterior | 0.325 | 0.030 | 0.189 |
| LR discriminativa | 0.340 | 0.012 | 0.196 |
| LR discriminativa + L2-SP | 0.338 | 0.014 | 0.204 |
| L2-SP, pseudo-período | 0.341 | 0.116 | 0.258 |

L2-SP reduziu o deslocamento relativo da layer 2 de `0.0416` para
`0.0294`, mas não preservou seu ranking semântico. A melhora de layer 1
vem da LR discriminativa, não da regularização. Isso mostra que proximidade
de pesos não garante proximidade funcional neste regime.

Decisão: manter LR discriminativa; encerrar L2-SP como solução principal;
o próximo teste é distillation da representação da layer 2 em
frases-âncora, com controle pseudo-período e uma segunda seed apenas se o
efeito aparecer na primeira. Não abrir uma grade de `lambda` contra os 37
alvos. Esta decisão foi superada pelo adendo seguinte após bootstrap e
alinhamento do orçamento de treino.

## Adendo (2026-06-13) -- bootstrap e correção do orçamento pseudo

Uma segunda opinião identificou duas limitações: incerteza de `n=37` e a
comparação entre `D1@2` cronológico e `pseudo-D1@0,5`. Executamos bootstrap
pareado por palavra (20.000 réplicas) e avaliamos os checkpoints de D1 em
orçamentos iguais.

Nenhuma melhora de layer 1 contra `init` apresentou IC pareado que
excluísse zero. Para layer 2, todos os ICs individuais incluíram zero.

| Épocas D1 | cronológico | pseudo | IC 95% crono-pseudo |
|---:|---:|---:|---:|
| 0,25 | 0.012 | 0.046 | [-0.132; 0.062] |
| 0,5 | 0.062 | 0.153 | [-0.238; 0.048] |
| 1 | 0.059 | 0.176 | [-0.278; 0.031] |
| 2 | 0.030 | 0.088 | [-0.170; 0.044] |

O pseudo permanece numericamente acima, mas nenhuma diferença alinhada
exclui zero. Portanto, “redistribuição entre camadas” e “efeito específico
da cronologia” deixam de ser conclusões e passam a padrões exploratórios.

Decisão revisada: não implementar distillation agora. Pausar a otimização
do MLM temporal e avançar para a Porta 1 de WSD externo congelado,
testando compatibilidade contexto-gloss nos subconjuntos heurísticos
predefinidos de `plane`, sem ajuste no SemEval.

## Adendo (2026-06-13) -- Porta 1 com LMMS-SP congelado

A Porta 1 foi implementada com LMMS-SP WSD, `bert-large-cased` congelado,
WordNet 3.0 e os pesos de camadas publicados, sem ajuste no SemEval.

| Subconjunto | N | Acurácia | IC 95% | Decisão |
|---|---:|---:|---:|---|
| D0 geometria | 182 | 0.984 | [0.962; 1.000] | passou |
| D0 ferramenta | 19 | 0.211 | [0.053; 0.421] | falhou |
| D1 avião | 208 | 1.000 | [1.000; 1.000] | passou |

A acurácia macro foi `0.731`, IC 95% `[0.677; 0.800]`, acima do baseline
de `1/3`. A frase `plate figure represent an inclined plane` permaneceu
geométrica sob a régua fixa.

Decisão: `NO-GO` estrito porque ferramenta histórica não superou o acaso.
Não escalar ainda para os 37 alvos. O próximo passo é adjudicação cega das
19 ocorrências de ferramenta, seguida de uma única replicação com outro
WSD externo congelado, preferencialmente ConSeC. Se ambos falharem nos
casos adjudicados, encerrar o atlas WordNet geral ou reformulá-lo com
inventário/adaptação histórica. Detalhes em
`docs/15-external_wsd_plane_gate1_results.md`.

## Adendo (2026-06-13) -- Gate 1 completo com ConSeC

Após adjudicação humana do subconjunto de ferramenta, executamos o checkpoint
oficial ConSeC SemCor+WNGT, congelado, nas 409 ocorrências dos três estratos
originais.

| Estrato pós-adjudicação | N | Acurácia | IC 95% |
|---|---:|---:|---:|
| D0 geometria | 183 | 0,995 | [0,984; 1,000] |
| D0 ferramenta | 16 | 0,875 | [0,688; 1,000] |
| D1 aviação | 208 | 0,990 | [0,976; 1,000] |

A macro acurácia foi `0,953`, IC 95% `[0,891; 0,997]`. Todos os checks
originais passaram, incluindo a frase-âncora geométrica.

Decisão: `Gate 1 ConSeC = GO`. Abrir uma Porta 2 pequena com `graft_nn`,
`chairman_nn` e `tree_nn`, congelando inventários, subconjuntos e cortes antes
da execução. Não escalar diretamente para os 37 alvos. Detalhes em
`docs/19-consec_plane_gate1_full_results.md`.

## Adendo (2026-06-13) -- Pré-registro da Porta 2

Os inventários e subconjuntos foram congelados antes da inferência. A parte
discriminativa usa `graft` corrupção versus médico e `tree` planta versus
diagrama. `chairman` é monossêmico e funciona apenas como controle de
cobertura. O uso botânico histórico de `graft` como scion foi registrado como
lacuna de inventário.

Todos os mínimos foram satisfeitos. Uma ficha cega com 87 contextos foi
gerada em `annotations/consec_gate2_audit/`. O ConSeC só deve ser executado
depois dessa auditoria. Critérios completos em
`docs/20-consec_gate2_preregistration.md`.

## Adendo (2026-06-13) -- Resultado da Porta 2

A auditoria humana foi concluída antes da inferência e concordou com 84 de 87
rótulos heurísticos. As três discordâncias foram preservadas sem correção
retroativa.

O ConSeC congelado obteve, depois da auditoria:

| Estrato | N | Acurácia | IC 95% |
|---|---:|---:|---:|
| `graft/corruption` | 22 | 1,000 | [1,000; 1,000] |
| `graft/medical` | 17 | 1,000 | [1,000; 1,000] |
| `tree/diagram` | 16 | 0,875 | [0,688; 1,000] |
| `tree/plant` | 138 | 0,986 | [0,964; 1,000] |

A macro foi `0,965`, IC 95% `[0,918; 0,998]`. Todos os cortes
pré-registrados passaram: `Porta 2 ConSeC = GO`.

O controle `chairman` foi coberto, mas permanece tautológico por ser
monossêmico. Nos 90 usos botânicos de `graft`, o inventário não contém
adequadamente o objeto enxertado; as previsões se dividiram entre ato de
enxertar (54), enxerto médico (35) e corrupção (1). Portanto, o próximo
gargalo é cobertura lexical, não a discriminação contexto-glossa.

Decisão: antes de escalar, construir uma matriz de cobertura WordNet dos 37
alvos e abrir uma Porta 3 somente para palavras e sentidos com inventário
adequado e ocorrências auditáveis. Detalhes em
`docs/21-consec_gate2_results.md`.

## Adendo (2026-06-13) -- Matriz de cobertura para a Porta 3

A matriz dos 37 alvos foi preparada antes de novas previsões. Ela contém 229
sensekeys WordNet 3.0 e 296 contextos históricos, com quatro amostras
determinísticas por alvo e período.

Distribuição da carga de revisão:

```text
3 palavras monossêmicas
13 com 2-3 sentidos
10 com 4-8 sentidos
11 com 9 ou mais sentidos
```

As decisões já estabelecidas para `plane`, `tree`, `graft` e `chairman` foram
transportadas explicitamente. As outras 33 palavras aguardam auditoria em
`annotations/wordnet_coverage_gate3/coverage_review.csv`.

Nenhum gold do SemEval ou previsão do ConSeC foi incluído. A Porta 3 só será
desenhada depois de congelar `coverage_status` e `gate3_decision` para todas
as palavras. Protocolo em `docs/22-wordnet_coverage_gate3_protocol.md`.

## Adendo (2026-06-13) -- Resultado da Porta 3

A revisão classificou 25 alvos polissêmicos como cobertura suficiente, nove
como cobertura parcial e três como controles monossêmicos. O conjunto
confirmatório foi congelado antes da inferência.

Com 25 ocorrências balanceadas por período e palavra, a JSD entre
distribuições de sentido do ConSeC obteve:

```text
Spearman = 0,586
IC 95% = [0,231; 0,818]
p por permutação = 0,0028
ROC-AUC = 0,714
```

Nos 21 alvos de alta confiança, Spearman foi `0,600`, com `p=0,0055`.
Portanto, `Porta 3 = GO`.

`plane` apresentou a maior JSD (`0,334`), `tree` ficou em `0,0135` e os
controles monossêmicos ficaram em zero. `graft`, diagnóstico, apresentou
`0,164`, com crescimento forte de corrupção em D1.

O principal risco novo é o tamanho do inventário:

```text
rho(JSD, número de sentidos) = 0,533
```

Uma correlação parcial exploratória ainda foi positiva (`0,454`, `p=0,023`),
mas precisa ser pré-registrada. Próximo passo: replicar com novas seeds,
estimar incerteza por ocorrência e calibrar JSD pelo número de candidatos.
Detalhes em `docs/24-consec_gate3_results.md`.

## Adendo (2026-06-14) -- Replicação da Porta 3

Duas novas amostras foram executadas com seeds `20260614` e `20260615`.

| Seed | Spearman | Spearman parcial por nº sentidos |
|---|---:|---:|
| 20260613 | 0,586 | 0,454 |
| 20260614 | 0,549 | 0,402 |
| 20260615 | 0,621 | 0,522 |

A média bruta foi `0,585 ± 0,036`; a média parcial, `0,459 ± 0,060`. A
permutação conjunta produziu `p=0,00115`. Os rankings de JSD entre seeds
correlacionaram de `0,808` a `0,885`.

Todos os critérios pré-registrados passaram. A associação com o gold não
depende de uma única amostra e não é explicada apenas pelo número de
candidatos WordNet.

Decisão: a próxima etapa é um nulo dentro de cada palavra, permutando período
entre suas ocorrências, para produzir JSD excedente e incerteza individual.
Detalhes em `docs/26-consec_gate3_replication_results.md`.

## Adendo (2026-06-14) -- Nulo intrapalavra

Os rótulos temporais das 50 ocorrências de cada palavra foram permutados 20.000
vezes em cada uma das três seeds.

| Score | Spearman médio com gold | |Spearman| médio com nº sentidos |
|---|---:|---:|
| JSD bruta | 0,585 | 0,509 |
| JSD excedente | 0,410 | 0,165 |
| z nulo | 0,319 | 0,034 |

A JSD excedente permaneceu positiva nas três seeds e reduziu fortemente a
associação com o tamanho do inventário. Todos os critérios pré-registrados
passaram.

Decisão: manter JSD bruta como ranking principal, usar JSD excedente como
controle confirmatório e z nulo como diagnóstico por palavra. O próximo passo
é comparar essas medidas externas com `layer 1` e com a resposta adaptativa
`layer 2 - layer 1` do TimeFormer, sem selecionar configuração pelo gold.
Detalhes em `docs/28-consec_within_word_null_results.md`.

## Adendo (2026-06-14) -- Integração ConSeC-TimeFormer

A validade convergente entre scores agregados por palavra foi pré-registrada e
testada nos 25 alvos confirmatórios:

| Comparação | Spearman | p permutação |
|---|---:|---:|
| `layer_1` × JSD bruta | -0,031 | 0,884 |
| `layer_1` × JSD excedente | -0,024 | 0,908 |

Os controles por seed, gold, número de sentidos, cobertura e LR não alteraram
a conclusão. `Validade convergente escalar = NO-GO`.

No subconjunto confirmatório, ConSeC continuou associado ao gold
(`rho=0,604` bruto), enquanto `layer_1` ficou em `rho=0,076`. APD contextual
e mudança explícita da mistura de sentidos não são medidas intercambiáveis.

Decisão: não interpretar APD como substituto direto de mudança lexical. O
próximo passo é uma análise no nível da ocorrência, aplicando ConSeC e
TimeFormer aos mesmos contextos para testar alinhamento entre posterior de
sentido e geometria contextual, além de separar mudança de mistura e deriva
dentro do sentido. Detalhes em
`docs/30-consec_timeformer_integration_results.md`.

## Adendo (2026-06-14) -- Alinhamento no nível da ocorrência

Foram extraídos vetores TimeFormer para 3.383 ocorrências confirmatórias
únicas da Porta 3. Dentro de cada palavra, a distância cosseno foi comparada à
JSD entre posteriores ConSeC, controlando se o par atravessava períodos.

```text
layer_1: rho médio = 0,062
IC 95% = [0,047; 0,078]
23/25 palavras positivas
p por inversão de sinais = 0,00005
```

Todos os critérios pré-registrados passaram. O efeito também apareceu usando
apenas pares dentro do mesmo período (`rho=0,074`), descartando a explicação
de um eixo temporal compartilhado.

Exploratoriamente, `layer_2` apresentou alinhamento local mais forte
(`rho=0,187`, 25/25 palavras positivas), embora sua APD temporal agregada
continue fraca. Isso mostra que conter estrutura local de sentidos e produzir
um bom ranking temporal por APD são propriedades diferentes.

Após controlar diferença semântica, permaneceu pequena separação geométrica
entre períodos. Decisão: decompor o deslocamento vetorial em mudança da
composição dos sentidos e deriva interna aos sentidos, usando pesos suaves dos
posteriores. Detalhes em
`docs/32-occurrence_level_consec_timeformer_results.md`.

## Adendo (2026-06-14) -- Decomposição suave por sentidos

O deslocamento do centróide temporal foi decomposto simetricamente em mudança
da composição dos sentidos e deriva dos centróides condicionais. Um nulo
embaralhou posteriores entre vetores dentro de cada palavra e período,
preservando as marginais.

Na análise principal (`layer_2`):

```text
excesso médio da share de composição = 0,048
IC 95% = [0,024; 0,081]
23/25 palavras positivas
p por inversão de sinais = 0,00005
rho(excesso, JSD ConSeC) = 0,615
```

Todos os critérios pré-registrados passaram. `plane_nn` apresentou a maior
contribuição de composição (`0,341`). A componente complementar,
aproximadamente `0,952`, não tem sua natureza identificada por esta análise e
não deve receber automaticamente um rótulo semântico ou contextual.

Decisão: a troca de mistura de sentidos explica uma parcela pequena, mas
mensurável e replicada, do deslocamento vetorial. O próximo passo é bootstrap
estratificado por ocorrência para obter incerteza por palavra e auditar os
casos extremos. Detalhes em
`docs/34-soft_sense_vector_decomposition_results.md`.

## Adendo (2026-06-14) -- Bootstrap da decomposição

Foram executadas 2.000 reamostragens estratificadas por palavra e período,
preservando o pareamento entre as duas seeds TimeFormer.

```text
layer_2:
mediana bootstrap da média = 0,042
IC 95% = [0,032; 0,053]
rho observado × mediana bootstrap = 0,919
```

A conclusão agregada permaneceu estável e todos os critérios pré-registrados
passaram. Dez palavras tiveram IC individual inteiramente positivo:

```text
plane, multitude, gas, record, land,
attack, bit, thump, risk, fiction
```

Nenhuma palavra foi robustamente negativa. `player`, `donkey` e `stab`
permaneceram incertas.

A auditoria por sentido confirmou `plane` como o exemplo mais transparente.
`multitude` é robusta, mas requer inspeção textual por causa da rotulagem pouco
intuitiva de seu synset WordNet.

Decisão: encerrar a busca aberta por novas métricas. Os próximos passos são
auditar os exemplos destinados ao artigo e consolidar método, resultados,
tabelas e figuras reproduzíveis. Detalhes em
`docs/36-soft_decomposition_bootstrap_results.md`.

## Adendo (2026-06-14) -- Consolidação para o artigo

Foi criado um gerador reproduzível com três figuras, três tabelas e oito
contextos auditados. Os exemplos `plane`, `multitude`, `gas` e `record`
confirmam que os casos individuais robustos possuem interpretações textuais
concretas, embora `multitude` exija apresentar a definição do sentido em vez
do nome pouco intuitivo do synset.

Os artefatos estão em:

```text
outputs/paper_assets/consec_timeformer/
```

Decisão: a linha de validação por sentidos está consolidada, mas o eixo
principal `token@time` ainda deve ser apresentado diretamente por vizinhanças
temporais, relações ganhas e perdidas e estabilidade entre seeds. O pacote
ConSeC será uma análise secundária de validade semântica. Síntese e limites
estão em `docs/37-consec_timeformer_article_package.md`.

## Adendo (2026-06-14) -- Hierarquia científica corrigida

O objetivo principal do TimeFormer não é inferir automaticamente sentidos
WordNet. É produzir perfis relacionais temporalmente consultáveis:

```text
w@t -> vizinhos e relações de w no período t
```

Esses vizinhos são o resultado. ConSeC responde a uma pergunta posterior:
quanto do deslocamento relacional coincide com recomposição de sentidos de um
inventário externo?

A parcela associada a sentidos foi estabelecida. A natureza da componente
complementar permanece indeterminada pelo método e pode ser investigada por
especialistas a partir dos vizinhos, contextos e fontes históricas.

Próximo passo: relatório reproduzível de `token@time` no melhor encoder,
mantendo a análise ConSeC como camada interpretativa, não como filtro das
vizinhanças. O posicionamento canônico está em
`docs/38-scientific_positioning_token_time_and_sense_analysis.md`.

## Adendo (2026-06-14) -- Framework de análise `token@time`

O planejamento operacional foi consolidado em seis capacidades:

```text
consultar token@time
comparar dois token@time
descrever uma trajetória
comparar trajetórias
encontrar trajetórias semelhantes
rankear quem mais e menos mudou
```

O SemEval possui somente D0 e D1. Portanto, nesta base podemos comparar
deslocamentos, direções e turnover de vizinhança, mas não inferir forma
multitemporal, reversão ou oscilação. Essas propriedades exigem três ou mais
períodos.

As primitivas de relações e métricas estruturais já existem, mas faltam uma
API unificada, busca palavra-palavra, incerteza integrada e aplicação ao
`bert-tiny` completo. A ordem definida é:

1. perfis e vizinhanças D0/D1 no melhor encoder;
2. ganhos, perdas, estabilidade e rankings separados;
3. busca por deslocamentos semelhantes;
4. corpus com 3+ períodos;
5. comparação e busca de trajetórias completas.

Especificação: `docs/39-token_time_analysis_framework.md`.
