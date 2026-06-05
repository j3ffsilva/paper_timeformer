# Pre-registro: forma temporal do deslocamento semântico relacional

## Status e regra de decisão

Este documento fixa as hipóteses, condições, métricas e critérios de decisão do
próximo experimento antes de implementar o novo gerador ou observar seus
resultados.

O experimento não assume que mudanças pequenas são irrelevantes. Ele testa se o
treinamento contínuo registra diferenças entre:

- mudança pequena acumulada;
- mudança abrupta persistente;
- perturbação transitória;
- movimento intenso, mas incoerente;
- ausência de mudança semântica.

Os resultados não serão usados para redefinir retrospectivamente as condições
ou os critérios abaixo. Alterações necessárias após o primeiro teste de
sanidade serão documentadas como desvios deste pre-registro antes das execuções
confirmatórias.

## Pergunta central

Um Transformer padrão treinado continuamente e em ordem cronológica registra a
**forma temporal** da mudança semântica, ou registra apenas a magnitude das
diferenças entre checkpoints?

O objeto de estudo é a arquitetura contínua:

```text
theta_0 = treino(D_0)
theta_t = continua_treino(theta_(t-1), D_t)
```

Nenhum identificador temporal é fornecido ao modelo. O deslocamento é medido
posteriormente por perfis relacionais:

```text
delta_rel(w, a, b) = r_b(w) - r_a(w)
```

O vetor `delta_rel` permanece como produto descritivo principal. As métricas
temporais não o substituem; elas descrevem a forma de sua trajetória.

## Hipóteses pré-registradas

### H1: acumulação gradual

Pequenos deslocamentos consecutivos, coerentes e persistentes devem produzir
uma mudança acumulada `t0 -> t9` detectável e direcionalmente correta.

### H2: equivalência de destino

Trajetórias graduais e abruptas com os mesmos estados inicial e final devem
apresentar deslocamentos acumulados `t0 -> t9` semelhantes, mas formas
temporais diferentes.

### H3: reversão

Uma perturbação transitória deve apresentar grande deslocamento intermediário,
seguido de retorno relacional próximo ao estado inicial.

### H4: incoerência temporal

Uma trajetória oscilatória pode apresentar grande comprimento total de caminho,
mas não deve ser interpretada como mudança final persistente quando retorna
próxima ao estado inicial.

### H5: valor da continuidade cronológica

O treinamento contínuo cronológico deve recuperar melhor a forma temporal
plantada que baselines sem continuidade cronológica.

## O que não será afirmado neste estágio

Ainda não definiremos uma classificação definitiva
`estável/transição/reorganização estrutural`.

Também não combinaremos magnitude, direção, persistência e coerência em uma
única pontuação ajustada após observar resultados. Cada propriedade será
avaliada separadamente.

Baixa detecção de uma condição será tratada como limitação ou questão aberta,
não como evidência automática de irrelevância semântica.

## Benchmark sintético temporal

### Estrutura geral

- `10` períodos: `t0` a `t9`;
- `40` sujeitos: `16` âncoras estáveis e `24` palavras-alvo;
- `6` quartetos pareados entre as palavras-alvo;
- os perfis confirmatórios das palavras-alvo são calculados contra as mesmas
  `16` âncoras estáveis;
- cada quarteto compartilha os mesmos estados semânticos inicial e alternativo;
- dentro de cada quarteto, cada sujeito recebe uma forma temporal diferente;
- metade dos quartetos se desloca de `N1` para `N2`;
- metade se desloca de `N2` para `N1`;
- os valores exatos dos extremos são amostrados uma vez por quarteto e
  registrados no arquivo de trajetórias;
- todas as condições usam a mesma fidelidade, número de exemplos e orçamento
  de atualizações.

O pareamento evita que uma condição seja favorecida apenas por possuir estados
iniciais ou finais mais fáceis.

As âncoras são necessárias para isolar a forma temporal da palavra-alvo. Se
todos os sujeitos em um perfil se moverem simultaneamente, uma palavra abrupta
também apresenta mudanças graduais em suas relações com palavras graduais. O
perfil completo entre todas as palavras continua cientificamente válido para
mudança relacional geral, mas não isola a trajetória própria necessária neste
experimento de falsificação.

### Condições principais

Para ilustração, considere um quarteto com estado inicial `0.9` e estado
alternativo `0.1`.

#### G: gradual persistente

Pequenos passos na mesma direção até alcançar o estado alternativo:

```text
[0.90, 0.81, 0.72, 0.63, 0.54, 0.46, 0.37, 0.28, 0.19, 0.10]
```

#### A: abrupta persistente

Permanece inicialmente estável, sofre um salto e mantém o novo estado:

```text
[0.90, 0.90, 0.90, 0.90, 0.90, 0.10, 0.10, 0.10, 0.10, 0.10]
```

#### T: transitória reversível

Sofre uma mudança forte, permanece brevemente no estado alternativo e retorna:

```text
[0.90, 0.90, 0.90, 0.90, 0.10, 0.10, 0.90, 0.90, 0.90, 0.90]
```

#### O: oscilatória

Executa mudanças grandes e repetidas, mas termina no estado inicial:

```text
[0.90, 0.10, 0.90, 0.10, 0.90, 0.10, 0.90, 0.10, 0.90, 0.90]
```

### Controle nulo ressampleado

O nulo recebe novos textos em todos os períodos, mas mantém constante a
distribuição semântica de cada sujeito:

```text
[p0, p0, p0, p0, p0, p0, p0, p0, p0, p0]
```

Esse controle calibra variação causada por amostragem e treinamento sem mudança
semântica plantada.

### Controle placebo repetido

O placebo continua treinando repetidamente sobre o mesmo `D0`. Ele permanece
como diagnóstico de deriva de otimização, não como distribuição nula semântica
principal.

## Contrastes pré-registrados

### Contraste C1: gradual versus abrupta

`G` e `A` compartilham estados inicial e final.

Esperado:

- magnitude e direção acumuladas `t0 -> t9` semelhantes;
- `G` distribui mudança ao longo dos períodos;
- `A` concentra mudança ao redor do ponto de ruptura.

Esse contraste verifica se as métricas temporais distinguem caminhos com o
mesmo destino.

### Contraste C2: transitória versus nulo

`T` e o nulo terminam próximos ao estado inicial.

Esperado:

- ambos apresentam pequena mudança final `t0 -> t9`;
- `T` apresenta grande mudança intermediária e posterior reversão;
- o nulo não apresenta evento intermediário consistente.

Esse contraste verifica se a trajetória preserva eventos que desapareceriam em
uma comparação apenas entre início e fim.

### Contraste C3: oscilatória versus gradual

`O` pode possuir comprimento de caminho maior que `G`, mas termina próxima ao
estado inicial.

Esperado:

- `O` possui grande atividade relacional total;
- `G` possui maior deslocamento final persistente;
- uma medida baseada apenas na soma das magnitudes confundiria as condições.

### Contraste C4: real versus nulo ressampleado

Cada condição deve ser comparada com o nulo usando a mesma configuração
computacional.

Esperado:

- o nulo mantém a taxa nominal de falsos positivos;
- sinais temporais das condições reais superam o nulo nas métricas específicas
  correspondentes, não necessariamente em todas as métricas.

## Representação principal

A análise confirmatória usará:

```text
q_t(w) = P_t(contextos | [CLS] w [MASK] [MASK] [SEP])
r_t(w)[v] = 1 - JS(q_t(w), q_t(v)) / log(2)
```

Jensen-Shannon sobre distribuições preditas permanece como representação
principal. Geometrias ocultas serão apenas ablações posteriores.

## Métricas pré-registradas

Todas as métricas abaixo serão calculadas por palavra-alvo sobre seu perfil
relacional contra as âncoras estáveis. Não há diagonal nesse perfil retangular.

### 1. Vetor de deslocamento relacional

```text
delta_t0_t(w) = r_t(w) - r_0(w)
delta_step_t(w) = r_t(w) - r_(t-1)(w)
```

Os vetores serão preservados nos resultados para interpretação posterior.

### 2. Magnitude acumulada

```text
M_final(w) = mean(abs(delta_t0_t9(w)))
```

Mede a diferença entre início e fim.

### 3. Comprimento do caminho

```text
L(w) = sum_t mean(abs(delta_step_t(w)))
```

Mede toda a atividade relacional, incluindo oscilações e reversões.

### 4. Razão de eficiência do deslocamento

```text
E(w) = M_final(w) / max(L(w), epsilon)
```

Interpretação:

- valor alto: o caminho produz deslocamento final persistente;
- valor baixo: muito movimento com pouco deslocamento final.

Esta métrica não será chamada de significância estrutural; ela mede apenas a
eficiência relacional da trajetória.

### 5. Fidelidade direcional por período

```text
F_step(w) = mean_t cosine(delta_step_observado_t(w),
                          delta_step_oraculo_t(w))
```

Mede se cada passo observado acompanha a direção temporal plantada.

Passos cujo delta-oráculo seja exatamente zero serão excluídos da média e
contabilizados separadamente como períodos estáveis.

### 6. Fidelidade da trajetória acumulada

```text
F_acc(w) = mean_t cosine(delta_t0_t_observado(w),
                         delta_t0_t_oraculo(w))
```

Mede se a sequência acumulada aponta para as direções corretas ao longo do
tempo.

### 6.1 Vantagem direcional sobre placebo

Como o placebo repetido já apresentou direção positiva em experimentos
anteriores, também calcularemos:

```text
F_step_adv(w) = F_step_observado(w) - F_step_placebo(w)
F_acc_adv(w) = F_acc_observado(w) - F_acc_placebo(w)
```

Uma direção absoluta positiva não será suficiente para sustentar recuperação
temporal se a vantagem sobre placebo não for positiva.

### 7. Erro da forma temporal

Para cada sujeito, normalizamos a sequência de magnitudes acumuladas pelo maior
valor da própria sequência:

```text
shape_obs_t = M_t0_t_observado / max_t(M_t0_t_observado)
shape_oracle_t = M_t0_t_oraculo / max_t(M_t0_t_oraculo)

ShapeError(w) = mean_t(abs(shape_obs_t - shape_oracle_t))
```

Mede a forma da trajetória sem exigir igualdade da escala absoluta.

### 8. Recuperação após reversão

```text
Recovery(w) = 1 - M_final(w) / max_t(M_t0_t(w), epsilon)
```

Interpretação:

- próximo de `1`: retorno forte ao perfil inicial;
- próximo de `0`: deslocamento máximo permanece no final.

Será aplicada principalmente às condições `T` e `O`.

### 9. Taxa de falsa mudança no nulo

O percentil 95 do nulo será calculado somente para métricas cuja hipótese exija
um teste de magnitude. A taxa de sujeitos nulos acima desse limiar deverá
permanecer próxima de `5%` por construção.

Não usaremos `p0` oculto ou a classe sintética para calibrar o limiar
confirmatório. Calibrações condicionais por entropia, frequência ou incerteza
observável serão tratadas como experimento posterior.

## Critérios confirmatórios

Os critérios abaixo serão avaliados sobre as seeds confirmatórias agregadas.
Não serão redefinidos após observar os resultados.

### Q1: o Timeformer registra acumulação gradual?

H1 será sustentada se, para `G`:

1. as medianas de `F_acc` e `F_acc_adv` forem positivas;
2. a mediana de `M_final` superar o percentil 95 do nulo;
3. `ShapeError` for menor que o erro obtido ao comparar `G` com a forma
   abrupta-oráculo.

Falha em qualquer item mantém aberta a hipótese de que o método não possui
resolução suficiente para mudança gradual.

### Q2: distingue caminhos com o mesmo destino?

H2 será sustentada se:

1. `G` e `A` apresentarem medianas positivas de `F_acc_adv`;
2. a diferença absoluta de suas medianas de `M_final` não exceder `25%` da
   maior mediana;
3. cada condição apresentar `ShapeError` menor para seu próprio oráculo que
   para o oráculo da outra condição.

### Q3: registra reversão?

H3 será sustentada se:

1. `T` apresentar magnitude intermediária acima do p95 nulo;
2. a mediana de `Recovery(T)` for maior que `Recovery(A)`;
3. o pico temporal observado de `T` ocorrer no mesmo período do pico-oráculo,
   com tolerância de um período.

### Q4: distingue atividade de deslocamento persistente?

H4 será sustentada se:

1. `L(O) > L(G)`;
2. `M_final(O) < M_final(G)`;
3. `E(O) < E(G)`;
4. `Recovery(O) > Recovery(G)`.

### Regra geral de interpretação

- Sustentar H1-H4 indica que os checkpoints registram aspectos da forma
  temporal, e não somente magnitude final.
- Detectar apenas `M_final` sustenta deslocamento relacional entre checkpoints,
  mas não estrutura temporal.
- Falhar em detectar `G` enquanto detecta `A` indica limitação de resolução
  para mudanças graduais.
- Detectar `O` como grande mudança final indica confusão entre atividade e
  deslocamento persistente.
- Nenhum resultado autoriza declarar mudanças pequenas irrelevantes.

## Regimes e baselines

### Experimento P: mínimo decisivo

Executar primeiro:

1. `continual_chronological`;
2. `resampled_null`;
3. `continual_placebo`;
4. `frozen`.

O Experimento P verifica se o pipeline atual consegue registrar as quatro
formas temporais.

### Experimento A: valor arquitetural da continuidade

Somente após o teste de sanidade do Experimento P, comparar:

1. `continual_chronological`: treinamento contínuo na ordem correta;
2. `independent_period`: um modelo independente por período, com inicialização
   controlada;
3. `cumulative_retrain`: um modelo treinado do zero em `D0 + ... + Dt` para
   cada checkpoint;
4. `joint_all_periods`: um modelo único treinado com todos os períodos
   misturados, sem sequência temporal.

`joint_all_periods` não produz trajetória por si só e funciona como controle de
ausência de temporalidade. `independent_period` testa se os perfis relacionais
já bastam sem continuidade. `cumulative_retrain` testa se memória dos dados
anteriores, e não continuidade dos pesos, explica os resultados.

O valor específico da arquitetura contínua será sustentado apenas se ela
recuperar a forma temporal melhor que `independent_period` e
`cumulative_retrain`, sob orçamento computacional documentado e comparável.

H5 não será julgada pelo Experimento P. Antes de executar o Experimento A,
escreveremos um adendo a este pre-registro fixando:

- como igualar ou contabilizar o orçamento entre regimes;
- quais métricas serão primárias;
- qual contraste estatístico definirá superioridade arquitetural;
- como tratar a ausência de trajetória em `joint_all_periods`.

## Adendo do Experimento A: valor arquitetural da continuidade

Este adendo foi escrito após a execução confirmatória do Experimento P e antes
da implementação dos baselines arquiteturais.

### Motivação

O Experimento P mostrou que o regime contínuo registra deslocamento persistente,
acumulação gradual, reversão e atividade temporal sem deslocamento final. Também
mostrou uma limitação: rupturas abruptas tendem a aparecer suavizadas.

O Experimento A testa se esses resultados dependem da arquitetura contínua ou
se perfis relacionais extraídos de modelos sem continuidade cronológica já
seriam suficientes.

### Hipótese H5 operacional

H5 será sustentada apenas se `continual_chronological` recuperar a forma
temporal melhor que baselines sem continuidade cronológica, sob orçamento
documentado.

O primeiro contraste confirmatório será:

```text
continual_chronological vs independent_period
```

`independent_period` treina um modelo separado para cada período `t`, usando
somente `D_t`. Cada modelo é inicializado do zero com semente controlada. Não há
memória dos períodos anteriores.

### Por que este baseline vem primeiro

`independent_period` responde a pergunta mínima:

> A continuidade dos pesos é necessária para recuperar a forma temporal, ou os
> perfis preditivos por período bastam?

Se `independent_period` empatar ou superar `continual_chronological`, a tese
arquitetural do Timeformer enfraquece. Ainda poderemos ter uma boa metodologia
relacional, mas não uma evidência forte de que o treinamento contínuo seja
essencial.

Se `continual_chronological` superar `independent_period`, haverá evidência de
que continuidade cronológica ajuda a registrar trajetória.

### Orçamento

O orçamento não será forçado a ser numericamente idêntico ao do regime contínuo,
porque os regimes respondem perguntas diferentes:

- `continual_chronological` usa uma sequência única de atualizações;
- `independent_period` treina `n_periods` modelos separados.

Em vez de igualar artificialmente os passos, o Experimento A reportará:

- passos por checkpoint;
- passos totais;
- exemplos vistos por checkpoint;
- exemplos vistos totais;
- mesma arquitetura, batch size, learning rate, dropout e epochs por período
  usados no regime contínuo, exceto quando explicitamente documentado.

Para reduzir assimetria injustificada, cada modelo independente será treinado
com:

```text
epochs = base_epochs para t0
epochs = epochs_per_period para t > 0
```

Assim, cada checkpoint recebe a mesma quantidade de treinamento que o período
correspondente receberia no regime contínuo, mas sem herdar pesos anteriores.

### Métricas primárias

As métricas primárias serão as mesmas do Experimento P:

- `M_final`;
- `path_length`;
- `displacement_efficiency`;
- `recovery`;
- `F_acc`;
- `F_acc_adv`, quando houver placebo apropriado;
- `shape_error`;
- erro de forma cruzado `G` versus `A`.

Para `independent_period`, a vantagem sobre o `continual_placebo` não será
usada como critério primário, pois não há placebo independente diretamente
equivalente à repetição de `D0` com pesos contínuos. A comparação primária será
entre cada regime e o oráculo temporal.

### Critério de superioridade arquitetural

`continual_chronological` será considerado superior a `independent_period` se,
nas três seeds agregadas:

1. tiver `shape_error` mediano menor em `gradual`;
2. tiver `shape_error` mediano menor em `abrupt_persistent`;
3. mantiver `Recovery(transient)` e `Recovery(oscillating)` pelo menos tão altas
   quanto `independent_period`, com tolerância de `0.05`;
4. mantiver `M_final(gradual)` e `M_final(abrupt_persistent)` acima do p95 do
   nulo em proporção não inferior à de `independent_period` por mais de `10`
   pontos percentuais.

Se apenas os itens 1, 3 e 4 forem satisfeitos, mas o item 2 falhar, a conclusão
será:

> a continuidade ajuda a trajetória gradual e a recuperação, mas suaviza ou não
> melhora rupturas abruptas.

Se `independent_period` tiver menor `shape_error` em `abrupt_persistent`, isso
será interpretado como evidência de que a suavização de rupturas pode vir da
continuidade dos pesos.

### O que não será concluído

O Experimento A com `independent_period` não decide sozinho entre:

- continuidade dos pesos;
- memória explícita dos dados anteriores;
- efeito de orçamento total;
- estabilidade causada por inicialização compartilhada ou distinta.

Esses pontos exigirão o baseline posterior `cumulative_retrain`.

### Próximo baseline após `independent_period`

Se o contraste com `independent_period` for informativo, o próximo baseline será
`cumulative_retrain`:

```text
theta_t = modelo treinado do zero em D0 + ... + Dt
```

Esse baseline testa se a memória dos dados anteriores basta sem continuidade
dos pesos.

## Adendo do Experimento A2: treinamento acumulativo do zero

Este adendo foi escrito após a comparação com `independent_period` e antes da
implementação de `cumulative_retrain`.

### Motivação

`independent_period` testou se perfis por período, sem continuidade e sem
memória dos períodos anteriores, bastam para recuperar a trajetória. O resultado
foi negativo: o regime contínuo preservou melhor forma temporal, recuperação e
detecção acima do nulo.

`cumulative_retrain` testa uma pergunta diferente:

> A vantagem do Timeformer vem da continuidade dos pesos ou apenas do fato de
> o modelo ver dados acumulados de períodos anteriores?

### Definição do regime

Para cada período `t`, treinar um modelo novo, inicializado do zero, usando:

```text
D_0 + D_1 + ... + D_t
```

Isso produz uma sequência de checkpoints:

```text
phi_0, phi_1, ..., phi_t
```

mas cada `phi_t` é treinado independentemente, sem herdar pesos de
`phi_(t-1)`.

### Regra de orçamento

Usaremos um orçamento de atualizações equivalente ao número de atualizações que
o regime contínuo já teria acumulado até o período correspondente:

```text
steps(phi_t) = steps_continuo_ate_t
```

Na configuração confirmatória:

```text
steps(phi_0) = 1500
steps(phi_t) = 1500 + 750 * t, para t > 0
```

Esse regime é computacionalmente mais caro que o contínuo, mas responde a uma
pergunta diferente: se a memória dos dados acumulados basta sem continuidade
dos pesos. Todos os passos serão registrados.

### Critério de leitura

Se `cumulative_retrain` igualar ou superar `continual_chronological`, então a
contribuição arquitetural específica da continuidade dos pesos enfraquece: a
memória acumulada dos dados pode bastar.

Se `continual_chronological` superar `cumulative_retrain`, então a continuidade
dos pesos ganha evidência própria, pois o baseline acumulativo teve acesso aos
dados passados, mas sem trajetória de otimização contínua.

As métricas primárias permanecem:

- `shape_error` em `gradual` e `abrupt_persistent`;
- `Recovery` em `transient` e `oscillating`;
- proporção acima do p95 nulo em `gradual` e `abrupt_persistent`;
- `M_final`, `path_length` e `F_acc` como diagnósticos.

### Resultado esperado que favorece o Timeformer

O Timeformer será considerado superior ao `cumulative_retrain` se:

1. tiver `shape_error` mediano menor em `gradual`;
2. tiver `shape_error` mediano menor em `abrupt_persistent`;
3. mantiver recuperação em `transient` e `oscillating` pelo menos tão alta
   quanto `cumulative_retrain`, com tolerância de `0.05`;
4. mantiver taxa acima do p95 nulo em `gradual` e `abrupt_persistent` não
   inferior por mais de `10` pontos percentuais.

Se `cumulative_retrain` for melhor apenas em `abrupt_persistent`, isso será
interpretado como evidência de que reiniciar do zero com dados acumulados pode
representar rupturas melhor que continuidade pura, enquanto o contínuo ainda
pode ser melhor para acumulação e reversão.

## Configuração computacional confirmatória

Depois de um teste de sanidade com uma seed:

- seeds confirmatórias: `1000`, `1001`, `1002`;
- `10` períodos;
- `100` exemplos por sujeito e período;
- checkpoint final, sem seleção desigual por parada antecipada;
- orçamento fixo e registrado para todos os regimes comparáveis;
- probe preditivo fixo;
- relação principal: Jensen-Shannon;
- resultados por sujeito, condição, seed e período;
- intervalos ou dispersões entre seeds reportados;
- todos os checkpoints e perfis preservados.

Se uma condição exigir configuração diferente por impossibilidade técnica, a
mudança será documentada antes da execução confirmatória.

## Sequência de implementação

1. Criar um gerador separado para quartetos temporais pareados, sem alterar o
   comportamento padrão do corpus atual.
2. Adicionar testes unitários para as trajetórias `G`, `A`, `T`, `O` e para o
   balanceamento direcional.
3. Implementar as métricas `L`, `E`, `F_step`, `F_acc`, `ShapeError` e
   `Recovery` como funções independentes.
4. Adicionar testes com perfis relacionais artificiais, nos quais os resultados
   esperados sejam conhecidos exatamente.
5. Criar um runner separado para o Experimento P, reutilizando treinamento,
   extração de perfis e controles existentes.
6. Executar uma seed de sanidade com configuração reduzida.
7. Documentar qualquer desvio necessário antes da execução confirmatória.
8. Executar três seeds confirmatórias com orçamento fixo.
9. Decidir, com base nos critérios pré-registrados, se a formulação estrutural
   deve entrar no planejamento principal.
10. Somente depois implementar os baselines do Experimento A.

## Desvio registrado após o smoke test

Um smoke test reduzido foi executado antes da configuração confirmatória. Ele
revelou que o desenho inicial com `10` quartetos e nenhuma âncora misturava a
trajetória própria de cada sujeito com o movimento dos demais sujeitos.

Exemplo: mesmo quando uma palavra abrupta permanecia constante nos primeiros
períodos, seu perfil relacional mudava porque palavras graduais se moviam em
relação a ela. Assim, todos os passos de seu delta-oráculo relacional podiam ser
ativos.

Antes de qualquer execução confirmatória, o benchmark foi alterado para `16`
âncoras estáveis e `24` alvos em `6` quartetos. As hipóteses, condições
temporais e critérios confirmatórios permanecem os mesmos. O smoke test
anterior não será usado como evidência científica.

## Resultados exploratórios permitidos

Podem ser explorados, mas não usados para declarar sucesso confirmatório:

- limiares condicionais por entropia ou frequência;
- Rank-Biased Overlap;
- coerência por comunidades semânticas;
- métricas topológicas;
- optimal transport;
- perfis separados por sentido;
- ajustes dos limiares confirmatórios;
- benchmark multidimensional.

Essas análises devem ser rotuladas explicitamente como exploratórias.

## Condições para avançar ao corpus real

Antes de usar COHA ou outro corpus real, precisamos:

1. concluir o Experimento P;
2. comparar continuidade cronológica com ao menos `independent_period`;
3. documentar sensibilidade ao orçamento de atualizações;
4. definir tratamento para tamanhos desiguais de corpus por período;
5. definir vocabulário de referência e política para palavras que entram ou
   desaparecem;
6. criar benchmark sintético com ao menos dois eixos semânticos;
7. declarar explicitamente a limitação atual para sentidos coexistentes.
