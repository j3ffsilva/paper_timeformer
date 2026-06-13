# Pedido de segunda opinião: plano para perfis relacionais comunitários

Queremos uma avaliação independente e crítica de uma reformulação conceitual do
nosso método de mudança semântica temporal. O ponto central é distinguir
**mudança de vizinhos individuais** de **mudança de comunidade semântica**.

Não assuma que a proposta abaixo está correta. Procure erros conceituais,
problemas de identificabilidade, vazamento de informação, escolhas arbitrárias
e alternativas mais simples.

## Objetivo

Treinamos continuamente um Transformer MLM:

```text
theta_0 = treino em D0 (1810-1860)
theta_1 = continuação de theta_0 em D1 (1960-2010)
```

Queremos detectar palavras que mudaram de sentido entre D0 e D1 no
SemEval-2020 Task 1.

## O que já sabemos

Depois de corrigir fronteiras documentais e substituir mascaramento central por
MLM dinâmico, o modelo aprendeu estruturas semanticamente coerentes.

Exemplo `plane_nn`:

```text
theta_0@D0:
line, angle, plate, column, canal, coast

theta_1@D1:
boat, ship, rail, route, engine, machine
```

Isso corresponde aproximadamente a:

```text
D0: sentido geométrico/material
D1: sentido de avião/transporte
```

Apesar disso, APD relacional coloca `plane_nn` em 35º entre 37 alvos, embora o
gold graded seja 0,882, o maior do dataset.

Exemplo `chairman_nn`, gold estável:

```text
theta_0@D0:
secretary, editor, commander, director, president, committee, jury

theta_1@D1:
secretary, director, commander, president, commissioner, governor, publisher
```

O campo continua sendo liderança institucional. Porém, D1 fica muito mais
concentrado em certos cargos:

```text
secretary 86%
commander 79%
commissioner 72%
director 71%
president 69%
```

APD relacional coloca `chairman_nn` em 1º, como se fosse a palavra mais mudada.

Balancear as ocorrências não resolveu:

```text
APD original Spearman = 0,210
APD balanceado, 100 ocorrências x 20 seeds = 0,212

plane_nn:    rank 35
chairman_nn: rank 1
graft_nn:    rank 9
```

## A confusão conceitual

Nosso perfil atual é:

```text
R_t(w)[v] = similaridade de w com cada token de referência v
```

E o deslocamento mede a diferença entre `R_0(w)` e `R_1(w)`.

Esse score responde corretamente:

> Quanto mudaram as relações específicas de w com seus vizinhos?

Mas o benchmark parece perguntar algo diferente:

> A palavra passou a ocupar outro campo ou sentido semântico?

Essas grandezas não são equivalentes.

### Mudança microrrelacional

Em `chairman_nn`, os vizinhos específicos e suas intensidades mudaram. Portanto,
o perfil relacional realmente mudou. O APD não está matematicamente errado ao
detectar isso.

Porém, os vizinhos antigos e modernos continuam pertencendo ao mesmo campo:

```text
Liderança institucional
```

### Mudança comunitária

Em `plane_nn`, os vizinhos mudam de um conjunto geométrico/material para um
conjunto de transporte. Não é apenas troca de membros ou concentração dentro
do mesmo campo; ocorre transferência entre campos semânticos.

Nossa interpretação atual é:

```text
chairman_nn:
grande mudança microrrelacional
pouca mudança comunitária

plane_nn:
mudança microrrelacional talvez moderada
grande mudança comunitária
```

O método atual mede principalmente a primeira. Queremos medir a segunda.

## O que chamamos de “bairro”

Um bairro é uma comunidade de palavras ou usos que compartilham relações
semânticas internas, mesmo que seus membros individuais e frequências mudem.

Exemplo conceitual:

```text
Bairro Liderança:
secretary, director, president, commissioner,
chairman, governor, committee, senator

Bairro Geometria:
line, angle, surface, plate, plane, axis

Bairro Transporte:
aircraft, engine, flight, route, pilot, airport
```

Não queremos fornecer esses nomes ou listas manualmente. Operacionalmente, um
bairro deve emergir da estrutura relacional dos dados.

## Nova representação proposta

Em vez de usar diretamente:

```text
R_t(w) = relações de w com milhares de tokens individuais
```

queremos construir:

```text
B_t(w) = afinidade ou massa de w em cada comunidade semântica
```

Exemplo idealizado:

```text
chairman_nn:

D0:
Liderança    0,85
Geometria    0,02
Transporte   0,03

D1:
Liderança    0,88
Geometria    0,01
Transporte   0,02

Resultado: pouca mudança comunitária
```

```text
plane_nn:

D0:
Geometria    0,79
Transporte   0,12

D1:
Geometria    0,15
Transporte   0,81

Resultado: grande transferência de massa
```

```text
graft_nn:

D0:
Botânica     0,80
Corrupção    0,08
Medicina     0,05

D1:
Botânica     0,20
Corrupção    0,42
Medicina     0,32

Resultado: mudança e diversificação de sentidos
```

## Plano de ataque proposto

### Etapa 1 — Construir um grafo relacional estável

Usar o vocabulário compartilhado e confiável já extraído:

```text
3.216 tokens
frequência mínima de 100 em cada período
sem palavras-alvo
```

Para cada período/checkpoint, cada token `v` possui um centroide contextual
`h_t(v)`.

Construímos um grafo:

```text
nó = token de referência
aresta(v,u) = similaridade contextual centrada entre v e u
```

Para evitar um grafo completo e anisotropia:

```text
manter somente mutual-kNN
usar similaridade centrada
testar k em uma faixa pré-definida
```

Questão crítica: introduzir `k` contradiz parcialmente nossa motivação inicial
de evitar vizinhanças arbitrárias. Alternativas seriam threshold adaptativo,
backbone de disparidade ou um grafo ponderado completo.

### Etapa 2 — Encontrar comunidades semânticas

Candidatos:

```text
Leiden
Louvain
Infomap
spectral clustering
hierarchical clustering
```

Preferência inicial: Leiden em grafo mutual-kNN ponderado, por permitir
comunidades de tamanhos diferentes.

Precisamos testar estabilidade:

```text
variação de k/resolução
bootstrap das ocorrências
seeds do algoritmo
adjusted mutual information entre partições
```

Se as comunidades forem instáveis, não podemos interpretá-las como bairros.

### Etapa 3 — Resolver o alinhamento entre períodos

Há três desenhos possíveis.

#### A. Comunidades conjuntas

Construir um único grafo de referência usando uma representação agregada dos
dois períodos:

```text
S_joint(v,u) = média ou combinação de S_0(v,u) e S_1(v,u)
```

Detectar uma única partição `C`, compartilhada entre D0 e D1.

Vantagem:

```text
dimensões de B_0 e B_1 já são idênticas
sem problema de alinhamento de comunidades
```

Risco:

```text
a agregação pode apagar comunidades que existem apenas num período
ou usar informação futura para definir o bairro histórico
```

Como o objetivo é avaliação retrospectiva, o uso dos dois períodos talvez seja
aceitável, mas precisa ser declarado.

#### B. Comunidades independentes e alinhadas

Detectar `C_0` e `C_1` separadamente e alinhá-las por:

```text
Jaccard dos membros
optimal transport entre comunidades
matching máximo bipartido
similaridade de perfis relacionais agregados
```

Vantagem:

```text
permite nascimento, morte, fusão e divisão de comunidades
```

Risco:

```text
o alinhamento pode recriar exatamente o problema que tentávamos evitar
e adicionar muitos graus de liberdade
```

#### C. Comunidades congeladas em D0

Detectar bairros apenas em `theta_0@D0` e projetar as ocorrências de D1 nesses
bairros.

Vantagem:

```text
sem vazamento futuro
interpretação clara: quanta massa deixou os bairros históricos?
```

Risco:

```text
novos sentidos podem não corresponder a nenhuma comunidade de D0
```

Podemos adicionar uma categoria de novidade/resíduo.

Nossa preferência inicial para o primeiro teste é **A, comunidades conjuntas**,
por ser a opção mais simples e por não exigir alinhamento. B e C seriam
controles/ablação.

### Etapa 4 — Projetar palavras e ocorrências nos bairros

Para cada ocorrência `o` de uma palavra-alvo `w`, já temos:

```text
r_t(w,o)[v] = similaridade com cada token de referência v
```

Para cada comunidade `C_j`, calcular afinidade:

```text
a_t(w,o,j) =
    logsumexp_{v em C_j}(r_t(w,o)[v] / tau)
    - log |C_j|
```

Ou, como baseline simples:

```text
a_t(w,o,j) = média dos top-m valores de r_t(w,o)[v], v em C_j
```

Depois normalizar:

```text
b_t(w,o) = softmax(a_t(w,o) / tau)
```

`b_t(w,o)` é uma distribuição de massa comunitária para uma ocorrência.

Agregações possíveis:

```text
B_t(w) = média_o b_t(w,o)
```

e também manter a distribuição de `b_t(w,o)` para preservar polissemia.

### Etapa 5 — Definir scores separados

Não queremos voltar a confundir grandezas. Reportaremos pelo menos:

#### 1. Mudança microrrelacional

```text
Delta_micro(w) = APD entre r_0(w,o) e r_1(w,o)
```

#### 2. Mudança comunitária média

```text
Delta_community(w) = JSD(B_0(w), B_1(w))
```

#### 3. Mudança distributiva de sentidos

Comparar distribuições de `b_t(w,o)`:

```text
energy distance
MMD
optimal transport
```

#### 4. Novidade

Massa de D1 que não é bem explicada por nenhuma comunidade histórica, no
desenho C.

Assim poderemos dizer explicitamente:

```text
chairman_nn:
Delta_micro alto
Delta_community baixo

plane_nn:
Delta_community alto

graft_nn:
Delta_community alto e possível aumento de entropia de sentidos
```

### Etapa 6 — Controles fatoriais

Para cada score comunitário, calcular:

```text
theta_0@D0 versus theta_0@D1
theta_1@D0 versus theta_1@D1
theta_0@D0 versus theta_1@D0
theta_0@D1 versus theta_1@D1
theta_0@D0 versus theta_1@D1
```

Objetivo:

```text
se a troca de bairro aparece ao trocar o corpus com checkpoint congelado,
há evidência de mudança contextual;

se aparece principalmente ao trocar checkpoint no mesmo corpus,
há evidência de deriva do modelo.
```

Não assumiremos que essas distâncias sejam aditivas.

### Etapa 7 — Validação

#### Quantitativa

```text
Spearman graded
ROC-AUC binário
average precision
intervalos bootstrap
permutation tests
correlação com frequência
análise por faixa de frequência
```

#### Qualitativa pré-especificada

Auditar pelo menos:

```text
plane_nn      mudança esperada
graft_nn      mudança/polissemia
chairman_nn   estável com mudança microrrelacional
tree_nn       estável e frequente
```

Não escolher exemplos apenas depois de olhar o resultado.

#### Estabilidade dos bairros

```text
AMI entre partições
persistência dos membros centrais
robustez a k, resolução, seed e bootstrap
```

## Experimento mínimo proposto

Antes de construir toda a infraestrutura:

1. usar os centroides em cache da camada 2;
2. construir uma matriz conjunta centrada para os 3.216 tokens;
3. criar grafo mutual-kNN para `k in {10, 20, 40}`;
4. detectar comunidades com Leiden em algumas resoluções;
5. escolher configurações por estabilidade da partição, não pelo gold;
6. calcular afinidade comunitária média das 37 palavras;
7. avaliar `JSD(B_0, B_1)`;
8. verificar especificamente se:

```text
plane_nn sobe substancialmente;
chairman_nn cai;
graft_nn permanece alto;
correlação com frequência diminui.
```

Critério preliminar de continuação:

```text
Spearman claramente acima do APD=0,21
e melhoria qualitativa simultânea de plane/chairman
e partições estáveis entre configurações próximas.
```

Se não ocorrer, testar comunidades congeladas em D0 antes de abandonar a
hipótese comunitária.

## Principais riscos que reconhecemos

1. Comunidades de palavras podem refletir sintaxe, frequência ou domínio, não
   sentidos.
2. Uma palavra pode pertencer simultaneamente a vários bairros; hard clustering
   pode ser inadequado.
3. Bairros podem ser hierárquicos, sobrepostos e dependentes da granularidade.
4. O parâmetro de resolução pode ser tão arbitrário quanto o antigo `k`.
5. Comunidades conjuntas podem incorporar informação futura.
6. Agregar por comunidade pode esconder mudanças internas semanticamente
   importantes.
7. O modelo pequeno pode produzir geometria insuficiente para comunidades
   robustas.
8. O gold SemEval mede mudança lexical agregada, não necessariamente
   transferência entre comunidades discretas.

## Perguntas para sua avaliação

1. A distinção entre mudança microrrelacional e mudança comunitária é
   conceitualmente defensável?
2. “Bairro semântico” deve ser operacionalizado como comunidade de tokens,
   comunidade de ocorrências/usos ou estrutura hierárquica?
3. Qual dos desenhos A/B/C é o mais defensável para o primeiro experimento?
4. A construção conjunta dos bairros constitui vazamento metodológico?
5. Leiden sobre mutual-kNN é adequado, ou há uma alternativa mais natural e
   menos paramétrica?
6. Como projetar uma ocorrência numa comunidade sem favorecer comunidades
   grandes?
7. JSD entre massas comunitárias realmente separa `plane_nn` de
   `chairman_nn`, ou ainda confundirá concentração?
8. Devemos usar soft clustering/topic models/mixture models em vez de
   comunidades rígidas?
9. Como definir novidade semântica quando um sentido de D1 não existia em D0?
10. Qual experimento mínimo falsifica esta hipótese com o menor número de
    graus de liberdade?

## Arquivos relevantes

```text
outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/hidden_relational_profiles/report.md
outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/balanced_apd_layer2/report.md
outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/hidden_relational_profiles/cache/
scripts/evaluate_hidden_relational_profiles.py
scripts/diagnose_balanced_relational_apd.py
docs/09-relational_profile_formalization.md
```

Por favor, responda com:

1. falhas conceituais ou estatísticas, ordenadas por severidade;
2. recomendação entre A/B/C ou uma quarta alternativa;
3. formulação matemática precisa do score comunitário recomendado;
4. desenho do experimento mínimo;
5. critérios objetivos para manter ou abandonar esta direção.
