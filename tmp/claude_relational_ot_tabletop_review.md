# Pedido de crítica: teste de mesa para deslocamento semântico por transporte relacional

Queremos avaliar uma ideia antes de transformá-la em um novo estimador completo.
O objetivo imediato é medir o deslocamento semântico de uma palavra entre dois
períodos usando somente a geometria relacional aprendida pelo TimeFormer:

```text
plane em D0 (1810-1860)
versus
plane em D1 (1960-2010)
```

A hipótese é que não basta medir quantos vizinhos mudaram. Devemos medir o
**custo semântico da substituição dos vizinhos**.

Este documento faz:

1. um exemplo fictício mínimo;
2. o mesmo cálculo com números reais do TimeFormer;
3. análise de sensibilidade a `k`, temperatura e seleção;
4. avaliação preliminar nas 37 palavras;
5. enumeração dos pontos que podem invalidar a proposta.

Não queremos uma defesa da ideia. Queremos saber se existe uma formulação
robusta ou se os resultados já aconselham abandoná-la.

## Contexto

O TimeFormer é um único Transformer MLM treinado continuamente:

```text
theta_0 = treino em D0
theta_1 = continuação de theta_0 em D1
```

Para cada checkpoint e corpus, calculamos centroides contextuais locais:

```text
h_t(v) =
1 / N_t(v) sum_o h_theta_t(v,o)
```

O perfil relacional de `w` é:

```text
r_t(w)[v] =
cos(h_t(w), h_t(v))
```

Cada similaridade é calculada inteiramente dentro do checkpoint correspondente.
Não comparamos diretamente as coordenadas de `theta_0` e `theta_1`.

Resultados anteriores:

```text
APD relacional:
Spearman = 0,210

plane_nn:
gold 0,882, mas rank baixo no APD

chairman_nn:
gold 0,000, mas falso positivo dominante
```

As vizinhanças médias, porém, são interpretáveis:

```text
plane D0:
line, angle, plate, column

plane D1:
boat, ship, rail, route, machine
```

```text
chairman D0:
secretary, editor, commander, director, president

chairman D1:
secretary, director, commander, president, commissioner
```

## Intuição da nova hipótese

APD pode considerar duas mudanças grandes apenas porque os membros mudaram:

```text
secretary -> director
angle     -> aircraft
```

Mas semanticamente:

```text
cost(secretary, director) deve ser pequeno
cost(angle, aircraft) deve ser grande
```

Assim, representamos cada período como massa distribuída sobre vizinhos e
calculamos quanto custa transportar a massa antiga para a nova.

## Passo 1: construir a distribuição de vizinhança

Selecionamos os `k` vizinhos mais próximos de `w` em cada período.

Para D0:

```text
N_0^k(w) = top-k de r_0(w)
```

Para D1:

```text
N_1^k(w) = top-k de r_1(w)
```

Uma distribuição ponderada possível é:

```text
p_0(u|w) =
exp(r_0(w,u) / tau)
/
sum_{x in N_0^k(w)} exp(r_0(w,x) / tau)
```

e analogamente:

```text
p_1(v|w)
```

Também testamos massa uniforme:

```text
p_t(v|w) = 1/k
```

### Problema já visível

Selecionar top-k separadamente cria suportes diferentes e descarta toda a
cauda. Depois da renormalização:

```text
sum_{v in N_t^k(w)} p_t(v|w) = 1
```

mesmo que os `k` vizinhos expliquem proporções muito diferentes da estrutura
total nos dois períodos.

Portanto, o OT pode medir a troca **condicionada ao top-k**, e não a mudança da
distribuição relacional completa.

## Passo 2: definir o custo entre referências

Para dois tokens de referência `u` e `v`, calculamos sua distância dentro de
cada checkpoint:

```text
d_0(u,v) = 1 - cos(h_0(u), h_0(v))
d_1(u,v) = 1 - cos(h_1(u), h_1(v))
```

O custo usado no teste de mesa é:

```text
c(u,v) =
1 - 1/2 [
    cos(h_0(u), h_0(v))
    +
    cos(h_1(u), h_1(v))
]
```

Equivalentemente:

```text
c(u,v) = 1/2 [d_0(u,v) + d_1(u,v)]
```

Isso não exige alinhamento vetorial, pois cada cosseno é calculado localmente.

### Problema já visível

O custo não é uma régua fixa. Se `u` ou `v` também mudou semanticamente:

```text
d_0(u,v) != d_1(u,v)
```

A média pode:

1. ocultar uma distância histórica importante;
2. reduzir artificialmente o custo porque os referenciais convergiram;
3. aumentar o custo porque um dos referenciais mudou por conta própria.

Logo, `c(u,v)` ainda incorpora a régua móvel do TimeFormer.

## Passo 3: resolver o transporte

Buscamos uma matriz de fluxo:

```text
pi(u,v) >= 0
```

com marginais:

```text
sum_v pi(u,v) = p_0(u|w)
sum_u pi(u,v) = p_1(v|w)
```

e minimizamos:

```text
Delta_OT(w) =
min_pi
sum_{u in N_0^k(w)}
sum_{v in N_1^k(w)}
pi(u,v) c(u,v)
```

Quanto mais semanticamente distantes forem os vizinhos novos dos antigos,
maior será o score.

---

# Exemplo fictício mínimo

## `chairman`

Suponha:

```text
D0:
secretary 0,50
president 0,30
committee 0,20

D1:
director     0,45
president    0,35
commissioner 0,20
```

Custos fictícios:

| D0 \ D1 | director | president | commissioner |
|---|---:|---:|---:|
| secretary | 0,10 | 0,12 | 0,14 |
| president | 0,08 | 0,00 | 0,10 |
| committee | 0,15 | 0,16 | 0,12 |

Um fluxo plausível:

```text
secretary   -> director      0,45 * 0,10
secretary   -> commissioner  0,05 * 0,14
president   -> president     0,30 * 0,00
committee   -> president     0,05 * 0,16
committee   -> commissioner  0,15 * 0,12
```

Então:

```text
Delta_OT(chairman)
= 0,045 + 0,007 + 0 + 0,008 + 0,018
= 0,078
```

A lista mudou, mas a massa permaneceu no campo institucional.

## `plane`

Suponha:

```text
D0:
line  0,40
angle 0,35
plate 0,25

D1:
aircraft 0,50
flight   0,30
engine   0,20
```

Custos fictícios:

| D0 \ D1 | aircraft | flight | engine |
|---|---:|---:|---:|
| line | 0,75 | 0,72 | 0,65 |
| angle | 0,70 | 0,74 | 0,68 |
| plate | 0,58 | 0,66 | 0,50 |

Mesmo o fluxo ótimo precisa atravessar campos:

```text
Delta_OT(plane) aproximadamente 0,65
```

Nesse mundo fictício:

```text
plane >> chairman
```

Esse é o comportamento desejado.

---

# Teste com números reais do TimeFormer

Usamos:

```text
camada 2
3.216 referências compartilhadas
centroides contextuais
geometria centrada
custo médio entre theta_0@D0 e theta_1@D1
```

Primeiro, escolhemos quatro vizinhos semanticamente representativos de cada
período apenas para entender a matriz. Esse não é ainda um estimador automático.

## Caso real 1: `plane_nn`

Similaridades observadas:

```text
D0:
line   0,932
angle  0,924
plate  0,887
column 0,859

D1:
boat    0,916
ship    0,894
route   0,835
machine 0,784
```

Matriz de custos:

| D0 \ D1 | boat | ship | route | machine |
|---|---:|---:|---:|---:|
| line | 0,583 | 0,500 | 0,585 | 0,616 |
| angle | 0,495 | 0,460 | 0,383 | 0,532 |
| plate | 0,299 | 0,313 | 0,408 | 0,353 |
| column | 0,365 | 0,328 | 0,414 | 0,420 |

Com massa uniforme `0,25`, o fluxo ótimo encontrado foi:

```text
line   -> ship     custo 0,500
angle  -> route    custo 0,383
plate  -> machine  custo 0,353
column -> boat     custo 0,365
```

Logo:

```text
Delta_OT(plane) =
0,25 * (0,500 + 0,383 + 0,353 + 0,365)
= 0,400
```

### Primeira surpresa

Os custos não são tão altos quanto a narrativa “geometria versus transporte”
sugere. Por exemplo:

```text
c(plate, boat) = 0,299
c(column, ship) = 0,328
```

Isso acontece porque o espaço relacional considera relações compartilhadas
como objeto físico, estrutura, material e localização. OT encontra exatamente
essas pontes baratas.

Isso pode ser:

```text
uma virtude:
a mudança não é uma troca entre categorias totalmente desconectadas;

ou um defeito:
o custo semântico do modelo não separa adequadamente os campos relevantes.
```

## Caso real 2: `chairman_nn`

Similaridades:

```text
D0:
secretary 0,934
editor    0,878
commander 0,877
director  0,875

D1:
secretary 0,944
director  0,939
commander 0,894
president 0,888
```

Matriz de custos:

| D0 \ D1 | secretary | director | commander | president |
|---|---:|---:|---:|---:|
| secretary | 0,000 | 0,072 | 0,082 | 0,082 |
| editor | 0,103 | 0,128 | 0,137 | 0,136 |
| commander | 0,082 | 0,144 | 0,000 | 0,169 |
| director | 0,072 | 0,000 | 0,144 | 0,152 |

Fluxo ótimo uniforme:

```text
secretary -> secretary  custo 0,000
commander -> commander  custo 0,000
director  -> director   custo 0,000
editor    -> president  custo 0,136
```

Portanto:

```text
Delta_OT(chairman) =
0,25 * 0,136
= 0,034
```

Esse é o resultado ideal para nosso diagnóstico:

```text
Delta_OT(plane)    = 0,400
Delta_OT(chairman) = 0,034
```

## Caso real 3: `tree_nn`

Escolhemos:

```text
D0:
rock, water, leaf, grass

D1:
wood, stone, garden, forest
```

Matriz de custos:

| D0 \ D1 | wood | stone | garden | forest |
|---|---:|---:|---:|---:|
| rock | 0,134 | 0,151 | 0,223 | 0,140 |
| water | 0,112 | 0,206 | 0,194 | 0,181 |
| leaf | 0,148 | 0,175 | 0,228 | 0,196 |
| grass | 0,066 | 0,114 | 0,153 | 0,149 |

OT uniforme:

```text
Delta_OT(tree) = 0,143
```

Assim:

```text
chairman 0,034
tree     0,143
plane    0,400
```

O score reconhece que `tree` reorganizou sua vizinhança, mas não atravessou uma
distância comparável a `plane`.

## Caso real 4: `graft_nn`

Escolhemos:

```text
D0:
globe, bee, planet, vine

D1:
compound, machinery, acid, cell
```

A vizinhança D0 é heterogênea, portanto este caso não é uma demonstração limpa
de botânica.

OT uniforme:

```text
Delta_OT(graft) = 0,494
```

No exemplo manual:

```text
graft    0,494
plane    0,400
tree     0,143
chairman 0,034
```

A ordenação qualitativa coincide com nossas expectativas.

---

# O que acontece quando removemos a seleção manual

Selecionamos automaticamente os top-k de cada período e testamos:

```text
k = {4, 10, 20, 50}

massas:
uniforme
softmax tau = {0,05, 0,10, 0,20}
```

## Scores dos quatro casos

| alvo | k | uniforme | tau=.05 | tau=.10 | tau=.20 |
|---|---:|---:|---:|---:|---:|
| plane | 4 | 0,313 | 0,412 | 0,366 | 0,340 |
| plane | 10 | 0,231 | 0,328 | 0,270 | 0,248 |
| plane | 20 | 0,198 | 0,283 | 0,224 | 0,207 |
| plane | 50 | 0,176 | 0,253 | 0,195 | 0,182 |
| chairman | 4 | 0,034 | 0,035 | 0,035 | 0,034 |
| chairman | 10 | 0,104 | 0,059 | 0,079 | 0,091 |
| chairman | 20 | 0,155 | 0,085 | 0,120 | 0,139 |
| chairman | 50 | 0,165 | 0,093 | 0,129 | 0,147 |
| graft | 4 | 0,641 | 0,681 | 0,658 | 0,649 |
| graft | 10 | 0,663 | 0,678 | 0,665 | 0,663 |
| graft | 20 | 0,529 | 0,621 | 0,569 | 0,547 |
| graft | 50 | 0,491 | 0,574 | 0,515 | 0,500 |
| tree | 4 | 0,139 | 0,145 | 0,142 | 0,141 |
| tree | 10 | 0,120 | 0,129 | 0,125 | 0,122 |
| tree | 20 | 0,159 | 0,151 | 0,154 | 0,157 |
| tree | 50 | 0,119 | 0,132 | 0,124 | 0,121 |

### Leitura

Com `tau=0,05`, a ordenação permanece:

```text
graft > plane > tree > chairman
```

para todos os valores testados de `k`.

Com massas uniformes, a separação entre `plane` e `chairman` quase desaparece
em `k=50`:

```text
plane    0,176
chairman 0,165
```

Por quê?

```text
top-4:
chairman compartilha 3 vizinhos idênticos entre os períodos

top-50:
chairman compartilha 26 vizinhos, mas os outros 24 precisam transportar
massa uniforme, mesmo sendo periféricos
```

Massas uniformes dão aos vizinhos de posição 50 o mesmo peso dos primeiros.
A temperatura baixa reduz esse problema, mas introduz um hiperparâmetro
decisivo.

## Seleção manual versus top-k automático

Para `plane`:

```text
quatro vizinhos semanticamente representativos:
0,400

top-4 estritamente automático:
0,313
```

O top-4 automático de D1 é:

```text
boat, ship, fence, rail
```

enquanto o exemplo manual usou:

```text
boat, ship, route, machine
```

Essa diferença é pequena conceitualmente, mas altera o score em cerca de 28%.
Isso mostra que não podemos usar exemplos escolhidos semanticamente como
evidência quantitativa.

---

# Resultado preliminar nas 37 palavras

Sem field-control e sem escolher hiperparâmetros pelo gold:

| k | massa | Spearman | plane rank | chairman rank | graft rank | tree rank |
|---:|---|---:|---:|---:|---:|---:|
| 4 | uniforme | 0,145 | 11 | 36 | 2 | 29 |
| 4 | tau=.05 | 0,155 | 7 | 36 | 1 | 27 |
| 10 | uniforme | 0,142 | 18 | 35 | 1 | 31 |
| 10 | tau=.05 | **0,196** | 8 | 36 | 1 | 30 |
| 20 | uniforme | 0,089 | 22 | 28 | 1 | 27 |
| 20 | tau=.05 | 0,168 | 9 | 36 | 1 | 27 |
| 50 | uniforme | 0,129 | 25 | 26 | 2 | 35 |
| 50 | tau=.05 | 0,183 | 12 | 36 | 1 | 30 |

Baseline anterior:

```text
APD relacional = 0,210
```

Portanto:

```text
OT melhora fortemente plane/chairman;
OT não melhora a correlação global;
o melhor valor observado, 0,196, ainda é inferior ao APD.
```

Não interpretamos `k=10, tau=0,05` como configuração vencedora. Ela foi
identificada olhando os resultados e seria tuning no gold.

---

# Pontos problemáticos

## 1. O custo aprende as mesmas confusões que queremos corrigir

Se o TimeFormer aproxima `plate` e `boat` por ambos serem objetos
físicos/estruturais:

```text
c(plate, boat) = 0,299
```

o transporte declara essa substituição relativamente barata.

Assim, OT não acrescenta semântica externa. Ele reorganiza a própria semântica
já presente no TimeFormer.

Pergunta:

> Por que esperar que o custo entre referências corrija uma falha que nasceu
> na mesma geometria usada para definir o custo?

## 2. O custo médio mistura dois tempos

Usamos:

```text
c = (d_0 + d_1) / 2
```

Alternativas:

```text
c = d_0
c = d_1
c = max(d_0, d_1)
c = min(d_0, d_1)
```

Cada uma responde a uma pergunta diferente:

```text
d_0:
quanto o novo bairro está distante segundo a semântica histórica?

d_1:
quanto o antigo bairro está distante segundo a semântica moderna?

média:
compromisso retrospectivo entre as duas réguas

max:
exige que a substituição seja barata nos dois tempos

min:
aceita qualquer ponte barata e pode subestimar a mudança
```

Não há escolha obviamente neutra.

## 3. Top-k cria descontinuidade

Uma alteração mínima pode trocar o vizinho de rank `k` pelo de rank `k+1`.
A massa desse token muda de zero para positiva abruptamente.

## 4. Softmax cria sensibilidade à temperatura

```text
tau pequeno:
quase toda massa nos primeiros vizinhos

tau grande:
vizinhos periféricos ganham massa
```

No nosso teste, temperatura baixa é precisamente o que mantém `chairman`
baixo quando `k` cresce.

## 5. A cauda desaparece

As distribuições renormalizadas não guardam quanta afinidade total o top-k
representava. Uma categoria residual seria necessária:

```text
OTHER_t(w)
```

Mas não existe custo semântico natural entre `OTHER` e um token específico.

## 6. OT encontra pontes semanticamente convenientes

No exemplo de `plane`:

```text
line   -> ship
angle  -> route
plate  -> machine
column -> boat
```

Esse fluxo minimiza custo, mas não afirma que esses pares representam
substituições históricas reais. O acoplamento é explicativo apenas no sentido
matemático.

## 7. O score confunde mudança do alvo e mudança das referências

Se `route`, `machine` ou `plate` também mudam, o custo e as massas de `plane`
mudam juntos. Isso pode amplificar ou cancelar o deslocamento.

## 8. A boa separação dos quatro casos pode ser local

Temos:

```text
plane > chairman
graft > tree
```

mas:

```text
Spearman máximo observado = 0,196
```

Logo, resolver os casos canônicos não basta para validar o estimador geral.

## 9. O baseline de campo ainda não está definido automaticamente

Poderíamos calcular:

```text
Delta_res(w) =
Delta_OT(w)
-
median_{c in Controls(w)} Delta_OT(c)
```

Mas escolher `Controls(w)` automaticamente pode recriar:

```text
clustering de campos;
dependência de top-k;
contaminação por controles polissêmicos;
remoção de mudanças reais compartilhadas.
```

---

# Reformulações possíveis antes de abandonar

## Opção A: suporte comum, sem top-k separado

Definir:

```text
U_k(w) = N_0^k(w) union N_1^k(w)
```

e calcular as duas distribuições no mesmo suporte:

```text
p_t(v|w), v in U_k(w)
```

Vantagem:

```text
um token não desaparece apenas porque caiu abaixo de k em um período
```

Limite:

```text
continuamos truncando a cauda e escolhendo k
```

## Opção B: distribuição rank-based

Em vez de temperatura:

```text
p_t(v|w) proporcional a 1 / rank_t(v)^alpha
```

Vantagem:

```text
reduz dependência da escala de cosseno
```

Limite:

```text
troca tau por alpha e mantém top-k
```

## Opção C: custo conservador

Usar:

```text
c_max(u,v) = max(d_0(u,v), d_1(u,v))
```

Uma substituição só é barata quando os tokens são próximos nas duas réguas.

Risco:

```text
qualquer deriva de uma referência aumenta o custo, mesmo sem mudança do alvo
```

## Opção D: custo com referências estáveis

Estimar estabilidade:

```text
stability(v) =
similaridade entre o perfil relacional de v em D0 e D1
```

E permitir no custo ou no suporte apenas referências estáveis.

Risco:

```text
as referências semanticamente mais informativas podem ser justamente as que
mudaram;
```

e:

```text
avaliar estabilidade já exige outro estimador de mudança.
```

## Opção E: Sinkhorn divergence no perfil completo

Evitar top-k e usar todas as 3.216 referências, com regularização entrópica:

```text
S_epsilon(P_0,P_1) =
OT_epsilon(P_0,P_1)
- 1/2 OT_epsilon(P_0,P_0)
- 1/2 OT_epsilon(P_1,P_1)
```

Vantagem:

```text
reduz viés entrópico e evita corte abrupto
```

Problemas:

```text
matriz de custo 3.216 x 3.216;
escolha de epsilon;
como converter cossenos positivos e negativos numa distribuição;
o custo ainda vem da mesma geometria móvel.
```

## Opção F: admitir que OT é score qualitativo complementar

OT talvez seja útil para:

```text
explicar por que duas listas diferem;
separar trocas internas de trocas entre campos;
produzir fluxos auditáveis;
```

sem ser o estimador principal do ranking SemEval.

Nesse desenho:

```text
APD = quantidade de reorganização contextual
OT  = custo semântico da substituição da vizinhança média
```

Os dois scores responderiam a perguntas diferentes.

---

# Nossa leitura provisória

O teste de mesa confirma que a ideia tem conteúdo:

```text
chairman:
troca de vizinhos dentro do mesmo campo -> custo baixo

plane:
troca entre geometria/material e transporte -> custo maior
```

Mas também mostra que a proposta ainda não é um estimador confiável:

```text
1. depende materialmente de k e tau;
2. o custo é produzido pela mesma geometria móvel;
3. a seleção manual melhora o exemplo;
4. a correlação global não supera APD;
5. não existe baseline de campo automático defensável.
```

Assim, não recomendamos implementar ainda um pipeline completo com bootstrap,
field-control e tuning.

## Próximo teste mínimo, se a direção for mantida

Antes de qualquer infraestrutura maior:

1. usar suporte comum `N_0^k union N_1^k`;
2. comparar custos `d_0`, `d_1`, média e `max`;
3. usar pesos uniformes, softmax predefinido e rank-based;
4. avaliar sensibilidade sem selecionar pelo gold;
5. verificar se existe uma região de configurações, não um único ponto, em que:

```text
plane > chairman
graft > tree
Spearman >= APD 0,210
```

Se nenhuma família de configurações satisfizer simultaneamente esses critérios,
OT deve ser mantido apenas como diagnóstico interpretável ou abandonado como
score principal.

# Perguntas para sua avaliação

1. A formulação mede deslocamento semântico ou apenas reorganização de uma
   vizinhança truncada?
2. Existe uma definição de custo entre referências que seja defensável sem
   introduzir um encoder externo?
3. `d_0`, `d_1`, média ou `max` tem interpretação científica superior?
4. Suporte comum resolve o problema principal ou apenas suaviza top-k?
5. A categoria `OTHER` é necessária? Como definir seu custo?
6. Rank-based weighting é mais defensável que softmax?
7. Devemos exigir melhora global sobre APD ou aceitar OT como dimensão
   complementar?
8. Os fluxos ótimos podem ser usados como explicação ou seriam facilmente
   superinterpretados?
9. Existe uma versão mais simples que capture a mesma distinção entre
   `plane` e `chairman`?
10. Diante dos números atuais, você continuaria, reformularia ou abandonaria OT
    como estimador principal?

# Formato solicitado

Responda com:

1. diagnóstico em uma frase;
2. erro conceitual mais grave;
3. interpretação dos quatro casos reais;
4. correções matemáticas necessárias;
5. próximo teste mínimo;
6. critério explícito de falsificação;
7. recomendação sobre OT como score principal ou complementar.

# Artefatos usados

```text
outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/
temporal_relational_neighborhoods/neighborhoods.csv

outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/
temporal_relational_neighborhoods/report.md

outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/
hidden_relational_profiles/cache/

scripts/report_temporal_relational_neighborhoods.py
```
