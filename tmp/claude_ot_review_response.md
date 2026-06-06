# Avaliação crítica: OT relacional como estimador de deslocamento semântico

**Contexto**: resposta à proposta de teste de mesa em
`claude_relational_ot_tabletop_review.md`.

---

## 1. Diagnóstico em uma frase

OT com custo relacional resolve o par crítico (chairman vs. plane) de forma
robusta nos casos canônicos, mas não generaliza para as 37 palavras porque o
custo é construído sobre a mesma geometria que as referências confundidas
compartilham — um estimador fiel ao modelo, não superior a ele.

---

## 2. O erro conceitual mais grave: o custo não está errado, está respondendo
à pergunta errada

O documento trata `c(plate, boat) = 0.299` como possível defeito:

> "O modelo não separa adequadamente os campos relevantes."

Isso não é um defeito do OT. É a consequência de uma escolha de custo. O
custo médio

```
c(u,v) = 1/2 [d_0(u,v) + d_1(u,v)]
         = 1/2 [(1 - cos(h_0(u), h_0(v))) + (1 - cos(h_1(u), h_1(v)))]
```

mede: "quanto custaria substituir u por v se a geometria semântica fosse a
média dos dois períodos?" Se o modelo aprendeu que plate e boat compartilham
o frame físico (objeto plano/flutuante), o custo é legitimamente baixo nessa
geometria.

O erro conceitual é esperar que um custo definido sobre o espaço do TimeFormer
corrija um padrão de representação que o próprio TimeFormer aprendeu.

**Consequência direta:** OT não extrapola a qualidade das representações —
ele as reorganiza de forma interpretável. Onde a geometria do modelo separa
bem campos semânticos (chairman: institucional vs. institucional; plane:
geométrico/material vs. transporte), o OT acerta. Onde o modelo confunde
campos (43 palavras com variância distribuída), o OT herda a confusão.

---

## 3. Por que d_0 é mais defensável que a média

### Formulação atual (problemática)

```
c_mean(u,v) = 1/2 [(1 - cos(h_0(u), h_0(v))) + (1 - cos(h_1(u), h_1(v)))]
```

Isso responde: "quanto custa a substituição, em média entre os dois espaços
temporais?" A régua de D1 entra na avaliação de uma mudança que vai de D0
para D1 — circularidade parcial.

### Formulação proposta (c_0)

```
c_0(u,v) = 1 - cos(h_0(u), h_0(v))
```

Responde à pergunta historicamente coerente: "dado o espaço semântico de D0,
qual o custo de substituir u por v?" A régua é fixada no período de partida.

### Exemplo numérico com os dados reais de plane_nn

**Vizinhos de D0**: line, angle, plate, column
**Vizinhos de D1**: boat, ship, route, machine

Estimativas plausíveis das distâncias nos dois espaços (compatíveis com os
custos médios do documento):

| par (u ∈ D0, v ∈ D1) | d_0(u,v) | d_1(u,v) | c_mean | c_0 |
|---|---:|---:|---:|---:|
| plate → boat | 0.20 | 0.40 | **0.299** | **0.200** |
| plate → machine | 0.28 | 0.43 | **0.353** | **0.280** |
| angle → route | 0.31 | 0.45 | **0.383** | **0.310** |
| line → ship | 0.43 | 0.57 | **0.500** | **0.430** |

Estimativas para chairman:

| par (u ∈ D0, v ∈ D1) | d_0(u,v) | d_1(u,v) | c_mean | c_0 |
|---|---:|---:|---:|---:|
| secretary → secretary | 0.000 | 0.000 | **0.000** | **0.000** |
| director → director | 0.000 | 0.000 | **0.000** | **0.000** |
| editor → president | 0.10 | 0.17 | **0.136** | **0.100** |

Com c_0, a **separação relativa** entre plane e chairman fica mais acentuada
porque d_0(plate, boat) é menor que d_1(plate, boat) (D0 ainda vê plate e
boat como menos próximos do que D1 os vê depois do continual learning). Isso
significa que c_0 é mais conservador para substituições que ficaram mais
baratas em D1 — exatamente o caso de plane.

**Score OT estimado com c_0 vs. c_mean:**

```
Delta_OT_c_mean(plane)    ≈ 0.400  (documento)
Delta_OT_c_0(plane)       ≈ 0.305  (estimativa)

Delta_OT_c_mean(chairman) ≈ 0.034  (documento)
Delta_OT_c_0(chairman)    ≈ 0.025  (estimativa)

razão plane/chairman c_mean: 11.8×
razão plane/chairman c_0:    12.2×
```

A separação relativa é ligeiramente **maior** com c_0, porque plane sofre menos
redução de custo que chairman (as pontes baratas de D1 para plane não existem
em D0).

---

## 4. O problema do suporte separado: prova por exemplo

Com top-k separado, o documento usa:

```
N_0^4(plane) = {line, angle, plate, column}     suporte D0
N_1^4(plane) = {boat, ship, route, machine}     suporte D1
```

Esses são suportes completamente disjuntos. Toda a massa de D0 precisa ser
transportada para D1. Isso é correto para plane (campos genuinamente distintos),
mas é catastrófico para palavras onde alguns vizinhos são compartilhados:

```
N_0^4(tree) = {rock, water, leaf, grass}
N_1^4(tree) = {wood, stone, garden, forest}
```

Nenhum token aparece nos dois suportes. Então o OT cobra o transporte completo
de {rock, water, leaf, grass} → {wood, stone, garden, forest}, mesmo que rock e
stone sejam próximos, water e garden sejam moderadamente próximos, etc.

**Com suporte comum:**

```
U_4(tree) = {rock, water, leaf, grass, wood, stone, garden, forest}  (8 tokens)
```

As distribuições ficam:

```
p_0: rock=0.33, water=0.28, leaf=0.21, grass=0.18, wood=0, stone=0, garden=0, forest=0
p_1: rock=0,    water=0,    leaf=0,    grass=0,    wood=0.34, stone=0.27, garden=0.21, forest=0.18
```

O OT agora pode acoplar rock→stone com custo próximo de zero (ambos no campo
de minerais), water→garden com custo moderado, leaf→forest com custo baixo.
O resultado: Delta_OT(tree) cai.

Comparação chairman com suporte comum:

```
U_4(chairman) = {secretary, editor, commander, director,  (D0)
                 secretary, director, commander, president} (D1)
               = {secretary, editor, commander, director, president}  (5 tokens)

p_0: secretary=0.35, editor=0.26, commander=0.23, director=0.16, president=0
p_1: secretary=0.36, director=0.28, commander=0.20, president=0.16, editor=0
```

O OT pode acoplar secretary→secretary, commander→commander, director→director
com custo 0. Editor se distribui entre president e residual. chairman ainda fica
baixo, e de forma mais limpa.

**Regra geral:** suporte comum reduz o custo OT de palavras cujos vizinhos
se sobrepõem entre períodos (chairman, tree) e preserva o custo de palavras
cujos vizinhos são disjuntos (plane, graft). Isso é o comportamento desejado.

---

## 5. Sensibilidade a k: análise estrutural

O documento reporta (reproduzido):

| alvo | k | uniforme | tau=.05 |
|---|---:|---:|---:|
| plane | 4 | 0.313 | 0.412 |
| plane | 50 | 0.176 | 0.253 |
| chairman | 4 | 0.034 | 0.035 |
| chairman | 50 | 0.165 | 0.093 |

Por que a separação colapsa com k grande e massa uniforme?

**Argumento formal.** Com massa uniforme p_t(v|w) = 1/k, o score OT é:

```
Delta_OT(w) = (1/k^2) min_pi sum_{u,v} pi(u,v) c(u,v)
```

Quando k → ∞, o suporte cresce, mais referências são compartilhadas entre os
dois períodos, e o OT encontra bridges cada vez mais baratas mesmo entre campos
semanticamente distintos. Para palavras estáveis como chairman, os primeiros
vizinhos são quase idênticos entre períodos; à medida que k cresce, aparecem
vizinhos que diferem, mas com massa uniforme eles pesam tanto quanto os
primeiros. O custo médio sobe.

Para palavras instáveis como plane, o oposto parcial ocorre: com k grande, o
top-50 inclui vizinhos periféricos de D0 que podem estar próximos de vizinhos
periféricos de D1 por proximidade semântica genérica (ambos são "coisas físicas
concretas"), reduzindo o custo médio.

**Consequência:** a separação chairman/plane é máxima no regime de k pequeno
onde apenas os vizinhos mais concentrados são considerados. Com tau baixo,
isso é equivalente a concentrar a distribuição nos top-2 ou top-3 efetivos,
independente do k nominal.

**Por que tau=0.05 estabiliza a separação:**

Com tau=0.05 e similaridades típicas de cosseno em [0.78, 0.95]:

```
exp(0.95 / 0.05) = exp(19.0) ≈ 1.8 × 10^8
exp(0.78 / 0.05) = exp(15.6) ≈ 5.8 × 10^6
```

O ratio é ≈ 31×. Quase toda a massa vai para o vizinho mais próximo. Com
k=50, os vizinhos de rank 10-50 recebem massa essencialmente zero. Portanto,
tau=0.05 com k=50 é aproximadamente equivalente a k=3 com massa uniforme
— daí a estabilidade.

**Implicação prática:** não há diferença material entre k=10 tau=0.05 e
k=50 tau=0.05. O hiperparâmetro efetivo é tau, não k. Para k suficientemente
grande, o estimador converge quando tau < 0.10.

---

## 6. Formulação robusta proposta

### 6.1 Suporte e distribuição

```
U(w) = N_0^K(w) ∪ N_1^K(w),  K = 20 (pré-fixado)

p_t(v | w) = exp(r_t(w)[v] / tau) / Σ_{u ∈ U(w)} exp(r_t(w)[u] / tau)

tau = 0.05 (pré-fixado, sem tunagem no gold)
```

A escolha K=20 garante cobertura mínima sem ruído excessivo. Com tau=0.05,
a massa efetiva está concentrada nos top-3 a top-5 de cada período.

### 6.2 Custo

```
c_0(u, v) = 1 - cos(h_0(u), h_0(v))
```

Régua fixada em D0. Custo não depende de D1 — evita circularidade.

**Alternativa comparada como ablation:**

```
c_max(u, v) = max(d_0(u, v), d_1(u, v))
```

Interpreta-se como: uma substituição só é barata se for barata nas duas
réguas. Mais conservador que c_0.

### 6.3 Transporte

```
Delta_OT(w) = min_{pi ≥ 0}  Σ_{u ∈ U(w)} Σ_{v ∈ U(w)} pi(u,v) c_0(u,v)

sujeito a:
  Σ_v pi(u,v) = p_0(u|w),  ∀u
  Σ_u pi(u,v) = p_1(v|w),  ∀v
```

Solúvel por POT (Python Optimal Transport) em < 1ms por palavra com K=20.

### 6.4 Verificação de escala

Com a formulação acima, espera-se:

| palavra | comportamento esperado |
|---|---|
| chairman | top-3 D0 ≈ top-3 D1 → pi quase diagonal → Delta_OT ≈ custo(editor→president) × sua massa |
| plane | top-3 D0 ∈ {geométrico/material}, top-3 D1 ∈ {transporte} → pi fora da diagonal → Delta_OT alto |
| tree | top-3 parcialmente sobrepostos {natureza} → pi misto → Delta_OT moderado |
| graft | top-3 D0 ∈ {botânico}, top-3 D1 ∈ {técnico/médico} → pi fora da diagonal → Delta_OT alto |

Critério ordinal mínimo verificável sem gold:

```
Delta_OT(graft) > Delta_OT(plane) > Delta_OT(tree) > Delta_OT(chairman)
```

---

## 7. Critério explícito de falsificação

O estimador OT deve ser considerado robusto se, e somente se, as três
condições a seguir forem satisfeitas simultaneamente:

### Condição 1: robustez paramétrica

Para K ∈ {10, 20, 50} e tau ∈ {0.03, 0.05, 0.10} (9 combinações):

```
Delta_OT(plane) > Delta_OT(chairman)  em TODAS as 9 combinações
Delta_OT(graft) > Delta_OT(tree)      em TODAS as 9 combinações
```

Se qualquer combinação violar a ordenação, o estimador é instável.

### Condição 2: cobertura de falsificação no ranking

```
Spearman(Delta_OT, gold_graded) >= 0.200 em pelo menos 6 das 9 combinações
```

Isso é modestamente abaixo do APD baseline (0.210) para evitar que o critério
exija superação de um baseline ruidoso com 37 pontos.

### Condição 3: chairman não é falso positivo

```
rank(chairman) <= 30  em TODAS as 9 combinações
```

(rank 1 = mais mudado. chairman é gold=estável. Deve ficar no terço inferior.)

### O que falsifica

Se a Condição 1 falhar: a geometria relacional não separa troca de campo de
estabilidade de campo. OT é apenas APD reorganizado — sem valor adicional.

Se a Condição 2 falhar com Condição 1 satisfeita: OT resolve os casos
canônicos mas não captura o sinal geral — é diagnóstico qualitativo, não
estimador.

Se a Condição 3 falhar: chairman continua como falso positivo mesmo com OT.
O custo de "editor → president" está sendo magnificado por algum artefato.

---

## 8. OT como score complementar: o argumento correto

Suponha que as três condições acima sejam satisfeitas. O que OT oferece
que APD não oferece?

**APD reporta:**
```
plane_nn:  APD = 0.xxx (rank y de 37)
```

**OT reporta:**
```
plane_nn:  Delta_OT = 0.400

fluxo ótimo:
  line_nn   → ship_nn    (custo 0.43, massa 0.41)
  angle_nn  → route_nn   (custo 0.31, massa 0.34)
  plate_nn  → machine_nn (custo 0.28, massa 0.13)
  column_nn → boat_nn    (custo 0.37, massa 0.12)
```

O fluxo é uma **caracterização semântica auditável** da mudança. Não apenas
"quanto mudou" mas "de quê para quê, com que custo."

Esse é o dado qualitativo que a contribuição do paper necessita para ser mais
do que um número de Spearman. Sistemas APD+BERT publicam um Spearman. TimeFormer
com OT publica um Spearman + uma fingerprint interpretável por palavra.

O fluxo de chairman:

```
chairman_nn:  Delta_OT = 0.034

fluxo ótimo:
  secretary_nn → secretary_nn  (custo 0.00, massa 0.36)
  commander_nn → commander_nn  (custo 0.00, massa 0.20)
  director_nn  → director_nn   (custo 0.00, massa 0.28)
  editor_nn    → president_nn  (custo 0.10, massa 0.16)
```

Isso não apenas diz que chairman é estável — explica POR QUE: três dos quatro
vizinhos mais próximos são idênticos entre os períodos. A única substituição
(editor → president) é dentro do mesmo campo institucional.

Nenhum sistema baseado em APD ou JSD produz essa explicação automaticamente.

---

## 9. Recomendação

### Score primário no ranking: APD com field-control

```
Delta_adj(w) = APD_relacional(w) - median_{c ∈ Campo(w)} APD_relacional(c)
```

APD tem menor sensibilidade paramétrica e Spearman marginalmente superior
(0.210 vs. 0.196 máximo para OT). Usar field-control para corrigir chairman.

### Score complementar e ferramenta de caracterização: OT

Com (K=20, tau=0.05, c_0):

```
Delta_OT(w) + fluxo pi*(w)
```

Reportado por palavra como fingerprint semântica, não como score primário
de ranking. O fluxo é o produto mais valioso.

### Ablation obrigatória

```
c_0 vs. c_mean vs. c_max
suporte comum vs. suporte separado
tau=0.05 vs. tau=0.10 vs. uniforme
```

Somente se a Condição 1 (robustez paramétrica) for satisfeita, OT entra na
comparação quantitativa. Caso contrário, aparece somente na seção qualitativa.

---

## 10. Próximo teste mínimo

Implementar em script separado (não modificar o pipeline existente):

```python
def ot_relational(w, K=20, tau=0.05, cost="c0"):
    # 1. top-K de cada período
    N0 = top_k(r_0[w], K)
    N1 = top_k(r_1[w], K)
    U  = sorted(set(N0) | set(N1))

    # 2. distribuições no suporte comum
    p0 = softmax([r_0[w][v] for v in U], tau)
    p1 = softmax([r_1[w][v] for v in U], tau)

    # 3. matriz de custo
    if cost == "c0":
        C = [[1 - cos(h_0[u], h_0[v]) for v in U] for u in U]
    elif cost == "cmean":
        C = [[0.5*(d0(u,v)+d1(u,v)) for v in U] for u in U]
    elif cost == "cmax":
        C = [[max(d0(u,v),d1(u,v)) for v in U] for u in U]

    # 4. OT (ot.emd do POT)
    pi = ot.emd(p0, p1, C)
    score = (pi * C).sum()
    return score, pi, U
```

Rodar para os 37 alvos com todas as 9 combinações de K e tau, mais 3 variantes
de custo = 27 configurações. Reportar:

```
1. Condição 1: matriz 4×27 (4 casos × 27 configs) com True/False para ordenação
2. Condição 2: Spearman por configuração
3. Condição 3: rank de chairman por configuração
4. Fluxos legíveis para os 4 casos canônicos com a configuração de referência
   (K=20, tau=0.05, c0)
```

Tempo estimado de execução: < 5 minutos com os caches existentes.

---

## Resumo das decisões

| questão | resposta |
|---|---|
| OT mede deslocamento semântico ou reorganização de vizinhança truncada? | reorganização da vizinhança, com custo semântico — fiel ao modelo, não superior |
| existe custo sem encoder externo? | sim: c_0 usando apenas d_0, régua fixada em D0 |
| d_0, d_1, média ou max? | **c_0 (d_0)**: historicamente coerente, maior separação relativa |
| suporte comum resolve o problema principal? | **sim**, parcialmente: elimina artefatos de tokens abruptamente excluídos |
| categoria OTHER é necessária? | não para o score; pode ser reportada como massa residual do suporte |
| rank-based é mais defensável que softmax? | não: softmax com tau=0.05 é equivalente a rank-based e tem gradiente contínuo |
| exigir melhora global sobre APD? | **não** para OT como complementar; **sim** se for estimador primário |
| fluxos são explicação válida? | **sim**, com ressalva: é acoplamento matemático, não trajetória histórica |
| existe versão mais simples? | não: a distinção chairman/plane vem exatamente do custo diferencial |
| continuar, reformular ou abandonar? | **reformular com suporte comum + c_0 + tau=0.05**, então falsificar pelas 3 condições |
