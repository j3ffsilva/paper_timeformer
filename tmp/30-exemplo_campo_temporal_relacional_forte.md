# Exemplo trivial: campo temporal relacional como geometria temporal forte

**Objetivo:** pensar a versão mais forte e academicamente competitiva da ideia:
não apenas medir que um vetor mudou, mas aprender/representar um **campo de
deslocamento temporal** que transforma o perfil relacional de uma palavra de um
período para outro.

---

## 1. Ideia central

A versão fraca seria:

```text
embedding_t1(w) - embedding_t0(w)
```

Isto mede quanto o ponto `w` se moveu no espaço vetorial. Essa é uma ideia
parecida com Hamilton depois de alinhar espaços.

A versão forte seria:

```text
R_t(w) = perfil relacional de w no período t
Δ_t(w) = deslocamento temporal relacional de w

R_{t+1}(w) ≈ R_t(w) + Δ_t(w)
```

Aqui, `R_t(w)` não é o embedding bruto de `w`. É uma assinatura composta pelas
relações de `w` com outras palavras nomeáveis.

Exemplo:

```text
R_t(plane)[line]   = força da relação de plane com line
R_t(plane)[angle]  = força da relação de plane com angle
R_t(plane)[ship]   = força da relação de plane com ship
R_t(plane)[pilot]  = força da relação de plane com pilot
```

O deslocamento temporal passa a ser interpretável:

```text
Δ(plane)[line]  < 0  perdeu relação com line
Δ(plane)[angle] < 0  perdeu relação com angle
Δ(plane)[ship]  > 0  ganhou relação com ship
Δ(plane)[pilot] > 0  ganhou relação com pilot
```

Essa é uma geometria temporal em que as dimensões são relações linguísticas,
não coordenadas internas opacas.

---

## 2. Vocabulário trivial

Vamos imaginar um vocabulário de referência com apenas seis palavras:

```text
V = [line, angle, surface, ship, pilot, flight]
```

Queremos estudar a palavra:

```text
w = plane
```

Vamos supor dois períodos:

```text
t0 = 1900
t1 = 2000
```

---

## 3. Perfil relacional em 1900

Em 1900, `plane` aparece majoritariamente em contextos geométricos:

```text
"inclined plane"
"geometrical plane"
"angle of the plane"
"surface of the plane"
```

Seu perfil relacional poderia ser:

```text
R_1900(plane)

line    = 0.90
angle   = 0.85
surface = 0.80
ship    = 0.10
pilot   = 0.05
flight  = 0.05
```

Em forma vetorial:

```text
R_1900(plane) = [0.90, 0.85, 0.80, 0.10, 0.05, 0.05]
```

Interpretação:

```text
plane@1900 está no campo geométrico/material.
```

---

## 4. Perfil relacional em 2000

Em 2000, `plane` aparece majoritariamente em contextos de transporte aéreo:

```text
"the plane landed"
"the pilot entered the plane"
"the flight was delayed"
"passengers boarded the plane"
```

Seu perfil relacional poderia ser:

```text
R_2000(plane)

line    = 0.15
angle   = 0.10
surface = 0.20
ship    = 0.65
pilot   = 0.90
flight  = 0.95
```

Em forma vetorial:

```text
R_2000(plane) = [0.15, 0.10, 0.20, 0.65, 0.90, 0.95]
```

Interpretação:

```text
plane@2000 está no campo transporte/aviação.
```

---

## 5. Deslocamento temporal relacional

O deslocamento temporal é:

```text
Δ(plane, 1900→2000) = R_2000(plane) - R_1900(plane)
```

Calculando dimensão por dimensão:

```text
line    = 0.15 - 0.90 = -0.75
angle   = 0.10 - 0.85 = -0.75
surface = 0.20 - 0.80 = -0.60
ship    = 0.65 - 0.10 = +0.55
pilot   = 0.90 - 0.05 = +0.85
flight  = 0.95 - 0.05 = +0.90
```

Logo:

```text
Δ(plane, 1900→2000) = [-0.75, -0.75, -0.60, +0.55, +0.85, +0.90]
```

Esse vetor é semanticamente legível:

```text
perdas:
line, angle, surface

ganhos:
ship, pilot, flight
```

Essa é a “seta temporal” de `plane`.

---

## 6. Ir para frente no tempo

Se temos o perfil em 1900 e o deslocamento, podemos reconstruir o perfil em
2000:

```text
R_1900(plane) + Δ(plane, 1900→2000)
```

Substituindo:

```text
[0.90, 0.85, 0.80, 0.10, 0.05, 0.05]
+[-0.75,-0.75,-0.60,+0.55,+0.85,+0.90]
=
[0.15, 0.10, 0.20, 0.65, 0.90, 0.95]
```

Que é exatamente:

```text
R_2000(plane)
```

Consulta:

```text
similares(plane@2000)
```

Ordenando o perfil de 2000:

```text
flight  0.95
pilot   0.90
ship    0.65
surface 0.20
line    0.15
angle   0.10
```

Resultado:

```text
plane@2000 → flight, pilot, ship
```

---

## 7. Voltar no tempo

Também podemos inverter:

```text
R_1900(plane) = R_2000(plane) - Δ(plane, 1900→2000)
```

Substituindo:

```text
[0.15, 0.10, 0.20, 0.65, 0.90, 0.95]
-[-0.75,-0.75,-0.60,+0.55,+0.85,+0.90]
=
[0.90, 0.85, 0.80, 0.10, 0.05, 0.05]
```

Consulta:

```text
similares(plane@1900)
```

Ordenando o perfil de 1900:

```text
line    0.90
angle   0.85
surface 0.80
ship    0.10
pilot   0.05
flight  0.05
```

Resultado:

```text
plane@1900 → line, angle, surface
```

---

## 8. Inserir um período intermediário

Agora vem o ponto onde o TimeFormer pode ser mais interessante que uma simples
comparação D0/D1.

Suponha que exista um período intermediário:

```text
t_mid = 1950
```

Poderíamos modelar uma transição parcial:

```text
R_1950(plane) = R_1900(plane) + 0.5 * Δ(plane, 1900→2000)
```

Calculando:

```text
0.5 * Δ = [-0.375, -0.375, -0.300, +0.275, +0.425, +0.450]
```

Então:

```text
R_1950(plane)
= [0.90, 0.85, 0.80, 0.10, 0.05, 0.05]
+[-0.375,-0.375,-0.300,+0.275,+0.425,+0.450]
= [0.525, 0.475, 0.500, 0.375, 0.475, 0.500]
```

Ordenando:

```text
line    0.525
surface 0.500
flight  0.500
angle   0.475
pilot   0.475
ship    0.375
```

Interpretação:

```text
plane@1950 está em transição:
mantém resíduos geométricos, mas já começa a ativar aviação.
```

Esse é o tipo de continuidade temporal que o SemEval D0/D1 não consegue testar
bem, mas que é central para a contribuição.

---

## 9. Exemplo de palavra estável

Agora usemos uma palavra estável:

```text
w = tree
```

Mesmo vocabulário simplificado, mas com referências adequadas:

```text
V = [leaf, bark, root, forest, pilot, flight]
```

Em 1900:

```text
R_1900(tree) = [0.90, 0.85, 0.80, 0.75, 0.05, 0.05]
```

Em 2000:

```text
R_2000(tree) = [0.88, 0.80, 0.77, 0.82, 0.04, 0.06]
```

Deslocamento:

```text
Δ(tree) = [-0.02, -0.05, -0.03, +0.07, -0.01, +0.01]
```

Interpretação:

```text
tree sofreu pequenas oscilações internas, mas o campo relacional permaneceu
estável.
```

A métrica não exige que tudo seja idêntico. Ela exige que a ordem de grandeza
do deslocamento seja pequena e que os vizinhos centrais permaneçam no mesmo
campo.

---

## 10. Por que isso é mais forte que Hamilton?

Hamilton mede:

```text
distância entre vetor estático de plane em 1900
e vetor estático de plane em 2000, após alinhamento
```

Ele pode dizer:

```text
plane mudou muito.
```

E pode listar vizinhos nos dois períodos.

Mas a versão forte do TimeFormer quer dizer algo mais estruturado:

```text
plane perdeu relações com line/angle/surface
plane ganhou relações com ship/pilot/flight
esse deslocamento pode ser visto como um vetor temporal
esse vetor permite consultar, avançar, retroceder e interpolar períodos
```

A diferença não é apenas computacional. É epistemológica:

```text
Hamilton:
mudança = distância entre dois pontos alinhados

Campo temporal relacional:
mudança = vetor de transformação entre perfis de relações linguísticas
```

---

## 11. Como transformar isso em solução computacional

### Passo 1 — construir perfis relacionais

Para cada palavra `w` e período `t`:

```text
R_t(w)[v] = força da relação de w com v no período t
```

Essa força pode ser calculada por:

```text
cos(centroide_t(w), centroide_t(v))
```

ou por uma versão mais robusta:

```text
distribuição contextual de vizinhos
probabilidade de substituição
PMI relacional
OT entre distribuições de ocorrência
```

O ponto essencial é que `R_t(w)` deve viver num espaço auditável:

```text
dimensões = palavras/referências/atributos linguísticos
```

### Passo 2 — calcular deslocamentos observados

Para períodos consecutivos:

```text
Δ_t(w) = R_{t+1}(w) - R_t(w)
```

Se houver muitos períodos:

```text
Δ_1900(plane)
Δ_1920(plane)
Δ_1940(plane)
...
```

Isso produz uma trajetória.

### Passo 3 — aprender um campo temporal

Em vez de apenas guardar cada deslocamento, podemos treinar um módulo:

```text
F(R_t(w), t) → Δ_t(w)
```

Ou:

```text
G(R_t(w), t) → R_{t+1}(w)
```

Objetivo:

```text
min || R_{t+1}(w) - (R_t(w) + F(R_t(w), t)) ||²
```

Com regularizações:

```text
deslocamento pequeno para palavras estáveis
suavidade temporal entre períodos próximos
capacidade de mudança abrupta quando o dado exigir
esparsidade para tornar ganhos/perdas interpretáveis
```

### Passo 4 — consultar no tempo

Depois de aprendido:

```text
R_1950(plane) = consulta direta do checkpoint 1950
```

ou, se o período não existir:

```text
R_1950(plane) ≈ R_1900(plane) + α * Δ(plane, 1900→2000)
```

Então:

```text
similares(plane@1950) = top_k(R_1950(plane))
```

---

## 12. O experimento mínimo

Com o SemEval, só temos dois períodos. Então o experimento mínimo não prova
continuidade, mas testa a formulação:

1. Calcular `R_D0(w)` e `R_D1(w)` para todos os 37 alvos.
2. Calcular:

```text
Δ(w) = R_D1(w) - R_D0(w)
```

3. Relatar:

```text
top ganhos
top perdas
turnover de vizinhos
norma de Δ
JSD/OT entre distribuições normalizadas de R
```

4. Comparar com Hamilton:

```text
Hamilton: vetor mudou quanto?
TimeFormer: perfil relacional mudou como?
```

O ponto não é apenas vencer no Spearman. O ponto é mostrar que o deslocamento
do TimeFormer tem conteúdo linguístico auditável.

---

## 13. O que precisamos para a versão forte de verdade

Precisamos de um corpus com mais períodos:

```text
1900, 1920, 1940, 1960, 1980, 2000
```

Então podemos mostrar:

```text
R_1900(plane)
R_1920(plane)
R_1940(plane)
R_1960(plane)
R_1980(plane)
R_2000(plane)
```

E estudar:

```text
velocidade da mudança
direção persistente
mudança abrupta
mudança reversível
estabilização
```

Esse é o território onde a contribuição fica mais forte:

> TimeFormer como instrumento de geometria temporal relacional, capaz de
> consultar e acompanhar a reorganização das vizinhanças lexicais ao longo de
> checkpoints cronológicos.

---

## 14. Resumo em uma frase

A versão forte não diz:

```text
o vetor de plane andou muito.
```

Ela diz:

```text
o perfil relacional de plane perdeu massa em line/angle/surface,
ganhou massa em ship/pilot/flight, e essa transformação pode ser tratada como
um vetor temporal interpretável que permite avançar, voltar e interpolar no
tempo.
```

