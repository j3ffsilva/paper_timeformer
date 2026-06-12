# Perfil Relacional e Trajetória Semântica — Definição Canônica v2

**Status:** definição canônica (substitui a v1)
**Projeto:** Timeformer
**Escopo:** perfil relacional, decomposição em modos semânticos, persistência de modos, trajetória semântica, protocolo de validação e implementação

---

## 0. O que muda em relação à v1

| # | Mudança | Tipo | Seção |
|---|---|---|---|
| 1 | Centralização dos embeddings por período antes de qualquer cosseno | correção (translação não cancela no cosseno) | 4 |
| 2 | Proposição de invariância formalizada, com a classe de invariância correta | correção | 4.2 |
| 3 | Normalização da distribuição removida do critério de gap | correção (etapa matematicamente inerte; bug latente com componentes negativas) | 8.1 |
| 4 | Critério de gap computável sobre top-k truncado, com condição de segurança | ampliação | 8.2, 14.1 |
| 5 | Decomposição espectral via SVD de DÊ — a matriz M nunca é formada | ampliação (custo O(\|V_w\|²·d) → O(\|V_w\|·d²)) | 7.5 |
| 6 | Atribuição de ocorrências a modos — a nuvem de embeddings passa a ser usada de fato | correção (inconsistência nuvem-vs-média da v1) | 9 |
| 7 | Persistência de modos entre períodos resolvida por emparelhamento no espaço de tokens | resolução de questão aberta da v1 | 11 |
| 8 | Limiar de emparelhamento θ calibrado por distribuição nula de permutação | ampliação (evita hiperparâmetro arbitrário) | 11.4 |
| 9 | Condição γ conectada à estabilidade espectral (Davis–Kahan) | ampliação | 11.6 |
| 10 | Protocolo de validação empírica: piso de drift, split-half, shuffle temporal, validação WSI | ampliação | 13 |
| 11 | Seção de custo computacional e implementação | ampliação | 14 |

---

## 1. Motivação

Queremos medir como uma palavra w evolui semanticamente ao longo do tempo. A pergunta central não é "onde w está no espaço de embeddings" mas "como w se relaciona com todas as outras palavras do vocabulário, e como essas relações mudam".

Isso exige uma representação que:

- não dependa de alinhamento entre espaços de períodos diferentes (sem Procrustes)
- seja invariante ao sistema de coordenadas interno do modelo, na medida do que é matematicamente garantível, com o resíduo não garantível medido empiricamente
- opere sobre vocabulário fixo, garantindo comparabilidade entre períodos
- capture não só estado pontual mas trajetória — aproximação, distanciamento, surgimento, desaparecimento de relações semânticas
- distinga os usos de uma palavra polissêmica e rastreie cada uso separadamente ao longo do tempo

A v2 acrescenta o último ponto como requisito de primeira classe: a trajetória não é apenas do perfil agregado de w, mas também de cada **modo semântico** de w, com critérios explícitos de continuidade entre períodos.

---

## 2. Notação

| Símbolo | Significado |
|---|---|
| V | Vocabulário fixo, construído sobre todo o corpus de todos os períodos |
| W | Léxico de estudo — subconjunto de V para o qual perfis são efetivamente computados |
| w | Palavra alvo |
| v | Token qualquer em V |
| t | Período (ano, década, etc.) |
| t₀, t₁, ..., tₙ | Sequência ordenada de períodos |
| d | Dimensão do espaço de embeddings do encoder |
| eᵏ_t(w) | Embedding contextual da k-ésima ocorrência de w no período t |
| N_t(w) | Número de ocorrências de w no corpus do período t |
| μ_t | Vetor médio dos embeddings contextuais do período t (centro do espaço) |
| ê_t(·) | Embedding centralizado e normalizado (ver seção 4) |
| e_t(w) | Embedding contextual médio (centralizado) de w no período t |
| P_t(w) | Perfil relacional de w no período t |
| V_w | Suporte do perfil — tokens relevantes para w em t, após filtragem por τ |
| M_t(w) | Matriz de coesão semântica de w em t (objeto conceitual; nunca formado explicitamente) |
| aᵢ | i-ésimo autovetor de M_t(w) — vetor de cargas do modo i sobre V_w |
| λᵢ | i-ésimo autovalor de M_t(w) |
| k | Número de modos semânticos identificados |
| τ, γ, θ, ε | Limiares: relevância, validade de gap, emparelhamento de modos, segurança de truncamento |
| T(w) | Trajetória semântica de w — sequência de perfis ao longo de todos os períodos |

---

## 3. Vocabulário Fixo

V é construído uma única vez, sobre a união de todos os corpora de todos os períodos:

```
V = build_vocab( corpus_t₀ ∪ corpus_t₁ ∪ ... ∪ corpus_tₙ )
```

Isso garante que a posição i de qualquer token v é a mesma em todos os períodos. Sem V fixo, a comparação entre períodos não tem sentido — seria comparar vetores com eixos diferentes.

**Exemplo:** "identidade" ocupa a posição 4.231 em V tanto em 1960 quanto em 2020, mesmo que em 1960 sua frequência fosse baixa e sua relação com "negro" fosse fraca. O valor muda; a posição não.

O vocabulário fixo é a fundação de duas propriedades centrais do framework: (i) perfis relacionais de períodos diferentes são diretamente comparáveis (seção 10); (ii) **modos semânticos de períodos diferentes também são diretamente comparáveis**, pois vivem no espaço de cargas sobre tokens de V, não no espaço de coordenadas do encoder (seção 11.1).

---

## 4. Pré-processamento: Centralização por Período

### 4.1 Definição

Para cada período t, computa-se o vetor médio sobre todas as ocorrências de todos os tokens do corpus de t:

```
μ_t = média de todos os embeddings contextuais produzidos pelo encoder sobre corpus_t
```

Todo embedding é centralizado e normalizado antes de qualquer operação subsequente:

```
ê_t(x) = ( e_t(x) − μ_t ) / ‖ e_t(x) − μ_t ‖
```

Todos os cossenos deste documento — perfil relacional, matriz de coesão, centróides de modo, atribuição de ocorrências — são computados sobre embeddings centralizados. A centralização é um passo declarado do pipeline, não uma opção.

### 4.2 Proposição de invariância

**Proposição.** Sejam e′ = s·R·e + c os embeddings do período t após uma transformação global com s > 0 (escala isotrópica), R ortogonal (rotação/reflexão) e c ∈ ℝᵈ (translação). Então todos os objetos derivados — P_t(w), M_t(w), seus autovalores e autovetores, e tudo a jusante — são invariantes.

**Prova.** A média transforma do mesmo modo: μ′ = s·R·μ + c. Logo:

```
e′ − μ′ = s·R·(e − μ)
```

A translação c cancela na centralização. Rotação preserva produtos internos e normas; escala isotrópica positiva cancela na normalização do cosseno. Portanto:

```
cos( e′ − μ′ , f′ − μ′ ) = cos( e − μ , f − μ )
```

para qualquer par de embeddings do período. Como todos os objetos do framework são funções exclusivamente desses cossenos, todos são invariantes. □

### 4.3 Decomposição do drift e divisão de trabalho

A proposição delimita exatamente o que é garantido analiticamente e o que precisa ser medido empiricamente. O drift de representação de um encoder sob treinamento contínuo decompõe-se em:

| Componente do drift | Tratamento | Onde |
|---|---|---|
| Rotação / reflexão global | cancela analiticamente no cosseno | 4.2 |
| Escala isotrópica global | cancela analiticamente no cosseno | 4.2 |
| Translação global (deslocamento do vetor médio) | eliminada por construção pela centralização | 4.1 |
| Drift anisotrópico e não-linear residual | **não cancela** — medido empiricamente | 13.1–13.3 |

Esta tabela substitui a alegação informal da v1 de que "o cosseno é invariante ao sistema de coordenadas". A afirmação correta é mais fraca e mais precisa: a classe de invariância do cosseno sobre embeddings centralizados é o grupo de similaridade mais translação; o resíduo fora dessa classe é objeto do protocolo de validação (seção 13).

**Nota.** A direção média é o principal componente anisotrópico documentado em embeddings contextuais; a centralização ataca, portanto, também parte da anisotropia. *Whitening* por período é o passo seguinte natural, mas envolve trade-offs (amplificação de ruído em direções de baixa variância) e fica como ablação, não como padrão (seção 16).

**Nota (2026-06-12, Fase 1).** Na implementação, "média sobre todas as
ocorrências de todos os tokens" foi testada em duas formas: ponderada por
ocorrência (uma palavra de alta frequência pesa mais em μ_t) e uniforme por
tipo (cada palavra do vocabulário de suporte conta igual,
independentemente de quão frequente é). As duas NÃO são intercambiáveis: a
versão ponderada acaba dominada por palavras de função (the, of, and...) e
seu desempenho na validação cai ao acaso; a versão uniforme por tipo
performou melhor e foi adotada como padrão. Ver
`docs/perfil_relacional_v2_resultados_fase1.md` §2 (variante "D").

---

## 5. Perfil Relacional

### 5.1 Definição

O perfil relacional de w no período t é o vetor de similaridades cosseno entre w e cada token v ∈ V, sobre embeddings centralizados:

```
P_t(w)[v] = cos( e_t(w), ê_t(v) )     para todo v ∈ V
```

onde e_t(w) é o embedding contextual médio centralizado de w em t (na seção 9, perfis por modo substituem o perfil do embedding médio quando w é polissêmica).

P_t(w) ∈ [−1, +1]^|V|

| Valor | Significado |
|---|---|
| próximo de +1 | w e v habitam a mesma região semântica em t |
| próximo de 0 | w e v são ortogonais — sem relação específica |
| negativo | w e v apontam para regiões opostas do espaço centralizado |

### 5.2 Exemplo concreto

Palavra: **negro** — Períodos: **1960** e **2020**

| v | P_1960(negro)[v] | P_2020(negro)[v] |
|---|---|---|
| raça | 0.81 | 0.74 |
| discriminação | 0.79 | 0.31 |
| preconceito | 0.74 | 0.28 |
| identidade | 0.12 | 0.91 |
| pertencimento | 0.08 | 0.88 |
| resistência | 0.15 | 0.85 |
| cultura | 0.11 | 0.79 |
| ancestralidade | 0.04 | 0.71 |
| fotossíntese | 0.01 | 0.01 |

Leitura: negro se afastou do campo semântico de discriminação/preconceito e se aproximou do campo de identidade/pertencimento/cultura. A posição de fotossíntese é corretamente próxima de zero em ambos os períodos — sem relação específica em nenhum dos dois.

### 5.3 Hipótese de encoder compartilhado

P_t(w) é definido sobre tokens de linguagem natural (V), não sobre coordenadas internas do Transformer. Isso torna os perfis de períodos diferentes comparáveis, sob a seguinte hipótese:

**Hipótese (encoder compartilhado):** os embeddings de todos os períodos são produzidos pelo mesmo encoder, atualizado por treinamento contínuo, de modo que e_t(w) e e_t′(w) vivem no mesmo espaço de representação a menos de transformações na classe de invariância da seção 4.2 e de um drift residual pequeno.

A v2 substitui a postura da v1 ("assume-se a hipótese") por uma postura operacional: a parte da hipótese coberta pela classe de invariância é garantida analiticamente (4.2); a parte não coberta — drift residual anisotrópico/não-linear — é **medida**, não assumida, pelo protocolo da seção 13. Toda medida de mudança Δ reportada pelo framework deve ser calibrada contra o piso de drift (13.1).

Se o encoder não for compartilhado (modelos independentes por período), a comparabilidade direta entre perfis não vale; a justificativa alternativa — comparação sobre a estrutura de relações internas a cada período — é mais fraca e está fora do escopo desta definição.

**Nota (2026-06-12).** A expectativa de "drift residual pequeno" (acima)
foi testada diretamente: as MESMAS frases, passadas pelo checkpoint theta0
e pelo checkpoint theta1 (treino contínuo entre os dois), produzem nuvens
de embeddings quase perfeitamente separáveis por um corte simples em dois
grupos (para a palavra mediana, quase 100% de acerto). Ou seja, na prática
o drift entre dois checkpoints de um treino contínuo NÃO é pequeno — é a
maior fonte de diferença entre as representações de t0 e t1, maior até do
que a diferença entre os dois corpora. Boa parte (mas não toda) dessa
diferença é um deslocamento simples (a "translação global" de 4.3), o que
explica por que a centralização por μ_t já ajuda; o resíduo (rotação/
reescala) é pequeno o suficiente para não dominar, mas grande o suficiente
para inflar artificialmente qualquer medida de "número de modos" feita
sobre o perfil agregado (ver nota em 7.1). **Implicação prática:** para
comparar t0 e t1, prefira aplicar o MESMO checkpoint (de preferência o mais
recente, que já viu os dois corpora) às ocorrências dos dois períodos, em
vez de aplicar o checkpoint "nativo" de cada período. Ver
`docs/perfil_relacional_v2_resultados_fase1.md` §7.14-7.19.

### 5.4 Esparsidade natural e componentes negativas

A maioria dos tokens em V tem P_t(w)[v] próximo de zero para qualquer w. Isso não é ruído — é informação correta. A esparsidade reflete a estrutura real do léxico e fundamenta a representação esparsa (seção 14.2).

Componentes negativas existem (cossenos vivem em [−1, +1]) e carregam informação de oposição, mas todas as operações de filtragem e decomposição operam sobre o prefixo positivo da distribuição ordenada: V_w é definido por P_t(w)[v] > τ com τ > 0 por construção (seção 8.2). Componentes negativas participam apenas das comparações de perfil completo (seções 10 e 12).

### 5.5 Robustez a mudanças de frequência

A similaridade cosseno é normalizada por magnitude — vetores de alta e baixa frequência contribuem igualmente, desde que suas direções sejam as mesmas. Se a frequência de w muda entre t₀ e t₁ mas seus contextos de ocorrência permanecem semanticamente similares, então e_t₀(w) e e_t₁(w) apontam em direções similares e P_t₀(w) ≈ P_t₁(w). O perfil muda quando os **contextos** mudam — não quando a frequência muda.

A ressalva honesta: frequência muito baixa afeta a **variância** da estimativa de e_t(w) e da nuvem de ocorrências, mesmo sem afetar sua direção esperada. Isso é tratado pelo limiar mínimo de ocorrências e pelo protocolo de bootstrap (seções 13.1 e 16).

---

## 6. Nuvem de Embeddings e o Problema da Média

O Timeformer produz embeddings contextuais — cada ocorrência de w numa sentença produz um vetor diferente. Para cada palavra w no período t, o resultado é uma **nuvem**:

```
{ ê¹_t(w), ê²_t(w), ..., ê^N_t(w) },     N = N_t(w)
```

O embedding médio colapsa todos os usos num único ponto, o que é enganoso para palavras polissêmicas.

**Exemplo:** "negro" em 2020 aparece em contextos de cor, identidade e mercado ilegal. A média dos três embeddings não representa fielmente nenhum dos três usos — é um centróide sem referente semântico claro.

O papel da nuvem no pipeline da v2 é duplo e explícito (corrigindo a inconsistência da v1, em que a nuvem era declarada como input mas não usada):

1. A **estrutura** dos modos é identificada no espaço de tokens, via decomposição espectral da matriz de coesão (seções 7–8).
2. As **ocorrências** da nuvem são então atribuídas aos modos identificados (seção 9), produzindo frequências por modo, embeddings por modo e perfis relacionais por modo.

---

## 7. Matriz de Coesão Semântica

### 7.1 Motivação

**Nota (2026-06-12).** Esta seção (matriz de coesão M_t(w), construída a
partir do PERFIL P_t(w)) foi implementada e testada em três formulações
diferentes -- resultado: NO-GO. Em todos os casos, o primeiro autovalor
domina os demais por 10-30x para QUALQUER palavra, inclusive palavras com
mudança de sentido conhecida (`plane_nn`, `graft_nn`), de forma
indistinguível de palavras estáveis. Ou seja, k=1 sempre, sem
discriminação. O diagnóstico (ver
`docs/perfil_relacional_v2_resultados_fase1.md` §3, §7.1-7.21): o
CENTRÓIDE de uma palavra (médio sobre todas as suas ocorrências, de todos
os sentidos misturados) já "achata" a multimodalidade antes de chegar ao
perfil -- não há mais nada para a matriz de coesão separar. A reformulação
proposta (ainda em avaliação, §7.21) inverte a ordem: agrupar primeiro a
NUVEM DE OCORRÊNCIAS de w (sem passar por um centróide único) em modos, e
só então calcular um perfil relacional POR MODO (cada modo tem seu próprio
"centróide" e, portanto, seu próprio perfil). As seções 7-9 abaixo
permanecem como definição de referência de M_t(w) e dos modos via SVD, mas
sua aplicação ao centróide agregado de w está descartada; a aplicação
correta (se confirmada) é por modo, após o agrupamento de 7.21.

Identificar modos semânticos sem fixar k a priori exige uma estrutura que capture simultaneamente:

- quais tokens são relevantes para w
- quais tokens relevantes pertencem ao mesmo campo semântico

A matriz de coesão semântica captura as duas condições num único objeto.

### 7.2 Filtragem e definição

Primeiro, filtra-se V pelo perfil relacional:

```
V_w = { v ∈ V : P_t(w)[v] > τ }
```

onde τ > 0 é determinado automaticamente (seção 8.2). Para cada par (v, v′) em V_w:

```
M_t(w)[v, v′] = P_t(w)[v] · P_t(w)[v′] · cos( ê_t(v), ê_t(v′) )
```

M_t(w) ∈ ℝ^{|V_w| × |V_w|}, simétrica.

### 7.3 Interpretação dos três fatores

O produto garante que M_t(w)[v, v′] é alto apenas quando três condições são satisfeitas simultaneamente:

| Fator | Condição |
|---|---|
| P_t(w)[v] | v é relevante para w |
| P_t(w)[v′] | v′ é relevante para w |
| cos(ê_t(v), ê_t(v′)) | v e v′ habitam a mesma região semântica |

Se qualquer condição falha, o produto cai. Dois tokens podem ser igualmente próximos de w mas pertencer a campos semânticos diferentes — o terceiro fator os separa.

**Exemplo:** em negro@2020, "identidade" e "discriminação" são ambos próximos de negro. Mas cos(ê(identidade), ê(discriminação)) é baixo — pertencem a campos distintos. M_t(negro)[identidade, discriminação] será pequeno, separando os dois modos corretamente.

### 7.4 M_t(w) é semidefinida positiva

**Proposição.** M_t(w) é semidefinida positiva (PSD).

**Prova.** Escreva M_t(w) = D C D, onde:

```
D       = diag( P_t(w)[v₁], ..., P_t(w)[v_|V_w|] )      (diagonal; entradas > 0 pois τ > 0)
C[v,v′] = cos( ê_t(v), ê_t(v′) ) = ê_t(v)ᵀ ê_t(v′)
```

C = Ê Êᵀ, onde Ê ∈ ℝ^{|V_w| × d} é a matriz cujas linhas são os embeddings centralizados e normalizados de V_w. C é uma matriz Gram e portanto PSD. Como D tem entradas positivas e M = DCD, para qualquer x:

```
xᵀ M x = (Dx)ᵀ C (Dx) ≥ 0
```

Portanto M é PSD. □

**Consequência:** todos os autovalores são reais e não negativos; a decomposição espectral é bem definida.

### 7.5 Decomposição via SVD — M nunca é formada

A prova acima dá de graça a fatoração computacional:

```
M = D Ê Êᵀ D = (DÊ)(DÊ)ᵀ
```

Seja DÊ = U Σ Wᵀ a SVD fina de DÊ ∈ ℝ^{|V_w| × d}. Então:

```
λᵢ = σᵢ²          (autovalores de M = quadrados dos valores singulares de DÊ)
aᵢ = uᵢ           (autovetores de M = vetores singulares à esquerda de DÊ)
```

**Implicações:**

- M nunca precisa ser materializada. A SVD de DÊ custa O(|V_w| · d²) contra O(|V_w|³) da autodecomposição de M, e O(|V_w|²) de memória são economizados.
- **rank(M) ≤ min(|V_w|, d).** Em particular, há no máximo d autovalores não nulos — um teto estrutural para o número de modos, que o critério de gap deve respeitar.

Convenção de sinal: cada autovetor aᵢ é orientado de modo que sua componente de maior magnitude seja positiva.

### 7.6 Modos semânticos

Os autovalores satisfazem λ₁ ≥ λ₂ ≥ ... ≥ 0. Cada autovetor aᵢ tem uma componente para cada v ∈ V_w. Os tokens com carga positiva alta em aᵢ formam o i-ésimo **modo semântico** de w em t. O número de modos k é determinado pelo critério de gap (seção 8.3).

**Nota terminológica:** o método identifica comunidades semânticas coesas — grupos de tokens simultaneamente relevantes para w e similares entre si. Chamamos esses grupos de *modos semânticos* em vez de *sentidos lexicais*; a correspondência com sentidos no sentido linguístico tradicional é objeto do protocolo de validação (seção 13.4) e permanece, no plano conceitual, uma questão aberta (seção 16).

**Exemplo** para negro@2020:

```
modo 1 (λ₁ = 0.41):  identidade(0.41), pertencimento(0.39), resistência(0.38)
modo 2 (λ₂ = 0.31):  discriminação(0.44), preconceito(0.42), exclusão(0.38)
modo 3 (λ₃ = 0.18):  cultura(0.45), ancestralidade(0.43), herança(0.40)
modo 4 (λ₄ = 0.04):  ...  ← ruído
```

---

## 8. Critério de Gap — Determinação Automática de τ e k

### 8.1 Forma invariante do gap relativo

A v1 normalizava a distribuição (P̃ = P/ΣP, λ̃ = λ/Σλ) antes de computar gaps relativos. Essa etapa é **matematicamente inerte** e foi removida: o gap relativo é invariante a reescalamento positivo, pois o fator de normalização cancela no quociente:

```
hᵢ = (Pᵢ − Pᵢ₊₁) / Pᵢ  =  (P̃ᵢ − P̃ᵢ₊₁) / P̃ᵢ
```

Remover a normalização também elimina um bug latente da v1: P_t(w) tem componentes negativas, de modo que ΣP pode ser pequeno ou de sinal instável, tornando P̃ mal definido. A forma não normalizada não tem esse problema.

A invariância tem uma segunda consequência, central para a implementação: **o gap relativo é computável exatamente sobre qualquer prefixo ordenado da distribuição** — não requer a distribuição completa. Isso licencia o cálculo sobre o top-k recuperado por busca aproximada (seção 14.1), sem distorção em nenhuma posição interna ao prefixo.

### 8.2 Determinação de τ

Ordena-se os valores positivos de P_t(w)[v] em ordem decrescente: P₁ ≥ P₂ ≥ ... > 0. O gap relativo entre posições consecutivas:

```
hᵢ = (Pᵢ − Pᵢ₊₁) / Pᵢ
```

τ corresponde ao valor de P na posição i* onde hᵢ é máximo, **sujeito à condição**:

```
h_{i*} > γ
```

onde γ é um limiar mínimo de validade (sugestão: γ = 0.3). Se nenhum gap exceder γ, a distribuição não tem estrutura clara — a palavra tem campo semântico difuso naquele período, e o procedimento sinaliza isso explicitamente em vez de retornar uma fronteira arbitrária.

Quando a distribuição é computada sobre um prefixo top-k truncado, a aceitação de τ exige adicionalmente as condições de segurança da seção 14.1.

### 8.3 Determinação de k

Sobre os autovalores ordenados λ₁ ≥ λ₂ ≥ ... ≥ 0 (obtidos da SVD de DÊ), o gap relativo:

```
gᵢ = (λᵢ − λᵢ₊₁) / λᵢ
```

k = índice i* onde gᵢ é máximo, sujeito a g_{i*} > γ. Se nenhum gap exceder γ, os modos não são distinguíveis espectralmente — a palavra é tratada como monossêmica naquele período.

**Exemplo** para negro@2020:

```
λ₁ = 0.41
λ₂ = 0.31   g₁ = (0.41−0.31)/0.41 = 0.24  < γ
λ₃ = 0.18   g₂ = (0.31−0.18)/0.31 = 0.42  > γ
λ₄ = 0.04   g₃ = (0.18−0.04)/0.18 = 0.78  ← máximo, > γ
λ₅ = 0.03

k = 3
```

### 8.4 γ como condição de estabilidade espectral

A condição γ não é apenas uma proteção contra fronteiras arbitrárias — ela coincide com a condição de **estabilidade dos autovetores** dada pelo teorema de Davis–Kahan: a perturbação de um autovetor sob perturbação E da matriz é controlada por ‖E‖ dividido pelo *eigengap* adjacente. Autovetores separados por gap pequeno são intrinsecamente instáveis (rotacionam livremente dentro do subespaço quase-degenerado); autovetores separados por gap grande são robustos.

Consequência operacional: **apenas modos delimitados por gaps > γ são rastreáveis individualmente entre períodos** (seção 11.6). O mesmo γ que valida a fronteira valida o rastreamento — um único princípio operando em três pontos do pipeline (τ, k, e elegibilidade de tracking).

### 8.5 Consistência do critério

| Nível | Distribuição | Gap determina |
|---|---|---|
| Filtragem | P_t(w)[v] (prefixo positivo ordenado) | τ → define V_w |
| Decomposição | λᵢ (autovalores ordenados) | k → define número de modos |
| Rastreamento | eigengaps adjacentes | elegibilidade de tracking individual |

O princípio geral do framework, do qual o gap é uma instância: *nenhuma fronteira é fixada a priori; toda fronteira emerge dos dados, acompanhada de uma condição de validade que sinaliza quando a estrutura não existe*. A calibração do limiar θ de emparelhamento (seção 11.4) é outra instância do mesmo princípio, com mecanismo diferente.

---

## 9. Atribuição de Ocorrências a Modos

Esta seção resolve a inconsistência da v1: a nuvem de embeddings era declarada como o input da decomposição, mas a matriz de coesão é construída a partir do perfil do embedding médio. Na v2, a divisão de trabalho é: a decomposição espectral identifica a **estrutura** dos modos no espaço de tokens; a atribuição de ocorrências conecta essa estrutura à **nuvem**.

### 9.1 Centróides de modo

Para cada modo i, define-se o centróide no espaço de embeddings como combinação dos embeddings dos tokens do modo, ponderada pelas cargas retificadas do autovetor:

```
a⁺ᵢ[v] = max( aᵢ[v], 0 )

cᵢ = Σ_v  a⁺ᵢ[v] · ê_t(v)      normalizado:  ĉᵢ = cᵢ / ‖cᵢ‖
```

A retificação descarta cargas negativas, que indicam oposição ao modo e não pertencimento.

### 9.2 Atribuição soft e hard

Cada ocorrência êᵏ_t(w) recebe responsabilidades proporcionais à sua similaridade retificada aos centróides:

```
rᵏᵢ = max( 0, cos( êᵏ_t(w), ĉᵢ ) )  /  Σⱼ max( 0, cos( êᵏ_t(w), ĉⱼ ) )
```

A atribuição **soft** é o padrão: ocorrências genuinamente ambíguas existem, e forçar atribuição dura nelas injeta ruído que não é culpa do método. A atribuição **hard** (argmax sobre i) é o caso degenerado, usado quando a comparação externa o exige — em particular na validação contra anotação de sentidos (seção 13.4), cujas métricas pressupõem rótulos discretos.

### 9.3 Frequência por modo

```
f_t(w, i) = Σ_k rᵏᵢ / N_t(w)
```

A distribuição { f_t(w, i) } responde "quanto do uso de w em t pertence a cada modo" — por exemplo, qual fração do uso de "negro" em 2020 é modo identidade vs. modo discriminação. A evolução de f ao longo da trajetória é um dos objetos descritivos centrais para a narrativa diacrônica: mudança semântica frequentemente se manifesta como redistribuição de massa entre modos persistentes, não apenas como surgimento de modos novos.

### 9.4 Embedding e perfil relacional por modo

```
e_t(w | i) = Σ_k  rᵏᵢ · êᵏ_t(w)  /  Σ_k rᵏᵢ        (normalizado)

P_t(w | i)[v] = cos( e_t(w | i), ê_t(v) )
```

P_t(w | i) é o perfil relacional **do modo i**, livre do colapso da média. Para palavras identificadas como monossêmicas (k = 1), P_t(w | 1) coincide com P_t(w) a menos de ruído. Para palavras polissêmicas, as trajetórias por modo (seção 11.7) operam sobre P_t(w | i), e o perfil agregado P_t(w) permanece disponível como medida global.

---

## 10. Comparação entre Dois Períodos

### 10.1 Diferença de perfil

```
ΔP(w, t₀→t₁)[v] = P_t₁(w)[v] − P_t₀(w)[v]     para todo v ∈ V
```

Cada componente diz quanto w se aproximou ou distanciou de v nesse intervalo.

**Exemplo** para negro entre 1960 e 2020:

```
ΔP[identidade]    = 0.91 − 0.12 = +0.79   ← aproximou fortemente
ΔP[pertencimento] = 0.88 − 0.08 = +0.80   ← aproximou fortemente
ΔP[discriminação] = 0.31 − 0.79 = −0.48   ← distanciou
ΔP[preconceito]   = 0.28 − 0.74 = −0.46   ← distanciou
ΔP[fotossíntese]  = 0.01 − 0.01 =  0.00   ← nunca foi relevante
```

### 10.2 Deslocamento semântico global

```
Δ(w, t₀, t₁) = 1 − cos( P_t₀(w), P_t₁(w) )
```

Resultado em [0, 2]:

| Valor | Significado |
|---|---|
| 0 | perfis idênticos, sem mudança |
| ≈ 1 | perfis ortogonais, campo semântico completamente reorganizado |
| 2 | perfis invertidos |

**Reporte calibrado (novo na v2):** Δ nunca é reportado como número absoluto isolado. Todo Δ é acompanhado de sua posição (percentil ou z-score) na distribuição do piso de drift (seção 13.1), que estabelece quanto Δ é esperado para palavras sabidamente estáveis sob o mesmo par de períodos. Exemplo de reporte:

```
Δ(negro, 1960, 2020) = 0.88   → percentil > 99 do piso de drift  → mudança forte
Δ(casa,  1960, 2020) = 0.02   → percentil 38 do piso de drift    → indistinguível de estável
```

---

## 11. Persistência de Modos entre Períodos

Esta seção resolve a principal questão aberta da v1. O problema: os autovetores de M_t(w) e M_{t+1}(w) não têm correspondência automática — um modo pode rotacionar, fragmentar ou colapsar entre períodos. Sem critério de continuidade, afirmações do tipo "o modo X em 1980 é o mesmo modo X em 1990" não têm fundamento, e a trajetória de modos não existe como objeto.

### 11.1 O fato estrutural: modos vivem no espaço de tokens

Um autovetor aᵢ de M_t(w) é um vetor de cargas sobre V_w(t) ⊆ V — não um vetor no espaço de coordenadas do encoder. Como V é fixo (seção 3), modos de períodos diferentes são **diretamente comparáveis**: basta estender cada vetor de cargas ao suporte comum

```
U = V_w(t) ∪ V_w(t+1)
```

preenchendo com zeros as posições fora do suporte original. Nenhum alinhamento de Procrustes é necessário — pela mesma razão que o perfil relacional o dispensa. Esta é a extensão natural da filosofia do framework do nível dos perfis para o nível dos modos.

### 11.2 Similaridade entre modos

Para o modo i de t (cargas estendidas ãᵢ) e o modo j de t+1 (cargas estendidas b̃ⱼ):

```
s(i, j) = cos( ã⁺ᵢ , b̃⁺ⱼ )
```

sobre cargas retificadas. Alternativa mais interpretável, recomendada para inspeção qualitativa: Jaccard ponderado sobre os conjuntos de top tokens de cada modo. As duas medidas devem concordar em casos claros; a definição canônica usa o cosseno.

### 11.3 Emparelhamento ótimo

Entre os k_t modos de t e os k_{t+1} modos de t+1, resolve-se o emparelhamento um-para-um que maximiza a soma das similaridades (algoritmo húngaro sobre a matriz s). Um par (i, j) do emparelhamento é **aceito** somente se:

```
s(i, j) ≥ θ
```

### 11.4 Calibração de θ por distribuição nula de permutação

θ não pode ser fixado arbitrariamente — isso reintroduziria o hiperparâmetro mágico que o framework foi desenhado para eliminar. Tampouco pode ser determinado por critério de gap sobre os scores do próprio emparelhamento: com k ≈ 3–4 modos de cada lado, há apenas uma dúzia de scores — material estatístico insuficiente para um gap significar algo.

A calibração canônica é por **distribuição nula de permutação**:

```
nula = { s( modo de w em t , modo de w′ em t′ ) :  w ≠ w′, pares amostrados sobre o léxico de estudo W }
```

isto é, similaridades entre modos de **palavras diferentes** — pares que sabidamente não são continuação um do outro. Então:

```
θ = quantil_{1−α} ( nula )         sugestão: α = 0.05
```

Interpretação estatística direta: a probabilidade de um emparelhamento espúrio ser aceito é < α. O nível α é uma convenção de significância declarada, não um número mágico: θ é calibrado pelos dados, e toda a arbitrariedade restante está concentrada num único parâmetro com semântica estatística padrão.

Isso preserva o princípio do framework (seção 8.5) com mecanismo distinto do gap: o gap exige uma distribuição com estrutura interna (centenas de similaridades, dezenas de autovalores); o emparelhamento não a tem, e a permutação fornece a referência externa que falta.

### 11.5 Eventos de trajetória: nascimento, morte, cisão, fusão

Após o emparelhamento, os modos não pareados e os pareamentos múltiplos residuais definem um vocabulário de eventos:

| Evento | Definição |
|---|---|
| **continuação** | par (i, j) aceito pelo emparelhamento com s ≥ θ |
| **morte** | modo de t sem nenhum candidato em t+1 com s ≥ θ |
| **nascimento** | modo de t+1 sem nenhum candidato em t com s ≥ θ |
| **cisão** | modo i de t com dois ou mais candidatos em t+1 acima de θ, com scores comparáveis (razão entre o segundo e o primeiro acima de ρ; sugestão: ρ = 0.7) |
| **fusão** | simétrico da cisão: dois ou mais modos de t convergindo para um modo de t+1 |

Cisões e fusões são detectadas examinando, após o emparelhamento um-para-um, as similaridades residuais acima de θ que o emparelhamento descartou.

Este vocabulário de eventos é um resultado descritivo em si: a história semântica de uma palavra passa a ser narrável como sequência tipada — "o modo discriminação persiste de 1960 a 2020 perdendo massa (f cai de 0.7 para 0.2); o modo identidade nasce nos anos 1980 e domina a partir de 2000" — com cada afirmação ancorada num critério explícito.

### 11.6 Estabilidade: quando o rastreamento individual é válido

Pelo argumento de Davis–Kahan (seção 8.4), autovetores associados a autovalores quase-degenerados são instáveis: rotacionam livremente dentro do subespaço do grupo, e compará-los individualmente entre períodos compara ruído.

Regra canônica:

- Um modo é **individualmente rastreável** apenas se está separado dos vizinhos espectrais por gaps relativos > γ de ambos os lados.
- Modos dentro de um grupo quase-degenerado (gaps ≤ γ entre si) são rastreados **como grupo**: compara-se o subespaço gerado pelas suas cargas, via ângulos principais entre subespaços, e o grupo inteiro é tratado como uma unidade de trajetória até que os autovalores se separem.

O mesmo γ da seção 8 governa esta regra — coerência interna, não um novo parâmetro.

Para robustez adicional, o rastreamento por sobreposição de conjuntos de tokens (comunidades no grafo de coesão) é menos sensível a rotação espectral que o rastreamento por autovetor, e serve como verificação cruzada nos casos limítrofes.

### 11.7 Cadeias de modos e trajetória por modo

Uma **cadeia de modos** é uma sequência maximal de continuações aceitas:

```
cadeia: modo i₀ @ t₀ → modo i₁ @ t₁ → ... → modo iₘ @ tₘ
```

possivelmente iniciada por um nascimento e terminada por uma morte, e possivelmente anotada com eventos de cisão/fusão nas extremidades. Para cada cadeia, a **trajetória do modo** é a sequência dos perfis relacionais por modo (seção 9.4):

```
T(w | cadeia) = [ P_t₀(w | i₀), P_t₁(w | i₁), ..., P_tₘ(w | iₘ) ]
```

acompanhada da série de frequências [ f_t₀(w, i₀), ..., f_tₘ(w, iₘ) ]. Todas as operações da seção 12 (distâncias, produtos internos, busca por trajetória similar) aplicam-se igualmente a trajetórias de modos, restritas à interseção dos períodos das cadeias comparadas.

---

## 12. Trajetória Semântica

### 12.1 Definição

Com três ou mais períodos, a trajetória semântica de w é a sequência completa de perfis relacionais:

```
T(w) = [ P_t₀(w), P_t₁(w), ..., P_tₙ(w) ]
```

T(w) representa a história semântica completa de w — não um snapshot, mas a evolução através de todos os períodos observados. Na v2, o objeto completo de trajetória de uma palavra é o par: trajetória do perfil agregado + conjunto de cadeias de modos com suas trajetórias e eventos (seção 11.7).

### 12.2 Diferença de trajetórias

```
T(w) − T(w′) = [ P_t₀(w) − P_t₀(w′),  P_t₁(w) − P_t₁(w′),  ... ]
```

O resultado é a trajetória da **divergência semântica** entre w e w′ ao longo do tempo.

**Exemplo:** T(negro) − T(branco) captura quando e em quais dimensões semânticas as duas palavras divergiram — não num único período, mas ao longo de toda a história observada.

### 12.3 Distância entre trajetórias

```
d( T(w), T(w′) ) = Σ_t  [ 1 − cos( P_t(w), P_t(w′) ) ]
```

Métrica bem definida no espaço de trajetórias: mede o quanto duas palavras estiveram semanticamente afastadas ao longo de toda a história observada.

### 12.4 Distância ponderada

```
d_α( T(w), T(w′) ) = Σ_t  α_t · [ 1 − cos( P_t(w), P_t(w′) ) ],     Σ_t α_t = 1
```

**Exemplo:** para estudar a divergência de "negro" e "preto" durante a redemocratização (1980–1990), atribui-se α_t alto para esses anos e baixo para os demais.

### 12.5 Similaridade de trajetória

```
⟨T(w), T(w′)⟩ = Σ_t  cos( P_t(w), P_t(w′) )
```

Mede o alinhamento semântico de duas palavras ao longo de todo o tempo observado.

### 12.6 Busca por trajetória similar

```
vizinhos_históricos(w) = argmin_{w′ ∈ W}  d( T(w), T(w′) )
```

"Quais palavras fizeram uma história semântica similar à de w?" — fundamentalmente diferente de buscar palavras similares num período: é buscar palavras que **percorreram caminhos semânticos análogos**.

**Exemplo:**

```
vizinhos_históricos(negro) → {queer, gay, mulher, ...}
```

Palavras que passaram por processos similares de reapropriação semântica, independentemente de coabitarem um campo semântico em qualquer período específico. A versão por modos (sobre T(w | cadeia)) refina a busca: trajetórias de **usos** análogos, não apenas de palavras agregadas.

### 12.7 Distância local vs. global

A distância d agrega todos os períodos e pode obscurecer divergências localizadas. O perfil de distância período a período:

```
δ_t(w, w′) = 1 − cos( P_t(w), P_t(w′) )     para cada t
```

revela **quando** a divergência ocorreu — não apenas quanto divergiram no total. O custo passa de O(1) para O(n) por par; sobre representação esparsa (seção 14.2) isso permanece barato.

---

## 13. Protocolo de Validação e Calibração

O framework garante analiticamente a invariância da seção 4.2 e elimina a translação por construção. O que resta — drift anisotrópico/não-linear do encoder sob treinamento contínuo, e a correspondência entre modos e sentidos — não é assumido: é medido pelos experimentos abaixo. Os três primeiros calibram a hipótese de encoder compartilhado; o quarto valida a decomposição em modos.

### 13.1 Piso de drift (conjunto de controle estável)

**Construção do controle:** palavras cuja estabilidade semântica é defensável a priori — substantivos concretos, numerais, termos técnicos estáveis (elementos químicos, partes do corpo). Palavras funcionais **não** servem: são contextuais demais e seus embeddings carregam pouca semântica lexical.

**Procedimento:** computar Δ(w_estável, t₀, t₁) para todo o conjunto de controle, para cada par de períodos de interesse. A distribuição resultante é o **piso de drift** — quanto Δ é gerado por drift residual + ruído de amostragem na ausência de mudança semântica real.

**Uso:** todo Δ de palavra de interesse é reportado como percentil/z-score nessa distribuição (seção 10.2). Mudança semântica é afirmada apenas quando Δ é extremo relativo ao piso.

**Condição de aplicabilidade:** impor limiar mínimo de ocorrências por período (a definir empiricamente via bootstrap — seção 16); palavras de controle e de interesse devem satisfazê-lo igualmente.

### 13.2 Split-half intra-período (limite inferior de detecção)

Dividir o corpus de **um mesmo período** em duas metades aleatórias; rodar o treinamento contínuo em cada metade; medir Δ entre as metades para o conjunto de controle e para o léxico de estudo.

Como não há mudança semântica real possível entre as metades, **qualquer Δ medido é artefato** (drift de treino + variância de amostragem). Esse é o limite inferior honesto do que o método consegue detectar: diferenças de perfil abaixo desse nível são indistinguíveis de ruído por construção.

### 13.3 Shuffle temporal (teste de vazamento)

Embaralhar os rótulos temporais dos documentos e retreinar a sequência de períodos sintéticos. Sob shuffle, Δ deve colapsar para o piso de ruído do split-half para todas as palavras. Se Δ permanecer elevado para alguma classe de palavras, há vazamento de artefato de treino (ordem de apresentação, curriculum implícito) contaminando a medida — e o artefato fica identificado antes de contaminar resultados substantivos.

### 13.4 Validação de modos contra anotação de sentidos

**Mudança global:** benchmark SemEval-2020 Task 1 (detecção graded de mudança semântica; inglês, alemão, latim, sueco; anotação DWUG). Valida Δ contra julgamento humano de mudança — e, indiretamente, a robustez ao drift, pois o benchmark exige comparação entre períodos reais.

**Modos vs. sentidos:** sobre um corpus com anotação de sentidos, comparar as atribuições hard de ocorrências a modos (seção 9.2) com os rótulos anotados, via métricas padrão de Word Sense Induction: B-Cubed e V-measure.

**Português:** dado que recursos anotados em PT são escassos, o caminho canônico é duplo — (i) validar o método nos benchmarks existentes em outras línguas, estabelecendo que o pipeline funciona; (ii) anotar manualmente uma amostra pequena em PT (algumas dezenas de palavras × ~50 ocorrências cada), suficiente para reportar B-Cubed/V-measure no domínio de aplicação.

**Critério de sucesso declarado:** a concordância entre anotadores humanos sobre inventários de sentidos é notoriamente baixa. O framework não precisa demonstrar que modos *são* sentidos; precisa demonstrar que modos correlacionam com julgamentos humanos de uso **tanto quanto julgamentos humanos correlacionam entre si**. Esse é o teto empírico realista de qualquer método de indução de sentidos, e é contra ele que os números devem ser lidos.

### 13.5 Resumo do protocolo

| Experimento | O que mede | O que requer |
|---|---|---|
| Piso de drift (13.1) | drift residual + amostragem entre períodos reais | nenhum retreino adicional |
| Split-half (13.2) | limite inferior absoluto de detecção | retreino sobre metades de um período |
| Shuffle temporal (13.3) | vazamento de artefato de treino | retreino sob permutação |
| SemEval / WSI (13.4) | correspondência com julgamento humano | corpora de benchmark + amostra PT anotada |

Ordem de prioridade quando recursos são limitados: 13.1 (sem retreino, calibra todos os reportes) → 13.4 (valida a contribuição central) → 13.2 → 13.3.

---

## 14. Custo Computacional e Implementação

### 14.1 Perfis via busca aproximada (ANN)

P_t(w) ∈ ℝ^|V| denso, para todo w ∈ W e todo período, é proibitivo e desnecessário: o critério de gap só consulta o topo da distribuição ordenada, e a cauda quase-zero é descartada pela própria definição de τ.

**Pipeline canônico:**

1. Centralizar e normalizar todos os embeddings do período (seção 4). Cosseno vira produto interno.
2. Indexar { ê_t(v) : v ∈ V } num índice ANN (HNSW / FAISS-IVF ou equivalente), uma vez por período.
3. Para cada w ∈ W, recuperar os top-k vizinhos de e_t(w), com k inicial k₀ (sugestão: k₀ = 1024).
4. Aplicar o critério de gap (seção 8.2) **dentro do prefixo recuperado** — exato, pela invariância da seção 8.1.

**Condições de segurança do truncamento** (ambas exigidas para aceitar τ):

```
(i)  i* < 0.8 · k         o gap máximo está confortavelmente interno ao prefixo
(ii) P[v_k] / P[v_1] < ε   a similaridade na borda já é fração negligível do topo
                            (sugestão: ε = 0.05)
```

A condição (ii) é livre de escala e computável inteiramente dentro do top-k — não requer a soma da distribuição completa. Se qualquer condição falha, re-consultar com 2k e repetir. Na prática, a esparsidade natural (seção 5.4) faz a primeira consulta bastar na esmagadora maioria dos casos.

Custo por palavra-período: O(k log |V|) contra O(|V|) do cálculo denso.

### 14.2 Representação esparsa e comparações

P_t(w) é armazenado como vetor esparso com suporte = top-k recuperado. Cossenos entre perfis (Δ, δ_t, distâncias de trajetória) são produtos internos esparsos sobre a **união dos suportes** dos dois perfis.

**Viés de truncamento:** componentes fora do suporte são ≈ 0 em ambos os perfis, e sua contribuição ao produto interno é o produto de dois quase-zeros — desprezível. Protocolo: validar numa amostra de pares contra o cálculo denso exato e reportar o erro de aproximação observado, uma vez, como parte da documentação de implementação.

### 14.3 Decomposição sem formar M

Conforme a seção 7.5: SVD fina de DÊ ∈ ℝ^{|V_w| × d}. Nunca formar M.

### 14.4 Complexidades resumidas

| Operação | Denso / ingênuo | Canônico v2 |
|---|---|---|
| Perfil de uma palavra num período | O(\|V\| · d) | O(k log \|V\|) via ANN |
| Armazenamento de um perfil | O(\|V\|) | O(k) esparso |
| Decomposição em modos | O(\|V_w\|³) + O(\|V_w\|²) memória | O(\|V_w\| · d²) via SVD de DÊ |
| Cosseno entre dois perfis | O(\|V\|) | O(\|suporte₁ ∪ suporte₂\|) |
| Emparelhamento de modos entre períodos | — | O(k_t · k_{t+1}) similaridades + húngaro (negligível) |

### 14.5 Léxico de estudo

Perfis e decomposições são computados apenas para w ∈ W — o léxico de estudo —, não para todo V. O índice ANN cobre V inteiro (qualquer v pode aparecer como vizinho), mas o lado "consulta" do pipeline é restrito a W e computado sob demanda.

---

## 15. O que este Framework Não Requer

| Escolha eliminada | Razão |
|---|---|
| Alinhamento de Procrustes entre períodos | Perfis **e modos** definidos sobre V fixo, não sobre coordenadas internas (seções 3, 11.1) |
| k fixado a priori | k emerge do gap espectral com condição de validade γ (8.3) |
| Threshold τ fixado | τ emerge do gap na distribuição de similaridades (8.2) |
| Limiar de emparelhamento θ fixado | θ calibrado por distribuição nula de permutação com nível de significância declarado (11.4) |
| Normalização da distribuição antes do gap | Etapa inerte — o gap relativo é invariante a reescalamento positivo (8.1) |
| Lista de âncoras pré-definidas | Palavras estáveis emergem como dimensões com ΔP ≈ 0; o controle da seção 13.1 é para calibração de reporte, não pré-requisito do método |
| Filtragem manual por POS | Tokens irrelevantes têm P_t(w)[v] ≈ 0 naturalmente |
| Comparação restrita a dois períodos | Trajetória opera sobre a história completa (12), inclusive por modo (11.7) |
| Materialização de M ou de P denso | SVD de DÊ (7.5); ANN + representação esparsa (14) |
| Alegação não qualificada de invariância | Classe de invariância provada (4.2); translação eliminada por centralização (4.1); resíduo medido (13) |

---

## 16. Questões Abertas

**Limiar mínimo de ocorrências e estabilidade com palavras raras.** Com poucas ocorrências em t, a nuvem é pequena, DÊ é ruidosa e os eigengaps são instáveis. O limiar mínimo de ocorrências por período permanece a definir. Protocolo sugerido: bootstrap sobre as ocorrências (reamostrar a nuvem, recomputar autovalores, medir a variância dos gaps) para estimar, por corpus, o N mínimo a partir do qual o critério de gap é estável.

**Modos semânticos vs. sentidos lexicais (questão conceitual).** O protocolo da seção 13.4 fornece a validação empírica, mas a questão conceitual permanece: comunidades espectrais coesas e sentidos lexicais no sentido linguístico tradicional são objetos de naturezas diferentes, e a correspondência observada empiricamente não estabelece identidade. A posição canônica do framework é deflacionária: modos são o objeto definido e medido; "sentido" é uma interpretação externa a ser ganha caso a caso.

**Parâmetros menores da atribuição de ocorrências.** A retificação na atribuição soft (seção 9.2) é a escolha mais simples; alternativas (softmax com temperatura) introduzem um hiperparâmetro adicional. A comparação fica como ablação.

**Whitening por período.** A centralização elimina translação e parte da anisotropia; whitening completo eliminaria mais, ao custo de amplificar ruído em direções de baixa variância. Fica como ablação contra o padrão (centralização apenas).

**Drift residual não-linear.** O protocolo da seção 13 mede o drift residual, mas não o corrige. Se o piso de drift se mostrar alto demais relativo aos efeitos de interesse, técnicas de correção (regularização de continuidade no treino, replay de corpus antigo) entram em cena — são intervenções no treino do encoder, fora do escopo desta definição.

**Formalização do espaço de trajetórias.** T(w) tem estrutura de espaço métrico com produto interno (12.3, 12.5). A formalização completa — propriedades topológicas, completude, relação entre a métrica agregada e a família {δ_t} — é extensão natural fora do escopo desta definição canônica.

**Sensibilidade de ρ na detecção de cisões/fusões.** O parâmetro ρ (razão entre candidatos comparáveis, seção 11.5) é o único limiar do pipeline ainda fixado por convenção sem calibração própria. Candidato natural: calibrá-lo pela mesma distribuição nula da seção 11.4. Fica registrado como pendência.

---

## Apêndice A — Resumo do pipeline

```
por período t:
  1. encoder → embeddings contextuais de corpus_t
  2. centralizar (μ_t) e normalizar                                  [§4]
  3. indexar ê_t(v), v ∈ V, em índice ANN                            [§14.1]

por palavra w ∈ W, por período t:
  4. top-k vizinhos → prefixo ordenado de P_t(w)                     [§14.1]
  5. gap relativo → τ → V_w   (com condições de segurança)           [§8.2, §14.1]
  6. SVD de DÊ → autovalores λᵢ, autovetores aᵢ                      [§7.5]
  7. gap relativo → k → modos semânticos                             [§8.3]
  8. centróides de modo → atribuição soft das ocorrências            [§9]
  9. frequências por modo f_t(w,i); perfis por modo P_t(w|i)         [§9.3–9.4]

entre períodos consecutivos:
  10. estender cargas a V_w(t) ∪ V_w(t+1); similaridades s(i,j)      [§11.1–11.2]
  11. emparelhamento húngaro; aceitar pares com s ≥ θ (θ via nula)   [§11.3–11.4]
  12. eventos: nascimento, morte, cisão, fusão                       [§11.5]
  13. cadeias de modos → trajetórias por modo                        [§11.7]

global:
  14. trajetórias T(w); distâncias, ponderações, vizinhos históricos [§12]
  15. calibração: piso de drift; reporte por percentil/z             [§13.1, §10.2]
```