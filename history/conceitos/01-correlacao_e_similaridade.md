# Conceitos 1 — Correlação, similaridade e informação

Este arquivo reúne as ferramentas matemáticas mais básicas usadas em quase
todos os capítulos: como comparar dois números, duas distribuições, ou
duas representações geométricas. Sempre que possível, os exemplos usam os
quatro personagens do projeto: `plane_nn`, `chairman_nn`, `graft_nn` e
`tree_nn`.

<a id="cosseno"></a>
## Similaridade e distância de cosseno

A similaridade de cosseno compara a **direção** de dois vetores, ignorando
grande parte da diferença de magnitude:

```text
cos(x, y) = (x . y) / (||x|| ||y||)
```

- `cos = 1`: mesma direção;
- `cos = 0`: direções ortogonais;
- `cos = -1`: direções opostas.

A distância usada frequentemente no projeto é:

```text
distancia_cosseno(x, y) = 1 - cos(x, y)
```

No projeto, cosseno aparece em perfis relacionais, APD entre ocorrências,
matching contexto-sentido no LMMS, estabilidade de frases-âncora e
comparação de vizinhanças.

O cosseno não torna automaticamente dois espaços comparáveis. Ele é
invariante a uma rotação comum aplicada a todos os vetores de um espaço, mas
comparar checkpoints diferentes ainda pode misturar deformação do encoder com
mudança do objeto. Ver a
[grade checkpoint x corpus](08-desenhos_temporais_e_reguas.md).

<a id="spearman"></a>
## Spearman

A correlação de Spearman (`rho`, ou `Spearman's rho`) mede **o quanto duas
listas concordam em ordenação**, não em valores exatos.

Como calcular, na prática:

1. Ordene as 37 palavras pelo seu score do modelo (por exemplo, APD).
2. Ordene as mesmas 37 palavras pelo gold do SemEval (`graded`).
3. Para cada palavra, compare sua posição (rank) nas duas listas.
4. `rho = 1` significa "as duas ordens são idênticas"; `rho = -1` significa
   "as ordens são exatamente opostas"; `rho = 0` significa "não há relação
   sistemática entre as ordens".

Por que Spearman, e não correlação de Pearson (que compara valores
diretamente)? Porque o que o projeto quer responder é "o modelo concorda
com o gold sobre **quais** palavras mudaram mais"? — uma pergunta de
**ranking**, não de magnitude. Uma métrica pode ter valores numéricos
muito diferentes do gold (escalas diferentes) e ainda assim ter Spearman
alto, se a ordem relativa estiver certa.

### Quão confiável é um Spearman com `n=37`?

O erro padrão aproximado de uma correlação de Spearman é:

```text
SE(rho) ~ 1/sqrt(n - 3)
```

Com `n=37`: `SE ~ 1/sqrt(34) ~ 0,17`. Isso significa que uma diferença
entre `rho=0,12` e `rho=0,20` está **dentro de uma margem de ruído
plausível** — não é, por si só, evidência de que um método é melhor que
outro. Esse cálculo foi o ponto de virada do capítulo 09: várias
"melhorias" de poucos centésimos discutidas nos capítulos 06-08 não
resistem a essa régua. Veja
[bootstrap](05-estatistica_experimental.md#bootstrap) para a forma mais
robusta de quantificar essa incerteza (em vez de uma fórmula aproximada).

<a id="pmi"></a>
## PMI

PMI (Pointwise Mutual Information, "informação mútua pontual") responde à
pergunta: **"`v` aparece ao lado de `w` mais do que seria esperado por
puro acaso de frequência?"**

```text
PMI(w, v) = log( P(v | w) / P(v) )
```

- `P(v | w)`: com que frequência `v` aparece em contextos de `w`.
- `P(v)`: com que frequência `v` aparece em geral, independente de `w`.
- Se `PMI(w, v) > 0`: `v` é mais comum perto de `w` do que em geral — `v`
  é um "marcador" de `w`.
- Se `PMI(w, v) ≈ 0`: nenhuma associação especial.
- Se `PMI(w, v) < 0`: `v` é *menos* comum perto de `w` — repulsão.

No capítulo 04, essa ideia foi adaptada para a saída do MLM head:
`R_t(w)[v] = log(q_t(w)[v] / p_t[v])`, onde `q_t(w)` é "o que o modelo
prevê no lugar de `w`" e `p_t` é "o que o modelo prevê em geral". Essa é
literalmente a fórmula do PMI, com `q_t(w)` no papel de `P(v|w)` e `p_t`
no papel de `P(v)`.

**Exemplo esperado** (capítulo 04): para `graft_nn` em D1 (1960-2010), o
perfil log-PMI deveria ter valores positivos simultaneamente para palavras
do campo botânico ("scion", "stock") *e* do campo médico/corrupção
("transplant", "bribery") — um perfil "mais largo" do que em D0, refletindo
a diversificação de sentido.

**PPMI** ("Positive PMI") é o PMI truncado em zero: `max(0, PMI(w,v))`.
Serve para descartar repulsões (que tendem a ser ruidosas) e manter só
associações positivas, antes de normalizar para uma distribuição (ver
[Jensen-Shannon](#jensen-shannon)).

<a id="jensen-shannon"></a>
## Jensen-Shannon

A divergência de Jensen-Shannon (JSD) mede **a distância entre duas
distribuições de probabilidade**, e tem duas propriedades convenientes:
é simétrica (`JSD(P,Q) = JSD(Q,P)`, diferente da KL-divergence) e é
limitada (entre 0 e `log(2)`, ou entre 0 e 1 se usar log na base 2).

```text
JSD(P, Q) = 1/2 * KL(P || M) + 1/2 * KL(Q || M),   onde M = (P+Q)/2
```

Intuição: `JSD = 0` significa "as duas distribuições são idênticas";
`JSD` alto significa "as duas distribuições têm massa de probabilidade
concentrada em lugares muito diferentes".

No projeto, JSD aparece em dois lugares principais:

1. **`Delta_JSD(w)`** (capítulo 04): compara o perfil PPMI normalizado de
   `w` em D0 com o de D1. Para `plane_nn`, espera-se `Delta_JSD` alto (os
   marcadores positivos em D0 — geometria — e em D1 — aviação — são quase
   disjuntos); para `tree_nn`, espera-se `Delta_JSD` baixo.
2. **`Delta_usage(w) = JSD(P(cluster|w,D0), P(cluster|w,D1))`** (capítulo
   06 e protótipo "modos primeiro" do capítulo 08): em vez de comparar
   perfis de vizinhos, compara **a proporção de ocorrências de `w` em cada
   cluster/modo** entre D0 e D1. `graft_nn` teve o maior `JSD` desse tipo
   no capítulo 08 (0,473), refletindo que a composição de modos mudou de
   "majoritariamente modo botânico" para "majoritariamente modo
   comercial/tecnológico".

<a id="entropia"></a>
## Entropia

A entropia de uma distribuição mede **quão "espalhada"/incerta ela é**:

```text
H(P) = - sum_v P(v) * log(P(v))
```

- Se `P` está concentrada num único valor (`P(v*)=1`, todo o resto 0),
  `H(P) = 0` — totalmente previsível.
- Se `P` é uniforme sobre `|V|` valores, `H(P) = log(|V|)` — o máximo
  possível, totalmente imprevisível.

**Entropia normalizada**: `H(P) / log(|V|)`, entre 0 e 1, facilita
comparar distribuições sobre vocabulários de tamanhos diferentes.

**Perplexidade**: `exp(H(P))` (ou `2^H(P)` em log base 2) — pode ser lida
como "o modelo está, em média, tão incerto quanto se escolhesse
uniformemente entre `perplexidade` opções". Um BERT-base bem treinado tem
perplexidade ~4-6 em texto em inglês; o modelo do capítulo 05, com
perplexidade ~120-130, estava "tão incerto quanto escolher entre 120-130
palavras igualmente prováveis" — evidência direta de subtreinamento.

No capítulo 05, `q_t(graft\_nn)` tinha entropia normalizada `H/log|V| ≈
0,78` — perto do máximo de 1,0 — o que significava que o modelo "não tinha
opinião" sobre o que deveria substituir `graft_nn`, mesmo em contextos
inequívocos.

<a id="softmax"></a>
## Softmax

A função softmax transforma um vetor de números reais quaisquer (chamados
**logits**) numa distribuição de probabilidade — valores entre 0 e 1, que
somam 1:

```text
softmax(z)_i = exp(z_i) / sum_j exp(z_j)
```

É a função usada na última camada de um MLM head: o modelo produz um
"score" (logit) para cada token do vocabulário, e o softmax converte esses
scores em `q_t(w)[v]` — "a probabilidade que o modelo atribui ao token `v`
no lugar de `w`". É essa distribuição que entra na fórmula do
[PMI](#pmi) do capítulo 04.

Uma propriedade importante do softmax para este projeto: ele é **invariante
a deslocamentos constantes** — somar uma constante `c` a todos os logits
não muda a distribuição resultante (`softmax(z+c) = softmax(z)`). Isso é
parte de por que comparar distribuições de saída (em vez de logits ou
vetores ocultos crus) ajuda a cancelar certos tipos de deriva entre
checkpoints (capítulo 04, propriedade "normalização por deriva de
domínio").

<a id="cka"></a>
## CKA

CKA (Centered Kernel Alignment) é uma métrica para comparar **dois
conjuntos de representações** (por exemplo, os vetores ocultos de
`theta_0` e de `theta_1` para as mesmas frases) e responder "quão
parecidas são essas duas geometrias, a menos de rotação/reflexão/escala?".

Intuição: para cada conjunto de representações, calcula-se a matriz de
similaridades entre todos os pares de exemplos (um "kernel"); depois
centraliza-se essa matriz (remove a média) e compara-se as duas matrizes
centralizadas via um produto interno normalizado:

```text
CKA(X, Y) = || centralizar(K_X) . centralizar(K_Y) || / (|| K_X || * || K_Y ||)
```

`CKA = 1` significa que as duas geometrias são equivalentes a menos de uma
transformação ortogonal e reescalonamento; `CKA = 0` significa nenhuma
relação estrutural. No início do projeto (capítulos 01-03), CKA foi uma
das métricas candidatas para comparar representações entre checkpoints —
ela é mencionada como referência de "comparação invariante a sistema de
coordenadas", uma propriedade que o **perfil relacional** (capítulos 04 e
08) persegue por um caminho diferente (centralização + cosseno sobre um
vocabulário compartilhado, em vez de uma matriz de Gram completa entre
exemplos).
