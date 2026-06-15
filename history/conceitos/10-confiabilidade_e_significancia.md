# Conceitos 10 — `D_obs` é grande? Comparado com o quê?

Os capítulos 26-27 recolocaram `token@time` no centro do projeto: para cada
palavra `w`, comparamos o perfil relacional `R_0(w)` (D0, 1810-1860) com
`R_1(w)` (D1, 1960-2010) e calculamos

```text
D_obs(w) = 1 - cos(R_0(w), R_1(w))
```

Esse número por si só não responde a nada. `D_obs(plane) = 0.0655` —
é muito? É pouco? Comparado com quem? Este arquivo reúne as ferramentas que o
capítulo 29 usa para transformar `D_obs(w)` num número interpretável, com uma
régua de ruído própria de cada palavra.

<a id="o-problema"></a>
## O problema: `D_obs` mistura sinal e ruído de amostragem

`R_t(w)` é calculado a partir de um **número finito** de ocorrências de `w`
no corpus daquele período: 301 para "plane" em D0, 827 em D1; mas só 64 para
"graft" em D1. Mesmo que o "sentido verdadeiro" de uma palavra não tivesse
mudado nem um pouco entre os dois períodos, `R_0(w)` e `R_1(w)` ainda
difeririam — porque são médias de amostras diferentes, de tamanhos
diferentes, de textos diferentes.

Analogia: se você pesa uma pessoa de manhã e de noite numa balança de
banheiro, vai ver números um pouco diferentes mesmo que o peso real não tenha
mudado — a balança tem ruído. Antes de dizer "essa pessoa engordou 0,3 kg",
você precisa saber: **qual é o ruído típico desta balança, para este tipo de
medição?** Um `Delta` de 0,3 kg pode ser enorme numa balança de precisão de
laboratório e ser pura flutuação numa balança de banheiro.

`D_obs(w)` é o "Delta" lido na balança. As próximas seções constroem a régua
de ruído — específica para cada palavra `w`, porque palavras com poucas
ocorrências (`graft`) têm uma balança muito mais "ruidosa" que palavras
frequentes (`chairman`, indo de 147 para 683 ocorrências).

<a id="nulo-b"></a>
## Nulo B: o que `D_obs(w)` pareceria se não houvesse mudança real

A ideia do **nulo B** (`document_permutation_null`,
`src/tracoformer/token_time_null.py`) é simular a pergunta: *"se as
ocorrências de `w` em D0 e D1 viessem todas do **mesmo** lugar — se a
diferença entre os dois grupos fosse só uma divisão arbitrária dos mesmos
dados — qual `D` eu mediria?"*

Procedimento, para uma palavra `w`:

1. Junte todas as ocorrências de `w` em D0 e D1 num único poço.
2. Agrupe por **documento** (todas as ocorrências de `w` no mesmo documento
   se movem juntas — preserva o fato de que ocorrências do mesmo texto não são
   independentes).
3. Embaralhe os documentos e redivida o poço em dois grupos do mesmo tamanho
   que os originais (`n0` e `n1`).
4. Recalcule o centróide de `w` em cada grupo pseudo-aleatório, monte os
   perfis relacionais (mantendo `mu_0`, `mu_1` e as referências fixas nos
   valores **observados** — só a linha de `w` muda) e calcule
   `D = 1 - cos(R_pseudoA, R_pseudoB)`.
5. Repita `B` vezes (ex. `B=200`). Isso dá uma distribuição `D_null` — os
   valores de `D` que você obteria "por acaso", sem nenhuma diferença real
   entre os grupos.

A partir de `D_null`, duas estatísticas resumem onde `D_obs(w)` cai:

```text
Z_robusto = (D_obs - mediana(D_null)) / (1.4826 * MAD(D_null))
p = (1 + #{D_null >= D_obs}) / (B + 1)
```

`MAD` (median absolute deviation, mediana dos desvios absolutos em relação à
mediana) é como um desvio-padrão, mas menos sensível a outliers; o fator
`1.4826` faz com que `1.4826*MAD` seja comparável a um desvio-padrão para
dados aproximadamente normais. `p` é o "one-sided": a fração das permutações
que produziram `D_null` tão grande quanto `D_obs` (ou maior) — quanto menor,
mais incomum é `D_obs` sob o nulo.

**Resultado real (seed1000, `layer_2`, `B=200`)**:

| palavra | `D_obs` | mediana(`D_null`) | MAD(`D_null`) | `Z_robusto` | `p` |
|---|---:|---:|---:|---:|---:|
| plane | 0.0655 | 0.0217 | 0.0016 | +18.10 | 0.005 |
| chairman | 0.0554 | 0.0261 | 0.0012 | +16.54 | 0.005 |
| graft | 0.0344 | 0.0227 | 0.0009 | +9.04 | 0.005 |

`p=0.005` é o menor valor possível com `B=200` (`1/(200+1)`) — as três
palavras estão tão fora da distribuição nula observada que nenhuma das 200
permutações chegou perto de `D_obs`. Isso já é evidência de que algo muda
entre D0 e D1 para essas palavras — mas "algo muda" ainda não é "o sentido
mudou": ver [resíduo de pré-treino](#residuo-pre-treino) abaixo.

<a id="z-robusto-vs-dobs"></a>
## Por que `Z_robusto` é uma métrica melhor que `D_obs` cru

A pergunta natural é: já que `Z_robusto` é só `D_obs` reescalado, por que se
incomodar? A resposta vem de comparar os dois com o gold do SemEval2020 (37
palavras, `truth.tsv`):

| | seed1000 | seed1001 |
|---|---|---|
| Spearman(`D_obs`, graded) | 0.061 (p=0.72) | 0.068 (p=0.69) |
| Spearman(`Z_robusto`, graded) | **0.269** | **0.295** |
| point-biserial(`binary`, `D_obs`) | 0.201 (p=0.23) | 0.209 (p=0.21) |
| point-biserial(`binary`, `Z_robusto`) | **0.334** (p=0.04) | **0.342** (p=0.04) |

`D_obs` cru é quase descorrelacionado do julgamento humano de mudança
(`rho~=0.06`). `Z_robusto` chega a `rho~=0.27-0.30` e a correlação ponto-bisserial
com o rótulo binário (`mudou`/`não mudou`) passa a ser significativa a 5%.

Por que dividir pelo `MAD(D_null(w))` ajuda tanto? Porque `D_obs(w)` mistura
duas coisas: **quanto `w` de fato se deslocou** e **quão ruidosa é a
estimativa do centróide de `w`**, que depende fortemente de `n0`/`n1` (de
50 a >5000 ocorrências entre as 37 palavras). Palavras raras têm `D_obs`
inflado mesmo sem mudança real, simplesmente porque a média de poucos
pontos é instável. `MAD(D_null(w))` mede exatamente essa instabilidade
*daquela palavra*, e dividir por ela remove boa parte dessa heterogeneidade —
o que resta se aproxima mais da mudança semântica real.

Isso é a mesma lição do [nulo específico por
palavra](05-estatistica_experimental.md#nulo-por-palavra), aplicada
concretamente ao `token@time`.

<a id="pseudo-periodos"></a>
## Calibração com pseudo-períodos: o nulo está calibrado?

Antes de confiar no nulo B para as 37 palavras reais, é preciso perguntar:
**o próprio nulo está bem calibrado?** Ou seja, se eu aplicar todo o
protocolo a um par onde sei que não há mudança real nenhuma, ele me dá falsos
positivos?

Teste: dividir os ~253 mil documentos de `1810-1860.txt` aleatoriamente
(seed fixa) em duas metades `pseudo_a`/`pseudo_b` — **sem nenhuma diferença
cronológica real**, ambas são amostras do mesmo período. Repetir o protocolo
completo (extrair perfis, rodar nulo B, calcular `D_obs`/`Z_robusto`/`p` para
as 37 palavras) sobre esse par pseudo.

Resultado (B=200, ambos os seeds):

| métrica | seed1000 | seed1001 |
|---|---|---|
| mean percentile | 0.371 | 0.383 |
| FP rate @ alpha=0.05 | 0/37 | 0/37 |
| FP rate @ alpha=0.10 | 0/37 | 0/37 |

Zero falsos positivos em ambos os seeds — não há evidência de que o nulo B
seja "estreito demais" (o que inflaria a taxa de falsos positivos). Se algo,
o nulo é levemente conservador (`mean percentile` um pouco abaixo de 0.5).

Um achado lateral interessante: `MAD(D_null)` no par pseudo é
**menor** (~0.58x) que `MAD(D_null)` no par real D0-vs-D1, mesmo que cada
metade pseudo tenha menos documentos que D0 inteiro. A explicação: no nulo B,
`mu_t` e as centroides de referência ficam fixos nos valores **observados**
de cada período (só a linha de `w` é recomputada na permutação). Entre
`pseudo_a` e `pseudo_b` (mesmo século), esses sistemas de referência são quase
idênticos; entre D0 e D1 (séculos diferentes), o vocabulário inteiro mudou, e
essa diferença "de fundo" entra no nulo B junto com o ruído amostral de `w`.
Ou seja: **o nulo B não mede só "quão instável é o centróide de `w`"** — ele
também herda o quanto os dois sistemas de referência `(mu_t, V_referência)`
diferem entre os períodos comparados. Isso é discutido com mais detalhe em
`tmp/36` (passo 5); não invalida o uso do nulo B no passo 7, mas é um
lembrete de que "nulo" aqui significa "o que se observaria sob este desenho
específico de permutação", não "ausência de qualquer diferença entre D0 e
D1".

<a id="split-half"></a>
## Split-half: `D_obs(w)` é uma propriedade do corpus ou de poucos documentos?

O nulo B responde "isso é maior do que o acaso explicaria?". O **split-half**
responde uma pergunta diferente e complementar: "esse resultado se sustenta
se eu olhar para metades diferentes dos dados?" — não é um nulo (não simula
"ausência de mudança"), é um diagnóstico de **repetibilidade**.

Procedimento (`split_half_displacement`,
`scripts/token_time/split_half_repeatability.py`): para cada palavra `w`,
divida as ocorrências de `w` em D1 em duas metades aleatórias por documento
(`OccurrenceCache.doc_index`), recompute o centróide de `w` em cada metade
(mantendo `mu_1` e as referências fixas no valor observado de D1 inteiro,
mesmo truque do nulo B) e compare `D_half1`, `D_half2` contra `R_0(w)` e
contra `D_obs` (D1 completo).

Resultado (B=1 split, ambos os seeds, n=37):

| | seed1000 | seed1001 |
|---|---|---|
| Spearman(`D_half1`, `D_half2`) | 0.958 | 0.962 |
| Spearman(`D_obs`, mean(`D_half1`,`D_half2`)) | 0.999 | 0.996 |

As duas metades independentes do corpus de D1 concordam fortemente entre si e
com a amostra completa, para as 37 palavras. Palavras raras (`graft`, ~64
ocorrências por metade; `lass`, 65) mostram a maior variação *absoluta* entre
metades, mas a *ordenação* relativa entre as 37 palavras (o que importa para
Spearman/ranking) é preservada. Conclusão prática: `D_obs(w)` não é dominado
por um punhado de documentos — é uma propriedade estável do corpus, mesmo
para as palavras menos frequentes da lista.

<a id="lexical-validity"></a>
## `lexical_validity`: filtrar fragmentos de WordPiece das referências

Esta seção depende do conceito de
[WordPiece](09-dados_tokenizacao_e_contexto.md#token-do-projeto-vs-wordpiece):
palavras raras são quebradas em pedaços (`"unfamiliarword" -> "un" "##fam"
"##iliar" "##word"`). O `reference_set` — o conjunto de ~3200 palavras usadas
como "eixos" do perfil relacional — é construído a partir do vocabulário do
tokenizer, filtrado para excluir pedaços que claramente não são WordPieces
(sem `"##"`, só letras). Mas esse filtro **não é suficiente**: alguns tokens
passam (`"graf"`, `"wil"`, `"mit"`) por serem alfabéticos e não terem o
prefixo `"##"`, mas na prática **quase nunca aparecem como palavra inteira** —
são quase sempre o início de um WordPiece partido (`"graf" + "##t"` =
"graft", `"wil" + "##l"` = "will"/"wild"/etc.).

`lexical_validity(token) = (ocorrências standalone) / (ocorrências totais)`
mede exatamente essa fração. Para `"graf"`, `lexical_validity ~= 0.5-9%`; para
uma palavra real como `"cotton"`, é próxima de 100%.

Um bug concreto (corrigido no capítulo 29): `TokenTimeIndex.reference_set()`
calculava `lexical_validity` mas não a passava para o filtro — então, antes
da correção, os vizinhos mais próximos de `"graft"` incluíam `"graf"`
(cos=+0.76) e `"wil"` (cos=+0.40), que não são palavras reais e não dizem nada
interpretável sobre `"graft"`. Depois da correção
(`min_lexical_validity=0.5`), 356 dos 3216 candidatos são substituídos por
palavras de fato standalone, e os vizinhos de `"graft"` passam a ser
`"net"`/`"cotton"`/`"wit"`/`"fit"`/`"cut"` — interpretáveis. As métricas
agregadas (`Z_robusto`, correlação com gold, split-half) mudam muito pouco
(3ª casa decimal) — o ganho é puramente **qualitativo**, na legibilidade das
listas de vizinhos usadas para interpretação humana.

<a id="residuo-pre-treino"></a>
## Um tipo diferente de "ruído": resíduo do pré-treino

Tudo até aqui mede **ruído de amostragem** — o quanto `D_obs(w)` varia se eu
reamostrar/reembaralhar os dados. Existe outro tipo de limitação, discutida
no [capítulo 28](../28-residuo_de_pre_treino_e_limitacao_de_corpus.md), que
nenhuma das ferramentas acima detecta: o **resíduo do pré-treino original do
encoder**.

Em D0, "plane" é quase só o sentido geométrico (~94% das ocorrências), mas seu
vizinho relacional mais próximo em D0 inclui "flight" — associação típica do
sentido moderno "avião". Comparando os embeddings estáticos de entrada em dois
checkpoints:

| | `init` (bert-tiny original) | `period1_epoch2` (após treino contínuo) |
|---|---:|---:|
| cos(plane, flight) | +0.5873 | +0.5725 |
| cos(plane, angle) | +0.4184 | +0.4573 |

A proximidade `plane`-`flight` já existe **antes** de qualquer treino
contínuo (herdada do pré-treino do bert-tiny em inglês moderno, onde "plane"
≈ "avião") e quase não muda depois. O treino contínuo move o embedding na
direção certa (cos(plane, angle) sobe), mas não o suficiente para superar um
prior tão mais "pesado" — com apenas 301 ocorrências de "plane" em D0 contra
os bilhões de tokens do pré-treino original.

A diferença prática entre os dois tipos de "ruído":

```text
ruído de amostragem  -> "será que D_obs(w) seria parecido com outra
                          amostra dos mesmos dados?" (nulo B, split-half)
resíduo de pré-treino -> "será que a posição absoluta de w num período
                          já carrega informação de fora desse período?"
                          (não tem teste estatístico simples; é uma
                          limitação de corpus x capacidade do encoder)
```

`Delta(plane, D0, D1)` continua sendo a maior `Z_robusto` das 37 palavras —
a *mudança* é real e grande. O que o resíduo de pré-treino afeta é a leitura
da posição *absoluta* de `plane` num único período (seus vizinhos em D0 já
"anteveem" parte de D1).

<a id="checklist"></a>
## Checklist: como ler um resultado de `token@time` hoje

Diante de um `D_obs(w)` e seus vizinhos, em ordem:

1. **`Z_robusto`/`p` (nulo B)**: `D_obs(w)` é maior do que uma reorganização
   aleatória dos mesmos documentos produziria? Se não, pare aqui — não há
   sinal a interpretar.
2. **split-half**: o resultado se sustenta em duas metades independentes do
   corpus, ou depende de poucos documentos?
3. **vizinhos do `reference_set`** (com `lexical_validity` aplicada): os
   vizinhos listados são palavras reais e interpretáveis?
4. **resíduo de pré-treino**: para leituras de posição *absoluta* num único
   período (não de `Delta` entre períodos), os vizinhos podem refletir, em
   parte, associações herdadas do pré-treino do encoder, não só do corpus
   daquele período.

O capítulo [29](../29-confiabilidade_do_token_time.md) percorre essas quatro
etapas para os quatro personagens do projeto.
