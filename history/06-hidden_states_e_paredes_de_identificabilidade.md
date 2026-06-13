# Capítulo 6 — Hidden states e as paredes de identificabilidade

> Fontes originais: `tmp/20-claude_hidden_relational_wall_review.md`,
> `tmp/21-claude_semantic_neighborhood_plan_review.md`,
> `tmp/22-claude_contextual_usage_wall_review.md`,
> `tmp/23-claude_scalable_temporal_semantic_architecture_review.md`.

O capítulo 05 terminou com duas correções concretas pendentes — o bug de
fronteiras e o mascaramento central determinístico — e a pergunta em
aberto: corrigir essas duas coisas seria suficiente?

## As correções acontecem — e o cloze ainda falha

Este capítulo começa com as duas correções já feitas:

1. **Fronteiras de documento** corrigidas (capítulo 05).
2. **MLM dinâmico**: em vez de mascarar sempre a posição central, agora
   15% dos tokens são selecionados a cada época, com a política
   80%/10%/10% padrão do BERT (substituir por `[MASK]`, por token
   aleatório, ou manter). As máscaras mudam por época, mas são
   reproduzíveis pela seed.
3. **Modelo maior**: `d_model=128`, 3 camadas, 4 cabeças, FFN=384 — quase
   o dobro do modelo anterior — treinado por 12 épocas em D0 e 8 em D1
   (40.188 passos de gradiente, ~34,7 milhões de alvos mascarados, ~2h em
   GPU).

O efeito sobre `graft_nn` em D0 foi dramático:

```text
antes: theta_0@D0, rank ≈ 2930, p ≈ 5,41e-5
agora: theta_0@D0, rank = 288,  p ≈ 5,89e-4
```

e os top tokens previstos passaram a incluir `river, water, tree_nn,
ground, bank, sea, wood, land_nn, hill` — claramente do campo botânico/
geográfico. **As correções funcionaram**: "poucas máscaras, baixa
capacidade e subtreino grosseiro deixaram de ser explicações
satisfatórias".

Mas em D1, o quadro continuou ruim:

```text
theta_1@D1: rank de graft_nn = 3316
top: and, the, be, it, of, that, have, time, in, to
```

E a avaliação global continuou em nível de acaso:

| Score | Spearman graded | ROC-AUC |
|---|---:|---:|
| PMI cosseno | -0,070 | 0,509 |
| PPMI-JSD | 0,042 | 0,518 |

A conclusão provisória desta etapa cristaliza algo que já se suspeitava
desde o capítulo 04: **o cloze responde "qual token completa esta posição
sintaticamente?", não "quais palavras são semanticamente próximas de
w?"**. Corrigir a frequência (via PMI) não transforma substituibilidade
posicional em proximidade semântica ampla. Esse é o primeiro "muro":
**o muro do cloze/PMI**.

## Tentativa 2: abandonar o MLM head, usar os estados ocultos

A resposta a esse muro foi mudar de representação por completo: em vez de
olhar a *saída* do MLM head (uma distribuição sobre tokens), olhar os
**estados ocultos** do Transformer — sem mascarar a palavra-alvo. Para
cada ocorrência de uma palavra `w`, extrai-se `h_t(w, ocorrência)` e
compara-se com os centroides de um vocabulário de referência (3.216
tokens não-alvo, frequência mínima 100 em cada período):

```text
r_t(w, ocorrência)[v] = cos(h_t(w, ocorrência), centroide_t(v))
```

A métrica usada foi **APD relacional** ([Average Pairwise Distance —
conceitos/04](conceitos/04-perfis_relacionais_e_apd.md#apd)): a distância
média entre as ocorrências de `w` em D0 e em D1, medida nesse espaço
relacional.

| Método | Camada | Spearman | ROC-AUC |
|---|---:|---:|---:|
| APD relacional | 2 | **0,210** | 0,542 |
| APD relacional | média últimas 2 | 0,199 | 0,542 |
| APD relacional | 1 | 0,185 | 0,565 |
| APD relacional centrado | 3 | 0,133 | 0,595 |
| centroide relacional centrado | 3 | 0,125 | 0,551 |
| perfil dos embeddings de entrada | embedding | 0,101 | 0,598 |
| energy distance relacional | 2 | 0,021 | 0,473 |

Nenhum desses valores é estatisticamente significativo com `n=37`
([conceitos/05](conceitos/05-estatistica_experimental.md)), mas o `0,210`
do APD na camada 2 foi o melhor sinal visto até agora — e não dependia de
um limiar de frequência específico para o vocabulário de referência
(testado com mínimos de 50, 100 e 200 ocorrências, todos perto de 0,20-0,23).

Para `graft_nn`, na camada 2, os vizinhos do centroide pós-centralização
mostraram exatamente a reorganização esperada:

```text
theta_0@D0: soil, stock, vine, road, plate, boundary
theta_1@D1: cell, compound, machinery, tool, commodity, exposure
```

botânico → técnico/médico — qualitativamente plausível.

## O segundo muro: o sinal de APD é, na verdade, frequência

Mas esse `0,210` veio com um problema sério: **correlação entre APD e a
frequência total da palavra-alvo = -0,436**, enquanto a correlação entre
o gold do SemEval e a frequência é só -0,113. Ou seja, **o APD está em
boa parte respondendo "essa palavra é rara?"**, não "essa palavra mudou
de sentido?".

O exemplo mais direto:

| Palavra | APD (camada 2) | Rank no APD | D0 | D1 | Gold |
|---|---:|---:|---:|---:|---|
| `chairman_nn` | 0,247 | **1** (maior mudança) | 147 | 683 | **estável (binary=0)** |
| `graft_nn` | 0,194 | 9 | 119 | 109 | mudou (graded=0,554) |
| `tree_nn` | 0,115 | 37 (menor mudança) | 2322 | 1596 | estável (graded baixo) |

`chairman_nn` — o personagem que existe exatamente para detectar esse
tipo de falso positivo (capítulo 00) — aparece como **a palavra de maior
mudança segundo o APD**, embora o gold diga que ele é estável. A
explicação: `chairman_nn` é relativamente rara em D0 (147 ocorrências) e
muito mais comum em D1 (683), e palavras raras tendem a ter APD mais alto
simplesmente porque suas poucas ocorrências são mais dispersas.

A tentativa de corrigir isso — normalizar o APD pela dispersão intrínseca
de cada palavra, usando **energy distance** com estimador intra-período —
**eliminou o sinal inteiro**: Spearman cai de 0,210 para 0,021. Ou seja:

> o APD positivo não demonstra que as distribuições temporais se separaram
> além de sua variabilidade interna comum.

Esse é o **segundo muro: o sinal de "mudança" não supera a variabilidade
de fundo de cada palavra** — quando se controla por essa variabilidade, o
sinal desaparece junto com o ruído.

## O experimento que mostra que o modelo "sabe", mas não diz sozinho

Antes de declarar o caminho sem saída, foi feito um teste mais direto:
fornecer ao modelo, manualmente, seis "campos semânticos" — geometria,
transporte, liderança, botânica, medicina, corrupção — representados
localmente em cada checkpoint (para não comparar coordenadas rotacionadas
entre `theta_0` e `theta_1`). Com esses eixos *dados* (não descobertos),
o resultado foi nítido:

```text
plane_nn D0:  geometria 0,718 | transporte 0,081
plane_nn D1:  geometria 0,053 | transporte 0,934

chairman_nn D0:  liderança 0,923
chairman_nn D1:  liderança 0,999
```

Com esses campos manuais, `plane_nn` vai para o rank 1 (maior mudança) e
`graft_nn` para o rank 2 — exatamente como o gold prevê — enquanto
`chairman_nn` (rank 32) e `tree_nn` (rank 26) ficam baixo, como controles
estáveis. Spearman global = 0,184.

**Conclusão central deste experimento**: o modelo *contém* informação
suficiente para separar `plane_nn` de `chairman_nn` — quando alguém diz a
ele quais são as dimensões relevantes. O problema não é (só) falta de
sinal no modelo. É **não saber, sem supervisão, quais eixos usar**.

## O terceiro muro: comunidades não supervisionadas misturam campos

A tentativa óbvia seguinte foi descobrir esses "campos" automaticamente,
via comunidades de tokens em grafos k-NN (3.216 tokens compartilhados,
frequência mínima 100). Mesmo depois de ajustar `k` e resolução para
evitar singletons (`k=40`, 11 comunidades, AMI médio=0,917, melhor
Spearman ≈ 0,168-0,200), o resultado qualitativo foi:

```text
plane_nn:    rank ~15-23
chairman_nn: rank ~6-10
graft_nn:    alto
tree_nn:     baixo
```

Uma comunidade ampla de 535 tokens misturava geometria, transporte,
botânica e outros campos físicos inteiros — comunidades globais de tokens
não respeitam o fato de que **uma palavra pertence a vários campos
hierárquicos e sobrepostos** (polissemia).

## O quarto muro: separar D0/D1 não é o mesmo que separar sentidos

A última tentativa desta fase foi a mais sofisticada: em vez de comunidades
de *tokens*, comunidades de **ocorrências contextualizadas**
`h_t(w, ocorrência)`. Para cada palavra, juntam-se ocorrências de D0 e D1
(rótulo temporal escondido do algoritmo de clustering), agrupam-se em `K`
clusters, e mede-se:

```text
Delta_usage(w) = JSD(P(cluster | w, D0), P(cluster | w, D1))
```

calculado separadamente sob `theta_0` congelado e `theta_1` congelado
(para não comparar coordenadas entre checkpoints), com `K ∈ {2,...,6}` e
5 seeds, **escolhidos antes de olhar o gold** — uma disciplina herdada do
capítulo 03.

Resultado, com vetores ocultos crus (camada 2):

| Alvo | JSD | `theta0` congelado | `theta1` congelado | Rank |
|---|---:|---:|---:|---:|
| `graft_nn` | 0,243 | 0,242 | 0,244 | **1** |
| `chairman_nn` | 0,168 | 0,170 | 0,165 | **2** |
| `plane_nn` | 0,024 | 0,030 | 0,018 | 25 |
| `tree_nn` | 0,004 | 0,006 | 0,003 | 36 |

Spearman global = **-0,155**. `graft_nn` no topo (correto), mas
`chairman_nn` também no topo (errado) e `plane_nn` quase no fundo
(errado, dado que seu gold é o segundo maior do dataset).

Para `chairman_nn`, os vizinhos em ambos os períodos continuam claramente
no campo de liderança:

```text
D0: secretary, editor, commander, director, president, committee, jury
D1: secretary, director, commander, president, commissioner, governor, publisher
```

— ou seja, o *sentido* de `chairman_nn` não mudou. Mas os **clusters de
ocorrências separam fortemente D0 de D1** sob ambos os checkpoints. A
hipótese mais provável é que esses clusters estejam capturando:

- tipo de instituição (parlamento → empresa/comitê);
- registro e gênero textual;
- nomes e cargos associados;
- construção sintática;
- concentração colocacional;
- composição amostral do corpus.

Tudo isso é **variação temporal contextual real** — mas não é mudança do
significado lexical de `chairman_nn`. É exatamente o tipo de falso
positivo que o personagem `chairman_nn` foi escolhido para expor
(capítulo 00).

## A formulação do problema de identificabilidade

A síntese desta fase nomeia o problema de forma precisa. A formulação
conceitual

```text
Delta_sem(w) = D(P_0(z|w), P_1(z|w))
```

continua parecendo correta — `z` é "o sentido ou tipo semântico de uso".
O problema é que, na prática,

```text
P_t(cluster | w) != P_t(sense | w)
```

Clusters não supervisionados de ocorrências encontram **variação
temporal real**, mas não distinguem mudança de sentido de mudança de
tópico, gênero/registro, sintaxe, entidades/colocações ou composição
amostral. E há um risco mais profundo, nomeado explicitamente:

> Usando apenas corpus, período e representações de um MLM, talvez não
> exista critério não supervisionado capaz de dizer se uma separação
> temporal é semântica ou apenas contextual. **O período é precisamente o
> fator cujo efeito queremos medir, mas ele também carrega todos os
> nuisances históricos.**

Essas são as "paredes A-D" mencionadas no capítulo 00: (A) cloze/PMI mede
substituibilidade, não proximidade semântica; (B) sinais de geometria
oculta (APD) são confundidos por frequência; (C) controlar por
variabilidade interna elimina o sinal junto com o ruído; (D) clustering
não supervisionado separa por *qualquer* fator que varie com o tempo, não
especificamente por sentido — e não há, a priori, garantia de que esse
fator seja identificável a partir só de corpus + período.

## O que vale levar deste capítulo

- As correções do capítulo 05 (fronteiras, MLM dinâmico, modelo maior)
  **funcionaram** — `graft_nn` em D0 passou de rank 2930 para 288. Mas
  corrigir os problemas de engenharia não resolveu o problema científico:
  ele só ficou mais visível.
- O experimento dos "campos manuais" é a peça-chave: ele mostra que **a
  informação existe** no modelo (`plane_nn` e `graft_nn` no topo,
  `chairman_nn` e `tree_nn` no fundo, quando os eixos certos são dados) —
  mas **descobrir esses eixos sem supervisão é o problema real**, não a
  capacidade do modelo.
- `chairman_nn` continua sendo o detector de falso positivo mais
  confiável do projeto: ele aparece no topo do ranking em pelo menos três
  métodos diferentes (APD bruto, clustering de ocorrências cru e
  relacional), sempre pela mesma razão — variação contextual real sem
  mudança de sentido.
- A pergunta que fecha este capítulo — "existe um critério não
  supervisionado capaz de separar mudança de sentido de mudança de
  contexto, usando só corpus e período?" — é a pergunta que os capítulos
  07 e 08 tentam responder por caminhos diferentes (realinhamento como
  instrumento de consulta, e depois desambiguação de sentido externa no
  capítulo 10).

## Conceitos novos usados neste capítulo

- [Mudança de contexto, representação e sentido lexical](conceitos/07-o_que_esta_sendo_medido.md)
- [Grade checkpoint x corpus e encoder fixo](conceitos/08-desenhos_temporais_e_reguas.md)
- [APD (Average Pairwise Distance)](conceitos/04-perfis_relacionais_e_apd.md#apd)
- [Energy distance e dispersão intra-período](conceitos/04-perfis_relacionais_e_apd.md#energy-distance)
- [Clustering não supervisionado e NMI/AMI](conceitos/04-perfis_relacionais_e_apd.md#nmi)
- [Identificabilidade e variáveis de confusão (confounders)](conceitos/05-estatistica_experimental.md#identificabilidade)
