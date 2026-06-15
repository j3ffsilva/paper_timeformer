# 29. Quanto confiar em um `D_obs`?

## A pergunta que faltava

O capítulo 27 definiu o framework de seis consultas do `token@time`. O
capítulo 28 mostrou que, mesmo com o método funcionando corretamente, a
posição absoluta de uma palavra num período pode carregar resíduo do
pré-treino do encoder. Faltava uma pergunta mais básica, sobre a peça central
de todas as consultas: o deslocamento `D_obs(w) = 1 - cos(R_0(w), R_1(w))`.
`D_obs(plane) = 0.0655`. É grande? Pequeno? Comparado com o quê?

Sem uma resposta, qualquer ranking ("quem mudou mais") ou afirmação de
mudança ("`graft` mudou de sentido") fica sem chão. Esta sessão de trabalho
(registrada em detalhe em `tmp/36`) percorreu uma ordem de oito passos para
construir essa resposta. O capítulo
[conceitos/10](conceitos/10-confiabilidade_e_significancia.md) explica cada
ferramenta isoladamente; este capítulo narra a sequência e o que ela mudou na
leitura dos quatro personagens do projeto.

## Passo 4: o nulo B

A primeira ferramenta construída foi o **nulo B**
(`document_permutation_null`): embaralhar os documentos de d0+d1 de `w`,
redividir em dois grupos do tamanho original, e recalcular `D` — repetindo
200 vezes para formar uma distribuição `D_null`. Isso responde "o que eu
mediria se a divisão d0/d1 fosse arbitrária?".

Teste em três das quatro palavras-chave (seed1000, `layer_2`, B=200):

| palavra | `D_obs` | mediana(`D_null`) | `Z_robusto` | `p` |
|---|---:|---:|---:|---:|
| `plane_nn` | 0.0655 | 0.0217 | 18.10 | 0.005 |
| `chairman_nn` | 0.0554 | 0.0261 | 16.54 | 0.005 |
| `graft_nn` | 0.0344 | 0.0227 | 9.04 | 0.005 |

`p=0.005` é o piso de `B=200` (`1/201`) — as três ficam totalmente fora da
distribuição nula. Mas aqui aparece o primeiro alerta do projeto: `Z` alto
para `chairman_nn` é **esperado** segundo o capítulo 00 (linha da tabela dos
quatro personagens: "métodos baseados em vizinhos/clusters acusam mudança
porque o *contexto* varia, mesmo que o *sentido* não mude — falso positivo
clássico"). Um nulo que apenas confirma "há uma diferença entre d0 e d1" não
basta para separar mudança de sentido de variação de contexto. Faltava saber
se esse nulo está calibrado.

## Passo 5: pseudo-períodos, o achado inesperado

Para calibrar sem depender do gold (que só existe para as 37 palavras do
SemEval), o passo 5 dividiu os 253644 documentos de `1810-1860.txt`
aleatoriamente em duas metades (`pseudo_a`/`pseudo_b`) — **sem diferença
temporal real** — e repetiu todo o protocolo sobre esse par.

Resultado: zero falsos positivos em 37 palavras, em ambos os seeds,
em `alpha=0.05` e `alpha=0.10`. O nulo não está "estreito demais" — se algo,
é levemente conservador.

O achado inesperado veio de comparar a *largura* do nulo:
`MAD(D_null)` no par pseudo é só ~0.58x o `MAD(D_null)` no par real
(d0 vs d1) — mesmo que cada metade pseudo tenha *menos* documentos que d0
inteiro, o que pela intuição "menos dados -> nulo mais largo" deveria ir na
direção contrária. A explicação (detalhada em
[conceitos/10](conceitos/10-confiabilidade_e_significancia.md#pseudo-periodos)):
o nulo B mantém `mu_t` e as referências fixos nos valores observados de cada
período; em `pseudo_a`/`pseudo_b` esses sistemas de referência são quase
idênticos (mesmo século), enquanto em d0 vs d1 o vocabulário inteiro do
corpus muda. O nulo B não mede só "ruído amostral de `w`" — herda também o
quanto os dois sistemas de referência diferem entre os períodos comparados.

## Passo 7: as 37 palavras reais, e o achado principal

Com o protocolo congelado (passo 6: `layer_2`, `V_active` com
`n_min_active=10`, `reference_set` com `max_references=3216` e
`min_lexical_validity=0.5`, encoder `period1_epoch2`, nulo B com B=200),
o passo 7 rodou as 37 palavras-alvo do SemEval contra d0 vs d1 de verdade.

Primeiro, o contraste com o passo 5 confirma que o nulo distingue os dois
cenários:

| | d0 vs d1 (real) | pseudo vs pseudo |
|---|---:|---:|
| significativos @ alpha=0.05 | 20/37 e 18/37 (dois seeds) | 0/37 |
| significativos @ alpha=0.10 | 23/37 (ambos os seeds) | 0/37 |

Mas o achado principal foi outro: comparando `D_obs` e `Z_robusto` com o gold
do SemEval2020 (`graded`/`binary` de `truth.tsv`, n=37, valores após a
correção do passo 9):

| | seed1000 | seed1001 |
|---|---:|---:|
| Spearman(`D_obs`, graded) | 0.061 | 0.068 |
| Spearman(`Z_robusto`, graded) | **0.269** | **0.295** |
| point-biserial(`binary`, `D_obs`) | 0.201 | 0.209 |
| point-biserial(`binary`, `Z_robusto`) | **0.334** | **0.342** |

`D_obs` cru é quase descorrelacionado do julgamento humano de mudança
(`rho~=0.06`). `Z_robusto` — `D_obs` dividido pelo `MAD(D_null(w))`, a régua
de ruído específica de cada palavra — chega a `rho~=0.27-0.30`, e a
correlação com o rótulo binário passa a ser significativa a 5%
(`r_pb~=0.33-0.34`).

A leitura para os quatro personagens: `plane_nn` e `graft_nn` (gold alto)
têm `Z_robusto` entre os maiores das 37 palavras — consistente. `chairman_nn`
(gold baixo) também tem `Z` alto (16.54) — o falso positivo previsto pelo
capítulo 00 continua presente mesmo em `Z_robusto`; normalizar pelo nulo B
reduz o problema agregado (a correlação geral sobe), mas não elimina casos
individuais como `chairman_nn`. Isso é esperado: o nulo B testa "há diferença
entre d0 e d1 maior que o acaso", não "essa diferença é uma mudança de
*sentido* e não de *contexto de uso*" — essa segunda pergunta é o que os
capítulos do ConSeC (29-32 da numeração de `docs/`) tentam responder por
outra via.

## Passo 8: split-half confirma que não é ruído de poucos documentos

Por fim, o passo 8 perguntou algo diferente: será que `D_obs(w)` depende de
um punhado de documentos de sorte? Dividindo as ocorrências de cada palavra
em d1 em duas metades aleatórias por documento e recalculando `D_half1`,
`D_half2`:

| | seed1000 | seed1001 |
|---|---:|---:|
| Spearman(`D_half1`, `D_half2`) | 0.958 | 0.962 |
| Spearman(`D_obs`, mean(halves)) | 0.999 | 0.996 |

Mesmo `graft_nn`, com apenas ~64 ocorrências por metade em d1, preserva a sua
posição relativa entre as 37 palavras. `D_obs(w)` é uma propriedade estável
do corpus.

## Onde isso deixa o capítulo 28

O capítulo 28 mostrou que a posição *absoluta* de `plane` em d0 carrega
resíduo do pré-treino (cos(plane,flight)=0.59 já no `init`). Este capítulo
mostra que, apesar disso, o *deslocamento* `D_obs(plane, d0, d1)` continua
sendo o maior `Z_robusto` das 37 palavras (18.10), robusto a permutação
(nulo B), a recombinação amostral (split-half) e consistente com o gold
(`Z_robusto` correlaciona com `graded`/`binary`). Os dois capítulos descrevem
limitações de natureza diferente — uma de viés sistemático do encoder na
leitura de posição absoluta, outra de ruído amostral na leitura de
deslocamento — e ambas apontam para a mesma conclusão prática: reportar
`Z_robusto` (não `D_obs` cru) como métrica primária de "sinal" do
`token@time`, com a checklist de quatro itens de
[conceitos/10](conceitos/10-confiabilidade_e_significancia.md#checklist)
como roteiro de leitura.

## Próximo passo

Construir uma tabela consolidada de evidência — `D_obs`, `Z_robusto`, `p`,
split-half e correlação com o gold para as 37 palavras × 2 seeds — como
peça central da seção de resultados do paper.
