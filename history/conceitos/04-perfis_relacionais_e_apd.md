# Conceitos 4 — Perfis relacionais, APD e ferramentas de comparação geométrica

Este arquivo cobre as ferramentas usadas para responder à pergunta central
do projeto a partir de representações vetoriais: **dadas duas nuvens de
pontos (vetores) representando ocorrências de uma palavra em dois
períodos, o quanto elas "se afastaram"?**

## Perfil relacional (recapitulação)

Um perfil relacional `R_t(w)` é um vetor que descreve `w` **em relação a
um conjunto fixo de outras palavras `V`** (o "vocabulário de referência"),
em vez de descrever `w` em coordenadas internas do modelo (que mudam de
checkpoint para checkpoint). A versão canônica (capítulo 08, `docs/12`)
usa:

```text
ê_t(x) = (e_t(x) - mu_t) / ||e_t(x) - mu_t||      (centralização)
P_t(w)[v] = cos(ê_t(w), ê_t(v))                    (perfil relacional)
```

`mu_t` é um vetor "médio" do espaço no período `t` — subtraí-lo remove
translações sistemáticas entre `theta_0` e `theta_1`. O cosseno depois da
centralização é invariante a rotação, reflexão e reescalonamento
isotrópico — mas **não** a deformações anisotrópicas/não-lineares, que
precisam ser medidas empiricamente (foi isso que o "eixo de época",
[conceitos/03](03-deriva_e_esquecimento.md), revelou).

<a id="apd"></a>
## APD (Average Pairwise Distance)

APD mede **o quanto duas nuvens de pontos estão, em média, distantes uma
da outra** — sem assumir que cada nuvem tem um único "centro" representativo.

```text
APD(w) = média sobre (i em D0, j em D1) de  distancia(x_i, x_j)
```

onde `x_i` são as representações das ocorrências de `w` em D0 e `x_j` as
de D1, e `distancia` é tipicamente `1 - cos(x_i, x_j)`.

Por que não simplesmente comparar os **centróides** (a média de cada
nuvem)? Porque o centróide colapsa toda a variação interna de uma nuvem
num único ponto — duas nuvens podem ter centróides parecidos mas formas/
dispersões muito diferentes (ou vice-versa). APD captura a relação entre
as nuvens *inteiras*.

No capítulo 06, APD sobre hidden states relacionais (camada 2) deu o
melhor resultado da fase (Spearman=0,210) — mas com um problema sério:
correlação de `-0,436` com a frequência total da palavra. `chairman_nn`
(147 ocorrências em D0, 683 em D1 — bastante desbalanceado) apareceu como
**rank 1** no APD, embora seu gold seja "estável". Palavras raras tendem a
ter nuvens mais dispersas, e nuvens mais dispersas tendem a ter APD mais
alto — *independentemente* de mudança de sentido.

<a id="energy-distance"></a>
## Energy distance

Energy distance é uma forma de **normalizar o APD pela dispersão interna
de cada nuvem** — respondendo "a distância entre D0 e D1 é maior do que a
variabilidade que já existe *dentro* de D0 (ou de D1) sozinho?".

```text
energy(D0, D1) = 2 * APD(D0,D1) - APD(D0,D0) - APD(D1,D1)
```

`APD(D0,D0)` é a distância média entre pares de pontos *dentro* de D0 (a
"dispersão interna" de D0); analogamente para `APD(D1,D1)`. Se
`APD(D0,D1)` não é maior do que a dispersão interna de cada nuvem,
`energy ~ 0` — as duas nuvens são indistinguíveis de duas amostras da
*mesma* distribuição.

No capítulo 06, essa normalização **eliminou inteiramente** o sinal do APD
(Spearman cai de 0,210 para 0,021). A leitura foi que o `0,210` original
não representava "as nuvens de D0 e D1 se separaram além de sua
variabilidade comum" — representava, em boa parte, a mesma confusão de
frequência descrita acima. É um exemplo de "controlar por uma variável de
confusão pode eliminar o sinal junto com o ruído" — ver
[identificabilidade](05-estatistica_experimental.md#identificabilidade).

<a id="nmi"></a>
## NMI / AMI

NMI (Normalized Mutual Information) e AMI (Adjusted Mutual Information)
medem **o quanto duas formas de agrupar os mesmos itens concordam entre
si** — por exemplo, "os clusters encontrados por k-means" vs. "os rótulos
de período (D0/D1)".

- `NMI = 0`: os clusters não têm relação nenhuma com os rótulos
  (independência estatística).
- `NMI = 1`: os clusters reproduzem os rótulos perfeitamente (cada cluster
  corresponde exatamente a um rótulo, possivelmente com nomes
  diferentes).
- AMI é uma versão de NMI corrigida para o "acaso" — duas partições
  aleatórias com muitos clusters pequenos podem ter NMI>0 só por azar;
  AMI corrige isso.

O uso mais marcante de NMI no projeto foi no capítulo 08: particionar a
nuvem combinada (D0+D1) de **qualquer** palavra em 2 clusters via k-means e
medir `NMI(cluster, período)`. Resultados como `tree_nn`: NMI=1,000 em
`layer_2` (a maior associação cluster×período de toda a amostra, para um
controle *estável*!) revelaram o "eixo de época" — um corte simples
recupera o checkpoint de origem quase perfeitamente, para quase qualquer
palavra, mascarando completamente qualquer estrutura de *sentido*.

Esse mesmo NMI, quando calculado sob o **oráculo BERT-base** (capítulo 08),
teve o comportamento oposto e esperado: `plane_nn` (mudou de sentido)
NMI=0,487 (alto — os clusters correspondem a "modo geométrico" vs. "modo
aviação", que coincidem com período por acaso histórico, não por drift de
checkpoint); `tree_nn` (estável) NMI≈0,002 (quase zero). A diferença entre
"NMI alto porque é drift de checkpoint" e "NMI alto porque há de fato dois
sentidos correspondendo aos dois períodos" só pôde ser resolvida com a
[grade 2x2 de encoder fixo](02-encoders_e_camadas.md#encoder-fixo).

<a id="rbo"></a>
## RBO (Rank-Biased Overlap)

RBO mede **a sobreposição entre dois rankings (listas ordenadas)**, dando
mais peso ao topo das listas. Se as duas listas têm os mesmos itens nas
primeiras posições (mesmo que em ordem ligeiramente diferente), RBO é
alto; se concordam só nas posições finais (menos importantes), RBO é
baixo.

No capítulo 04, RBO foi uma das três operacionalizações discutidas para
"comparar os vizinhos de `w` em D0 com os vizinhos de `w` em D1, sem usar
uma lista de âncoras pré-definida" — comparando o **top-k de vizinhos**
via RBO em vez de comparar o perfil inteiro. Foi descartada nessa fase
porque introduz um parâmetro `k` arbitrário, mas a ideia — "o que importa é
o topo da lista de vizinhos, e quanto eles mudaram" — reaparece de forma
mais madura nas comparações de vizinhança qualitativa do capítulo 07.

<a id="svd"></a>
## SVD (Singular Value Decomposition)

SVD decompõe qualquer matriz em três partes:

```text
M = U . Sigma . V^T
```

Para uma matriz **simétrica e positiva semi-definida** (PSD) — como a
matriz de coesão `M_t(w)` do capítulo 08 — a SVD coincide com a
decomposição em **autovalores e autovetores**:

```text
M = sum_i  lambda_i * u_i * u_i^T
```

- `u_i` (autovetores): direções/"modos" — cada `u_i` é um vetor de cargas
  sobre o vocabulário de referência `V`, candidato a representar um
  "campo semântico" ou "sentido" de `w`.
- `lambda_i` (autovalores): "quanto peso"/importância cada modo tem —
  `lambda_i = sigma_i^2`, onde `sigma_i` são os valores singulares.

A premissa do capítulo 08 (Fase 1.5) era: se `w` tem múltiplos sentidos,
`M_t(w)` deveria ter **vários** `lambda_i` grandes (vários modos
relevantes). O que se observou foi `lambda_1` dominando `lambda_2` por
**10-30x** em todos os 48 casos testados — ou seja, `k=1` (um único modo)
sempre, mesmo para palavras com mudança de sentido conhecida. Daí o
critério de [gap](#gap) entrar em jogo: ele tenta detectar automaticamente
*quantos* `lambda_i` são "grandes o suficiente" para contar como modos
separados.

<a id="gap"></a>
## Gap (critério de gap relativo)

O "critério de gap" (capítulo 08, `docs/12` §8) é uma forma de decidir
automaticamente **onde cortar uma lista ordenada de valores** (autovalores,
ou os próprios valores de `P_t(w)[v]` ordenados), sem fixar um limiar
arbitrário de antemão.

```text
h_i = (X_i - X_{i+1}) / X_i      (gap relativo entre o i-ésimo e o (i+1)-ésimo valor)
```

Se `h_i > gamma` (um limiar pequeno, sugerido ~0,3 = 30%), há uma "queda
relativa grande" entre `X_i` e `X_{i+1}` — interpretada como uma fronteira
natural: tudo até `i` fica de um lado, tudo depois fica do outro. Se
**nenhum** `h_i` excede `gamma`, o critério declara explicitamente "sem
estrutura clara" — em vez de forçar uma fronteira onde não há nenhuma.

A propriedade importante é que `h_i` é **invariante a reescalonamento
positivo**: multiplicar toda a lista por uma constante não muda nenhum
`h_i`. Isso o torna adequado tanto para escolher `tau` (limiar de
similaridade que define o vocabulário relevante `V_w`) quanto `k` (número
de modos, via gap nos autovalores) quanto `theta` (limiar de matching entre
modos de períodos diferentes) — três decisões diferentes, um único
princípio.

A causa do NO-GO do capítulo 08 foi descoberta inspecionando diretamente
os valores aos quais esse critério era aplicado: `P_t(plane\_nn)[v]`
ordenado decai **suavemente** de ~0,95 a ~-0,76 ao longo de milhares de
tokens, com `h_i` da ordem de `0,003-0,05` — nenhum gap próximo de `0,3`
(exceto um artefato perto do cruzamento de zero, sem significado
semântico). O critério de gap em si estava correto; é a *entrada* (uma
"rampa" suave, sem degraus) que não tinha a estrutura que o critério
precisa para funcionar.

<a id="ot"></a>
## OT (Optimal Transport)

Optimal Transport (Transporte Ótimo) responde à pergunta: **dado que eu
preciso "mover" toda a massa de probabilidade de uma distribuição `A` para
coincidir com uma distribuição `B`, qual é o "plano de movimento" mais
barato, e quanto custa?**

No contexto do projeto (capítulo 07), `A` e `B` são duas distribuições
sobre o mesmo conjunto de "pontos de referência" (vizinhos no vocabulário
`V`) — por exemplo, o perfil de vizinhos de `plane_nn` em D0 e em D1. O
"custo de transporte" `c(v, v')` entre dois pontos de referência é
tipicamente `1 - cos(ê(v), ê(v'))` — quão "longe" semanticamente `v` está
de `v'`. O custo total ótimo, `Delta_OT(w)`, mede quanto "trabalho
semântico" é necessário para transformar o perfil de D0 no perfil de D1.

A vantagem do OT sobre uma simples distância de cosseno entre os perfis
agregados é que ele é sensível a **como** a massa se moveu — por exemplo,
distinguir "todo o peso se moveu de um campo para outro, de forma
concentrada" (substituição de sentido, como `plane_nn`) de "o peso se
espalhou para um campo adicional, mantendo o original" (diversificação,
como `graft_nn`). No capítulo 07, a tabela manual deu `Delta_OT`:
`chairman_nn=0,034` (baixo, estável), `tree_nn=0,143`, `plane_nn=0,400`,
`graft_nn=0,494` — ordenação qualitativamente coerente com o gold, embora
a versão automatizada não tenha superado o APD em Spearman (0,196 vs
0,210).

Um ponto de cuidado discutido no capítulo 07 (`tmp/26`): o custo `c(v,v')`
pode ser calculado usando o cosseno em D0 (`c_0`) ou a média entre D0 e D1
(`c_mean`); `c_0` foi considerada mais defensável, porque usa apenas
informação do período de referência, sem "espiar" o período de destino ao
construir a métrica de custo.

<a id="procrustes"></a>
## Procrustes

A análise de Procrustes encontra a **melhor transformação rígida**
(rotação + reflexão + reescalonamento, sem distorção) que alinha um
conjunto de pontos a outro, minimizando a distância total depois do
alinhamento.

No contexto deste projeto, Procrustes seria a ferramenta "clássica" para
tentar comparar coordenadas internas de `theta_0` e `theta_1` diretamente
— "encontre a rotação que melhor alinha o espaço de `theta_0` ao espaço de
`theta_1`, e então compare". O perfil relacional (capítulo 04) **evita**
precisar de Procrustes: ao comparar tudo via cosseno sobre um vocabulário
de referência compartilhado (coordenadas "externas", baseadas em tokens, não
nas dimensões internas do modelo), a necessidade de um alinhamento
explícito entre espaços de coordenadas desaparece — qualquer rotação
consistente de `theta_1` em relação a `theta_0` já é cancelada
automaticamente pelo cosseno (capítulo 08, prova de invariância). Isso é
parte do motivo pelo qual a persistência de modos entre períodos (capítulo
08, `docs/12` §11) pode usar **matching húngaro** sobre cargas de
vocabulário, em vez de Procrustes sobre coordenadas internas.

<a id="controles-pareados"></a>
## Controles pareados

Um "controle pareado" é um grupo de itens selecionados especificamente
para serem **comparáveis** ao grupo de interesse numa dimensão que poderia
confundir o resultado — para isolar se um efeito é específico do grupo de
interesse, ou se aparece igualmente no controle (e portanto não é
específico).

No capítulo 09, foram selecionadas 37 palavras **fora** do benchmark
SemEval, com frequências em D0 e D1 pareadas às 37 palavras-alvo. O
resultado — APD absoluto praticamente igual entre alvos e controles, em
todas as condições — foi um dos achados mais importantes do capítulo,
embora não meça diretamente "mudança semântica": ele mostra que **o sinal
do SemEval inteiro vem do ranking relativo entre as 37 palavras-alvo**,
não de uma diferença de magnitude entre "palavras que mudaram" e "palavras
quaisquer". Isso reforça, de um ângulo diferente, a lição da [energy
distance](#energy-distance): a magnitude absoluta do APD não é, por si só,
informativa sobre mudança de sentido.
