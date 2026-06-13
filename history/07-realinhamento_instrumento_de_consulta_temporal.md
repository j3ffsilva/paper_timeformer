# Capítulo 7 — Realinhamento: de detector a instrumento de consulta temporal

> Fontes originais: `tmp/24-codex_realinhamento_contribuicao.md`,
> `tmp/25-claude_relational_ot_tabletop_review.md`,
> `tmp/26-claude_ot_review_response.md`,
> `tmp/27-codex_reframing_instrumento_temporal.md`,
> `tmp/30-exemplo_campo_temporal_relacional_forte.md`.

O capítulo 06 terminou em quatro muros e numa pergunta desconfortável:
"existe um critério não supervisionado capaz de separar mudança de
sentido de mudança de contexto, usando só corpus e período?". Em vez de
tentar um quinto método para atravessar esse muro, este capítulo registra
uma virada diferente: **mudar a pergunta que o projeto está tentando
responder**.

## Primeiro passo: nomear o que já temos, sem mais rodeios

`tmp/24` abre com um balanço direto de tudo o que foi tentado desde o
capítulo 04 — cloze-PMI (capítulo 05, descartado), clustering de
ocorrências (capítulo 06, "muro de identificabilidade formal: mais
épocas ou melhores algoritmos não atravessam essa parede"), APD
relacional (capítulo 06, sinal de 0,210 mas confundido por frequência), e
um experimento novo, **atlas externo de WSD**, que também foi testado e
rejeitado:

| checkpoint | conjunto | acurácia contra glosses do WordNet |
|---|---|---:|
| `theta_0` | D0 geometria | 23,6% |
| `theta_0` | D0 ferramenta | 10,5% |
| `theta_1` | D1 avião | 63,9% |

A conclusão foi que estados ocultos de um MLM genérico não são
compatíveis com representações de glosses sem treinamento conjunto
explícito — isso exigiria um encoder externo de WSD (BEM/ConSeC), o que
"transfere a contribuição central para um componente de terceiros".
**Decisão registrada: o atlas WSD é uma direção de paper separada, não o
TimeFormer.**

## A "régua móvel": de problema a achado

Um experimento de controle — projetar ocorrências reais nos campos
manuais "geometria" e "transporte" (o mesmo experimento do capítulo 06) —
revelou algo sobre o próprio treino contínuo:

```text
plate figure represent an inclined plane

theta_0: geometria 0,841 | transporte 0,008
theta_1: geometria 0,043 | transporte 0,951
```

A **mesma frase**, processada por `theta_0` e por `theta_1`, é classificada
de formas opostas. Em 218 ocorrências de D0, o argmax passa de "geometria"
(sob `theta_0`) para "transporte" (sob `theta_1`); o inverso acontece com
contextos modernos de aviação sob `theta_0`.

`tmp/24` propõe ler isso não como um defeito a corrigir, mas como **o
próprio dado**:

> A mudança da régua IS o dado. O modelo que aprendeu D0 e depois D1
> incorpora a transição em seus pesos.

Essa frase é o gancho conceitual de todo o capítulo: se a "régua" (a
geometria interna do modelo) se move de forma sistemática entre `theta_0`
e `theta_1`, talvez o produto interessante do TimeFormer não seja "um
número que diz o quanto uma palavra mudou", mas **a própria régua em cada
momento** — isto é, a vizinhança lexical de uma palavra, vista por aquele
checkpoint específico.

## Onde o TimeFormer está, na paisagem da literatura

`tmp/24` situa o projeto em relação a dois tipos de trabalho prévio:

| propriedade | Hamilton et al. 2016 | APD+BERT (SemEval) | TimeFormer |
|---|---|---|---|
| embeddings | estáticos (1 vetor/palavra/período) | contextuais | contextuais |
| modelos | 2 independentes | 1 fixo externo | 1 contínuo |
| domínio | in-domain | out-of-domain (BERT moderno em texto de 1810) | in-domain |
| alinhamento | Procrustes ortogonal | não necessário | não necessário (perfis relacionais) |
| resolução temporal | 2 pontos | 2 pontos | N checkpoints (em princípio) |

A limitação de Hamilton 2016: embeddings estáticos borram polissemia —
"bank perto de rio" e "bank perto de dinheiro" contribuem igualmente para
um único vetor. A limitação dos sistemas BERT do SemEval (melhor
Spearman ≈ 0,42 para inglês): aplicam um modelo pré-treinado em inglês
*moderno* a texto de 1810 — saem do domínio de treino. A proposta do
TimeFormer não é "transformers em vez de word2vec" (isso já existe) — é
**um único modelo que viveu a transição cronológica**, com representações
in-domain nos dois períodos e sem necessidade de alinhamento post-hoc.

## A demonstração qualitativa: vizinhanças temporais

A "prioridade 1" executada nesta fase foi simples de enunciar: para cada
palavra-alvo, calcular o top-20 de vizinhos por `r_t(w)[v] =
cos(centroide_t(w), centroide_t(v))` em cada período (camada 2, geometria
centrada, 3.216 referências), e também `delta_z(w)[v]` — o ganho/perda de
proximidade padronizado entre períodos, restrito a `v` que aparece entre
os 50 vizinhos mais próximos em pelo menos um período.

| Personagem | D0 (top vizinhos) | D1 (top vizinhos) | Leitura |
|---|---|---|---|
| `plane_nn` | `line, angle, plate, column, canal, coast` | `boat, ship, rail, route, machine` | transição clara para **transporte amplo** — mas `aircraft, airline, pilot, airport` não dominam o top-20, então a leitura "geometria → aviação especificamente" não é totalmente sustentada |
| `chairman_nn` | `secretary, editor, commander, director, president, committee, jury` | `secretary, director, commander, president, commissioner, governor, publisher` | núcleo institucional **estável**; ganhos (`bush, clinton, board, republican, executive`) e perdas (`emperor, prophet, preacher, merchant, king`) são sobre a *realização histórica* do campo, não sobre o campo em si |
| `tree_nn` | `rock, water, leaf, valley, grass, grove, stem, bud, bark, vine, soil` | `wood, fountain, stone, sand, garden, mountain, grass, forest, valley, leaf, bird` | reorganização interna, sem transferência para campo alheio |
| `graft_nn` | `globe, bee, planet, road, chapel, horizon, platform, frontier, village, ship, town, vine` (heterogêneo) | `compound, machinery, currency, commodity, mechanic, utility, consumption, tool, facility, acid, organ, substance, cell` | forte transição relacional, mas a vizinhança D0 já era heterogênea — não é uma trajetória limpa "botânica → medicina/corrupção" |

A conclusão registrada para esta etapa é deliberadamente modesta:

> TimeFormer consegue caracterizar **algumas** transições relacionais
> temporais por vizinhanças lexicais interpretáveis e locais a cada
> checkpoint, sem alinhamento geométrico post-hoc.

E explicitamente **não**:

> TimeFormer identifica consistentemente os sentidos lexicais de
> qualquer palavra.

O critério qualitativo foi satisfeito claramente por `plane_nn` e pela
estabilidade de `chairman_nn`; parcialmente por `tree_nn`; e ficou misto
para `graft_nn` — o mesmo personagem que já era o caso difícil nos
capítulos 02 (Amnesia by Design, falha de protótipo médio) e 06
(clustering).

## O reframing: instrumento de consulta, não detector

`tmp/27` é o documento que nomeia a virada. A frase central:

> Estávamos tentando construir um *detector* de mudança semântica — um
> sistema que produz um score escalar e compete com o Spearman do
> SemEval. Isso criou uma barreira artificial e um objetivo errado. O
> TimeFormer não é um detector. É um **instrumento de consulta temporal**.

A pergunta que o instrumento responde é simplesmente:

```text
Quais eram os vizinhos lexicais de w no corpus de D0, e quais são em D1?
```

Sem pressupor nenhuma teoria de sentido. O pesquisador recebe as
vizinhanças e interpreta — pode ser mudança de sentido lexical, mudança de
registro, deriva de domínio, mudança social.

Sob esse reframing, dois resultados que pareciam "problemas" no capítulo
06 são reinterpretados:

- **`plane_nn`** não é um falso negativo do SemEval — é uma demonstração
  bem-sucedida do instrumento: o pesquisador vê a transição
  geométrico/material → transporte diretamente nas vizinhanças, sem
  precisar de um score que diga "mudou X%".
- **`chairman_nn`** não é um falso positivo — é um resultado informativo:
  o núcleo institucional permanece estável, mas a *ecologia lexical* da
  palavra mudou (figuras históricas religiosas/militares → executivos
  corporativos/políticos contemporâneos). "Isso é mudança da ecologia
  lexical da palavra, não do seu sentido. O instrumento capturou
  corretamente. **O Spearman do SemEval não mede isso** porque SemEval
  pergunta sobre mudança de sentido lexical, não sobre mudança de
  ecologia."

### O exemplo canônico que articula a contribuição

`tmp/27` propõe um exemplo em português para tornar a ideia concreta:
`negro@1950` vs. `negro@2020`. Não há mudança de sentido lexical — a
palavra continua referindo-se a pessoas negras. Mas as vizinhanças
lexicais mudam profundamente: contextos de submissão e discriminação dão
lugar a contextos de resistência e identidade. Um instrumento que mostre
essa mudança **sem qualquer anotação de sentido** é diretamente útil para
sociolinguistas e historiadores da língua — esse é o público que o
reframing tem em mente.

### A contribuição em três componentes

1. **Arquitetura**: um modelo único contínuo, vs. dois modelos
   independentes alinhados (Hamilton) ou um modelo externo fora do
   domínio (APD+BERT).
2. **Representações contextuais in-domain**: cada ocorrência é
   representada no seu contexto real, treinada sobre o corpus histórico —
   diferente de word2vec (estático) e de BERT fora do domínio.
3. **Consulta temporal sem alinhamento**: `similares(w@t)` é respondido
   diretamente pelos vizinhos no checkpoint `theta_t`, sem projetar dois
   espaços num referencial comum.

### O papel do SemEval, depois do reframing

O Spearman não desaparece — muda de **objetivo primário** para
**validação secundária**: "um Spearman positivo e acima de zero com 37
palavras mostra que as vizinhanças que o instrumento produz têm
correlação com julgamentos humanos de 'algo linguisticamente relevante
mudou'". O resultado já disponível (APD = 0,210) é tratado como
**suficiente para essa função** — melhorar esse número é desejável, mas
não é mais bloqueador.

## Um desvio explorado e parcialmente abandonado: transporte ótimo (OT)

Em paralelo ao reframing, surgiu uma ideia técnica para refinar o APD:
em vez de contar *quantos* vizinhos mudaram, medir o **custo semântico**
de substituir os vizinhos antigos pelos novos — via
[transporte ótimo (OT)](conceitos/04-perfis_relacionais_e_apd.md#ot).

### A intuição

```text
secretary -> director   (deveria custar pouco: mesmo campo institucional)
angle     -> aircraft   (deveria custar muito: campos muito diferentes)
```

`tmp/25` formaliza isso: cada palavra é representada como uma distribuição
de massa sobre seus `k` vizinhos mais próximos em cada período; o custo
`c(u,v)` entre dois tokens de referência é a distância relacional entre
eles; e `Delta_OT(w)` é o custo mínimo de transportar a distribuição de D0
para a distribuição de D1.

### O teste de mesa com números reais

Com massa uniforme e os quatro personagens (vizinhos escolhidos
manualmente):

| Personagem | `Delta_OT` (exemplo manual) |
|---|---:|
| `chairman_nn` | 0,034 |
| `tree_nn` | 0,143 |
| `plane_nn` | 0,400 |
| `graft_nn` | 0,494 |

Essa ordem (`graft > plane > tree > chairman`) é exatamente a esperada.
Mas ao remover a seleção manual e usar top-k automático com várias
combinações de `k` e temperatura `tau`:

| `k` | massa | Spearman (37 alvos) | rank `plane` | rank `chairman` |
|---:|---|---:|---:|---:|
| 4 | uniforme | 0,145 | 11 | 36 |
| 10 | tau=0,05 | **0,196** | 8 | 36 |
| 50 | uniforme | 0,129 | 25 | 26 |

O melhor valor (0,196) ainda fica **abaixo do APD (0,210)**. E com massa
uniforme em `k=50`, a separação `plane`/`chairman` quase desaparece
(0,176 vs. 0,165) — o oposto do que se queria.

### Por que o custo "não corrige, reorganiza"

A resposta de `tmp/26` à proposta identifica o erro conceitual central:

> O documento trata `c(plate, boat) = 0,299` como possível defeito... Isso
> não é um defeito do OT. É a consequência de uma escolha de custo... O
> erro conceitual é esperar que um custo definido sobre o espaço do
> TimeFormer corrija um padrão de representação que o próprio TimeFormer
> aprendeu.

Em outras palavras: **OT não extrapola a qualidade das representações —
ele as reorganiza de forma interpretável**. Onde a geometria do modelo já
separa bem os campos (`chairman`: institucional vs. institucional;
`plane`: geométrico/material vs. transporte), o OT acerta. Onde o modelo
confunde campos, o OT herda a confusão.

A resposta também propõe correções concretas: usar **suporte comum**
`U(w) = N_0^k(w) ∪ N_1^k(w)` em vez de top-k separados (isso evita que
`tree_nn`, cujos vizinhos parcialmente se sobrepõem entre períodos, seja
penalizado por ter que "transportar" tokens que na prática já eram
parecidos); usar custo `c_0(u,v) = 1 - cos(h_0(u), h_0(v))` fixado em D0
em vez da média entre D0 e D1 (evita circularidade); e fixar `tau=0,05`
**antes** de olhar o gold.

Três condições de falsificação foram definidas para essa formulação
revisada (K∈{10,20,50}, tau∈{0,03;0,05;0,10}):

1. `Delta_OT(plane) > Delta_OT(chairman)` e `Delta_OT(graft) >
   Delta_OT(tree)` em **todas** as 9 combinações;
2. `Spearman >= 0,20` em pelo menos 6 das 9 combinações;
3. `rank(chairman) <= 30` em todas as 9 combinações.

### Veredito: OT como fingerprint, não como score

A recomendação final é dupla:

- **Score primário de ranking**: continua sendo APD com controle de
  campo (`Delta_adj(w) = APD(w) - mediana(APD(controles de campo))`),
  porque tem Spearman marginalmente superior e menor sensibilidade a
  hiperparâmetros.
- **OT entra como ferramenta de caracterização**: para cada palavra, além
  de um número, reportar o **fluxo ótimo** — por exemplo, para `plane_nn`:

  ```text
  line_nn   -> ship_nn    (custo 0,43, massa 0,41)
  angle_nn  -> route_nn   (custo 0,31, massa 0,34)
  plate_nn  -> machine_nn (custo 0,28, massa 0,13)
  column_nn -> boat_nn    (custo 0,37, massa 0,12)
  ```

  Esse fluxo é "uma caracterização semântica auditável da mudança" — não
  apenas "quanto mudou", mas "de quê para quê, com que custo". É
  precisamente o tipo de produto que o reframing de `tmp/27` pede: uma
  saída interpretável para o pesquisador, não um score competindo no
  Spearman.

## A visão de longo prazo: campo temporal relacional

`tmp/30` fecha o capítulo com um exercício de pensamento sobre a "versão
forte" da ideia, usando um vocabulário de referência trivial (`line,
angle, surface, ship, pilot, flight`) e dois períodos fictícios (1900 e
2000) para `plane`:

```text
R_1900(plane) = [0.90, 0.85, 0.80, 0.10, 0.05, 0.05]   # line,angle,surface,ship,pilot,flight
R_2000(plane) = [0.15, 0.10, 0.20, 0.65, 0.90, 0.95]

Delta(plane, 1900->2000) = R_2000 - R_1900
                          = [-0.75, -0.75, -0.60, +0.55, +0.85, +0.90]
```

O vetor `Delta` é legível por si só: `plane` perdeu relação com `line,
angle, surface` e ganhou relação com `ship, pilot, flight` — "a seta
temporal de `plane`". Com esse vetor, em princípio é possível **avançar**
(`R_1900 + Delta = R_2000`), **retroceder** (`R_2000 - Delta = R_1900`), e
até **interpolar** um período intermediário fictício (`R_1950 = R_1900 +
0.5 * Delta`), que produziria um perfil "em transição" — resíduos
geométricos ainda presentes, mas com aviação já ativada.

A diferença em relação a Hamilton é descrita como epistemológica, não só
computacional:

> Hamilton: mudança = distância entre dois pontos alinhados. Campo
> temporal relacional: mudança = **vetor de transformação entre perfis de
> relações linguísticas**.

Essa visão exige, para ser realizada de verdade, um corpus com mais de
dois períodos (`1900, 1920, ..., 2000`) — algo que o SemEval-2020 Task 1
não oferece (só D0 e D1). Por isso, `tmp/30` é registrado como **direção
de longo prazo**, não como próximo experimento: o experimento mínimo
proposto é calcular `Delta(w) = R_{D1}(w) - R_{D0}(w)` para os 37 alvos e
relatar ganhos/perdas/turnover — essencialmente, formalizar o que já foi
feito qualitativamente nas vizinhanças temporais desta fase.

## O que vale levar deste capítulo

- A virada deste capítulo não é tecnológica — é de **enquadramento**. Os
  mesmos números e vizinhanças que pareciam "falso positivo"
  (`chairman_nn`) e "falso negativo" (`plane_nn`) sob a moldura "detector
  vs. gold do SemEval" tornam-se **resultados corretos e informativos**
  sob a moldura "instrumento de consulta temporal".
- A "régua móvel" — o mesmo contexto sendo classificado de forma oposta
  por `theta_0` e `theta_1` — deixa de ser tratada como ruído a eliminar
  e passa a ser **o próprio fenômeno que o treino contínuo captura**.
- O desvio de OT foi valioso mesmo não "vencendo": ele não superou o APD
  como score (0,196 < 0,210), mas produziu uma representação auxiliar — o
  fluxo de transporte — que é exatamente o tipo de saída interpretável que
  o reframing de instrumento pede. "OT não corrige a geometria do modelo,
  ele a reorganiza de forma legível" é uma lição que se generaliza: **nem
  todo método precisa ganhar no Spearman para ser útil ao instrumento**.
- A pendência mais concreta que sai deste capítulo é a comparação direta
  com Hamilton 2016 (word2vec + Procrustes) sob o mesmo protocolo de
  vizinhanças — ela está marcada como "prioridade máxima" em `tmp/24` e
  `tmp/27`, mas, pelos registros desta fase, ainda não foi executada.
  Junto com o relatório qualitativo dos 37 alvos (não só dos quatro
  personagens), essa comparação é o que o capítulo 08 e seguintes
  precisam decidir como/quando endereçar.

## Conceitos novos usados neste capítulo

- [Transporte ótimo (OT) e custo entre distribuições](conceitos/04-perfis_relacionais_e_apd.md#ot)
- [Procrustes e alinhamento de espaços (Hamilton 2016)](conceitos/04-perfis_relacionais_e_apd.md#procrustes)
- [Softmax e temperatura](conceitos/01-correlacao_e_similaridade.md#softmax)
- [Vizinhanças lexicais como saída interpretável vs. score escalar](conceitos/06-wsd_e_sentido_lexical.md)
