# Capítulo 10 — A régua externa: WSD congelado e o "NO-GO estrito" da Porta 1

> Fonte original: `docs/15-external_wsd_plane_gate1_results.md`.

O capítulo 09 terminou com uma decisão explícita: parar de tentar
"salvar" o encoder MLM temporal por engenharia (L2-SP, distillation,
congelamento) e testar, em paralelo, uma **régua externa** — um modelo de
desambiguação de sentido (WSD) já treinado, congelado, sem nenhum ajuste
no SemEval. A ideia, que vem desde o capítulo 06 ("a parede da
identificabilidade": `P_t(cluster|w) != P_t(sense|w)`), é que talvez o
caminho mais direto para medir "mudança de sentido" não seja inferir
sentidos a partir de geometria contextual — é **usar um modelo que já foi
treinado especificamente para prever sentidos**, e perguntar a ele
diretamente.

Este capítulo é sobre a primeira execução desse plano — chamada de
"Porta 1" — e seu resultado: **NO-GO estrito**, mas um NO-GO informativo,
não um beco sem saída.

## O desenho: tudo decidido antes de rodar

Seguindo a disciplina que já apareceu nos capítulos 03 e 07 (decidir
critérios antes de olhar resultados), a Porta 1 foi inteiramente
especificada com antecedência:

- **Modelo**: LMMS-SP, que usa `bert-large-cased` **congelado** como
  encoder e um inventário de sentidos do **WordNet 3.0**. A decisão é por
  vizinho mais próximo (cosseno) entre os cinco sentidos nominais de
  `plane` no WordNet.
- **Palavra-alvo única**: `plane_nn` — escolhida porque é o personagem com
  a mudança de sentido mais documentada e mais citada em todo o projeto
  (capítulos 06, 07, 08).
- **Subconjuntos heurísticos**, com cortes de aprovação definidos
  *a priori*:

  | Subconjunto | N | Corte de aprovação |
  |---|---:|---|
  | D0 geometria | 182 | acurácia >= 0,75 |
  | D0 ferramenta | 19 | acima do acaso (1/3), com IC |
  | D1 avião | 208 | acurácia >= 0,80 |

- **Verificação extra**: a acurácia macro estratificada deve ter IC 95%
  acima de `1/3` (acaso entre 3 classes), e a frase histórica usada como
  exemplo em vários capítulos anteriores —

  ```text
  "plate figure represent an inclined plane"
  ```

  — deve ser classificada como `geometry`.

- **Regra de decisão conjuntiva**: Gate 1 = GO somente se **todas** as
  verificações passarem.

Os cinco sensekeys do WordNet usados para agrupar os sentidos:

```text
aircraft: plane%1:06:01::
geometry: plane%1:25:00::
tool:     plane%1:06:00:: e plane%1:06:02::
other:    plane%1:26:00::
```

## Resultado: 4 de 5 passam — mas a regra é conjuntiva

| Subconjunto | Acurácia | IC 95% | Corte | Passou |
|---|---:|---:|---:|---|
| D0 geometria | **0,984** | [0,962; 1,000] | >= 0,75 | sim |
| D0 ferramenta | **0,211** | [0,053; 0,421] | acima de 1/3 | **não** |
| D1 avião | **1,000** | [1,000; 1,000] | >= 0,80 | sim |

Acurácia macro estratificada: **0,731**, IC 95% `[0,677; 0,800]` —
confortavelmente acima do acaso (`1/3`). A frase-âncora também foi
classificada corretamente:

```text
esperado: geometry
predito:  geometry
sensekey: plane%1:25:00::
margem:   0,041
```

Ou seja: **quatro das cinco verificações pré-registradas passaram, com
margens folgadas** — a oposição temporal principal (geometria em D0 vs.
aviação em D1) foi lida quase perfeitamente (98,4% e 100%). Mas a regra de
decisão era conjuntiva, e o subconjunto "D0 ferramenta" (19 ocorrências)
ficou em 21,1% — **abaixo até do acaso de 1/3**, com IC que não exclui
valores baixos. Pela regra pré-definida:

```text
Gate 1 = NO-GO
```

## Por que "ferramenta" falhou: heurística ruim, ou WSD ruim, ou os dois?

A auditoria das 19 ocorrências do subconjunto "ferramenta" mostrou uma
distribuição de predições:

| Predição do LMMS | N |
|---|---:|
| geometry | 12 |
| tool | 4 |
| aircraft | 3 |

Inspecionando os exemplos, dois problemas distintos aparecem:

1. **A heurística de rotulagem tem falsos positivos**: uma ocorrência
   contendo `plane tree` (uma espécie de árvore — sentido botânico, nem
   geometria nem ferramenta!) foi rotulada como "ferramenta" só porque a
   frase também contém a palavra `timber`. Esse exemplo nunca deveria ter
   sido "ferramenta" para começo.
2. **O LMMS também tem dificuldade real com o sentido raro**: exemplos
   inequivocamente de ferramentas de marcenaria históricas — `mould
   plane`, `bench planes`, `jack plane`, `smooth plane` — foram
   frequentemente classificados como `geometry` em vez de `tool`.

A leitura honesta, registrada explicitamente no documento: **não é
possível, neste momento, separar quanto da falha vem da heurística de
avaliação e quanto vem de uma limitação real do LMMS no sentido
histórico/raro de "ferramenta"**. E — ponto importante de rigor
metodológico — **remover esses 19 exemplos agora, depois de ver as
predições, invalidaria a Porta 1** (seria exatamente o tipo de ajuste
post-hoc que a disciplina de pré-registro existe para evitar).

## Interpretação: nem "incompatível" nem "suficiente"

O resultado rejeita as duas leituras extremas que poderiam ter sido feitas
antes do experimento:

- **Não é verdade que "WSD externo não funciona neste corpus lematizado"**.
  Os sentidos dominantes e bem representados (geometria em D0, aviação em
  D1 — exatamente a oposição temporal que `plane_nn` é citado para
  ilustrar desde o capítulo 00) foram lidos quase perfeitamente, a frase
  histórica de referência foi classificada corretamente, e a acurácia
  macro fica bem acima do acaso.
- **Mas também não é verdade que "o LMMS, do jeito que está, é suficiente
  como atlas lexical geral para este projeto"**. Ele falhou exatamente no
  caso que mais importa para um estudo de mudança semântica histórica: um
  sentido raro, de cauda longa, em registro antigo (`plane` como
  ferramenta de marcenaria do século XIX).

Em outras palavras: **a régua externa funciona bem onde o sinal é forte e
falha onde ele é fraco** — o que é, em si, um resultado mais informativo
do que um "GO" ou "NO-GO" simples teria sido. Um GO total poderia ter
escondido essa fragilidade na cauda; um NO-GO categórico ("WSD não
funciona aqui") teria descartado uma ferramenta que, no caso dominante,
funciona quase perfeitamente.

## Próximos passos definidos

O documento já define cinco passos concretos, em ordem:

1. **Congelar este resultado como confirmatório** — não alterar palavras-
   chave, cortes ou exemplos da Porta 1 original (preservar a integridade
   do pré-registro para uso futuro).
2. **Adjudicação cega das 19 ocorrências de ferramenta**: dois anotadores
   humanos rotulam cada ocorrência como `tool`, `geometry`, `aircraft`,
   `botanical` ou `unclear`, **sem ver as predições do LMMS**. Isso separa
   "erro da heurística de avaliação" de "erro real do WSD" — mas não
   reverte o NO-GO já registrado.
3. **Testar um segundo WSD externo congelado** (ConSeC é a opção preferida;
   BEM foi descartado porque seu checkpoint oficial não está mais
   disponível publicamente) no mesmo conjunto adjudicado — para saber se a
   falha em "ferramenta" é específica do LMMS ou generaliza.
4. **Regra de parada explícita**: se LMMS **e** ConSeC falharem nos exemplos
   adjudicados de ferramenta, a conclusão passa a ser que **sentidos
   históricos raros exigem adaptação de domínio, um inventário de sentidos
   histórico, ou supervisão adicional** — não que "WSD externo não
   funciona" em geral.
5. **Só abrir a "Porta 2"** (estender de `plane_nn` para `plane_nn,
   graft_nn, chairman_nn, tree_nn` — os quatro personagens — sem
   engenharia por palavra) **se algum modelo externo passar** na etapa 3-4.

A reflexão final do documento reposiciona o que é "o experimento de maior
valor informacional agora": não é mais nenhuma forma de treino temporal —
é **separar, de forma cega, erro de heurística de falha real de WSD na
cauda histórica**, e verificar replicação num segundo modelo.

## O que vale levar deste capítulo

- Este é o primeiro experimento do projeto desenhado especificamente para
  responder à "parede de identificabilidade" do capítulo 06 de forma
  direta — usando um modelo que prediz sentidos, não geometria contextual
  que precisa ser interpretada como proxy de sentido.
- A regra conjuntiva pré-registrada fez exatamente o que deveria fazer:
  impediu que um resultado "4 de 5, e o macro está ótimo" fosse
  arredondado para "GO". Sem essa disciplina, seria tentador ignorar o
  subconjunto de 19 exemplos (o menor dos três) e declarar sucesso — e a
  fragilidade na cauda histórica passaria para o capítulo seguinte sem ser
  notada.
- `plane_nn` segue sendo o personagem mais informativo do projeto: a mesma
  palavra que mostrou o "eixo geometria vs. transporte" mais limpo no
  experimento dos campos manuais (capítulo 06) e no oráculo BERT-base
  (capítulo 08) agora também mostra, com uma régua de WSD de fato treinada
  para sentidos, exatamente onde a dificuldade remanescente está: não na
  oposição principal (quase perfeita, 98-100%), mas num terceiro sentido
  raro que nenhuma das abordagens anteriores havia isolado explicitamente.
- O NO-GO não encerra a linha de WSD externo — ele a torna **mais
  específica**: a próxima pergunta não é "WSD externo funciona?", é "essa
  falha específica em sentidos raros e históricos replica num segundo
  modelo, ou é uma idiossincrasia do LMMS?".

## Conceitos novos usados neste capítulo

- [WSD (Word Sense Disambiguation) e inventários de sentido (WordNet)](conceitos/06-wsd_e_sentido_lexical.md#wsd)
- [Inventário, open set e cauda longa](conceitos/06-wsd_e_sentido_lexical.md#inventario)
- [Do corpus lematizado ao vetor da ocorrência](conceitos/09-dados_tokenizacao_e_contexto.md)
- [LMMS-SP e representações de sentido via embeddings congelados](conceitos/06-wsd_e_sentido_lexical.md#lmms)
- [Acurácia macro estratificada e IC por bootstrap](conceitos/05-estatistica_experimental.md#bootstrap)
- [Regras de decisão conjuntivas e pré-registro](conceitos/05-estatistica_experimental.md#pre-registro)
- [Adjudicação cega entre anotadores](conceitos/06-wsd_e_sentido_lexical.md#adjudicacao)
