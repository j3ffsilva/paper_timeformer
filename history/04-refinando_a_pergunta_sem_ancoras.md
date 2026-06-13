# Capítulo 4 — Refinando a pergunta: tirando as âncoras do caminho

> Fontes originais: `tmp/11-claude_anchorless_relational_displacement_prompt.md`,
> `docs/09-relational_profile_formalization.md`.

## O problema que ainda sobrava

Depois da validação do capítulo 03, a formulação relacional
(`r_t(w)[v] = similaridade_t(w,v)`, `delta_rel = r_b(w) - r_a(w)`) estava
de pé. Mas, na prática, toda vez que se calculava `r_t(w)`, era preciso
decidir *contra o quê* — contra qual conjunto de palavras `v`?

Nos primeiros pilotos com corpus real (mencionados de forma prospectiva
em `docs/02`, mas executados nesta fase), a resposta foi "uma lista fixa
de palavras-âncora", escolhidas por frequência ou por serem
presumivelmente estáveis. Mesmo depois de trocar o probe sintético por
ocorrências reais, os resultados no SemEval-2020 Task 1 continuaram
ruins. O `tmp/11` levanta a hipótese de que **o problema não era o probe,
era a própria ideia de âncora pré-definida**:

> a noção de "âncoras" entrou cedo demais e talvez tenha deformado o
> problema.

## A intuição: o círculo social, não os amigos individuais

`tmp/11` propõe uma analogia que se tornaria recorrente no projeto:

> Uma pessoa pode continuar no mesmo círculo social mesmo que todos os
> amigos mudem um pouco. O que importa é se a **estrutura do círculo**
> mudou, não cada microvariação local.

Aplicado a uma palavra: `gay` em 1920 tinha como vizinhos `happy`,
`cheerful`, `merry`; em 1980, `lesbian`, `homosexual`, `queer`. A pergunta
certa não é "a distância de `gay` até `happy` especificamente mudou?" —
é "o tipo de palavra que cerca `gay` mudou de campo semântico?". Se a
resposta depende de qual lista de âncoras foi escolhida de antemão, a
medida está sendo deformada pela escolha, não pelo fenômeno.

A formulação proposta nesse prompt — ainda abstrata — era:

```text
R_t(w): V_t \ {w} -> R
R_t(w)[v] = sim_t(w, v)

Delta_t(w) = d(R_t(w), R_{t+1}(w))
```

Três operacionalizações foram colocadas em discussão: (1) restringir a
`V* = V_t ∩ V_{t+1}` (vocabulário comum aos dois períodos); (2) comparar
apenas o top-k de vizinhos via Jaccard ou
[Rank-Biased Overlap (RBO)](conceitos/04-perfis_relacionais_e_apd.md#rbo);
(3) transformar similaridades em distribuição via softmax e comparar com
JSD. Cada uma tinha um problema: `V*` ainda é "uma âncora gigante
implícita"; top-k introduz um parâmetro `k` arbitrário; e softmax sobre
similaridades ainda depende de como `sim_t` foi calculada.

## A formalização que ficou: perfil relacional via log-PMI

A resposta que se consolidou como "definição canônica" — `docs/09`,
status explicitamente marcado como "base matemática para implementação e
paper" — resolve as três objeções de uma vez, mudando *onde* o perfil
relacional é calculado: não nas coordenadas internas do Transformer
(que mudam de checkpoint para checkpoint), mas na **saída do MLM head**,
que é sempre um vetor de probabilidades sobre o vocabulário de tokens —
um espaço que não muda de sistema de coordenadas entre `theta_0` e
`theta_1`.

### Os dois ingredientes

Para cada palavra `w` e checkpoint `theta_t`:

1. **Distribuição condicional** — pega todas as ocorrências reais de `w`
   no corpus do período `t`, mascara `w` em cada uma, e tira a média das
   distribuições previstas pelo MLM head na posição mascarada:

   ```text
   q_t(w) = média sobre c em C_t(w) de P_{theta_t}(. | c com w mascarado)
   ```

   `q_t(w)` é uma distribuição sobre todo o vocabulário `V`: "que palavras
   o modelo espera ver no lugar de `w`, dado tudo que já viu até `t`?"

2. **Distribuição marginal (prior do modelo)** — a mesma pergunta, mas sem
   nenhuma informação sobre `w`, usando o probe neutro `[CLS] [MASK]
   [SEP]`:

   ```text
   p_t = P_{theta_t}(. | [CLS] [MASK] [SEP])
   ```

   `p_t` captura "o que o modelo prevê em geral", incluindo o efeito de
   palavras simplesmente terem se tornado mais ou menos frequentes no
   corpus do período `t` — independentemente de `w`.

### O perfil relacional

```text
R_t(w)[v] = log( q_t(w)[v] / p_t[v] ),   para todo v em V
```

Isso é exatamente **log-PMI** (Pointwise Mutual Information,
[conceitos/01](conceitos/01-correlacao_e_similaridade.md#pmi)): o quanto
`v` é mais (ou menos) esperado ao lado de `w` do que seria por puro acaso
de frequência. `R_t(w)[v] > 0` significa "`v` é um marcador semântico
positivo de `w` em `t`"; `R_t(w)[v] ≈ 0` significa "sem associação
específica"; `R_t(w)[v] < 0` significa "repulsão semântica".

### Por que isso resolve o problema das âncoras

| O que a formulação elimina | Por quê |
|---|---|
| Lista de âncoras pré-definidas | `p_t` já é o "fundo" do modelo; as dimensões de `R_t(w)` que ficam perto de zero em todos os períodos *são* as âncoras — descobertas pela análise, não escolhidas antes |
| Parâmetro `k` de vizinhança | `R_t(w)` é definido sobre `V` inteiro, sem corte de top-k |
| Alinhamento entre checkpoints | A comparação acontece sobre **tokens de string** (o vocabulário `V`, fixo), não sobre coordenadas internas que mudam de checkpoint para checkpoint |
| Filtragem manual por classe gramatical | Palavras funcionais (`a`, `the`, `of`) têm `p_t[v]` alto em qualquer corpus, então seu PMI já fica baixo automaticamente |

Esta última coluna também responde a uma preocupação do capítulo 03: como
o perfil é calculado sobre a saída do MLM head (probabilidades sobre
tokens), e não sobre os vetores ocultos do Transformer, rotações,
reflexões ou reescalamentos anisotrópicos de `theta_t` **não afetam**
`R_t(w)` — eles afetam `q_t(w)` e `p_t` da mesma forma, e o log-PMI
cancela esse efeito comum (propriedade **P2** do documento: "normalização
por deriva de domínio").

### Duas formas de medir o deslocamento

```text
# versão cosseno — sensível a reorientação do campo semântico
Delta(w, t0, t1) = 1 - cos(R_t0(w), R_t1(w))

# versão JSD sobre associações positivas (PPMI)
R_t+(w)[v] = max(0, R_t(w)[v])
pi_t(w) = R_t+(w) / ||R_t+(w)||_1
Delta_JSD(w, t0, t1) = JSD(pi_t0(w), pi_t1(w))
```

A versão JSD é interpretável em bits: "quanta informação distingue o
campo semântico de `w` em `t0` do campo em `t1`". Esse `Delta_JSD` é, na
prática, a métrica `direct_jsd`/`log-PMI` que reaparece nos capítulos 05
e 06 como uma das poucas abordagens que sobrevivem ao teste do corpus
real.

## Exemplo concreto com os personagens

Para tornar isso menos abstrato, o que essa formalização prevê para os
quatro personagens (capítulo 00), antes de qualquer medição real:

- **`tree_nn`** (controle estável): `R_{t0}(tree\_nn)` e `R_{t1}(tree\_nn)`
  deveriam ter PMIs altos para um conjunto parecido de palavras
  ("planta", "madeira", "floresta" — ou seus equivalentes em 1810-1860 e
  1960-2010). `Delta_JSD` baixo.
- **`plane_nn`** (substituição de sentido): em D0, PMI alto com palavras
  do campo "superfície geométrica"; em D1, PMI alto com palavras do campo
  "aviação". Os dois conjuntos de marcadores positivos deveriam ser quase
  disjuntos — `Delta_JSD` alto.
- **`graft_nn`** (diversificação): em D1, o perfil deveria mostrar PMI
  positivo simultaneamente para marcadores agrícolas *e* para marcadores
  médicos/de corrupção — um perfil "mais largo", não simplesmente
  deslocado. Isso é uma previsão que distingue diversificação de
  substituição, algo que a métrica de cosseno isolada pode não capturar
  bem (o capítulo 08 volta a essa distinção).
- **`chairman_nn`** (falso positivo clássico): a expectativa é que, mesmo
  que o *contexto* mude (parlamento → empresa), o núcleo de marcadores
  ligados a "presidir", "liderar", "organização" permaneça com PMI
  positivo nos dois períodos — então `Delta_JSD` deveria ficar moderado,
  não alto.

## Questões deixadas abertas

O próprio `docs/09` lista quatro questões que não foram resolvidas nesta
fase, e que retornam nos capítulos seguintes:

1. **Estabilidade de `p_t`**: o probe neutro `[CLS] [MASK] [SEP]` pode não
   ser representativo se o modelo aprender a tratar essa sequência como
   um padrão especial, diferente de frases naturais.
2. **Cobertura de `C_t(w)`**: palavras raras com poucas ocorrências em
   `D_t` produzem `q_t(w)` ruidoso — sugerida uma contagem mínima (≥10
   ocorrências).
3. **Tamanho de `V`**: 30 mil dimensões é tratável, mas esparso na
   prática.
4. **Trajetória como sequência de perfis**: para mais de dois períodos,
   `R_{t0}(w), ..., R_{tn}(w)` define uma trajetória — métricas de
   velocidade, aceleração e persistência podem ser computadas sobre essa
   sequência sem componente adicional. (Essa ideia ecoa diretamente as
   métricas de forma temporal do capítulo 02 — `M_final`, `L`,
   `ShapeError` — agora aplicadas a perfis log-PMI em vez de perfis
   `r_t(w)` baseados em JSD entre distribuições de contexto.)

O item 2 (cobertura mínima por palavra) é particularmente importante: ele
prenuncia o tipo de problema de "poucas ocorrências, estimativa ruidosa"
que se tornaria central no capítulo 05, quando essa formulação encontrou
o corpus real do SemEval pela primeira vez.

## O que vale levar deste capítulo

- A virada deste capítulo não é uma nova métrica — é uma mudança de
  **onde** a comparação acontece: da geometria interna do Transformer
  (que muda de checkpoint para checkpoint) para a saída do MLM head sobre
  o vocabulário de tokens (que é compartilhado por construção).
- "Âncoras emergem, não são pressupostas" (propriedade P3) resolve, de
  forma elegante, a preocupação levantada no capítulo 03 sobre listas de
  âncoras arbitrárias deformarem o resultado.
- O log-PMI sobre `q_t(w)/p_t` também responde, de forma natural, à
  propriedade P2 (normalização por deriva de domínio) e à preocupação do
  capítulo 03 sobre deformações anisotrópicas — porque a comparação não
  depende mais de coordenadas internas do modelo.
- As quatro questões abertas (estabilidade de `p_t`, cobertura mínima,
  esparsidade, trajetória) formam a lista de verificação que o capítulo 05
  carrega para o primeiro contato com o SemEval real.

## Conceitos novos usados neste capítulo

- [PMI e log-PMI](conceitos/01-correlacao_e_similaridade.md#pmi)
- [PPMI e Jensen-Shannon sobre distribuições](conceitos/01-correlacao_e_similaridade.md#jensen-shannon)
- [Rank-Biased Overlap (RBO)](conceitos/04-perfis_relacionais_e_apd.md#rbo)
- [MLM head e probe neutro](conceitos/02-encoders_e_camadas.md#mlm-head)
