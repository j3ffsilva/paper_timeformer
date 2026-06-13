# Capítulo 2 — A virada relacional: medir depois, não ensinar antes

> Fontes originais: `tmp/01-claude_displacement_reorientation_prompt.md`,
> `docs/02-novo_planejamento.md`,
> `docs/06-structural_relational_experiment_preregistration.md`,
> `docs/13-paper1_iberamia_timeformer.pdf` ("Amnesia by Design").

## A pergunta que disparou a virada

No fim do capítulo 01, a engenharia (Token-Time/FiLM, `L_traj`, Set
Transformer, teacher/student) estava funcionando no sintético e tinha
recebido revisões favoráveis ("prosseguir, com correções"). Mesmo assim,
um novo pedido de revisão externa (`tmp/01`, o "prompt de reorientação")
levantou uma pergunta mais incômoda, que não era sobre bugs:

> Se o modelo *recebe o tempo como entrada* e é treinado para produzir uma
> representação de trajetória, como saber se a geometria temporal que
> observamos no final é uma propriedade do *significado das palavras*, ou
> apenas um *artefato de como ensinamos o modelo a representar o tempo*?

Em outras palavras: ao colocar `token@time = (h_s(t), m_s(t))` dentro da
arquitetura e treinar tudo de ponta a ponta, o sistema de coordenadas
semântico do modelo deixa de ser fixo — ele próprio é modelado em função
do tempo. Isso torna quase impossível separar "a palavra mudou de
sentido" de "o modelo aprendeu a desenhar o espaço de forma diferente
para cada período".

## A proposta inicial: deslocamento sobre um espaço-base congelado

O `tmp/01` propôs uma primeira correção: treinar um Transformer **padrão**
(sem nenhum sinal de tempo) só no período inicial `t0`, **congelar** esse
modelo para sempre — ele define o "sistema de coordenadas de referência" —
e aprender, por fora, um módulo de deslocamento `delta(w,t)`:

```text
e(w@t) = b(w) + delta(w,t)
```

com `delta(w,t0) = 0` por definição. A trajetória de uma palavra deixaria
de ser algo que o modelo aprende diretamente; seria derivada, depois do
treino, da sequência `delta(w,t0), delta(w,t1), ..., delta(w,tn)`.

Essa proposta tinha um mérito real: tornava o "ponto zero" explícito e
movia a trajetória para *análise posterior*, em vez de *representação
aprendida* — um princípio que sobrevive até hoje. Mas a revisão desse
prompt (que não está detalhada aqui, mas seu resultado está em
`docs/02`) identificou um problema prático: um Transformer-base treinado
só em `t0` e depois **congelado** teria que processar textos de `t1`
(1960-2010, no caso do SemEval) sem nunca ter visto o vocabulário, a
sintaxe ou os tópicos desse período. Isso poderia confundir "o modelo não
reconhece o texto novo" com "a palavra mudou de sentido".

## A formulação que ficou: perfis relacionais entre checkpoints cronológicos

A formulação que efetivamente foi adotada (`docs/02-novo_planejamento.md`,
"versão reorientada") resolve isso de outra forma. Em vez de congelar um
modelo em `t0`, ela deixa o modelo **continuar treinando cronologicamente**
— exatamente como antes — mas muda *o que é comparado entre checkpoints*.

```text
theta_0 = treino(D_0)
theta_1 = continua_treino(theta_0, D_1)
```

Nenhum identificador de período é dado ao modelo — `theta_1` é só um
Transformer comum que continuou sendo treinado. A diferença está em como
medimos mudança *depois*:

```text
r_t(w)[v] = similaridade_t(w, v)
delta_rel(w, a, b) = r_b(w) - r_a(w)
```

Ou seja: em vez de perguntar "a coordenada absoluta de `plane_nn` mudou
entre `theta_0` e `theta_1`?" (pergunta sensível a qualquer rotação ou
reescala global do espaço, que pode acontecer só por re-otimização),
perguntamos **"o círculo de palavras que `plane_nn` considera parecidas
mudou?"**. Se o espaço inteiro girar ou for reescalado de forma uniforme,
as relações internas — e portanto `delta_rel` — permanecem
aproximadamente zero. Esse é o teste de sanidade conceitual da ideia
relacional: **uma transformação global do espaço não deve, por si só,
produzir um falso sinal de mudança semântica.**

A frase-guia adotada para o projeto a partir daqui é:

> Modelamos mudança semântica não como movimento em um sistema absoluto de
> coordenadas de embeddings, mas como alteração no perfil relacional de
> uma palavra ao longo de checkpoints de Transformer treinados
> cronologicamente.

### Como `r_t(w)` é calculado na prática

A primeira implementação concreta de `r_t(w)` não usa o vetor oculto do
sujeito diretamente — o próprio `docs/02` registra que "consultar
diretamente `h_subj` não recupera de forma confiável a direção semântica
plantada". Em vez disso, usa um **probe preditivo pós-Transformer**:

```text
[CLS] palavra [MASK] [MASK] [SEP]
```

O checkpoint `theta_t` prevê uma distribuição de probabilidade sobre
contextos plausíveis para essa frase, `q_t(w)`. O perfil relacional é a
similaridade entre essas distribuições, via divergência de
Jensen-Shannon:

```text
r_t(w)[v] = 1 - JS(q_t(w), q_t(v)) / log(2)
```

(JSD e o que ela mede de forma intuitiva estão em
[`conceitos/01-correlacao_e_similaridade.md`](conceitos/01-correlacao_e_similaridade.md#jensen-shannon).)

## O parente próximo: "Amnesia by Design" (paper IberAMIA)

Em paralelo a essa reorientação, o mesmo grupo produziu um trabalho
formal — submetido ao IberAMIA, `docs/13-paper1_iberamia_timeformer.pdf`,
"Amnesia by Design: Diagnosing Temporal Traceability in Transformer
Representations" — que **não é o TimeFormer em si**, mas testa, num
ambiente totalmente controlado, uma pergunta vizinha e complementar:

> Se um Transformer recebe o tempo como entrada explícita (Token-Time,
> Memory-Augmented, Additive), a representação de uma palavra realmente
> "se move" para a vizinhança correta do período consultado — ou o modelo
> apenas aprende a *rotular* o período sem que a geometria interna mude de
> forma rastreável?

Esse artigo define a propriedade de **traceabilidade temporal**: uma
representação é "temporalmente traceável" se, ao consultar o mesmo token
em períodos diferentes, ele ocupa vizinhanças geometricamente consistentes
com o significado daquele período — não basta o modelo "saber" em que
período está, ele precisa *posicionar* a palavra de forma diferente no
espaço.

Para testar isso, o artigo constrói uma suíte sintética com gramática
SVO (sujeito-verbo-objeto), 30 sujeitos divididos em três classes —
**stable**, **drifting**, **bifurcating** — e quatro arquiteturas:
`Standard` (sem condicionamento, baseline), `Additive` (deslocamento
global por período), `Token-Time` (projeção conjunta token-período) e
`Memory-Augmented` (histórico causal de protótipos). Três achados:

1. **Qualquer condicionamento temporal mais que dobra o deslocamento de
   vizinhança** em relação ao baseline `Standard` — condicionar pelo
   tempo, de fato, muda a geometria, não é só rótulo.
2. Sob marcadores de contexto confiáveis, `Additive` (o mais simples) e
   `Token-Time` (o mais complexo) ficam **equivalentes** em deriva
   agregada; `Token-Time` só se destaca quando os marcadores ficam
   degradados/ambíguos.
3. **Para sujeitos `bifurcating`** (que desenvolvem dois sentidos
   coexistentes), **a agregação por protótipo médio falha
   sistematicamente**: a média colapsa os dois sentidos num centroide que
   não representa nenhum dos dois.

Esse terceiro achado é o elo direto com o capítulo 01 (onde o `D6` de
bimodalidade só melhorava com supervisão sintética do Set Transformer) e
com o capítulo 04 (onde a questão "como detectar sentidos coexistentes
sem rótulos sintéticos" volta a aparecer). `graft_nn`, o personagem que
*diversifica* sentido (capítulo 00), é exatamente o tipo de caso que esse
achado prevê ser difícil para qualquer método baseado em protótipo médio.

A relação entre os dois trabalhos, então, é de **complementaridade, não
de sequência**: "Amnesia by Design" testa arquiteturas *com*
condicionamento temporal explícito e mostra onde a agregação por média
falha; o TimeFormer (a partir deste capítulo) testa a hipótese
**oposta** — um modelo *sem* condicionamento temporal, medido depois via
perfis relacionais — e tenta evitar exatamente esse tipo de armadilha
desde o desenho.

## Validando a virada: o experimento estrutural

A nova formulação só seria adotada como plano principal se sobrevivesse a
um teste honesto. `docs/06` é o pré-registro desse teste — fixa
hipóteses, condições e critérios de decisão **antes** de rodar o
experimento (essa disciplina de pré-registro reaparece em vários pontos
da história e está detalhada em
[`conceitos/05-estatistica_experimental.md`](conceitos/05-estatistica_experimental.md)).

O desenho usa um corpus sintético com 16 "âncoras" estáveis e 24
palavras-alvo organizadas em 6 quartetos — cada quarteto compartilha o
mesmo estado inicial e final, mas cada palavra do quarteto segue uma
**forma temporal** diferente:

| Forma | Padrão (exemplo `0.90 -> 0.10`) | Pergunta que testa |
|---|---|---|
| **Gradual (G)** | `0.90, 0.81, 0.72, ..., 0.10` | mudança lenta e acumulada é detectada? |
| **Abrupta (A)** | `0.90, 0.90, 0.90, 0.90, 0.90, 0.10, 0.10, 0.10, 0.10, 0.10` | um salto súbito é localizado corretamente no tempo? |
| **Transitória (T)** | `0.90, ..., 0.10, 0.10, 0.90, ...` | uma mudança que se reverte é "esquecida" ou deixa rastro? |
| **Oscilatória (O)** | `0.90, 0.10, 0.90, 0.10, ...` | muita atividade sem mudança final líquida é distinguida de mudança real? |

`G` e `A` compartilham começo e fim — testam se o método distingue
**como** se chegou ao mesmo destino. `T` e o controle nulo terminam no
mesmo lugar onde começaram — testam se o método enxerga um evento
intermediário que uma comparação simples início-vs-fim perderia.

### Resultados do experimento estrutural principal

| Condição | `M_final` (mediano) | % acima do p95 nulo | Caminho `L` | `Recovery` | `F_acc` | `ShapeError` |
|---|---:|---:|---:|---:|---:|---:|
| Gradual | 0.0719 | 77.8% | 0.196 | 0.000 | 0.653 | 0.117 |
| Abrupta persistente | 0.0722 | 94.4% | 0.292 | 0.209 | 0.911 | 0.194 |
| Transitória | 0.0273 | 0.0% | 0.353 | 0.660 | 0.914 | 0.228 |
| Oscilatória | 0.0245 | 11.1% | 0.683 | 0.703 | 0.925 | 0.204 |

Leitura, condição por condição:

- **Gradual**: o deslocamento final (`M_final`) fica acima do limiar do
  nulo em 77.8% dos casos — a mudança lenta acumulada é detectável.
- **Abrupta**: detectada com força ainda maior (94.4%) e direção correta
  (`F_acc=0.911`).
- **Transitória**: termina com `M_final` baixo (como esperado, ela
  "volta" ao estado inicial) e **0% acima do nulo** — corretamente, porque
  o destino final é igual ao do nulo. Mas `Recovery=0.660` mostra que o
  modelo "viu" um pico no meio do caminho e depois retornou — o evento
  intermediário deixou rastro, mesmo sem mudança final.
- **Oscilatória**: maior `Caminho` (0.683, mais atividade que todas as
  outras) mas `M_final` baixo e quase não passa do nulo (11.1%) — o
  método não confunde "muita atividade" com "mudança persistente".

Esse padrão é exatamente o que o pré-registro definia como sucesso: as
quatro formas temporais produzem assinaturas diferentes nas métricas, não
apenas magnitudes diferentes no final.

### A continuidade cronológica importa — mas rupturas ficam parcialmente "borradas"

Um segundo bloco de resultados comparou o regime `continual_real`
(treino cronológico de verdade) com `independent_period` (um modelo do
zero por período) e `cumulative_retrain` (um modelo do zero, mas vendo
`D_0 + ... + D_t`). Conclusão registrada:

> a trajetória relacional não é apenas uma propriedade dos dados vistos em
> cada checkpoint; ela também depende da história de otimização.

Ou seja, *como* o modelo chegou até `theta_t` (e não só *o que* ele viu)
afeta o perfil relacional — uma evidência a favor de treinar
continuamente em vez de, por exemplo, re-treinar do zero a cada novo
período.

Por outro lado, uma ablação testando rupturas abruptas em diferentes
pontos do tempo (`t3`, `t5`, `t7`) mostrou uma limitação importante: o
modelo **localiza corretamente o momento e a direção** do salto (erro de
localização = 0 nos três casos), mas **só cerca de 29-30% do caminho
relacional total fica concentrado no evento** — o resto se distribui
como "deriva" antes e depois do salto. A leitura honesta registrada foi:

> Timeformer recupera o momento e a direção local de rupturas abruptas,
> mas distribui parte do caminho relacional em deriva pré- e pós-evento.

Essa limitação — rupturas abruptas (como `plane_nn`, que troca de sentido
dominante quase de uma vez) ficam parcialmente "borradas" no perfil
relacional — é uma sombra que acompanha o projeto até os capítulos finais.

## O que vale levar deste capítulo

- A virada relacional resolve o problema central do capítulo 01 (mudança
  pode ser artefato de como o modelo representa o tempo) ao **remover
  qualquer sinal de tempo da entrada do modelo** e mover toda a medição
  de mudança para *depois* do treino, comparando perfis relacionais entre
  checkpoints.
- "Amnesia by Design" (mesmo grupo, IberAMIA) é um trabalho-irmão que
  testa o cenário *oposto* (condicionamento temporal explícito) e mostra
  que, **mesmo nesse cenário**, agregação por protótipo médio falha para
  palavras com sentidos coexistentes — um aviso que se aplica a
  `graft_nn` e que o TimeFormer também precisará enfrentar (capítulos 04 e
  08).
- O experimento estrutural (`docs/06`) validou as quatro formas temporais
  (gradual/abrupta/transitória/oscilatória) com controles nulos e
  placebo, e confirmou que continuidade cronológica importa — mas também
  documentou, desde o início, que rupturas abruptas são parcialmente
  "borradas" no perfil relacional. Esse resultado **passou** o teste e
  abriu caminho para o capítulo 03 (validação mais ampla no sintético) e,
  depois, para a tentativa no SemEval real (capítulo 05).

## Conceitos novos usados neste capítulo

- [Jensen-Shannon (JSD)](conceitos/01-correlacao_e_similaridade.md#jensen-shannon)
- [Perfil relacional e `delta_rel`](conceitos/04-perfis_relacionais_e_apd.md)
- [Checkpoints cronológicos (`theta_0`, `theta_1`, `theta_t`)](conceitos/02-encoders_e_camadas.md#checkpoints)
- [Treino contínuo, modelos independentes e réguas temporais](conceitos/08-desenhos_temporais_e_reguas.md)
- [Estimando: mudança de uso vs. mudança de sentido](conceitos/07-o_que_esta_sendo_medido.md)
- [Pré-registro e controles nulo/placebo](conceitos/05-estatistica_experimental.md#pre-registro)
