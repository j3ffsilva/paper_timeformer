# Conceitos 5 — Estatística experimental: como não se enganar com `n=37`

O SemEval-2020 Task 1 (inglês) tem **37 palavras-alvo**. Esse número
pequeno aparece, direta ou indiretamente, atrás de quase toda decisão
metodológica deste projeto a partir do capítulo 05. Este arquivo reúne as
ferramentas para lidar com ele honestamente.

<a id="p-valor"></a>
## Efeito, p-valor e intervalo de confiança

São três perguntas diferentes:

- **tamanho do efeito**: quão grande é o resultado observado?
- **p-valor**: quão incompatível o resultado é com um nulo específico?
- **intervalo de confiança**: quais valores do efeito continuam plausíveis
  diante da variabilidade amostral?

Um Spearman `rho=0,20` pode parecer relevante, mas com `n=37` seu intervalo
pode ser largo e incluir zero. Dizer apenas `p>0,05` também não prova que o
efeito é zero; significa que os dados não o distinguem com segurança do nulo
adotado.

O projeto passou a privilegiar:

```text
efeito pontual + intervalo + controle apropriado
```

em vez de classificar resultados apenas como "significativos" ou "não
significativos".

<a id="metricas-classificacao"></a>
## Acurácia, macro acurácia, ROC-AUC e AP

**Acurácia** é a fração de decisões corretas. Pode esconder classes raras:
acertar sempre a classe majoritária gera boa acurácia global.

**Macro acurácia** calcula a acurácia separadamente por grupo/classe e tira a
média, dando peso igual a grupos grandes e pequenos. Na Porta 1:

```text
macro = media(acuracia_geometria,
              acuracia_ferramenta,
              acuracia_aviao)
```

Assim, as 19 ferramentas pesam tanto quanto os 208 aviões.

**ROC-AUC** mede a probabilidade de um item positivo receber score maior que um
negativo ao variar todos os limiares. `0,5` corresponde ao acaso; `1,0`, à
ordenação perfeita.

**Average Precision (AP)** resume a curva precisão-revocação. É especialmente
útil quando as classes são desbalanceadas, pois penaliza listas cujo topo está
cheio de falsos positivos.

ROC-AUC/AP avaliam separação binária (`mudou` vs. `estável`), enquanto Spearman
avalia a ordenação graduada dos 37 scores de mudança. Um método pode ir bem em
uma tarefa e não na outra.

<a id="identificabilidade"></a>
## Identificabilidade e confundidores

Um problema é "identificável" se, em princípio (mesmo com dados
infinitos), é possível distinguir a explicação que nos interessa de
explicações alternativas. Uma variável de confusão (**confundidor**) é
algo que varia *junto* com a variável de interesse, de forma que um efeito
observado pode ser atribuído a qualquer uma das duas.

O projeto encontrou confundidores em várias camadas:

- **Frequência vs. mudança de sentido** (capítulo 06): o APD correlaciona
  com a frequência total da palavra (`rho=-0,436`). `chairman_nn`
  (147→683 ocorrências) tem APD alto não porque seu sentido mudou, mas
  porque é relativamente rara em D0.
- **Variação de entropia vs. mudança semântica** (capítulo 05): `pmi_cosine`
  correlacionava `rho=0,946` com a variação de entropia do corpus entre
  checkpoints — um sinal "forte", mas da variável errada (capacidade/
  calibração do modelo, não significado de uma palavra).
- **Drift de checkpoint vs. diferença de corpus** (capítulo 08): o "eixo de
  época" é, esmagadoramente, o primeiro, não o segundo.
- **A formulação mais geral** (capítulo 06): clusters não supervisionados de
  ocorrências encontram variação temporal real — mas essa variação pode
  vir de tópico, registro, sintaxe, entidades, ou composição amostral, não
  necessariamente de sentido lexical. Em notação:

  ```text
  P_t(cluster | w) != P_t(sense | w)
  ```

  E o risco mais profundo nomeado no capítulo 06: **o período é
  precisamente a variável cujo efeito queremos medir, mas ele também
  carrega todos os "nuisances" históricos** — então pode não haver, a
  partir só de corpus+período, um critério não supervisionado capaz de
  separar as duas coisas. Essa é a motivação central para a linha de
  [WSD externo](06-wsd_e_sentido_lexical.md) do capítulo 10.

Uma armadilha relacionada: **controlar por um confundidor pode eliminar o
sinal de interesse junto com o ruído**, se os dois estiverem
suficientemente entrelaçados — foi o que aconteceu com a [energy
distance](04-perfis_relacionais_e_apd.md#energy-distance) no capítulo 06
(Spearman caiu de 0,210 para 0,021 ao normalizar por dispersão interna).
Isso não significa necessariamente "o controle estava errado" — pode
significar que o sinal original **era**, em boa parte, o confundidor.

<a id="controle-aleatorio"></a>
## Controle com pesos aleatórios / pseudo-períodos

Um "controle" experimental responde à pergunta "o que eu observaria se a
coisa que estou tentando medir **não existisse**, mas tudo o mais
permanecesse igual?". Se o resultado no controle é parecido com o
resultado real, o efeito observado provavelmente não é o que se pensava.

Dois controles desse tipo aparecem no projeto:

- **Pesos aleatórios** (proposto no capítulo 05, não executado nessa fase):
  comparar `JSD(theta_0, theta_1)` (dois checkpoints treinados) com
  `JSD(random_init_1, random_init_2)` (dois modelos com pesos aleatórios,
  nunca treinados). Se os dois JSDs forem parecidos, o "efeito de
  checkpoint" não é evidência de aprendizado — é ruído de inicialização.
- **Pseudo-períodos** (capítulo 09): embaralhar os documentos de D0+D1 e
  redividir em dois conjuntos do mesmo tamanho, sem nenhuma relação com
  cronologia real, e repetir todo o protocolo de treino. Se o efeito
  observado no treino "cronológico real" também aparece no pseudo-período,
  o efeito não é específico da ordem temporal real — pode ser "qualquer
  fine-tuning faz isso" (adaptação geral ao domínio).

Um cuidado importante descoberto no capítulo 09: para que a comparação
cronológico-vs-pseudo seja válida, os dois precisam ter recebido o
**mesmo orçamento de treino** (mesmo número de passos/épocas) no
checkpoint comparado — senão, "o pseudo preservou mais sinal" pode
significar simplesmente "o pseudo recebeu menos gradiente", não "a ordem
cronológica é mais nociva".

<a id="oraculo"></a>
## Oráculo (teto de referência)

Um "oráculo" é uma versão do experimento usando um componente **melhor do
que o disponível na prática** — não para ser usado na solução final, mas
para responder "a tarefa, em princípio, é solúvel com componentes
melhores?".

No capítulo 08, o oráculo foi `bert-base-uncased`: um modelo **muito**
maior (12 camadas, `d=768`) que o encoder próprio do projeto (`d=128`, 2-3
camadas), **congelado e nunca treinado** nos dados do projeto. O resultado
— Spearman=0,594 (p=0,0001), o primeiro estatisticamente significativo do
projeto — não é uma solução utilizável (o projeto não vai simplesmente
"usar BERT-base"), mas é evidência de que:

1. a tarefa é solúvel com as frases e o gold disponíveis;
2. a distância entre o resultado atual (~0,13-0,20) e o oráculo (~0,59) é,
   em boa medida, atribuível à **capacidade do encoder**, não à formulação
   da métrica (APD) ou à pergunta de pesquisa.

Um oráculo bem escolhido transforma "não sabemos se o problema é a
fórmula ou o modelo" em "sabemos que pelo menos parte do problema é o
modelo" — uma redução de incerteza importante, mesmo sem apontar a solução
completa.

<a id="gmm-bic"></a>
## GMM e critério BIC

Um GMM (Gaussian Mixture Model, "modelo de mistura de gaussianas") modela
uma nuvem de pontos como uma combinação de `k` "blobs" gaussianos. BIC
(Bayesian Information Criterion) é um critério para escolher `k`: ele
recompensa o modelo por explicar bem os dados, mas penaliza modelos com
mais parâmetros (mais blobs) — `k=2` só "vence" `k=1` se a melhora no
ajuste compensar a complexidade extra.

No capítulo 08, um teste de bimodalidade comparou `k=1` vs `k=2` (via
`delta_bic`) para a nuvem de ocorrências de cada palavra. O resultado foi
`delta_bic` positivo (favorecendo `k=2`) para **todas** as palavras — e,
de forma reveladora, `tree_nn` (controle estável) teve o **maior**
`delta_bic` de toda a amostra. Isso não significa que `tree_nn` tem dois
sentidos; significa que, com centenas/milhares de pontos em alta dimensão,
`k=2` quase sempre "ganha" estatisticamente — o teste não estava
discriminando a propriedade de interesse (multimodalidade de *sentido*),
estava apenas confirmando que a nuvem não é perfeitamente gaussiana, o que
é quase sempre verdade. Um teste que dá "positivo" para tudo, inclusive
para o exemplo que deveria ser o controle negativo, não está medindo o que
se pretendia.

<a id="erro-padrao-spearman"></a>
## Erro padrão de Spearman

Para uma correlação de Spearman calculada sobre `n` pares, uma
aproximação útil do erro padrão é:

```text
SE(rho) ~ 1/sqrt(n - 3)
```

Com `n=37`: `SE ~ 1/sqrt(34) ~ 0,17`. Isso significa, grosso modo, que um
intervalo de confiança de 95% para um `rho` observado tem largura da ordem
de `±0,33` (aproximadamente `2 * SE`).

**Por que isso importa tanto neste projeto**: a maioria das comparações
discutidas nos capítulos 06-09 — "o método A deu 0,124 e o método B deu
0,210", "antes do treino era 0,298, depois é 0,325" — envolve diferenças
de **0,01 a 0,09**. Todas essas diferenças são muito menores que `0,17`.
Tomadas isoladamente, nenhuma dessas comparações permite concluir "o
método B é melhor" com confiança — a diferença observada é compatível com
pura variação amostral.

Essa fórmula é uma *aproximação rápida*; o
[bootstrap](#bootstrap) é a ferramenta para quantificar isso de forma mais
precisa e específica aos dados em mãos (em particular, para diferenças
*pareadas* entre duas condições sobre as mesmas 37 palavras, onde parte da
variação se cancela).

<a id="bootstrap"></a>
## Bootstrap

Bootstrap é uma técnica para estimar a incerteza de uma estatística (como
um Spearman) **sem assumir uma fórmula matemática para essa incerteza**:
em vez disso, reamostra-se os próprios dados, repetidamente, com
reposição.

Procedimento usado no capítulo 09 (`scripts/bootstrap_bert_apd_spearman.py`,
20.000 réplicas):

1. Você tem 37 pares (APD do modelo, gold do SemEval), um por palavra.
2. Sorteie, com reposição, 37 pares dessa mesma lista (alguns pares podem
   se repetir, outros podem não aparecer) — isso é "uma réplica".
3. Calcule o Spearman dessa réplica.
4. Repita 20.000 vezes. Você agora tem 20.000 valores de Spearman.
5. O intervalo de confiança de 95% é, aproximadamente, o intervalo entre o
   percentil 2,5% e o percentil 97,5% dessas 20.000 réplicas.

**Para comparar duas condições** (por exemplo, `layer_1` antes e depois do
treino), a forma correta é **pareada**: em cada réplica do bootstrap, use
**o mesmo sorteio de palavras** para calcular o Spearman de ambas as
condições, e subtraia. Isso cancela a variação que é comum às duas
condições (por exemplo, "esta réplica sorteou `plane_nn` duas vezes"
afeta ambas igualmente), deixando apenas a variação que é específica da
diferença entre elas.

No capítulo 09, esse procedimento pareado revelou que diferenças como
`full_seed1000 - init` (em `layer_1`, `0,325 - 0,298`) têm IC 95%
`[-0,081; 0,125]` — **inclui zero**. Ou seja: não há evidência, com estes
dados, de que o treino mudou `layer_1` em qualquer direção.

<a id="harking"></a>
## HARKing

HARKing significa "Hypothesizing After the Results are Known" —
formular (ou ajustar) uma hipótese **depois** de já ter visto o
resultado, mas apresentá-la como se tivesse sido decidida antes. É um dos
principais mecanismos pelos quais pesquisas com `n` pequeno produzem
conclusões que não replicam: com 37 palavras, é sempre possível encontrar
*alguma* explicação pós-hoc para *qualquer* padrão nos dados, porque há
muitas variáveis candidatas (frequência, polissemia, registro, etc.) e
poucos pontos para distingui-las.

O projeto evita HARKing através de **decisões pré-registradas** — ver
[pré-registro](#pre-registro). Um exemplo concreto de onde HARKing *quase*
aconteceu e foi evitado: no capítulo 10, o subconjunto "ferramenta" de
`plane_nn` (19 ocorrências) falhou (21,1%, abaixo do acaso). O documento
nota explicitamente que **remover esses 19 exemplos agora, depois de ver
as predições, invalidaria a Porta 1** — exatamente o tipo de ajuste que o
pré-registro existe para impedir, mesmo quando a "justificativa" (a
heurística de rotulagem tem ruído, o que é verdade) parece razoável.

<a id="pre-registro"></a>
## Pré-registro e regras de decisão conjuntivas

"Pré-registrar" um experimento significa escrever, **antes de rodá-lo**,
exatamente quais critérios serão usados para julgar o resultado — quais
subconjuntos, quais cortes de aprovação, e como combinar múltiplos
critérios numa decisão final.

Uma **regra de decisão conjuntiva** ("E", não "OU") exige que **todos** os
critérios passem para uma aprovação geral. No capítulo 10 (Porta 1), três
critérios foram pré-registrados:

```text
D0 geometria  >= 0,75   (passou: 0,984)
D0 ferramenta >  acaso  (falhou: 0,211 < 1/3)
D1 avião      >= 0,80   (passou: 1,000)
```

Apesar de dois dos três critérios passarem com folga e a acurácia macro
estar bem acima do acaso (0,731), a regra conjuntiva pré-registrada produz
`NO-GO` — porque um critério falhou. Isso pode parecer "rígido demais"
quando 4 de 5 verificações passam, mas é exatamente o ponto: **a regra
conjuntiva existe para impedir que um resultado "majoritariamente bom"
seja arredondado para "aprovado"**, escondendo a fragilidade específica
(neste caso, sentidos raros/históricos) que o critério reprovado estava
desenhado para detectar. O capítulo 10 trata esse NO-GO não como um
beco sem saída, mas como informação específica sobre *onde* investigar a
seguir — sem reabrir os critérios já decididos.
