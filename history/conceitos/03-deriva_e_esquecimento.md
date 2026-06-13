# Conceitos 3 — Deriva e esquecimento

Este arquivo trata de um tema que aparece, sob formas diferentes, em quase
todos os capítulos: **o que acontece com um modelo quando ele continua
sendo treinado**, e como distinguir "o modelo aprendeu algo novo sobre o
mundo" de "o modelo simplesmente mudou de uma forma que não tem a ver com
o fenômeno que queremos medir".

## Deriva (drift) entre checkpoints

"Deriva" é qualquer mudança nos pesos/representações de um modelo entre
dois pontos do treino, **independentemente de essa mudança refletir
algo sobre os dados**. Todo treino contínuo produz deriva — a pergunta
interessante é sempre "deriva em relação a quê, e o que ela significa?".

O projeto encontrou deriva em vários níveis:

- **Deriva de domínio/calibração** (capítulo 04-05): a distribuição geral
  de previsões `p_t` do modelo muda entre `theta_0` e `theta_1` por
  motivos que não têm relação com o significado de uma palavra específica
  — por exemplo, porque o vocabulário ou a composição lexical de D1 é
  diferente de D0. O log-PMI (capítulo 04) foi desenhado especificamente
  para cancelar esse tipo de deriva: ele compara `q_t(w)` com `p_t` *do
  mesmo checkpoint*, então uma deriva que afeta `q_t(w)` e `p_t` por igual
  se cancela na razão.
- **"Eixo de época"** (capítulo 08): a forma mais dramática de deriva
  encontrada no projeto. Um corte simples (k-means com 2 clusters) na
  nuvem de ocorrências (D0+D1 combinadas) de **quase qualquer palavra**
  recupera o período de origem quase perfeitamente (NMI ~0,8-1,0) — não
  porque os dois corpora sejam tão diferentes assim, mas porque
  `theta_0` e `theta_1` são, eles mesmos, dois pontos muito diferentes no
  espaço de representações. O experimento decisivo (a "grade 2x2", capítulo
  08) isolou essa deriva do efeito do corpus: com **encoder fixo**, o
  corpus por si só dá NMI~0,01-0,03 (quase nada); com **dados fixos**, o
  checkpoint por si só dá NMI~0,85-1,00 (quase tudo). Daí a recomendação
  de [encoder fixo](02-encoders_e_camadas.md#encoder-fixo).
- **Recentralização aditiva** (capítulo 08, §7.15): subtrair a média global
  de cada "lado" (cada checkpoint/corpus) das representações reduz o NMI
  do eixo de época de ~0,98 para ~0,05-0,09 — mas **não melhora** o
  Spearman do APD. Conclusão: "remover a deriva visível" e "destravar o
  sinal semântico" são dois problemas diferentes — a deriva de época não
  era, ela mesma, o que estava *escondendo* o sinal.

<a id="segundo-momento"></a>
## Por que o treino deforma o espaço de forma desigual (Adam/AdamW)

O [cosseno](01-correlacao_e_similaridade.md#cosseno) entre representações é
invariante a rotação, reflexão e escalamento **uniforme** — mas não a
escalamento **anisotrópico** (cada dimensão crescendo numa taxa diferente).
A razão pela qual essa diferença importa na prática é o otimizador.

Adam/AdamW não dão o mesmo passo de atualização para todos os parâmetros: a
cada passo, cada parâmetro é dividido por uma estimativa móvel da raiz do
**segundo momento** dos seus gradientes (a média móvel de
`gradiente^2`), o que faz parâmetros com gradientes historicamente grandes
receberem passos menores, e vice-versa:

```text
passo(parametro) ∝ lr * gradiente / sqrt(segundo_momento(parametro))
```

Como cada dimensão do espaço de representações acumula seu próprio segundo
momento, dimensões diferentes evoluem em taxas diferentes ao longo do
treino — exatamente o escalamento anisotrópico ao qual o cosseno **não** é
invariante. Por isso, "o cosseno entre `theta_0` e `theta_1` deveria ser
preservado porque o treino só gira/reescala o espaço" é uma afirmação que
**precisa ser medida** (por exemplo, via [CKA](01-correlacao_e_similaridade.md#cka)
entre checkpoints), não assumida — é o "Problema 1" do capítulo 03.

## Esquecimento catastrófico (catastrophic forgetting)

"Esquecimento catastrófico" é o fenômeno, bem documentado em redes neurais
treinadas sequencialmente em tarefas/dados diferentes, em que **continuar
o treino num novo conjunto de dados degrada o desempenho no conjunto de
dados anterior** — o modelo "esquece" o que sabia antes.

No capítulo 05, há uma primeira pista: a *loss* de D1 (`5,4283` no início
do treino contínuo) é **maior** que a loss final de D0 (`4,7883`) — ou
seja, o checkpoint `theta_0`, ao ser exposto pela primeira vez a D1, está
pior em D1 do que estava em D0 ao final de seu próprio treino. Isso é
esperado (D1 é um domínio "novo"), mas é também o tipo de sintoma que
torna plausível a hipótese de esquecimento.

No capítulo 09, a hipótese de esquecimento catastrófico foi testada
diretamente: a primeira tentativa de inicializar com `bert-tiny` (adaptado
ao vocabulário próprio) mostrou um sinal forte em `theta_init`
(Spearman=0,337, `p=0,041` — o primeiro resultado significativo com o
encoder próprio!) que **colapsava** já em `theta_0` (Spearman=-0,016) e se
recuperava só parcialmente em `theta_1` (0,204). Essa trajetória —
sinal alto, queda, recuperação parcial — é exatamente o formato esperado
de esquecimento catastrófico.

**Mas** quando o experimento foi repetido preservando o BERT
**integralmente** (Option D, capítulo 09), essa narrativa não se sustentou:
não houve colapso global (cosseno com a inicialização permanece ~0,95), e
as diferenças entre `theta_init` e os checkpoints treinados ficaram dentro
do [intervalo de bootstrap](05-estatistica_experimental.md#bootstrap), ou
seja, indistinguíveis de zero. A conclusão revisada foi que a primeira
versão sofria principalmente de **incompatibilidade arquitetural** (MLM
head aleatório, falta de weight tying, embeddings incompletos) — um
problema de engenharia que *parecia* esquecimento catastrófico, mas era
outra coisa.

**Lição geral**: "o sinal caiu depois do treino" tem pelo menos três
explicações possíveis, que precisam ser desenredadas uma a uma: (1)
esquecimento catastrófico genuíno; (2) incompatibilidade
arquitetural/inicialização ruim, que produz colapso por razões não
relacionadas ao *conteúdo* do treino; (3) o "sinal" original nunca era
robusto o suficiente para sobreviver a qualquer perturbação — ver
[erro padrão de Spearman](05-estatistica_experimental.md#erro-padrao-spearman).

<a id="l2-sp"></a>
## L2-SP

L2-SP ("L2 Starting Point") é uma técnica de regularização para continual
learning: além da loss principal (MLM), adiciona-se um termo que penaliza
o quanto os pesos de uma parte do modelo **se afastaram do checkpoint
inicial** (`theta_init`):

```text
loss = loss_MLM + lambda * ||theta_atual - theta_init||^2 / ||theta_init||^2
```

A ideia é "permitir que o modelo aprenda, mas sem se afastar demais de onde
começou" — uma forma suave de continual learning, menos drástica que
congelar completamente uma camada.

No capítulo 09, L2-SP foi aplicado só na `layer_2` de `bert-tiny`, com
`lambda=10`. O resultado foi revelador, mas não da forma esperada: o L2-SP
**funcionou tecnicamente** — a distância L2 relativa da `layer_2` ao
`theta_init` caiu de `0,0416` (sem L2-SP) para `0,0294` (com L2-SP), ou
seja, os pesos ficaram mais próximos da inicialização, como pretendido.
Mas o Spearman da `layer_2` permaneceu praticamente igual (`0,012` ->
`0,014`).

Esse resultado deu nome a um princípio que se tornou central no capítulo
09:

> **Proximidade paramétrica não é o mesmo que preservação funcional.**

Manter os *pesos* perto de `theta_init` não garante que a *função*
computada por essa camada (o mapeamento de entradas para representações,
e portanto o ranking semântico que essas representações produzem) também
fique perto da função original. Espaços de pesos de alta dimensão
permitem mudanças funcionais relevantes mesmo com pequenos deslocamentos
de norma — e foi exatamente esse tipo de mudança "pequena em norma, mas
funcionalmente relevante" que o L2-SP não conseguiu evitar (ou que,
alternativamente, nunca existiu de forma robusta — ver novamente
[erro padrão de Spearman](05-estatistica_experimental.md#erro-padrao-spearman)).

A proposta seguinte, **distillation funcional** (preservar a *saída* da
camada em frases-âncora, em vez dos pesos), foi desenhada precisamente
para testar essa distinção — mas foi descartada antes de ser executada,
porque o bootstrap (capítulo 09) mostrou que não havia, com confiança
estatística, nada de concreto para "resgatar" na `layer_2`.
