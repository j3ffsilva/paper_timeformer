# Capítulo 9 — `bert-tiny`, Option D e L2-SP: quando o ruído de n=37 vira o protagonista

> Fontes originais: `tmp/31-codex_bert_tiny_continual_finetuning_second_opinion_report.md`,
> `tmp/32-codex_option_d_execution_results.md`,
> `tmp/33-codex_option_d_l2sp_update_for_claude.md`,
> `tmp/34-claude_option_d_l2sp_second_opinion.md`,
> `tmp/35-codex_followup_to_claude_option_d_second_opinion.md`.

O capítulo 08 terminou com uma recomendação concreta: inicializar o
TimeFormer a partir de um checkpoint pré-treinado (`bert-tiny`) em vez de
treinar do zero, porque o oráculo BERT-base mostrou que a tarefa é
solúvel (Spearman=0,594) enquanto o encoder próprio (mesmo com `d=128`)
ficava em torno de 0,13-0,20. Este capítulo conta o que aconteceu quando
essa recomendação foi testada — e como, no caminho, o projeto descobriu
um problema metodológico mais geral: **com `n=37` palavras-alvo, é muito
fácil confundir "achei algo" com "reordenei dois ou três pares por
acaso"**.

## Primeira tentativa: adaptar `bert-tiny` ao `RealStaticMLM`

A primeira versão (já registrada ao fim do capítulo 08, mas detalhada
aqui) transplantou pesos de `prajjwal1/bert-tiny` (2 camadas, `d=128`,
2 heads, `d_ff=512`) para a arquitetura própria do projeto
(`RealStaticMLM`), mapeando:

```text
Q/K/V do BERT       -> self_attn.in_proj
attention.output    -> self_attn.out_proj
intermediate.dense  -> linear1
output.dense        -> linear2
LayerNorms          -> norm1/norm2
```

Os 27.311 tokens do vocabulário próprio (lematizado, palavras inteiras)
foram inicializados pela **média dos embeddings dos WordPieces** do BERT
correspondentes (depois de remover sufixos como `_nn`/`_vb`). Ficaram
aleatórios: `pos_emb` e `mlm_head` — e o modelo próprio não tem embedding
LayerNorm, token-type embeddings nem weight tying entre embedding e
decoder (diferenças importantes do BERT original).

Depois de 12 épocas em D0 + 8 em D1 (mesmo protocolo do capítulo 08, LR
`1e-4` constante), o resultado foi:

| Checkpoint | Layer/readout | APD Spearman | p |
|---|---|---:|---:|
| `theta_init` (antes do treino) | layer_2 | 0,277 | 0,097 |
| `theta_init` | mean_last_2 | 0,224 | 0,183 |
| `theta_init`, melhor método (`centered_relational_apd`) | mean_last_2 | **0,337** | **0,041** |
| `theta_0` (após D0) | layer_2 | -0,055 | 0,747 |
| `theta_0` | mean_last_2 | -0,016 | 0,926 |
| `theta_1` (após D0+D1) | layer_2 | 0,123 | 0,467 |
| `theta_1` | mean_last_2 | **0,204** | 0,225 |

Como referência, o `bert-tiny` original (sem nenhuma adaptação,
congelado) deu Spearman=0,399 (p=0,014) em `mean_last_4`. A leitura
imediata foi: **`theta_init` preserva boa parte do sinal pré-treinado
(0,337, primeira vez com `p<0,05` usando o encoder próprio do projeto!),
mas o sinal colapsa já depois de D0 (-0,016) e só se recupera
parcialmente em D1 (0,204)** — de volta ao mesmo patamar do encoder
treinado do zero.

A hipótese de trabalho foi nomeada **"catastrophic forgetting durante o
fine-tuning contínuo"**, mas o próprio relatório (`tmp/31`) já listou dez
explicações alternativas não descartadas — entre elas: o `mlm_head`
aleatório força o encoder inteiro a se reorganizar desde o passo 1; não
há weight tying; faltam positional embeddings e embedding LayerNorm
copiados do BERT; a média estática de WordPieces destrói a estrutura
sublexical; `mean_last_4` do BERT original (que inclui a saída do
embedding) não é diretamente comparável a `mean_last_2` do modelo
adaptado (que não inclui); e — o ponto que se mostraria mais importante —
**não havia checkpoints intermediários, então não se sabia *quando*,
dentro de 20 épocas, o sinal caía**.

## Option D: preservar o BERT inteiro, sem adaptação

A recomendação consensual (e a preferência técnica do próprio relatório)
foi testar primeiro a "Opção D": abandonar a arquitetura/vocabulário
próprios para o encoder e usar `prajjwal1/bert-tiny` **integralmente** —
tokenizer WordPiece original, embeddings lexicais e posicionais,
embedding LayerNorm, MLM head e weight tying, tudo preservado. As
palavras-alvo passam a ser representadas pela média de seus subtokens
WordPiece no momento da avaliação.

Protocolo de treino, desta vez bem mais conservador que o capítulo 08:

```text
D0: 3 épocas    (vs. 12 antes)
D1: 2 épocas    (vs. 8 antes)
LR: 3e-5 com warmup 10% + linear decay   (vs. 1e-4 constante antes)
batch: 192, seq_len: 32
checkpoints salvos em 0,25 / 0,5 / 1 / 2(+) épocas
2 seeds: 1000 e 1001
holdout por documento, seleção de checkpoint SEM gold
```

A regra de seleção sem gold: para `theta_0`, escolher pelo menor loss de
validação em D0; para `theta_1`, pela média das losses de D0 e D1; entre
checkpoints empatados (até 1% do melhor loss), preferir o que mantém
maior cosseno com representações-âncora da inicialização.

### O baseline alinhado — e por que 0,399 não era comparável

Antes de qualquer treino, o relatório recalculou o `bert-tiny` original
com um avaliador que usa exatamente as mesmas frases, mesmo tokenizer,
mesmo `max_length=32`, e separa explicitamente cada readout:

| Readout (congelado, sem treino) | APD Spearman |
|---|---:|
| embedding | -0,018 |
| **layer 1** | **0,298** |
| layer 2 (última) | 0,136 |
| média layers 1+2 | 0,241 |
| média embedding+layers | 0,160 |

Ou seja, o `0,399` do capítulo 08 (`mean_last_4`) misturava a saída do
embedding com as duas camadas e usava outro pré-processamento — não é um
"teto" comparável. **O baseline relevante e alinhado é `layer_1 = 0,298`.**
Esse realinhamento por si só já é uma lição: o número mais impressionante
de um relatório anterior pode não sobreviver a uma comparação apples-to-apples.

### Resultado do fine-tuning cronológico (D0 -> D1)

| Readout | Seed 1000 | Seed 1001 |
|---|---:|---:|
| embedding | 0,038 | 0,032 |
| **layer 1** | **0,325** | **0,322** |
| layer 2 | 0,030 | 0,038 |
| média layers 1+2 | 0,189 | 0,194 |

Primeira leitura: `layer_1` (já a melhor no baseline) sobe levemente, de
0,298 para ~0,32; `layer_2` (já a pior, 0,136) cai para perto de zero. Na
seed 1000, essa reorganização **já estava presente em `theta_0`**
(`layer_1=0,324`, `layer_2=-0,019`) — ou seja, acontece durante D0, não
é um efeito acumulado de D0+D1.

### Controles: pseudo-períodos e camada congelada

Para saber se essa reorganização é específica da ordem cronológica real
(1810-1860 -> 1960-2010) ou apenas "qualquer fine-tuning faz isso", os
documentos de D0+D1 foram embaralhados e redivididos em dois conjuntos do
mesmo tamanho ("pseudo-períodos"), e o mesmo protocolo de treino foi
repetido:

| Readout | Cronológico (seed 1000) | Pseudo-período |
|---|---:|---:|
| layer 1 | 0,325 | **0,332** |
| layer 2 | 0,030 | **0,153** |
| média layers | 0,189 | 0,270 |

`layer_1` sobe de forma parecida em ambas — sugere "adaptação geral ao
domínio/corpus", não algo específico da cronologia. Mas `layer_2` fica
muito melhor no pseudo-controle (0,153) do que no cronológico (0,030) —
*à primeira vista*, sugerindo que a ordem cronológica real prejudica
especificamente `layer_2`.

Uma terceira variante — congelar embeddings + `layer_1`, treinando só
`layer_2` + MLM head — deu `layer_1=0,298` (idêntico ao baseline, como
esperado) e `layer_2=0,017` (ainda ruim). Conclusão parcial: congelar a
parte inferior não protege a camada superior, e impede a pequena melhora
de `layer_1` vista no fine-tuning completo.

### Controles lexicais pareados por frequência

Um teste adicional, importante para o capítulo 06 (lembra do problema
"o APD mede frequência, não mudança de sentido"?): foram selecionadas 37
palavras **fora** do benchmark, com frequências pareadas às 37
palavras-alvo em D0 e D1.

| Condição | Grupo | layer 1 | layer 2 |
|---|---|---:|---:|
| cronológico | alvos (37 do SemEval) | 0,318 | 0,339 |
| cronológico | controles (37 pareados por freq.) | 0,304 | 0,343 |
| pseudo | alvos | 0,323 | 0,383 |
| pseudo | controles | 0,309 | 0,372 |

A magnitude **absoluta** do APD é praticamente igual entre alvos e
controles em todas as condições. Isso confirma, de outro ângulo, o que o
capítulo 06 já havia mostrado: **o sinal do SemEval não vem de "as
palavras que mudaram têm APD alto"** — vem inteiramente do *ranking
relativo* entre as 37 palavras, e essas diferenças de ranking são de
`0,01` a `0,04` em APD absoluto. Uma margem muito estreita para qualquer
conclusão sobre camadas individuais ser robusta.

## L2-SP: restringir os pesos não restringe o ranking

Com a hipótese "a ordem cronológica prejudica especificamente `layer_2`"
ainda em aberto, testou-se uma regularização clássica de continual
learning, **L2-SP** (L2 Starting Point — penaliza o afastamento dos pesos
em relação ao checkpoint inicial), aplicada só na camada 2, combinada com
LR discriminativa (camadas inferiores com LR menor que a camada
superior):

```text
embeddings + layer 1: LR 1e-5
layer 2 + MLM head:   LR 3e-5
loss = loss_MLM + lambda * ||theta_layer2 - theta_init||^2 / ||theta_init||^2
lambda = 10
```

| Condição | layer 1 | layer 2 | média layers |
|---|---:|---:|---:|
| full anterior (LR única) | 0,325 | 0,030 | 0,189 |
| LR discriminativa, `lambda=0` | **0,340** | 0,012 | 0,196 |
| LR discriminativa + L2-SP (`lambda=10`) | 0,338 | 0,014 | 0,204 |
| L2-SP, pseudo-período | 0,341 | 0,116 | 0,258 |

No checkpoint cronológico selecionado, o L2-SP de fato **restringiu os
pesos**: a distância L2 relativa de `layer_2` ao `theta_init` caiu de
0,0416 (`lambda=0`) para 0,0294 (`lambda=10`). Mas o Spearman de `layer_2`
ficou praticamente igual (0,012 -> 0,014). **Proximidade paramétrica não
é o mesmo que preservação funcional** — manter os pesos perto de
`theta_init` não preserva o que esses pesos *fazem* com o ranking
semântico.

## A virada: bootstrap revela que quase tudo é ruído

Antes de seguir para "distillation funcional" (o próximo passo natural
proposto em `tmp/33`), a segunda opinião de `tmp/34` fez uma pergunta
simples e devastadora: **com `n=37`, qual é o erro padrão de um
Spearman?**

```text
SE(rho) ~ 1/sqrt(n-3) ~ 1/sqrt(34) ~ 0,17
```

Quase todas as diferenças discutidas até aqui — `layer_1`: `0,298 ->
0,322-0,341`; `layer_2`: `0,136 -> 0,012-0,153` — são **menores que esse
erro padrão**. A "queda dramática" de `layer_2` para `-0,019` já em
`theta_0` pode ser, literalmente, **a reordenação de dois ou três pares de
palavras com APD muito parecido**. E há um segundo problema, um confound
de orçamento: o checkpoint pseudo-período selecionado para `theta_1` tinha
visto só **0,5 época** de "D1", contra **2 épocas completas** do
cronológico — então "pseudo é melhor em `layer_2`" podia significar
simplesmente "pseudo recebeu 4x menos gradiente", não "pseudo não tem o
efeito nocivo da ordem cronológica".

A resposta (`tmp/35`) foi rodar dois controles baratos, sem treinar nada
de novo — reaproveitando os checkpoints já salvos.

### Bootstrap pareado por palavra (20.000 réplicas)

**Layer 1**:

| Condição | Spearman | IC 95% |
|---|---:|---:|
| init (congelado) | 0,298 | [-0,018; 0,561] |
| full seed 1000 | 0,325 | [-0,011; 0,607] |
| full seed 1001 | 0,322 | [-0,016; 0,603] |
| pseudo seed 1000 | 0,332 | [-0,007; 0,616] |
| LR discriminativa | 0,340 | [0,014; 0,611] |
| LR + L2-SP cronológico | 0,338 | [0,011; 0,609] |
| LR + L2-SP pseudo | 0,341 | [0,014; 0,608] |

A diferença `full_seed1000 - init` tem IC `[-0,081; 0,125]` — **inclui
zero**. Todas as outras diferenças contra `init` também incluem zero.

**Layer 2**:

| Condição | Spearman | IC 95% |
|---|---:|---:|
| init (congelado) | 0,136 | [-0,197; 0,453] |
| full seed 1000 | 0,030 | [-0,285; 0,334] |
| full seed 1001 | 0,038 | [-0,274; 0,343] |
| pseudo seed 1000 | 0,153 | [-0,171; 0,444] |
| LR discriminativa | 0,012 | [-0,300; 0,316] |
| LR + L2-SP cronológico | 0,014 | [-0,298; 0,318] |
| LR + L2-SP pseudo | 0,116 | [-0,203; 0,413] |

A diferença `full_seed1000 - init`, que era a "queda dramática" original,
tem IC `[-0,302; 0,084]` — **também inclui zero**.

### Trajetória de D1 com orçamento alinhado

Reavaliando `layer_2` em **todos** os checkpoints salvos de D1 (0,25,
0,5, 1, 2 épocas), nas duas condições, com a mesma amostra de ocorrências:

| Épocas em D1 | Cronológico | Pseudo | IC 95% (crono - pseudo) |
|---:|---:|---:|---:|
| 0,25 | 0,012 | 0,046 | [-0,132; 0,062] |
| 0,5 | 0,062 | 0,153 | [-0,238; 0,048] |
| 1 | 0,059 | 0,176 | [-0,278; 0,031] |
| 2 | 0,030 | 0,088 | [-0,170; 0,044] |

O pseudo fica numericamente acima em **todos** os marcos com orçamento
igual — então o confound de orçamento não explica tudo. Mas **nenhum**
desses quatro intervalos exclui zero.

## A decisão: não perseguir mais a layer 2, não fazer distillation

A síntese final (`tmp/35`) retira a conclusão "a ordem cronológica causa
perda adicional específica na layer 2" e mantém apenas o que sobrevive ao
bootstrap:

- `layer_1` é o melhor readout entre os avaliados (mas sem diferença
  estabelecida entre congelado e treinado);
- `layer_2` é fraca e instável tanto antes quanto depois do treino — em
  `bert-tiny` (2 camadas), a "última camada" é também "a camada mais
  próxima da saída MLM", sem o espaço de abstração hierárquica que uma
  "última camada" teria num BERT-base/large;
- L2-SP restringe drift de pesos sem produzir melhora observável;
- a APD absoluta não separa alvos de controles pareados por frequência em
  nenhuma condição;
- **os dados atuais não permitem afirmar redistribuição causal entre
  camadas** — é um padrão descritivo nas estimativas pontuais, não um
  efeito estabelecido.

A distillation funcional proposta em `tmp/33` foi **descartada antes de
ser implementada** — ela tentaria preservar uma camada que (1) já era
inferior no baseline congelado, (2) não mostrou diferença
estatisticamente estabelecida, (3) não é necessária como régua porque
`layer_1` já cumpre esse papel, e (4) não resolveria, de qualquer forma, a
identificabilidade entre contexto e sentido discutida no capítulo 06.

## O que isso significa para "gargalo = encoder" (capítulo 08)

Este capítulo não invalida o achado central do capítulo 08 — o oráculo
BERT-base congelado continua dando Spearman=0,594 (p=0,0001), muito acima
de qualquer coisa medida aqui. O que muda é a **interpretação do caminho
entre 0,13 e 0,59**:

- a versão "fine-tuning contínuo causa esquecimento catastrófico
  generalizado" (a leitura inicial deste capítulo, baseada na adaptação
  parcial de `bert-tiny`) **não se sustenta** quando se preserva o BERT
  integralmente — não há colapso global (cosseno com a inicialização
  permanece ~0,95 em todas as condições);
- mas a versão otimista — "basta inicializar com BERT, treinar com
  cuidado, e o sinal de 0,59 vai aparecer no encoder próprio" — **também
  não se sustenta**: mesmo no melhor caso (`layer_1`, LR discriminativa),
  o Spearman fica em ~0,34, com IC que inclui zero, longe de 0,59;
- a explicação mais provável para a distância restante não é mais
  "esquecimento durante o fine-tuning" — é que **`bert-tiny` (2 camadas,
  `d=128`) simplesmente não tem a mesma capacidade representacional que
  `bert-base` (12 camadas, `d=768`)**, independentemente de como o
  fine-tuning é conduzido. O ganho de 0,13 (encoder próprio do zero) para
  ~0,30-0,34 (`bert-tiny`, com ou sem fine-tuning) é real e considerável;
  o ganho restante até 0,59 provavelmente exige mais capacidade do que
  `bert-tiny` oferece.

## O que vale levar deste capítulo

- **A descoberta mais importante deste capítulo não é sobre o encoder — é
  sobre o tamanho da amostra.** Com `n=37`, `SE(Spearman) ≈ 0,17`: a
  maioria das comparações "ganhou 0,02" ou "perdeu 0,03" que apareceram
  nos capítulos 06-09 estão dentro dessa margem. Daqui em diante, qualquer
  comparação de Spearman entre condições deveria vir acompanhada de
  bootstrap pareado por palavra — não como formalidade, mas porque, como
  visto aqui, ele pode reverter inteiramente a narrativa (de "esquecimento
  catastrófico" para "ruído de seleção de checkpoint").
- **Controles importam mais que a métrica principal.** Os dois controles
  mais informativos deste capítulo — palavras pareadas por frequência (que
  mostraram que alvos e controles têm APD absoluto igual) e pseudo-períodos
  com orçamento alinhado (que mostraram que o "efeito cronológico" não
  sobrevive ao bootstrap) — não mediram diretamente "mudança semântica":
  mediram se as outras medições eram interpretáveis. Em projetos com `n`
  pequeno, controles desse tipo costumam valer mais que mais uma rodada de
  treino.
- **Realinhar comparações entre relatórios é um trabalho contínuo**: o
  `0,399` do capítulo 08 (`mean_last_4` do BERT original) não sobreviveu a
  uma comparação apples-to-apples — o valor relevante e comparável é
  `layer_1=0,298`. Isso não muda a conclusão "gargalo = encoder" do
  capítulo 08 (BERT-base ainda dá 0,594), mas muda a régua contra a qual
  `bert-tiny` deve ser comparado.
- O capítulo termina com uma decisão explícita de **não** seguir engenharia
  de regularização do encoder (L2-SP, distillation, congelamento) além
  deste ponto, e de redirecionar o esforço para uma régua **externa**
  (WSD contexto-gloss, sem ajuste no SemEval) — preparando o terreno do
  capítulo 10, que já nasce com um conjunto de critérios pré-registrados
  para `plane_nn` (geometria D0 >= 0,75; aeronave D1 >= 0,80; ferramenta D0
  substancialmente acima do acaso).

## Conceitos novos usados neste capítulo

- [Bootstrap e intervalos de confiança para Spearman](conceitos/05-estatistica_experimental.md#bootstrap)
- [Erro padrão de uma correlação de Spearman em função de n](conceitos/05-estatistica_experimental.md#erro-padrao-spearman)
- [Pseudo-períodos e controles de orçamento de treino](conceitos/05-estatistica_experimental.md#controle-aleatorio)
- [L2-SP e regularização de continual learning](conceitos/03-deriva_e_esquecimento.md#l2-sp)
- [LR discriminativa e congelamento de camadas](conceitos/02-encoders_e_camadas.md#lr-discriminativa)
- [Fine-tuning, warmup e scheduler](conceitos/02-encoders_e_camadas.md#fine-tuning)
- [Token inteiro vs. WordPiece e embeddings de entrada](conceitos/09-dados_tokenizacao_e_contexto.md)
- [Pseudo-períodos e comparações com orçamento alinhado](conceitos/08-desenhos_temporais_e_reguas.md#pseudo-periodos)
- [Controles pareados por frequência](conceitos/04-perfis_relacionais_e_apd.md#controles-pareados)
