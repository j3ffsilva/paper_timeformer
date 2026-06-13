# Capítulo 1 — O ponto de partida: condicionar o modelo pelo tempo

> Fontes originais: `docs/01-synthetic_results_current.md`,
> `docs/03-synthetic_experiment_memo.md`,
> `tmp/04-code_quality_review.md`, `tmp/06-timeformer_external_review.md`.

## A primeira ideia: ensinar o modelo "em que ano estamos"

A primeira formulação do projeto (o que os documentos chamam de "paper
2", herdeiro de um "paper 1" que já havia mostrado que diferentes formas
de injetar tempo num Transformer — Token-Time, FiLM, Memory-Augmented —
são equivalentes) partia de uma ideia direta: **se queremos um modelo que
entenda mudança ao longo do tempo, façamos o modelo receber o tempo como
entrada explícita**.

Concretamente, cada token de cada ocorrência era representado como um par

```text
token@time = (h_s(t), m_s(t))
```

onde `h_s(t)` é a posição semântica de um sujeito `s` numa ocorrência num
instante `t`, e `m_s(t)` é um estado agregado de sua "trajetória" (como
esse sujeito mudou até `t`). A arquitetura condicionava a self-attention
do Transformer por esse sinal de tempo (Token-Time, FiLM ou
Memory-Augmented — mecanismos diferentes, mas que o "paper 1" havia
mostrado serem equivalentes em efeito).

Em volta dessa arquitetura, montou-se um pipeline em duas etapas:

1. **Teacher**: treinado de forma auto-supervisionada, com uma loss de
   reconstrução mais uma "loss anti-identidade" baseada em
   [CKA](conceitos/01-correlacao_e_similaridade.md#cka) — para evitar que
   a trajetória aprendida fosse trivialmente igual à sequência de
   representações de entrada.
2. **Student**: treinado por "masked trajectory distillation" — aprende a
   reconstruir a trajetória do teacher a partir de versões mascaradas,
   numa variante bidirecional, causal ou linear.

## O corpus sintético e as quatro classes

Para testar essa arquitetura antes de qualquer corpus real, foi
construído um corpus sintético com 40 "sujeitos" (análogos a palavras),
cada um pertencente a uma de quatro classes de trajetória semântica:

| Classe | O que significa | Análogo no SemEval (capítulo 00) |
|---|---|---|
| **Stable** | A posição semântica do sujeito não muda ao longo do tempo | `tree_nn`, `chairman_nn` |
| **Drift** | A posição muda gradualmente, de forma contínua | mudança lenta de uso |
| **Bifurcating** | O sujeito passa a ter *dois* sentidos coexistindo | `graft_nn` (diversificação) |
| **Abrupt** | A posição muda de forma súbita, num único ponto no tempo | `plane_nn` (substituição de sentido) |

Cada exemplo tinha um parâmetro de **fidelidade** (`fidelity`, entre 0.50
e 0.75): a fração das ocorrências em que os marcadores de contexto
(verbo/objeto) realmente indicam a classe correta do sujeito. Fidelidade
mais baixa = contexto mais "ruidoso"/ambíguo.

## Resultado 1: FiLM + L_traj funciona melhor que apenas mudar a arquitetura

O primeiro ciclo experimental (`docs/03`, 2026-05-30) comparou quatro
modelos:

- `TokenTime` / `FiLM`: só condicionamento arquitetural pelo tempo, MLM
  puro.
- `TokenTimeTraj` / `FiLMTraj`: condicionamento + uma loss adicional de
  trajetória, `L_traj`, que força a geometria das representações ao longo
  do tempo a respeitar a trajetória plantada (via ranking).

A pergunta era: o ganho vem da arquitetura (condicionamento) ou da loss
(`L_traj`)? Resultado, em `fidelity=0.75` (31 seeds):

| Modelo | MLM | Spearman Drift | Spearman Bifurcating |
|---|---:|---:|---:|
| TokenTime | 0.1789 | 0.6069 | 0.5311 |
| FiLM | 0.2448 | 0.5552 | 0.4375 |
| TokenTimeTraj | 0.1727 | 0.8018 | 0.6914 |
| **FiLMTraj** | **0.2412** | **0.9657** | **0.8777** |

`FiLMTraj` venceu `FiLM` puro em **todos os 7 níveis de fidelidade
testados**, com ganhos grandes (+0.41 a +0.58 em Spearman Drift,
intervalos de confiança de 95% sempre positivos) e custo quase nulo em
MLM (-0.004 a +0.003). Uma ablação adicional (`StandardTraj`: só a loss,
sem condicionamento arquitetural) mostrou que a loss por si só **não**
basta — em `fidelity=0.75`, `StandardTraj` chega a Spearman Drift=0.497,
muito abaixo dos 0.966 de `FiLMTraj`. Conclusão da fase: **arquitetura e
loss são complementares**, nenhuma das duas resolve isoladamente.

Um teste de tendência (o ganho `FiLMTraj - FiLM` cresce conforme a
fidelidade piora?) deu `p=0.22` e `p=0.57` — **não confirmado**. Como
explicado em [efeito, p-valor e intervalo de
confiança](conceitos/05-estatistica_experimental.md#p-valor), esses
p-valores altos não provam que a tendência não existe; apenas que os dados
não a confirmam. A conclusão honesta registrada foi "o efeito é robusto à
degradação dos marcadores", não "o efeito cresce monotonicamente".

## Resultado 2: o "agregador" (Set Transformer) só funciona com supervisão sintética

A segunda etapa do pipeline (`docs/01`, 2026-06-01) testava se um **Set
Transformer** — um módulo que agrega várias ocorrências de um sujeito numa
representação de trajetória — conseguia, sem qualquer rótulo externo,
capturar a **bimodalidade** de sujeitos `Bifurcating` (a classe onde dois
sentidos coexistem).

A métrica era `D6`, um escore de bimodalidade baseado em
[silhouette](conceitos/04-perfis_relacionais_e_apd.md#silhouette).
Resultado:

| Configuração | D6 |
|---|---:|
| `mean:bidirectional` (baseline, sem agregador) | 0.130 |
| `set:bidirectional` (Set Transformer, **com** supervisão `true_context`) | **0.298** |
| `set:bidirectional --skip-set-training` (Set Transformer, **sem** supervisão) | 0.130 |

A leitura honesta registrada no próprio memo: **a melhora de 0.130 para
0.298 vinha inteiramente da supervisão sintética** (`true_context`, um
rótulo que só existe porque o corpus é artificial). Sem essa supervisão,
o Set Transformer treinado do zero não fazia melhor que uma simples média
(`mean pooling`). Ou seja: a arquitetura não estava, por si só,
descobrindo a bimodalidade — ela precisava ser ensinada a encontrá-la.

Esse resultado foi corretamente classificado como "sanidade
supervisionada / upper bound", não como "método principal" — mas já
sinalizava um problema que voltaria a aparecer várias vezes no projeto
(capítulos 04, 06 e 08): **sem supervisão externa, é difícil saber se uma
representação "contém" uma estrutura semântica ou se ela só parece conter
quando se already sabe o que procurar**.

Por outro lado, o teste `D5a` (reconstrução mascarada, comparando
variantes bidirectional/causal/linear do student) confirmou a hipótese
arquitetural esperada — em `Abrupt`:

| Student | Overall | Abrupt |
|---|---:|---:|
| `bidirectional` | **1.996** | **1.766** |
| `causal` | 2.197 | 2.090 |
| `linear` | 4.659 | 4.598 |

`bidirectional < causal << linear`, exatamente como previsto: olhar para
o futuro e o passado ajuda a reconstruir uma ruptura abrupta.

## A revisão externa: "prosseguir, mas pausar para corrigir"

Em 2026-06-03, duas revisões avaliaram esse pipeline:

- **Qualidade de código** (`tmp/04`): nota B+. Estrutura modular boa
  (`corpus.py → dataset.py → models.py → ... → trajectory_metrics.py`),
  mas com duplicações concretas (um bloco de cálculo de protótipos
  copiado entre `trajectory_axis_loss` e `trajectory_ranking_loss`,
  `CLASS_NAMES` definido duas vezes, o loop de treino replicado em 4
  lugares).

- **Revisão científica** (`tmp/06`): veredito "a direção conceitual está
  certa, mas pausar antes de ir para corpus real (COHA)". Apontou três
  problemas críticos:
  1. o sinal `context` usado pela loss SSL do agregador **não** era o
     `true_context` sintético, mas embeddings de verbo/objeto — o que é
     defensável, mas sua qualidade dependeria inteiramente da fidelidade
     do corpus e não estava documentado;
  2. a métrica `D2` calculava seus protótipos de referência usando *toda*
     a partição de avaliação (incluindo o ponto sendo avaliado) — uma
     forma de vazamento de avaliação;
  3. um bug de dimensão no `SetSlotsAggregator` (`R` tinha dimensão
     `2*d_model`, mas um script auxiliar assumia `d_model`), que podia
     produzir resultados silenciosamente errados.

## Por que esse caminho foi abandonado

Apesar dos resultados sintéticos serem positivos e das revisões dizerem
"prosseguir, com correções", **este caminho inteiro foi abandonado dias
depois** — não por estar errado tecnicamente, mas por uma reconsideração
mais profunda sobre *o que* deveria ser medido.

A questão de fundo (detalhada no capítulo 02) é esta: toda essa
arquitetura — Token-Time/FiLM, `L_traj`, teacher/student, Set Transformer
— treina o modelo para *aprender explicitamente sobre o tempo* (cada
token "sabe" em que período está, e a trajetória é uma saída direta do
modelo). Mas isso significa que qualquer geometria temporal observada
pode ser, em parte, **um artefato de como o modelo foi treinado para
representar o tempo**, não uma propriedade emergente do significado das
palavras.

A proposta que toma forma no capítulo 02 inverte essa lógica: treinar um
modelo **sem nenhum sinal de tempo explícito** (um Transformer comum,
treinado cronologicamente em D0 e depois em D1) e **medir a mudança
depois**, comparando como as relações entre palavras mudaram de um
checkpoint para o outro. Essa mudança de framing tornou toda a engenharia
deste capítulo — Set Transformer, `L_traj`, teacher/student, D2/D5a/D6 —
um **baseline histórico**: útil para entender de onde o projeto veio, mas
não parte do pipeline atual.

## O que vale levar deste capítulo

- A lição mais durável não é nenhum número, é metodológica: **um
  resultado "positivo" obtido com supervisão sintética
  (`true_context`/D6=0.298) pode não significar nada sobre o regime
  real**, onde esse rótulo não existe. Essa lição reaparece, em formas
  diferentes, em quase todos os capítulos seguintes.
- As quatro classes sintéticas (Stable/Drift/Bifurcating/Abrupt) e a ideia
  de "fidelidade" do contexto continuaram relevantes — elas reaparecem
  como base do experimento estrutural do capítulo 02/03, e a distinção
  Drift vs. Abrupt vs. Bifurcating ecoa diretamente nos quatro
  personagens do SemEval (capítulo 00).
- `bidirectional < causal << linear` em rupturas abruptas é um resultado
  arquitetural que permanece válido e não foi contestado depois — só
  deixou de ser o foco do projeto.

## Conceitos novos usados neste capítulo

- [CKA (Centered Kernel Alignment)](conceitos/01-correlacao_e_similaridade.md#cka)
- [Spearman](conceitos/01-correlacao_e_similaridade.md#spearman)
- [MLM (Masked Language Modeling)](conceitos/02-encoders_e_camadas.md#mlm)
- [Silhouette score](conceitos/04-perfis_relacionais_e_apd.md#silhouette)
- [Efeito, p-valor e intervalo de confiança](conceitos/05-estatistica_experimental.md#p-valor)
