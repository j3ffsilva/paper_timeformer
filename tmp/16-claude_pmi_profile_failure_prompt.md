# Prompt para 2a opinião independente: por que o perfil log-PMI não recupera mudança semântica?

Você deve realizar uma revisão técnica e científica independente do projeto:

`/home/jeff/Documentos/trabalhos/papers/paper_timeformer`

Não altere código, documentação, dados, checkpoints ou resultados. Sua tarefa é
somente ler os arquivos indicados, auditar a implementação e escrever um
parecer crítico e autocontido.

Escreva obrigatoriamente o parecer final em:

`./tmp/17-timeformer_pmi_profile_failure_review.md`

Não assuma que nossa hipótese diagnóstica está correta. Procure ativamente
explicações alternativas, erros de implementação, inconsistências matemáticas e
experimentos capazes de distinguir as hipóteses.

---

## 1. Pergunta central

Estamos tentando medir mudança semântica temporal no SemEval-2020 Task 1 com um
Transformer MLM treinado continuamente em ordem cronológica.

O protocolo atual produz deslocamento numérico entre períodos, mas quase nenhum
alinhamento com o gold. Mesmo após aumentar muito o número de épocas, os
marcadores de associação de palavras concretas continuam semanticamente
incoerentes.

Queremos uma segunda opinião sobre:

> Qual é a causa mais provável da ausência de sinal semântico, e qual é a
> sequência mínima de experimentos que pode identificá-la de forma causal?

---

## 2. Protocolo conceitual

O modelo não recebe período como entrada. O treinamento é contínuo:

```text
theta_0 = treino(D_0)
theta_1 = continua_treino(theta_0, D_1)
```

Para uma palavra-alvo `w` no período `t`, usamos suas ocorrências reais no
corpus. Em cada ocorrência, substituímos `w` por `[MASK]` e extraímos a
distribuição do MLM head:

```text
q_t(w) = média_{c em C_t(w)} P_theta_t(. | c com w mascarada)
```

O prior do checkpoint é estimado por:

```text
p_t = P_theta_t(. | [CLS] [MASK] [SEP])
```

O perfil relacional é calculado sobre o vocabulário inteiro:

```text
R_t(w)[v] = log((q_t(w)[v] + eps) / (p_t[v] + eps))
```

Os scores atuais são somente:

```text
pmi_cosine(w) = 1 - cos(R_t0(w), R_t1(w))

PPMI_t(w) = max(0, R_t(w))
pi_t(w) = PPMI_t(w) / ||PPMI_t(w)||_1
ppmi_jsd(w) = JSD(pi_t0(w), pi_t1(w))
```

Leia a definição canônica:

`docs/09-relational_profile_formalization.md`

---

## 3. Limpeza recente do protocolo

O runner antes misturava o protocolo log-PMI com medidas legadas baseadas em
listas de âncoras:

- `direct_jsd`;
- `mean_abs_delta`;
- `max_abs_delta`;
- matriz target por target/anchor;
- `top_anchors.csv`.

Isso foi removido do caminho `--profile-mode pmi`.

No modo PMI atual:

- não há lista de âncoras: `anchors=0`;
- `q_t(w)` é calculado sobre todo o vocabulário;
- o CSV contém apenas `pmi_cosine` e `ppmi_jsd`;
- os avaliadores SemEval só aceitam essas duas métricas.

As mudanças ainda estão locais e não commitadas. Leia o estado atual dos
arquivos, não apenas o histórico Git:

- `scripts/run_diachronic_relational_experiment.py`
- `scripts/evaluate_semeval2020_relational.py`
- `scripts/diagnose_semeval2020_relational.py`
- `src/timeformers/relational.py`
- `src/timeformers/real_corpus.py`
- `src/timeformers/real_models.py`

---

## 4. Dataset

SemEval-2020 Task 1, inglês lematizado:

```text
data/processed/semeval2020_task1/eng_lemma/
```

Períodos:

```text
1810-1860: 409.401 janelas MLM
1960-2010: 420.743 janelas MLM
```

Vocabulário compartilhado:

```text
27.311 tokens
```

Alvos:

```text
37 palavras com gold graduado e binário
```

Os corpora e contextos reais estão em:

- `data/processed/semeval2020_task1/eng_lemma/corpus/1810-1860.txt`
- `data/processed/semeval2020_task1/eng_lemma/corpus/1960-2010.txt`
- `data/processed/semeval2020_task1/eng_lemma/truth.tsv`
- `data/processed/semeval2020_task1/eng_lemma/metadata.json`

---

## 5. Rodada curta

Configuração principal:

```text
d_model=96
layers=2
heads=4
d_ff=192
batch_size=256
learning_rate=1e-4
base_epochs=3
epochs_per_period=2
probe_mode=occurrence
profile_mode=pmi
```

Resultados:

```text
pmi_cosine:
  Spearman graded = 0.00545
  ROC-AUC binary = 0.52976
  Average precision = 0.45918

ppmi_jsd:
  Spearman graded = -0.05703
  ROC-AUC binary = 0.48214
  Average precision = 0.45401
```

Loss:

```text
t0: 7.2719 -> 6.1366 -> 5.8025
t1: 6.2094 -> 5.8854
```

Resultados e checkpoints:

`outputs/semeval2020_pmi_pilot/`

Arquivos importantes:

- `outputs/semeval2020_pmi_pilot/config.json`
- `outputs/semeval2020_pmi_pilot/continual_real/continual_history.json`
- `outputs/semeval2020_pmi_pilot/diachronic_relational_changes.csv`
- `outputs/semeval2020_pmi_pilot/profiles/prediction_anchor_js/t00.pt`
- `outputs/semeval2020_pmi_pilot/profiles/prediction_anchor_js/t01.pt`
- `outputs/semeval2020_pmi_pilot/diagnostics_pmi/summary.json`

---

## 6. Rodada longa para eliminar a hipótese de poucas épocas

Mantivemos corpus, arquitetura, seed, learning rate, batch, probe e métricas.
Alteramos somente:

```text
base_epochs: 3 -> 12
epochs_per_period: 2 -> 8
```

Foram 32.352 gradient steps no total. O treino levou aproximadamente 2h06 em
uma RTX 3060.

Loss:

```text
t0:
7.2086, 6.1155, 5.8147, 5.6099, 5.4495, 5.3201,
5.2081, 5.1113, 5.0200, 4.9342, 4.8543, 4.7777

t1:
6.0168, 5.6277, 5.4518, 5.3103,
5.1881, 5.0777, 4.9745, 4.8788
```

Resultados:

```text
pmi_cosine:
  Spearman graded = 0.11406
  p = 0.50148
  ROC-AUC binary = 0.55952
  Average precision = 0.61857

ppmi_jsd:
  Spearman graded = 0.09177
  p = 0.58908
  ROC-AUC binary = 0.53571
  Average precision = 0.56797
```

Houve melhora modesta, mas nenhum sinal graduado convincente ou
estatisticamente significativo.

Resultados e checkpoints:

`outputs/semeval2020_pmi_long_epochs_12_8/`

Leia especialmente:

- `outputs/semeval2020_pmi_long_epochs_12_8/config.json`
- `outputs/semeval2020_pmi_long_epochs_12_8/continual_real/continual_history.json`
- `outputs/semeval2020_pmi_long_epochs_12_8/diachronic_relational_changes.csv`
- `outputs/semeval2020_pmi_long_epochs_12_8/profiles/prediction_anchor_js/t00.pt`
- `outputs/semeval2020_pmi_long_epochs_12_8/profiles/prediction_anchor_js/t01.pt`
- `outputs/semeval2020_pmi_long_epochs_12_8/eval_pmi_cosine/metrics.json`
- `outputs/semeval2020_pmi_long_epochs_12_8/eval_ppmi_jsd/metrics.json`
- `outputs/semeval2020_pmi_long_epochs_12_8/diagnostics_pmi/summary.json`

---

## 7. Exemplo concreto: `graft_nn`

O corpus contém uma mudança real e facilmente interpretável.

Em `1810-1860`, os contextos são majoritariamente botânicos:

```text
graft, stock, bark, sap, scion, bud, tree
```

Em `1960-2010`, aparecem sentidos de corrupção e medicina:

```text
government graft, graft scandal, corruption, bribe,
bone marrow graft, skin graft, surgical graft
```

Na rodada longa, `graft_nn` tornou-se o primeiro colocado por `pmi_cosine`:

```text
pmi_cosine = 0.45221
ppmi_jsd = 0.26588
```

Porém, as maiores dimensões PMI continuam semanticamente incoerentes.

Top PMI em t0:

```text
steppe, reprobate, pus, baptist, ruff, pliable, plurality, coucy,
polo, vestige, petal, cove, dutchman...
```

Top PMI em t1:

```text
assiduously, rid, spite, despicable, ostentation, traceable,
explanatory, ennoble, sweetly, misunderstand...
```

Os maiores ganhos/perdas também são dominados por tokens raros aparentemente
aleatórios, e não por `corruption`, `bribe`, `skin`, `marrow`, `stock`,
`bark`, `bud` ou `scion`.

Isso sugere que o ranking numericamente reconhece algum deslocamento, mas o
perfil que o sustenta não é semanticamente confiável.

---

## 8. Diagnóstico de entropia

Na rodada longa, para `pmi_cosine`:

```text
score vs gold graduado:
rho = 0.1141

score vs frequência em t0:
rho = 0.3899

score vs soma das frequências:
rho = 0.3578

score vs entropia PPMI em t0:
rho = -0.9388

score vs variação absoluta da entropia PPMI:
rho = 0.9443
```

Estatísticas médias dos perfis:

```text
rodada curta:
  t0: R mean=0.759, R std=0.559, q entropy=7.934,
      PPMI entropy=9.959, positive mass=21166
  t1: R mean=-0.262, R std=0.687, q entropy=7.818,
      PPMI entropy=8.849, positive mass=4533

rodada longa:
  t0: R mean=1.822, R std=1.875, q entropy=7.709,
      PPMI entropy=9.683, positive mass=51351
  t1: R mean=1.557, R std=1.229, q entropy=7.620,
      PPMI entropy=9.941, positive mass=44092
```

O score parece refletir fortemente a forma/entropia do perfil PPMI, em vez do
conteúdo semântico das associações.

---

## 9. Nossa hipótese atual

Nossa hipótese principal, ainda não confirmada, é:

> O probe neutro `[CLS] [MASK] [SEP]` não é uma boa estimativa de `p_t` para
> este MLM, porque o modelo foi treinado mascarando posições em janelas naturais
> longas. Dividir `q_t(w)` por esse prior fora de distribuição amplifica
> probabilidades minúsculas e produz PMI enorme para tokens raros e aleatórios.

Isso explicaria:

- top-PMI dominado por tokens raros incoerentes;
- sensibilidade extrema à entropia;
- mudança global de escala entre checkpoints;
- melhora pequena com mais épocas, sem interpretabilidade semântica;
- ausência de correlação consistente com o gold.

Alternativas candidatas para `p_t`:

1. Média das distribuições MLM sobre máscaras em contextos naturais aleatórios
   de cada período.
2. Média de `q_t(w)` sobre um conjunto amplo de palavras/ocorrências.
3. Frequência empírica do corpus, com smoothing.
4. Prior único compartilhado entre checkpoints.
5. Estimar `p_t(v)` como marginal conjunta sobre posições mascaradas naturais,
   e não por uma única sequência artificial.

Não aceite essa hipótese sem auditoria. Considere também:

- o MLM pode não ter capacidade ou objetivo suficiente para aprender semântica;
- o treinamento do zero pode ser inadequado;
- a arquitetura ou masking pode conter erro;
- `q_t(w)` pode medir substituibilidade sintática mais que semântica;
- a média sobre ocorrências pode apagar mudança de sentidos multimodais;
- log-PMI em toda a cauda do vocabulário pode ser estatisticamente instável;
- cosseno sobre componentes PMI negativas pode ser inadequado;
- PPMI-JSD pode ser dominado por milhares de dimensões frágeis;
- comparação de checkpoints continuamente treinados pode misturar forgetting,
  adaptação global e mudança lexical;
- o corpus lematizado/POS e o vocabulário podem estar introduzindo ruído;
- o modelo pode não predizer o token-alvo real com qualidade suficiente;
- pode haver bug em `epoch_idx`, masking, checkpoint loading, normalização,
  agregação de ocorrências ou extração na posição mascarada.

---

## 10. Perguntas obrigatórias

O parecer deve responder, de maneira direta e tecnicamente fundamentada:

1. A implementação atual corresponde à formalização matemática declarada?
   Liste qualquer divergência, bug ou ambiguidade.

2. O uso de:

   ```text
   p_t = P(. | [CLS] [MASK] [SEP])
   ```

   é defensável neste modelo? Ele é a causa mais provável do comportamento
   observado?

3. A interpretação de `q_t(w)` como perfil semântico é válida? O MLM head
   prediz substitutos semânticos, reconstrução lexical, concordância
   morfossintática ou uma mistura difícil de interpretar?

4. O top-PMI incoerente pode ser explicado apenas por baixa probabilidade e
   instabilidade da cauda? Que smoothing, truncamento, suporte mínimo ou
   shrinkage seria matematicamente justificável?

5. A forte correlação entre score e variação de entropia é:

   - uma propriedade esperada da métrica;
   - um artefato de calibração;
   - evidência de colapso/deriva global;
   - ou consequência do próprio fenômeno semântico?

6. A rodada de 12+8 épocas é suficiente para retirar "poucas épocas" como
   explicação principal? A loss ainda decrescente invalida essa conclusão?

7. Há evidência de que aumentar ainda mais as épocas poderia resolver o
   problema, ou isso apenas tornaria o modelo mais confiante nos mesmos
   artefatos?

8. Devemos manter `pmi_cosine`, `ppmi_jsd`, ambos ou nenhum? Proponha uma
   alternativa somente se ela medir melhor o objeto conceitual.

9. Qual alternativa de `p_t` deve ser testada primeiro? Especifique a fórmula,
   smoothing, amostragem e como compará-la usando os checkpoints existentes,
   sem novo treinamento.

10. Que testes unitários e invariâncias deveriam ser exigidos antes de outra
    rodada cara?

11. Como verificar se o modelo aprendeu de fato informação semântica antes de
    avaliar mudança temporal? Proponha probes/controles concretos.

12. Qual é o menor conjunto de experimentos que separa causalmente:

    - problema no treinamento do MLM;
    - problema em `q_t(w)`;
    - problema em `p_t`;
    - problema na transformação log-PMI;
    - problema na métrica temporal?

---

## 11. Formato obrigatório do parecer

Estruture o arquivo `./tmp/17-timeformer_pmi_profile_failure_review.md` assim:

1. **Veredito executivo**
2. **Auditoria da implementação**
3. **Causa mais provável**
4. **Hipóteses alternativas, ordenadas por plausibilidade**
5. **O que os resultados de 12+8 épocas realmente demonstram**
6. **Análise do caso `graft_nn`**
7. **Avaliação crítica de `q_t`, `p_t`, log-PMI e métricas**
8. **Experimentos discriminativos usando checkpoints existentes**
9. **Testes necessários antes de novo treinamento**
10. **Recomendação final: continuar, reformular ou abandonar esta medida**

Para cada recomendação experimental, informe:

- hipótese testada;
- mudança exata;
- resultado esperado se a hipótese estiver correta;
- resultado esperado se estiver errada;
- custo computacional aproximado;
- critério objetivo de decisão.

Priorize experimentos baratos que reutilizem:

`outputs/semeval2020_pmi_long_epochs_12_8/continual_real/checkpoint_t00.pt`

e:

`outputs/semeval2020_pmi_long_epochs_12_8/continual_real/checkpoint_t01.pt`

Não proponha uma nova rodada longa antes de esgotar diagnósticos que podem ser
feitos com esses checkpoints.
