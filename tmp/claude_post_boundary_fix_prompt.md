# Pedido de posicionamento independente após correção das fronteiras do corpus

Você deve realizar uma nova auditoria técnica e científica do projeto:

`/home/jeff/Documentos/trabalhos/papers/paper_timeformer`

Não altere código, dados, checkpoints, documentação ou resultados. Leia o
estado atual do workspace, incluindo mudanças locais não commitadas, e escreva
um parecer independente.

Escreva obrigatoriamente sua resposta em:

`./tmp/timeformer_post_boundary_fix_review.md`

O objetivo é decidir se a ausência de sinal decorre principalmente:

1. de um erro restante no treinamento ou na extração;
2. da baixa qualidade/capacidade do MLM atual;
3. do mascaramento central determinístico;
4. da média de distribuições cloze sobre ocorrências heterogêneas;
5. da transformação por prior/log-PMI;
6. ou da própria definição de perfil relacional por substituibilidade cloze.

Não aceite nossa hipótese automaticamente. Audite os exemplos e resultados
diretamente nos arquivos indicados.

---

## 1. Definição atualmente testada

O modelo é treinado continuamente:

```text
theta_0 = treino(D0)
theta_1 = continua_treino(theta_0, D1)
```

Para cada palavra `w`, período/corpus `D_t` e checkpoint `theta_s`, calculamos:

```text
q(theta_s, D_t, w)
  = média das distribuições MLM sobre ocorrências reais de w em D_t,
    substituindo w por [MASK]
```

O protocolo log-PMI usa:

```text
p_t = P_theta_t(. | [CLS] [MASK] [SEP])

R_t(w)[v] = log((q(theta_t, D_t, w)[v] + eps) /
                (p_t[v] + eps))
```

E então:

```text
pmi_cosine = 1 - cos(R_0(w), R_1(w))
ppmi_jsd = JSD(normalize(max(R_0(w), 0)),
               normalize(max(R_1(w), 0)))
```

Formalização:

`docs/relational_profile_formalization.md`

Implementação:

- `src/timeformers/real_corpus.py`
- `src/timeformers/real_models.py`
- `src/timeformers/relational.py`
- `scripts/run_diachronic_relational_experiment.py`

---

## 2. O bug concreto que encontramos

O preparador SemEval escreve uma janela/sentença independente por linha:

```python
dst.write(" ".join(tokens))
dst.write("\n")
```

Arquivo:

`scripts/prepare_semeval2020_task1.py`

Porém, o leitor anterior fazia:

```python
documents = [tokenize(path.read_text(...))]
```

Isto transformava cada arquivo de período inteiro em um único documento. Assim:

- linhas independentes eram concatenadas;
- janelas MLM atravessavam fronteiras de exemplos;
- probes de ocorrência podiam incluir o final de uma linha e o início de uma
  linha sem relação;
- o sliding window criava exemplos artificiais entre documentos.

O código atual foi corrigido para:

```python
documents = [
    tokenize(line)
    for line in path.read_text(...).splitlines()
    if line.strip()
]
```

Leia a mudança local em:

`src/timeformers/real_corpus.py`

E o teste de regressão em:

`tests/test_real_corpus.py`

Todos os 22 testes relevantes passaram.

---

## 3. Evidência quantitativa do impacto do bug

Antes da correção, o runner criava:

```text
1810-1860: 409.401 janelas MLM
1960-2010: 420.743 janelas MLM
```

Depois de preservar cada linha como documento:

```text
1810-1860: 299.534 janelas MLM
1960-2010: 366.340 janelas MLM
```

Logo, cerca de 110 mil janelas no período antigo e 54 mil no moderno dependiam
da concatenação indevida entre linhas.

Para `graft_nn`, estimamos que uma janela centrada na ocorrência cruzaria a
fronteira da linha em:

```text
1810-1860: 88 de 119 ocorrências, 73,9%
1960-2010: 96 de 109 ocorrências, 88,1%
```

Exemplo de contexto contaminado observado antes da correção:

```text
... burn victim [MASK] of brand-new skin
to the edge_nn he swing down under the eave ...
```

O trecho após `skin` vinha da linha seguinte e não pertencia ao mesmo exemplo.

Outro exemplo:

```text
... government [MASK] be really old news lacayo say
then in a misplace effort to regain ...
```

O contexto continuava em outra linha independente.

Este bug invalida como avaliação limpa os checkpoints antigos:

- `outputs/semeval2020_pmi_pilot/`
- `outputs/semeval2020_pmi_long_epochs_12_8/`

Inclusive a rodada 12+8 foi treinada sobre janelas contaminadas.

---

## 4. Diagnóstico novo de `q_t` bruto

Criamos um diagnóstico que não usa `p_t`, PMI ou métricas temporais:

`scripts/diagnose_cloze_semantics.py`

Ele produz:

- top tokens de `q(theta_s, D_t, w)`;
- rank e probabilidade do próprio alvo;
- entropia e massa top-k;
- previsões por ocorrência;
- grupos heurísticos de sentido para `graft_nn`;
- matriz checkpoint por corpus:

```text
q(theta_0, D0), q(theta_0, D1),
q(theta_1, D0), q(theta_1, D1)
```

O script e seus critérios também devem ser auditados. Em particular, a
classificação heurística de sentidos não é gold e pode conter erros.

---

## 5. Nova baseline treinada após a correção

Treinamos novamente do zero, preservando linhas/documentos:

```text
output: outputs/semeval2020_pmi_line_documents_3_2/

d_model=96
layers=2
heads=4
d_ff=192
batch_size=256
learning_rate=1e-4
base_epochs=3
epochs_per_period=2
seq_len=32
stride=16
probe_mode=occurrence
profile_mode=pmi
seed=1000
```

Loss:

```text
D0:
epoch 0 = 7.4467
epoch 2 = 6.0530

D1:
epoch 0 = 6.3068
epoch 1 = 5.9568
```

Arquivos:

- `outputs/semeval2020_pmi_line_documents_3_2/config.json`
- `outputs/semeval2020_pmi_line_documents_3_2/continual_real/continual_history.json`
- `outputs/semeval2020_pmi_line_documents_3_2/diachronic_relational_changes.csv`
- `outputs/semeval2020_pmi_line_documents_3_2/profiles/prediction_anchor_js/t00.pt`
- `outputs/semeval2020_pmi_line_documents_3_2/profiles/prediction_anchor_js/t01.pt`

---

## 6. Resultado SemEval após a correção

`pmi_cosine`:

```text
Spearman graded = -0.02490
p = 0.88371
ROC-AUC binary = 0.49405
Average precision = 0.45427
```

`ppmi_jsd`:

```text
Spearman graded = -0.06402
p = 0.70658
ROC-AUC binary = 0.47917
Average precision = 0.45256
```

Médias por gold binário:

```text
pmi_cosine:
  estáveis = 1.08843
  mudadas  = 1.11991
  diferença = 0.03148

ppmi_jsd:
  estáveis = 0.30337
  mudadas  = 0.31243
  diferença = 0.00907
```

O diagnóstico continua mostrando:

```text
rho(pmi_cosine, entropy_abs_delta) = 0.94618
p ~= 1e-18
```

Arquivos:

- `outputs/semeval2020_pmi_line_documents_3_2/eval_pmi_cosine/metrics.json`
- `outputs/semeval2020_pmi_line_documents_3_2/eval_ppmi_jsd/metrics.json`
- `outputs/semeval2020_pmi_line_documents_3_2/diagnostics_pmi/summary.json`

Esta baseline é curta. Não use o resultado isoladamente para concluir que um
modelo bem treinado também falharia.

---

## 7. Caso concreto: `graft_nn`

O corpus apresenta uma mudança semanticamente clara.

Em `1810-1860`, a maioria dos contextos refere-se a enxertia botânica:

```text
the bark of the graft and the stock precisely meet

after a graft be insert and as soon as the tree commence growth
the bud on the stock must be rub off

the most perfect way to fit the graft be to make a long sloping cut
```

Em `1960-2010`, aparecem pelo menos:

```text
government graft
graft scandal
wholesale graft payment to police

bone marrow graft
graft of brand-new skin
surgeon ... attach the other end of the graft
```

Gold:

```text
binary = changed
graded = 0.55398
```

---

## 8. `q_t` bruto para `graft_nn` após treino corrigido

Relatório completo:

`outputs/semeval2020_pmi_line_documents_3_2/cloze_diagnostics/graft_nn/report.md`

Dados:

- `outputs/semeval2020_pmi_line_documents_3_2/cloze_diagnostics/graft_nn/report.json`
- `outputs/semeval2020_pmi_line_documents_3_2/cloze_diagnostics/graft_nn/occurrences.csv`

### 8.1 Médias por checkpoint e corpus

```text
theta_0 em D0:
  H(q)/log|V| = 0.781
  rank de graft_nn = 2930
  p(graft_nn) = 5.41e-5
  top = [UNK], and, the, be, of, to, a, that, it, as

theta_0 em D1:
  H(q)/log|V| = 0.716
  rank de graft_nn = 3691
  p(graft_nn) = 3.20e-5
  top = and, the, be, [UNK], to, of, a, in, that, have

theta_1 em D0:
  H(q)/log|V| = 0.780
  rank de graft_nn = 3481
  p(graft_nn) = 4.29e-5
  top = [UNK], and, be, it, to, man, one, the, that, a

theta_1 em D1:
  H(q)/log|V| = 0.736
  rank de graft_nn = 4715
  p(graft_nn) = 2.44e-5
  top = [UNK], the, and, be, a, of, in, to, that, have
```

### 8.2 Médias heurísticas por sentido em `theta_1/D1`

```text
botânico, n=4:
  rank de graft_nn = 4721
  top = [UNK], and, he, have, be

corrupção, n=24:
  rank de graft_nn = 5751
  top = [UNK], and, be, the, in

médico, n=22:
  rank de graft_nn = 5388
  top = [UNK], the, and, be, a
```

Não aparecem substitutos semanticamente claros como:

```text
botânico: scion, cutting, bud
corrupção: corruption, bribery, fraud
médico: transplant, tissue, implant
```

### 8.3 Exemplos individuais auditáveis

Contexto botânico:

```text
so that it fit neatly in the cleft ...
the bark of the [MASK] and the stock precisely meet
```

No checkpoint `theta_0`, baseline corrigida:

```text
rank de graft_nn = 3307
p(graft_nn) = 4.67e-5
top:
[UNK], and, be, more, of, as, the, man, one, other
```

Contexto de corrupção:

```text
that nicaraguan business be inefficient and corrupt ...
except from government [MASK] be really old news
```

No checkpoint `theta_1`:

```text
rank de graft_nn = 4881
p(graft_nn) = 1.42e-6
top:
be, do, make, get, have, see, take, find, know, use
```

Contexto médico:

```text
the clon cell could provide an infusion of fresh bone marrow
and for the burn victim [MASK] of brand new skin
```

No checkpoint `theta_1`:

```text
rank de graft_nn = 6207
p(graft_nn) = 4.06e-6
top:
and, or, with, york, [UNK], from, but, as, than, up
```

Esses resultados sugerem que a falha já existe em `q_t`, antes de `p_t` e
log-PMI.

---

## 9. Matriz checkpoint por corpus

JSD entre médias brutas de `q` para `graft_nn`:

```text
theta_0@D0 vs theta_0@D1 = 0.02219
theta_1@D0 vs theta_1@D1 = 0.03892

theta_0@D0 vs theta_1@D0 = 0.10902
theta_0@D1 vs theta_1@D1 = 0.08617
```

Dentro do mesmo checkpoint, trocar o corpus/período altera menos `q` do que
trocar o checkpoint mantendo o corpus:

```text
efeito aproximado de corpus: 0.022-0.039
efeito aproximado de checkpoint: 0.086-0.109
```

Isto sugere que `q` está mais sensível à deriva global do modelo do que às
diferenças contextuais entre os sentidos de `graft_nn`.

Audite se esta interpretação é válida. JSD não é uma decomposição causal e a
comparação usa apenas uma palavra.

---

## 10. Nossa hipótese atual

Nossa hipótese revisada é:

> O problema principal ocorre antes do log-PMI. O modelo atual não produz uma
> distribuição cloze sensível ao sentido para as ocorrências de `graft_nn`.

Os principais suspeitos são:

### H1. Incompatibilidade de posição de máscara

No treinamento:

```python
mask_pos = candidate_positions[len(candidate_positions) // 2]
```

O token mascarado é sempre a posição central da janela.

No probe:

```python
mask_pos = mask_pos_in_window + 1
```

O alvo pode aparecer em posições variadas, principalmente em linhas menores que
30 tokens.

Com embeddings posicionais aprendidos, o modelo pode ter aprendido a resolver
MLM apenas na posição central. Aplicá-lo em outras posições seria fora de
distribuição.

### H2. Treino curto/modelo pouco capaz

A baseline corrigida usa apenas 3+2 épocas. A loss ainda está caindo e termina
perto de 6.0. É possível que `q_t` ainda seja pouco informativa apenas por
subtreino.

Contudo, a rodada longa anterior não pode resolver esta dúvida porque foi
treinada com fronteiras contaminadas.

### H3. A média cloze não representa o perfil relacional pretendido

Mesmo um MLM competente produz uma distribuição de substitutos na posição de
`w`. Isso não é o mesmo que:

- palavras do mesmo campo;
- colocados;
- vizinhos de embedding;
- relações com entidades e conceitos do contexto;
- estrutura do “círculo social” lexical.

Por exemplo:

```text
bark, sap, stock
```

são relações semanticamente importantes de `graft` no sentido botânico, mas
geralmente não podem substituir `graft` na mesma posição.

Além disso, fazer:

```text
q_t(w) = média de todas as distribuições por ocorrência
```

pode apagar multimodalidade. No período moderno, `graft_nn` mistura corrupção,
medicina e usos botânicos.

### H4. O prior neutro e o log-PMI amplificam um problema já existente

O prior `[CLS] [MASK] [SEP]` continua fora da distribuição de treinamento e o
score continua correlacionado quase perfeitamente com variação de entropia.

Porém, corrigir `p_t` não criará sensibilidade semântica se ela não existir em
`q_t`.

---

## 11. Perguntas obrigatórias

Responda diretamente:

1. A correção de cada linha como documento é semanticamente correta para este
   dataset? Leia o README bruto do SemEval se necessário:

   `data/raw/semeval2020_task1/semeval2020_ulscd_eng/README.md`

2. Os checkpoints anteriores devem ser considerados inválidos para avaliar o
   método?

3. O diagnóstico `diagnose_cloze_semantics.py` mede corretamente `q_t` bruto?
   Há bugs, leakage, desalinhamento entre contextos e tensores ou problemas na
   classificação heurística?

4. Os exemplos concretos permitem concluir que `q_t` não é sensível ao sentido,
   ou o critério top-k/rank do alvo é inadequado?

5. A matriz checkpoint por corpus sustenta a hipótese de deriva global maior que
   efeito contextual? Que métrica ou controle tornaria essa conclusão causal?

6. O mascaramento central determinístico é agora o suspeito prioritário? Como
   testá-lo com custo mínimo?

7. Podemos testar o efeito de posição sem novo treinamento, por exemplo:

   - comparar probes cujo alvo naturalmente cai perto do centro versus bordas;
   - recentralizar todas as ocorrências no probe;
   - medir probabilidade/rank por posição da máscara;
   - aplicar o checkpoint a frases idênticas deslocadas por padding/contexto?

   Quais desses testes são válidos?

8. Se recentralizar o probe melhorar muito `q_t`, isso valida a definição cloze
   ou apenas mostra compatibilidade com o treino?

9. Se nem ocorrências recentralizadas produzirem substitutos plausíveis, qual é
   o próximo passo mais informativo:

   - treinar novamente com masking aleatório;
   - treinar por mais épocas;
   - aumentar o modelo;
   - comparar com um MLM pré-treinado;
   - abandonar `q_t` médio e voltar a perfis relacionais de embeddings/grafos?

10. A distribuição cloze média pode, em princípio, medir mudança de sentido
    multimodal? Devemos comparar conjuntos/mixtures de distribuições por
    ocorrência em vez de suas médias?

11. Qual definição operacional de “perfil relacional” é mais fiel à motivação
    original de mudança no círculo semântico?

12. Qual é a sequência mínima de experimentos que separa:

    - efeito de posição;
    - capacidade/subtreino;
    - inadequação da média;
    - inadequação do prior;
    - inadequação conceitual da definição cloze?

---

## 12. Formato obrigatório do parecer

Escreva em:

`./tmp/timeformer_post_boundary_fix_review.md`

Com as seções:

1. **Veredito executivo**
2. **Validade da correção de fronteiras**
3. **Auditoria do diagnóstico de q bruto**
4. **Interpretação dos exemplos de graft_nn**
5. **Análise da matriz checkpoint por corpus**
6. **Causa mais provável, com grau de confiança**
7. **A definição cloze ainda é defensável?**
8. **Experimentos mínimos, ordenados por informação/custo**
9. **Critérios objetivos para continuar ou abandonar cada caminho**
10. **Recomendação final**

Para cada conclusão, marque explicitamente:

- **fato observado**;
- **inferência**;
- **hipótese não testada**.

Para cada experimento proposto, informe:

- hipótese;
- mudança exata;
- arquivos/checkpoints usados;
- custo;
- resultado esperado sob cada hipótese;
- critério objetivo de decisão.

Não recomende outra rodada longa antes de esgotar testes de posição e
sensibilidade contextual que reutilizam os checkpoints corrigidos em:

`outputs/semeval2020_pmi_line_documents_3_2/continual_real/`
