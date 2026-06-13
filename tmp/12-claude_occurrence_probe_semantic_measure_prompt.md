# Prompt para 2a opinião externa: ainda estamos medindo o objeto errado?

Você deve realizar uma revisão técnica e científica independente do projeto:

`/Users/jeff/Documents/trabalhos/papers/paper-timeformers`

Não altere nenhum arquivo de código, documentação, dados ou resultados. Sua
tarefa é somente analisar a formulação atual, os resultados recentes e escrever
um parecer crítico.

Escreva obrigatoriamente o parecer final em:

`./tmp/14-timeformer_occurrence_probe_semantic_measure_review.md`

O parecer deve ser autocontido e deve responder diretamente se ainda estamos
medindo o objeto errado.

---

## 1. Contexto da tese atual

O Paper 2 foi reorientado para a seguinte tese:

> mudança semântica temporal é mudança no perfil relacional de uma palavra
> entre checkpoints de Transformer treinados cronologicamente.

O modelo não recebe tempo como input. O treino é cronológico:

```text
theta_0 = treino(D_0)
theta_1 = continua_treino(theta_0, D_1)
...
```

Depois medimos mudança semanticamente, não por coordenada absoluta do embedding,
mas por relações.

A intuição conceitual é:

> uma palavra muda semanticamente quando seu "círculo social" lexical muda de
> forma estrutural. Pequenas perturbações locais não bastam; queremos detectar
> mudança de ordem de grandeza/estrutura no conjunto de relações.

Analogias usadas internamente:

- duas pessoas podem mudar pequenos detalhes e ainda pertencer ao mesmo círculo
  social;
- dois furacões da mesma classe não são idênticos, mas pertencem à mesma ordem
  de grandeza;
- talvez estejamos interessados em estabilidade/mudança estrutural, não em
  microdiferenças vetoriais.

Essa é a dúvida principal:

> Mesmo após corrigir o probe, ainda estamos medindo diferença pequena demais,
> local demais, ou errada demais?

---

## 2. O que já foi corrigido desde o parecer anterior

Leia estes arquivos:

- `src/timeformers/real_corpus.py`
- `src/timeformers/relational.py`
- `scripts/run_diachronic_relational_experiment.py`
- `scripts/evaluate_semeval2020_relational.py`
- `scripts/diagnose_semeval2020_relational.py`
- `docs/07-data_layout.md`

Antes, o probe era artificial:

```text
[CLS] palavra [MASK] [SEP]
```

e o score principal era derivado de uma matriz target-vs-target 37x37.

Após sua/uma revisão externa anterior, implementamos:

### 2.1 Probe por ocorrência real

Novo dataset:

```text
RealTargetOccurrenceDataset
```

Para cada ocorrência real da palavra-alvo no corpus:

```text
... contexto esquerdo palavra contexto direito ...
```

criamos:

```text
... contexto esquerdo [MASK] contexto direito ...
```

e extraímos a distribuição prevista pelo checkpoint sobre uma lista de
âncoras.

Ou seja:

```text
q_t(w) = média_ocorrências P_t(anchor | contexto real com w mascarada)
```

Isso é mais próximo da tarefa MLM que o modelo viu no treino.

### 2.2 Score direto

Implementamos:

```text
direct_jsd(w) = JSD(q_t0(w), q_t1(w))
```

onde `q_t0(w)` e `q_t1(w)` são distribuições da mesma palavra sobre as mesmas
âncoras em dois checkpoints.

Também mantivemos scores antigos como diagnóstico:

- `mean_abs_delta`
- `max_abs_delta`

mas `direct_jsd` passou a ser a métrica principal do novo modo.

---

## 3. Dataset e pilotos

Dataset:

```text
SemEval-2020 Task 1, inglês, lematizado
data/processed/semeval2020_task1/eng_lemma/
```

Períodos:

```text
1810-1860
1960-2010
```

Alvos:

```text
37 targets com gold binário e graduado
```

Âncoras:

```text
~932 âncoras após filtragem do runner
```

---

## 4. Resultados antes da correção

### Probe antigo, piloto 10k

Configuração:

```text
10k janelas por período
d_model=32
1 camada
1 época por período
probe artificial [CLS] word [MASK] [SEP]
score max_abs_delta
```

Resultado:

```text
Spearman graded: -0.051
AUC binary:       0.560
AP binary:        0.479
```

Diagnóstico:

```text
predicted_vs_entropy_t0: rho=-0.383, p=0.019
```

Interpretação anterior:

- score antigo tinha algum sinal binário fraco;
- mas provavelmente estava contaminado por entropia/confiança do probe;
- não parecia uma boa medida semântica.

---

## 5. Resultados depois da correção

### Probe por ocorrência real + direct_jsd

Usamos os mesmos checkpoints do piloto 10k antigo para isolar o efeito do novo
probe/score:

```text
outputs/semeval2020_eng_lemma_pilot_10k_occurrence/
```

Configuração:

```text
checkpoints reaproveitados do piloto 10k
d_model=32
1 camada
1 época por período
probe-mode occurrence
max_probe_occurrences_per_target=500
score principal: direct_jsd
```

Resultado:

```text
Spearman graded: -0.032
AUC binary:       0.402
AP binary:        0.387
```

Diagnóstico do `direct_jsd`:

```text
predicted_vs_graded:            rho=-0.032, p=0.851
predicted_vs_binary:            rho=-0.169, p=0.318
predicted_vs_freq_t0:           rho= 0.159, p=0.348
predicted_vs_freq_t1:           rho= 0.194, p=0.251
predicted_vs_freq_min:          rho= 0.247, p=0.140
predicted_vs_freq_sum:          rho= 0.189, p=0.263
predicted_vs_freq_abs_delta:    rho= 0.141, p=0.406
predicted_vs_freq_ratio:        rho= 0.074, p=0.665
predicted_vs_entropy_t0:        rho=-0.064, p=0.707
predicted_vs_entropy_t1:        rho=-0.164, p=0.333
predicted_vs_entropy_abs_delta: rho=-0.009, p=0.958
```

Interpretação interna:

- o viés por entropia praticamente desapareceu;
- mas o alinhamento com o gold piorou;
- os scores `direct_jsd` estão muito comprimidos, em torno de `0.003`;
- isso pode indicar que o modelo/checkpoint é fraco;
- mas também pode indicar que `JSD` entre distribuições médias sobre âncoras
  ainda não mede mudança semântica estrutural do modo desejado.

Top do ranking por `direct_jsd`:

```text
ounce_nn          binary=0 graded=0.285
fiction_nn        binary=0 graded=0.021
record_nn         binary=1 graded=0.427
stroke_vb         binary=0 graded=0.176
contemplation_nn  binary=0 graded=0.071
gas_nn            binary=0 graded=0.160
rag_nn            binary=1 graded=0.277
pin_vb            binary=0 graded=0.207
face_nn           binary=0 graded=0.138
chairman_nn       binary=0 graded=0.000
plane_nn          binary=1 graded=0.882
prop_nn           binary=1 graded=0.625
```

Bottom por `direct_jsd` inclui palavras mudadas fortes:

```text
graft_nn          binary=1 graded=0.554
tip_vb            binary=1 graded=0.679
```

Isso aumenta a suspeita de que ainda estamos medindo o objeto errado.

---

## 6. Pergunta central

O usuário/pesquisador ainda suspeita que estamos medindo errado.

Queremos que você avalie criticamente:

1. O `direct_jsd` entre distribuições médias sobre âncoras realmente mede
   mudança semântica estrutural?
2. A média sobre ocorrências não colapsa polissemia e apaga mudança de sentido?
3. A lista de âncoras frequentes é um espaço semântico adequado?
4. `JSD(q_t0, q_t1)` é sensível a microdiferenças de distribuição que não
   representam mudança estrutural?
5. O score deveria ser rank-based/top-k, community-based ou thresholded em vez
   de divergência contínua?
6. O problema principal ainda é escala do modelo ou é formulação da métrica?
7. Como distinguir:
   - mudança semântica real;
   - mudança sintática;
   - mudança de frequência;
   - mudança de domínio;
   - mudança de distribuição das âncoras;
   - ruído de checkpoint?

Não assuma que a hipótese relacional está correta. Se a métrica atual ainda não
testa a hipótese, diga isso claramente.

---

## 7. Alternativas que queremos avaliar

### 7.1 Rank/top-k sobre âncoras

Em vez de comparar distribuições inteiras por JSD:

```text
q_t(w) = distribuição sobre 932 âncoras
```

usar apenas estrutura de vizinhança:

```text
N_k,t(w) = top-k âncoras mais prováveis/similares
score(w) = 1 - Jaccard(N_k,t0(w), N_k,t1(w))
```

ou:

```text
Rank-Biased Overlap
Spearman/Kendall entre rankings
weighted top-k displacement
```

Isso parece mais alinhado com a ideia do “círculo social”: não importa se todas
as relações variam um pouco, importa se o conjunto estrutural de vizinhos muda.

### 7.2 Medida por ordem de grandeza

O usuário sugeriu que talvez precisemos sair de diferenças absolutas e pensar
em mudança por ordem de grandeza, como uma escala logarítmica.

Pergunta:

```text
score = log(1 + alteração estrutural / nulo esperado)
```

ou z-score contra nulo por palavra seria mais adequado?

### 7.3 Nulo por resampling

Criar dois pseudo-períodos dentro do mesmo período real:

```text
1810-1860_A vs 1810-1860_B
1960-2010_A vs 1960-2010_B
```

Medir quanto `direct_jsd` ou top-k muda sem mudança temporal real. Depois usar:

```text
score_calibrado = score_observado / score_nulo
```

ou:

```text
z = (score_observado - mean_null) / std_null
```

Pergunta: isso é obrigatório antes de qualquer conclusão?

### 7.4 Preservar multimodalidade

Se `q_t(w)` é média das ocorrências, uma palavra com dois sentidos pode ter:

```text
sentido A: ancora x,y,z
sentido B: ancora p,q,r
```

A média pode apagar a mudança se um sentido cresce e outro diminui.

Alternativas:

- clusterizar ocorrências por palavra/período;
- comparar distribuições de clusters;
- usar distância entre conjuntos/distribuições de ocorrências;
- usar optimal transport entre ocorrências ou entre clusters;
- medir mudança top-k separadamente por modo.

Isso é necessário já no SemEval ou só como extensão?

### 7.5 Contextual embeddings em vez de logits sobre âncoras

Talvez o erro esteja em usar distribuição preditiva do MLM. Alternativa:

- extrair embeddings contextuais das ocorrências reais;
- agregar por palavra/período;
- calcular vizinhos entre âncoras/targets;
- comparar top-k relacional.

Isso seria mais próximo de LSCD contextual tradicional. Mas queremos manter a
contribuição relacional e cronológica.

### 7.6 Âncoras semânticas melhores

Âncoras atuais são frequentes e compartilhadas.

Talvez precisem ser:

- POS-compatible com o alvo;
- semanticamente estáveis;
- selecionadas por baixa mudança no nulo;
- menos numerosas;
- balanceadas por campo semântico;
- não apenas frequentes;
- filtradas para excluir funcionais ou quase funcionais.

Pergunta: a escolha das âncoras pode explicar os resultados ruins?

### 7.7 Escala/modelo

O modelo usado nos pilotos é pequeno:

```text
d_model=32
1 camada
1 época por período
10k janelas por período
```

Talvez nenhuma métrica funcione com checkpoints tão fracos.

Mas antes de gastar muito em escala, queremos saber:

- há algum teste barato para distinguir métrica errada vs. modelo fraco?
- por exemplo, avaliar perplexidade/loss por target?
- medir se o modelo realmente prediz âncoras relevantes?
- comparar ranking de anchors manualmente para palavras conhecidas?

---

## 8. O que esperamos do parecer

Escreva em:

`./tmp/14-timeformer_occurrence_probe_semantic_measure_review.md`

Use esta estrutura:

1. **Resumo executivo**
2. **Sua compreensão da tese relacional**
3. **O novo probe por ocorrência corrigiu o problema anterior?**
4. **O `direct_jsd` mede mudança semântica estrutural?**
5. **Por que o sinal SemEval ainda não apareceu?**
6. **Estamos medindo diferença local demais em vez de mudança estrutural?**
7. **Riscos de média sobre ocorrências e perda de polissemia**
8. **Riscos das âncoras atuais**
9. **Rank/top-k, nulo por resampling, multimodalidade ou escala: o que vem
   primeiro?**
10. **Mudanças concretas recomendadas no código**
11. **Experimento mínimo recomendado**
12. **Veredito**

No veredito, responda diretamente:

- Ainda estamos medindo o objeto errado?
- Devemos abandonar `direct_jsd` como score principal?
- O próximo passo deve ser top-k/rank, nulo por resampling, mudança de âncoras,
  embeddings contextuais, preservação multimodal ou escala maior?
- Qual teste barato pode dizer se o problema é métrica ou modelo fraco?

Novamente: não altere outros arquivos. Escreva apenas o parecer solicitado em
`./tmp/14-timeformer_occurrence_probe_semantic_measure_review.md`.
