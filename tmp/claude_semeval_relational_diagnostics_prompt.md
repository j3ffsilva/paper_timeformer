# Prompt para 2a opinião externa: diagnóstico do piloto SemEval relacional

Você deve realizar uma revisão técnica e científica independente do projeto:

`/Users/jeff/Documents/trabalhos/papers/paper-timeformers`

Não altere nenhum arquivo de código, documentação, dados ou resultados. Sua
tarefa é somente analisar o estado atual do pipeline, os resultados do piloto
SemEval e escrever um parecer.

Escreva obrigatoriamente o parecer final em:

`./tmp/timeformer_semeval_relational_diagnostics_review.md`

O parecer deve ser autocontido e detalhado o suficiente para orientar a próxima
decisão experimental.

---

## 1. Contexto científico

O Paper 2 foi reorientado. A ideia atual não é mais inserir tempo dentro do
Transformer, nem aprender uma representação temporal explícita concatenada ao
embedding.

A nova tese é:

> mudança semântica temporal é mudança no perfil relacional de uma palavra
> entre checkpoints de Transformer treinados cronologicamente.

Treinamento:

```text
theta_0 = treino(D_0)
theta_1 = continua_treino(theta_0, D_1)
...
theta_t = continua_treino(theta_{t-1}, D_t)
```

O modelo não recebe embedding de período, token-time, adapter temporal ou
qualquer sinal temporal explícito. A temporalidade entra pela ordem cronológica
dos dados.

Depois do treino, medimos a mudança por perfis relacionais:

```text
r_t(w)[v] = similaridade_t(w, v)
delta_rel(w, a, b) = r_b(w) - r_a(w)
```

O ponto conceitual é evitar depender de coordenadas absolutas dos embeddings.
Se o espaço vetorial inteiro gira, translada ou muda de sistema de coordenadas,
mas as relações entre palavras permanecem equivalentes, então a mudança
semântica relacional deveria ser aproximadamente zero.

Uma analogia usada na discussão interna:

> Um círculo social pode permanecer estável mesmo que cada pessoa mude um
> pouco. O que interessa não é toda microvariação individual, mas uma alteração
> estrutural no círculo de relações.

Portanto, pequenas diferenças vetoriais absolutas não devem ser tratadas
automaticamente como mudança semântica relevante.

---

## 2. Implementação atual relevante

Leia estes arquivos:

- `docs/novo_planejamento.md`
- `docs/data_layout.md`
- `src/timeformers/real_corpus.py`
- `src/timeformers/real_models.py`
- `scripts/prepare_semeval2020_task1.py`
- `scripts/run_diachronic_relational_experiment.py`
- `scripts/evaluate_semeval2020_relational.py`
- `scripts/diagnose_semeval2020_relational.py`

Pipeline atual:

1. `scripts/prepare_semeval2020_task1.py`
   - prepara SemEval-2020 Task 1 inglês;
   - usa corpora lematizados;
   - preserva tokens `target_pos`, como `attack_nn`, `circle_vb`;
   - gera:

```text
data/processed/semeval2020_task1/eng_lemma/
  corpus/1810-1860.txt
  corpus/1960-2010.txt
  targets.txt
  anchors.txt
  truth.tsv
  metadata.json
```

2. `scripts/run_diachronic_relational_experiment.py`
   - lê os dois períodos;
   - treina um `RealStaticMLM` continuamente;
   - checkpoint 0 após treino em `1810-1860`;
   - checkpoint 1 após continuação em `1960-2010`;
   - usa probe:

```text
[CLS] palavra [MASK] [SEP]
```

   - calcula:

```text
q_t(w) = P_t(anchor | [CLS] w [MASK] [SEP])
```

   - compara distribuições sobre âncoras com Jensen-Shannon;
   - exporta `diachronic_relational_changes.csv` com:
     - `mean_abs_delta`;
     - `max_abs_delta`.

3. `scripts/evaluate_semeval2020_relational.py`
   - compara ranking produzido com gold do SemEval:
     - Spearman contra score graduado;
     - AUC/AP/F1 contra rótulo binário.

4. `scripts/diagnose_semeval2020_relational.py`
   - junta score relacional, gold, frequência por período e entropia do perfil
     de âncoras;
   - procura sinais de artefato por frequência, cobertura ou incerteza.

---

## 3. Dados usados

Dataset real gratuito:

```text
data/raw/semeval2020_task1/semeval2020_ulscd_eng/
```

Processado:

```text
data/processed/semeval2020_task1/eng_lemma/
```

Resumo:

- 37 alvos;
- 2 períodos:
  - `1810-1860`;
  - `1960-2010`;
- 939 âncoras no preparo completo;
- runner usou 932 âncoras depois de filtragem por vocabulário;
- `metadata.json` registra cobertura dos alvos em cada período.

---

## 4. Pilotos executados

### 4.1 Piloto 2k

Saída:

```text
outputs/semeval2020_eng_lemma_pilot_2k/
```

Configuração:

```text
2k janelas por período
d_model=32
1 camada
1 época por período
```

Resultado:

```text
mean_abs_delta:
  Spearman graded: -0.030
  AUC binary:       0.476

max_abs_delta:
  Spearman graded: -0.085
  AUC binary:       0.478
```

Leitura interna: pipeline funciona, mas configuração não captura o gold.

### 4.2 Piloto 10k

Saída:

```text
outputs/semeval2020_eng_lemma_pilot_10k/
```

Configuração:

```text
10k janelas por período
d_model=32
1 camada
1 época por período
tempo aproximado: 150s em CPU
```

Resultado:

```text
mean_abs_delta:
  Spearman graded: -0.151
  AUC binary:       0.473

max_abs_delta:
  Spearman graded: -0.051
  AUC binary:       0.560
  AP binary:        0.479
```

Leitura interna:

- `max_abs_delta` melhorou levemente para classificação binária;
- Spearman graduado continua ruim;
- ranking ainda mistura palavras claramente mudadas com estáveis.

Top do ranking por `max_abs_delta` no piloto 10k:

```text
bag_nn       binary=0 graded=0.100
graft_nn     binary=1 graded=0.554
stroke_vb    binary=0 graded=0.176
word_nn      binary=0 graded=0.179
risk_nn      binary=0 graded=0.000
tip_vb       binary=1 graded=0.679
tree_nn      binary=0 graded=0.071
circle_vb    binary=1 graded=0.171
prop_nn      binary=1 graded=0.625
attack_nn    binary=1 graded=0.144
```

Há acertos parciais (`graft_nn`, `tip_vb`, `prop_nn`), mas também falsos
positivos altos (`bag_nn`, `stroke_vb`, `risk_nn`, `tree_nn`).

---

## 5. Diagnóstico atual

Rodamos:

```text
outputs/semeval2020_eng_lemma_pilot_10k/diagnostics_max/
```

Para `max_abs_delta`, correlações de Spearman:

```text
predicted_vs_graded:            rho=-0.051, p=0.763
predicted_vs_binary:            rho= 0.102, p=0.547
predicted_vs_freq_t0:           rho=-0.051, p=0.763
predicted_vs_freq_t1:           rho=-0.075, p=0.661
predicted_vs_freq_min:          rho=-0.031, p=0.854
predicted_vs_freq_sum:          rho=-0.118, p=0.488
predicted_vs_freq_abs_delta:    rho=-0.157, p=0.354
predicted_vs_freq_ratio:        rho=-0.182, p=0.282
predicted_vs_entropy_t0:        rho=-0.383, p=0.019
predicted_vs_entropy_t1:        rho=-0.305, p=0.067
predicted_vs_entropy_abs_delta: rho= 0.033, p=0.848
```

Para `mean_abs_delta`, houve correlação positiva com frequência inicial:

```text
predicted_vs_freq_t0: rho=0.380, p=0.020
```

Interpretação interna provisória:

- `mean_abs_delta` parece mais contaminado por frequência/cobertura;
- `max_abs_delta` não parece dominado por frequência;
- `max_abs_delta` parece associado à entropia inicial do perfil relacional;
- o score pode favorecer palavras cujo perfil preditivo inicial é menos
  espalhado/menos incerto;
- talvez a medição esteja capturando confiança/estrutura do probe, não mudança
  semântica lexical.

---

## 6. Questão central para você avaliar

Estamos tentando medir mudança estrutural no perfil relacional de uma palavra,
mas os primeiros pilotos SemEval ainda não alinham bem com o gold.

Queremos sua avaliação crítica:

1. O problema parece estar na formulação científica?
2. O problema parece estar no probe `[CLS] palavra [MASK] [SEP]`?
3. O problema parece estar na escala de treino, que ainda é pequena?
4. O problema parece estar no score escalar (`mean_abs_delta`/`max_abs_delta`)?
5. O problema parece estar na escolha de âncoras?
6. O problema parece estar em catastrophic forgetting, domínio, vocabulário ou
   treino contínuo?
7. O diagnóstico por entropia aponta para um viés real ou pode ser artefato
   do tamanho pequeno da amostra de 37 alvos?

Não assuma que nossa hipótese está correta. Se achar que estamos medindo o
objeto errado, diga claramente.

---

## 7. Alternativas que queremos que você avalie

Avalie se devemos testar, em ordem de prioridade:

### 7.1 Score rank-based/top-k

Em vez de usar diferença média/máxima no vetor relacional inteiro:

```text
delta_rel = r_1(w) - r_0(w)
```

medir alteração estrutural no conjunto de vizinhos:

```text
Jaccard(kNN_0(w), kNN_1(w))
Rank-biased overlap
Spearman entre rankings de vizinhança
```

Essa opção parece mais alinhada com a analogia do “círculo social”: pequenas
variações individuais são ignoradas, mas troca estrutural de vizinhança aparece.

### 7.2 Normalização por entropia/incerteza

Se o score atual favorece perfis menos entrópicos, talvez usar:

```text
score_norm = score / f(entropy_t0, entropy_t1)
```

ou calibrar o score por bandas de entropia/frequência.

Pergunta: isso é estatisticamente defensável ou arriscaria remover sinal real?

### 7.3 Controle nulo com resampling

Criar dois pseudo-períodos a partir do mesmo corpus/período e medir o deslocamento
esperado sem mudança semântica real:

```text
D_0a -> D_0b
D_1a -> D_1b
```

Depois usar z-score ou percentil:

```text
score_calibrado(w) = (score_observado(w) - mean_null(w)) / std_null(w)
```

Pergunta: esse deveria ser o próximo passo obrigatório?

### 7.4 Melhorar o probe

O probe atual:

```text
[CLS] palavra [MASK] [SEP]
```

Talvez seja fraco porque remove contexto demais. Alternativas:

- usar contextos reais contendo a palavra e extrair distribuição sobre âncoras;
- usar templates múltiplos;
- usar embedding contextual médio por ocorrências;
- calcular vizinhança diretamente em embeddings de ocorrência agregados;
- usar pseudo-log-likelihood de âncoras em contextos reais.

Qual dessas alternativas é mais coerente com a tese relacional?

### 7.5 Âncoras

As âncoras atuais são palavras frequentes presentes nos dois períodos.

Riscos:

- âncoras também mudam semanticamente;
- âncoras muito genéricas achatam o perfil;
- âncoras escolhidas por frequência podem induzir entropia alta;
- 932 âncoras talvez sejam demais para um modelo pequeno.

Como escolher âncoras melhores?

### 7.6 Escala de treino

O piloto 10k usa só parte do corpus. O corpus completo tem centenas de milhares
de janelas disponíveis por período.

Pergunta: antes de mudar método, devemos rodar mais escala?

Ou os resultados atuais já indicam uma falha conceitual/medição que escala não
resolverá?

---

## 8. O que esperamos do parecer

Escreva em:

`./tmp/timeformer_semeval_relational_diagnostics_review.md`

Use a seguinte estrutura:

1. **Resumo executivo**
2. **Sua compreensão da proposta atual**
3. **O piloto SemEval está testando a hipótese correta?**
4. **Interpretação dos resultados 2k e 10k**
5. **Interpretação do diagnóstico por frequência e entropia**
6. **Principais riscos de validade**
7. **O que provavelmente está errado**
8. **Alternativas recomendadas**
9. **Próximo experimento mínimo**
10. **Mudanças necessárias no código**
11. **Veredito**

No veredito, responda diretamente:

- Devemos continuar com o paradigma de mudança semântica relacional?
- Devemos abandonar `mean_abs_delta`/`max_abs_delta`?
- O próximo teste deve ser rank-based/top-k, normalização por entropia,
  resampling null, melhoria do probe ou simplesmente escala maior?
- Qual é o menor experimento que pode falsificar ou fortalecer a direção atual?

Novamente: não altere outros arquivos. Escreva apenas o parecer solicitado em
`./tmp/timeformer_semeval_relational_diagnostics_review.md`.
