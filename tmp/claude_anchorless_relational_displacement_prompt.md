# Prompt para 2a opinião externa: deslocamento relacional sem âncoras pré-definidas

Você deve realizar uma revisão técnica e científica independente do projeto:

`/Users/jeff/Documents/trabalhos/papers/paper-timeformers`

Não altere nenhum arquivo de código, documentação, dados ou resultados. Sua
tarefa é somente analisar a nova formulação conceitual abaixo e escrever um
parecer crítico.

Escreva obrigatoriamente o parecer final em:

`./tmp/timeformer_anchorless_relational_displacement_review.md`

O parecer deve ser autocontido e deve nos ajudar a decidir se esta é a
formulação matemática correta antes de reescrever código e planejamento.

---

## 1. Contexto do problema

O projeto está tentando formular uma abordagem para mudança semântica temporal.
Após vários pilotos, percebemos que a noção de "âncoras" entrou cedo demais e
talvez tenha deformado o problema.

Até agora, tentamos medir:

```text
q_t(w) = distribuição prevista sobre uma lista fixa de âncoras
score(w) = JSD(q_t0(w), q_t1(w))
```

Mesmo após trocar o probe artificial por ocorrências reais, os resultados no
SemEval-2020 Task 1 continuaram ruins. Uma hipótese forte é que ainda faltava
uma formulação matemática mais precisa para o que chamamos de:

```text
deslocamento semântico temporal
```

O usuário/pesquisador agora defende que:

> O perfil relacional de uma palavra deve ser comparado com o perfil relacional
> da mesma palavra em um período anterior. Não deve haver âncoras pré-definidas.

Ou seja, uma âncora, se existir, deve ser resultado da análise:

```text
âncora = palavra cujo perfil relacional muda pouco ao longo do tempo
```

e não uma lista escolhida antes por frequência, POS ou heurística.

---

## 2. Intuição científica

A intuição central é:

> Uma palavra muda semanticamente quando seu círculo relacional muda de forma
> estrutural.

Exemplo:

```text
gay@t0 -> happy, cheerful, merry
gay@t1 -> lesbian, homosexual, queer
```

Não precisamos escolher `happy`, `cheerful` ou `lesbian` como âncoras antes.
Queremos que o próprio perfil relacional revele que o círculo de vizinhança
mudou.

Outra analogia:

> Uma pessoa pode continuar no mesmo círculo social mesmo que todos os amigos
> mudem um pouco. O que importa é se a estrutura do círculo mudou, não cada
> microvariação local.

Assim, deslocamento semântico temporal não deve ser:

```text
diferença absoluta entre vetores
```

nem:

```text
diferença contra uma lista fixa de âncoras escolhidas previamente
```

mas sim:

```text
distância entre perfis relacionais induzidos em períodos diferentes.
```

---

## 3. Formulação candidata

Para cada período `t`:

```text
V_t = vocabulário observável no período t
x_t(w) = representação de w no período t
sim_t(w, v) = similaridade entre x_t(w) e x_t(v)
```

O perfil relacional de `w` é uma função:

```text
R_t(w): V_t \ {w} -> R
R_t(w)[v] = sim_t(w, v)
```

O deslocamento semântico temporal é:

```text
Delta_t(w) = d(R_t(w), R_{t+1}(w))
```

Essa é a definição abstrata. O ponto difícil é escolher `d` e lidar com
vocabulários diferentes entre períodos.

---

## 4. Operacionalizações candidatas

### 4.1 Vocabulário comum

Usar o vocabulário comparável:

```text
V* = V_t ∩ V_{t+1}
```

Então:

```text
R_t(w)     = [sim_t(w, v)     for v in V*]
R_{t+1}(w) = [sim_{t+1}(w, v) for v in V*]
Delta(w) = d(R_t(w), R_{t+1}(w))
```

Isso não é uma lista de âncoras pré-definidas. É o universo comparável mínimo.

Perguntas:

- Isso resolve o problema das âncoras?
- Ou `V*` acaba funcionando como uma âncora gigante implícita?
- Há risco de perder palavras emergentes/desaparecidas?

### 4.2 Ranking/top-k de vizinhos

Definir:

```text
N_{k,t}(w) = top-k vizinhos de w em V_t
N_{k,t+1}(w) = top-k vizinhos de w em V_{t+1}
```

E medir:

```text
Delta_Jaccard@k(w) =
  1 - |N_{k,t}(w) ∩ N_{k,t+1}(w)| / |N_{k,t}(w) ∪ N_{k,t+1}(w)|
```

Ou:

```text
Delta_RBO(w) = 1 - RBO(rank_t(w), rank_{t+1}(w))
```

Essa opção parece mais alinhada com a ideia de "o círculo mudou?".

Perguntas:

- Top-k é melhor que JSD para mudança estrutural?
- Como escolher `k`?
- Devemos reportar uma curva em `k`?
- O ranking deve ser sobre vocabulário comum ou vocabulário de cada período?

### 4.3 Distribuição relacional sem âncoras

Transformar similaridades em distribuição:

```text
P_t(v | w) = softmax(sim_t(w, v) / tau), v in V*
```

Então:

```text
Delta_JSD(w) = JSD(P_t(.|w), P_{t+1}(.|w))
```

Isso é parecido com o que tentamos, mas sem lista fixa de âncoras escolhidas
antes.

Perguntas:

- Isso é matematicamente defensável?
- Como escolher temperatura `tau`?
- JSD fica sensível demais à cauda do vocabulário?
- Top-k/RBO seria mais robusto?

### 4.4 Grafo kNN temporal

Para cada período, construir:

```text
G_t = grafo kNN das palavras em V_t
```

O perfil de `w` é seu ego-grafo:

```text
Ego_t(w) = subgrafo local ao redor de w em G_t
```

O deslocamento é:

```text
Delta_graph(w) = graph_distance(Ego_t(w), Ego_{t+1}(w))
```

Perguntas:

- Essa formulação captura melhor "círculo social"?
- É complexa demais para o Paper 2?
- Há uma versão simples e defensável?

### 4.5 Perfil de segunda ordem

Em vez de comparar vizinhos diretos, comparar padrões de vizinhança:

```text
R2_t(w)[u] = similarity(N_{k,t}(w), N_{k,t}(u))
```

Ou seja, `w` é descrita por "com quais palavras ela compartilha vizinhos".

Perguntas:

- Isso seria mais robusto a pequenas trocas locais?
- Ou torna a interpretação científica opaca demais?

---

## 5. Relação com âncoras

Nesta nova formulação:

```text
âncora não é input do método.
```

Âncora é uma palavra que emerge depois:

```text
w é âncora se Delta_t(w) é baixo ao longo do tempo
w é palavra em mudança se Delta_t(w) é alto
```

Portanto:

```text
âncoras = resultado da investigação dos deslocamentos
```

e não:

```text
âncoras = referência fixa escolhida antes da investigação
```

Perguntas:

- Essa mudança é cientificamente defensável?
- Há circularidade em descobrir âncoras e depois usá-las?
- Devemos evitar totalmente a palavra "âncora" e falar apenas em estabilidade
  relacional?

---

## 6. Como obter x_t(w)

Ainda há uma questão aberta: como obter a representação de `w` por período.

Possibilidades:

1. Média de embeddings contextuais das ocorrências de `w` no checkpoint `t`;
2. Clusters de embeddings contextuais para preservar sentidos;
3. Distribuição de ocorrência em vez de centroide;
4. Representação por probe MLM;
5. Vizinhança induzida por probabilidades de substituição.

Perguntas:

- Qual é a forma mínima e defensável para o primeiro experimento?
- Média contextual é aceitável para SemEval?
- O problema de polissemia exige clusters já no primeiro teste?
- Como evitar que o método vire apenas LSCD contextual clássico?

---

## 7. O que queremos que você avalie

Não assuma que a nova formulação está correta. Queremos uma crítica dura.

Avalie:

1. A definição abstrata `Delta_t(w) = d(R_t(w), R_{t+1}(w))` é uma boa definição
   de deslocamento semântico temporal?
2. O perfil relacional deve ser vetor de similaridades, ranking, top-k,
   distribuição, grafo ou outra coisa?
3. Como lidar com vocabulários diferentes entre períodos sem reintroduzir
   âncoras?
4. Como distinguir deslocamento semântico de mudança geral de domínio/corpus?
5. O que deve ser medido para todas as palavras?
6. Como identificar estabilidade sem pressupor estabilidade?
7. Como avaliar no SemEval-2020 Task 1?
8. Qual métrica tem maior chance de correlacionar com o gold humano:
   Jaccard@k, RBO, JSD sobre softmax de similaridades, distância de grafos,
   ou outra?
9. Qual é o menor experimento que testa a formulação sem implementar uma
   arquitetura grande?
10. Quais pressuposições do benchmark sintético deixaram de valer no corpus real?

---

## 8. Arquivos relevantes para contexto

Leia, se necessário:

- `docs/novo_planejamento.md`
- `docs/data_layout.md`
- `src/timeformers/real_corpus.py`
- `src/timeformers/relational.py`
- `scripts/run_diachronic_relational_experiment.py`
- `scripts/evaluate_semeval2020_relational.py`
- `scripts/diagnose_semeval2020_relational.py`
- `tmp/timeformer_occurrence_probe_semantic_measure_review.md`

Mas o foco principal deve ser a formulação matemática, não uma auditoria de
código.

---

## 9. Formato obrigatório do parecer

Escreva em:

`./tmp/timeformer_anchorless_relational_displacement_review.md`

Use esta estrutura:

1. **Resumo executivo**
2. **Sua compreensão da nova formulação**
3. **A definição abstrata é adequada?**
4. **Qual deve ser o perfil relacional?**
5. **Como comparar perfis entre períodos?**
6. **Como lidar com vocabulário mutável?**
7. **O papel das âncoras deve desaparecer ou emergir depois?**
8. **Como obter `x_t(w)` no primeiro experimento?**
9. **Como avaliar no SemEval?**
10. **Principais riscos de validade**
11. **Experimento mínimo recomendado**
12. **Mudanças concretas no código**
13. **Veredito**

No veredito, responda diretamente:

- Esta formulação sem âncoras pré-definidas é melhor do que a anterior?
- Qual operacionalização devemos implementar primeiro?
- O que deve ser abandonado da implementação atual?
- Qual é o teste mínimo para saber se estamos finalmente medindo
  deslocamento semântico temporal?

Novamente: não altere outros arquivos. Escreva apenas o parecer solicitado em
`./tmp/timeformer_anchorless_relational_displacement_review.md`.
