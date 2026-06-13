# Formalização do Perfil Relacional de uma Palavra

**Status:** definição canônica — base matemática para implementação e paper  
**Relacionado a:** `05-relational_change_current_plan.md`, `07-data_layout.md`

---

## Motivação

Queremos medir mudança semântica temporal de uma palavra `w` sem depender de:

- coordenadas absolutas do espaço de embeddings (sensíveis a rotações e deriva de treinamento);
- uma lista de âncoras pré-definidas por heurística (POS, frequência, etc.);
- um parâmetro `k` de vizinhança.

A intuição central é que uma palavra muda semanticamente quando a **natureza do seu entorno** muda, não apenas quando membros individuais do entorno são substituídos. O médico que passa de um hospital para uma clínica continua no campo médico; `gay` que passa do campo de humor para o campo de identidade mudou estruturalmente.

A formulação abaixo opera sobre a distribuição de saída do MLM head — que está definida sobre tokens de linguagem natural (strings), não sobre coordenadas internas do Transformer. Isso a torna invariante a mudanças no sistema de coordenadas dos vetores ocultos entre checkpoints.

---

## Notação

| Símbolo | Significado |
|---|---|
| `V` | Vocabulário fixo do modelo (construído de todo o corpus, compartilhado entre períodos) |
| `θ_t` | Checkpoint após treinamento contínuo em `D_0, …, D_t` |
| `P_{θ_t}(· | c)` | Distribuição sobre `V` produzida pelo MLM head dado input `c` |
| `C_t(w)` | Conjunto de contextos (sentenças) em que `w` aparece em `D_t` |
| `c̃` | Versão de `c` com `w` substituído por `[MASK]` |
| `Δ^|V|` | Simplex de probabilidade sobre `V` |

---

## Definição

### 1. Distribuição condicional por ocorrência

Para cada contexto `c ∈ C_t(w)`, seja `c̃` a sentença com `w` mascarado. O modelo produz `P_{θ_t}(· | c̃)` na posição do `[MASK]`.

A distribuição condicional de `w` no checkpoint `θ_t` é a média sobre suas ocorrências reais:

```
q_t(w) = (1 / |C_t(w)|) · Σ_{c ∈ C_t(w)} P_{θ_t}(· | c̃)
```

`q_t(w) ∈ Δ^|V|` — o modelo diz, em média, quais tokens ele associa à posição de `w`, dado tudo que aprendeu até `t`.

### 2. Distribuição marginal (probe neutro)

O prior do modelo — o que ele prevê sem informação sobre a palavra-alvo — é estimado pelo probe neutro:

```
p_t = P_{θ_t}(· | [CLS] [MASK] [SEP])
```

`p_t` captura a distribuição de base do checkpoint `θ_t`: essencialmente a expectativa de frequência de cada token segundo o que o modelo aprendeu até o período `t`.

> **Nota sobre alternativas para `p_t`:**  
> - **Probe neutro** (padrão): uma chamada ao modelo, captura o prior aprendido.  
> - **Frequência de corpus**: `p_t[v] = count_t(v) / Σ count_t(v')` — independente do modelo, útil para separar o efeito do dado do efeito do modelo.  
> - **Média sobre alvos**: `p_t = (1/|W|) · Σ_{w ∈ W} q_t(w)` — internamente consistente, depende da escolha de `W`.  
>
> Para o primeiro experimento com corpus real, usar o probe neutro.

### 3. Perfil relacional

O perfil relacional de `w` no período `t` é o vetor de log-PMI sobre o vocabulário inteiro:

```
R_t(w)[v] = log( q_t(w)[v] / p_t[v] ),   para todo v ∈ V
```

`R_t(w) ∈ ℝ^|V|`

**Interpretação componente a componente:**

| Valor | Significado |
|---|---|
| `R_t(w)[v] ≫ 0` | `w` atrai especificamente `v`, muito além da frequência base de `v`. `v` é um marcador semântico positivo de `w` em `t`. |
| `R_t(w)[v] ≈ 0` | `v` aparece com `w` na mesma proporção que aparece em geral. Sem associação específica. |
| `R_t(w)[v] ≪ 0` | `v` é menos provável dado `w` do que seria aleatoriamente. Repulsão semântica. |

---

## Propriedades

**P1 — Invariância a mudanças de coordenada**

`R_t(w)` é definido sobre tokens de linguagem natural (`V`), não sobre coordenadas do espaço de embeddings. Rotações, reflexões ou reescalamentos dos vetores ocultos de `θ_t` não alteram `q_t(w)` nem `p_t`, e portanto não alteram `R_t(w)`.

**P2 — Normalização por deriva de domínio**

Se um token `v` ficou mais frequente em `t₁` por mudança de domínio (e não por mudança semântica de `w`), então `p_{t₁}[v]` também aumentou proporcionalmente. O perfil `R_{t₁}(w)[v] = log(q_{t₁}(w)[v] / p_{t₁}[v])` é normalizado por essa deriva. O que sobra é a associação específica de `w` com `v`, líquida do efeito geral do corpus.

**P3 — Âncoras emergem, não são pressupostas**

As dimensões de `R_t(w)` com maior valor absoluto ao longo do tempo são os marcadores semânticos mais informativos de `w`. Palavras com `|R_t(w)[v]|` pequeno em todos os períodos são as âncoras naturais — estáveis o suficiente para não discriminar `w` de outras palavras. A lista de âncoras é resultado da análise, não input.

**P4 — Palavras funcionais têm peso automaticamente reduzido**

Palavras funcionais (`a`, `the`, `of`, etc.) têm `p_t[v]` alto em qualquer corpus. Isso reduz seu PMI independentemente de `q_t(w)[v]`. Não é necessário filtragem manual por POS.

---

## Deslocamento semântico temporal

### Versão cosseno (recomendada para o primeiro experimento)

```
Δ(w, t₀, t₁) = 1 - cos(R_{t₀}(w), R_{t₁}(w))
```

Mede a mudança angular entre os perfis. Insensível a escala, sensível a reorientação do campo semântico.

### Versão JSD sobre associações positivas

A parte positiva do perfil (PPMI) concentra as associações genuínas:

```
R_t⁺(w)[v] = max(0, R_t(w)[v])

π_t(w) = R_t⁺(w) / ‖R_t⁺(w)‖₁

Δ_JSD(w, t₀, t₁) = JSD(π_{t₀}(w), π_{t₁}(w))
```

`Δ_JSD ∈ [0, log 2]` — interpretável em bits como quantidade de informação que distingue o campo semântico de `w` em `t₀` do campo em `t₁`.

---

## O que esta formulação elimina

| Escolha eliminada | Razão |
|---|---|
| Lista de âncoras pré-definidas | `p_t` é o prior do modelo; âncoras emergem como dimensões estáveis de `R_t` |
| Parâmetro `k` de vizinhança | Perfil definido sobre `V` inteiro |
| Alinhamento de espaços entre checkpoints | Comparação sobre tokens de string, não coordenadas |
| Filtragem manual por POS | Palavras funcionais têm `p_t[v]` alto → PMI baixo → peso natural pequeno |

---

## Pseudocódigo

```python
def compute_relational_profile(model, word, corpus_t, vocab, seq_len, batch_size):
    """
    Retorna R_t(word) como vetor de log-PMI sobre o vocabulário.
    """
    # Passo 1: distribuição marginal (probe neutro)
    neutral_probe = make_neutral_probe(vocab, seq_len)      # [CLS] [MASK] [SEP]
    p_t = softmax(model(neutral_probe).logits[mask_pos])    # |V|

    # Passo 2: distribuição condicional por ocorrência
    occurrences = find_occurrences(word, corpus_t)          # lista de sentenças
    distributions = []
    for batch in batched(occurrences, batch_size):
        masked = mask_target(batch, word)                   # w → [MASK]
        logits = model(masked).logits[:, mask_pos, :]       # (batch, |V|)
        distributions.append(softmax(logits))
    q_t_w = mean(distributions, dim=0)                      # |V|

    # Passo 3: log-PMI
    eps = 1e-9
    R_t_w = log(q_t_w + eps) - log(p_t + eps)              # |V|
    return R_t_w


def semantic_displacement(R_t0, R_t1, method="cosine"):
    if method == "cosine":
        return 1 - cosine_similarity(R_t0, R_t1)
    elif method == "jsd":
        ppmi_t0 = normalize(relu(R_t0), p=1)
        ppmi_t1 = normalize(relu(R_t1), p=1)
        return jensen_shannon_divergence(ppmi_t0, ppmi_t1)
```

---

## Questões abertas

1. **Estabilidade numérica de `p_t`**: o probe neutro `[CLS] [MASK] [SEP]` pode não ser representativo do prior real se o modelo aprendeu a distinguir esse padrão de sequências naturais. Alternativa: média de `q_t(w)` sobre um subconjunto amplo de palavras.

2. **Cobertura de `C_t(w)`**: palavras raras com poucas ocorrências em `D_t` terão `q_t(w)` ruidoso. Necessário limiar mínimo de ocorrências (sugestão: `|C_t(w)| ≥ 10`).

3. **Tamanho do vocabulário**: `|V| = 30.000` é tratável. O vetor `R_t(w)` tem 30k dimensões, mas é esparso na prática (a maior parte das dimensões tem PMI ≈ 0).

4. **Trajetória como sequência de perfis**: para corpora com mais de dois períodos, `R_{t_0}(w), R_{t_1}(w), …, R_{t_n}(w)` define uma trajetória no espaço dos perfis relacionais. Métricas de trajetória (velocidade, aceleração, persistência da direção) podem ser computadas sobre essa sequência sem nenhum componente adicional.
