# Method

Formal description of the implemented method, using the notation from
`NOTATION.md`. Describes what is running in `src/tracoformer/` and
`scripts/token_time/` as of 2026-06-16. All sections are implemented
unless explicitly marked otherwise.

---

## 1. Vocabulary

Let V be the vocabulary shared across all periods:

```
V = vocabulary of prajjwal1/bert-tiny   (30 522 tokens, WordPiece)
```

V is fixed before training begins and shared across all periods.
No tokens are added or removed between periods.

The active support V_ativo ⊂ V is determined after extraction:

```
V_ativo = { v ∈ V : count_0(v) ≥ n_min  AND  count_1(v) ≥ n_min }
```

Default n_min = 10. V_ativo is the coordinate system for both the
mean μ_t and the reference set V_ref.

---

## 2. Encoder architecture

The model is `prajjwal1/bert-tiny`:

- Hidden dimension: 128
- Feed-forward size: 512
- Attention heads: 2
- Layers: 2
- Vocabulary: 30 522 WordPiece tokens

The model has an MLM head used only during training. At inference,
hidden states from the encoder layers are used directly.

---

## 3. Continual training

Let D_0 and D_1 be the period corpora in chronological order. Starting
from the pre-trained `prajjwal1/bert-tiny` checkpoint θ_init, training
proceeds sequentially:

```
θ_after_0 = train(θ_init,    D_0,   n_epochs = n_epochs_0)
θ         = train(θ_after_0, D_1,   n_epochs = n_epochs_1)
```

The training objective is the standard MLM loss over each period's corpus.
`ContinualPeriodTrainer` (`bert_continual.py`) drives this loop.

There is no regularization against forgetting. The final checkpoint θ is
used for all subsequent extraction (see Section 4).

---

## 4. Encoder fixo (key design decision)

The single final checkpoint θ (trained through D_1) is applied to BOTH
period corpora to extract hidden states:

```
hidden states of D_0  ←  E_θ applied to D_0
hidden states of D_1  ←  E_θ applied to D_1
```

This is implemented in `scripts/token_time/build_profiles.py`.

**Why encoder fixo:** applying the same θ to both periods guarantees that
centroids c_0(w) and c_1(w) live in the same hidden coordinate system. A
period-specific θ_t would rotate and rescale the hidden space between
periods, making direct centroid comparison invalid.

The alternative (period-specific checkpoints) is used only in the log-PMI
method (Section 9), which is invariant to hidden-space rotations.

---

## 5. Contextual centroid

For each period t ∈ {0, 1} and target word w, the contextual centroid is:

```
c_t(w) = (1 / |C_t(w)|) · Σ_{c ∈ C_t(w)} h_θ(w | c)
```

where C_t(w) is the set of contexts in D_t containing w, and h_θ(w | c)
is the hidden state at the position of w in context c, extracted from
the fixed encoder θ at the chosen layer (default: layer 2).

Stored in `PeriodStatistics` (sums and counts per token per layer).
Per-occurrence hidden states and their document indices are stored in
`OccurrenceCache` (required for the permutation null, Section 7).

---

## 6. Active support and type-uniform mean

The type-uniform mean μ_t is the unweighted average centroid over V_ativo:

```
μ_t = (1 / |V_ativo|) · Σ_{v ∈ V_ativo} c_t(v)
```

Unweighted (not frequency-weighted) so that high-frequency tokens do not
dominate the center of the embedding space. Implemented in
`relational.type_uniform_mean`.

The reference set V_ref is a whole-word, alphabetic subset of V_ativo,
filtered to exclude WordPiece fragments and special tokens, and capped
at max_references = 3216 by minimum period count. Used as the axis for
r_t(w). Implemented in `token_time_repository.build_reference_set`.

---

## 7. Relational profile (canonical)

The relational profile of word w at period t is:

```
r_t(w)[v] = cos( c_t(w) − μ_t,  c_t(v) − μ_t )   for v ∈ V_ref
```

r_t(w) ∈ [−1, 1]^{|V_ref|}. Each component encodes how similar w's
contextual position is to reference word v's, after centering both on
the mean of the embedding space at period t.

Implemented in `relational.relational_profile`. The typed wrapper is
`TokenTimeProfile` (`token_time.py`).

---

## 8. Relational displacement

The primary scalar metric is the angular change in r_t(w) between periods:

```
δ(w) = 1 − cos( r_0(w), r_1(w) )   ∈ [0, 2]
```

- δ(w) ≈ 0: w's relational profile unchanged (stable semantic field).
- δ(w) ≈ 1: profiles are orthogonal (uncorrelated neighbor sets).
- δ(w) > 1: profiles are anti-correlated.

Implemented in `relational.displacement`. The typed wrapper is
`TokenTimeDisplacement.score` (`token_time.py`).

The per-reference difference vector Δr(w) = r_1(w) − r_0(w) is available
as `TokenTimeDisplacement.delta` and used in neighborhood reports, but
does not appear in the theory section.

---

## 9. Permutation null

To assess whether δ(w) exceeds sampling chance, repeat B times:

1. Pool all per-occurrence hidden states of w from D_0 and D_1 (pooling
   at the document level using `OccurrenceCache.doc_index`).
2. Randomly repartition the pooled documents into two pseudo-groups of
   sizes |C_0(w)| and |C_1(w)|.
3. Recompute pseudo-centroids c_0^b(w) and c_1^b(w) from each group.
4. Compute r_0^b(w) and r_1^b(w) using the real μ_0, μ_1 and reference
   centroids (held fixed).
5. Compute δ_b(w) = 1 − cos(r_0^b(w), r_1^b(w)).

This yields {δ_b(w)}_{b=1}^B. The repartition is at the **document**
level: all occurrences from the same source document move together,
preserving within-document correlation (topic, register, sense priming).

Implemented in `token_time_null.document_permutation_null`. Driven by
`scripts/token_time/calibrate_null.py` and exposed via
`TokenTimeIndex.null_b`.

---

## 10. Robust standardized displacement

```
ζ(w) = ( δ(w) − median({δ_b(w)}) ) / ( 1.4826 · MAD({δ_b(w)}) )
```

where MAD({δ_b(w)}) = median(|δ_b(w) − median({δ_b(w)})|).

The factor 1.4826 makes MAD a consistent estimator of σ under normality
(report in a footnote, not the main text). ζ(w) measures how many robust
standard deviations the observed displacement lies above the null.

The corresponding one-sided p-value:

```
p(w) = (1 + #{δ_b(w) ≥ δ(w)}) / (B + 1)
```

Implemented as the local variable `z` in `calibrate_null.py` and
`explore_index.py`.

---

## 11. Split-half diagnostic

A repeatability check (not a null). Split the occurrences of w in D_1
into two random halves by document. Compute δ(w) independently on each
half against the full D_0 profile. The Spearman rank correlation ρ_sh
between the two half-displacement rankings across all target words
indicates whether the δ(w) ranking is driven by a few documents (artifact)
or by a corpus-wide property of w.

ρ_sh does not appear in the theory section. Report only in result tables.

Implemented in `token_time_null.split_half_displacement`. Driven by
`scripts/token_time/split_half_repeatability.py` and exposed via
`TokenTimeIndex.split_half`.

---

## 12. Alternative method: log-PMI profile

For comparison, `relational.log_pmi_profiles` implements the log-PMI
relational profile using period-specific checkpoints θ_t:

```
q_t(w)[v] = mean_{c ∈ C_t(w)} P_{θ_t}(v | c̃)
p_t[v]    = P_{θ_t}(v | [CLS][MASK][SEP])
R_t(w)[v] = log( q_t(w)[v] / p_t[v] )
δ_pmi(w)  = 1 − cos( R_0(w), R_1(w) )
```

R_t(w) is invariant to rotations of the hidden space across periods
because it is defined over string token probabilities, not hidden
coordinates. This makes it valid under period-specific θ_t.

Evaluated via `scripts/research/semeval2020/evaluate_relational.py`.

---

## 13. Baseline: Hamilton 2016

`run_hamilton2016_baseline.py` implements the Hamilton et al. (2016)
aligned word2vec method for comparison:

```
E_0^{Ham} = word2vec(D_0)
E_1^{Ham} = word2vec(D_1) · R,   R = argmin_{R^T R = I} ‖E_0 − E_1 R‖_F
r_t^{Ham}(w)[v] = cos(E_t^{Ham}[w], E_t^{Ham}[v])
score^{Ham}(w)  = 1 − cos(E_0^{Ham}[w], E_1^{Ham}[w])
```

Notation uses the `^{Ham}` superscript throughout to distinguish from the
paper's symbols. Not part of the proposed framework.
