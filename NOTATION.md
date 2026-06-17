# Notation Reference

Maps implementation names to formal paper notation.
Code names prioritize readability and stability; paper names prioritize
mathematical precision. The implemented codebase is `src/tracoformer/`.

---

## 1. Canonical method: centroid-based relational profile

The primary contribution of the paper. Implemented in `src/tracoformer/`.

### Training and encoder

| Code | Paper | LaTeX | Notes |
|---|---|---|---|
| `prajjwal1/bert-tiny` | `E_{θ_0}` | `E_{\theta_0}` | Pre-trained starting checkpoint |
| `ContinualPeriodTrainer` | — | — | Trains sequentially on D_0, D_1 |
| Final checkpoint after D_1 | `θ` | `\theta` | Fixed encoder used for ALL periods |
| `D_t`, corpus file for period t | `D_t` | `D_t` | Corpus for period t |

**Encoder fixo (key design):** a single final checkpoint `θ` (trained
through D_1) is applied to both D_0 and D_1 to extract hidden states.
This guarantees centroids from different periods live in the same hidden
coordinate system, making direct comparison valid.

### Contextual centroid

| Code | Paper | LaTeX | Notes |
|---|---|---|---|
| `contextual_centroids(stats, layer)` | `c_t(w)` | `c_t(w)` | Mean hidden state of w across occurrences in D_t, at fixed θ |
| `PeriodStatistics` cache | `{h_θ(w|c)}` | — | Raw per-occurrence hidden states stored in `OccurrenceCache` |

```
c_t(w) = (1 / |C_t(w)|) · Σ_{c ∈ C_t(w)} h_θ(w | c)
```

where `C_t(w)` is the set of contexts in D_t containing w, and
`h_θ(w | c)` is the hidden state at the position of w in context c,
under the fixed encoder θ.

### Active support and type-uniform mean

| Code | Paper | LaTeX | Notes |
|---|---|---|---|
| `build_active_support(...)` | `V_ativo` | `\mathcal{V}_{\mathrm{act}}` | Tokens with count ≥ n_min in both periods |
| `type_uniform_mean(stats, layer, support)` | `μ_t` | `\mu_t` | Unweighted mean centroid over V_ativo |
| `build_reference_set(...)` | `V_ref` | `\mathcal{V}_{\mathrm{ref}}` | Whole-word subset of V_ativo used as reference axis |

```
μ_t = (1 / |V_ativo|) · Σ_{v ∈ V_ativo} c_t(v)
```

μ_t is unweighted (not frequency-weighted) so that high-frequency tokens
do not dominate the center of the embedding space.

### Relational profile (canonical symbol: lowercase r)

| Code | Paper | LaTeX | Notes |
|---|---|---|---|
| `relational_profile(centroids, mu, target_id, support_ids)` | `r_t(w)` | `r_t(w)` | Centered centroid cosine profile over V_ref |
| `TokenTimeProfile.vector` | `r_t(w)` | `r_t(w)` | Same object, typed wrapper |
| `standardize(vector)` | `std_r_t(w)` | — | Per-reference z-score of r_t(w); NOT ζ(w) |

```
r_t(w)[v] = cos( c_t(w) − μ_t,  c_t(v) − μ_t )   for v ∈ V_ref
```

**Symbol rule:** `r_t(w)` (lowercase r) always refers to the centroid
cosine profile. `R_t(w)` (uppercase R) is reserved for the log-PMI
profile (Section 2). Never swap the cases.

### Relational displacement (scalar)

| Code | Paper | LaTeX | Notes |
|---|---|---|---|
| `displacement(profile_t0, profile_t1)` | `δ(w)` | `\delta(w)` | Primary scalar metric |
| `TokenTimeDisplacement.score` | `δ(w)` | `\delta(w)` | Same value, typed wrapper |
| `TokenTimeDisplacement.delta` | `Δr(w)` | `\Delta r(w)` | Per-reference difference vector r_1(w) − r_0(w) |

```
δ(w) = 1 − cos( r_0(w), r_1(w) )   ∈ [0, 2]
```

δ(w) ≈ 0: w's relational profile unchanged (same contextual neighbors).
δ(w) near 1: orthogonal profiles (different sets of neighbors).
δ(w) > 1: anti-correlated profiles.

**Symbol rule:** `δ(w)` (without subscript) is the primary displacement
scalar. The per-reference vector `Δr(w)` = r_1(w) − r_0(w) is only used
in neighborhood reports, not in the theory section.

### Permutation null

| Code | Paper | LaTeX | Notes |
|---|---|---|---|
| `document_permutation_null(...)` | `{δ_b(w)}_{b=1}^B` | `\{\delta_b(w)\}_{b=1}^{B}` | B null samples via document-level repartition |
| `TokenTimeIndex.null_b(...)` | same | same | Facade over above |
| `OccurrenceCache.doc_index` | document block id | — | Groups occurrences by source document |

The null resamples w's occurrences at the **document** level: all
occurrences in the same document move together between pseudo-groups.
This preserves within-document correlation (topic, register, sense
priming) that an occurrence-level shuffle would destroy.

**Symbol rule:** never introduce a distribution symbol `𝒫_w` for the
null. Reference it as the list `{δ_b(w)}` or "permutation null statistics
for w".

### Robust standardized displacement

| Code | Paper | LaTeX | Notes |
|---|---|---|---|
| `z` (local variable in `calibrate_null.py`, `explore_index.py`) | `ζ(w)` | `\zeta(w)` | Robust standardized displacement |

```
ζ(w) = ( δ(w) − median({δ_b(w)}) ) / ( 1.4826 · MAD({δ_b(w)}) )
```

where MAD({δ_b(w)}) = median(|δ_b(w) − median({δ_b(w)})|).

The factor 1.4826 makes MAD a consistent estimator of σ under normality.
Explain in a footnote; do not derive in the main text.

**Symbol rule:** `ζ(w)` is the ONLY robust standardized scalar. The
per-reference standardized profile `std_r_t(w)[v] = standardize_v(r_t(w)[v])`
is a diagnostic output variable, not a paper notation.

### Split-half diagnostic

| Code | Paper | LaTeX | Notes |
|---|---|---|---|
| `split_half_displacement(...)` | ρ_sh | `\rho_{\mathrm{sh}}` | Spearman correlation of half-sample δ rankings |
| `TokenTimeIndex.split_half(...)` | same | same | Facade |

ρ_sh does not appear in the theory section. Use it only in result tables
as a diagnostic of whether the δ(w) ranking is reproducible under
resampling.

---

## 2. Alternative method: log-PMI relational profile

Used for comparison experiments. NOT the canonical paper method.
Implemented in `src/tracoformer/relational.py` (`log_pmi_profiles`,
`pmi_cosine_displacement`, `ppmi_jsd_displacement`).

Uses period-specific checkpoints θ_t (not encoder fixo), and operates
over string token probability distributions rather than hidden states.
This makes it invariant to hidden-space rotations across periods.

### Distributions

| Code | Paper | LaTeX | Notes |
|---|---|---|---|
| `q_t` | `q_t(w)` | `q_t(w)` | Mean of P_{θ_t}(·|c̃) over c ∈ C_t(w), c̃ has w masked |
| `p_t` | `p_t` | `p_t` | Neutral probe marginal: P_{θ_t}(·|[CLS][MASK][SEP]) |

### Log-PMI profile (alternative symbol: uppercase R)

| Code | Paper | LaTeX | Notes |
|---|---|---|---|
| `log_pmi_profiles(q, p)` | `R_t(w)` | `R_t(w)` | Returns (n_words, \|V\|) tensor |
| — | `R_t(w)[v] = log(q_t(w)[v] / p_t[v])` | `R_t(w)[v] = \log\!\left(\frac{q_t(w)[v]}{p_t[v]}\right)` | Component-wise |

**Symbol rule:** `R_t(w)` (uppercase R) always refers to log-PMI.
Never use it for centroid profiles.

### PMI displacement metrics

| Code | Paper | LaTeX | Notes |
|---|---|---|---|
| `pmi_cosine_displacement(R_t0, R_t1)` | `δ_pmi(w)` | `\delta_{\mathrm{pmi}}(w)` | `1 - cos(R_0(w), R_1(w))` ∈ [0, 2] |
| `ppmi_jsd_displacement(R_t0, R_t1)` | `δ_JSD(w)` | `\delta_{\mathrm{JSD}}(w)` | JSD between PPMI distributions ∈ [0, log 2] nats |

---

## 3. Symbol conflict table

| Symbol | Reserved for | Never use for |
|---|---|---|
| `r_t(w)` | centroid cosine profile (canonical) | log-PMI profile |
| `R_t(w)` | log-PMI profile (alternative) | centroid cosine profile |
| `c_t(w)` | contextual centroid (hidden state mean) | distribution mean |
| `μ_t` | type-uniform mean over V_ativo | any other mean |
| `δ(w)` | centroid displacement scalar: 1−cos(r_0, r_1) | PMI displacement |
| `δ_pmi(w)` | PMI cosine displacement | centroid displacement |
| `δ_JSD(w)` | PPMI JSD displacement | raw JSD |
| `{δ_b(w)}` | permutation null samples | any other distribution |
| `ζ(w)` | robust standardized displacement scalar | per-reference z-score |
| `ρ_sh` | split-half Spearman (tables only) | theory section |
| `Δr(w)` | per-reference difference vector r_1−r_0 | displacement scalar |
| `std_r_t(w)` | per-reference z-score of r_t (code/reports only) | paper notation |
| `q_t(w)` | conditional occurrence distribution (PMI method) | anything else |
| `p_t` | neutral probe marginal (PMI method) | anything else |
| `V_ativo` | active support (count ≥ n_min in both periods) | full vocabulary |
| `V_ref` | whole-word reference subset of V_ativo | active support |
| `V` | full vocabulary | any subset |
| `𝒫_w` | do not use | permutation null (use list notation) |

---

## 4. Code ↔ paper notation quick-reference

| File | Object/function | Paper symbol |
|---|---|---|
| `relational.py` | `contextual_centroids` | c_t(w) |
| `relational.py` | `type_uniform_mean` | μ_t |
| `relational.py` | `relational_profile` | r_t(w) |
| `relational.py` | `displacement` | δ(w) |
| `relational.py` | `log_pmi_profiles` | R_t(w) |
| `relational.py` | `pmi_cosine_displacement` | δ_pmi(w) |
| `relational.py` | `ppmi_jsd_displacement` | δ_JSD(w) |
| `token_time.py` | `TokenTimeProfile.vector` | r_t(w) |
| `token_time.py` | `TokenTimeDisplacement.score` | δ(w) |
| `token_time.py` | `TokenTimeDisplacement.delta` | Δr(w) |
| `token_time_null.py` | `document_permutation_null` | {δ_b(w)} |
| `token_time_null.py` | `split_half_displacement` | ρ_sh (indirectly) |
| `token_time_repository.py` | `TokenTimeIndex.null_b` | {δ_b(w)} |
| `calibrate_null.py` (local var `z`) | — | ζ(w) |
| `explore_index.py` (local var `z`) | — | ζ(w) |
| `bert_continual.py` | `ContinualPeriodTrainer` | training of θ |
| `build_profiles.py` | encoder fixo extraction | θ applied to all D_t |
