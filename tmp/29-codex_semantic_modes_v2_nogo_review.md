# Prompt para codex: auditoria do "no-go" da Fase 1.5 (decomposição em modos espectrais, §7-9 de 12-novo_perfil_relacional.md)

## Contexto

Estamos implementando o "Perfil Relacional v2" descrito em
`docs/12-novo_perfil_relacional.md` (documento canônico, 813 linhas). Antes de
construir a infraestrutura completa de "modos semânticos" (§7-11: matriz de
coesão M_t(w), decomposição espectral, persistência de modos via Hungarian
matching), concordamos em rodar primeiro um experimento barato de "go/no-go"
(Fase 1.5): aplicar o critério de gap (§8) + SVD da matriz de coesão (§7.5)
a um conjunto pequeno de palavras auditadas, e ver se a estrutura espectral
é (a) detectável e (b) discrimina palavras com mudança semântica conhecida
de palavras estáveis.

**Resultado obtido: NO-GO em dois testes independentes**, com um diagnóstico
de causa-raiz que aponta para uma propriedade geométrica do perfil em si
(curva suave, sem gaps internos), não para um problema de parametrização.
Como essa conclusão tem implicações grandes (descartar §7-9 do documento
canônico para este regime de d_model), pedimos uma segunda opinião
independente: **há algum erro de cálculo, de implementação ou de julgamento
que invalide essa conclusão?**

## O que pedimos ao codex

1. Auditar a matemática e o código abaixo (não apenas ler, mas verificar
   passo a passo se cada fórmula implementa corretamente o que o documento
   canônico especifica).
2. Verificar se a interpretação do resultado ("a curva de P_t(w)[v] decai
   suavemente, sem gaps internos, exceto perto do cruzamento por zero, que é
   onde o critério de gap acaba selecionando V_w") é a explicação mais
   provável para os números observados, ou se há uma explicação alternativa
   (bug, escolha de eixo/sinal errada, normalização incorreta, vazamento de
   dados, etc.) que produziria o mesmo padrão.
3. Se a conclusão estiver correta, sugerir se existe alguma variação
   *dentro do espírito do documento canônico* (ainda usando o critério de
   gap em §8, ainda baseada em cosenos de embeddings centralizados) que
   poderia recuperar estrutura -- ou se isso realmente fecha a porta para
   §7-9 neste regime (d_model=128, ~5700 V_ativo).

## 1. Definições do documento canônico (resumo das seções relevantes)

### §4 -- Centralização por período
mu_t = média dos embeddings de um conjunto de referência V_ref no período t
(estamos usando a "variante D" da Fase 1: média não ponderada dos centróides
por tipo de token sobre V_ativo -- ver código abaixo).
e_hat_t(w) = normalize(e_t(w) - mu_t)  (L2-normalizado)

### §5 -- Perfil relacional
P_t(w)[v] = cos(e_hat_t(w), e_hat_t(v)) = e_hat_t(w) . e_hat_t(v)
para v em V (vocabulário fixo / V_ativo).

### §7 -- Matriz de coesão
M_t(w)[v, v'] = P_t(w)[v] * P_t(w)[v'] * cos(e_hat_t(v), e_hat_t(v'))
para v, v' em V_w = {v : P_t(w)[v] > tau}.
M_t(w) é PSD e nunca materializada: M = (D E)(D E)^T onde D = diag(P_t(w)[V_w])
e E são os embeddings centralizados/normalizados de V_w. SVD de D E dá
lambda_i = sigma_i^2, a_i = u_i (vetores singulares à esquerda).

### §8 -- Critério de gap (para tau e para k)
Dada uma sequência não-negativa ordenada decrescente X_1 >= X_2 >= ... > 0,
o "gap relativo" h_i = (X_i - X_{i+1}) / X_i é invariante a reescala positiva
(X -> c*X). O índice selecionado i* é a posição do maior gap, aceito apenas
se h_{i*} > gamma (gamma ~ 0.3). Se nenhum gap excede gamma, retorna None
("sem estrutura clara nesta resolução").

Usos:
- tau (e V_w) a partir dos componentes positivos, ordenados decrescentemente,
  de P_t(w)  (§8.2)
- k (número de modos semânticos) a partir dos autovalores ordenados de
  M_t(w)  (§8.3)

Exemplo do documento (§8.3, usado como teste unitário):
values = [0.41, 0.31, 0.18, 0.04, 0.03], gamma=0.3
gaps = [(0.41-0.31)/0.41, (0.31-0.18)/0.31, (0.18-0.04)/0.18, (0.04-0.03)/0.04]
     = [0.244, 0.419, 0.778, 0.25]
argmax(gaps) = índice 2 (0-indexado) -> i* = 3 (seleciona os 3 primeiros)

## 2. Código relevante (implementação atual)

### `src/timeformers/gap_criterion.py` (completo)

```python
"""Gap criterion for automatic threshold/rank selection (§8 of
docs/12-novo_perfil_relacional.md).
...
"""
from __future__ import annotations

import torch
from torch import Tensor


def relative_gaps(values: Tensor) -> Tensor:
    if values.ndim != 1:
        raise ValueError("values must be a 1D tensor")
    if values.numel() < 2:
        return values.new_zeros(0)
    if torch.any(values <= 0):
        raise ValueError("values must be strictly positive")
    if torch.any(values[:-1] < values[1:]):
        raise ValueError("values must be sorted in descending order")
    return (values[:-1] - values[1:]) / values[:-1]


def select_gap_index(values: Tensor, gamma: float) -> int | None:
    gaps = relative_gaps(values)
    if gaps.numel() == 0:
        return None
    best_index = int(torch.argmax(gaps))
    best_gap = float(gaps[best_index])
    if best_gap <= gamma:
        return None
    return best_index + 1


def adjacent_gaps_valid(values: Tensor, index: int, gamma: float) -> bool:
    gaps = relative_gaps(values)
    if gaps.numel() == 0:
        return True
    left_ok = index == 0 or float(gaps[index - 1]) > gamma
    right_ok = index == values.numel() - 1 or float(gaps[index]) > gamma
    return left_ok and right_ok
```

### `src/timeformers/semantic_modes.py` (completo, incluindo a nova `filter_support_topn`)

```python
"""Semantic mode decomposition via SVD of the cohesion matrix (§7-9 of
docs/12-novo_perfil_relacional.md).
...
"""
from __future__ import annotations

import torch
from torch import Tensor

from .gap_criterion import select_gap_index


def filter_support(profile: Tensor, gamma: float) -> tuple[Tensor, float | None]:
    """Select V_w = {v : P_t(w)[v] > tau} via the gap criterion on the
    positive, descending-sorted components of profile (§8.2).
    Returns (indices into `profile` of V_w, tau), or (empty, None) if no gap
    exceeds gamma.
    """
    positive_mask = profile > 0
    positive_values = profile[positive_mask]
    if positive_values.numel() < 2:
        return profile.new_zeros(0, dtype=torch.long), None
    order = torch.argsort(positive_values, descending=True)
    sorted_values = positive_values[order]
    i_star = select_gap_index(sorted_values, gamma)
    if i_star is None:
        return profile.new_zeros(0, dtype=torch.long), None
    tau = float(sorted_values[i_star - 1])
    positive_indices = torch.nonzero(positive_mask, as_tuple=False).flatten()
    selected = positive_indices[order[:i_star]]
    return selected, tau


def filter_support_topn(profile: Tensor, gamma: float, top_n: int) -> tuple[Tensor, float | None]:
    """Like filter_support, but first restricts the candidate set to the
    top_n components of `profile` by absolute value before applying the gap
    criterion to their positive, descending-sorted subset.
    """
    if profile.numel() == 0:
        return profile.new_zeros(0, dtype=torch.long), None
    top_n = min(top_n, profile.numel())
    candidate_order = torch.argsort(profile.abs(), descending=True)[:top_n]
    candidate_profile = profile[candidate_order]
    selected_local, tau = filter_support(candidate_profile, gamma)
    if tau is None:
        return profile.new_zeros(0, dtype=torch.long), None
    return candidate_order[selected_local], tau


def cohesion_svd(profile_vw: Tensor, embeddings_vw: Tensor) -> tuple[Tensor, Tensor]:
    """SVD-based eigendecomposition of M_t(w) over V_w (§7.5).

    Args:
        profile_vw: (|V_w|,) -- P_t(w)[v] for v in V_w, all > 0.
        embeddings_vw: (|V_w|, d) -- centralized, L2-normalized embeddings
            of V_w (i.e. e_hat_t(v)).
    Returns:
        eigenvalues (descending, >= 0), eigenvectors (columns = a_i, sign
        convention: largest-magnitude component is positive, §7.5).
    """
    d_matrix = profile_vw.unsqueeze(1) * embeddings_vw
    u, sigma, _ = torch.linalg.svd(d_matrix, full_matrices=False)
    eigenvalues = sigma**2
    eigenvectors = u
    for i in range(eigenvectors.shape[1]):
        column = eigenvectors[:, i]
        argmax = torch.argmax(torch.abs(column))
        if column[argmax] < 0:
            eigenvectors[:, i] = -column
    return eigenvalues, eigenvectors


def select_num_modes(eigenvalues: Tensor, gamma: float) -> int | None:
    """k via the gap criterion on the eigenvalues (§8.3). None if the modes
    are not spectrally distinguishable (treated as monosemous)."""
    positive = eigenvalues[eigenvalues > 0]
    if positive.numel() < 2:
        return None
    return select_gap_index(positive, gamma)


def top_tokens_per_mode(eigenvectors, vw_tokens, k, *, top_n=10):
    modes = []
    for i in range(k):
        loadings = eigenvectors[:, i]
        positive = loadings.clamp_min(0)
        order = torch.argsort(positive, descending=True)[:top_n]
        modes.append(
            [(vw_tokens[index], float(loadings[index])) for index in order if loadings[index] > 0]
        )
    return modes
```

### Como o perfil P_t(w) é construído (`scripts/evaluate_semantic_modes_v2.py`, trechos relevantes)

```python
def contextual_centroids(stats: dict, layer: str) -> Tensor:
    counts = stats["counts"].float().unsqueeze(1).clamp_min(1.0)
    return stats["sums"][layer].float() / counts  # média dos hidden states por token, sobre todas as ocorrências


def build_active_support(stats_t0, stats_t1, *, vocab, excluded, n_min) -> Tensor:
    mask = (stats_t0["counts"] >= n_min) & (stats_t1["counts"] >= n_min)
    for index, token in enumerate(vocab):
        if token in SPECIAL_TOKENS or token in excluded:
            mask[index] = False
    return mask  # V_ativo: tokens com >= n_min ocorrências em AMBOS os períodos, excluindo specials e as 8 palavras auditadas


def centralized_embeddings(centroids: Tensor, mu: Tensor) -> Tensor:
    return F.normalize(centroids - mu.unsqueeze(0), dim=1)  # e_hat_t(v)


# No main():
support_mask = build_active_support(stats_t0, stats_t1, vocab=vocab, excluded=excluded, n_min=10)
support_ids = torch.nonzero(support_mask, as_tuple=False).flatten()   # ~5700 tokens
support_tokens = [vocab[i] for i in support_ids.tolist()]

centroids_t0 = contextual_centroids(stats_t0, "mean_last_2")   # (27311, 128)
centroids_t1 = contextual_centroids(stats_t1, "mean_last_2")

mu_t0 = centroids_t0[support_ids].mean(dim=0)   # variante D: média NÃO ponderada dos centróides por tipo, sobre V_ativo
mu_t1 = centroids_t1[support_ids].mean(dim=0)

support_emb_t0 = centralized_embeddings(centroids_t0[support_ids], mu_t0)  # e_hat_t0(v) para v em V_ativo, (5700, 128)
support_emb_t1 = centralized_embeddings(centroids_t1[support_ids], mu_t1)

# Para cada palavra-alvo w:
word_id = token_to_id[word]
word_emb_t0 = F.normalize(centroids_t0[word_id:word_id+1] - mu_t0, dim=1).squeeze(0)  # e_hat_t0(w), (128,)
word_emb_t1 = F.normalize(centroids_t1[word_id:word_id+1] - mu_t1, dim=1).squeeze(0)

# decompose():
profile = support_embeddings @ word_embedding   # P_t(w)[v] para v em V_ativo, (5700,) -- produto escalar de vetores L2-normalizados = cos
vw_indices, tau = filter_support_topn(profile, gamma, top_n)
```

`stats["sums"][layer]` e `stats["counts"]` vêm de um cache pré-computado:
para cada token do vocabulário, a soma (e contagem) dos hidden states de
TODAS as ocorrências desse token no corpus daquele período, extraídos do
checkpoint do modelo (`theta0_d0.pt` para t0, `theta1_d1.pt` para t1).
`mean_last_2` é a média das duas últimas camadas do encoder (d_model=128,
3 camadas).

## 3. O experimento e os resultados

Vocabulário: 27311 tokens. V_ativo (n_min=10, excluindo as 8 palavras
auditadas e tokens especiais): ~5700-5900 tokens (varia ligeiramente por
período, porque o mask depende de count_t0 E count_t1 >= 10, e count_t1 é
fixo mas as 8 palavras excluídas mudam o tamanho total -- na prática
|support_b_ids| ~ 5700).

Palavras auditadas:
- "mudança esperada" (SemEval truth.tsv, binary=1): plane_nn, graft_nn
- "estável" (binary=0), usadas como controle de campo: chairman_nn, tree_nn,
  ball_nn, face_nn, lane_nn, multitude_nn
  (chairman_nn e tree_nn também aparecem como "is_target": true no JSON por
  motivos de rastreamento histórico, mas são controles estáveis segundo o
  truth.tsv)

### Rodada 1: `filter_support` sem restrição (gap criterion sobre TODO o V_ativo positivo)

Resultado (resumo): para TODAS as 8 palavras, em ambos os períodos, tau é da
ordem de 1e-4 a 1e-3 (praticamente zero), |V_w| ~ 4800-5900 (i.e. quase todo
o V_ativo positivo entra em V_w), e k=1 para a maioria (exceções: face_nn em
t1 dá k=2, multitude_nn em t0 dá k=2 -- ambas palavras de CONTROLE estável,
não as mudadas). plane_nn e graft_nn (mudança esperada) deram k=1 nos dois
períodos. n_min_sensitivity é muito instável: plane_nn perde todo o suporte
em n_min=50 (n_vw=0), multitude_nn perde suporte em n_min=20 (n_vw=0).

### Rodada 2: `filter_support_topn` com top_n in {100, 500}

Para TODAS as 8 palavras, em ambos os períodos, com top_n=100 E com
top_n=500: tau=None, k=None, n_vw=0. Ou seja, restringindo aos 100 (ou 500)
candidatos com maior |P_t(w)[v]|, nenhum gap relativo excede gamma=0.3 em
lugar nenhum.

### Diagnóstico: inspeção direta do perfil (plane_nn, t0, layer mean_last_2)

```python
# (script ad-hoc, reproduzindo a construção acima para plane_nn / t0)
sorted_vals, order = torch.sort(profile, descending=True)
print('top 15:', sorted_vals[:15])
print('bottom 5:', sorted_vals[-5:])
```

Saída:
```
top 15: tensor([0.9456, 0.9424, 0.9030, 0.8990, 0.8975, 0.8944, 0.8928, 0.8826, 0.8749,
        0.8739, 0.8739, 0.8737, 0.8720, 0.8674, 0.8661])
bottom 5: tensor([-0.7146, -0.7216, -0.7296, -0.7571, -0.7649])
```

Ou seja: P_t(w)[v] varia continuamente de ~0.95 a ~-0.76 ao longo dos ~5700
tokens de V_ativo, com gaps consecutivos no topo da ordem de 0.001-0.03 (ex.:
(0.9456-0.9424)/0.9456 ~= 0.0034; (0.9030-0.8990)/0.9030 ~= 0.0044). Não há
nenhum salto >30% em lugar nenhum perto do topo. O único gap >30% que o
`filter_support` (sem topn) encontra fica perto do cruzamento por zero,
selecionando ~5000-5900 elementos como V_w com tau~1e-4 -- o que é
interpretado como "o critério está pegando o cruzamento de sinal, não uma
fronteira semântica".

## 4. Interpretação proposta (a verificar)

1. Com d_model=128 e e_hat_t(v) = normalize(centroid(v) - mu_t) onde mu_t é
   a média de ~5700 centróides, os vetores e_hat_t(v) ficam concentrados
   numa região relativamente estreita da esfera unitária (alta dimensão
   efetiva pequena relativa ao número de pontos -> "concentração de medida"),
   fazendo com que cos(e_hat_t(w), e_hat_t(v)) varie suavemente e
   quase-monotonicamente conforme v percorre o conjunto ordenado por
   similaridade, sem clusters discretos.
2. Isso significa que P_t(w) não tem "estrutura de gap" em lugar nenhum
   (exceto, por acaso, perto do cruzamento de zero, que não tem significado
   semântico -- é só onde a maioria dos vetores deixa de ter projeção
   positiva sobre w).
3. Logo, nem tau (V_w), nem k (via autovalores de M_t(w), que dependem de
   V_w e do próprio profile) podem ser determinados de forma confiável pelo
   critério de gap §8 neste regime -- para NENHUMA palavra, mudada ou
   estável. O critério não está "errando a discriminação", está retornando
   None (ou um tau espúrio perto de zero) de forma praticamente uniforme.
4. Conclusão proposta: NO-GO para §7-9 (modos semânticos via SVD de M_t(w))
   neste regime (d_model=128, |V_ativo|~5700, 3 camadas, embeddings médias
   de centróides). Não é um problema de escolha de gamma, n_min ou top_n --
   é uma propriedade da distribuição de cossenos do perfil em si.

## 5. Perguntas específicas para o codex

1. A implementação de `relative_gaps` / `select_gap_index` /
   `filter_support` / `filter_support_topn` / `cohesion_svd` /
   `select_num_modes` está fiel às fórmulas de §7.5, §8.2, §8.3 do
   documento canônico? (Os 14 testes unitários em
   `tests/test_gap_criterion.py` passam -- mas isso testa a lógica isolada,
   não a integração com o perfil real.)
2. Existe algum erro de sinal, eixo, normalização ou indexação na construção
   de `profile = support_embeddings @ word_embedding` ou em
   `centralized_embeddings` que poderia produzir artificialmente essa curva
   suave (por exemplo: `mu_t` calculado sobre o conjunto errado, embeddings
   não realmente L2-normalizados, mistura entre `t0`/`t1`, etc.)?
3. A interpretação "perfis de cosseno concentram-se e decaem suavemente
   devido à baixa dimensionalidade efetiva (d=128) relativa a |V_ativo|~5700"
   é plausível e suficiente para explicar os números observados (top15 em
   [0.86, 0.95], gaps relativos ~0.003-0.03; cauda até -0.76)? Há alguma
   forma rápida de testar essa hipótese (ex.: distribuição de
   cos(e_hat_t(v), e_hat_t(v')) para pares aleatórios v, v' em V_ativo --
   se também for concentrada e suave, reforça a hipótese de concentração de
   medida; se for bimodal/multimodal, sugere que o problema é específico da
   centralização ou do perfil por palavra).
4. Dado que o mesmo padrão (tau~0, k=1 quase sempre, exceções aleatórias em
   palavras de controle) ocorre tanto para plane_nn/graft_nn (mudança
   conhecida) quanto para chairman_nn/tree_nn/ball_nn/face_nn/lane_nn/
   multitude_nn (estáveis), e tanto em t0 quanto em t1 -- isso é evidência
   suficiente de que o critério não está simplesmente "subdimensionado"
   (precisaria de gamma menor, top_n diferente, etc.) mas sim que a
   informação que §7-9 pressupõe (clusters discretos de vizinhos por sentido)
   não está presente nesta representação?
5. Existe alguma variação da definição de P_t(w) ou de V_w -- ainda dentro
   do espírito de "perfil relacional + critério de gap" -- que poderia
   recuperar estrutura? Por exemplo: usar cos(e_t(w), e_t(v)) SEM
   centralizar (e_t bruto, não e_hat_t); usar uma métrica diferente de
   cosseno (distância euclidiana nos embeddings brutos); ou aplicar o
   critério de gap sobre os k maiores AUTOVALORES de uma matriz de Gram
   *sem* o fator P_t(w) (ou seja, decompor diretamente a vizinhança de w por
   PCA/SVD dos embeddings de V_w, independente do perfil)?
6. Se a conclusão NO-GO estiver correta: existe algum risco de que o mesmo
   problema de concentração de medida também afete a métrica de
   deslocamento da Fase 1 (Delta(w) = 1 - cos(P_t0(w), P_t1(w)), que já está
   em uso e deu spearman~0.124 com a variante D)? Ou o deslocamento de um
   vetor de perfil de alta dimensão (~5700) é robusto a essa concentração de
   forma que P_t0(w) e P_t1(w) ainda diferem de modo informativo, mesmo que
   cada P_t individualmente não tenha "gaps"?

## 6. Arquivos para referência (se o codex tiver acesso ao repo)

- `docs/12-novo_perfil_relacional.md` (documento canônico completo)
- `src/timeformers/gap_criterion.py`
- `src/timeformers/semantic_modes.py`
- `tests/test_gap_criterion.py`
- `scripts/evaluate_relational_profile_v2.py` (Fase 0A/1, variante D)
- `scripts/evaluate_semantic_modes_v2.py` (Fase 1.5, script que produziu os
  resultados acima)
- Resultados Fase 1: `outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/relational_profile_v2_phase1/metrics.json`
- Resultados Fase 1.5 (sem topn): `outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/relational_profile_v2_phase1_5/modes.json`
- Resultados Fase 1.5 (topn=100): `outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/relational_profile_v2_phase1_5_topn/modes.json`
- Resultados Fase 1.5 (topn=500): `outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/relational_profile_v2_phase1_5_topn500/modes.json`
