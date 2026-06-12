# Perfil Relacional v2 -- Resultados das Fases 0A/1/1.5 e diagnóstico para revisão

## Status e propósito deste documento

`docs/novo_perfil_relacional.md` define o "Perfil Relacional e Trajetória
Semântica v2" (documento canônico, §1-14). Este documento registra, com
detalhe reprodutível, o que foi testado das Fases 0A/1/1.5 do plano de
implementação, o que funcionou (parcialmente) e o que **não** funcionou,
para servir de base a uma próxima rodada de re-projeto de §7-11. Não
reescreve o documento canônico -- é o "diário de bordo empírico" que
motiva essa revisão.

Tudo abaixo roda sobre os caches já existentes
(`outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/hidden_relational_profiles/cache/theta{0,1}_d{0,1}.pt`),
sem reextração: para cada token do vocabulário (27311 tokens), há soma e
contagem de hidden states sobre todas as ocorrências no corpus de cada
período (`sums[layer]`, `counts`), de onde se deriva o centróide
contextual `centroid_t(v) = sums_t(v) / counts_t(v)`.

Modelo: `outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/config.json` --
vocab_size=27311, d_model=128, 3 camadas, períodos `["1810-1860",
"1960-2010"]`, 37 palavras-alvo do SemEval-2020 Task 1 (eng_lemma).

---

## 1. Fase 0A -- Contrato de dados

Definições adotadas (compatíveis com §3-4 do documento canônico):

- **V_ativo** (= "support"): tokens com `count_t >= n_min` em AMBOS os
  períodos, excluindo tokens especiais e (quando aplicável) as palavras
  auditadas. Com `n_min=10`, `|V_ativo| = 11.621` tokens (confirmado pelo
  codex; os números de 5000-5900 mencionados abaixo são os componentes
  *positivos* de `P_t(w)` dentro desse V_ativo, não o V_ativo inteiro).
- **mu_t**: ponto de centralização do período t (§4.1). Quatro variantes
  testadas na Fase 1 (ver §2).
- **e_hat_t(v) = normalize(centroid_t(v) - mu_t)**: embedding centralizado
  e L2-normalizado.
- **P_t(w)[v] = e_hat_t(w) . e_hat_t(v) = cos(e_hat_t(w), e_hat_t(v))**:
  perfil relacional (§5).

Implementação: `scripts/evaluate_relational_profile_v2.py`
(`contextual_centroids`, `build_active_support`, `relational_profile`).

---

## 2. Fase 1 -- Ablação de centralização (mu_t)

Quatro formas de calcular `mu_t`, todas avaliadas com o MESMO `P_t(w)`
sobre o mesmo suporte (apenas a centralização varia):

| Variante | Definição de mu_t |
|---|---|
| A_reference_mean | média dos centróides de um conjunto de referência pequeno (top ~3216 tokens por `min(count_t0,count_t1)`, n_min=100) -- replica a abordagem v1 |
| B_global_mu | `sum(sums_t) / sum(counts_t)` sobre TODO o vocabulário (média ponderada por ocorrência, literal ao §4.1 "média de todas as ocorrências") |
| C_global_mu_active_support | igual a B, mas restrito a V_ativo com n_min=50 |
| D_type_uniform_mu | média NÃO ponderada dos centróides por tipo de token, sobre V_ativo (n_min=10) -- cada *tipo* de palavra conta igual, independente de frequência |

Métrica: `Delta(w) = 1 - cos(P_t0(w), P_t1(w))`, avaliada contra
`truth.tsv` (binary/graded) via Spearman, ROC-AUC, AP, e
`changed_above_stable_p95` (fração de alvos "mudados" com Delta acima do
percentil 95 dos alvos "estáveis").

### Resultados (37 alvos, ordenado por Spearman desc)

| Variante | Layer | Spearman (p) | ROC-AUC | AP | changed_above_stable_p95 |
|---|---|---|---|---|---|
| D_type_uniform_mu | mean_last_2 | 0.124 (p=0.466) | 0.601 | 0.513 | 0.000 |
| A_reference_mean | mean_last_2 | 0.108 (p=0.524) | 0.592 | 0.589 | 0.125 |
| D_type_uniform_mu | layer_2 | 0.083 (p=0.625) | 0.568 | 0.498 | 0.000 |
| A_reference_mean | layer_2 | 0.078 (p=0.645) | 0.565 | 0.559 | 0.188 |
| C_global_mu_active_support | mean_last_2 | 0.044 (p=0.794) | 0.500 | 0.468 | 0.063 |
| C_global_mu_active_support | layer_2 | 0.013 (p=0.941) | 0.491 | 0.428 | 0.000 |
| B_global_mu | mean_last_2 | 0.005 (p=0.976) | 0.506 | 0.469 | 0.063 |
| B_global_mu | layer_2 | -0.025 (p=0.884) | 0.482 | 0.463 | -- |

(arquivo completo: `outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/relational_profile_v2_phase1/metrics.json`)

### Interpretação

- **B e C (mu_t ponderado por ocorrência, §4.1 literal) performam pior que
  o acaso** em ROC-AUC (~0.49-0.51). Diagnóstico: `mu_t` ponderado por
  ocorrência é dominado por palavras de função de altíssima frequência
  (ex.: "the" tem `count ≈ 448771`, vs. média ≈ 240 e mediana muito menor
  entre os ~11.6k tokens de V_ativo). Isso empurra `mu_t` para perto do
  centróide de "the"/"of"/"and"/etc., tornando `e_hat_t(v)` quase
  uniforme entre os tokens de conteúdo.
- **D (média não ponderada por tipo) é a melhor**, marginalmente acima de
  A (abordagem v1), mas a diferença não é estatisticamente significativa
  com n=37 (p>0.4 em ambos).
- **Nenhuma variante separa bem "mudado" de "estável"**:
  `changed_above_stable_p95` é 0 ou próximo de 0 em quase todas (o "piso de
  drift" não é superado pelos alvos conhecidos como mudados, na maioria das
  variantes).

### Decisão tomada

Adotar D como centralização de trabalho (`mu_t` = média não ponderada dos
centróides por tipo sobre V_ativo). **Não revisar §4.1** do documento
canônico com base só nisso -- a diferença A vs. D é pequena e não
significativa; a interpretação de `mu_t` em §4.1 pode precisar de uma nota
("ponderado por ocorrência" vs. "uniforme por tipo" produzem resultados
muito diferentes na prática e a escolha não é neutra), mas a definição
formal (§4, prova de invariância) permanece válida para qualquer das duas
interpretações de "média".

---

## 3. Fase 1.5 -- Go/no-go espectral (§7-9): decomposição em modos via SVD da matriz de coesão

### 3.1 O que o documento canônico propõe (resumo)

- **§7**: matriz de coesão `M_t(w)[v,v'] = P_t(w)[v] * P_t(w)[v'] *
  cos(e_hat_t(v), e_hat_t(v'))` para `v,v' em V_w = {v : P_t(w)[v] > tau}`.
  PSD, nunca materializada: `M = (D E)(D E)^T`, SVD de `D E` dá
  `lambda_i = sigma_i^2`, `a_i = u_i` (§7.5).
- **§8.2**: `tau` (e portanto `V_w`) é escolhido pelo critério de gap sobre
  os componentes positivos de `P_t(w)`, ordenados decrescentemente: gap
  relativo `h_i = (X_i - X_{i+1})/X_i`, aceita-se o maior gap se
  `h > gamma` (gamma~0.3).
- **§8.3**: `k` (número de modos semânticos) é escolhido pelo mesmo
  critério de gap, agora sobre os autovalores `lambda_i` de `M_t(w)`.
- Expectativa: para uma palavra polissêmica/em mudança (ex.: "plane" ->
  avião vs. superfície geométrica), `M_t(w)` deveria ter `k>=2` modos
  razoavelmente separados, cada um carregando um conjunto de tokens
  semanticamente coerente (§7.6, top tokens por `a_i[v]>0`); para uma
  palavra estável/monosêmica, `k=1` (ou indistinguível).

### 3.2 Palavras auditadas

- "Mudança esperada" (SemEval `truth.tsv`, `binary=1`): **plane_nn**,
  **graft_nn**.
- "Estável" (`binary=0`), usadas como controle de campo: **chairman_nn**,
  **tree_nn**, **ball_nn**, **face_nn**, **lane_nn**, **multitude_nn**.

(`chairman_nn`/`tree_nn` aparecem como `is_target: true` no JSON de saída
por motivos de rastreamento de uma auditoria qualitativa anterior, mas
contam como `binary=0`/estáveis no truth.tsv.)

### 3.3 Implementação

- `src/timeformers/gap_criterion.py`: `relative_gaps`, `select_gap_index`,
  `adjacent_gaps_valid` -- 14 testes unitários (`tests/test_gap_criterion.py`),
  incluindo o exemplo numérico de §8.3 do documento canônico
  (`[0.41,0.31,0.18,0.04,0.03]`, gamma=0.3 -> i*=3).
- `src/timeformers/semantic_modes.py`: `filter_support` (gap sobre
  positivos de `P_t(w)`), `filter_support_topn` (variante: restringe a
  top-N candidatos por `|P_t(w)[v]|` antes do gap), `cohesion_svd` (SVD de
  `D E`), `select_num_modes` (gap sobre autovalores), `top_tokens_per_mode`.
- `scripts/evaluate_semantic_modes_v2.py`: orquestra tudo, com
  sensibilidade a gamma (0.2/0.3/0.4), n_min (10/20/50), top_n (50/100/200)
  e bootstrap (subamostragem 80% de `V_w`, 30 repetições, overlap de
  Jaccard dos top tokens por modo).

A auditoria do codex (segunda opinião, ver `tmp/codex_semantic_modes_v2_nogo_review.md`)
confirmou que `relative_gaps`/`select_gap_index`/`filter_support`/
`cohesion_svd`/`select_num_modes` são fiéis a §8.2/§8.3/§7.5, e que
`centralized_embeddings`/`profile = support_embeddings @ word_embedding`
estão corretos (normas ~1.0 verificadas numericamente).

### 3.4 Resultado 1 -- `filter_support` puro (gap sobre componentes positivos de P_t(w))

Para as 8 palavras x 2 períodos (16 casos):

- `tau` da ordem de `1e-4` a `1e-3` (praticamente zero) em todos os casos.
- `|V_w|` entre ~4800 e ~5900 -- i.e., **quase todos os componentes
  positivos de `P_t(w)`** entram em `V_w` (o gap selecionado pelo critério
  é o que ocorre perto do cruzamento de zero de `P_t(w)`, não uma fronteira
  semântica no topo).
- `k=1` para a maioria dos casos, incluindo **plane_nn (t0 e t1) e graft_nn
  (t0 e t1)** -- as duas palavras com mudança semântica conhecida.
- `k=2` aparece apenas em **face_nn (t1)** e **multitude_nn (t0)** -- ambas
  palavras de controle ESTÁVEL, não nos alvos de mudança. Ou seja, quando
  o critério encontra "estrutura extra", ela aparece nos controles, não nos
  alvos.
- `n_min_sensitivity`: extremamente instável. `plane_nn` perde todo o
  suporte em `n_min=50` (`n_vw=0`); `multitude_nn` perde suporte em
  `n_min=20` (`n_vw=0`).
- bootstrap: `k_recovery_rate` em torno de 0.73-1.0 -- mas estabiliza em
  torno de um `k` que já é, na maioria dos casos, `k=1` degenerado (todo
  `V_w` quase igual a V_ativo positivo).

(arquivo completo: `outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/relational_profile_v2_phase1_5/modes.json`)

### 3.5 Resultado 2 -- `filter_support_topn` (top-N por `|P_t(w)[v]|`, N=100 e N=500)

Para as 8 palavras x 2 períodos, com **N=100 e N=500**:

- `tau = None`, `k = None`, `n_vw = 0` em **todos os 16 casos**, para
  ambos os valores de N.
- Ou seja: restringindo aos N candidatos mais correlacionados (em valor
  absoluto) com `w`, nenhum gap relativo excede `gamma=0.3` entre eles --
  o critério não encontra nenhuma estrutura, em nenhum caso.

(arquivos: `outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/relational_profile_v2_phase1_5_topn/modes.json`,
`..._topn500/modes.json`)

### 3.6 Diagnóstico -- forma do perfil P_t(w)

Inspeção direta de `P_t(w)` para `plane_nn`/t0/`mean_last_2` (ordenado
decrescente):

```
top 15:    [0.9456, 0.9424, 0.9030, 0.8990, 0.8975, 0.8944, 0.8928, 0.8826,
            0.8749, 0.8739, 0.8739, 0.8737, 0.8720, 0.8674, 0.8661]
bottom 5:  [-0.7146, -0.7216, -0.7296, -0.7571, -0.7649]
```

`P_t(w)[v]` decai de forma **suave e quase monotônica** de ~0.95 a ~-0.76
ao longo de ~11.6k tokens de V_ativo, com gaps relativos consecutivos no
topo da ordem de 0.003-0.05. Não há salto >30% em lugar nenhum perto do
topo. O único gap >30% (que `filter_support` sem topn encontra) ocorre
perto do cruzamento de sinal -- artefato do sinal, não uma fronteira
semântica.

Checagem adicional do codex (distribuição **par-a-par** de
`cos(e_hat_t(v), e_hat_t(v'))` para `v,v'` em V_ativo, plane_nn/t0):

```
quantis: [-0.81, -0.61, -0.49, -0.27, -0.05, 0.23, 0.67, 0.85, 1.0]
```

Essa distribuição PAR-A-PAR ainda é larga (não está toda concentrada perto
de 0) -- ou seja, o problema não é "concentração de medida" simples
(cossenos de pares aleatórios colapsando para 0). O problema é mais
específico: **a projeção de V_ativo sobre a direção `e_hat_t(w)` de QUALQUER
palavra w produz uma rampa suave de 1 dimensão**, sem fronteira discreta de
vizinhança -- mesmo que a geometria par-a-par subjacente tenha estrutura.

### 3.7 Resultado 3 -- Top-N positivo fixo (sem gap em P_t(w)), gap só nos autovalores de M_t(w)

Formulação recomendada pelo codex como mais próxima do espírito do
documento canônico, sem a ambiguidade de tau: tomar `V_w` = top-N
componentes POSITIVOS de `P_t(w)` (N fixo, sem critério de gap em `P_t(w)`),
construir `M_t(w)` sobre esse `V_w`, e aplicar o critério de gap (§8.3)
SÓ aos autovalores, para obter `k`.

Para as 8 palavras x 2 períodos x N em {50, 100, 200} (48 casos):

- **`k=1` em TODOS os 48 casos**, sem excecão.
- O primeiro autovalor domina o segundo por **10-30x** em todos os casos.
  Exemplos (N=100):
  - plane_nn/t0: `lambda = [53.39, 3.24, 2.69, 2.05]` (17x)
  - graft_nn/t1: `lambda = [38.98, 3.03, 2.82, 2.06]` (13x)
  - chairman_nn/t1: `lambda = [44.97, 4.38, 2.12, 1.66]` (10x)
  - face_nn/t1: `lambda = [33.45, 4.91, 3.14, 2.11]` (7x, o menor gap
    observado, mas ainda k=1 pelo critério gamma=0.3)

Ou seja: para QUALQUER palavra (mudada ou estável), em QUALQUER período, com
QUALQUER tamanho razoável de `V_w`, `M_t(w)` é dominado por um único modo.

(`/tmp/fixed_topn_check.py`, não versionado -- script ad-hoc de 70 linhas,
reproduzível a partir das definições do §1-2 deste documento)

### 3.8 Conclusão da Fase 1.5

**NO-GO para §7-9 (decomposição em modos semânticos via SVD da matriz de
coesão) neste regime** (d_model=128, 3 camadas de encoder, |V_ativo|~11.6k,
SemEval eng_lemma), confirmado por:

1. três formulações distintas do critério de seleção de `V_w`/`k`
   (gap em P_t(w) completo; gap em top-N por |P_t(w)|; gap só nos
   autovalores com top-N positivo fixo);
2. ausência de discriminação entre palavras com mudança semântica
   conhecida (plane_nn, graft_nn -- sempre k=1) e palavras de controle
   estável (k=2 espúrio aparece só nos controles, na formulação 1; k=1
   uniforme na formulação 3);
3. instabilidade severa a `n_min` na formulação 1 (suporte zera para
   plane_nn em n_min=50);
4. auditoria de implementação independente (codex) sem identificar erro de
   sinal/eixo/normalização/indexação;
5. diagnóstico geométrico direto: `P_t(w)` é uma rampa suave e
   quase-monotônica, sem clusters/fronteiras discretas, para qualquer `w`
   testado.

O problema não parece ser de calibração (gamma, n_min, tau, top_n) -- é
que a premissa de §7-9 (que `P_t(w)` ou `M_t(w)` tenham estrutura de
cluster multimodal detectável por gap espectral) não se sustenta nesta
representação.

---

## 4. Impacto sobre o restante do pipeline

- **§4-6 (centralização, perfil relacional, deslocamento)**: permanecem em
  uso. `Delta(w) = 1 - cos(P_t0(w), P_t1(w))` com centralização D é a
  métrica de trabalho atual (Fase 1), com sinal fraco mas não-nulo
  (spearman~0.124, p~0.47, n=37). O codex notou que essa métrica pode
  continuar informativa mesmo que `P_t(w)` individualmente seja uma rampa
  sem gaps, porque `Delta` mede a mudança da rampa inteira entre períodos
  -- mas o sinal já era fraco antes deste diagnóstico, então o NO-GO de
  §7-9 não piora a situação da Fase 1, apenas não a melhora como se
  esperava.
- **§8 (critério de gap)**: a implementação está correta e testada; o
  critério em si não está "quebrado" -- é de uso geral (qualquer sequência
  decrescente positiva). O que falhou é a aplicação específica em §7-9 a
  `P_t(w)`/autovalores de `M_t(w)` nesta representação.
- **§10-11 (persistência de modos via Hungarian matching, theta
  calibrado)**: dependem de `k>=2` e modos identificáveis de §7-9; ficam em
  suspenso até haver uma formulação de `V_w`/`M_t(w)` que produza `k>=2`
  de forma não-espúria nos alvos conhecidos.
- **§12-14 (validação, ANN/implementação)**: não testados ainda; a maior
  parte depende de §7-11 estarem funcionando.

---

## 5. Possíveis direções para a próxima rodada (não avaliadas ainda)

Sugestões levantadas (codex + discussão), para priorizar na próxima
rodada de re-projeto:

1. **Ablação de capacidade**: repetir a Fase 1.5 (formulação 3, top-N fixo
   + gap nos autovalores) com um encoder de d_model maior (256/512), para
   checar se o colapso em modo único é efeito de capacidade/dimensão do
   encoder ou algo estrutural da formulação. Mais barato que mudar a
   formulação, já que reaproveita o mesmo código.
2. **Gap relativo a um "piso nulo"**: aplicar o critério de gap não a
   `P_t(w)` ou aos autovalores brutos, mas à diferença em relação a uma
   distribuição de referência (cossenos de pares aleatórios, ou autovalores
   de `M_t(w)` para palavras de controle conhecidas como monosêmicas). Sai
   parcialmente da invariância simples de §8, mas pode recuperar contraste.
3. **WSI direto sobre nuvem de ocorrências**: em vez de decompor o perfil
   médio `P_t(w)`, aplicar clustering (ou a mesma decomposição espectral)
   diretamente sobre os embeddings de OCORRÊNCIAS individuais de `w` (já
   extraídos em `occurrence_vectors` no cache, para as 37 palavras-alvo do
   SemEval) -- mais próximo de WSI clássico, abandona a ideia de "modos
   emergem do perfil agregado".
4. **V_w por top-N fixo + k via outro critério** (não gap): já que
   `lambda_1 >> lambda_2` sempre, talvez `k` precise de um critério
   diferente do gap relativo (ex.: razão `lambda_2/lambda_1` acima de um
   limiar absoluto, ou teste de significância via bootstrap/permutação em
   vez de gap determinístico).
5. Não recomendado pelo codex (reabrir problemas que §4 já resolveu):
   perfil sem centralização (`cos(e_t(w), e_t(v))` bruto), distância
   euclidiana bruta, ou decompor o Gram de `V_w` sem o fator `P_t(w)` (isso
   seria PCA/comunidades locais, um método diferente de §7).

---

## 6. Adendo (2026-06-12) -- Diagnóstico estrutural e Passo 0 (APD + bimodalidade sobre ocorrências)

### 7.1 Contexto

Após este documento ser escrito, recebemos uma segunda revisão (externa,
fora do codex) que propõe uma explicação estrutural unificada para os
resultados das Fases 1 e 1.5, e um plano de próximos passos ordenado por
custo. Este adendo registra (a) o diagnóstico estrutural proposto, (b) o
"Passo 0" (teste discriminante barato) que executamos para decidir entre
os dois ramos do plano, e (c) o resultado e sua interpretação.

### 7.2 Diagnóstico estrutural proposto: "lambda_1 >> lambda_2 é quase garantido pela construção"

A revisão argumenta que o NO-GO da Fase 1.5 (§3 acima) e o sinal fraco da
Fase 1 (§2 acima) **não são dois problemas independentes -- são a mesma
causa em dois lugares**:

- `V_w` é selecionado por similaridade com uma ÚNICA direção `e_hat_t(w)`
  (o centróide de w, um único ponto). Vetores selecionados por proximidade
  a um vetor fixo formam, geometricamente, um cone em torno dele -- logo
  são similares entre si por construção, e o Gram desse conjunto
  necessariamente tem um primeiro autovetor apontando essencialmente para
  `e_hat_t(w)`. A centralização (variante D) reforça esse efeito (todos os
  `e_hat_t(v)` já compartilham a remoção do mesmo `mu_t`).
- Ou seja: **a multimodalidade é destruída no passo zero**, quando o
  conjunto de ocorrências de w é colapsado num único centróide `e_t(w)`.
  Um único vetor projetado sobre o vocabulário produz necessariamente uma
  função suave do ângulo -- exatamente a "rampa" observada em §3.6. A
  informação sobre múltiplos sentidos de w, se existir, vive na NUVEM de
  ocorrências de w, não no seu centróide.
- Consequência para a Fase 1: `Delta(w) = 1 - cos(P_t0(w), P_t1(w))` é
  calculado sobre o perfil INTEIRO (~11.6k componentes). Para uma palavra
  como "plane", o que muda entre períodos é a vizinhança LOCAL no topo do
  perfil (surge o sentido aeronáutico); os ~11k componentes restantes da
  rampa quase não mudam e diluem esse sinal local no cosseno global. O
  cosseno do perfil inteiro é, por essa leitura, a métrica errada para
  capturar uma mudança que é por natureza local.
- Consequência adicional: `changed_above_stable_p95 ~= 0` (§2, tabela)
  pode ser efeito de RUÍDO amostral em `mu_t`/centróides de vizinhos raros
  (n_min=10), não de ausência de sinal -- o piso de drift dos estáveis
  fica inflado por variância de amostragem, competindo com o drift
  semântico real.

Implicação para o §5 original (direções para a próxima rodada): as
direções 1, 2 e 4 (ablação de capacidade, gap relativo a piso nulo, k via
outro critério) **calibram um objeto (`M_t(w)` a partir do perfil agregado)
que é rank-1 por construção** -- não atacam a causa. A direção 3 (WSI sobre
a nuvem de ocorrências, ANTES de agregar em perfil) é, por essa leitura, a
única que ataca a causa diretamente: clusterizar primeiro (onde a
multimodalidade genuína pode existir), e só depois construir um perfil
relacional por modo (`P_t(w, j)[v] = cos(e_hat_t(w, j), e_hat_t(v))` para
cada modo `j`).

### 7.3 Passo 0 -- teste discriminante (decide entre os dois ramos)

Antes de investir na reformulação de §7-9 (direção 3) ou na ablação de
capacidade (direção 1), a revisão propõe um teste barato que decide qual
delas é prioritária, usando `occurrence_vectors`/`occurrence_targets` já
presentes no cache (27173 ocorrências das 37 palavras-alvo, sem
reextração):

1. **APD(w)** ("Average Pairwise Distance", baseline clássico do
   SemEval-2020): `APD(w) = média sobre (i em ocorrências_t0(w), j em
   ocorrências_t1(w)) de (1 - cos(v_i, v_j))`. Calculado diretamente sobre
   ocorrências individuais -- sem nunca passar por um centróide. Se APD
   correlacionar bem com `truth.tsv`, o encoder presta e o problema é
   especificamente a agregação por centróide (-> prioriza direção 3). Se
   APD também não correlacionar, o encoder de d_model=128/3 camadas
   provavelmente não codifica a distinção de sentidos de forma utilizável,
   e nenhuma reformulação de §7-9 vai resolver isso por si só (-> prioriza
   direção 1, ablação de capacidade).

2. **Teste de bimodalidade**: para a nuvem de ocorrências de w (em t0, t1,
   e t0+t1 combinados), ajustar GMM com k=1 e k=2 componentes
   (covariância diagonal, embeddings L2-normalizados) e comparar BIC
   (`delta_bic = BIC(k=1) - BIC(k=2)`, positivo = k=2 preferido) e
   silhouette da partição k=2. Se nem plane_nn/graft_nn (mudança conhecida)
   mostrarem `delta_bic`/silhouette consistentemente maiores que
   chairman_nn/tree_nn (controles estáveis), a distinção de sentidos não é
   separável nesta representação por clustering simples.

Implementação: `scripts/evaluate_occurrence_apd_v2.py`
(`occurrences_for_target`, `average_pairwise_distance`,
`bimodality_check`, `evaluate`). Roda sobre
`outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/hidden_relational_profiles/cache/theta{0,1}_d{0,1}.pt`,
camadas `layer_2` e `mean_last_2`, mesmas 37 palavras-alvo do SemEval e
mesmo `truth.tsv` da Fase 1.

### 7.4 Resultado -- APD

| Layer | Spearman (p) | ROC-AUC | AP |
|---|---|---|---|
| layer_2 | 0.130 (p=0.444) | 0.592 | 0.615 |
| mean_last_2 | 0.127 (p=0.452) | 0.577 | 0.600 |

(arquivo completo: `outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/occurrence_apd_phase_passo0/occurrence_apd_results.json`,
linhas por alvo em `apd_rows_{layer}.csv`)

**APD (occurrence-level, sem centróide) é estatisticamente indistinguível
do `Delta` baseado em centróide da Fase 1** (0.130/0.577 vs. 0.124/0.601,
ambos n=37, p~0.45). Bypassar completamente a agregação em centróide não
recuperou sinal.

### 7.5 Resultado -- bimodalidade (GMM k=1 vs k=2)

`delta_bic` (positivo = k=2 preferido) e silhouette da partição k=2, para
`plane_nn`/`graft_nn` (mudança conhecida) vs. `chairman_nn`/`tree_nn`
(controles estáveis), camada `mean_last_2`:

| Palavra | Período | n | delta_bic | silhouette (k=2) |
|---|---|---|---|---|
| plane_nn | t0 | 278 | 4 747 | 0.072 |
| plane_nn | t1 | 792 | 14 262 | 0.104 |
| plane_nn | combined | 1070 | 29 085 | 0.268 |
| graft_nn | t0 | 119 | 4 207 | 0.295 |
| graft_nn | t1 | 109 | 1 088 | 0.127 |
| graft_nn | combined | 228 | 9 351 | 0.287 |
| chairman_nn | t0 | 147 | 3 096 | 0.246 |
| chairman_nn | t1 | 683 | 26 163 | 0.254 |
| chairman_nn | combined | 830 | 31 209 | 0.160 |
| tree_nn | t0 | 2322 | 36 602 | 0.099 |
| tree_nn | t1 | 1596 | 30 467 | 0.096 |
| tree_nn | combined | 3918 | 129 581 | 0.192 |

(layer_2 mostra o mesmo padrão; arquivo completo na mesma saída JSON)

Observações:

- `delta_bic` é GRANDE e positivo para TODAS as palavras, incluindo as
  estáveis -- k=2 é "preferido" por BIC universalmente. Com n na ordem de
  centenas/milhares e d=128 (covariância diagonal), o ganho de
  log-verossimilhança de adicionar um segundo componente supera
  facilmente a penalidade de BIC, mesmo para variação de densidade local
  genérica sem relação com sentido. **`delta_bic` não é informativo aqui.**
- Mais diretamente: **`tree_nn` (controle estável) tem o MAIOR
  `delta_bic`** (129 581 combinado, vs. 29 085 de plane_nn e 9 351 de
  graft_nn) -- o oposto do que se esperaria se `delta_bic` rastreasse
  estrutura de sentido.
- `silhouette(k=2)` também não discrimina: `graft_nn`/t0 (0.295, mudança
  conhecida) e `chairman_nn`/t0 (0.246, estável) são comparáveis;
  `chairman_nn`/t1 (0.254) é maior que `plane_nn`/t1 (0.104).
- Conclusão: GMM(k=1 vs k=2)/BIC + silhouette, aplicado à nuvem de
  ocorrências bruta, **não encontra estrutura de sentido específica de
  palavra** -- encontra variação de densidade local genérica, que existe
  em qualquer nuvem de centenas/milhares de pontos em d=128 e não
  correlaciona com `truth.tsv`.

### 7.6 Interpretação revisada

Os dois resultados do Passo 0 juntos **deslocam o diagnóstico** em relação
ao proposto em 7.2:

- Não é o caso "encoder ~= 0" extremo (APD não é zero/negativo; AP~0.60 é
  acima do acaso de 37/(37*ratio), embora spearman não seja significativo).
- Mas também não é o caso "aggregation artifact puro": se a agregação em
  centróide fosse a causa principal do sinal fraco, `APD` (que nunca passa
  por um centróide) deveria ter performado MELHOR que `Delta` da Fase 1.
  Performou igual (0.130 vs 0.124, mesma faixa de p).
- O teste de bimodalidade não mostra NENHUMA estrutura específica de
  palavra correlacionada com `truth.tsv` -- nem mesmo nas palavras com
  mudança conhecida, e a palavra com maior "preferência por k=2" é
  justamente um controle estável.

**Leitura proposta:** ao contrário da hipótese de 7.2 (multimodalidade
destruída na agregação), a evidência do Passo 0 sugere que, **neste regime
(d_model=128, 3 camadas, treino contínuo de MLM nos dados do SemEval
eng_lemma), as representações contextuais do encoder não carregam um
sinal forte e consistentemente recuperável da distinção de sentido que o
SemEval-2020 Task 1 alvo -- independentemente de como esse sinal é
agregado** (centróide médio, perfil relacional, ou distância par-a-par
entre ocorrências brutas). Isso não exclui que HAJA algum sinal (AP~0.60,
spearman~0.13 positivo e consistente entre 3 métricas independentes
-- Delta da Fase 1, APD, e a métrica A_reference_mean da v1 -- é mais
forte que ruído puro), mas sugere que o "teto" desse sinal, nesta
configuração, está perto de onde já estamos.

### 7.7 Prioridade revisada para a próxima rodada

Com base no Passo 0, a prioridade entre as direções do §5 muda:

1. **[Prioridade alta] Ablação de capacidade (d_model maior, 256/512)**
   -- já listada como direção 1 do §5, agora com motivação mais forte: o
   Passo 0 é consistente com "teto de capacidade/treino do encoder" mais
   do que com "problema de formulação do perfil/agregação". Se um encoder
   maior também platô em spearman~0.13, a limitação provavelmente está no
   sinal de treino contínuo de MLM em si (quantidade de dados, objetivo),
   não na dimensão.
2. **[Prioridade média, independente de (1)] Robustez estatística**: com
   n=37, `spearman` precisa de ~0.33 para p<0.05. Repetir Fase 1 e o Passo
   0 nas outras línguas do SemEval-2020 (alemão, sueco, latim -- n~120
   combinado) e reportar intervalos de bootstrap em vez de só p-valores,
   antes de descartar qualquer variante por "não significativa".
3. **[Reclassificada de "ataca a causa" para "vale tentar, custo baixo"]
   Direção 3 (WSI sobre nuvem de ocorrências, modos antes do perfil)**:
   o Passo 0 (GMM/BIC ingênuo) não encontrou estrutura, mas isso não
   esgota a hipótese -- métodos de WSI mais robustos (ex.: clustering
   sobre uma redução de dimensão prévia, ou sobre `layer_1`/camadas
   individuais em vez de `mean_last_2`, ou normalizando por remoção de
   anisotropia "all-but-the-top" antes do clustering) ainda podem revelar
   estrutura que GMM diagonal ingênuo não vê. Vale tentar, mas não é mais
   o passo "que decide tudo" -- é uma entre várias ablações de
   pré-processamento.
4. **[Direção 2 da Fase 1, independente da decomposição em modos]
   Conserto da métrica Delta**: três variantes baratas, ainda sobre a
   infraestrutura atual, valem ser testadas independentemente do resultado
   da ablação de capacidade:
   - **Delta local**: `1 - cos` apenas sobre a união dos top-N componentes
     de `P_t0(w)` e `P_t1(w)` (N~50-200), em vez do perfil inteiro --
     remove a diluição pela "rampa".
   - **Overlap de vizinhos**: Jaccard ou RBO (Rank-Biased Overlap) entre os
     top-k vizinhos de w em t0 e t1 -- baseline forte na literatura de LSC
     porque ignora magnitude e usa só a estrutura ordinal do topo.
   - **All-but-the-top**: remover as 1-3 primeiras componentes principais
     dos embeddings centralizados de cada período antes de calcular
     `P_t(w)` -- a rampa com cossenos de topo em ~0.95 (§3.6) é sugestiva
     de anisotropia residual que a centralização (§4) por si só não
     remove.
   - Também: aumentar `n_min` dos VIZINHOS (não de w) para reduzir o ruído
     amostral de centróides raros que infla `stable_delta_p95`.

### 7.8 Arquivos novos deste adendo

- `scripts/evaluate_occurrence_apd_v2.py` -- Passo 0 (APD + bimodalidade)
- `outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/occurrence_apd_phase_passo0/{occurrence_apd_results.json,apd_rows_layer_2.csv,apd_rows_mean_last_2.csv}`

### 7.9 Correção (segunda revisão, mesma data) -- as 3 métricas em ~0.12-0.13 não são evidências independentes

Releitura crítica de 7.6: `Delta` (Fase 1), `APD` (Passo 0) e
`A_reference_mean` (v1) são computados sobre as **mesmas** representações
contextuais e (no caso de Delta/APD) sobre as **mesmas** ocorrências --
divergem só na agregação (centróide global vs. centróide de referência vs.
par-a-par). Concordância entre elas mede principalmente **estabilidade da
métrica frente à agregação**, não acúmulo de evidência independente sobre o
encoder. A frase final de 7.6 ("é mais forte que ruído puro") deve ser lida
como "o sinal de spearman~0.12-0.13 é robusto à escolha de agregação", e não
como "três fontes independentes apontam no mesmo sentido".

### 7.10 Segundo teste barato -- `APD_ratio` e associação cluster x período

Duas extensões adicionais ao Passo 0, ainda sobre o mesmo cache de
ocorrências (sem reextração), propostas para des-confundir dois problemas
levantados na revisão de 7.6:

1. **`APD_ratio(w) = APD_inter / mean(APD_intra_t0, APD_intra_t1)`** --
   `APD_intra_t` = distância par-a-par média (1 - cos) entre ocorrências do
   *mesmo* período. Palavras naturalmente difusas (ex.: `tree_nn`, com 2322
   + 1596 ocorrências) têm `APD_inter` alto só porque têm `APD_intra` alto
   em ambos os períodos -- `APD_ratio` normaliza por essa dispersão de
   fundo.

2. **Associação cluster x período** -- em vez de testar se a nuvem
   combinada (t0+t1) de uma palavra "prefere" k=2 (teste de densidade do
   Passo 0, que deu `delta_bic` positivo para TODAS as palavras), particiona
   a nuvem combinada em 2 clusters via k-means e mede a associação entre o
   rótulo de cluster e o rótulo de período via **normalized mutual
   information (NMI)** e **adjusted mutual information (AMI)**. Esse teste
   é imune ao artefato do Passo 0: variação de densidade genérica não tem
   motivo para se alinhar com período, mas uma troca de sentido sim
   (intuição confirmada anecdoticamente em 7.x: silhouette da nuvem
   combinada plane=0.268, graft=0.287 > chairman=0.160, tree=0.192).

Implementado em `scripts/evaluate_cluster_period_apd_ratio_v2.py`,
reaproveitando `occurrences_for_target`/`average_pairwise_distance` do
Passo 0. k-means com `n_init=10`, seed=0.

**Resultado -- correlação com `truth.tsv` (n=37):**

| Layer | Métrica | Spearman (p) | ROC-AUC | AP |
|---|---|---|---|---|
| layer_2 | `apd_ratio` | -0.104 (p=0.538) | 0.482 | 0.468 |
| layer_2 | `nmi`/`ami` | 0.027 (p=0.872) | 0.455 | 0.397 |
| mean_last_2 | `apd_ratio` | -0.058 (p=0.733) | 0.491 | 0.515 |
| mean_last_2 | `nmi`/`ami` | 0.048 (p=0.778) | 0.543 | 0.458 |

Nenhuma das duas extensões discrimina `truth.tsv` -- `apd_ratio` fica
*abaixo* do acaso (AUC<0.5) e `nmi`/`ami` ficam dentro do ruído (AUC
0.45-0.54). À primeira vista, isso parece fechar a porta para a direção 3
(WSI sobre ocorrências) tão quanto o Passo 0 original.

**Mas os valores absolutos de NMI por palavra contam outra história:**

| Palavra (esperado) | layer_2 NMI | mean_last_2 NMI | n_t0 / n_t1 |
|---|---|---|---|
| `chairman_nn` (estável) | 0.002 | 0.965 | 147 / 683 |
| `graft_nn` (mudou) | 0.423 | 0.964 | 119 / 109 |
| `plane_nn` (mudou) | 0.988 | 0.988 | 278 / 792 |
| `tree_nn` (estável) | 1.000 | 0.997 | 2322 / 1596 |

Em `mean_last_2`, um corte k-means=2 da nuvem combinada recupera o rótulo de
período quase perfeitamente para **praticamente qualquer palavra**, mudada
ou estável (`chairman_nn`=0.965, `tree_nn`=0.997). Em `layer_2` o padrão é
mais variável, mas `tree_nn` (controle estável, 3918 ocorrências) ainda
atinge NMI=1.000 -- o caso de MAIOR associação cluster x período de toda a
amostra.

### 7.11 Interpretação revisada (2) -- eixo global de "época" domina o espaço de ocorrências em `mean_last_2`

A combinação de 7.10 com o `delta_bic` de 7.5 (onde `tree_nn` também tinha o
maior `delta_bic` combinado, 129581) converge para uma leitura mais forte:
existe, em `mean_last_2`, um **eixo dominante que separa "ocorrência veio de
t0" vs. "ocorrência veio de t1"**, compartilhado por praticamente todas as
palavras -- mudadas e estáveis. Esse eixo é forte o suficiente para que
qualquer particionamento k=2 (GMM ou k-means) o capture trivialmente antes
de qualquer estrutura específica de sentido, o que explica por que:

- `delta_bic` é positivo (e enorme) para todas as palavras (7.5) -- k=2
  sempre "ganha" porque k=2 = "separar por período", não porque há dois
  sentidos.
- NMI(cluster, período) é alto para quase todas as palavras (7.10) -- pelo
  mesmo motivo.
- Nenhuma das duas métricas discrimina `truth.tsv`, porque a separação por
  período não é específica de mudança semântica -- é uma propriedade global
  do espaço.

**Isso refina (não contradiz) a hipótese estrutural de 7.2.** A dominância
de "lambda_1" não vem de uma rampa geométrica artefatual da agregação
P_t(w), mas de um **eixo de "época"/checkpoint** que provavelmente reflete
o próprio treino contínuo (theta0 -> theta1 via MLM continuado): mudanças de
escala/norma, de calibração da LayerNorm final, ou de distribuição de
contexto entre os dois recortes temporais do corpus, que se propagam para
TODAS as representações em `mean_last_2`, não só para as palavras-alvo. A
centralização por mu_t (subtração da média por período, §4) remove um
deslocamento aditivo compartilhado, mas se o efeito do checkpoint for mais
parecido com uma rotação/reescala anisotrópica do que com um shift puro, ele
sobrevive à centralização e domina tanto P_t(w) (Fase 1.5) quanto a nuvem de
ocorrências brutas (Passo 0 / 7.10).

Achado adicional relevante: a discrepância `layer_2` (NMI=0.002 para
`chairman_nn`) vs. `mean_last_2` (NMI=0.965 para a mesma palavra) sugere que
esse eixo de época se acumula/intensifica nas camadas finais -- consistente
com a hipótese de anisotropia "all-but-the-top" já listada em 7.7 (item 4),
mas agora com motivação mais concreta: o candidato a remover não é uma
anisotropia genérica do espaço de embeddings, e sim especificamente um eixo
correlacionado com o checkpoint/período de treino.

### 7.12 Prioridade revisada (3)

Com 7.10/7.11, a ordem de 7.7 muda de novo:

1. **[Nova prioridade 1, custo baixo, minutos]** Antes de qualquer ablação
   de capacidade ou Passo 0' com BERT pré-treinado: **identificar e remover
   o eixo de época em `mean_last_2`**. Procedimento direto sobre o cache
   atual: (a) treinar uma regressão logística / LDA simples para prever
   período a partir das ocorrências combinadas (todas as palavras juntas,
   não por palavra) -- isso estima a direção do eixo de época globalmente;
   (b) projetar essa direção fora dos embeddings (remoção tipo
   "all-but-the-top" mas dirigida, não pelas top-PCs genéricas); (c)
   re-rodar Passo 0 (APD/Delta) e o teste de 7.10 sobre os embeddings
   corrigidos. Se o spearman de `Delta`/`APD` subir e a NMI cluster x
   período cair para a maioria das palavras estáveis (mantendo-se alta só
   para as que de fato mudaram), confirma o diagnóstico e dá um "Delta
   corrigido" de baixo custo -- sem treino novo.
2. **[Mantido, mas agora como verificação cruzada] Passo 0' -- teto de
   oráculo com BERT pré-treinado**: ainda útil para saber se esse eixo de
   época é um artefato do *treino contínuo from-scratch* (theta0/theta1 são
   checkpoints do mesmo modelo pequeno) ou algo que apareceria mesmo com um
   encoder pré-treinado grande rodando sobre os mesmos dois recortes de
   corpus. Mas deixa de ser bloqueante para (1).
3. **[Mantido de 7.7] Robustez estatística multi-língua (n~120)** e
   **[Mantido de 7.7, item 4] consertos de Delta** (Delta local, RBO,
   all-but-the-top) seguem em paralelo -- o item "all-but-the-top" de 7.7
   agora é um caso específico de (1).
4. **[Rebaixada] Ablação de capacidade (d_model 256/512)**: só faz sentido
   *depois* de (1), porque um eixo de época dominante em `mean_last_2`
   afetaria igualmente um encoder maior, e confundiria de novo a leitura de
   "capacidade" com "checkpoint drift não removido".

### 7.13 Arquivos novos deste segundo adendo

- `scripts/evaluate_cluster_period_apd_ratio_v2.py` -- `APD_ratio` +
  associação cluster x período (NMI/AMI)
- `outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/cluster_period_apd_ratio_phase/{cluster_period_apd_ratio_results.json,rows_layer_2.csv,rows_mean_last_2.csv}`

### 7.14 Terceiro adendo (mesma data) -- de-confundindo a origem do eixo de época: grade 2x2 do cache

A revisão seguinte a 7.13 apontou que o cache já contém a grade completa
`theta{0,1}_d{0,1}.pt` -- cada checkpoint (theta0, theta1) aplicado a CADA
recorte de corpus (d0=1810-1860, d1=1960-2010), 27173 e 31174 ocorrências
respectivamente. Isso permite separar duas fontes possíveis para o eixo de
época de 7.11 sem nenhuma extração nova:

- **(A) drift de checkpoint** -- theta0 -> theta1 via MLM continuado muda a
  geometria das representações independentemente do conteúdo.
- **(B) separabilidade do corpus** -- d0 e d1 têm gênero/ortografia/tópico
  tão diferentes que mesmo um encoder fixo separaria as duas épocas.

Implementado em `scripts/diagnose_period_axis_v2.py`. **Teste 1**: NMI
(cluster k-means=2, nuvem combinada) por palavra, para 4 combinações:

| Combinação | Isola | mean_last_2 mean NMI | mean_last_2 median NMI |
|---|---|---|---|
| `theta0_d0` vs `theta1_d1` (diagonal, original) | A+B | 0.805 | 0.982 |
| `theta1_d0` vs `theta1_d1` (encoder fixo) | B (corpus) | 0.029 | 0.012 |
| `theta0_d0` vs `theta1_d0` (dados fixos, d0) | A (checkpoint) | 0.857 | 1.000 |
| `theta0_d1` vs `theta1_d1` (dados fixos, d1) | A (checkpoint) | 0.755 | 0.995 |

**Resultado limpo**: com o encoder fixo (linha 2), o corpus por si só é
quase NÃO separável (NMI médio 0.029, mediana 0.012 -- na prática zero,
exceto `graft_nn`~0.20). Com os dados fixos (linhas 3-4), o checkpoint por
si só já reproduz quase todo o NMI da diagonal (0.86/0.76 vs 0.80). **O eixo
de época de 7.11 é, esmagadoramente, drift de checkpoint (A), não
separabilidade de corpus/registro (B).** As mesmas sentenças de d0,
codificadas por theta0 vs theta1, já são quase perfeitamente separáveis por
um corte k-means=2 -- a mudança de dados (d0 vs d1) contribui muito pouco
além disso.

### 7.15 Teste 2 -- a recentralização aditiva por "lado" remove a maior parte do eixo, mas não move o Spearman

Sobre a combinação diagonal (`theta0_d0` vs `theta1_d1`, a usada no Passo 0
original), subtrai-se de cada ocorrência a média global (sobre TODAS as
palavras daquele lado), renormaliza-se, e recalcula-se NMI e `APD`.

- `shift_cosine(mean_a, mean_b)` ~= 0.83-0.85 -- as médias globais dos dois
  lados são próximas em direção, mas não idênticas.
- `shift_norm_ratio` ~= 0.52-0.57 -- a diferença entre as médias tem
  magnitude comparável (~metade) à da própria média -- um shift aditivo
  substancial.
- Após recentralizar: `mean_nmi` cai de 0.805 -> 0.312 (mean_last_2) e de
  0.646 -> 0.247 (layer_2); `median_nmi` cai de 0.982 -> 0.088
  (mean_last_2) e de 0.908 -> 0.048 (layer_2).
- **Mas a discriminação de `APD` recentralizado contra `truth.tsv` não
  melhora**: spearman 0.100 (mean_last_2) / 0.093 (layer_2), praticamente
  igual ao APD original (0.127 / 0.130) -- dentro do ruído.

**Leitura**: um shift aditivo simples (diferença de médias globais por
"lado") explica a MAIOR PARTE do eixo de época para a maioria das palavras
(mediana de NMI cai de ~0.98 para ~0.05-0.09 -- ou seja, para a palavra
mediana o corte k-means deixa de recuperar o período quase totalmente). Isso
é uma correção barata e bem definida. **Porém ela não destrava o sinal
fraco de `Delta`/`APD`** -- sugerindo que "remover o eixo de época" (útil
para Fase 1.5, onde ele provavelmente infla `lambda_1`) e "destravar o
spearman~0.12-0.13 contra `truth.tsv`" são **dois problemas distintos**, não
o mesmo problema visto de duas formas.

Caso notável: `tree_nn` (controle estável, 3918 ocorrências) continua com
NMI alto mesmo após recentralização (0.90 layer_2 / 0.80 mean_last_2) --
um residual não-aditivo (rotação/reescala) sobrevive para palavras de alta
frequência, mas isso não se traduz em sinal contra `truth.tsv`.

### 7.16 Interpretação revisada (3) e conexão com a centralização da Fase 1

O resultado de 7.15 retroativamente valida a centralização por `mu_t` da
Fase 1 (§4): ela já faz, no nível de CENTRÓIDE e sobre `V_ativo`,
essencialmente a mesma correção aditiva que o Teste 2 faz no nível de
OCORRÊNCIA e sobre todo o vocabulário -- e, de fato, a Fase 1 com
centralização D já tem spearman~0.12, no mesmo patamar do `APD`
recentralizado (~0.10) e do `APD` original (~0.13). Ou seja: **a correção
aditiva (shift de médias) já está, em grande parte, "embutida" nos
resultados que já temos** -- não é uma alavanca nova.

O que sobra (componente não-aditivo, ilustrado por `tree_nn` em 7.15) é
mais compatível com uma correção tipo "all-but-the-top" (remover as top-k
componentes principais de variância, não só a média) do que com uma única
direção -- mas dado que mesmo a correção aditiva completa não moveu o
spearman, a expectativa para "all-but-the-top" deve ser modesta: é mais
provável que ajude a Fase 1.5 (reduzir a dominância de `lambda_1` para
viabilizar k>=2) do que a métrica `Delta`/`APD` em si.

### 7.17 Prioridade revisada (4)

1. **[Mantido, redefinido] Fase 1.5**: aplicar a correção aditiva de 7.15
   (ou all-but-the-top com poucas componentes, estimado fora das
   palavras-alvo per 7.12) ANTES do critério de gap (§8) sobre `P_t(w)`, e
   reavaliar se `lambda_1/lambda_2` deixa de ser 10-30x para
   `plane_nn`/`graft_nn`. Este é o teste mais direto de "o eixo de época é
   a causa do NO-GO da Fase 1.5" -- e agora com uma correção concreta e
   barata para aplicar (~minutos), em vez de uma hipótese.
2. **[Rebaixada para Delta/APD, mas ainda relevante] Passo 0' (BERT
   pré-treinado)**: como o drift de checkpoint (A) é a causa dominante e é,
   por definição, AUSENTE em um encoder pré-treinado único (sem
   continuação), Passo 0' deixa de ser um desempate "regime vs encoder" e
   passa a ser um teste limpo de (B) -- separabilidade de
   sentido/corpus sem nenhum drift de checkpoint. Se BERT também não
   discriminar `truth.tsv` bem acima de ~0.13, reforça que o teto está nos
   dados/tarefa (SemEval eng_lemma, n=37) mais do que no encoder.
3. **[Mantido de 7.7/7.12] Multi-língua (n~120)** e **consertos de Delta**
   (Delta local, RBO) seguem em paralelo -- mas com expectativa recalibrada
   por 7.15: provavelmente não vêm do eixo de época, então valem
   independentemente do resultado de (1).
4. **Ablação de capacidade (d_model 256/512)**: mantém-se como último passo,
   mas agora com um motivo mais específico para esperar pouco efeito: se
   nem a correção aditiva completa (7.15) nem a mudança de agregação (Passo
   0) moveram o spearman, um encoder maior treinado no MESMO regime
   provavelmente herda a mesma limitação de dados/objetivo.

### 7.18 Arquivos novos deste terceiro adendo

- `scripts/diagnose_period_axis_v2.py` -- grade 2x2 checkpoint x corpus
  (Teste 1) + recentralização aditiva por lado (Teste 2)
- `outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/period_axis_diagnosis/{period_axis_diagnosis.json,test1_*.csv,test2_recentered_*.csv}`

### 7.19 Quarto adendo (2026-06-12) -- medindo com o "encoder fixo" (Tarefa 1)

**Ideia em palavras simples:** até aqui, sempre comparamos "palavra em
1850 medida pelo modelo theta0" com "palavra em 2000 medida pelo modelo
theta1" (a configuração "diagonal"). O problema (7.14) é que theta0 e
theta1 são dois modelos ligeiramente diferentes -- então parte da diferença
que medimos pode ser "o modelo mudou", não "a palavra mudou".

A correção é simples: pegar **um único modelo** (theta0 OU theta1) e usá-lo
para medir as ocorrências de 1850 E as de 2000. Assim qualquer diferença
medida só pode vir da palavra/do texto, nunca do modelo.

Implementado em `scripts/evaluate_fixed_encoder_v2.py`, reaproveitando
`relational_profile`/`type_uniform_mean` (Fase 1) e
`average_pairwise_distance` (Passo 0). Repete duas métricas já conhecidas
(`APD` e `Delta` do perfil relacional), agora com modelo fixo:

| Modelo fixo | Layer | APD spearman (p) | APD AUC / AP | Delta spearman (p) | Delta AUC / AP |
|---|---|---|---|---|---|
| theta0 (só viu textos de 1850) | layer_2 | 0.016 (p=0.93) | 0.586 / 0.624 | -0.057 (p=0.74) | 0.548 / 0.489 |
| theta0 | mean_last_2 | 0.048 (p=0.78) | 0.568 / 0.608 | 0.045 (p=0.79) | 0.583 / 0.578 |
| theta1 (viu 1850 e 2000 no treino contínuo) | layer_2 | **0.204 (p=0.23)** | 0.604 / 0.635 | 0.141 (p=0.41) | 0.583 / 0.576 |
| theta1 | mean_last_2 | **0.202 (p=0.23)** | **0.619 / 0.663** | 0.164 (p=0.33) | 0.589 / 0.586 |

Para comparação, a configuração "diagonal" original tinha `APD`
spearman~0.13 (AUC~0.59) e `Delta` (Fase 1, variante D) spearman~0.12
(AUC~0.60).

**Resultado:** com `theta0` fixo o sinal praticamente desaparece (spearman
~0.02-0.05, perto do acaso). Com `theta1` fixo, `APD` sobe para
spearman~0.20 e AP~0.66 -- o melhor resultado de toda a investigação até
agora, embora ainda não estatisticamente significativo com n=37
(precisaria de spearman~0.33 para p<0.05).

**Interpretação:** isto bate com o cenário intermediário previsto: o
desenho "diagonal" não estava simplesmente errado a ponto de o encoder fixo
"explodir" o sinal (não foi de ~0.13 para ~0.40+), mas também não é
indiferente -- `theta1` fixo é claramente melhor que a diagonal, e MUITO
melhor que `theta0` fixo. Faz sentido: `theta1` "viu" os dois recortes de
corpus durante o treino contínuo, então é o modelo mais "neutro" para
comparar os dois períodos; `theta0` nunca viu os textos de 2000, então
mede-os mal. **Conclusão prática:** medições futuras (Fase 1, Passo 0,
Fase 1.5) devem usar `theta1` aplicado aos dois corpora (`theta1_d0` vs
`theta1_d1`) como configuração padrão, em vez da diagonal. O `NMI`
(separação cluster x período) continua sem discriminar `truth.tsv` em
nenhum dos dois casos -- não é uma métrica útil de mudança, mesmo sob
encoder fixo.

### 7.20 Arquivos novos deste quarto adendo

- `scripts/evaluate_fixed_encoder_v2.py` -- APD/Delta/NMI com encoder fixo
  (theta0 ou theta1) sobre d0 vs d1
- `outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/fixed_encoder_eval/{fixed_encoder_metrics.json,rows_theta{0,1}_{layer_2,mean_last_2}.csv}`

### 7.21 Tarefa 3 -- protótipo "modos primeiro, perfil depois" (substitui §7-9)

**Ideia em palavras simples:** a Fase 1.5 (§3, NO-GO) tentou achar "modos de
sentido" dentro do PERFIL (a lista de cossenos de uma palavra contra o
vocabulário) -- e isso falhou porque o perfil já chega achatado (sempre um
modo só). Aqui a ordem é invertida: primeiro agrupa-se a NUVEM DE
OCORRÊNCIAS de cada palavra (sob `theta1`, o encoder fixo de 7.19) em 2-5
grupos ("modos"); cada grupo só DEPOIS é descrito pelo seu próprio perfil
(top-15 vizinhos no vocabulário).

Implementado em `scripts/prototype_modes_first_v2.py`, para as 8 palavras
auditadas (4 mudaram: `plane_nn`, `graft_nn`, `chairman_nn`, `tree_nn`; 4
controles estáveis: `ball_nn`, `face_nn`, `lane_nn`, `multitude_nn`).
Antes de agrupar, removem-se as 2 direções principais (PCA) estimadas sobre
a nuvem combinada de TODAS as OUTRAS 7 palavras auditadas (proxy do "eixo de
época residual" de 7.15-7.16, sem usar as próprias ocorrências de w). O
número de modos `k` é escolhido por silhouette (grade k=2..5).

**Resultado -- número de modos e JSD(d0, d1) (divergência entre a
distribuição de modos em d0 e em d1; quanto maior, mais a palavra "trocou de
modo predominante" entre os dois períodos):**

| Palavra (esperado) | k escolhido | JSD(d0,d1) | silhouette do k escolhido |
|---|---|---|---|
| `graft_nn` (mudou) | 2 | **0.473** | 0.204 |
| `chairman_nn` (estável) | 3 | 0.367 | 0.249 |
| `multitude_nn` (estável) | 4 | 0.320 | 0.242 |
| `lane_nn` (estável) | 4 | 0.195 | 0.143 |
| `ball_nn` (estável) | 5 | 0.137 | 0.144 |
| `face_nn` (estável) | 4 | 0.125 | 0.154 |
| `plane_nn` (mudou) | 2 | 0.095 | 0.125 |
| `tree_nn` (estável) | 2 | **0.028** | 0.192 |

**Olhando os vizinhos de cada modo** (top-8, encoder `theta1`,
`mean_last_2`):

- **`graft_nn`** -- modo 0 (81% das ocorrências vêm de d0/1850):
  `prism, thermometer, populace, nucleus, iliad, excavation, projector,
  combatant`; modo 1 (73% de d1/2000): `data, commodity, innovation,
  nutrition, device, organism, dividend, statistics`. **Os dois modos têm
  vocabulários claramente diferentes e a composição temporal muda** -- é o
  caso mais limpo de "reorganização de sentido" entre os 8.
- **`tree_nn`** -- modo 0 e modo 1 têm vocabulários muito parecidos
  (`valley, wood, forest, river, land_nn, mountain/sun/stone`), ambos com
  composição ~60/40 d0/d1. **Os dois modos são "a mesma árvore"** -- bate
  com JSD baixo e com ser um controle estável.
- **`plane_nn`** -- modo 0 (`boat, ship, road, car, truck, sidewalk`) e modo
  1 (`route, vehicle, rail, panel, building, budget, troop`). Ambos os
  modos são do campo "veículo/transporte" -- **nenhum dos dois parece o
  sentido geométrico ("superfície plana")** que se esperava encontrar como
  um modo distinto em d1. Pode ser que, neste encoder pequeno, o sentido
  geométrico de "plane" simplesmente não tenha representação própria, ou
  que ele se misture com o sentido de veículo.
- **`chairman_nn`** -- 3 modos, mas os vizinhos dos 3 são quase o mesmo
  campo ("cargos/autoridades": `governor, president, secretary, commander,
  director`). JSD alto (0.367) aqui parece **falso positivo**: o k-means
  separou por algum eixo de uso (talvez frequência/registro) sem separar
  por SENTIDO -- os vizinhos não mudam de campo semântico entre modos.

**Interpretação:** o protótipo encontra UM caso claramente bom (`graft_nn`:
modos com vocabulários diferentes + composição temporal diferente + JSD
mais alto da amostra) e UM caso claramente "correto por estabilidade"
(`tree_nn`: modos parecidos + JSD mais baixo). Mas não é um GO limpo: (a)
`plane_nn`, o caso mais citado na literatura, não produziu o modo
geométrico esperado; (b) `chairman_nn` (estável) teve JSD alto por um
motivo que parece não ser de sentido. **Em palavras simples: a abordagem
"modos primeiro" às vezes funciona (graft) e às vezes não (plane,
chairman) -- não dá pra confiar nela sozinha ainda, mas ela já produz
saídas que fazem sentido olhar (os vizinhos por modo), o que o perfil
agregado da Fase 1.5 nunca produziu.** Próximo passo natural: repetir com
o encoder BERT da Tarefa 2 (quando disponível) para ver se a separação
geométrica/aviação de `plane_nn` aparece com um encoder melhor -- se nem o
BERT separar, o limite é da própria representação de "plane" como token
único (sem desambiguação por contexto suficiente), não do nosso pipeline.

### 7.22 Arquivos novos deste prototipo

- `scripts/prototype_modes_first_v2.py` -- agrupamento de ocorrências +
  perfil por modo, para as 8 palavras auditadas
- `outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/modes_first_phase/modes_first_results.json`

### 7.23 Tarefa 2 -- Passo 0': teto de oráculo com BERT pré-treinado congelado

**Ideia em palavras simples:** todos os resultados anteriores usam o nosso
encoder pequeno (128 dimensões, 3 camadas, treinado do zero nos dados do
SemEval). Esta tarefa repete a medida mais simples (`APD`: quão diferentes
são, em média, as ocorrências de uma palavra em 1850 vs em 2000) usando o
**BERT pronto** (`bert-base-uncased`, baixado da internet, usado "como
está", sem nenhum treino nos nossos dados), nas **mesmas frases** do
corpus. Isso responde: "se o problema é a qualidade do nosso encoder, um
encoder bem treinado resolve?"

Implementado em `scripts/evaluate_pretrained_oracle_v2.py`: para cada
palavra-alvo, lê até 150 frases de cada período do corpus lematizado
(`data/processed/semeval2020_task1/eng_lemma/corpus/*.txt`, onde a
palavra-alvo já vem marcada, ex. `plane_nn`), troca a marcação pela palavra
normal (`plane`), tokeniza com o tokenizer do BERT, e extrai o vetor da
palavra-alvo como a média dos seus subtokens na última camada (e na média
das últimas 4 camadas). Calcula `APD` (igual ao Passo 0, §7.3) e `NMI`
(cluster x período, igual a §7.10), e correlaciona com `truth.tsv`.

**Resultado -- correlação com `truth.tsv` (n=37):**

| Camada do BERT | Métrica | Spearman (p) | ROC-AUC | AP |
|---|---|---|---|---|
| última camada | `APD` | **0.594 (p=0.0001)** | 0.693 | 0.659 |
| última camada | `NMI` (cluster x período) | 0.341 (p=0.039) | 0.682 | 0.635 |
| média das últimas 4 | `APD` | **0.591 (p=0.0001)** | 0.676 | 0.592 |
| média das últimas 4 | `NMI` | 0.311 (p=0.061) | 0.634 | 0.588 |

**Pela primeira vez em toda a investigação, o resultado é
estatisticamente significativo** (p<0.001 para `APD`), e bem acima de
qualquer resultado anterior: o nosso encoder (diagonal ou `theta1` fixo,
§7.19) ficou sempre entre spearman 0.02 e 0.20; o BERT chega a 0.59 -- até
acima da literatura citada como referência (~0.4-0.55 para inglês no
SemEval-2020).

**Confirmação qualitativa nos 4 casos auditados:**

| Palavra (esperado) | APD (BERT, última camada) | NMI (BERT) |
|---|---|---|
| `plane_nn` (mudou) | **0.566** (maior dos 4) | **0.487** (muito acima dos outros) |
| `chairman_nn` (estável) | 0.466 | 0.092 |
| `graft_nn` (mudou) | 0.377 | 0.140 |
| `tree_nn` (estável) | 0.395 | **0.002** (quase zero) |

Olhando as frases de `plane_nn` (`qualitative_sentences.json`): em
1810-1860, as frases são todas do sentido GEOMÉTRICO ("the plane of
projection", "the intersection of the planes be call..."); em 1960-2010,
todas do sentido AVIAÇÃO ("by plane", "his private plane to Vienna's
airport"). O `NMI`=0.487 confirma que o BERT separa essas duas nuvens quase
ao longo da linha do período -- exatamente o "modo geométrico vs modo
aviação" que o nosso encoder pequeno NÃO conseguiu separar em `plane_nn`
(§7.21). Já `tree_nn` (estável) tem `NMI`≈0, ou seja, o BERT NÃO encontra
uma divisão por período -- como esperado para uma palavra que não mudou de
sentido.

### 7.24 Interpretação final (cruzando Tarefas 1 e 2)

Este resultado decide o cenário previsto em 7.17/Tarefa 2: **BERT
(~0.59) >> Tarefa 1 com `theta1` fixo (~0.20) >> diagonal original
(~0.13)**. Isso é o cenário "gargalo = encoder":

- A tarefa É solúvel com estas frases e este `truth.tsv` -- um encoder de
  qualidade suficiente chega a spearman~0.59, muito além do "teto" de
  ~0.13-0.20 que vínhamos vendo.
- O drift de checkpoint (7.14) e a falta de modos no perfil agregado (3.x,
  7.21) são problemas REAIS e foram corrigidos parcialmente nas Tarefas
  1 e 3 (ganho de 0.13 -> 0.20) -- mas a maior parte da distância até 0.59
  não vem deles. Vem da **qualidade/capacidade do encoder em si**
  (d_model=128, 3 camadas, treinado do zero só com MLM contínuo nos dados
  do SemEval, sem nenhum pré-treino).
- O caso `plane_nn`/`NMI` (7.23) mostra concretamente o que o BERT "vê" e o
  nosso encoder não: dois sentidos (geometria vs aviação) bem separados na
  nuvem de ocorrências. O protótipo "modos primeiro" (7.21) com o nosso
  encoder não encontrou esse modo geométrico para `plane_nn` -- agora temos
  evidência de que ele EXISTE na língua/no corpus, só não está representado
  pelo nosso encoder.

**Recomendação prática (substitui a prioridade 4 de 7.17 e a ablação de
capacidade "d_model 256/512 do zero"):** o próximo passo de maior retorno
não é aumentar `d_model` treinando do zero, e sim **inicializar o
Timeformer a partir de um checkpoint pré-treinado (ex.: BERT-base ou um
encoder pequeno pré-treinado equivalente) antes do treino contínuo
temporal**, mantendo o restante do pipeline (perfil relacional v2,
encoder fixo da Tarefa 1, agrupamento de ocorrências da Tarefa 3) como
está. As Tarefas 1 e 3 continuam válidas e devem ser aplicadas TAMBÉM ao
encoder pré-treinado, como infraestrutura de medição -- elas não competem
com essa mudança, são complementares a ela.

### 7.25 Arquivos novos desta tarefa

- `scripts/evaluate_pretrained_oracle_v2.py` -- `APD`/`NMI` com
  `bert-base-uncased` congelado sobre as mesmas ocorrências
- `outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/pretrained_oracle_phase/{pretrained_oracle_results.json,qualitative_sentences.json,rows_last.csv,rows_mean_last_4.csv}`

---

## 7. Arquivos de referência

- `docs/novo_perfil_relacional.md` -- documento canônico v2 (§1-14)
- `docs/relational_change_current_plan.md` -- log de decisões (seção
  "Fase 1 / 1.5 do Perfil Relacional v2 (2026-06-11)")
- `scripts/evaluate_relational_profile_v2.py` -- Fase 0A/1
- `scripts/evaluate_semantic_modes_v2.py` -- Fase 1.5
- `src/timeformers/gap_criterion.py`, `src/timeformers/semantic_modes.py`
- `tests/test_gap_criterion.py` (14 testes)
- `tmp/codex_perfil_relacional_v2_plan_review.md` -- revisão do plano de
  implementação (antes da Fase 1)
- `tmp/codex_semantic_modes_v2_nogo_review.md` -- auditoria do NO-GO
  (segunda opinião)
- Resultados:
  - `outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/relational_profile_v2_phase1/{metrics.json,deltas.csv}`
  - `outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/relational_profile_v2_phase1_5/modes.json`
  - `outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/relational_profile_v2_phase1_5_topn/modes.json`
  - `outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/relational_profile_v2_phase1_5_topn500/modes.json`
