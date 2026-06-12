# Pedido de avaliação: plano de implementação do "Perfil Relacional v2"

Queremos uma avaliação crítica e independente de um plano de implementação em
fases. Não assuma que o plano está correto nem que a ordem proposta é a
melhor. Procure: erros conceituais, dependências entre fases que invalidam a
ordem proposta, escolhas de engenharia que vão exigir retrabalho, e
oportunidades de simplificar ou eliminar fases.

O contexto é o projeto Timeformer: medir mudança semântica diacrônica via
embeddings contextuais de um Transformer MLM treinado continuamente
(`theta_0 = treino(D_0)`, `theta_t = continua_treino(theta_{t-1}, D_t)`).

## Documento de referência (definição canônica v2)

O plano abaixo implementa `docs/novo_perfil_relacional.md`, que define:

- **Centralização por período** (§4): `ê_t(x) = (e_t(x) - μ_t) / ||e_t(x) - μ_t||`,
  onde μ_t é a média de TODOS os embeddings do corpus do período t. Prova de
  invariância a rotação/reflexão/escala global; translação eliminada por
  construção.
- **Perfil relacional** (§5): `P_t(w)[v] = cos(ê_t(w), ê_t(v))` para todo
  v em vocabulário fixo V (construído sobre a união de todos os corpora).
- **Matriz de coesão semântica** M_t(w) (§7): para v,v' no suporte filtrado
  V_w = {v : P_t(w)[v] > τ}, `M_t(w)[v,v'] = P_t(w)[v] · P_t(w)[v'] ·
  cos(ê_t(v), ê_t(v'))`. M é PSD (prova: M = DCD com D diagonal positiva, C
  Gram). M nunca é formada explicitamente: decompõe-se via SVD de
  `DÊ = D · Ê` (D = diag(P_t(w)[V_w]), Ê = embeddings centralizados de V_w),
  dando `λᵢ = σᵢ²`, `aᵢ = uᵢ` (vetores singulares à esquerda).
- **Critério de gap** (§8): para qualquer distribuição ordenada decrescente
  X₁≥X₂≥..., `hᵢ = (Xᵢ - Xᵢ₊₁)/Xᵢ` é invariante a reescalamento positivo.
  τ = posição do gap máximo se h_max > γ (sugestão γ=0.3); senão "sem
  estrutura clara". Mesmo critério para k (número de modos) sobre os
  autovalores λᵢ.
- **Atribuição de ocorrências a modos** (§9): centróides de modo
  `cᵢ = Σ_v max(aᵢ[v],0)·ê_t(v)`; cada ocorrência ê^k_t(w) recebe
  responsabilidade soft `r^k_i ∝ max(0, cos(ê^k_t(w), ĉᵢ))`. Disso derivam
  frequência por modo f_t(w,i), embedding por modo e perfil por modo
  P_t(w|i).
- **Persistência de modos entre períodos** (§11): modos vivem no espaço de
  cargas sobre V (fixo), então são comparáveis entre períodos sem Procrustes.
  Emparelhamento húngaro sobre `s(i,j)=cos(ã⁺ᵢ, b̃⁺ⱼ)` (cargas estendidas ao
  suporte união). Limiar θ calibrado por **distribuição nula de permutação**:
  `θ = quantil_{1-α}({s(modo de w em t, modo de w'≠w em t')})`, α=0.05.
  Eventos: continuação/morte/nascimento/cisão/fusão. γ (do critério de gap)
  também governa quando um modo é individualmente rastreável (via
  Davis-Kahan: gaps pequenos = autovetores instáveis).
- **Protocolo de validação** (§13): piso de drift (Δ de palavras de controle
  estáveis), split-half intra-período, shuffle temporal, validação WSI
  (B-Cubed/V-measure) contra SemEval-2020 Task 1.
- **Implementação** (§14): ANN para top-k de P_t(w) (índice construído uma vez
  por período sobre V), critério de gap computável sobre prefixo top-k com
  duas condições de segurança (i* < 0.8k; P[v_k]/P[v_1] < ε=0.05),
  representação esparsa de perfis.

O documento completo está em `docs/novo_perfil_relacional.md` (789 linhas);
posso fornecer trechos adicionais se necessário.

## Estado atual do código (v1)

- `src/timeformers/real_corpus.py`: já constrói vocabulário fixo V sobre a
  união dos corpora de todos os períodos (✅ atende §3).
- `src/timeformers/real_models.py`: Transformer MLM (`RealStaticMLM`) treinado
  continuamente; checkpoints `theta0_d0.pt`, `theta1_d1.pt` etc. já existem
  para o corpus SemEval-2020 Task 1 (inglês, 2 períodos: 1810-1860, 1960-2010).
- Cache de extração atual (`cache/theta0_d0.pt` etc.) guarda **apenas**
  `sums` e `counts` por token por camada — ou seja, centróide agregado, **sem
  a nuvem de ocorrências individuais**.
- `scripts/report_temporal_relational_neighborhoods.py` (função
  `relational_profile`) já implementa algo parecido com §5, mas:
  - centraliza na **média das ~3.216 palavras de referência**, não em μ_t
    global sobre todo o corpus (diferença de §4.1);
  - opera só sobre essas ~3.216 referências, não sobre V completo;
  - não há decomposição em modos, não há nuvem de ocorrências, não há
    matching entre períodos.
- `src/timeformers/relational.py`: utilidades de similaridade/perfil log-PMI
  (abordagem log-PMI já foi **abandonada** por correlacionar com mudança de
  entropia, não com mudança semântica — ver `docs/relational_change_current_plan.md`).
- `src/timeformers/corpus.py`: gerador de corpus sintético com 10 períodos e
  classes de trajetória conhecidas (`stable`, `drift`, `abrupt`,
  `bifurcating`) — usado como banco de testes controlado.
- Não há FAISS/HNSW; `scipy.optimize.linear_sum_assignment` (Hungarian) está
  disponível via scipy mas não usado ainda.
- requirements.txt: torch, numpy, scikit-learn, scipy, matplotlib, gensim.
  d_model dos encoders treinados é pequeno (96-128), poucas camadas.

## Plano de implementação proposto (6 fases)

### Fase 0 — Decisão de dados
Decidir se a implementação v2 requer retreino do encoder ou apenas
reextração de caches a partir dos checkpoints `theta0`/`theta1` já
existentes. Proposta: V (vocabulário fixo) já é definido na tokenização,
independente dos pesos — então mudar V ou adicionar nuvem de ocorrências não
exige retreino, só reextração (forward passes sobre os checkpoints
existentes).

### Fase 1 — Centralização global + perfil sobre V completo
- Calcular μ_t = média de TODOS os embeddings do período (a partir dos
  `sums`/`counts` já cacheados de todos os tokens, não só referências) —
  **sem reextração**, reaproveitando cache existente.
- Modificar `relational_profile()` para centrar em μ_t global em vez da média
  das ~3.216 referências.
- Estender P_t(w) de ~3.216 referências para V completo (denso, sem ANN
  ainda — assumindo |V| seja da ordem de 10-50k, cabe em memória).
- Validar: recomputar Spearman/AUC do SemEval com a nova centralização e
  comparar com baseline atual (melhor resultado documentado: pmi_cosine
  Spearman=+0.114; perfil relacional centrado Spearman=0.210 com APD).

### Fase 2 — Critério de gap (τ, k) como módulo independente
- `gap_criterion.py`: `relative_gap()`, `select_threshold_index()`.
- Testes unitários com distribuições sintéticas (gap claro / ausente /
  poucos valores positivos).
- Aplicar sobre P_t(w) real (Fase 1) para inspecionar τ e |V_w| em palavras
  conhecidas (plane_nn, chairman_nn, graft_nn, tree_nn — os 4 alvos já
  auditados qualitativamente em trabalho anterior).

### Fase 3 — Decomposição espectral (SVD de DÊ)
- `semantic_modes.py`: `cohesion_svd()` monta DÊ = diag(P_t(w)[V_w])·Ê(V_w) e
  roda `torch.linalg.svd`; aplica gap criterion sobre λᵢ → k.
- Testar com palavras sintéticas da classe `bifurcating` em `corpus.py`
  (esperado k>1 após o ponto de bifurcação).
- Comparar k e tokens dominantes por modo com a leitura qualitativa já feita
  para plane_nn/chairman_nn/graft_nn/tree_nn.

### Fase 4 — Atribuição de ocorrências a modos (requer reextração)
- Modificar pipeline de extração para guardar amostra de embeddings por
  ocorrência por token (reservoir sampling, ~200/token/período), além de
  `sums`/`counts` — forward pass sobre checkpoints já treinados, sem
  retreino.
- Implementar centróides de modo, atribuição soft/hard, f_t(w,i),
  P_t(w|i) (§9).
- Validar qualitativamente em plane_nn (esperado: ocorrências D0 atribuídas
  ao modo geométrico, D1 ao modo transporte).

### Fase 5 — Persistência de modos entre períodos
- `mode_matching.py`: extensão de cargas ao suporte união, matriz de
  similaridade, Hungarian, calibração de θ por permutação (pares w≠w' do
  léxico de estudo W), classificação de eventos.
- Testar no corpus sintético de 10 períodos: bifurcação plantada deve gerar
  evento de cisão e duas cadeias de modos pós-bifurcação.

### Fase 6 — Protocolo de validação formal
- Piso de drift, split-half, shuffle temporal (adaptando
  `continual_placebo`/`resampled_null` já implementados).
- WSI: avaliar atribuição hard de modos contra SemEval-2020 Task 1 com
  B-Cubed/V-measure (sklearn).

## Riscos que já reconhecemos

1. |V| completo pode ser grande o suficiente para que o perfil denso da
   Fase 1 não escale, forçando ANN antes do planejado (§14.1 prevê isso, mas
   o plano o deixa só implícito/opcional).
2. d_model pequeno (96-128) pode limitar a separabilidade espectral exigida
   pelo critério de gap (§7.5: rank(M) ≤ min(|V_w|, d) — com d=96, no máximo
   96 modos, mas a separação real pode ser ainda mais pobre).
3. Com apenas 2 períodos no corpus SemEval, a Fase 5 (persistência de modos)
   tem material limitado — só 1 par de emparelhamento por palavra. O valor
   real do matching só aparece com 3+ períodos.
4. A calibração de θ por permutação (§11.4) precisa de um léxico de estudo W
   com tamanho razoável para a distribuição nula ter massa suficiente — não
   está claro se o léxico atual (37 alvos do SemEval) é suficiente.
5. Reextração com nuvem de ocorrências (Fase 4) muda o formato de cache;
   pode invalidar análises anteriores se não for versionado/separado
   claramente.
6. A Fase 1 muda a definição de centralização (μ_t global vs. média das
   referências) — pode alterar resultados já publicados/documentados em
   `docs/relational_change_current_plan.md`, exigindo reinterpretação.

## Perguntas para sua avaliação

1. A ordem das fases é logicamente sólida, ou há dependências que a
   invalidam (por exemplo, a Fase 3 depende de decisões da Fase 4 que só são
   tomadas depois)?
2. A Fase 1 (mudar a centralização de "média das referências" para μ_t
   global) é uma mudança pequena ou pode ter efeitos de segunda ordem não
   triviais sobre os resultados já documentados?
3. Dado d_model pequeno (96-128) nos encoders já treinados, o critério de gap
   espectral (§8.3) é factível, ou o plano deveria incluir uma fase de
   diagnóstico de separabilidade espectral ANTES de investir na Fase 3?
4. A Fase 5 (persistência de modos) tem valor demonstrável com apenas 2
   períodos (SemEval), ou deveria ser reordenada para depois de validar em
   corpus sintético de 10 períodos primeiro (antes mesmo da Fase 4, usando
   embeddings sintéticos onde a nuvem já está disponível)?
5. A decisão "reextração sem retreino" (Fase 0/4) é tecnicamente sólida, ou
   há algo no pipeline de treino atual (ex.: tokenização dinâmica, mascaramento
   estocástico) que tornaria a reextração inconsistente com os embeddings
   originais usados para treinar?
6. Existe uma fase que pode ser eliminada ou fundida com outra sem perda?
7. O plano deveria incluir, mais cedo, um experimento "go/no-go" de menor
   custo que valide a premissa central (separação espectral em modos
   produz algo melhor que o perfil agregado v1) antes de construir toda a
   infraestrutura de matching e validação?
8. Há riscos de identificabilidade/vazamento análogos aos já descobertos na
   tentativa anterior de "bairros semânticos" (`tmp/claude_semantic_neighborhood_plan_review.md`)
   que se aplicam também a esta proposta de modos espectrais?

## Arquivos relevantes

```text
docs/novo_perfil_relacional.md
docs/relational_change_current_plan.md
docs/relational_profile_formalization.md
src/timeformers/real_corpus.py
src/timeformers/real_models.py
src/timeformers/relational.py
src/timeformers/corpus.py
scripts/report_temporal_relational_neighborhoods.py
tmp/claude_semantic_neighborhood_plan_review.md
```

Por favor, responda com:

1. problemas na ordem/dependência das fases, ordenados por severidade;
2. recomendação de reordenação, fusão ou eliminação de fases, se aplicável;
3. avaliação específica da factibilidade do critério de gap espectral dado
   d_model pequeno;
4. desenho de um experimento "go/no-go" mínimo antes da Fase 4;
5. critérios objetivos para decidir, na Fase 0, entre reextração e retreino.
