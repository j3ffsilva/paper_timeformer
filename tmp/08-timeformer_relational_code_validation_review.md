# Auditoria Técnica e Científica — Pipeline Relacional Contínuo
**Data:** 2026-06-04
**Escopo:** Código relacional completo + outputs de seed_1000
**Baseado em:** leitura integral de `relational.py`, `relational_metrics.py`, `run_relational_continual_sanity.py`, `test_relational.py`, `05-relational_change_current_plan.md`, `dataset.py` (classes novas), `train.py:ContinualPeriodTrainer`, `config.json`, `counterfactual_summary.csv`, `relational_summary.csv`

---

## 1. Resumo executivo e veredito

**O código implementa corretamente o que afirma implementar.** Não há bugs de lógica nas funções críticas, o controle frozen funciona como esperado (zero delta em prediction probes — verificado nos dados), e o pipeline de treinamento contínuo preserva corretamente o estado do otimizador entre períodos.

**O resultado positivo é preliminarmente válido mas requer duas correções antes de múltiplas seeds:**

1. O checkpoint salvo não inclui o estado do otimizador — isso torna `--reuse-checkpoints` não-reproduzível e pode produzir trajetórias divergentes entre runs.
2. A direção do placebo (0.55–0.68) é substancialmente acima do acaso esperado e sua causa não está totalmente explicada. Esse é o principal risco à validade antes de escalar.

**Não há circularidade técnica entre o treinamento e o oráculo**, mas a relação entre a tarefa de treinamento e o oráculo é mais estreita do que o esperado — e precisa ser explicitada. O resultado de `stable` com direction cosine de +0.962 é matematicamente válido mas exige documentação cuidadosa para não ser mal interpretado.

**Veredicto:** rodar múltiplas seeds, mas corrigir os dois itens acima antes.

---

## 2. Compreensão do objetivo

O experimento pretende: treinar um Transformer padrão cronologicamente em D_0, D_1, ..., D_9 (sem sinal temporal explícito), medir como o perfil de similaridade interna de cada sujeito muda entre checkpoints, comparar essa mudança contra uma direção oráculo derivada das trajetórias sintéticas, e subtrair a deriva de otimização estimada pelo placebo (D_0 repetido).

---

## 3. O código implementa o objetivo?

**Sim, com fidelidade alta.** Ponto a ponto:

| Intenção | Implementação | Status |
|---|---|---|
| Único Transformer treinado cronologicamente | `ContinualPeriodTrainer` com único `opt` persistente | ✓ |
| Sem sinal temporal ao modelo | `Static` sem `needs_time`; `ContextPairMLMDataset` não passa `epoch_idx` útil | ✓ |
| Checkpoint por período | `torch.save(model.state_dict(), f"checkpoint_t{period:02d}.pt")` | Parcial — falta `opt.state_dict()` |
| Parada antecipada restaurando melhor estado | Salva `best_model_state` e `best_opt_state`, restaura ambos | ✓ |
| Placebo treina somente em D_0 | `placebo_datasets = [placebo_base] * n_periods` | ✓ |
| Probes separados do treino | `generate_subject_probe_examples()` usa `split="probe"` | ✓ |
| Controle frozen identidade | Verificado: delta = 0.000 em todos os períodos e métricas | ✓ |
| Métricas invariantes a rotação | Testado explicitamente em `test_relational.py` | ✓ |

---

## 4. Bugs ou inconsistências concretas

### 4.1 [BLOQUEADOR] Checkpoint não salva estado do otimizador

**Arquivo:** `train.py`, linha 268
**Código:** `torch.save(self.model.state_dict(), self.output_dir / f"checkpoint_t{period:02d}.pt")`

Apenas o estado do modelo é salvo. O estado do otimizador (momentos do Adam) **não é persistido**.

**Consequência:** quando `--reuse-checkpoints` é usado numa segunda execução, o modelo é restaurado mas o otimizador começa do zero (sem momentos acumulados). A trajetória do modelo a partir desse ponto diverge da trajetória original — mesmo com mesmo seed e mesmos dados. Isso invalida a reprodutibilidade quando se reutiliza checkpoints, e pode produzir resultados diferentes entre uma execução integral e uma execução com `--reuse-checkpoints`.

**Fix:** salvar também o estado do otimizador junto com o modelo. Dado que o otimizador é único e persiste entre períodos, o estado relevante após o período t é o restaurado após a parada antecipada:

```python
torch.save(
    {"model": self.model.state_dict(), "opt": opt.state_dict()},
    self.output_dir / f"checkpoint_t{period:02d}.pt",
)
```

E ao carregar em `load_checkpoint_model` (script), restaurar ambos.

### 4.2 [Importante] Placebo `in_corpus` ainda usa contextos de D_0, mas `subject_prediction_probes` não

Em `collect_profiles`, `placebo_contexts=True` faz `context_period = 0` — garantindo que o modo `in_corpus` do placebo use apenas contextos de D_0. Correto.

Mas o modo `subject_prediction_probes` usa `subject_only_dataset = ContextPairMLMDataset(generate_subject_probe_examples())` — as mesmas sondas neutras para real e placebo. Isso é correto e necessário. Porém, o script não deixa explícito que, para probes, real e placebo usam exatamente os mesmos inputs — isso deveria ser documentado no código.

### 4.3 [Importante] `in_corpus` inclui exemplos de treino

`examples_for_epoch(rows, period)` retorna todos os exemplos (train + test) do período. O modo `in_corpus` usa `RepresentationDataset(examples_for_epoch(rows, context_period))` — que inclui exemplos de treino.

Para extração de representações isso é aceitável (não há vantagem de treino vs. test na extração), mas é tecnicamente correto documentar: "perfis `in_corpus` foram extraídos do conjunto completo de ocorrências, incluindo o split de treino."

### 4.4 [Desejável] Seed no `build_training_dataset` do placebo

```python
# script linha 92
return dataset_type(examples_for_epoch(...), seed=args.seed + period)
```

Para `ContextPairMLMDataset`, a seed é recebida mas `_make_item` faz `del rng` — o mascaramento é determinístico. A seed não tem efeito. Isso não é um bug, mas é código morto que pode confundir. O construtor de `ContextPairMLMDataset` poderia ser adaptado para não aceitar seed, ou a `_make_item` deveria ser corrigida.

---

## 5. Riscos à validade científica

### 5.1 [Risco principal] Direção do placebo é ~0.55–0.68, muito acima do acaso

Para o modo `subject_prediction_probes`, from_t0 to t9:

| Classe | Placebo direction cosine |
|---|---|
| abrupt | +0.559 |
| bifurcating | +0.677 |
| drift | +0.549 |
| stable | +0.663 |

Em espaço de dimensão n−1 = 39 (39 outros sujeitos), o valor esperado do cosseno absoluto entre dois vetores aleatórios é `1/sqrt(39) ≈ 0.16`. Os valores observados são 3–4× mais altos.

**Hipótese mais plausível:** o modelo treinando continuamente em D_0 melhora progressivamente sua estimativa das p_n1 dos sujeitos em t0. À medida que as estimativas melhoram, a matriz de similaridade do probe se alinha melhor com o oráculo de t0. Mas o oráculo que usamos para comparar é de diferentes períodos t > 0. Pelo fato de a classe `stable` não mudar e de a classe `bifurcating` convergir para um platô intermediário, o oráculo de t9 ainda preserva parte da estrutura de t0 — e portanto a melhora da estimativa do placebo se alinha parcialmente com o oráculo de t9.

**Consequência:** a subtração `delta_real - delta_placebo` remove parte do sinal real (se a melhora do placebo é na mesma direção que a mudança semântica real), ou pode inflar o sinal se as melhoras forem em direções diferentes.

**Controles necessários antes de escalar:**
1. Permutação de período: executar o mesmo experimento com ordem de períodos embaralhada (D_3 → D_7 → D_1 → ...). O direction cosine excedente deveria degradar; se não degradar, o sinal não é específico à ordem cronológica.
2. Múltiplas seeds do placebo: rodar o placebo com 10 seeds diferentes e construir uma distribuição nula do direction cosine. O excedente do `continual_real` deve ser superior ao percentil 95 dessa distribuição nula.

### 5.2 [Risco médio] Stable tem direction cosine mais alto que as classes que mudam

Para `from_t0 → t9`, o cosseno da direção observada para `stable` é +0.962 — mais alto que `drift` (+0.944) e `abrupt` (+0.960). Matematicamente isso é válido: o perfil relacional de uma palavra estável muda quando as outras palavras se movem, e o oráculo captura exatamente isso. Mas a narrativa "palavras estáveis mudam seus perfis relacionais mais do que palavras que deveriam mudar" requer explicação cuidadosa.

A explicação correta: o oráculo delta para `stable` é dominado por grandes mudanças de drift/abrupt em relação ao stable, enquanto o oráculo delta para `drift` captura suas mudanças relativas entre si e em relação a stable. Os probe representations de stable capturam facilmente "os outros se afastaram de mim", enquanto os de drift precisam capturar "eu me movi em relação aos outros" — o que é uma tarefa mais sutil.

**Isso não invalida o resultado, mas deve ser explicitado no paper.**

### 5.3 [Risco médio] Tarefas de treino e oráculo são estruturalmente alinhados

O modelo treina em `ContextPairMLMDataset`: aprende a prever `(verb, obj)` a partir do sujeito. O oráculo é baseado em `p_n1` (probabilidade de usar contextos N1 vs. N2). O probe preditivo consulta o modelo sobre quais contextos ele prevê para cada sujeito.

Em síntese: modelo aprende → sujeitos com mesmo p_n1 têm a mesma distribuição de contextos → probe representations de sujeitos com p_n1 similar serão similares → oráculo mede exatamente essa similaridade. O direction cosine alto é esperado se o modelo aprendeu a tarefa.

Isso não é circularidade (o modelo não vê p_n1 diretamente), mas é menos surpreendente do que parece. O paper deve posicionar o resultado como "o Transformer organiza seus estados internos de acordo com a estrutura latente do corpus, e essa organização muda quando o corpus muda" — não como "descobrimos que o Transformer rastreia mudança semântica de forma misteriosa".

### 5.4 [Risco baixo] t1 e t2 têm direções negativas ou fracas nos períodos consecutivos

Para t1 e t2 consecutivos, os valores são fracos (±0.2) ou negativos. Isso pode indicar:
- Instabilidade inicial ao trocar de corpus — esperado
- Catastrophic forgetting seletivo no primeiro período de transição
- Ou simplesmente que a mudança entre t0→t1 e t1→t2 é pequena e o sinal é ruidoso

Com 1 seed não é possível distinguir. As múltiplas seeds resolverão isso.

---

## 6. Avaliação do probe preditivo

O probe preditivo (`subject_prediction_probes`) usa `ContextPairMLMDataset(generate_subject_probe_examples())`. A entrada é `[CLS, S_i, MASK, MASK, SEP]` — os mesmos tokens para todos os checkpoints, garantindo comparabilidade. A representação extraída é `hidden[:, [POS_VERB, POS_OBJECT], :].mean(dim=1)` — estados ocultos nas posições mascaradas, após o Transformer.

**Justificativa científica:** os estados ocultos nas posições mascaradas contêm a distribuição que o modelo prevê para os contextos do sujeito. Dois sujeitos cuja distribuição de contextos é similar terão estados similares. Isso captura exatamente o que queremos: "que contextos o modelo prevê que acompanham este sujeito?"

**Comparação com alternativas:**

| Representação | O que mede | Discriminação observada |
|---|---|---|
| `subject_prediction_probes` (masked) | Contextos previstos pelo modelo | Muito alta (+0.96) |
| `subject_only_probes` (h_subj) | Estado do sujeito sem contexto | Baixa (+0.08 para drift) |
| `fixed_probes` (centroides com contextos) | Média sobre contextos específicos | Baixa/nula (+0.025 para drift) |
| `in_corpus` (ocorrências reais) | Média sobre ocorrências do período | Idêntico a fixed_probes |

A diferença entre `subject_prediction_probes` e as outras representações é dramática. Isso indica que o sinal semântico temporal está principalmente codificado na **previsão de contexto**, não no estado direto do sujeito. Isso é um resultado científico interessante por si só.

---

## 7. Avaliação das métricas e do oráculo

### Oracle

```python
# script linha 217-221
points = [[trajectories[subject][period], 1.0 - trajectories[subject][period]] for subject in SUBJECTS]
profiles.append(cosine_similarity_matrix(points))
```

O oráculo é uma matriz de similaridade de cosseno entre vetores 2D `[p_n1, 1-p_n1]`. Como cada vetor está na semiesfera positiva do círculo unitário 2D, a matriz é monotonicamente relacionada a `|p_n1_i - p_n1_j|` — é uma ordenação 1D dos sujeitos por p_n1.

**O oráculo é válido como referência para o benchmark sintético**, dado que p_n1 é exatamente a propriedade que queremos rastrear. Não é circular porque o modelo não recebe p_n1 diretamente.

**Limitação:** o oráculo 2D é muito simples — é basicamente uma linha 1D. Em corpus real, não existirá um oráculo tão clean. O paper deve documentar que, no sintético, o oráculo é conhecido e ideal.

### Métricas relacionais

- **Jaccard de vizinhos:** correto, exclui diagonal, invariante a rotação ✓
- **Spearman do ranking:** correto, usa `1 - rho` como medida de mudança ✓
- **Mean abs similarity delta:** correto, exclui diagonal via máscara ✓
- **Direction cosine:** faz o produto escalar entre o delta vetorializado (exceto diagonal) e o oracle delta. Correto e matematicamente bem motivado ✓
- **CKA entre checkpoints:** usa `linear_cka` sobre os pontos brutos (centroides). Correto como diagnóstico global de estabilidade do espaço ✓

**Problema na normalização do Spearman:** `spearman_change = 1 - rho`. Se `rho = -1` (ranking completamente invertido), `spearman_change = 2`. Isso cria uma escala assimétrica (0 a 2) que não é imediatamente interpretável como "probabilidade de mudança". Considere usar `(1 - rho) / 2` para normalizar para [0,1], ou documentar explicitamente que 2 = máxima mudança.

### Tratamento da diagonal

`counterfactual_relational_change` usa máscara `~eye` e reshape correto para excluir diagonal:
```python
mask = ~torch.eye(n_subjects, dtype=torch.bool, device=excess.device)
excess_without_self = excess[mask].reshape(n_subjects, n_subjects - 1)
```
Correto. ✓

---

## 8. Avaliação do placebo e da mudança contrafactual

### Validade da subtração `delta_real - delta_placebo`

A subtração é válida **se e somente se** real e placebo partem do mesmo estado inicial e recebem tratamentos comparáveis em termos de número de atualizações e tamanho de batch. Verificação:

- Mesmo estado inicial: ambos usam `torch.manual_seed(args.seed)` antes de `build_static(args)` ✓
- Mesmo número de atualizações: real e placebo têm os mesmos `n_epochs_per_period` e `n_epochs_first_period` — **mas podem divergir pelo early stopping**. Se o real para mais cedo que o placebo num período (ou vice-versa), os dois modelos terão feito número diferente de atualizações totais no final. Isso torna os deltas não estritamente comparáveis.

**Fix necessário:** registrar e reportar o número total de atualizações de gradiente por período para real e placebo, para confirmar que são comparáveis.

### Por que o placebo tem direction cosine ~0.6?

Hipótese mais provável: o modelo continuando em D_0 refina progressivamente suas estimativas de p_n1, e a matriz de similaridade do checkpoint_t{k} (placebo) se alinha melhor com o oráculo de t0, que ainda tem estrutura similar ao oráculo de t9 (por causa da classe stable e da estrutura geral de p_n1 que não muda completamente).

Isso significa que parte do que o `excess` mede não é "mudança específica de período" mas "convergência adicional no espaço de p_n1". A subtração não remove completamente esse efeito.

**Alternativas melhores:**
1. **Permutação de período** (Caso E do prompt): embaralhar a ordem dos períodos. Se o direction cosine do excedente for similar ao resultado cronológico, o sinal não é de ordem temporal — é apenas de "treinamento em dados mais variados". Esta é a ablação mais crítica faltando.
2. **Normalização pela distribuição nula**: rodar placebo com 10+ seeds, construir a distribuição do direction cosine esperado por deriva pura, e reportar Z-score ou percentil do resultado real.

---

## 9. Auditoria por arquivo

### `src/timeformers/relational.py` (25 linhas)
**Classificação:** Manter sem alteração.
Implementação correta de `cosine_similarity_matrix`, `topk_neighbors`, `relational_delta`. Simples e bem testado.

### `src/timeformers/relational_metrics.py` (90 linhas)
**Classificação:** Manter com ajuste menor (Spearman normalização).
`relational_change_by_subject` e `counterfactual_relational_change` estão corretamente implementados. O único ajuste sugerido é normalizar `spearman_change` para [0,1].

### `scripts/run_relational_continual_sanity.py` (420 linhas)
**Classificação:** Manter com duas adições.
O script orquestra corretamente os três regimes (real, placebo, frozen), coleta perfis, calcula métricas e counterfactuals. As adições necessárias:
1. Salvar `opt.state_dict()` junto com o modelo em `ContinualPeriodTrainer` (via ajuste em `train.py`)
2. Adicionar registro de número de atualizações por período no history

### `src/timeformers/train.py:ContinualPeriodTrainer`
**Classificação:** Reutilizar com correção do checkpoint (salvar opt state).
O treinamento contínuo está correto: single optimizer, early stopping com restauração de ambos model e opt, sem CosineAnnealingLR (importante para evitar artefatos de restart). A única correção necessária é o salvamento do opt state.

### `src/timeformers/dataset.py:ContextPairMLMDataset`
**Classificação:** Manter sem alteração.
Mascaramento correto: ambas as posições de contexto são mascaradas, labels corretos nas duas posições, sujeito fica visível. O teste `test_context_pair_masking_hides_both_context_markers` valida explicitamente ✓.

### `src/timeformers/dataset.py:RepresentationDataset`
**Classificação:** Manter sem alteração.
Dataset sem mascaramento para extração de representações. Correto.

### `src/timeformers/corpus.py` (funções novas)
**Classificação:** Manter sem alteração.
`generate_fixed_probe_examples`, `generate_subject_probe_examples`, `examples_for_epoch` são implementações corretas e simples. O teste `test_subject_probes_have_one_neutral_example_per_subject` valida a estrutura dos probes ✓.

### `tests/test_relational.py` (81 linhas)
**Classificação:** Manter, adicionar testes.
Cobre invariância ortogonal, detecção de mudança, extração do sinal contrafactual e mascaramento. Faltam testes descritos na Seção 10.

---

## 10. Testes adicionais obrigatórios

### [BLOQUEADOR] Teste de permutação de período

```python
def test_shuffled_period_order_produces_lower_direction_cosine():
    # treinar real com períodos em ordem D_0,D_1,...D_9
    # treinar shuffled com períodos em ordem D_5,D_2,D_8,...
    # direction cosine excedente de shuffled deve ser < direction cosine do real
```

Este é o teste mais crítico para validade do sinal temporal. Se o sinal não depende da ordem, não está capturando mudança semântica temporal.

### [BLOQUEADOR] Teste de reprodutibilidade com `--reuse-checkpoints`

```python
def test_reuse_checkpoints_produces_same_profiles():
    # run 1: treina do zero, extrai perfis
    # run 2: --reuse-checkpoints, extrai perfis
    # perfis devem ser idênticos
```

Atualmente este teste falharia por causa do opt state não salvo.

### [Importante] Teste de case D: movimento conjunto preserva relações

```python
def test_all_subjects_shift_together_preserves_relations():
    # criar duas matrizes de representações onde todos os vetores são
    # transladados pelo mesmo delta
    # relational_change_by_subject deve retornar zero
```

Nota: cosseno não é invariante a translação — se todos os vetores forem transladados pelo mesmo vetor, as similaridades de cosseno podem mudar. Este teste verificaria se a medida relacional é ou não invariante a translação (provavelmente não é), o que é uma propriedade relevante para documentar.

### [Importante] Distribuição nula do direction cosine

```python
def test_direction_cosine_under_null():
    # gerar 100 pares de matrizes de similaridade aleatórias
    # calcular direction cosine com oráculo aleatório
    # confirmar que média e desvio padrão são próximos de E[|cos|] teórico
```

Isso estabelece a escala de referência para avaliar os valores observados.

### [Desejável] Teste do frozen control por construção

```python
def test_frozen_model_same_probes_gives_zero_delta():
    # criar modelo, extrair probes com mesmo input duas vezes
    # delta deve ser zero
```

Já está verificado empiricamente nos dados, mas um teste automatizado seria melhor.

---

## 11. Mudanças necessárias antes de múltiplas seeds

**1. [BLOQUEADOR] Salvar opt.state_dict() nos checkpoints**

```python
# train.py, linha 268
torch.save(
    {"model": self.model.state_dict(), "opt": opt.state_dict()},
    self.output_dir / f"checkpoint_t{period:02d}.pt",
)
```

Atualizar `load_checkpoint_model` no script para restaurar apenas `model` (ou ambos, dependendo do contexto).

**2. [BLOQUEADOR] Adicionar permutação de período como controle**

No script, acrescentar regime `continual_shuffled` que treina na mesma sequência de datasets mas com ordem embaralhada. Calcular direction cosine do excedente para este regime. Deve ser menor que o regime cronológico.

**3. [Importante] Registrar número de atualizações por período**

Adicionar `"n_gradient_steps"` no history do `ContinualPeriodTrainer`, para confirmar que real e placebo são comparáveis em termos de compute.

**4. [Importante] Normalizar `spearman_change` para [0,1]**

Em `relational_metrics.py`, linha 36: `(1.0 - rho) / 2.0 if not math.isnan(rho) else 1.0`

---

## 12. Mudanças necessárias antes de corpus real

**1. Estratégia para palavras ausentes em checkpoints anteriores**

No corpus sintético, o vocabulário é fixo. Em corpus real, algumas palavras não terão ocorrências em D_t0. A função `subject_centroids` em `representations.py` assume que todos os sujeitos aparecem em todos os períodos — isso quebrará em corpus real.

**2. Definição de conjunto de referência**

As métricas atuais usam todos os sujeitos (40) como referência uns para os outros. Em corpus real, com vocabulários de milhares de palavras, a dimensão do perfil relacional será enorme. Precisará de seleção de palavras-âncora ou redução.

**3. Probe sentences para corpus real**

`generate_subject_probe_examples` usa `Example(epoch=0, subject, V1, O1, ...)` — estrutura sintética. Para corpus real, probes serão sentenças reais com a palavra-alvo mascarada. A arquitetura do probe precisa ser repensada.

**4. Teste de permutação de período é crítico especialmente no real**

No corpus real, os períodos têm conteúdos genuinamente diferentes. O sinal de permutação testará se o modelo está capturando cronologia ou apenas diversidade de corpus.

---

## 13. Plano recomendado em ordem de prioridade

| Prioridade | Ação | Bloqueadora? |
|---|---|---|
| 1 | Corrigir salvamento de opt.state_dict() em checkpoints | Bloqueadora |
| 2 | Adicionar regime `continual_shuffled` ao script | Bloqueadora |
| 3 | Rodar 5+ seeds com as correções acima | — |
| 4 | Adicionar distribuição nula do direction cosine (10+ seeds do placebo) | Importante |
| 5 | Registrar n_gradient_steps por período em history | Importante |
| 6 | Normalizar spearman_change para [0,1] | Desejável |
| 7 | Adicionar testes unitários de permutação e reprodutibilidade | Importante |
| 8 | Documentar explicitamente no plan: stable direction cosine alto é válido e esperado | Importante |

---

## Apêndice: casos de teste mental respondidos

**Caso A (rotação global):** Todas as métricas de similaridade por cosseno são invariantes a rotação ortogonal. Confirmado pelo teste `test_orthogonal_coordinate_change_is_not_semantic_change`. ✓

**Caso B (D_0 repetido = placebo):** O placebo mostra direction cosine de 0.55–0.68, acima do acaso esperado. Isso é o risco principal identificado nesta auditoria. A causa provável é refinamento da estimativa de p_n1 com mais treinamento em D_0.

**Caso C (apenas S1 muda):** O perfil relacional de S1 muda. Os perfis de outros sujeitos também mudam por causa da relação deles com S1. Isso está documentado no plano (`05-relational_change_current_plan.md`) e é matematicamente correto. A métrica direcional capturará essa mudança corretamente.

**Caso D (todos se movem preservando relações):** A medida de cosseno é invariante a rotação mas NÃO a translação. Se todos os vetores forem transladados pelo mesmo vetor, as similaridades de cosseno mudarão. Um teste explícito desta propriedade está faltando e deveria ser adicionado.

**Caso E (permutação de rótulos de período):** Este controle não está no experimento atual e é identificado como bloqueador.

**Caso F (probe diferente, mesma semântica):** Os dados mostram claramente que diferentes representações dão resultados muito diferentes. `subject_prediction_probes` domina. Isso já está sendo avaliado e documentado no plano.
