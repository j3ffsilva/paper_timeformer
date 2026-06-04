# Revisão Técnica e Científica Independente — Timeformer Paper 2
**Data:** 2026-06-03
**Revisor:** Externo, independente
**Projeto:** `/Users/jeff/Documents/trabalhos/papers/paper-timeformers`
**Pergunta central:** Estamos na direção certa, e o código suporta as conclusões metodológicas que estão sendo tiradas?

---

## 1. Resumo Executivo

O projeto está na direção correta nas suas fundações conceituais. A distinção `token@time = (h_s(t), m_s(t))` é bem motivada, e o pipeline de dois passos com teacher self-supervised e student bidirecional é arquiteturalmente sólido. A hipótese D5a (bidirecional > causal > linear na classe Abrupt) passou com os resultados experimentais, o que é encorajador.

**Porém, há um problema de consistência semântica grave no componente SSL do agregador** que precisa ser resolvido antes de qualquer avanço para o COHA. O sinal `context` que está sendo passado para a loss contrastiva SSL (`context_similarity_contrastive_loss` em `aggregator_ssl.py`) **é derivado de embeddings do modelo semântico sobre os tokens de verbo e objeto** — não de `true_context`. Isso é defendível, mas tem implicações cruciais que não estão documentadas e que criam ambiguidade sobre o que exatamente o SSL está aprendendo.

Há também problemas menores mas importantes: (a) a métrica D2 usa protótipos derivados de `p_n1` de toda a split de avaliação, o que é uma forma leve de leakage; (b) o `SetSlotsAggregator` produz `R = slots.reshape(-1)` com dimensão `num_slots * d_model`, o que quebra a suposição de que `d_in` do teacher é `d_model`; (c) o D5a em `run_d5a_student_ablation.py` instancia o teacher com `d_in=args.d_model` hardcoded, enquanto o pipeline principal usa `d_in=d_aggregated` — inconsistência que pode causar erro silencioso com `set_slots`.

---

## 2. O que está sólido

### 2.1 Formulação conceitual
A distinção entre `h_s(t)` (posição semântica numa ocorrência contextual) e `m_s(t)` (estado agregado da trajetória) é conceitualmente limpa e bem articulada no `docs/novo_planejamento.md`. O documento reconhece explicitamente que `m_s(t)` carrega contexto local de forma agregada, não que seja livre de contexto — esta precisão é importante e é honesta.

### 2.2 Pipeline do teacher
O `TrajectoryTeacher` (`trajectory_models.py`) e o `TrajectoryTeacherTrainer` (`trajectory_train.py`) implementam corretamente `L_recon + β * L_anti_identidade`. O teacher é corretamente congelado antes de treinar o student (linhas 105-107 em `trajectory_train.py`):
```python
self.teacher.eval()
for param in self.teacher.parameters():
    param.requires_grad_(False)
```
Não há loop de feedback acidental — o alvo é fixo durante o treino do student.

### 2.3 Student bidirecional e D5a
O `TrajectoryStudent` (`trajectory_models.py`) usa `encoder_variant="bidirectional"` por padrão e implementa corretamente a máscara de posição. O `evaluate_all_masked_reconstruction` em `trajectory_train.py` avalia mascarando **todas** as posições válidas — não apenas uma amostra — o que é a abordagem correta para um D5a justo conforme descrito no planejamento.

O resultado documentado (bidirectional < causal << linear em Abrupt) está alinhado com a hipótese e com a motivação da arquitetura bidirecional para capturar rupturas abruptas.

### 2.4 Interpolação de trajetória
`build_trajectory_sequences` (`trajectories.py`) implementa corretamente a interpolação linear interna sem extrapolação nos extremos, exatamente como especificado no planejamento (Seção 2.4). A distinção entre `valid_mask` (períodos no span) e `observed_mask` (períodos efetivamente observados) é útil e está disponível, embora subutilizada nas métricas atuais.

### 2.5 Construção do corpus sintético
`generate_trajectories` e `generate_examples` (`corpus.py`) implementam corretamente as quatro classes (stable, drift, bifurcating, abrupt). A classe abrupt usa `step function` com `switch` sorteado uniformemente — alinhado com o planejamento. O ruído de 25% (fidelity=0.75 por padrão) está implementado corretamente em `_draw_marker`.

### 2.6 Separação entre regimes supervisionado e SSL
O `docs/synthetic_results_current.md` documenta honestamente que o resultado forte de D6 com `set:bidirectional` vem do treino supervisionado com `true_context`, e que isso é sanidade/upper bound, não a configuração principal. Isso é cientificamente correto e a narrativa está sendo gerenciada adequadamente.

### 2.7 Anti-identidade
`anti_identity_loss` (`trajectory_losses.py`) implementa corretamente o mecanismo: penalidade de CKA quando CKA(M, Seq) > τ_cka, mais regularização de variância via VICReg-style. A implementação do CKA linear é matematicamente correta (centralização, produto de Frobenius dos produtos cruzados normalizados).

---

## 3. O que é frágil ou potencialmente errado

### 3.1 [CRÍTICO] O sinal `context` no SSL do agregador é os embeddings do modelo, não `true_context` — mas isso precisa ser explicitado

Em `representations.py`, linha 33:
```python
context = model.token_emb(context_ids).mean(dim=1)
```
E em `dataset.py`, linhas 44:
```python
context_ids = [ids[POS_VERB], ids[POS_OBJECT]]
```

O `context` que alimenta o `context_similarity_contrastive_loss` no SSL do agregador são **os embeddings aprendidos dos tokens de verbo e objeto** (antes do transformer, portanto sem contextualização). Isso tem duas implicações opostas:

**A favor (defensável):** É genuinamente auto-supervisionado — não usa `true_context` nem `p_n1`. Os tokens de verbo e objeto são sinal observável no corpus natural.

**Problema:** No corpus sintético, VERBS_N1 = {V1,V2,V3,V4} e VERBS_N2 = {V5,V6,V7,V8}. Com fidelidade 0.75, 75% dos exemplos têm verb/object do contexto correto. Isso significa que `context_ids` carregam informação fortemente correlacionada com `true_context` — mas não são `true_context`. O SSL está, na prática, aprendendo a separar as ocorrências por contexto de vizinhança observável, que é exatamente o que queremos. Mas a magnitude do sinal é dependente da fidelidade do corpus sintético, e isso não está documentado.

**Consequência para o COHA:** No corpus natural, os embeddings de verbo e objeto são aprendidos por MLM sem sinal temporal explícito — a qualidade do sinal contextual para o SSL depende da qualidade do encoder semântico. Se o encoder semântico ainda não distingue bem os contextos (especialmente no início do treinamento), o SSL do agregador receberá ruído. Isso é um risco de fragilidade não documentado.

**Recomendação:** Documentar explicitamente que `context` = embeddings de co-ocorrentes, quantificar a correlação de `context_ids` com `true_context` no sintético em diferentes fidelidades, e testar no ablation se a performance do SSL degrada proporcionalmente com a fidelidade do corpus.

### 3.2 [CRÍTICO] D2 usa protótipos derivados de toda a split de avaliação — forma leve de leakage

Em `trajectory_metrics.py`, função `cosine_axis_scores` (linhas 47-56):
```python
n1 = flat[target >= 0.5]
n2 = flat[target < 0.5]
proto_n1 = torch.nn.functional.normalize(n1.mean(dim=0), dim=0)
proto_n2 = torch.nn.functional.normalize(n2.mean(dim=0), dim=0)
```

Os protótipos `proto_n1` e `proto_n2` são calculados **a partir de todos os dados de avaliação**, usando `p_n1` como limiar de classe. Isso significa que a métrica D2 (drift score) usa o "ground truth" — ainda que de forma agregada — para construir o eixo de comparação. O score de cada ponto individual é então calculado como `sim(rep, proto_n1) - sim(rep, proto_n2)`, onde os protótipos incluem o próprio ponto.

Isso **não é leakage no sentido de treinar no test set** — os protótipos não são usados durante o treinamento. Mas é leakage **de avaliação**: a métrica é construída usando informação que um avaliador real não teria. Para um corpus natural, os protótipos precisariam ser construídos de alguma outra forma.

O planejamento (Seção 5, D2) menciona:
> `score(t) = sim(rep(S,t), centroide_N1) − sim(rep(S,t), centroide_N2)`

Mas não especifica de onde vêm os centroides. No código, eles vêm de `p_n1 >= 0.5` aplicado à split de avaliação completa — essencialmente usando o ground truth sintético. Para o D2 ser uma métrica legítima no COHA, precisará de outra estratégia.

**Recomendação:** Para o sintético, documentar que D2 usa ground truth como supervisão de avaliação (aceitável para benchmark controlado, mas deve ser explicitado). Para o COHA, definir os protótipos usando os embeddings dos próprios períodos extremos como proxy (sem usar anotações de mudança).

### 3.3 [CRÍTICO] `SetSlotsAggregator` muda a dimensão de `R` — inconsistência com `d_in` do teacher

Em `aggregators.py`, linha 73:
```python
return {"R": slots.reshape(-1), "U": u, "slots": slots}
```

Com `num_slots=2` e `d_model=32`, `R` tem dimensão `64`, não `32`. Já `SetTransformerAggregator` retorna `R` com dimensão `d_model`.

Em `run_synthetic_pipeline.py`, linha 111:
```python
d_aggregated = sequences.values.size(-1)
teacher = TrajectoryTeacher(d_in=d_aggregated, ...)
```

O pipeline principal compensa porque usa `d_aggregated` dinamicamente. Mas em `run_d5a_student_ablation.py`, linha 122:
```python
teacher = TrajectoryTeacher(d_in=args.d_model, ...)
```

Aqui `d_in` é hardcoded para `args.d_model`. Se `set_slots` for usado nesse script, o teacher terá dimensão errada e ou falhará ou produzirá resultados silenciosamente errados (se `d_traj == args.d_model` por coincidência, não haverá erro de shape).

Adicionalmente, há um problema conceitual: com `set_slots`, a sequência de representações tem dimensão `num_slots * d_model`. A CKA entre `R` (dimensão `2*d_model`) e `m_student` (dimensão `d_traj`) em `d2_context_drift_metrics` e `cka_metric` usa as dimensões como veio do `sequences.values`, o que é correto — mas significa que os experimentos com `set_slots` e com `set` são comparando coisas de dimensões diferentes para `R`, o que pode inflar/deflacionar D2 sem que isso seja a arquitetura do encoder temporal.

### 3.4 D6 usa `true_context` como labels — aceitável no sintético, mas é leakage de supervisão

Em `trajectory_metrics.py`, linha 134:
```python
labels = group["true_context"].detach().cpu().numpy()
```

Para calcular o silhouette, D6 usa `true_context` (0 ou 1) como labels de cluster. No sintético, isso é aceitável — é o único modo de avaliar bimodalidade em ambiente controlado, e o documento reconhece que esse é um benchmark com ground truth.

No entanto, o planejamento (Seção 5, D6) diz:
> `bimodalidade(S, t) = silhouette({u_s^i(t)}, labels={N1_ctx, N2_ctx})`

E diz que os labels são derivados de `true_context`. Isso está corretamente implementado. Mas quando o paper afirmar "bimodalidade melhora com SSL", o crítico correto apontará que o silhouette está sendo calculado com as classes verdadeiras como referência — ou seja, D6 não é uma métrica cega. É um diagnóstico supervisionado com ground truth sintético, o que é legítimo para o benchmark controlado mas deve ser apresentado assim explicitamente.

Para o COHA, D6 precisará usar clusters não-supervisionados (GMM, K-Means com K=2) como proxy de bimodalidade, sem acessar `true_context`.

### 3.5 Fragmentação metodológica: SSL do agregador não está integrado ao pipeline end-to-end

O `run_ssl_aggregator_sanity.py` testa o SSL do agregador separadamente e depois avalia D6. O `run_synthetic_pipeline.py` tem suporte para `--set-training ssl`, mas os resultados documentados no `synthetic_results_current.md` mostram apenas `set:bidirectional` com supervisão (`set_training == "supervised"`) e sem treino (`--skip-set-training`).

Os resultados do SSL com `context` (que melhora D6 fortemente segundo o planejamento, mas não está no documento de resultados atuais) parecem vir de um experimento separado não documentado em `synthetic_results_current.md`. O documento de resultados diz:

> "SSL com observable local neighborhood (verb+object via context) improved D6 strongly WITHOUT using true_context"

Mas esse resultado **não aparece nas tabelas** do `synthetic_results_current.md`. Há um gap entre o que o planejamento menciona como resultado recente e o que está documentado formalmente. Isso é uma fragilidade de rastreabilidade científica.

### 3.6 Teacher usa encoder "linear" por padrão — inconsistência com o paper

Em `trajectory_models.py`, linha 56:
```python
class TrajectoryTeacher(nn.Module):
    def __init__(self, d_in, d_traj=32, encoder_variant="linear", max_len=32):
```

O teacher usa `encoder_variant="linear"` por padrão. Mas em `run_d5a_student_ablation.py`, linha 124, o teacher é explicitamente criado como bidirecional:
```python
teacher = TrajectoryTeacher(..., encoder_variant="bidirectional", ...)
```

E em `run_synthetic_pipeline.py`, o teacher usa o `temporal_name` da config (ex: `bidirectional`). Isso é correto, mas o padrão "linear" no construtor é potencialmente enganoso se alguém instanciar o teacher diretamente sem especificar o variant.

O planejamento (Seção 3.4) define o teacher como Variante A (bidirecional) na configuração principal. Um teacher linear é essencialmente um autoencoder sem memória temporal — que o planejamento descreve como "Variante C: lower bound". Usar o teacher linear para gerar os alvos do student bidirecional é uma escolha válida de ablação, mas não é a configuração "principal" que o paper defende.

---

## 4. Avaliação do código e potenciais bugs

### 4.1 Bug potencial: `class_id` na avaliação all_masked em `trajectory_train.py`

Em `evaluate_all_masked_reconstruction` (linhas 191-192):
```python
repeated_class = batch["class_id"].unsqueeze(1).expand_as(batch["valid_mask"])
class_ids.append(repeated_class[batch["valid_mask"]].cpu())
```

`batch["class_id"]` tem shape `(batch_size,)`. `batch["valid_mask"]` tem shape `(batch_size, seq_len)`. O expand e indexing com bool mask produz uma sequência de `class_id` para cada posição válida da sequência, o que é correto — cada posição na trajetória herda o `class_id` do subject. Não é bug, mas é um padrão de código não óbvio que poderia introduzir erros se a shape mudar.

Também vale notar que as `losses` são acumuladas por posição mascarada individualmente (loop interno sobre `pos`), enquanto `class_ids` é expandido por subject. Quando `torch.cat(losses)` e `torch.cat(class_ids)` são feitos no final, **o número de elementos deve coincidir**, mas como a acumulação de losses e class_ids ocorre em loops diferentes (`batch_losses` no loop `pos`, `class_ids` por batch), há risco de desalinhamento se um batch tiver posições completamente inválidas. A lógica `if batch_losses: losses.append(...) ... class_ids.append(...)` é correta para o caso de batch vazio, mas dentro de um batch com mistura de posições válidas e inválidas, o número de elementos pode divergir. **Recomendo uma verificação explícita de alinhamento.**

### 4.2 `context_similarity_contrastive_loss` vs. `supervised_contrastive_loss`

Há duas losses contrastivas no código:
- `aggregator_train.py`: `supervised_contrastive_loss(points, labels)` — usa `true_context` como labels
- `aggregator_ssl.py`: `context_similarity_contrastive_loss(points, context)` — usa similaridade de embeddings de co-ocorrentes como "soft labels"

A segunda é genuinamente auto-supervisionada. A diferença entre elas é fundamental para a tese do paper, e o código as separa corretamente. Porém, os nomes das funções não comunicam bem a distinção — `context_similarity_contrastive_loss` soa como que poderia usar `true_context`. Recomendo renomear para `cooccurrence_contrastive_loss` ou similar.

### 4.3 `_sample_subset` na loss de consistência — ruído artificial

Em `aggregator_ssl.py`, linhas 105-108:
```python
view_a = _sample_subset(h)
view_b = _sample_subset(h)
pool_a = aggregator(view_a)["R"]
pool_b = aggregator(view_b)["R"]
consistency = 1.0 - F.cosine_similarity(pool_a, pool_b, dim=0)
```

A consistência mede se o agregador produz outputs similares para subsets aleatórios do mesmo grupo. Isso é uma forma de augmentation SSL bem motivada. Mas `_sample_subset` sempre mantém 75% das ocorrências (`keep = max(2, int(round(0.75 * n)))`), o que significa que os dois subsets se sobrepõem muito para grupos pequenos. Para grupos de tamanho 4 (8 * 0.5 split, 4 por período em alguns casos), os dois subsets podem ser idênticos, zerando a perda de consistência. Não é um bug crítico, mas a força do sinal depende do tamanho do grupo.

### 4.4 `MeanAggregator` não retorna "U" — tratamento inconsistente em D6

Em `aggregators.py`:
```python
class MeanAggregator(nn.Module):
    def forward(self, occurrences):
        return {"R": occurrences.mean(dim=0)}
```

Em `trajectory_metrics.py`, `d6_bimodality_silhouette` (linha 141):
```python
if aggregator is not None:
    out = aggregator(h)
    points = out.get("U", h).detach().cpu().numpy()
else:
    points = h.detach().cpu().numpy()
```

Para `MeanAggregator`, o chamador passa `aggregator=None` (veja `run_synthetic_pipeline.py` linha 104):
```python
d6 = d6_bimodality_silhouette(
    reps,
    aggregator=None if aggregator_name == "mean" else aggregator,
    device=args.device,
)
```

Isso é correto e consistente com o planejamento (D6 para mean pooling usa as ocorrências brutas `h_s^i(t)` como baseline). Mas é um padrão frágil: se alguém passar um `MeanAggregator` explícito como argumento, `out.get("U", h)` retornaria `h` corretamente (porque "U" não existe), mas isso é dependente de `h` ainda estar em escopo — o que está, porque `h = group["h"].to(device_t)` na linha 138. Não é bug, mas é frágil.

### 4.5 `variance_regularizer` aplica `relu(gamma - std)` — penaliza baixa variância, não alta

Em `trajectory_losses.py` (linha 43):
```python
return F.relu(gamma - std).mean()
```

Isso retorna `max(0, γ - std)`, que é 0 quando `std >= γ`. Isso significa que a regularização de variância **penaliza quando a variância é baixa** (std < γ), forçando os vetores a ocupar o espaço. Isso é correto para evitar colapso — é o comportamento VICReg desejado. OK.

### 4.6 `linear_cka` com mask 3D — comportamento com tensors 3D

Em `trajectory_losses.py`, `linear_cka` (linhas 19-24):
```python
if mask is not None:
    x = x[mask]
    y = y[mask]
else:
    x = x.reshape(-1, x.size(-1))
    y = y.reshape(-1, y.size(-1))
```

`x` e `y` são tensors 3D `(n_subjects, seq_len, d_model)`. `mask` é `valid_mask` de shape `(n_subjects, seq_len)`. O indexing `x[mask]` produz shape `(n_valid_positions, d_model)`, que é o correto. Sem mask, `reshape(-1, ...)` achataria todos os períodos, incluindo os inválidos/padded — o que inflaria o CKA artificialmente. O uso de `valid_mask` é portanto necessário e está sendo usado corretamente em todas as chamadas de `cka_metric`.

---

## 5. Avaliação das métricas

### 5.1 D2 (context drift score)
**Implementação:** Em `trajectory_metrics.py`, `d2_context_drift_metrics`, os scores são cosine axis scores calculados contra protótipos globais. O `delta` é `score[-1] - score[0]` (diferença entre último e primeiro período com ocorrência válida). O Spearman é calculado entre os scores e `p_n1`.

**Adequação:** Para as classes Drift e Stable, é uma métrica razoável. Para Abrupt, o delta captura a mudança mas é insensível à *velocidade* da transição — duas palavras com transição no período 3 e no período 8 terão o mesmo delta se os valores inicial e final coincidirem. Isso está reconhecido implicitamente na escolha de D5a (masked reconstruction) como métrica primária para Abrupt.

**Problema:** Conforme identificado em 3.2, os protótipos são calculados a partir de dados de avaliação. No sintético, `p_n1 >= 0.5` é uma heurística de separação limpa e justificável. No COHA, seria necessário construir os protótipos de outra forma.

### 5.2 D5a (masked reconstruction)
**Implementação:** Em `evaluate_all_masked_reconstruction`, cada posição válida é mascarada independentemente e a loss MSE entre a predição do student e o target do teacher é registrada. Os class_ids são propagados corretamente para calcular a loss por classe.

**Adequação:** A comparação D5a entre variantes de student é **justa** porque: (a) todos os students são treinados contra o mesmo teacher congelado, (b) a avaliação usa o mesmo conjunto de dados e mascara todas as posições válidas, não apenas uma amostra. Isso está bem implementado.

**Cuidado:** D5a mede reconstrução de `M_teacher(t_k)`, não de `R_s(t_k)`. Se o teacher tiver aprendido algo degenerado (CKA(M,R) próximo de 1), um student trivialmente bom seria aquele que apenas copia a entrada — o que é exatamente o que a regularização anti-identidade tenta evitar. A Verificação de Sanidade 0 (CKA(M,R) << 1) é, portanto, pré-requisito para que D5a seja uma métrica informativa. Parece estar sendo verificada (`teacher_sanity_metrics` é chamado no pipeline), mas os resultados dessa verificação não estão explicitamente reportados nas tabelas atuais.

### 5.3 D6 (bimodalidade)
**Implementação:** Silhouette com `metric="cosine"` sobre `U` (para Set Transformer) ou `h` (para mean pooling), com `true_context` como labels. Calcula apenas nos períodos da segunda metade (`late_half_only=True`), quando Bifurcating já está no plateau de dois contextos.

**Adequação:** `late_half_only=True` é uma escolha razoável — nos primeiros períodos, Bifurcating ainda é praticamente N1-only, então o silhouette seria artificialmente baixo por falta de exemplos N2. A restrição à segunda metade captura o estado em que os dois contextos coexistem.

**Problema crítico na comparação mean vs. set:** Para mean pooling, D6 é calculado sobre `h_s^i(t)` (representações individuais antes de qualquer agregação). Para Set Transformer, é calculado sobre `U` (representações após a atenção cross-occurrence). Se o SetTransformer tem D6 alto, isso pode significar apenas que a atenção transformou as representações de forma que os clusters ficam mais separados geometricamente — **sem que isso implique que R_s(t) (o vetor agregado) preserva bimodalidade**. O planejamento reconhece que U é a etapa intermediária, mas a conexão entre "U tem bimodalidade alta" e "R_s captura bimodalidade para o encoder temporal" não é direta.

### 5.4 Comparação D5a entre students: fairness
A configuração em `run_d5a_student_ablation.py` é correta: mesmo teacher bidirecional congelado, mesmos dados, mesma sequência de representações. Os três students (bidirectional, causal, linear) são comparados contra o mesmo alvo. Isso é genuinamente fair.

**Potencial problema não verificado:** O teacher em `run_d5a_student_ablation.py` usa `d_in=args.d_model` hardcoded (não `d_aggregated`). Para `set_slots`, isso seria erro de shape. Para `set` sem slots e `mean`, coincide com `d_model`. O script atualmente só aceita `--aggregator mean/attention/set` (não `set_slots`), então o bug não é ativado, mas é latente.

---

## 6. Recomendações para próximos passos

### 6.1 Antes de avançar para COHA — obrigatório

**A. Documentar e verificar explicitamente a Verificação de Sanidade 0** nos resultados atuais. As tabelas em `synthetic_results_current.md` reportam D6, D2, D5a — mas não reportam `teacher_cka_M_R` nem `teacher_probe_r2_M vs teacher_probe_r2_R`. Se o teacher CKA for alto (> 0.7, que é o próprio limiar τ_cka), a regularização não está funcionando e D5a mede reconstrução de uma identidade aproximada.

**B. Resolver a ambiguidade de dimensão do `SetSlotsAggregator`.** Decidir se `R` é `mean(slots)` (mantendo `d_model`) ou `flatten(slots)` (dobrando a dimensão). Se o objetivo é múltiplos protótipos, considerar uma pooling não-linear dos slots em vez de concatenação.

**C. Documentar formalmente os resultados do SSL com vizinhança local** em `synthetic_results_current.md`. Os resultados que o documento de planejamento menciona ("SSL with local neighborhood improved D6 strongly") não aparecem nas tabelas formais. Isso é um problema de rastreabilidade.

**D. Corrigir ou documentar o `d_in` hardcoded em `run_d5a_student_ablation.py`** (linha 122). Mesmo que `set_slots` não seja usado hoje, o bug latente deve ser explicitado ou corrigido.

### 6.2 Qual variante usar como configuração principal?

Com base nos resultados atuais e nos problemas identificados:

- `mean:bidirectional` é o baseline honesto sem nenhum risco de leakage ou problemas arquiteturais. Tem D6 = 0.130 (baixo, esperado), mas D2 razoável.

- `set:bidirectional` com supervisão tem D6 alto, mas usa `true_context` no treino do agregador — **não pode ser a configuração principal do paper**. Serve como upper bound.

- `set:bidirectional` sem treino tem D6 = 0.130 (idêntico ao mean), ou seja, a arquitetura sozinha não traz bimodalidade.

- `set_ssl:bidirectional` (com contrastive loss sobre embeddings de co-ocorrentes) é o candidato mais promissor como configuração self-supervised, mas **os resultados não estão documentados nas tabelas formais** ainda.

**Recomendação:** Formalizar os experimentos com `set_ssl` em 10+ seeds antes de tomar qualquer decisão. Se `set_ssl` conseguir D6 > 0.20 sem usar `true_context`, isso é o resultado central do paper. Se não, a tese sobre bimodalidade precisa ser reformulada ou limitada.

### 6.3 Experimento mínimo de maior confiança

Um único experimento end-to-end com:
- 10 seeds
- `set_ssl:bidirectional` como configuração principal
- `mean:bidirectional` como baseline
- `set_supervised:bidirectional` como upper bound
- Reportando explicitamente: D6, D5a por classe, `teacher_cka_M_R`, `teacher_probe_r2_M`, `teacher_probe_r2_R`
- Comparando fidelidade 0.75 e 0.50

Este experimento cobrirá todas as verificações de sanidade e ablações principais em uma única rodada coerente.

---

## 7. Veredicto

### 7.1 Direção científica geral
**Prosseguir** — a direção é correta. A formulação `token@time = (h_s, m_s)` é bem motivada, o pipeline é arquiteturalmente defensável, e os resultados de D5a validam a hipótese central sobre bidirecionalidade para Abrupt.

### 7.2 Estado atual do código
**Pausar para corrigir** antes de avançar para o COHA. Os problemas são:
1. `run_d5a_student_ablation.py` tem `d_in` hardcoded inconsistente com o pipeline principal (linha 122)
2. Os resultados de `set_ssl` não estão documentados formalmente
3. A relação entre o sinal `context` (embeddings de co-ocorrentes) e `true_context` não está quantificada

### 7.3 Principais riscos de overclaiming
- Afirmar que "SSL preserva bimodalidade" com base apenas na Sanidade 2 atual, sem experimentos multi-seed com `set_ssl`
- Apresentar D2 como métrica cega quando ela usa ground truth (`p_n1`) para construir o eixo de comparação
- Apresentar D6 como "unsupervised bimodality detection" quando usa `true_context` como labels do silhouette

### 7.4 O que um revisor crítico atacaria primeiro
1. "Seu D2 é computado com ground truth como referência — isso não é uma métrica cega"
2. "Seu melhor resultado de D6 usa supervisão (`true_context`); o resultado sem supervisão é igual ao baseline mean pooling"
3. "O sinal SSL que você chama de self-supervised é baseado em embeddings do seu próprio modelo semântico — não é independente do step de pré-treinamento"
4. "Por que o teacher é às vezes linear e às vezes bidirecional — e qual é a configuração principal?"

Esses pontos **têm respostas defensáveis** (D2 é benchmark sintético controlado; set_ssl usa embeddings observáveis; teacher bidirecional é a configuração principal), mas precisam ser articulados explicitamente e documentados em código e no paper.

---

## Apêndice: Localização de problemas no código

| Problema | Arquivo | Linha(s) |
|---|---|---|
| `d_in` hardcoded no D5a | `scripts/run_d5a_student_ablation.py` | 122 |
| `context` = embeddings (não `true_context`) | `src/timeformers/representations.py` | 33 |
| Protótipos de D2 calculados de toda a split | `src/timeformers/trajectory_metrics.py` | 47-56 |
| `SetSlotsAggregator` retorna dimensão dupla | `src/timeformers/aggregators.py` | 73 |
| Teacher default = `linear` (não bidirecional) | `src/timeformers/trajectory_models.py` | 56 |
| D6 usa `true_context` como labels de silhouette | `src/timeformers/trajectory_metrics.py` | 134 |
| Resultados SSL não documentados formalmente | `docs/synthetic_results_current.md` | — |
