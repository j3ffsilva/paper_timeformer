# Avaliação de Maturidade do Código — Timeformers
**Data:** 2026-06-03
**Escopo:** `src/timeformers/` (16 arquivos, ~1.741 linhas)

---

## Resumo executivo

O código está bem acima da média de projetos de pesquisa. A decomposição modular é coerente, as responsabilidades dos arquivos são claras, e há cuidado com tipagem, constantes centralizadas e separação entre treino supervisionado e SSL. Os problemas existentes são rastreáveis e corrigíveis sem refatoração profunda.

**Nota geral: B+**
Pronto para publicação de código acompanhando paper. Não pronto para ser um pacote reutilizável.

---

## 1. O que está bem

### Decomposição modular clara

O pipeline tem uma lógica de camadas bem definida:

```
corpus.py → dataset.py → models.py → representations.py
  → aggregators.py → trajectories.py → trajectory_models.py
  → {losses,trajectory_losses}.py → {train,trajectory_train}.py
  → {metrics,trajectory_metrics}.py
```

Cada arquivo tem uma responsabilidade única e tamanho razoável (< 200 linhas). Nenhum arquivo faz tudo.

### Constantes centralizadas

`dataset.py` centraliza `POS_SUBJECT`, `POS_VERB`, `POS_OBJECT`, `SEQ_LEN`, `VOCAB_SIZE`, `TOKEN2ID` etc. Outras partes do código importam dali sem redefinir. `corpus.py` concentra a geração do corpus. Isso evita magic numbers.

### Convenções consistentes

- `from __future__ import annotations` em todos os arquivos.
- Tipagem presente e coerente: `dict[str, Tensor]` como contrato de retorno de módulos, `nn.Module` nas assinaturas.
- Decoradores `@torch.no_grad()` aplicados corretamente em todas as funções de avaliação/extração.
- `registry + factory`: `MODEL_REGISTRY`/`build_model` e `build_aggregator` evitam `if/elif` espalhado nos scripts.
- `dataclass(frozen=True)` para `Example` e `TrajectorySequences` — imutabilidade explícita onde faz sentido.

### Separação entre modos (supervisionado vs. SSL)

`aggregator_train.py` e `aggregator_ssl.py` separam corretamente as duas formas de treinar o agregador. O código não mistura os dois modos em um único arquivo com flags condicionais.

### Chaves de dicionário como contrato

`AGG_KEYS` em `aggregators.py` e `REP_KEYS` em `representations.py` documentam o contrato do dict de saída, garantindo que `save_representations`/`load_representations` valide as chaves na carga.

---

## 2. Duplicações concretas

### 2.1 Bloco de protótipos duplicado em `losses.py`

`trajectory_axis_loss` e `trajectory_ranking_loss` compartilham o mesmo bloco de cálculo de protótipos, copiado palavra por palavra:

```python
# trajectory_axis_loss, linhas 24-34
n1_mask = true_context == 0
n2_mask = true_context == 1
if int(n1_mask.sum()) < 2 or int(n2_mask.sum()) < 2:
    return h_subj.new_zeros(())
detached = h_subj.detach()
proto_n1 = detached[n1_mask].mean(dim=0, keepdim=True)
proto_n2 = detached[n2_mask].mean(dim=0, keepdim=True)
...
proto_n1 = proto_n1.to(device=h_subj.device, dtype=h_subj.dtype).view(1, -1)
proto_n2 = proto_n2.to(device=h_subj.device, dtype=h_subj.dtype).view(1, -1)

# trajectory_ranking_loss, linhas 51-62 — bloco idêntico
```

**Fix:** extrair `_compute_batch_prototypes(h_subj, true_context) -> tuple | None`.

### 2.2 `CLASS_NAMES` definido duas vezes

```python
# metrics.py, linha 13
CLASS_NAMES = {0: "stable", 1: "drift", 2: "bifurcating", 3: "abrupt"}

# trajectory_metrics.py, linha 14
CLASS_NAMES = {0: "stable", 1: "drift", 2: "bifurcating", 3: "abrupt"}
```

`corpus.py` já tem `SUBJECT_CLASSES = ("stable", "drift", "bifurcating", "abrupt")`. A solução é derivar de lá:

```python
# corpus.py (adicionar)
CLASS_ID = {name: i for i, name in enumerate(SUBJECT_CLASSES)}

# metrics.py e trajectory_metrics.py (substituir)
from .corpus import SUBJECT_CLASSES
CLASS_NAMES = {i: name for i, name in enumerate(SUBJECT_CLASSES)}
```

### 2.3 Loop de treinamento replicado em 4 lugares

O padrão abaixo aparece em `train.py`, `trajectory_train.py`, `aggregator_train.py` e `aggregator_ssl.py`:

```python
history = []
for epoch in range(n_epochs):
    totals = {...: 0.0}
    n_batches = 0  # ou n_groups
    for batch in loader:
        ...
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(..., 1.0)
        opt.step()
        for key, value in parts.items():
            totals[key] += ...
        n_batches += 1
    record = {"epoch": epoch, **{k: v / max(n_batches, 1) for k, v in totals.items()}}
    history.append(record)
    if verbose and (epoch == 0 or epoch == n_epochs - 1 or (epoch + 1) % 10 == 0):
        print(...)
```

A condição de verbose `(epoch == 0 or epoch == n_epochs - 1 or (epoch + 1) % 10 == 0)` aparece 3 vezes idêntica.

Para um código de pesquisa, tolerável. Se o projeto crescer, vale extrair um `TrainingLoop` genérico. Por agora, pelo menos a condição de verbose pode ser:

```python
def _should_log(epoch: int, n_epochs: int, every: int = 10) -> bool:
    return epoch == 0 or epoch == n_epochs - 1 or (epoch + 1) % every == 0
```

### 2.4 `clip_grad_norm_` com concatenação de listas de parâmetros

```python
# aggregator_ssl.py, linha 119
nn.utils.clip_grad_norm_(list(aggregator.parameters()) + list(decoder.parameters()), 1.0)

# aggregator_train.py, linha 76
nn.utils.clip_grad_norm_(list(aggregator.parameters()) + list(head.parameters()), 1.0)
```

O idioma correto é `itertools.chain` — evita construir uma lista intermediária:

```python
from itertools import chain
nn.utils.clip_grad_norm_(chain(aggregator.parameters(), decoder.parameters()), 1.0)
```

---

## 3. Nomes inapropriados ou ambíguos

### 3.1 `losses.py` vs `trajectory_losses.py` — o corte está errado

`losses.py` contém `trajectory_axis_loss` e `trajectory_ranking_loss`, que operam sobre representações de **ocorrências** (`h_subj`, shape `(batch, d_model)`).
`trajectory_losses.py` contém `masked_mse`, `linear_cka`, `variance_regularizer`, `anti_identity_loss`, que operam sobre **sequências** (`values`, shape `(batch, seq_len, d_model)`).

O prefixo `trajectory_` em `trajectory_axis_loss` cria a impressão de que ele pertence a `trajectory_losses.py`, mas não pertence. O corte real é:

| Arquivo atual | O que contém | Nome mais claro |
|---|---|---|
| `losses.py` | `mlm_loss`, `trajectory_axis_loss`, `trajectory_ranking_loss` | `occurrence_losses.py` |
| `trajectory_losses.py` | `masked_mse`, `linear_cka`, `variance_regularizer`, `anti_identity_loss` | `sequence_losses.py` |

Renomear os arquivos resolveria a confusão sem mudar nenhuma lógica.

### 3.2 `Trainer` em `train.py` — nome genérico demais

A classe é específica para treino de `BaseModel` (encoder MLM com loss de trajetória opcional). Deveria ser `TokenTimeTrainer` ou `MLMTrainer`. O nome `Trainer` colide com qualquer trainer genérico que alguém possa adicionar depois.

### 3.3 `R` tem dois significados no codebase

Em `aggregators.py`, `R` = representação agregada do período (output do aggregator, shape `(d_model,)` ou `(num_slots * d_model,)`).
Em `trajectory_train.py`, `TrajectoryTeacherTrainer.encode()` salva a entrada bruta das sequências também como `"R"`:

```python
# trajectory_train.py, linha 80
out["R"].append(batch["values"].cpu())
```

Aqui `R` = `values` = o tensor de entrada `R_s(t)` — mas é exatamente o mesmo conceito. O problema é que um dict de saída da função `encode()` tem chaves `{"M", "R", "valid_mask", "p_n1", "class_id"}` onde `R` é "a entrada" e `M` é "a saída do teacher". Isso é coerente com a notação do paper, mas pode confundir alguém que leia o código sem o contexto matemático. Um comentário inline ou docstring curta resolveria.

### 3.4 `OccurrenceDecoder` em `aggregator_ssl.py`

É uma cabeça MLP de reconstrução auxiliar para o SSL — não um "decodificador" no sentido do modelo principal. Nome melhor: `SSLReconHead` ou simplesmente `ReconHead`.

### 3.5 `encoder_variant` em `TrajectoryTeacher` tem default `"linear"`

```python
# trajectory_models.py, linha 56
class TrajectoryTeacher(nn.Module):
    def __init__(self, d_in, d_traj=32, encoder_variant="linear", max_len=32):
```

O default `"linear"` corresponde ao lower bound do ablation. A configuração principal do paper é `"bidirectional"`. Um default que não é a configuração recomendada é uma armadilha silenciosa. Ou mude o default, ou remova o default e force o caller a ser explícito.

---

## 4. Separação de preocupações

### 4.1 `representations.py` — bem separado, com um vazamento menor

`extract_occurrence_representations` faz duas coisas: extrai `h_subj` pelo forward do modelo **e** computa `context` diretamente como `model.token_emb(context_ids).mean(dim=1)`. Isso acopla a extração de `context` ao módulo `representations.py`, exigindo que o modelo tenha `token_emb` acessível.

Isso é OK dado que `context` é um conceito derivado dos embeddings e pertence à extração de representações. Mas o acoplamento ao atributo `token_emb` é implícito — se o modelo mudar o nome do embedding, quebra silenciosamente. Uma alternativa seria passar `context` como output do `model.forward()`, o que centralizaria essa lógica em `models.py`.

### 4.2 `train.py` — `compute_context_prototypes` pertence às métricas

`Trainer.compute_context_prototypes` calcula protótipos N1/N2 a partir de um DataLoader. É uma operação de avaliação/análise, não de treino. Poderia viver em `metrics.py` como uma função pura.

### 4.3 `trajectories.py` — `build_trajectory_sequences` está longa

A função tem 78 linhas com dois loops aninhados. O loop interno que constrói a sequência de um único subject poderia ser extraído:

```python
def _build_subject_sequence(local_R, local_p, epochs, full_epochs, max_len, d_model) -> tuple[Tensor, ...]:
    ...
```

Isso tornaria o loop externo legível e a lógica de interpolação testável isoladamente.

### 4.4 Duplicação conceitual: `evaluate_masked_reconstruction` vs `evaluate_all_masked_reconstruction`

`TrajectoryStudentTrainer` tem dois métodos de avaliação:
- `evaluate_masked_reconstruction` — mascara 1 posição aleatória por subject
- `evaluate_all_masked_reconstruction` — mascara cada posição válida individualmente (D5a)

O primeiro parece ser legacy do desenvolvimento inicial — é chamado em `run_sanity_1_student.py` mas não no pipeline principal. Se ainda é necessário, um comentário explicando a diferença seria útil. Se não, pode ser removido.

---

## 5. Problemas menores de boas práticas

### 5.1 `torch.load` sem `weights_only`

```python
# representations.py, linha 47
loaded = torch.load(path, map_location="cpu")
```

Em PyTorch >= 2.0, isso gera `FutureWarning`. Adicionar `weights_only=True` elimina o aviso e é mais seguro:

```python
loaded = torch.load(path, map_location="cpu", weights_only=True)
```

### 5.2 `_sample_subset` hardcoda 75%

```python
# aggregator_ssl.py, linha 31
keep = max(min_size, int(round(0.75 * h.size(0))))
```

Todos os outros hiperparâmetros do SSL são expostos como parâmetros de `train_set_aggregator_ssl`. A fração 0.75 não é. Deveria ser um parâmetro `subset_fraction: float = 0.75`.

### 5.3 `__init__.py` não exporta nada

```python
# src/timeformers/__init__.py
# 2 linhas, vazias ou com comentário
```

Para código de pesquisa é aceitável. Para quem importar o pacote de fora, `from timeformers import ...` não funcionará sem conhecer os submódulos. Se isso for para acompanhar um paper com código público, vale exportar pelo menos as funções principais do pipeline.

### 5.4 Scripts com boilerplate `argparse` duplicado

`--seed`, `--device`, `--d-model`, `--n-epochs`, `--output-dir` aparecem em todos os scripts. Um `common_args(parser)` ou dataclass de config eliminaria ~30 linhas duplicadas por script.

---

## 6. O que não é problema (embora possa parecer)

- **`R`, `M`, `U` como chaves de dicionário**: notação matemática curta, mas coerente com o paper. OK para código de pesquisa.
- **Ausência de testes unitários**: aceitável dado o contexto, mas `_compute_batch_prototypes`, `build_trajectory_sequences` e `linear_cka` seriam candidatos óbvios a testes simples.
- **Tamanho pequeno dos módulos** (alguns com 60–70 linhas): não é problema, é sinal de boa decomposição.

---

## 7. Lista de ações priorizadas

| Prioridade | Ação | Arquivo(s) |
|---|---|---|
| Alta | Extrair `_compute_batch_prototypes` para eliminar bloco duplicado | `losses.py` |
| Alta | Mover `CLASS_NAMES` para `corpus.py`, importar nos dois lugares | `metrics.py`, `trajectory_metrics.py` |
| Alta | Renomear default `encoder_variant="linear"` → sem default ou `"bidirectional"` | `trajectory_models.py:56` |
| Média | Renomear `losses.py` → `occurrence_losses.py` (e ajustar imports) | `losses.py`, scripts |
| Média | Renomear `Trainer` → `MLMTrainer` | `train.py` |
| Média | Extrair `_should_log(epoch, n_epochs)` para eliminar condição repetida 3x | `train.py`, `trajectory_train.py`, `aggregator_*.py` |
| Média | Adicionar `weights_only=True` no `torch.load` | `representations.py:47` |
| Média | Usar `itertools.chain` em vez de `list(...) + list(...)` para `clip_grad_norm_` | `aggregator_ssl.py:119`, `aggregator_train.py:76` |
| Baixa | Expor `subset_fraction` em `_sample_subset` | `aggregator_ssl.py:31` |
| Baixa | Extrair `_build_subject_sequence` de `build_trajectory_sequences` | `trajectories.py` |
| Baixa | Adicionar exports em `__init__.py` | `src/timeformers/__init__.py` |
