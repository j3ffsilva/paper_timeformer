# Handoff — próxima máquina (GPU)

**Data:** 2026-06-05  
**Estado:** repositório limpo, pronto para rodar

---

## O que este projeto faz

Estamos estudando **mudança semântica temporal** com um Transformer treinado
continuamente em ordem cronológica, sem nenhum sinal explícito de período.

A arquitetura ("Timeformer") é:

```
theta_0 = treino(D_0)
theta_1 = continua_treino(theta_0, D_1)
...
```

Mudança semântica é medida por **perfil relacional** de cada palavra:

```
R_t(w)[v] = log( q_t(w)[v] / p_t[v] )
```

Onde:
- `q_t(w)` = distribuição média do MLM head sobre ocorrências reais de `w` com
  `w` mascarada, no checkpoint `theta_t`
- `p_t` = distribuição do MLM head com probe neutro `[CLS] [MASK] [SEP]`
  (prior sem informação de palavra-alvo)

O deslocamento semântico entre dois checkpoints é:

```
Delta(w) = 1 - cos(R_t0(w), R_t1(w))        # pmi_cosine
Delta(w) = JSD(PPMI_t0(w), PPMI_t1(w))       # ppmi_jsd
```

Formalização completa em `docs/relational_profile_formalization.md`.

---

## Dataset

SemEval-2020 Task 1 inglês lematizado. Dois períodos:

```
data/processed/semeval2020_task1/eng_lemma/corpus/1810-1860.txt   (~410k janelas)
data/processed/semeval2020_task1/eng_lemma/corpus/1960-2010.txt   (~420k janelas)
data/processed/semeval2020_task1/eng_lemma/targets.txt            (37 alvos)
data/processed/semeval2020_task1/eng_lemma/truth.tsv              (gold SemEval)
```

---

## O que executar

### Passo 1 — Experimento principal

```bash
python scripts/run_diachronic_relational_experiment.py \
    --input-dir data/processed/semeval2020_task1/eng_lemma/corpus \
    --targets data/processed/semeval2020_task1/eng_lemma/targets.txt \
    --profile-mode pmi \
    --probe-mode occurrence \
    --d-model 96 --layers 2 --heads 4 --d-ff 192 \
    --base-epochs 3 --epochs-per-period 2 \
    --batch-size 256 \
    --device cuda \
    --output-dir outputs/semeval2020_pmi_pilot
```

Produz: `outputs/semeval2020_pmi_pilot/diachronic_relational_changes.csv`  
Colunas relevantes: `target`, `comparison`, `pmi_cosine`, `ppmi_jsd`, `direct_jsd`

### Passo 2 — Avaliação contra gold SemEval

```bash
# Score principal: pmi_cosine
python scripts/evaluate_semeval2020_relational.py \
    --predictions outputs/semeval2020_pmi_pilot/diachronic_relational_changes.csv \
    --truth data/processed/semeval2020_task1/eng_lemma/truth.tsv \
    --score-column pmi_cosine \
    --comparison from_t0 \
    --output-dir outputs/semeval2020_pmi_pilot/eval_pmi_cosine

# Score alternativo: ppmi_jsd
python scripts/evaluate_semeval2020_relational.py \
    --predictions outputs/semeval2020_pmi_pilot/diachronic_relational_changes.csv \
    --truth data/processed/semeval2020_task1/eng_lemma/truth.tsv \
    --score-column ppmi_jsd \
    --comparison from_t0 \
    --output-dir outputs/semeval2020_pmi_pilot/eval_ppmi_jsd
```

Reporta: `spearman_graded`, `roc_auc_binary`, `average_precision_binary`

### Passo 3 — Diagnóstico de artefatos

```bash
python scripts/diagnose_semeval2020_relational.py \
    --predictions outputs/semeval2020_pmi_pilot/diachronic_relational_changes.csv \
    --profile-dir outputs/semeval2020_pmi_pilot/profiles/prediction_anchor_js \
    --score-column pmi_cosine \
    --output-dir outputs/semeval2020_pmi_pilot/diagnostics_pmi
```

Verifica correlação do score com frequência e entropia (detecta artefatos).

---

## O que observar nos resultados

### Sinal esperado (se o modelo aprendeu algo)

Para palavras com mudança semântica real (`graft_nn`, `tip_vb`, `prop_nn`):
- `pmi_cosine` > 0.05 e maior que para palavras estáveis

Para palavras estáveis (`chairman_nn`, `ball_nn`, `face_nn`):
- `pmi_cosine` próximo de zero ou pequeno

### Diagnóstico rápido: inspecionar top âncoras por palavra

```python
import torch, json

anchors = json.load(open('outputs/semeval2020_pmi_pilot/anchors.json'))
for period in ['t00', 't01']:
    p = torch.load(f'outputs/semeval2020_pmi_pilot/profiles/prediction_anchor_js/{period}.pt',
                   weights_only=True)
    targets = p['targets']
    # pmi_profiles não está no perfil de âncoras — ver full_distributions no pt
    # Para inspecionar: carregar o profile e ver distributions (anchor-restricted)
    for word in ['graft_nn', 'tip_vb', 'chairman_nn']:
        if word in targets:
            idx = targets.index(word)
            dist = p['distributions'][idx]
            top5 = torch.topk(dist, 5)
            words = [anchors[i] for i in top5.indices.tolist()]
            print(f'{period} {word}: {words}')
```

Se `graft_nn` em t00 mostrar palavras agrícolas e em t01 palavras médicas/políticas,
o modelo está funcionando.

Se todos os alvos mostrarem as mesmas palavras funcionais (`the`, `a`, `of`),
o modelo ainda é fraco — aumentar `--base-epochs` e `--epochs-per-period`.

### Problema conhecido: âncoras ainda incluem palavras funcionais

O script atual seleciona âncoras por frequência bruta — pode incluir `a`, `the`, etc.
Se o diagnóstico manual mostrar isso, filtrar as âncoras por POS no prepare script
ou passar `--max-anchors 300` com uma lista filtrada manualmente.

---

## Contexto das iterações anteriores

### O que não funcionou

1. **Probe artificial `[CLS] word [MASK] [SEP]`**: incompatível com o treinamento MLM
   (que mascara posições centrais de janelas longas). Descartado.

2. **Score sobre matriz target×target 37×37**: media como o target se moveu em
   relação aos outros 36 targets — espaço de referência minúsculo e heterogêneo.
   Descartado como score principal.

3. **Modelo pequeno (d_model=32, 1 camada)**: produzia `direct_jsd` comprimido em
   ~0.003 para todos os alvos — sem discriminação possível.

4. **Âncoras com palavras funcionais**: dominavam as distribuições, tornando
   `q_t(w) ≈ p_t` para todos os alvos.

### O que foi corrigido na implementação atual

- Probe por ocorrência real (`RealTargetOccurrenceDataset`): mascara a palavra-alvo
  em seu contexto real do corpus
- Score direto `JSD(q_t0(w), q_t1(w))` sem passar pela matriz target×target
- Perfil log-PMI `R_t(w)[v] = log(q_t(w)[v] / p_t[v])` sobre vocabulário completo,
  com `p_t` do probe neutro
- Modelo maior: d_model=96, 2 camadas (padrão do `RealStaticMLM`)

---

## Arquivos relevantes

```
src/timeformers/real_corpus.py          # RealTargetOccurrenceDataset, vocabulário
src/timeformers/real_models.py          # RealStaticMLM
src/timeformers/relational.py           # log_pmi_profiles, pmi_cosine_displacement, ppmi_jsd_displacement
scripts/run_diachronic_relational_experiment.py   # pipeline completo
scripts/evaluate_semeval2020_relational.py        # avaliação gold
scripts/diagnose_semeval2020_relational.py        # diagnóstico de artefatos
docs/relational_profile_formalization.md          # formalização matemática
docs/relational_change_current_plan.md            # plano e resultados anteriores
```
