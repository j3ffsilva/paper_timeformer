# Organização de Dados e Saídas

Este projeto separa dados, saídas experimentais e materiais temporários para
manter a evolução do Paper 2 reprodutível.

## Diretórios

- `data/raw/`: datasets baixados sem transformação. Não é versionado.
- `data/processed/`: datasets convertidos para o formato esperado pelos scripts.
  Não é versionado.
- `data/smoke/`: corpora pequenos usados apenas para testes de integração. Não é
  versionado.
- `outputs/`: checkpoints, métricas, perfis relacionais e tabelas produzidas por
  experimentos. Não é versionado.
- `outputs/smoke/`: saídas de testes pequenos e baratos.
- `tmp/`: prompts, pareceres externos e notas transitórias.
- `docs/`: planejamento, memorandos científicos e documentação versionada.

## Dataset Real Atual

O dataset real escolhido para substituir COHA é o SemEval-2020 Task 1, por ser
gratuito e ter rótulos gold de mudança semântica lexical.

Layout baixado:

```text
data/raw/semeval2020_task1/
  semeval2020_ulscd_posteval/
  semeval2020_ulscd_posteval.zip
  semeval2020_ulscd_eng/
  semeval2020_ulscd_eng.zip
```

Arquivos relevantes do recorte inglês:

```text
data/raw/semeval2020_task1/semeval2020_ulscd_eng/
  corpus1/lemma/ccoha1.txt.gz
  corpus2/lemma/ccoha2.txt.gz
  targets.txt
  truth/binary.txt
  truth/graded.txt
```

Preparação para o runner diacrônico:

```bash
python3 scripts/prepare_semeval2020_task1.py
```

Para gerar uma versão com âncoras de conteúdo, removendo palavras funcionais
frequentes:

```bash
python3 scripts/prepare_semeval2020_task1.py \
  --output-dir data/processed/semeval2020_task1/eng_lemma_content \
  --exclude-stopwords \
  --anchor-alpha-only \
  --anchor-min-count 50 \
  --anchor-min-length 3 \
  --anchor-max-period-fraction 0.005 \
  --max-anchors 300
```

Layout processado esperado:

```text
data/processed/semeval2020_task1/eng_lemma/
  corpus/
    1810-1860.txt
    1960-2010.txt
  targets.txt
  anchors.txt
  truth.tsv
  metadata.json
```

## Smoke Diacrônico

O corpus mínimo usado para validar a integração do pipeline fica em:

```text
data/smoke/diachronic_relational/
  corpus/
    1950.txt
    1980.txt
    2000.txt
  targets.txt
  anchors.txt
```

A saída correspondente fica em:

```text
outputs/smoke/diachronic_relational/
```

Esse smoke não é evidência científica. Ele serve apenas para verificar que o
treino contínuo, os probes relacionais e a exportação de deltas funcionam de
ponta a ponta.

## Avaliação SemEval

Depois de rodar `scripts/run_diachronic_relational_experiment.py`, o ranking
relacional pode ser comparado com o gold do SemEval:

```bash
python3 scripts/evaluate_semeval2020_relational.py \
  --predictions outputs/semeval2020_eng_lemma_pilot_2k/diachronic_relational_changes.csv \
  --truth data/processed/semeval2020_task1/eng_lemma/truth.tsv \
  --output-dir outputs/semeval2020_eng_lemma_pilot_2k/eval_mean
```

O avaliador reporta Spearman contra o score graduado e métricas binárias
contra o rótulo de mudança.

Para diagnosticar se o ranking está capturando mudança semântica ou artefatos
de frequência/cobertura, use:

```bash
python3 scripts/diagnose_semeval2020_relational.py \
  --predictions outputs/semeval2020_eng_lemma_pilot_10k/diachronic_relational_changes.csv \
  --profile-dir outputs/semeval2020_eng_lemma_pilot_10k/profiles/prediction_anchor_js \
  --output-dir outputs/semeval2020_eng_lemma_pilot_10k/diagnostics_max
```

O diagnóstico une score relacional, gold do SemEval, frequência por período e
entropia normalizada das distribuições sobre âncoras.

O modo principal do runner para corpus real é o probe por ocorrência:

```bash
python3 scripts/run_diachronic_relational_experiment.py \
  --probe-mode occurrence \
  --max-probe-occurrences-per-target 500 \
  ...
```

Nesse modo, cada ocorrência real da palavra-alvo é mascarada em sua janela
original, e as distribuições previstas sobre âncoras são agregadas por
palavra/período. O score principal recomendado é `direct_jsd`, isto é,
`JSD(q_t0(w), q_t1(w))` calculado diretamente entre os dois perfis de âncoras
da mesma palavra.

O runner também exporta `top_anchors.csv`, com as âncoras mais prováveis por
alvo e período. Esse arquivo deve ser inspecionado antes de interpretar
qualquer métrica agregada.

## Convenção de Scripts

- Scripts sintéticos continuam usando `synthetic` no nome.
- Scripts aplicáveis a corpus real usam `diachronic`, não nomes de datasets
  específicos.
- Scripts específicos de preparação de dataset devem usar o nome do dataset,
  por exemplo `prepare_semeval2020_task1.py`.
