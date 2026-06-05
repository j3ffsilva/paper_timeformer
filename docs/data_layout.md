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

## Convenção de Scripts

- Scripts sintéticos continuam usando `synthetic` no nome.
- Scripts aplicáveis a corpus real usam `diachronic`, não nomes de datasets
  específicos.
- Scripts específicos de preparação de dataset devem usar o nome do dataset,
  por exemplo `prepare_semeval2020_task1.py`.
