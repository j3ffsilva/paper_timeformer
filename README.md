# paper_timeformer

Laboratorio experimental do paper 2 do Timeformer. Este repositório reimplementa
o necessário do paper 1 em uma base independente, para testar a hipótese antes
de partir para a escrita e para o corpus natural.

## Setup

```bash
venv/bin/pip install -r requirements.txt
```

## Piloto sintético

Smoke test rápido:

```bash
PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
venv/bin/python scripts/run_pilot.py \
  --models TokenTime,FiLM,TokenTimeTraj,FiLMTraj \
  --fidelities 0.75 \
  --seeds 1000 \
  --epochs 1 \
  --examples-per-subject-epoch 2 \
  --output-dir outputs/smoke
```

Piloto curto, ainda barato:

```bash
PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
venv/bin/python scripts/run_pilot.py \
  --models TokenTime,FiLM,TokenTimeTraj,FiLMTraj \
  --fidelities 0.75 \
  --seeds 1000 \
  --epochs 5 \
  --examples-per-subject-epoch 6 \
  --output-dir outputs/pilot_quick
```

Resultados principais são escritos em `outputs/<run>/pilot_results.csv` e
`outputs/<run>/pilot_results.json`.

Para usar protótipos globais `N1/N2` recalculados a cada época, adicione:

```bash
--traj-prototypes global
```

O modo padrão é `--traj-prototypes batch`, que calcula o eixo de trajetória
dentro de cada batch com stop-gradient.

Para a loss de ranking temporal por sujeito:

```bash
--traj-loss rank --traj-margin 0.05
```

Resultado piloto atual: `FiLMTraj` com `--traj-loss rank` e protótipos `batch`
é a configuração mais promissora até agora, pois melhora Spearman de trajetória
preservando melhor a acurácia MLM.

## Análise pareada

Depois de um piloto, gere deltas pareados, IC 95%, teste t e Wilcoxon:

```bash
venv/bin/python scripts/analyze_pilot.py outputs/pilot_rank_10seed/pilot_results.csv
```

Isso escreve `pilot_paired_stats.csv`, `.json` e `.md` no diretório do piloto.

Para testar tendência do gap ao longo da degradação dos marcadores:

```bash
venv/bin/python scripts/analyze_trend.py outputs/synthetic_rank_31seed_7fid_ablation/pilot_paired_stats.csv
```

Isso escreve `pilot_trend_stats.csv`, `.json` e `.md`.

## Figuras

Para gerar curvas por fidelidade, deltas pareados com IC 95% e o gráfico
de trade-off entre acurácia MLM e geometria temporal:

```bash
venv/bin/python scripts/plot_pilot.py outputs/synthetic_rank_31seed_5fid
```

As figuras são escritas em `outputs/<run>/figures/`, em `.png` e `.pdf`.
