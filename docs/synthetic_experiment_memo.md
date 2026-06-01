# Synthetic Experiment Memo — Timeformer Paper 2

Data: 2026-05-30

Este memo congela o primeiro ciclo experimental do paper 2. O objetivo
foi validar a ideia antes de avançar para escrita extensa ou corpus natural:
se o objetivo de trajetória melhora a geometria temporal sem destruir a
competência MLM.

## Pergunta

O modelo completo, `FiLM + L_traj`, aprende representações mais alinhadas
com a trajetória plantada do que modelos condicionados apenas por arquitetura?

Mais especificamente:

1. `FiLMTraj` deve superar `FiLM` em Spearman de trajetória.
2. O ganho geométrico não deve vir acompanhado de queda relevante em MLM.
3. `FiLMTraj` deve ser competitivo com `TokenTimeTraj`, mas preservando
   melhor a acurácia lexical.

## Protocolo Rodado

Diretório do experimento:

`outputs/synthetic_rank_31seed_5fid`

Configuração:

- Modelos: `TokenTime`, `FiLM`, `TokenTimeTraj`, `FiLMTraj`
- Fidelities: `0.50`, `0.5625`, `0.625`, `0.6875`, `0.75`
- Seeds: `31` (`1000` a `1030`)
- Épocas: `5`
- Exemplos por sujeito/época: `6`
- Loss auxiliar: `rank`
- Protótipos de trajetória: `batch`
- `lambda_traj`: `1.0`
- `traj_margin`: default do script

Arquivos principais:

- `pilot_results.csv`: resultados por seed/modelo/fidelity
- `pilot_summary.csv`: médias e desvios por modelo/fidelity
- `pilot_paired_stats.csv`: deltas pareados, IC 95%, t-test e Wilcoxon
- `figures/`: figuras em `.png` e `.pdf`

## Resultado Principal

`FiLMTraj` melhora fortemente a geometria temporal contra `FiLM` em todas
as fidelities.

### FiLMTraj - FiLM

| Fidelity | Δ Spearman Drift | IC 95% | p t-test | Δ Spearman Bifurcating | IC 95% | p t-test | Δ MLM |
|----------|------------------|--------|----------|-------------------------|--------|----------|-------|
| 0.5000 | +0.4812 | [+0.3662, +0.5962] | 1.56e-09 | +0.4527 | [+0.3429, +0.5626] | 2.15e-09 | -0.0030 |
| 0.5625 | +0.5896 | [+0.5082, +0.6710] | 2.53e-15 | +0.5814 | [+0.5065, +0.6563] | 3.99e-16 | -0.0011 |
| 0.6250 | +0.5630 | [+0.5023, +0.6237] | 3.05e-18 | +0.5634 | [+0.5100, +0.6168] | 8.32e-20 | +0.0026 |
| 0.6875 | +0.5108 | [+0.4666, +0.5550] | 6.27e-21 | +0.5159 | [+0.4722, +0.5597] | 3.54e-21 | -0.0040 |
| 0.7500 | +0.4105 | [+0.3720, +0.4490] | 6.06e-20 | +0.4401 | [+0.3928, +0.4875] | 2.87e-18 | -0.0037 |

Leitura: a melhora geométrica é grande, estável e pareada por seed. A
diferença em MLM é pequena. Em quatro dos cinco níveis ela não sugere uma
perda relevante; em `0.6875`, o t-test dá p próximo de 0.05 para uma queda
absoluta de `0.0040`, que é pequena demais para mudar a conclusão, mas deve
ser reportada sem exagero.

## Médias Por Modelo

### Fidelity 0.50

| Modelo | MLM | Spearman Drift | Spearman Bifurcating |
|--------|-----|----------------|----------------------|
| TokenTime | 0.1692 | 0.4470 | 0.3937 |
| FiLM | 0.2377 | 0.1211 | 0.0500 |
| TokenTimeTraj | 0.1791 | 0.7306 | 0.6378 |
| FiLMTraj | 0.2347 | 0.6023 | 0.5027 |

### Fidelity 0.625

| Modelo | MLM | Spearman Drift | Spearman Bifurcating |
|--------|-----|----------------|----------------------|
| TokenTime | 0.1822 | 0.4601 | 0.4019 |
| FiLM | 0.2443 | 0.3382 | 0.2503 |
| TokenTimeTraj | 0.1748 | 0.6496 | 0.5427 |
| FiLMTraj | 0.2469 | 0.9012 | 0.8138 |

### Fidelity 0.75

| Modelo | MLM | Spearman Drift | Spearman Bifurcating |
|--------|-----|----------------|----------------------|
| TokenTime | 0.1789 | 0.6069 | 0.5311 |
| FiLM | 0.2448 | 0.5552 | 0.4375 |
| TokenTimeTraj | 0.1727 | 0.8018 | 0.6914 |
| FiLMTraj | 0.2412 | 0.9657 | 0.8777 |

## Interpretação

O resultado apoia a tese central do paper 2: o gargalo não é apenas inserir
tempo na arquitetura; é treinar o espaço representacional com um objetivo
compatível com trajetórias.

O contraste `FiLMTraj` vs. `FiLM` isola bem o efeito de `L_traj`: mesma
arquitetura, objetivo diferente. O ganho é grande em drift e bifurcating,
com MLM praticamente preservado.

O contraste com `TokenTimeTraj` é mais sutil. Em fidelity `0.50`,
`TokenTimeTraj` atinge Spearman maior que `FiLMTraj`, mas com MLM muito
mais baixo. A narrativa correta não é que `FiLMTraj` maximiza Spearman em
todos os casos. A narrativa mais forte é:

> FiLM + L_traj oferece o melhor compromisso entre competência lexical e
> geometria temporal. Token-Time + L_traj pode forçar geometria em regime
> mais ruidoso, mas paga custo maior em MLM.

A partir de `0.5625`, `FiLMTraj` passa a ser também o melhor modelo em
Spearman médio, além de preservar MLM.

## Crítica Ao Plano Original

O plano em `paper2_journal.md` previa uma varredura completa de 7 níveis
e a hipótese de que o gap cresceria monotonicamente com degradação dos
marcadores.

Este ciclo rodou 5 níveis, não 7. Ele é suficiente para validar a ideia e
desbloquear o paper, mas ainda não deve ser chamado de varredura completa.

Além disso, a monotonicidade do gap não é trivial. Para `FiLMTraj-FiLM`, o
ganho em Spearman é positivo em todos os níveis, mas seu tamanho não cresce
linearmente conforme a fidelity aumenta ou diminui. A conclusão segura é:

- `L_traj` melhora consistentemente a geometria temporal.
- O efeito é robusto à degradação dos marcadores.
- A monotonicidade deve ser tratada como análise adicional, não como achado
  já estabelecido.

## Decisão

O ponto de decisão sintético foi ultrapassado.

Critério planejado:

`FiLM + L_traj > Token-Time + MLM` no sintético em Spearman, sem perda
relevante de MLM.

Resultado:

- Sim para Spearman Drift em todos os níveis.
- Sim para Spearman Bifurcating em quatro níveis de forma clara; em `0.50`,
  o ganho contra `TokenTime` é positivo mas o IC inclui zero.
- Sim para MLM: `FiLMTraj` supera `TokenTime` em acurácia MLM em todos os
  níveis com deltas grandes e significativos.

Portanto, faz sentido avançar.

## Próximos Experimentos Sugeridos

1. Rodar a versão realmente completa da varredura sintética:
   `7 fidelities × 31 seeds`, mantendo a configuração atual.

2. Adicionar `Standard` e `StandardTraj` para testar se o objetivo sozinho
   compensa a ausência de condicionamento temporal explícito.

3. Fazer teste formal de tendência do gap por fidelity. Se mantivermos a
   hipótese de monotonicidade, ela precisa ser testada diretamente.

4. Rodar ablação pequena de `lambda_traj` em torno da configuração vencedora:
   `0.25`, `0.5`, `1.0`, `2.0`.

5. Só depois disso partir para COHA/SemEval, porque o desenho sintético já
   terá virado protocolo fechado e defensável.

---

## Atualização — Varredura Completa Com Ablação Standard

Data: 2026-05-30

Diretório:

`outputs/synthetic_rank_31seed_7fid_ablation`

Configuração:

- Modelos: `Standard`, `StandardTraj`, `TokenTime`, `FiLM`,
  `TokenTimeTraj`, `FiLMTraj`
- Fidelities: `0.50`, `0.5416667`, `0.5833333`, `0.625`,
  `0.6666667`, `0.7083333`, `0.75`
- Seeds: `31`
- Loss auxiliar: `rank`
- Protótipos: `batch`
- `lambda_traj`: `1.0`

Arquivos:

- `pilot_results.csv`
- `pilot_summary.csv`
- `pilot_paired_stats.csv`
- `pilot_trend_stats.csv`
- `figures/`

### Resultado Principal Atualizado

O resultado de 5 níveis se reproduz no protocolo completo de 7 níveis.
`FiLMTraj` supera `FiLM` em Spearman Drift e Bifurcating em todos os
níveis, com deltas grandes e IC 95% sempre acima de zero.

Resumo `FiLMTraj - FiLM`:

| Fidelity | Δ Drift | IC 95% | Δ Bifurcating | IC 95% | Δ MLM |
|----------|---------|--------|---------------|--------|-------|
| 0.5000 | +0.4812 | [+0.3662, +0.5962] | +0.4527 | [+0.3429, +0.5626] | -0.0030 |
| 0.5417 | +0.5535 | [+0.4680, +0.6390] | +0.5471 | [+0.4698, +0.6244] | -0.0034 |
| 0.5833 | +0.5799 | [+0.5060, +0.6538] | +0.5702 | [+0.4965, +0.6440] | -0.0023 |
| 0.6250 | +0.5630 | [+0.5023, +0.6237] | +0.5634 | [+0.5100, +0.6168] | +0.0026 |
| 0.6667 | +0.5355 | [+0.4887, +0.5823] | +0.5405 | [+0.4972, +0.5838] | +0.0031 |
| 0.7083 | +0.4741 | [+0.4310, +0.5173] | +0.4897 | [+0.4497, +0.5297] | +0.0008 |
| 0.7500 | +0.4105 | [+0.3720, +0.4490] | +0.4401 | [+0.3928, +0.4875] | -0.0037 |

Leitura: o efeito geométrico é robusto, grande e replicado. A diferença
em MLM continua pequena e sem custo material para a narrativa.

### Ablação StandardTraj

`StandardTraj` não resolve o problema sozinho. Ele melhora pouco sobre
`Standard` em alguns níveis, mas fica muito abaixo de `FiLMTraj`.

Exemplo em `fidelity=0.75`:

| Modelo | MLM | Spearman Drift | Spearman Bifurcating |
|--------|-----|----------------|----------------------|
| Standard | 0.2374 | 0.4699 | 0.3801 |
| StandardTraj | 0.2455 | 0.4972 | 0.3865 |
| FiLMTraj | 0.2412 | 0.9657 | 0.8777 |

Interpretação: o objetivo ajuda, mas não substitui condicionamento temporal
explícito. O paper deve defender complementaridade entre arquitetura e loss,
não apenas supremacia da loss.

### Teste De Tendência

Foi gerado `pilot_trend_stats.md` usando a média dos deltas pareados por
fidelity, com `noise = 1 - fidelity`.

Resultado:

- `FiLMTraj-FiLM`, Drift: slope/noise `+0.3559`, p `0.2239`
- `FiLMTraj-FiLM`, Bifurcating: slope/noise `+0.1562`, p `0.5694`
- Spearman/Kendall também não confirmam monotonicidade.

Conclusão: a hipótese de monotonicidade do gap não está sustentada neste
ciclo. A afirmação defensável é robustez à degradação, não crescimento
monotônico do gap.

### Decisão Atualizada

O protocolo sintético controlado está suficientemente forte para avançar
para a próxima fase. Antes do corpus natural, ainda vale uma ablação pequena
de `lambda_traj`, mas ela é ajuste de hiperparâmetro, não bloqueador da tese.

Figuras da varredura completa:

`outputs/synthetic_rank_31seed_7fid_ablation/figures`

## Figuras Geradas

Diretório:

`outputs/synthetic_rank_31seed_5fid/figures`

Figuras:

- `spearman_drift_by_fidelity`
- `spearman_bifurcating_by_fidelity`
- `mlm_accuracy_by_fidelity`
- `paired_deltas_filmtraj`
- `mlm_vs_trajectory_tradeoff`
