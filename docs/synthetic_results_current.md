# Synthetic Results Current

Data: 2026-06-01

Este memo registra o estado atual dos experimentos sintéticos após a
mudança de planejamento para representações explícitas de trajetória.

## Regimes experimentais

Separar estes regimes é essencial para a narrativa do paper:

1. **Principal self-supervised**
   - Encoder semântico Token-Time treinado com MLM.
   - Sequências `Seq_s` construídas a partir de representações por período.
   - Teacher treinado com reconstrução + anti-identidade.
   - Student treinado com masked trajectory distillation.
   - Nenhum rótulo externo de mudança semântica no teacher/student.

2. **Sanidade supervisionada sintética do agregador**
   - Set Transformer treinado com `true_context` (`N1/N2`) no sintético.
   - Serve para testar capacidade arquitetural de preservar bimodalidade.
   - Não deve ser descrito como configuração principal para COHA.

3. **Upper bound supervisionado**
   - Futuro: variantes com `L_sup`.
   - Deve ser reportado como teto, não como método defendido.

## Pipeline end-to-end, 10 seeds

Configuração:

- Seeds: 1000-1009
- Fidelity: 0.75
- Exemplos por sujeito/período: 8
- `d_model=32`, `d_traj=16`
- `semantic_epochs=5`, `teacher_epochs=20`, `student_epochs=20`

### Com Set supervisionado no sintético

| Config | D6 | m Drift | m Bifurcating | m Abrupt | token Drift | token Bifurcating | token Abrupt |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mean:linear` | 0.130 | 0.578 | 0.526 | 0.567 | 0.642 | 0.515 | 0.733 |
| `mean:bidirectional` | 0.130 | 0.690 | 0.650 | 0.679 | 0.745 | 0.667 | 0.753 |
| `set:bidirectional` | **0.298** | **0.793** | **0.725** | **0.702** | **0.845** | **0.774** | **0.793** |

Leitura: o Set treinado com `true_context` melhora fortemente D6 e também
melhora D2 em `m` e em `token_time`. Isto valida a capacidade arquitetural
do Set Transformer no ambiente sintético controlado.

### Set sem treino supervisionado

| Config | D6 | m Drift | m Bifurcating | m Abrupt | token Drift | token Bifurcating | token Abrupt |
|---|---:|---:|---:|---:|---:|---:|---:|
| `set:bidirectional --skip-set-training` | 0.130 | 0.763 | 0.686 | 0.688 | 0.793 | 0.716 | 0.777 |

Leitura: sem supervisão no agregador, D6 cai para o nível de mean pooling
(`0.130`). Portanto, a preservação forte de bimodalidade observada em D6
vem do treino supervisionado sintético do agregador, não apenas da
arquitetura aleatória do Set Transformer.

Ao mesmo tempo, D2 continua razoavelmente alto mesmo sem treino
supervisionado do Set. Isso sugere que masked trajectory distillation
está capturando sinal temporal, mas a tese específica de bimodalidade
precisa de uma estratégia auto-supervisionada do agregador ou deve ficar
como sanidade/upper bound no sintético.

## D5a justo, 10 seeds

Configuração:

- Mesmo teacher bidirecional congelado.
- Mesmo agregador Set.
- Students comparados contra o mesmo alvo.
- Avaliação mascarando todas as posições válidas.

| Student | Overall | Stable | Drift | Bifurcating | Abrupt |
|---|---:|---:|---:|---:|---:|
| `bidirectional` | **1.996** | **2.150** | **1.963** | 2.105 | **1.766** |
| `causal` | 2.197 | 2.427 | 2.213 | **2.059** | 2.090 |
| `linear` | 4.659 | 4.768 | 4.645 | 4.628 | 4.598 |

Leitura: a hipótese central de D5a passa. Em `Abrupt`, a ordem esperada
aparece claramente:

```text
bidirectional < causal << linear
```

O causal empata ou vence levemente em `Bifurcating`, o que é aceitável:
a predição forte de bidirecionalidade era para rupturas abruptas mascaradas.

## Decisão atual

Para próximos experimentos sintéticos, manter duas trilhas separadas:

- **Trilha principal self-supervised:** reportar `mean:bidirectional` e
  `set:bidirectional --skip-set-training` como variantes sem rótulo no
  agregador.
- **Trilha de sanidade supervisionada:** reportar Set supervisionado com
  `true_context` apenas para demonstrar que o gargalo de bimodalidade pode
  ser removido quando o agregador recebe sinal apropriado.

Antes de COHA, a prioridade científica é substituir a supervisão sintética
do Set por um objetivo auto-supervisionado de agregador, ou então reduzir a
reivindicação sobre Set Transformer a uma sanidade controlada.
