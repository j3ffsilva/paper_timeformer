# Resposta à segunda opinião: bootstrap e trajetória com orçamento alinhado

## Decisão

Aceitamos os dois pontos centrais da segunda opinião:

1. `n=37` torna as diferenças de Spearman muito incertas;
2. comparar o checkpoint pseudo selecionado em `0,5` época com o
   cronológico selecionado em `2` épocas confundia ordem e orçamento.

Antes de qualquer distillation, executamos os dois controles baratos
propostos. Os resultados não sustentam continuar tentando resgatar a
`layer 2` neste momento.

## Bootstrap pareado por palavra

Implementamos:

```text
scripts/bootstrap_bert_apd_spearman.py
```

O procedimento reamostra com reposição os mesmos 37 alvos, calcula o
Spearman em cada réplica e, para comparações, subtrai as correlações
dentro da mesma réplica. Foram usadas 20.000 réplicas e seed fixa.

### Layer 1

| Condição | Spearman | IC bootstrap 95% |
|---|---:|---:|
| init | 0,298 | [-0,018; 0,561] |
| full seed 1000 | 0,325 | [-0,011; 0,607] |
| full seed 1001 | 0,322 | [-0,016; 0,603] |
| pseudo seed 1000 | 0,332 | [-0,007; 0,616] |
| LR discriminativa | 0,340 | [0,014; 0,611] |
| LR + L2-SP cronológico | 0,338 | [0,011; 0,609] |
| LR + L2-SP pseudo | 0,341 | [0,014; 0,608] |

As diferenças pareadas entre `init` e os modelos treinados incluem zero.
Exemplos:

```text
full_s1000 - init:      [-0,081; 0,125]
lower_lr - init:        [-0,020; 0,109]
l2sp_chrono - init:     [-0,024; 0,106]
```

Assim, não podemos afirmar que o treino melhorou a layer 1. O padrão é
estável nas estimativas pontuais, mas o ganho de poucos centésimos não é
distinguível de zero com estes 37 alvos.

### Layer 2

| Condição | Spearman | IC bootstrap 95% |
|---|---:|---:|
| init | 0,136 | [-0,197; 0,453] |
| full seed 1000 | 0,030 | [-0,285; 0,334] |
| full seed 1001 | 0,038 | [-0,274; 0,343] |
| pseudo seed 1000 selecionado | 0,153 | [-0,171; 0,444] |
| LR discriminativa | 0,012 | [-0,300; 0,316] |
| LR + L2-SP cronológico | 0,014 | [-0,298; 0,318] |
| LR + L2-SP pseudo | 0,116 | [-0,203; 0,413] |

Todas as condições incluem zero. A diferença `full seed 1000 - init`
também inclui zero:

```text
[-0,302; 0,084]
```

Logo, “a layer 2 perdeu sinal” deve ser tratada como padrão descritivo,
não como efeito estatisticamente estabelecido.

## Trajetória com orçamento de D1 alinhado

Reavaliamos os checkpoints de `0,25`, `0,5`, `1` e `2` épocas de D1 nas
condições cronológica e pseudo, sempre com a mesma amostra de ocorrências.

### Spearman da layer 2

| Épocas em D1 | Cronológico | Pseudo | Pseudo - cronológico |
|---:|---:|---:|---:|
| 0,25 | 0,012 | 0,046 | 0,033 |
| 0,5 | 0,062 | 0,153 | 0,091 |
| 1 | 0,059 | 0,176 | 0,117 |
| 2 | 0,030 | 0,088 | 0,058 |

O pseudo fica numericamente acima em todos os marcos, portanto a seleção
de `0,5` contra `2` épocas não explica sozinha todo o padrão. Entretanto,
nenhum IC pareado da diferença com orçamento igual exclui zero:

| Épocas | IC 95% de `cronológico - pseudo` |
|---:|---:|
| 0,25 | [-0,132; 0,062] |
| 0,5 | [-0,238; 0,048] |
| 1 | [-0,278; 0,031] |
| 2 | [-0,170; 0,044] |

A evidência correta é:

> Há um padrão consistente nas estimativas pontuais de layer 2, mas sua
> magnitude é instável e não é distinguível de zero com os 37 alvos e
> uma única seed pseudo.

## Revisão das conclusões anteriores

Retiramos como conclusão:

```text
a ordem cronológica causa perda adicional específica na layer 2
```

Mantemos apenas:

```text
layer 1 é o melhor readout entre os avaliados;
layer 2 é fraca e instável antes e depois do treino;
L2-SP reduz drift paramétrico, mas não produz melhora observável;
APD absoluta não separa alvos de controles pareados por frequência;
os dados atuais não identificam redistribuição causal entre camadas.
```

O cosseno de âncoras próximo de `0,95` continua válido somente como
checagem de ausência de colapso numérico grosseiro.

## Decisão sobre distillation

Não implementaremos distillation agora.

Ela tentaria preservar uma layer que:

1. já era inferior no baseline congelado;
2. não apresentou diferença estatisticamente estabelecida;
3. não é necessária como régua, porque layer 1 já é melhor;
4. não resolve a identificabilidade entre contexto e sentido lexical.

Caso essa linha seja retomada, o protocolo será:

```text
teacher = theta_init
loss ponto a ponto
beta relacional = 0 no piloto
âncoras sem os 37 alvos e sem os 37 controles
alpha calibrado por razão de gradientes, sem gold
```

Mas isso deixa de ser o próximo experimento prioritário.

## Próximo passo recomendado

Avançar para a Porta 1 da arquitetura externa de WSD:

1. escolher um modelo externo congelado treinado para compatibilidade
   contexto-gloss;
2. não ajustar nada no SemEval;
3. testar os subconjuntos heurísticos predefinidos de `plane`:

```text
D0 geometry: 182
D0 tool:       19
D1 aircraft:  208
```

4. reportar macro accuracy e intervalos;
5. exigir aproximadamente:

```text
D0 geometry >= 0,75
D1 aircraft >= 0,80
D0 tool substantivamente acima do acaso, com IC
```

6. verificar que uma ocorrência histórica inequívoca permanece geométrica
   sob a única régua externa.

Em paralelo, a linha MLM fica documentada como estudo de plasticidade e
seleção de readout, não como estimador direto de mudança de sentido.

## Artefatos

```text
scripts/bootstrap_bert_apd_spearman.py
outputs/bert_tiny_option_d_bootstrap/bootstrap_spearman.json
outputs/bert_tiny_option_d_bootstrap/period1_matched_trajectory.json
outputs/bert_tiny_option_d_full_seed1000/trajectory_apd/
outputs/bert_tiny_option_d_random_control_seed1000/trajectory_apd/
```
