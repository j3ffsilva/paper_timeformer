# Plano atual: mudança semântica relacional entre checkpoints

## Status

Este documento descreve a direção experimental principal atual. O pipeline de
teacher/student e trajetória aprendida descrito em `novo_planejamento.md` deixa
de ser a configuração principal e permanece somente como baseline histórico.

## Hipótese

Um Transformer padrão é treinado cronologicamente:

```text
theta_0 = treino(D_0)
theta_1 = continua_treino(theta_0, D_1)
...
theta_t = continua_treino(theta_{t-1}, D_t)
```

Nenhum identificador de período é fornecido ao modelo. A mudança semântica não
é a diferença entre coordenadas absolutas de embeddings. Ela é a mudança das
relações internas de uma palavra com as demais palavras no mesmo checkpoint.

```text
r_t(w)[v] = similaridade_t(w, v)
delta_rel(w, a, b) = r_b(w) - r_a(w)
```

Uma transformação global que preserve todas as relações deve produzir mudança
relacional zero. A trajetória é derivada posteriormente da sequência de perfis
e deltas relacionais; ela não é aprendida por teacher/student.

## Representação principal em avaliação

O experimento sintético mostrou que consultar diretamente `h_subj` não recupera
de forma confiável a direção semântica conhecida. A representação principal
passa a ser a distribuição de contextos produzida por um **probe preditivo
pós-Transformer**:

```text
[CLS] palavra [MASK] [MASK] [SEP]
```

Extraímos as distribuições previstas nas posições de verbo e objeto, restritas
aos 16 contextos sintéticos válidos. A relação entre duas palavras é calculada
pela similaridade de Jensen-Shannon entre suas distribuições previstas.

```text
q_t(w) = P_t(contextos | [CLS] w [MASK] [MASK] [SEP])
r_t(w)[v] = 1 - JS(q_t(w), q_t(v)) / log(2)
```

Essa formulação não depende das coordenadas ocultas do Transformer e possui
interpretação direta: duas palavras são semelhantes quando o checkpoint prevê
distribuições de contexto semelhantes para elas.

Isto não é um sinal temporal anterior ao Transformer. É uma consulta posterior
ao treinamento de cada checkpoint.

As geometrias ocultas permanecem como ablações:

- cosseno dos estados mascarados;
- cosseno centralizado, invariante a translação, rotação/reflexão e escala
  positiva global;
- distância euclidiana normalizada, com as mesmas invariâncias globais;
- estado direto do sujeito e centroides contextuais.

## Controles obrigatórios

- `continual_real`: `D_0 -> D_1 -> ... -> D_t`.
- `continual_placebo`: repete `D_0` para estimar deriva causada apenas por
  continuar a otimização.
- `frozen`: aplica `theta_0` aos diferentes corpora sem atualizar pesos.
- validação e parada antecipada por período, restaurando o melhor checkpoint;
- probes fixos e probes preditivos, separados das ocorrências do próprio corpus.
- registro de passos computados e do passo selecionado para cada checkpoint;
- estado retomável contendo modelo e otimizador.

Resultados brutos e comparações com placebo devem ser mantidos. Calculamos:

```text
vantagem_direcional = direção_real_vs_oráculo - direção_placebo_vs_oráculo
```

A antiga subtração vetorial `delta_real - delta_placebo` permanece somente como
diagnóstico exploratório. Não deve ser interpretada como efeito causal, pois
real e placebo podem percorrer quantidades diferentes de passos de otimização.

## Relações e métricas

- mudança de vizinhos por Jaccard;
- mudança de ranking relacional por Spearman normalizada em `[0,1]`;
- média da mudança absoluta de similaridade;
- CKA como diagnóstico global;
- cosseno entre a direção relacional aprendida e o oráculo sintético;
- direção observada, direção placebo e vantagem direcional pareada;
- Jensen-Shannon como relação principal;
- cosseno oculto, cosseno centralizado e euclidiana normalizada como ablações.

## Resultado atual em três seeds

Configuração: seeds `1000`, `1001` e `1002`; 100 ocorrências por
palavra/período; `t0` com até 60 épocas; períodos posteriores com até 30 épocas
e parada antecipada.

Para Jensen-Shannon sobre distribuições previstas, na mudança acumulada
`t0 -> t9`:

| Classe | Direção observada | DP | Direção placebo | Vantagem pareada | DP |
|---|---:|---:|---:|---:|---:|
| abrupt | +0.968 | 0.012 | +0.554 | +0.414 | 0.030 |
| bifurcating | +0.881 | 0.047 | +0.540 | +0.341 | 0.080 |
| drift | +0.955 | 0.014 | +0.546 | +0.409 | 0.035 |
| stable | +0.951 | 0.012 | +0.617 | +0.334 | 0.043 |

A vantagem pareada foi positiva nas três seeds para todas as classes. O placebo
também apresenta direção positiva substancial, confirmando que deriva de
otimização é uma ameaça real à validade.

Comparação média entre relações em `t0 -> t9`:

| Relação | Direção observada | Placebo | Vantagem |
|---|---:|---:|---:|
| Distribuição prevista + Jensen-Shannon | +0.939 | +0.564 | +0.375 |
| Estado oculto + cosseno | +0.933 | +0.632 | +0.300 |
| Estado oculto + cosseno centralizado | +0.537 | -0.024 | +0.561 |
| Estado oculto + euclidiana normalizada | +0.463 | +0.181 | +0.282 |

Jensen-Shannon é a configuração principal por combinar direção alta,
interpretação semântica direta e menor dependência da geometria interna. O
cosseno centralizado é a ablação geométrica mais forte em vantagem sobre
placebo.

O sinal é muito mais forte para mudança acumulada. Nos passos consecutivos
iniciais (`t1`, `t2`) e no passo final `t8 -> t9`, a direção é fraca ou
instável. Não devemos afirmar ainda que o método detecta mudanças pequenas.

### Controle com orçamento fixo

Uma ablação adicional executou a seed `1000` sem parada antecipada e selecionou
o checkpoint final de cada período. Real e placebo receberam exatamente `8250`
atualizações cada.

| Relação | Direção observada | Placebo | Vantagem |
|---|---:|---:|---:|
| Distribuição prevista + Jensen-Shannon | +0.893 | +0.601 | +0.292 |
| Estado oculto + cosseno | +0.931 | +0.646 | +0.284 |
| Estado oculto + cosseno centralizado | +0.446 | -0.008 | +0.455 |
| Estado oculto + euclidiana normalizada | +0.287 | +0.196 | +0.091 |

A vantagem Jensen-Shannon diminui em relação ao regime escolhido por validação
(`+0.375`), mas permanece positiva. Portanto, seleção desigual de checkpoints
explica parte, mas não todo, o sinal observado.

Uma palavra classificada como estável pelo gerador pode apresentar mudança
relacional: mesmo que sua propriedade própria permaneça constante, suas
relações mudam quando outras palavras se movem. Essa distinção deve ser
explicitada no paper.

## Riscos ainda abertos

- Na configuração principal, real e placebo executam números diferentes de
  passos devido à parada antecipada. Nas seeds `1001` e `1002`, o regime real
  executou respectivamente `6175` e `5825` passos, enquanto o placebo executou
  `4050` e `4525`. O controle de orçamento fixo reduz, mas não elimina, a
  vantagem observada.
- O placebo positivo exige distribuição nula com mais seeds e, possivelmente,
  um controle pareado por orçamento fixo de atualizações.
- O benchmark sintético alinha estruturalmente tarefa, probe e oráculo. Isso é
  apropriado para validação controlada, mas não demonstra ainda validade em
  corpus real.
- Uma palavra `stable` pode apresentar mudança relacional porque outras palavras
  se movem.

## Próximo critério para prosseguir

Antes de corpus real, executar:

1. distribuição nula mais ampla do placebo;
2. replicar a ablação com orçamento de passos pareado em múltiplas seeds;
3. teste explícito de sensibilidade a mudanças pequenas/consecutivas;
4. benchmark sintético com estrutura semântica multidimensional, além de
   `p_n1`.
