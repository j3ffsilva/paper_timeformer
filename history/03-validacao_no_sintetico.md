# Capítulo 3 — Antes de acreditar: validando a ideia relacional

> Fontes originais: `tmp/07-timeformer_relational_change_second_opinion.md`,
> `tmp/08-timeformer_relational_code_validation_review.md`,
> `tmp/10-timeformer_structural_relational_change_review.md`.

O capítulo 02 terminou com um resultado que parecia bom: o experimento
estrutural (`docs/06`) detectou as quatro formas temporais
(gradual/abrupta/transitória/oscilatória) com controles. Mas esse
resultado só existe na forma "limpa" que vimos porque, **antes** de
chegar lá, várias rodadas de revisão picaram a ideia em pedaços e exigiram
correções. Este capítulo conta essa parte — o trabalho de desconfiar da
própria ideia antes de comprometer-se com ela.

## Problema 1: "perfis relacionais são invariantes" — invariantes a quê, exatamente?

A virada relacional (capítulo 02) se apoia numa intuição forte: se o
espaço de representações girar ou for reescalado uniformemente entre
`theta_0` e `theta_1`, as relações internas (cossenos) não mudam, então
`delta_rel ≈ 0` — sem falso positivo.

Uma segunda opinião (`tmp/07`) testou essa intuição matematicamente e
encontrou um limite importante: a similaridade por cosseno é invariante a
**rotação**, **reflexão** e **escalamento uniforme** — mas **não** a:

- **escalamento anisotrópico** (cada dimensão crescendo numa taxa
  diferente) — exatamente o tipo de deformação que o
  [Adam/AdamW produz, porque cada dimensão tem seu próprio segundo
  momento](conceitos/03-deriva_e_esquecimento.md#segundo-momento);
- **translação** — se o embedding de `[CLS]` ou `[PAD]` deriva
  sistematicamente;
- **mudanças nos parâmetros de LayerNorm** (`gamma`, `beta`) ao longo do
  treino, que equivalem a uma transformação afim por dimensão.

Ou seja: **a premissa central só é válida na prática se as deformações
produzidas pelo treino contínuo forem pequenas**, e isso "precisa ser
medido, não assumido". A revisão sugeriu uma verificação concreta: medir
**[CKA](conceitos/01-correlacao_e_similaridade.md#cka)** entre as
representações de `theta_t` e `theta_(t+1)` sobre o mesmo conjunto de
frases. Se `CKA ≈ 1.0`, a geometria foi preservada e os perfis são
comparáveis; se cair muito, a comparação direta perde sentido. Essa
checagem de CKA entre checkpoints consecutivos passa a ser, a partir
daqui, um item de rotina sempre que se compara `theta_a` com `theta_b`
(reaparece nos capítulos 06 e 09).

A revisão também trouxe um critério de validação simples e direto, que
se tornaria o "teste de fumaça" padrão do projeto:

> Para palavras do grupo `Stable` do corpus sintético, `delta_rel` deve
> ser ≈ 0. Se palavras sabidamente estáveis mostram `delta_rel` alto, o
> método está medindo principalmente instabilidade, não mudança
> semântica.

No vocabulário dos quatro personagens (capítulo 00): isso é exatamente
testar se `tree_nn` (controle de estabilidade) fica perto de zero antes
de confiar em qualquer número para `plane_nn` ou `graft_nn`.

## Problema 2: o esquecimento catastrófico confunde sinal com ruído

A mesma revisão (`tmp/07`) apontou que o `MLMTrainer` da época não tinha
**nenhuma** proteção contra esquecimento catastrófico — sem replay, sem
EWC, sem learning rate diferenciado por camada
([conceitos/03](conceitos/03-deriva_e_esquecimento.md)). Isso significa
que qualquer `delta_rel` medido entre `theta_(t-1)` e `theta_t` é uma
mistura inseparável de:

- mudança semântica genuína no corpus;
- degradação de representações de períodos anteriores (esquecimento).

Esse ponto não foi "resolvido" neste capítulo — ele é adiado, mas
**registrado como uma fonte de confusão que precisa de controles**. É
exatamente para isso que servem os regimes `resampled_null` e
`continual_placebo` do pré-registro (`docs/06`, capítulo 02): eles
isolam quanto do `delta_rel` observado aconteceria *mesmo sem* mudança
semântica real, só pelo fato de o treino continuar.

## Problema 3: o placebo "sabe" demais

A auditoria de código (`tmp/08`, sobre a primeira rodada com `seed=1000`)
encontrou um sinal concreto desse problema. Para o regime placebo
(`continual_placebo`, treino repetido sempre em `D_0`), o cosseno entre a
direção observada e a direção-oráculo (de `t0` a `t9`) deveria ser
próximo de zero — afinal, nada deveria "saber" para onde os outros
sujeitos vão se mover. Os valores observados foram:

| Classe sintética | Cosseno direcional do placebo (`t0 -> t9`) |
|---|---:|
| `abrupt` | +0.559 |
| `bifurcating` | +0.677 |
| `drift` | +0.549 |
| `stable` | +0.663 |

Para comparação, o valor esperado entre dois vetores aleatórios num
espaço de 39 dimensões (39 outros sujeitos) é `1/sqrt(39) ≈ 0.16`. Os
valores observados são **3 a 4 vezes maiores** que o acaso.

A explicação mais plausível encontrada: o modelo, mesmo só vendo `D_0`
repetidamente, melhora sua estimativa da distribuição de contextos de
cada sujeito em `t0` — e essa melhora, por acaso, aponta parcialmente na
mesma direção que o oráculo de `t9` (porque sujeitos `stable` não mudam,
e `bifurcating` converge para um estado intermediário que ainda preserva
parte da estrutura de `t0`). Consequência prática:

> a subtração `delta_real - delta_placebo` pode remover parte do sinal
> real (se a melhora do placebo aponta na mesma direção da mudança real),
> ou inflar o sinal (se apontarem em direções diferentes).

A auditoria recomendou dois controles concretos antes de escalar para
múltiplas seeds: (1) **permutação de período** — embaralhar a ordem
cronológica e verificar se o excesso de cosseno desaparece (se não
desaparecer, o sinal não é específico da ordem cronológica); (2)
**múltiplas seeds do placebo**, para construir uma distribuição nula do
cosseno direcional e comparar o regime real contra o percentil 95 dessa
distribuição — em vez de assumir que zero é o nulo correto.

Esses dois controles são, na prática, os antepassados diretos dos regimes
`resampled_null` (controle principal contra falsos positivos) e
`continual_placebo` (diagnóstico de deriva de otimização) que aparecem no
pré-registro `docs/06` do capítulo 02.

A mesma auditoria também encontrou um bug concreto e de baixo nível:
**o checkpoint salvo não incluía o estado do otimizador** (`opt.state_dict()`),
o que tornava `--reuse-checkpoints` não-reprodutível — ao retomar de um
checkpoint, o Adam reiniciava seus momentos do zero, produzindo uma
trajetória diferente da que teria ocorrido numa execução contínua. Esse é
o tipo de bug "invisível" que só aparece quando se tenta reproduzir um
resultado depois — e por isso vale registrar como lição: **sempre que o
pipeline permitir retomar de um checkpoint, o estado do otimizador faz
parte do checkpoint.**

## Problema 4: "mudança pequena" é ruído ou é o próprio fenômeno?

A revisão mais filosófica desta fase (`tmp/10`) atacou uma tentação que
apareceu nos dados: o método detectava mudanças com confiança quando o
parâmetro de mudança do corpus sintético (`alpha`) era alto (≥0.75), mas
não quando era baixo (`alpha=0.25`). A proposta em avaliação era
reinterpretar `alpha=0.25` como "microvariação estruturalmente
irrelevante" — e não como "mudança real que o método não detecta".

A revisão chamou isso pelo nome: **HARKing aplicado à definição do
objeto** (Hypothesizing After Results are Known, mas em vez de mudar a
hipótese, muda-se a definição do que conta como "mudança relevante" para
que o método pareça correto por definição). O argumento histórico contra
essa tentação é direto: mudanças semânticas pequenas, graduais e
persistentes são exatamente como campos semânticos inteiros mudam na
prática — por exemplo, a expansão do campo de "computador" entre 1950 e
2000 aconteceu por acúmulo lento de pequenas associações novas, não por
um salto único. Declarar `alpha=0.25` "irrelevante por definição"
arriscava apagar justamente esse tipo de fenômeno — o tipo de mudança
gradual que o personagem hipotético "mudança lenta de uso" (linha
`Drift`, capítulo 01) representa.

A saída proposta, e que se tornou o padrão do projeto a partir daqui, foi
definir "reorganização estrutural" por **três critérios independentes**,
calibrados a partir do nulo — não por um limiar único ajustado aos dados:

1. **Magnitude**: supera a variação nula esperada *para aquela palavra
   específica* (o nulo não é global — sujeitos com `p_0` diferente têm
   nulos até 5x diferentes entre si);
2. **Persistência**: a mudança não reverte nos períodos seguintes, mas se
   mantém ou se intensifica;
3. **Coerência**: as dimensões/vizinhos que mudam formam um conjunto
   semanticamente interpretável, não um padrão espalhado aleatoriamente.

```text
estrutural(w, a, b) = magnitude(w) > nulo_especifico(w)
                  AND persistente(w)
                  AND coerente(w)
```

A revisão também propôs um teste que distingue "descoberta genuína" de
"racionalização": se a framing estrutural for real, ela deve **prever**
que mudanças pequenas mas *persistentes e coordenadas ao longo dos 10
períodos* sejam detectáveis como reorganização estrutural — mesmo que
cada passo individual fique abaixo do limiar de magnitude. Se o método
não detectar isso, o limiar é do método, não do fenômeno. Esse
experimento específico não foi rodado nesta fase, mas a exigência de
"magnitude E persistência E coerência" (em vez de um score único) é a
resposta estrutural a essa preocupação, e moldou os critérios
confirmatórios do pré-registro `docs/06`.

## Veredito desta fase

Nenhuma das três revisões deu um "sim" simples. Os veredictos foram:

- `tmp/07`: "promissor, mas não pronto para implementação — validar
  invariantes explicitamente e rodar o experimento mínimo de
  falsificação antes."
- `tmp/08`: "o código implementa corretamente o que afirma, mas corrigir
  o bug do otimizador e investigar o viés do placebo antes de múltiplas
  seeds."
- `tmp/10`: "não adotar a nova framing como posição principal ainda — fazer
  o experimento que distingue descoberta de racionalização."

O experimento estrutural pré-registrado do capítulo 02 (`docs/06`) é a
resposta a essas três exigências ao mesmo tempo: ele inclui controles
`frozen`, `resampled_null` e `continual_placebo` (resposta aos problemas 2
e 3), define magnitude por palavra contra o nulo específico em vez de um
limiar global (resposta ao problema 4), e — embora a verificação de CKA
entre checkpoints não apareça destacada nos resultados do capítulo 02 —
estabelece o hábito de tratar "os perfis são comparáveis?" como uma
pergunta a ser respondida, não assumida (resposta ao problema 1).

## O que vale levar deste capítulo

- A lição central é processual, não um número: **uma ideia que "faz
  sentido" matematicamente (cossenos são invariantes a rotação/escala)
  pode falhar silenciosamente na prática** se as condições reais do
  experimento (treino com Adam, LayerNorm, esquecimento) violarem as
  premissas que garantem essa invariância.
- O "teste de fumaça com `Stable`" (`delta_rel(tree_nn) ≈ 0`?) e a
  checagem de CKA entre checkpoints consecutivos tornam-se verificações
  de rotina — qualquer capítulo posterior que reporte um `delta_rel` alto
  para uma palavra deveria, implicitamente, já ter passado por essas
  checagens.
- A distinção entre os três critérios — magnitude, persistência,
  coerência — calibrados contra um nulo *específico da palavra* (não
  global) é uma ideia que reaparece, com nomes diferentes, em quase todo
  método estatístico introduzido depois (capítulos 05, 06 e 08).
- O bug do checkpoint sem estado do otimizador é um lembrete concreto:
  problemas de reprodutibilidade podem invalidar silenciosamente a
  comparação entre execuções, mesmo quando a lógica científica está
  correta.

## Conceitos novos usados neste capítulo

- [CKA entre checkpoints](conceitos/01-correlacao_e_similaridade.md#cka)
- [Catastrophic forgetting / esquecimento catastrófico](conceitos/03-deriva_e_esquecimento.md)
- [Nulo específico por palavra vs. limiar global](conceitos/05-estatistica_experimental.md#nulo-por-palavra)
- [Segundo momento do Adam/AdamW e escalamento anisotrópico](conceitos/03-deriva_e_esquecimento.md#segundo-momento)
- [HARKing](conceitos/05-estatistica_experimental.md#harking)
