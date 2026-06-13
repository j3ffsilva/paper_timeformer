# Conceitos 6 — WSD e sentido lexical

Este arquivo cobre a linha de investigação mais recente do projeto
(capítulo 10): em vez de inferir "mudança de sentido" a partir da
geometria de representações contextuais (capítulos 06-09), usar um modelo
treinado **especificamente** para identificar sentidos.

<a id="wsd"></a>
## WSD (Word Sense Disambiguation)

WSD é a tarefa de, dada uma palavra em contexto, decidir **qual dos
sentidos possíveis dessa palavra** está sendo usado. Por exemplo, para a
frase:

```text
"plate figure represent an inclined plane"
```

a palavra `plane` poderia significar, entre outros: uma superfície
geométrica (`plane%1:25:00::`), uma aeronave (`plane%1:06:01::`), ou uma
ferramenta de marcenaria (`plane%1:06:00::`/`plane%1:06:02::`). Um sistema
de WSD recebe a frase e devolve qual desses sentidos é o mais provável
naquele contexto — neste caso, "geometria".

Os "sentidos possíveis" vêm de um **inventário de sentidos**, tipicamente
o **WordNet** — um dicionário estruturado onde cada palavra tem uma lista
de "sensekeys" (identificadores únicos de sentido) com definições
("glosses") associadas.

<a id="inventario"></a>
## Inventário, granularidade e cobertura

Um sistema WordNet só pode prever sentidos que existem em seu inventário.
Isso cria três problemas:

1. **cobertura**: um sentido histórico pode não estar registrado;
2. **granularidade**: o WordNet pode separar distinções finas demais para a
   pergunta histórica;
3. **anacronismo**: um sentido moderno pode ser oferecido como candidato para
   uma frase do século XIX, mesmo que ainda não existisse.

Agrupar sensekeys em classes mais amplas (`geometry`, `aircraft`, `tool`) ajuda
na interpretação, mas também é uma decisão semântica. Se o agrupamento for
feito depois de ver os resultados, pode introduzir viés.

<a id="open-set"></a>
## Closed set, open set e abstention

No **closed set**, o modelo é obrigado a escolher um dos sentidos conhecidos:

```text
prediction = argmax entre os sensekeys disponíveis
```

Mesmo quando nenhum candidato é adequado, algum deles vence. Isso pode
transformar "sentido ausente/desconhecido" em um erro confiante.

No **open set**, existe uma saída adicional:

```text
UNK / sentido não coberto / abstenção
```

Ela pode depender de margem, distância absoluta, entropia ou calibração. O
desafio é definir o limiar sem usar o gold temporal e sem confundir linguagem
histórica difícil com sentido realmente novo.

O Gate 1 usou closed set para testar compatibilidade básica. Uma arquitetura
completa para mudança histórica provavelmente precisará de open set.

<a id="cauda-longa"></a>
## Cauda longa e sentidos raros

Distribuições lexicais têm poucos sentidos frequentes e muitos sentidos raros.
Modelos tendem a favorecer o sentido mais frequente aprendido no treino.

No projeto:

```text
geometry e aircraft -> muitos exemplos e pistas modernas claras
tool                -> 19 exemplos históricos, termos como jack/mould plane
```

O fracasso em `tool` é mais importante do que a acurácia macro sugere:
um atlas histórico precisa funcionar justamente na cauda, onde mudanças e
desaparecimentos de sentido podem ocorrer. Por isso a Porta 1 foi conjuntiva.

### Por que WSD é diferente do que os capítulos 04-09 faziam

Os capítulos 04-09 perguntavam, essencialmente, "os *contextos* de `w` em
D0 são parecidos com os contextos de `w` em D1?" — uma pergunta sobre
**distribuições de uso** (`P_t(contexto | w)`). WSD pergunta diretamente
"qual *sentido* está sendo usado aqui?" — uma pergunta sobre
**identidade de sentido** (`P_t(sentido | w)`). O capítulo 06 mostrou que
essas duas perguntas não são a mesma coisa: clusters de contexto podem
separar D0 de D1 por motivos que não têm nada a ver com sentido (registro,
tópico, entidades). Uma régua de WSD, ao prever sentidos diretamente,
contorna essa ambiguidade — **se** ela for confiável.

Comparar:

```text
abordagem dos capítulos 06-09:
  contexto -> representação geométrica -> "as nuvens de D0 e D1 se separaram?"
  (resposta indireta, depende de identificabilidade)

abordagem do capítulo 10 (WSD):
  contexto -> sentido (de um inventário fixo, ex: WordNet)
  "P(geometria | plane, D0) vs P(aviação | plane, D1)"
  (resposta direta, se o WSD for confiável)
```

<a id="lmms"></a>
## LMMS-SP

LMMS (Language Modelling Makes Sense) é uma família de métodos de WSD que
funciona assim: para cada sensekey do WordNet, calcula-se um **vetor de
sentido** (a partir de exemplos anotados e/ou da definição/gloss do
sentido, processados por um encoder como `bert-large-cased`). Para
desambiguar uma ocorrência nova, calcula-se o vetor da ocorrência (com o
mesmo encoder) e escolhe-se o sentido cujo vetor é **mais próximo por
cosseno**.

```text
para cada sensekey s de "plane":
   v_s = vetor de sentido pré-computado (LMMS)

para a ocorrência:
   v_ocorrencia = encoder(frase, posição de "plane")

sentido_previsto = argmax_s cos(v_ocorrencia, v_s)
```

A variante "SP" usa uma combinação ponderada de camadas do encoder
(`bert-large-cased`), com pesos publicados pelos autores do método.

No capítulo 10, LMMS-SP foi usado **integralmente congelado** — nenhum
ajuste no SemEval, nenhuma calibração com os dados do projeto. Isso o
torna um "oráculo" no mesmo espírito do capítulo 08 (ver
[oráculo](05-estatistica_experimental.md#oraculo)): se ele funcionar bem,
é evidência de que a tarefa "ler sentidos no corpus lematizado" é viável
com um modelo pronto; se falhar, é informação sobre os limites desse
modelo específico (cobertura de sentidos raros, domínio histórico, etc.).

O resultado do capítulo 10 foi misto e informativo: os dois sentidos
dominantes de `plane_nn` (geometria em D0, aviação em D1) foram
identificados quase perfeitamente (98,4% e 100%), mas o sentido raro
"ferramenta" (19 ocorrências históricas de marcenaria) teve acurácia de
21,1% — **abaixo do acaso de 1/3**. A causa provável combina dois fatores:
ruído na heurística de rotulagem usada para avaliação, **e** dificuldade
real do LMMS com vocabulário histórico de cauda longa (`mould plane`,
`jack plane` — termos que um corpus moderno de pré-treino dificilmente
contém em quantidade).

<a id="adjudicacao"></a>
## Adjudicação cega

"Adjudicação cega" é o processo de ter um ou mais humanos rotularem dados
**sem ver as predições do modelo** que está sendo avaliado — para que o
julgamento humano não seja influenciado (consciente ou
inconscientemente) pela resposta do modelo.

No capítulo 10, isso foi proposto especificamente para as 19 ocorrências
do subconjunto "ferramenta" de `plane_nn`, que falhou no Gate 1: dois
anotadores devem rotular cada ocorrência como `tool`, `geometry`,
`aircraft`, `botanical` ou `unclear`, **sem ver** se o LMMS previu
`tool`, `geometry` ou `aircraft` para aquela ocorrência.

Por que isso é necessário aqui especificamente: a auditoria do capítulo 10
já notou que a **heurística** original de rotulagem (usada para decidir
quais ocorrências contam como "ferramenta") tinha pelo menos um falso
positivo claro — uma ocorrência de `plane tree` (uma espécie de árvore,
sentido botânico) foi rotulada "ferramenta" só por conter a palavra
`timber` em algum lugar da frase. Sem saber quanto desse tipo de erro
existe no subconjunto, é impossível separar:

```text
"o LMMS errou porque o rótulo de referência estava errado" (problema da heurística)
   vs.
"o LMMS errou porque não sabe identificar 'ferramenta' em contexto histórico" (limite do modelo)
```

A adjudicação cega responde à primeira pergunta sem contaminar a segunda:
o rótulo humano (cego às predições) se torna a referência corrigida, e
**só então** se compara com as predições do(s) modelo(s) de WSD — incluindo
um segundo modelo (ConSeC), para testar se uma eventual falha persistente
é específica do LMMS ou generaliza.

## Visão geral: a "Porta 1" como template

O desenho do capítulo 10 — uma palavra-alvo (`plane_nn`), subconjuntos
heurísticos com cortes pré-registrados, uma regra de decisão conjuntiva
([pré-registro](05-estatistica_experimental.md#pre-registro)), e uma
"Porta 2" condicional ao resultado da "Porta 1" — é pensado como um
**template reutilizável**: se a Porta 1 (com ajustes pós-adjudicação)
eventualmente passar, a Porta 2 estenderia o mesmo protocolo, sem
engenharia por palavra, para os outros três personagens (`graft_nn`,
`chairman_nn`, `tree_nn`). Se não passar mesmo com um segundo modelo, a
conclusão documentada (capítulo 10) é que **sentidos históricos raros
exigem adaptação de domínio, um inventário de sentidos histórico, ou
supervisão adicional** — uma direção de pesquisa diferente, mas ainda
dentro do espírito de usar uma régua externa de sentido em vez de inferir
sentido a partir de geometria contextual.
