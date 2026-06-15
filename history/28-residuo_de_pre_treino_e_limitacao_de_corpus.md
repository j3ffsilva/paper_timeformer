# 28. O resíduo do pré-treino é uma limitação de corpus, não do método

## A pergunta

Em d0 (1810-1860), `plane` é quase exclusivamente o substantivo geométrico
(~94% de ocorrências standalone, contextos como "projection of any plane",
"horizontal plane"). Mesmo assim, seu vizinho relacional mais próximo em d0
inclui `flight` (cos = +0,4844), associação típica do sentido moderno
"avião". Isso é um artefato do método ou um limite do corpus?

## O diagnóstico

Comparando os embeddings estáticos de entrada (`model.bert.embeddings.word_embeddings.weight`)
em dois checkpoints do seed1000:

| | `init` (bert-tiny original) | `period1_epoch2` (após treino contínuo) |
|---|---|---|
| cos(plane, flight) | +0,5873 | +0,5725 |
| cos(plane, angle)  | +0,4184 | +0,4573 |
| cos(plane, gas)    | +0,2469 | +0,2506 |

A proximidade `plane`-`flight` já existe **antes** de qualquer treino
contínuo nosso (0,5873 no `init`) e quase não muda depois (0,5725). O
pré-treino original do bert-tiny (inglês moderno, onde `plane` ≈ "avião")
grava esse prior no embedding estático de `plane`, associando-o fortemente a
`flight`.

## A interpretação

Não é um defeito do `token@time`: o método mede corretamente o deslocamento
relacional `Delta(plane, d0, d1)`, que é de fato o maior entre as 37 palavras
(maior `Z_robusto`), exatamente porque a mudança geometria -> aviação é real e
grande. O que o diagnóstico expõe é outra coisa: a posição *absoluta* de
`plane` em d0 já carrega um resíduo do sentido moderno, antes mesmo de medir
qualquer mudança.

**Sim -- com dados suficientes de treino contínuo no sentido geométrico, essa
separação deveria ocorrer.** A representação contextual (`layer_2`) tem
capacidade para se afastar do prior do embedding estático quando o contexto
real e repetido empurra nessa direção; é isso que o treino contínuo faz a
cada época. O que falta aqui é volume: `plane` tem apenas 301 ocorrências em
d0, contra o volume (bilhões de tokens) que gravou o prior "avião" no
pré-treino do bert-tiny. O treino contínuo move o embedding estático na
direção certa (cos(plane, angle) sobe de 0,4184 para 0,4573), mas não o
suficiente para superar um prior tão mais "pesado".

## Conclusão

É uma limitação do **corpus de treino contínuo** (tamanho/cobertura),
combinada com a **capacidade do encoder** (bert-tiny, 2 camadas, hidden=128) -
não do framework `token@time`. Com um corpus maior de 1810-1860 ou um encoder
maior/mais épocas de treino contínuo, o resíduo "avião" em `plane`@d0 deveria
diminuir, e os vizinhos relacionais de `plane` em d0 deveriam refletir mais
puramente o sentido geométrico dominante naquele período.

Isso é consistente com a leitura do `token@time` como instrumento sensível à
*mudança* relativa (`Delta`), mesmo quando a posição absoluta em um período
carrega resíduo de outras fontes (pré-treino). Para leitura de posição
absoluta isolada (sem comparação entre períodos), esse resíduo é uma fonte de
ruído a se ter em mente ao reportar vizinhos de um único período.
