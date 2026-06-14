# Como ler este histórico

## Para que serve esta pasta

`docs/` e `tmp/` contêm o registro de trabalho real do projeto: planos,
pré-registros, pedidos de segunda opinião, pareceres e diários técnicos,
todos numerados em ordem cronológica de criação. Esse material é preciso,
mas é difícil de ler de uma vez — está espalhado em ~50 arquivos, em
formatos diferentes (prompt, parecer, diário, plano), com muita
informação repetida ou superada por etapas posteriores.

`history/` é uma segunda camada, escrita por cima desse material: uma
narrativa única, em ordem cronológica, que conta **o que foi tentado, o
que funcionou, o que não funcionou e por quê**, com exemplos concretos.
O objetivo é que alguém que não acompanhou o dia a dia do projeto consiga
entender, lendo do capítulo 00 ao 11, como chegamos do ponto de partida
até o estado atual — e por que cada virada de rumo aconteceu.

Cada capítulo de `history/`:

- conta a decisão central daquela fase e o(s) experimento(s) que a
  motivaram;
- usa pelo menos um exemplo concreto (sempre que possível, um dos quatro
  "personagens" descritos abaixo);
- reproduz as tabelas de resultados mais importantes daquela fase;
- termina com uma referência aos arquivos originais de `docs/`/`tmp/`
  para quem quiser os detalhes de implementação, código e discussão
  completa.

`history/conceitos/` é um glossário didático separado. Sempre que um
capítulo usar um termo técnico (CKA, APD, Spearman, L2-SP, drift, layer 1
vs layer 2, WSD, etc.), ele aponta para o arquivo de conceitos
correspondente, onde o termo é explicado com analogias e com um exemplo
numérico tirado do próprio projeto.

Para uma primeira leitura, não é necessário estudar todos os conceitos
antes dos capítulos. O percurso mais eficiente é:

```text
1. ler este capítulo 00;
2. ler conceitos/07 para saber o que pode estar mudando;
3. acompanhar os capítulos históricos em ordem;
4. abrir os demais conceitos quando os links aparecerem;
5. usar conceitos/08 e conceitos/09 para reconstruir concretamente
   qualquer experimento checkpoint x corpus.
```

## A pergunta central do projeto

O projeto se chama TimeFormer e investiga **mudança temporal relacional**:
como o entorno de uma palavra se reorganiza entre períodos históricos.

O objeto principal é diretamente inspecionável:

```text
token@time
  -> perfil relacional no período
  -> vizinhos, aproximações e afastamentos
  -> deslocamento temporal do perfil
```

Formalmente, o TimeFormer estima primeiro:

```text
Delta_rel(w) = D(R_0(w), R_1(w))
```

onde `R_t(w)` descreve `w` por suas relações com referências lexicais no
período `t`. Os próprios vizinhos e suas mudanças são resultados, mesmo quando
não correspondem um a um a sentidos discretos de um inventário.

Uma segunda pergunta, mais estrita, é:

```text
quanto de Delta_rel(w) se associa a
Delta_sense(w) = D(P_0(s | w), P_1(s | w))?
```

Essa pergunta exige informação adicional sobre sentidos. No projeto, ela é
investigada com ConSeC e WordNet, não pressuposta pela saída do TimeFormer.

O benchmark usado para validar qualquer método é o **SemEval-2020 Task
1** (mudança semântica lexical, inglês, lematizado), com:

```text
D0 = corpus de 1810-1860
D1 = corpus de 1960-2010
37 palavras-alvo, cada uma com um "gold score" de 0 (sem mudança) a 1
(mudança forte)
```

A história documenta primeiro como tornar `Delta_rel(w)` comparável apesar da
mudança do sistema de coordenadas e, depois, quanto desse deslocamento pode
receber uma interpretação lexical mais estrita.

## Os quatro personagens

Quatro palavras do SemEval aparecem repetidamente, do início ao fim da
história, porque formam um conjunto de teste muito informativo: dois
casos onde o gold diz "mudou muito", e dois onde diz "não mudou muito" —
mas cada par tem uma característica distinta que quebra métodos
diferentes.

| Palavra | Gold (mudou?) | O que muda concretamente | Por que é difícil |
|---|---|---|---|
| `plane_nn` | Mudou muito (gold alto) | De "superfície geométrica/plana" (1810-1860) para "avião" (1960-2010) — **substituição de sentido dominante** | A mudança é uma migração inteira de campo semântico ("comunitária"), não um deslocamento sutil de vizinhos individuais |
| `chairman_nn` | Não mudou muito (gold baixo) | Continua "pessoa que preside uma organização" nos dois períodos, mas o tipo de organização (parlamento, empresa, comissão) varia | Métodos baseados em vizinhos/clusters acusam mudança porque o *contexto* varia, mesmo que o *sentido* não mude — falso positivo clássico |
| `graft_nn` | Mudou muito (gold alto) | De "enxerto de planta" (agricultura, 1810-1860) para um campo mais diversificado, incluindo "enxerto médico" e "ganho corrupto" (1960-2010) | Envolve **diversificação de sentidos**, não substituição simples — mais de um sentido novo aparece |
| `tree_nn` | Não mudou muito (gold baixo) | Continua "planta lenhosa" nos dois períodos | Serve de controle de estabilidade — qualquer método que acuse mudança forte em `tree_nn` provavelmente está respondendo a ruído/composição do corpus, não a sentido |

Sempre que um capítulo introduzir uma nova métrica ou método, ele vai
mostrar o que essa métrica diz para esses quatro casos. Isso permite
comparar fases diretamente: por exemplo, "no capítulo 05, `graft_nn`
aparecia no fundo do ranking; no capítulo 08, ele já aparece perto do
topo".

## Mapa dos capítulos

| # | Capítulo | Em uma frase |
|---|---|---|
| 01 | Ponto de partida (Token-Time/FiLM) | Primeira arquitetura tentou *ensinar* o modelo sobre o tempo; foi abandonada |
| 02 | A virada relacional | Mudança semântica = mudança nas *relações* entre palavras, não nas coordenadas absolutas |
| 03 | Validação no sintético | A ideia relacional funciona num corpus de laboratório, com controles |
| 04 | Refinando a pergunta | "Estrutural vs microvariação"; abandono de listas fixas de palavras-âncora |
| 05 | Primeiras tentativas no SemEval real | PMI/cloze sobre o corpus real; um bug de fronteiras de documento invalida meses de resultados |
| 06 | Hidden states e as paredes de identificabilidade | Muda-se de "prever palavras" para "comparar representações internas"; aparecem as paredes A-D |
| 07 | Realinhamento: instrumento de consulta temporal | TimeFormer redefinido como instrumento de consulta, não "detector de mudança" |
| 08 | Perfil relacional v2 e o gargalo do encoder | Tentativa de detectar "modos de sentido" automaticamente falha (NO-GO); descobre-se que o gargalo real é a capacidade do encoder |
| 09 | bert-tiny, Option D e L2-SP | Inicializar com BERT pré-treinado; layer 1 vs layer 2; por que a regularização do encoder foi abandonada |
| 10 | WSD externo, Gate 1 | Primeira porta de um desambiguador de sentido externo congelado — resultado misto (NO-GO parcial) |
| 12 | Adjudicação humana | A heurística é majoritariamente válida, mas o LMMS falha no sentido histórico de ferramenta |
| 13 | Segundo WSD externo | ConSeC reconhece 14/16 ferramentas e localiza a falha como principalmente específica do LMMS |
| 14 | Gate 1 completo | ConSeC passa geometria, ferramenta, aviação e autoriza um piloto pequeno entre palavras |
| 15 | Pré-registro da Porta 2 | Inventários, subconjuntos, cortes e auditoria são congelados antes de novas previsões |
| 16 | Generalização do ConSeC | Porta 2 passa em `graft` e `tree`, mas expõe a cobertura WordNet como próximo gargalo |
| 17 | Matriz de cobertura | Os 37 alvos recebem inventários e contextos antes da definição da Porta 3 |
| 18 | Distribuições de sentido | A JSD do ConSeC correlaciona com o gold e a Porta 3 passa |
| 19 | Replicação da Porta 3 | O resultado persiste em três amostras e após controlar o número de sentidos |
| 20 | Nulo intrapalavra | A divergência excedente preserva sinal e reduz o viés do inventário |
| 21 | Duas réguas não equivalentes | APD contextual e JSD de sentidos não convergem como scores por palavra |
| 22 | Geometria local de sentidos | Nas mesmas ocorrências, distância semântica acompanha distância vetorial |
| 23 | Composição e componente não atribuída | A troca de sentidos explica uma parcela pequena e mensurável; o restante não é identificado |
| 24 | Incerteza por palavra | Bootstrap separa a conclusão global dos dez casos individuais robustos |
| 25 | Consolidação da análise de sentidos | A validação semântica externa vira um pacote reproduzível, sem encerrar o eixo `token@time` |
| 26 | Recolocando `token@time` no centro | Vizinhanças são o resultado principal; sentidos externos são uma análise adicional |
| 27 | Framework de consultas temporais | Consulta, comparação, busca e rankings de `token@time` recebem um contrato operacional |
| 11 | Estado em 2026-06-13 | Onde o projeto está agora e o que vem a seguir |

## Mapa dos conceitos

Os seis primeiros arquivos explicam ferramentas específicas; os três
últimos fornecem as pontes que permitem montar o projeto inteiro
mentalmente.

| # | Arquivo | Pergunta que responde |
|---|---|---|
| 01 | Correlação, similaridade e informação | Como comparar rankings, vetores e distribuições? |
| 02 | Encoders, camadas e checkpoints | O que existe dentro do modelo e o que é salvo em cada etapa? |
| 03 | Deriva e esquecimento | O modelo mudou numericamente, funcionalmente ou semanticamente? |
| 04 | Perfis relacionais e APD | Como comparar palavras e nuvens de ocorrências? |
| 05 | Estatística experimental | Como evitar conclusões frágeis com 37 palavras? |
| 06 | WSD e sentido lexical | Como prever sentidos explicitamente com um inventário externo? |
| 07 | O que está sendo medido | Qual a diferença entre mudança de corpus, contexto, representação e sentido? |
| 08 | Desenhos temporais e réguas | O que significam diagonal, encoder fixo, treino contínuo e controles? |
| 09 | Dados, tokenização e ocorrências | Como uma linha do corpus vira tokens, WordPieces, janelas e vetores? |

Se uma pessoa entender especialmente os conceitos 07, 08 e 09, ela
consegue visualizar o fluxo completo:

```text
texto histórico
  -> ocorrência contextual
  -> encoder/checkpoint escolhido
  -> vetor ou distribuição
  -> métrica
  -> estimando científico
  -> decisão com incerteza e controles
```

## Convenções usadas neste histórico

- Números de Spearman, AUC, AP, etc. são sempre correlacionados com o
  `gold` do SemEval (37 palavras), salvo indicação contrária — `n=37` é
  uma amostra pequena, e isso é discutido explicitamente em
  [`conceitos/05-estatistica_experimental.md`](conceitos/05-estatistica_experimental.md).
- "D0" e "D1" sempre significam os corpora 1810-1860 e 1960-2010.
- `theta_0`, `theta_1` (ou `theta0`/`theta1`) são os checkpoints do
  modelo após treinar em D0 e depois continuar em D1, respectivamente.
  `theta_init` é o checkpoint antes de qualquer treino temporal (ponto de
  partida pré-treinado).
- Quando um capítulo descreve uma linha de investigação que foi
  **abandonada**, isso é dito explicitamente — o objetivo não é apenas
  mostrar o que "deu certo", mas também preservar o raciocínio por trás
  de cada descarte, porque esse raciocínio é parte do valor científico do
  projeto.
- "Mudança" sem qualificador é evitada sempre que houver risco de
  ambiguidade. O projeto distingue mudança de corpus, de contexto, de
  representação e de sentido; ver
  [`conceitos/07-o_que_esta_sendo_medido.md`](conceitos/07-o_que_esta_sendo_medido.md).
