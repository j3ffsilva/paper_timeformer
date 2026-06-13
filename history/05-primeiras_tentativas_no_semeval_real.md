# Capítulo 5 — Primeiro contato com o SemEval real: ruído, entropia e um bug de fronteiras

> Fontes originais: `tmp/16-claude_pmi_profile_failure_prompt.md`,
> `tmp/17-timeformer_pmi_profile_failure_review.md`,
> `tmp/18-claude_post_boundary_fix_prompt.md`,
> `tmp/19-timeformer_post_boundary_fix_review.md`.

O capítulo 04 deixou uma formulação elegante — perfil relacional via
log-PMI sobre a saída do MLM head, sem âncoras pré-definidas — e uma
lista de quatro questões abertas (estabilidade de `p_t`, cobertura mínima
por palavra, esparsidade, trajetória). Este capítulo é sobre o que
aconteceu quando essa formulação encontrou, pela primeira vez, o corpus
real do SemEval-2020 Task 1: **399 mil e 420 mil janelas de treino**,
**27.311 tokens de vocabulário**, **37 palavras-alvo com gold conhecido**.

## O resultado: indistinguível de ruído

Depois de treinar `theta_0` em D0 (1810-1860) e continuar em D1
(1960-2010), e calcular `pmi_cosine` e `ppmi_jsd` para as 37 palavras-alvo,
o resultado contra o gold do SemEval foi:

```text
Spearman = -0.025  (p = 0.88)
ROC-AUC  = 0.494
```

Ou seja: **nenhuma correlação com o gold**, e a métrica binária acerta
exatamente o que uma moeda jogada acertaria. Aumentar o número de épocas
não ajudou. Pior: uma análise de correlação revelou que `pmi_cosine`
correlacionava fortemente — `rho = 0.946` — com a **variação de entropia
do corpus** entre os dois checkpoints, não com mudança semântica. O
método estava medindo "o quanto a distribuição geral do modelo mudou de
forma", não "o quanto o significado de uma palavra específica mudou".

## Diagnóstico, palavra por palavra: `graft_nn` como caso de estudo

Para entender *por que*, a investigação usou `graft_nn` (o personagem que
diversifica sentido — capítulo 00) como caso de estudo, porque ele tem
uma mudança de sentido bem documentada: 77 de 119 ocorrências em D0 são
do sentido botânico ("enxerto de planta"), enquanto em D1 apenas 4 de 109
ainda são botânicas, e surgem 24 ocorrências de "corrupção" e 22 de
"medicina".

O diagnóstico (`diagnose_cloze_semantics.py`) revelou que `q_t(graft\_nn)`
— a distribuição que o modelo prevê no lugar de `graft_nn` — era
**quase uniforme**: entropia normalizada `H/log|V| ≈ 0.78` (1.0 seria
totalmente uniforme). O rank de `graft_nn` na sua própria distribuição
prevista era **2930 de 27.311**. Nenhuma das palavras semanticamente
esperadas — `scion`, `branch`, `stock` (botânico), `bribery`, `scandal`
(corrupção), `transplant` (médico) — aparecia no top-20, em nenhuma
combinação de checkpoint × corpus.

Mesmo em contextos inequivocamente botânicos como:

> "cut off and graft the top first give the **[MASK]** there the best
> possible chance..."

o top da predição era `[UNK]`, `and`, `man`, `own`, `time` — sem nenhum
traço de "scion" ou qualquer palavra do campo botânico. **`q_t(w)` não
era sensível ao sentido — fato observado, não inferência.**

## Causas concorrentes: nenhuma isolada basta

A primeira revisão (`tmp/17`, antes da correção de fronteiras) identificou
quatro causas, hierarquizadas por confiança:

| Hipótese | Confiança | Evidência |
|---|---|---|
| **H1 — Mascaramento central determinístico cria desalinhamento de posição** | Alta (fato confirmado) | Treino sempre mascara a posição central da janela (`mask_pos = candidate_positions[len(candidate_positions)//2]`); o probe mascara a posição real da ocorrência. 24,4% das ocorrências de `graft_nn` caem em posições (>16) **nunca vistas no treino**. |
| **H2 — Modelo subtreinado/pequeno demais** | Alta (fato) | `d_model=96`, 2 camadas, 4 cabeças, para 27.311 tokens de vocabulário e ~800 mil janelas. Loss final ≈ 5.96-6.0 (perplexidade ~120-130 — um BERT-base chega a perplexidade ~4-6). |
| **H3 — A média cloze apaga multimodalidade** | Média (não separável de H2 ainda) | Em princípio, a média de distribuições por sentido poderia refletir a proporção entre sentidos. Mas como cada distribuição individual já é quase uniforme, o problema está antes da média. |
| **H4 — Prior `p_t` fora de distribuição** | Baixa (secundário) | O probe neutro `[CLS] [MASK] [SEP]` usa uma sequência de 3 tokens, mas o modelo foi treinado em janelas de 32. Posição 1 nunca foi mascarada no treino. |

A conclusão prática: **com `q_t(w)` quase uniforme, o log-PMI amplifica
ruído de cauda** — diferenças minúsculas entre tokens raros com
probabilidades como `1e-5` vs `1e-7` (puro acidente de calibração) passam
a dominar o perfil `R_t(w)`, sem qualquer relação com semântica. Esse é o
mecanismo concreto por trás do `rho=0.946` com a variação de entropia: o
score estava capturando "o quanto a calibração geral do modelo mudou
entre checkpoints", que é dominado por capacidade insuficiente e
esquecimento catastrófico parcial (a loss do período 1 começa em 6.02,
*acima* da loss final do período 0, indicando que o modelo "esqueceu"
parte do que sabia ao virar a página para D1 —
[conceitos/03](conceitos/03-deriva_e_esquecimento.md)).

## O bug de fronteiras

No meio dessa investigação, surgiu um achado independente, mais
estrutural: **as janelas MLM de treino e os probes de ocorrência estavam
sendo construídos sobre texto corrompido**.

O preparador do corpus (`prepare_semeval2020_task1.py`) escreve uma
sentença por linha — corretamente, porque o SemEval distribui o corpus
como "sentenças embaralhadas aleatoriamente, sem relação entre linhas
consecutivas" (confirmado no README oficial do dataset). Mas o leitor do
corpus (`real_corpus.py`) tratava **o arquivo inteiro de um período como
um único documento**:

```python
# antes (bug)
documents = [tokenize(path.read_text(...))]

# depois (correção)
documents = [
    tokenize(line)
    for line in path.read_text(...).splitlines()
    if line.strip()
]
```

A consequência: o *sliding window* de 30 tokens, ao varrer "um documento
gigante", criava janelas que **atravessavam fronteiras de sentenças
completamente não relacionadas** — concatenando o final de uma frase
sobre, digamos, agricultura em 1850 com o início de uma frase sobre
política, só porque elas ficaram adjacentes depois do embaralhamento do
corpus. O impacto quantitativo:

| | Antes (bug) | Depois (corrigido) |
|---|---:|---:|
| Janelas MLM, 1810-1860 | 409.401 | 299.534 |
| Janelas MLM, 1960-2010 | 420.743 | 366.340 |

Ou seja, **cerca de 110 mil janelas no período antigo e 54 mil no
moderno** dependiam dessa concatenação indevida. Para `graft_nn`
especificamente, estimou-se que **73,9% das ocorrências em D0 e 88,1% em
D1** tinham sua janela de contexto cruzando uma fronteira de sentença
espúria.

A revisão pós-correção (`tmp/19`) confirmou: a correção é **semanticamente
correta** — "tratar o arquivo como um único documento conectado criava
janelas MLM que conectam o fim de uma sentença com o início de outra,
produzindo sequências semanticamente incoerentes que corrompem o sinal de
treino". Os checkpoints anteriores a essa correção foram declarados
**inválidos** para avaliar o método.

## Corrigir o bug não bastou

Esse é o ponto mais importante deste capítulo, e a razão de incluí-lo na
história mesmo sendo "só um bug": **corrigir as fronteiras de documento
era necessário, mas não suficiente**. A revisão pós-correção refez o
diagnóstico do H1 (desalinhamento de posição de máscara) com o corpus
corrigido e **confirmou ambas as causas como co-primárias**:

- **H1 confirmado como fato**: a distribuição de posições de máscara no
  treino tem média 12.0 (desvio 4.1, faixa [2,16]) — porque 70% das
  sentenças do corpus têm menos de 30 tokens, então o mascaramento central
  cai sempre perto do meio de sentenças curtas. Já a distribuição de
  posições no probe de `graft_nn` tem média 14.2 (desvio 8.0, faixa
  [1,30]). **24,4% das ocorrências caem em posições nunca mascaradas no
  treino.** Com embedding posicional absoluto, essas posições produzem
  predições degeneradas.
- **H2 confirmado como fato**: mesmo restringindo a ocorrências cuja
  máscara cai na posição "vista" (posição 16, ~28% dos casos), as
  predições continuam dominadas por function words (`and`, `the`, `be`).
  O modelo simplesmente não tem capacidade para associações lexicais
  informativas.

A revisão também propôs um controle barato e definitivo, ainda não
executado nesta fase: comparar `JSD(theta_0, theta_1)` com
`JSD(random_init_1, random_init_2)` — dois modelos com pesos
**aleatórios**, nunca treinados. Se o JSD entre checkpoints treinados for
do mesmo tamanho que o JSD entre dois modelos aleatórios, o "efeito de
checkpoint" observado (4-5x maior que o "efeito de corpus") não é
evidência de aprendizado algum — é ruído de inicialização.

## O que vale levar deste capítulo

- **Um bug real e concreto** (fronteiras de documento) foi encontrado,
  documentado quantitativamente e corrigido — e isso por si só já
  invalidou meses de checkpoints anteriores. Esse tipo de achado é
  valioso mesmo quando (como aqui) não "resolve" o problema central.
- A lição mais geral, porém, é sobre **como interpretar uma correlação
  alta com a explicação errada**: `rho=0.946` com variação de entropia
  parecia, a princípio, "um sinal forte" — mas era o sinal forte de uma
  variável de confusão (capacidade/calibração do modelo), não da variável
  de interesse (mudança semântica). Distinguir as duas exigiu descer até
  o nível de uma única palavra (`graft_nn`) e uma única ocorrência.
- A combinação **modelo pequeno + mascaramento central determinístico +
  probe em posição real** é uma armadilha sutil: cada peça isolada parece
  uma escolha de implementação razoável, mas juntas produzem um
  desalinhamento sistemático entre o que o modelo foi treinado para fazer
  e o que o probe pede dele.
- Esse capítulo marca a primeira vez que o projeto precisa decidir entre
  "aumentar a capacidade do modelo" (capítulos 08-09, bert-tiny) e
  "trocar a representação por algo que não dependa de `q_t(w)` ser
  informativo" (capítulo 06, hidden states).

## Conceitos novos usados neste capítulo

- [Entropia normalizada e perplexidade](conceitos/01-correlacao_e_similaridade.md#entropia)
- [Dados lematizados, janelas e fronteiras de documento](conceitos/09-dados_tokenizacao_e_contexto.md)
- [Embedding posicional e mascaramento (MLM)](conceitos/02-encoders_e_camadas.md#mascaramento)
- [Esquecimento catastrófico entre checkpoints](conceitos/03-deriva_e_esquecimento.md)
- [Controle com pesos aleatórios](conceitos/05-estatistica_experimental.md#controle-aleatorio)
