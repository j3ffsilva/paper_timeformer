# Capítulo 8 — Perfil relacional v2: uma definição mais rigorosa, e o gargalo que ela revela

> Fontes originais: `docs/12-novo_perfil_relacional.md`,
> `docs/14-perfil_relacional_v2_resultados_fase1.md` (§1-§7.24),
> `tmp/28-codex_perfil_relacional_v2_plan_review.md`,
> `tmp/29-codex_semantic_modes_v2_nogo_review.md`.

O capítulo 07 terminou com um reframing (instrumento de consulta) e uma
pendência concreta: o relatório qualitativo de vizinhanças temporais foi
satisfatório para `plane_nn` e `chairman_nn`, mas insatisfatório para
`graft_nn`, e o sinal quantitativo (APD = 0,210) continuava sem
significância estatística. Em paralelo a esse reframing, teve início um
esforço para **reescrever a formalização do perfil relacional do zero**,
corrigindo problemas matemáticos sutis que se acumularam desde o capítulo
04 — esse é o "Perfil Relacional v2" (`docs/12`). Este capítulo conta o
que essa reformulação corrigiu, o que ela tentou (e não conseguiu) ir
além, e o diagnóstico que ela acabou produzindo sobre **onde está o
verdadeiro limite do projeto**.

## O que a v2 corrige na v1

`docs/12` lista onze mudanças em relação à formulação anterior. As mais
importantes:

1. **Centralização por período antes de qualquer cosseno**:
   `ê_t(x) = (e_t(x) - mu_t) / ||e_t(x) - mu_t||`, onde `mu_t` é um vetor
   médio do período `t`. Sem essa centralização, uma translação
   sistemática do espaço entre `theta_0` e `theta_1` (algo que o capítulo
   03 já havia identificado como não-cancelável pelo cosseno cru) inflaria
   artificialmente qualquer `Delta`.
2. **Uma prova de invariância mais honesta**: o cosseno sobre embeddings
   centralizados é invariante a rotação, reflexão e escala isotrópica — e
   a translação é eliminada *por construção* pela centralização. O que
   **não** é coberto por essa prova — "drift anisotrópico e não-linear
   residual" — é explicitamente declarado como algo que **precisa ser
   medido empiricamente**, não assumido. Essa é uma formulação mais
   cautelosa do que a v1, que afirmava informalmente "o cosseno é
   invariante ao sistema de coordenadas" sem qualificar a classe de
   invariância.
3. **A matriz de coesão semântica `M_t(w)`** (§7): para o suporte filtrado
   `V_w = {v : P_t(w)[v] > tau}`,
   ```text
   M_t(w)[v,v'] = P_t(w)[v] · P_t(w)[v'] · cos(ê_t(v), ê_t(v'))
   ```
   — uma matriz PSD que combina três condições simultâneas (`v` relevante
   para `w`, `v'` relevante para `w`, `v` e `v'` no mesmo campo), decomposta
   via SVD sem nunca ser formada explicitamente. Os autovetores dessa
   matriz seriam os **modos semânticos** de `w`.
4. **Critério de gap** (§8): para detectar automaticamente tanto o limiar
   `tau` (que define `V_w`) quanto o número de modos `k`, sem
   hiperparâmetros fixos — usando o gap relativo `h_i = (X_i -
   X_{i+1})/X_i` entre valores ordenados, aceito apenas se exceder um
   limiar `gamma` (sugestão 0,3). Se nenhum gap excede `gamma`, o método
   **declara explicitamente "sem estrutura clara"** em vez de forçar uma
   fronteira arbitrária — um princípio que o documento chama de "nenhuma
   fronteira é fixada a priori; toda fronteira emerge dos dados,
   acompanhada de uma condição de validade que sinaliza quando a estrutura
   não existe".
5. **Persistência de modos entre períodos via espaço de tokens** (§11):
   como os modos vivem como vetores de cargas sobre o vocabulário fixo `V`
   (não em coordenadas internas do encoder), eles são comparáveis entre
   `theta_0` e `theta_1` sem Procrustes — extensão direta da lógica que já
   havia tornado os perfis relacionais comparáveis (capítulo 04). O
   emparelhamento usa o algoritmo húngaro, com limiar `theta` calibrado por
   **distribuição nula de permutação** (similaridades entre modos de
   palavras *diferentes* — pares que sabidamente não são continuação um do
   outro), em vez de um valor fixo arbitrário.
6. **Protocolo de validação em camadas** (§13): piso de drift (palavras de
   controle estáveis), split-half intra-período (limite inferior de
   detecção), shuffle temporal (teste de vazamento), e validação contra
   anotação de sentido (B-Cubed/V-measure).

A revisão de `tmp/28` (antes de qualquer implementação) já sinalizou os
riscos principais: `d_model` pequeno (96-128) pode limitar a
separabilidade espectral exigida por §7-8 ("rank(M) ≤ min(|V_w|, d)"); com
apenas 2 períodos no SemEval, a persistência de modos (§11) tem material
limitado; e — mais importante — recomendou um **experimento go/no-go
barato antes de construir toda a infraestrutura de matching e validação**.
Essa recomendação se tornou a "Fase 1.5" descrita a seguir.

## Fase 1: a centralização certa, sinal ainda fraco

A primeira etapa de implementação testou quatro formas de calcular `mu_t`
(o vetor de centralização), todas avaliadas com o mesmo
`Delta(w) = 1 - cos(P_{t0}(w), P_{t1}(w))`:

| Variante | Definição de `mu_t` | Spearman (melhor camada) | ROC-AUC |
|---|---|---:|---:|
| A — média das ~3.216 referências (≈ abordagem v1) | média dos centróides de um conjunto pequeno de referência | 0,108 | 0,592 |
| B — média global ponderada por ocorrência | `sum(sums_t)/sum(counts_t)` sobre todo o vocabulário | 0,005 | 0,506 |
| C — igual a B, restrito ao suporte ativo | idem, mas só sobre V_ativo (n_min=50) | 0,044 | 0,500 |
| **D — média não ponderada por tipo** | cada *tipo* de token conta igual, independente de frequência, sobre V_ativo (n_min=10) | **0,124** | **0,601** |

A diferença entre B/C e D é qualitativa, não apenas quantitativa: `mu_t`
ponderado por ocorrência (B/C) é dominado por palavras de função de
altíssima frequência (`the` tem ~448.771 ocorrências, contra uma mediana
muito menor entre os ~11.600 tokens do suporte ativo) — isso empurra
`mu_t` para perto do centróide de `the/of/and`, deixando `ê_t(v)` quase
uniforme entre palavras de conteúdo, e o resultado **abaixo do acaso**
(ROC-AUC ≈ 0,49-0,51). A variante D (cada palavra-tipo conta igual) foi
adotada como padrão — mas a diferença para A não é estatisticamente
significativa com `n=37` (`p>0,4` em ambas), e nenhuma variante separa
bem "mudado" de "estável" (`changed_above_stable_p95` ≈ 0 em quase todas).

## Fase 1.5: o NO-GO da decomposição em modos

A "Fase 1.5" — o experimento go/no-go recomendado por `tmp/28` — testou se
a matriz de coesão `M_t(w)` realmente produz `k >= 2` modos para palavras
com mudança conhecida (`plane_nn`, `graft_nn`) e `k = 1` para controles
estáveis (`chairman_nn`, `tree_nn`, `ball_nn`, `face_nn`, `lane_nn`,
`multitude_nn`).

Três formulações foram testadas:

1. **Gap sobre todo `P_t(w)` positivo**: `tau` saiu da ordem de `1e-4`
   (praticamente zero), `|V_w|` entre ~4.800 e ~5.900 (quase todo o
   suporte ativo positivo). `k=1` para `plane_nn` e `graft_nn` (as duas
   palavras com mudança conhecida!) em ambos os períodos; `k=2`
   apareceu só em `face_nn`@D1 e `multitude_nn`@D0 — **ambos controles
   estáveis**. Além disso, extremamente instável a `n_min`: `plane_nn`
   perde todo o suporte em `n_min=50`.
2. **Top-N por `|P_t(w)[v]|` (N=100 ou 500), gap depois**: `tau=None`,
   `k=None`, `|V_w|=0` em **todos os 16 casos** — nenhum gap relativo
   excede `gamma=0,3` entre os candidatos mais correlacionados.
3. **Top-N positivo fixo, gap só nos autovalores de `M_t(w)`**: `k=1` em
   **todos os 48 casos** (8 palavras × 2 períodos × N∈{50,100,200}), sem
   exceção. O primeiro autovalor domina o segundo por **10-30x** — por
   exemplo, `plane_nn`@D0 com N=100: `lambda = [53,39; 3,24; 2,69; 2,05]`
   (razão 17x).

### O diagnóstico: o perfil é uma rampa, não tem clusters

A causa-raiz foi encontrada inspecionando `P_t(plane\_nn)[v]` diretamente,
ordenado:

```text
top 15:   [0.9456, 0.9424, 0.9030, 0.8990, 0.8975, ..., 0.8661]
bottom 5: [-0.7146, -0.7216, -0.7296, -0.7571, -0.7649]
```

O perfil decai de forma **suave e quase monotônica** de ~0,95 a ~-0,76 ao
longo de ~5.700-11.600 tokens, com gaps relativos consecutivos da ordem de
0,003-0,05 — nenhum salto >30% perto do topo. O único gap >30% que o
critério encontra fica perto do cruzamento de sinal, que não tem
significado semântico. A auditoria independente (`tmp/29`) confirmou que a
implementação está fiel ao documento canônico — não é um bug de sinal,
eixo ou normalização.

**Decisão registrada: NO-GO para §7-9 (modos semânticos via SVD de
`M_t(w)`) neste regime** (`d_model=128`, 3 camadas, |V_ativo|~5.700-11.600).
O problema não é de calibração (`gamma`, `n_min`, `top_n`) — é que a
premissa de §7-9 (que `P_t(w)` ou `M_t(w)` tenham estrutura de cluster
multimodal detectável por gap espectral) **não se sustenta nesta
representação**.

## A cadeia de diagnósticos: de "rampa" a "eixo de época"

A partir do NO-GO, uma sequência de testes baratos (reaproveitando o cache
já existente, sem reextração) foi montada para entender *por que* a rampa
existe e se ela também explica o sinal fraco da Fase 1 (Spearman ≈ 0,12).

### Passo 0: bypass do centróide não ajuda

Calcular APD diretamente sobre ocorrências individuais (sem nunca passar
por um centróide) deu Spearman = 0,130/0,127 (layer_2/mean_last_2) —
**estatisticamente indistinguível** do `Delta` da Fase 1 (0,124). Hipótese
"a agregação em centróide é a causa" descartada: bypassá-la não recupera
sinal.

### Teste de bimodalidade: não discrimina

GMM (k=1 vs. k=2) sobre a nuvem de ocorrências de cada palavra: `delta_bic`
positivo para **todas** as palavras (k=2 sempre "ganha" estatisticamente
com centenas/milhares de pontos em d=128) — e, mais revelador,
**`tree_nn` (controle estável) tem o maior `delta_bic` de todos** (129.581,
contra 29.085 de `plane_nn` e 9.351 de `graft_nn`). O oposto do esperado
se `delta_bic` rastreasse estrutura de sentido.

### O eixo de época: k-means(2) recupera o período, não o sentido

Particionando a nuvem combinada (D0+D1) de cada palavra em 2 clusters via
k-means e medindo a associação cluster×período via NMI:

| Palavra (esperado) | NMI (`layer_2`) | NMI (`mean_last_2`) | n_D0 / n_D1 |
|---|---:|---:|---:|
| `chairman_nn` (estável) | 0,002 | 0,965 | 147 / 683 |
| `graft_nn` (mudou) | 0,423 | 0,964 | 119 / 109 |
| `plane_nn` (mudou) | 0,988 | 0,988 | 278 / 792 |
| `tree_nn` (estável) | 1,000 | 0,997 | 2322 / 1596 |

Em `mean_last_2`, um corte k-means=2 da nuvem combinada recupera o
**período** quase perfeitamente para qualquer palavra — mudada ou
estável. `tree_nn` (3.918 ocorrências, controle estável) chega a NMI=1,000
em `layer_2` — a maior associação cluster×período de toda a amostra. Essa
descoberta explica retroativamente o `delta_bic` do teste de bimodalidade
anterior: k=2 "ganha" porque k=2 ≈ "separar por período", não porque há
dois sentidos.

### Decompondo o eixo de época: drift de checkpoint, não corpus

Usando a grade 2×2 já disponível no cache (cada checkpoint × cada corpus),
foi possível separar duas explicações possíveis para esse eixo:

| Combinação | Isola | NMI médio (`mean_last_2`) | NMI mediano |
|---|---|---:|---:|
| `theta0_d0` vs `theta1_d1` (diagonal, original) | checkpoint + corpus | 0,805 | 0,982 |
| `theta1_d0` vs `theta1_d1` (encoder fixo) | só corpus | 0,029 | 0,012 |
| `theta0_d0` vs `theta1_d0` (dados fixos D0) | só checkpoint | 0,857 | 1,000 |
| `theta0_d1` vs `theta1_d1` (dados fixos D1) | só checkpoint | 0,755 | 0,995 |

Com o encoder fixo (segunda linha), o corpus por si só é **quase não
separável** (NMI ≈ 0,01-0,03). Com os dados fixos (terceira/quarta linha),
o checkpoint por si só já reproduz quase todo o NMI da diagonal. **O eixo
de época é, esmagadoramente, drift de checkpoint — não diferença de
corpus.** As mesmas sentenças de D0, codificadas por `theta_0` vs.
`theta_1`, já são quase perfeitamente separáveis por um corte k-means=2.

Esse resultado é uma reformulação, com evidência direta, do que o capítulo
07 já havia chamado de "régua móvel" — e quantifica precisamente o quão
grande é essa régua móvel: **maior que a diferença entre os dois corpora**.

### Recentralização aditiva: corrige o eixo, não corrige o Spearman

Subtraindo a média global por "lado" (a mesma ideia da centralização `mu_t`
da Fase 1, mas no nível de ocorrência individual em vez de centróide), o
NMI mediano cai de ~0,98 para ~0,05-0,09 — a correção aditiva resolve a
maior parte do eixo de época para a palavra mediana. **Mas o Spearman do
APD recentralizado (0,093-0,100) fica praticamente igual ao original
(0,127-0,130)** — dentro do ruído. Conclusão: "remover o eixo de época"
(útil para destravar a Fase 1.5) e "destravar o Spearman contra o gold" são
**dois problemas distintos**, não o mesmo problema visto de duas formas. E
a centralização `mu_t` da Fase 1 já fazia, em boa parte, essa correção —
por isso ela já estava "embutida" no resultado de 0,124.

## Encoder fixo: `theta_1` é melhor régua que a diagonal

Uma correção mais direta do que recentralizar: medir D0 e D1 com o **mesmo
checkpoint**, em vez de cada período com seu checkpoint "nativo"
(configuração "diagonal" usada até aqui).

| Modelo fixo | Layer | APD Spearman (p) | APD ROC-AUC / AP |
|---|---|---:|---:|
| `theta_0` (só viu 1810-1860) | layer_2 | 0,016 (p=0,93) | 0,586 / 0,624 |
| `theta_0` | mean_last_2 | 0,048 (p=0,78) | 0,568 / 0,608 |
| `theta_1` (viu D0 e D1 no treino contínuo) | layer_2 | **0,204** (p=0,23) | 0,604 / 0,635 |
| `theta_1` | mean_last_2 | **0,202** (p=0,23) | **0,619 / 0,663** |

Com `theta_0` fixo, o sinal praticamente desaparece (≈ acaso). Com
`theta_1` fixo, o APD sobe para Spearman ≈ 0,20 e AP ≈ 0,66 — **o melhor
resultado da investigação até este ponto**, embora ainda não significativo
com `n=37` (precisaria de ≈ 0,33 para `p<0,05`). Faz sentido: `theta_1` viu
os dois recortes de corpus durante o treino contínuo, então é o modelo mais
"neutro" para compará-los; `theta_0` nunca viu os textos de D1 e mede-os
mal. **Daqui em diante, `theta_1` aplicado aos dois corpora passa a ser a
configuração padrão de medição.**

## Modos primeiro, perfil depois: um protótipo qualitativo

Com `theta_1` fixo, foi tentado um protótipo que inverte a ordem do
documento canônico: em vez de decompor o **perfil agregado** em modos
(o que falhou na Fase 1.5), primeiro agrupar a **nuvem de ocorrências** de
cada palavra em 2-5 grupos (via silhouette), e só depois descrever cada
grupo por seu próprio perfil de vizinhos.

| Personagem | `k` escolhido | JSD(D0,D1) | Leitura |
|---|---:|---:|---|
| `graft_nn` | 2 | **0,473** (maior) | modo 0 (81% D0): `prism, thermometer, populace, nucleus, ...`; modo 1 (73% D1): `data, commodity, innovation, device, organism, ...` — **vocabulários claramente diferentes, composição temporal muda** |
| `chairman_nn` | 3 | 0,367 | os 3 modos compartilham o mesmo campo (`governor, president, secretary, commander, director`) — JSD alto aqui parece **falso positivo**: separação por algum eixo de uso, não por sentido |
| `plane_nn` | 2 | 0,095 | modo 0 (`boat, ship, road, car, truck`) e modo 1 (`route, vehicle, rail, panel, budget`) — **ambos do campo "transporte"**; o sentido geométrico esperado não aparece como modo distinto |
| `tree_nn` | 2 | **0,028** (menor) | os dois modos têm vocabulários muito parecidos (`valley, wood, forest, river, mountain`) — "a mesma árvore", consistente com ser controle estável |

`graft_nn` é o caso "claramente bom" (modos com vocabulários diferentes +
composição temporal diferente + maior JSD da amostra) e `tree_nn` é o caso
"correto por estabilidade". Mas `plane_nn` — o caso mais citado na
literatura do projeto — **não produziu o modo geométrico esperado**, e
`chairman_nn` teve JSD alto por um motivo aparentemente não relacionado a
sentido. Conclusão: a abordagem "modos primeiro" às vezes funciona,
às vezes não — não é confiável isoladamente, mas produz saídas que valem
inspecionar, o que o perfil agregado da Fase 1.5 nunca produziu.

## O teste decisivo: o teto de oráculo com BERT pré-treinado

A pergunta que ficou pendente em todos os diagnósticos anteriores é: será
que o **encoder pequeno** (`d_model=128`, 3 camadas, treinado do zero só
com MLM contínuo no SemEval) é o limite, ou o limite está nos dados/tarefa?

A resposta veio de repetir a medida mais simples (APD) usando
**`bert-base-uncased`**, baixado pronto, **sem nenhum treino** nos dados do
projeto, sobre as mesmas frases (apenas destokenizando `plane_nn` →
`plane`):

| Encoder | Camada | APD Spearman (p) | ROC-AUC | AP |
|---|---|---:|---:|---:|
| TimeFormer, diagonal | mean_last_2 | 0,124-0,130 (p≈0,45) | ≈0,59 | ≈0,60 |
| TimeFormer, `theta_1` fixo | mean_last_2 | 0,202-0,204 (p≈0,23) | ≈0,62 | ≈0,66 |
| **BERT-base, congelado** | última camada | **0,594 (p=0,0001)** | **0,693** | **0,659** |
| BERT-base, congelado | média das últimas 4 | 0,591 (p=0,0001) | 0,676 | 0,592 |

**Pela primeira vez em toda a investigação, o resultado é estatisticamente
significativo** — e bem acima dos ≈0,40-0,55 citados como referência da
literatura do SemEval-2020 para inglês.

A confirmação qualitativa nos quatro personagens é direta:

| Personagem (esperado) | APD (BERT) | NMI cluster×período (BERT) |
|---|---:|---:|
| `plane_nn` (mudou) | **0,566** (maior) | **0,487** (muito acima dos outros) |
| `chairman_nn` (estável) | 0,466 | 0,092 |
| `graft_nn` (mudou) | 0,377 | 0,140 |
| `tree_nn` (estável) | 0,395 | **0,002** (quase zero) |

Inspecionando as frases reais: em D0, todas as ocorrências de `plane_nn`
são do sentido **geométrico** ("the plane of projection", "the
intersection of the planes"); em D1, todas são do sentido **aviação**
("by plane", "his private plane to Vienna's airport"). O BERT separa essas
duas nuvens quase exatamente ao longo da linha do período (NMI=0,487) —
**exatamente o "modo geométrico vs. modo aviação" que o encoder pequeno do
TimeFormer não conseguiu separar** no protótipo "modos primeiro" desta
mesma seção. E `tree_nn` (estável) tem NMI≈0 sob BERT — como esperado para
uma palavra que não mudou de sentido.

## A conclusão: gargalo = encoder

Cruzando os dois resultados — `theta_1` fixo (≈0,20) e BERT-base (≈0,59) —
o quadro fica:

```text
BERT-base (≈0,59)  >>  theta_1 fixo (≈0,20)  >>  diagonal original (≈0,13)
```

Esse é o cenário "**gargalo = encoder**":

- A tarefa **é solúvel** com estas frases e este gold — um encoder de
  qualidade suficiente chega a Spearman ≈ 0,59, muito além do "teto" de
  ≈0,13-0,20 visto até aqui.
- O drift de checkpoint (eixo de época) e a falta de modos no perfil
  agregado eram problemas **reais**, e corrigi-los parcialmente (`theta_1`
  fixo) gerou ganho real (0,13 → 0,20) — mas a maior parte da distância até
  0,59 não vem deles. Vem da **qualidade/capacidade do encoder em si**:
  `d_model=128`, 3 camadas, treinado do zero apenas com MLM contínuo no
  corpus do SemEval, sem nenhum pré-treino.

A recomendação que sai deste capítulo, e que abre o capítulo 09, é
explícita: o próximo passo de maior retorno não é aumentar `d_model`
treinando do zero — é **inicializar o TimeFormer a partir de um checkpoint
pré-treinado** (BERT-base ou um encoder pequeno pré-treinado equivalente)
**antes** do treino contínuo temporal, mantendo o restante do pipeline
(perfil relacional v2, encoder fixo, agrupamento de ocorrências) como
infraestrutura de medição.

## O que vale levar deste capítulo

- A v2 não foi um exercício puramente formal: a centralização por tipo
  (variante D) e a identificação do "eixo de época" são correções reais
  que sobrevivem e são incorporadas ao protocolo. Mas a peça mais ambiciosa
  — decomposição espectral em modos semânticos (§7-9) — recebeu um **NO-GO
  bem documentado**, com uma causa-raiz geométrica clara (o perfil é uma
  rampa suave, sem clusters discretos) confirmada por auditoria
  independente.
- A cadeia de diagnósticos (Passo 0 → bimodalidade → eixo de época → drift
  de checkpoint → recentralização → encoder fixo → oráculo BERT) é um
  exemplo de **eliminação sistemática de hipóteses**: cada passo descarta
  uma explicação específica (agregação, dispersão, eixo aditivo) antes de
  chegar à causa que realmente explica a distância entre 0,13 e 0,59 —
  capacidade do encoder.
- O "eixo de época" — checkpoints consecutivos de um treino contínuo são
  quase perfeitamente separáveis por um corte simples, mais até do que os
  dois corpora — é uma quantificação direta da "régua móvel" do capítulo 07,
  e um achado que provavelmente generaliza para qualquer experimento futuro
  com `theta_0`/`theta_1`: **medições devem usar um encoder fixo**, nunca a
  configuração "diagonal".
- `plane_nn` continua sendo o personagem mais informativo para *mostrar* o
  gargalo: o protótipo "modos primeiro" com o encoder pequeno não encontrou
  o modo geométrico; o oráculo BERT o encontra imediatamente (NMI=0,487). A
  diferença entre os dois é, neste capítulo, a evidência mais direta de que
  o problema é de capacidade do encoder, não de formulação da métrica.

## Conceitos novos usados neste capítulo

- [SVD, autovalores/autovetores e matriz PSD](conceitos/04-perfis_relacionais_e_apd.md#svd)
- [Critério de gap relativo](conceitos/04-perfis_relacionais_e_apd.md#gap)
- [NMI/AMI (normalized/adjusted mutual information)](conceitos/04-perfis_relacionais_e_apd.md#nmi)
- [GMM e critério BIC](conceitos/05-estatistica_experimental.md#gmm-bic)
- [Encoder fixo vs. "diagonal"](conceitos/02-encoders_e_camadas.md#encoder-fixo)
- [A grade 2x2 completa e a analogia da régua](conceitos/08-desenhos_temporais_e_reguas.md)
- [Ocorrência, WordPiece, hidden state e amostragem](conceitos/09-dados_tokenizacao_e_contexto.md)
- [Oráculo pré-treinado como teto de referência](conceitos/05-estatistica_experimental.md#oraculo)
