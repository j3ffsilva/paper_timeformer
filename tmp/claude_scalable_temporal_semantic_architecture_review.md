# Pedido de segunda opinião: arquitetura escalável para mudança semântica temporal

Queremos avaliar uma proposta arquitetural para superar as paredes encontradas
em nossos experimentos de mudança semântica temporal. A proposta não deve ser
avaliada pela capacidade de consertar apenas `plane_nn`, `chairman_nn`,
`graft_nn` e `tree_nn`. Esses casos serão usados como testes de contrato de uma
arquitetura que precisa operar, sem engenharia manual por palavra, sobre todo um
vocabulário.

Não assuma que a proposta está correta. Procure especialmente:

1. vazamento do benchmark ou de conhecimento temporal;
2. dependência excessiva do inventário lexical;
3. falhas de calibração e identificabilidade;
4. problemas com sentidos novos ou históricos ausentes do inventário;
5. escolhas que parecem escaláveis computacionalmente, mas não
   metodologicamente;
6. uma alternativa mais simples que preserve a mesma reivindicação.

## Objetivo científico

Temos dois corpora:

```text
D0 = 1810-1860
D1 = 1960-2010
```

Treinamos continuamente um Transformer MLM:

```text
theta_0 = treino em D0
theta_1 = continuação de theta_0 em D1
```

O benchmark atual é SemEval-2020 Task 1 em inglês lematizado, com 37
palavras-alvo. Porém, a arquitetura deve ser aplicável a outras palavras,
períodos e corpora.

O estimando desejado é:

```text
Delta_sem(w) = D(P_0(s | w), P_1(s | w))
```

onde `s` representa sentido lexical, e não qualquer cluster contextual.

## O que os experimentos anteriores demonstraram

### 1. O modelo temporal contém sinal semântico

Com campos semânticos definidos manualmente:

```text
plane_nn D0:
geometria  0,718
transporte 0,081

plane_nn D1:
geometria  0,053
transporte 0,934
```

E:

```text
chairman_nn D0:
liderança 0,923

chairman_nn D1:
liderança 0,999
```

Assim, eixos semânticos adequados distinguem a substituição de sentido de
`plane` da deriva interna do campo de `chairman`.

### 2. Clusters não supervisionados não identificam sentidos

Clusters de ocorrências com perfis relacionais produziram:

```text
Spearman = -0,089

graft_nn:    rank 1
chairman_nn: rank 4
plane_nn:    rank 11
tree_nn:     rank 23
```

O algoritmo encontra as direções de maior variância, que podem corresponder a
tópico, gênero, construção sintática, instituição ou entidades, e não a
sentido lexical.

### 3. Parte do sinal é deriva coletiva do campo

Usando controles semânticos emparelhados:

| alvo | JSD observado | mediana do campo | resíduo |
|---|---:|---:|---:|
| `chairman_nn` | 0,121 | 0,102 | 0,019 |
| `plane_nn` | 0,075 | 0,045 | 0,030 |
| `graft_nn` | 0,220 | 0,032 | 0,187 |
| `tree_nn` | 0,027 | 0,032 | -0,005 |

Isso explica boa parte do falso positivo de `chairman`, mas não recupera o
forte deslocamento esperado de `plane`.

### 4. Os checkpoints temporais não fornecem uma régua fixa

A mesma ocorrência histórica:

```text
plate figure represent an inclined plane
```

foi projetada nos mesmos campos lexicais como:

```text
theta_0:
geometria  0,841
transporte 0,008

theta_1:
geometria  0,043
transporte 0,951
```

No conjunto de D0, 218 ocorrências passaram de argmax geométrico sob `theta_0`
para argmax transporte sob `theta_1`.

Inversamente, `theta_0` interpretou exemplos modernos inequívocos de aviação
como geometria:

```text
take the eight o'clock plane out of Charles de Gaulle airport

theta_0:
geometria  0,976
transporte 0,006
```

Portanto:

```text
theta_0 não representa adequadamente alguns sentidos modernos;
theta_1 reinterpreta alguns sentidos históricos pela geometria moderna.
```

Comparar medidas produzidas por esses checkpoints mistura mudança lexical e
mudança da própria régua.

### 5. Hidden states de MLM não formam naturalmente um espaço gloss-contexto

Testamos um atlas de três sentidos de `plane`:

```text
plane#geometry
plane#aircraft
plane#tool
```

Codificamos glosses e ocorrências com o mesmo checkpoint e usamos similaridade
entre seus hidden states.

Resultados em subconjuntos heurísticos:

| checkpoint | conjunto | acurácia |
|---|---|---:|
| `theta_0` | D0 geometria | 23,6% |
| `theta_0` | D0 ferramenta | 10,5% |
| `theta_0` | D1 avião | 63,9% |
| `theta_1` | D0 geometria | 12,1% |
| `theta_1` | D0 ferramenta | 15,8% |
| `theta_1` | D1 avião | 79,8% |

Os protótipos apresentaram correlações de aproximadamente `0,73-0,90`.
Portanto, não basta adicionar definições a um MLM genérico. Contextos e
definições precisam ser treinados para compartilhar um espaço de WSD.

## Diagnóstico atual: quatro paredes distintas

### Parede A: identificabilidade

```text
P_t(c | w) =
sum_s P_t(s | w) P_t(c | w,s)
```

Observar mudança em `P_t(c|w)` não identifica qual fator mudou:

```text
P_t(s|w)       distribuição dos sentidos
P_t(c|w,s)     realização contextual de um mesmo sentido
```

Esse é o problema dominante de `chairman`.

### Parede B: estimação

Mesmo quando existe substituição forte de sentido, KMeans pode preferir
partições por domínio ou registro porque minimiza inércia, e não erro de WSD.

Esse é um problema central de `plane`.

### Parede C: régua temporal móvel

`theta_0` e `theta_1` não são apenas observadores diferentes do mesmo objeto.
Eles implementam espaços semânticos diferentes. Um sentido pode aparecer,
desaparecer ou dominar o próprio codificador.

### Parede D: compatibilidade entre definições e ocorrências

Um hidden state de MLM não é necessariamente comparável a uma representação de
gloss. Precisamos de um modelo treinado explicitamente para aproximar:

```text
contexto de uma ocorrência <-> definição do sentido correto
```

## Proposta: atlas lexical fixo com WSD probabilístico e open set

A arquitetura proposta é:

```text
inventário lexical fixo
        +
codificador externo de WSD congelado
        |
        v
P(sense | ocorrência)
        |
        v
distribuição temporal de sentidos
        |
        v
distância entre períodos + incerteza
```

Os checkpoints `theta_0` e `theta_1` deixam de definir a régua principal de
sentido. Eles podem permanecer como objeto de análise auxiliar, por exemplo,
para estudar plasticidade e forgetting, mas não classificam os sentidos usados
no score temporal principal.

## Componente 1: inventário de sentidos

Para cada lema e POS `w`, obtemos automaticamente:

```text
S(w) = {s_1, ..., s_K}
```

de um inventário pré-especificado, inicialmente WordNet.

Cada sentido contém:

```text
synset
gloss
exemplos
sinônimos
relações hierárquicas
```

Não criaremos campos específicos depois de observar o gold. A mesma rotina de
consulta e preparação será usada para todas as palavras.

O inventário precisa permitir dois níveis:

```text
nível fino: synsets
nível agregado: supersenses ou ancestrais semânticos
```

Isso é importante porque a granularidade fina da WordNet pode separar
distinções irrelevantes para mudança lexical temporal.

## Componente 2: codificador externo de WSD

Usaremos um modelo externo e congelado com codificação compatível de contexto e
gloss. A primeira família candidata é um bi-encoder informado por glosses,
treinado explicitamente para WSD:

```text
e_c(w,o) = E_context(contexto da ocorrência, posição de w)
e_g(s)   = E_gloss(gloss, exemplos e lema do sentido s)
```

O ponto arquitetural não depende de uma implementação específica. Precisamos
comparar pelo menos:

```text
bi-encoder gloss-contexto:
rápido, glosses pré-computáveis, adequado para grande escala

cross-encoder ou reranker:
mais caro, mas pode melhorar candidatos ambíguos

modelo contextual de WSD:
pode usar sentidos atribuídos às palavras vizinhas
```

Uma implementação do tipo BEM é candidata natural ao primeiro teste porque seus
encoders de contexto e gloss são treinados conjuntamente. ConSeC pode funcionar
como controle mais contextual, porém é computacionalmente mais caro e menos
direto para produzir probabilidades independentes por ocorrência.

Referências primárias:

```text
BEM:
https://aclanthology.org/2020.acl-main.95/

ConSeC:
https://aclanthology.org/2021.emnlp-main.112/

WordNet:
https://wordnet.princeton.edu/
```

## Componente 3: posterior probabilístico de sentidos

Para uma ocorrência `o` de `w`, calculamos:

```text
z_s(w,o) =
cos(e_c(w,o), e_g(s)) / tau
```

para todo `s in S(w)`.

No baseline:

```text
q(s | w,o) =
exp(z_s) / sum_{r in S(w)} exp(z_r)
```

`tau` não será escolhido pelo SemEval. Será calibrado uma única vez em um
conjunto externo de WSD, ou fixado pelo modelo pré-treinado.

Como palavras possuem números muito diferentes de sentidos candidatos, a
calibração deve ser auditada por:

```text
|S(w)|
POS
frequência do lema
frequência lexical dos sentidos
comprimento e qualidade do contexto
```

Caso contrário, palavras altamente polissêmicas podem receber mais entropia e
mais `UNK` apenas por terem um inventário maior.

Não queremos usar apenas `argmax`, pois:

1. ocorrências podem ser ambíguas;
2. sentidos próximos devem compartilhar incerteza;
3. pequenas diferenças de logit não devem virar contagens rígidas;
4. a agregação temporal precisa propagar a confiança do classificador.

## Componente 4: sentido desconhecido

Forçar toda ocorrência para um synset conhecido produz falsa certeza,
especialmente para:

```text
sentidos históricos ausentes da WordNet;
sentidos realmente novos;
usos técnicos;
erros de lematização;
contextos insuficientes.
```

Adicionaremos uma classe:

```text
UNK_w
```

O score de evidência conhecida será:

```text
m(w,o) = max_{s in S(w)} z_s(w,o)
```

Uma versão simples do posterior aberto é:

```text
z_UNK(w,o) = (gamma_w - m(w,o)) / tau_UNK

q(y | w,o) =
softmax({z_s : s in S(w)} union {z_UNK})
```

`gamma_w` não pode ser ajustado pelo gold temporal. Deve ser calibrado por
dados externos, por validação leave-one-sense-out ou por um procedimento
conformal.

Uma alternativa metodologicamente mais limpa é abstention:

```text
se confiança(w,o) < gamma:
    y = UNK
senão:
    preservar q(s | w,o)
```

Queremos que a revisão diga qual formulação é mais defensável e calibrável.

## Componente 5: agregação temporal

Para o período `t` com ocorrências `O_t(w)`:

```text
P_t(s | w) =
1 / |O_t(w)| sum_{o in O_t(w)} q(s | w,o)
```

incluindo `UNK`.

Isso estima a massa esperada de cada sentido, em vez de contar apenas rótulos
duros.

O score primário mínimo é:

```text
M(w) = 1/2 (P_0 + P_1)

JSD(P_0, P_1) =
1/2 KL(P_0 || M) + 1/2 KL(P_1 || M)

Delta_fine(w) =
sqrt(JSD(P_0(.|w), P_1(.|w)))
```

Usaremos logaritmo de base 2. Assim, a raiz da JSD é uma métrica simétrica no
intervalo `[0,1]`.

### Correção do piso amostral

Mesmo sob ausência de mudança, amostras finitas e erros do classificador podem
produzir distância positiva. Para cada palavra, agrupamos as ocorrências dos
dois períodos, permutamos os rótulos temporais preservando `N_0` e `N_1`, e
recalculamos a distância:

```text
Delta_null^(b)(w), b = 1,...,B
```

Definimos:

```text
Delta_adj(w) =
max(0, Delta_obs(w) - median_b Delta_null^(b)(w))
```

e também reportamos:

```text
Z_perm(w) =
(Delta_obs(w) - mean_b Delta_null^(b)(w))
/ sd_b Delta_null^(b)(w)
```

`Delta_adj` é o score principal candidato; `Z_perm` é diagnóstico, pois pode
ficar instável quando a variância nula é muito pequena. A permutação corrige
somente o piso amostral. Ela não remove mudança de gênero, domínio ou qualidade
do corpus.

## Componente 6: comparação hierárquica

A WordNet pode ser fina demais. Duas distribuições podem trocar massa entre
synsets quase equivalentes e gerar uma JSD alta sem mudança relevante.

Se `A_l` agrega synsets no nível hierárquico `l`, definimos:

```text
P_t^(l) = A_l P_t
```

e:

```text
Delta_hier(w) =
sum_{l=1}^L alpha_l
sqrt(JSD(P_0^(l), P_1^(l)))

alpha_l >= 0
sum_l alpha_l = 1
```

Para evitar graus de liberdade excessivos, o primeiro experimento deve usar
somente:

```text
Delta_fine
Delta_coarse em um único nível predefinido
```

sem aprender `alpha_l` no SemEval.

Uma alternativa é optimal transport entre synsets:

```text
Delta_OT(w) =
min_pi sum_{i,j} pi_ij d_sem(s_i,s_j)
```

sujeito às marginais `P_0` e `P_1`, com `d_sem` derivada da hierarquia lexical.

Nossa preferência inicial é JSD fina + JSD agregada, porque OT introduz escolhas
adicionais de custo e regularização.

## Componente 7: novidade temporal

`UNK` mistura pelo menos três fenômenos:

```text
sentido ausente do inventário;
contexto pouco informativo;
falha de domínio do codificador.
```

Não podemos chamar toda massa `UNK` de inovação.

Definimos apenas um score de novidade candidata:

```text
Novelty(w) =
max(0, P_1(UNK|w) - P_0(UNK|w))
```

Ele só será interpretado como possível sentido novo se:

1. a massa `UNK` de D1 for estável por bootstrap;
2. essas ocorrências formarem estrutura reproduzível;
3. o aumento não ocorrer amplamente em palavras-controle;
4. uma auditoria lexical mostrar coerência interna.

Assim, a arquitetura separa:

```text
mudança entre sentidos conhecidos;
aumento de massa não explicada;
incerteza geral do codificador.
```

## Componente 8: incerteza e controles

Para cada palavra:

1. bootstrap estratificado de ocorrências em D0 e D1;
2. intervalo para `Delta_fine`, `Delta_coarse` e `Novelty`;
3. baseline nula por permutação temporal;
4. sensibilidade ao tamanho da amostra;
5. taxa média de `UNK`;
6. entropia média dos posteriors;
7. desempenho por número de sentidos candidatos;
8. estabilidade sob pequenas variações de contexto;
9. resultados separados para texto original e lematizado, se disponível.

Uma palavra não receberá score semanticamente interpretável quando:

```text
cobertura do inventário for insuficiente;
massa UNK for excessiva nos dois períodos;
posterior for quase uniforme;
intervalo bootstrap for amplo demais.
```

O sistema deve poder dizer “não sei”, em vez de sempre produzir um ranking.

## Exemplos esperados, sem usá-los para ajustar o método

Os números abaixo são ilustrativos, não resultados.

### `plane_nn`

Inventário mínimo automático esperado:

```text
surface/geometry
aircraft
woodworking tool
outros synsets da WordNet
UNK
```

Exemplo de agregação:

```text
D0:
geometry 0,66
aircraft 0,04
tool     0,24
other    0,03
UNK      0,03

D1:
geometry 0,10
aircraft 0,80
tool     0,03
other    0,03
UNK      0,04
```

Nesse caso, aviação comercial, militar e notícias podem variar muito no espaço
contextual, mas transferem massa para o mesmo sentido lexical de aeronave. Isso
resolve o problema de KMeans perseguir a maior variância interna.

### `chairman_nn`

```text
D0:
person presiding over an organization 0,92
other known senses                  0,04
UNK                                 0,04

D1:
person presiding over an organization 0,94
other known senses                  0,03
UNK                                 0,03
```

Mudanças entre parlamento, empresa, comissão e associações podem alterar o
contexto, mas não precisam alterar a distribuição de sentidos.

Esse é o teste direto contra a parede de identificabilidade.

### `graft_nn`

```text
D0:
plant graft       0,79
corrupt gain      0,10
medical tissue    0,05
UNK               0,06

D1:
plant graft       0,23
corrupt gain      0,38
medical tissue    0,31
UNK               0,08
```

A arquitetura deve preservar tanto substituição quanto diversificação.

### `tree_nn`

```text
D0:
woody plant 0,89
diagram     0,04
other/UNK   0,07

D1:
woody plant 0,86
diagram     0,06
other/UNK   0,08
```

Mudanças de tópico botânico ou composição do corpus não deveriam gerar grande
transferência entre sentidos.

## Escalabilidade arquitetural

### Escalabilidade computacional

No bi-encoder:

```text
glosses são codificadas uma única vez;
ocorrências são codificadas em batches;
somente sentidos do mesmo lema e POS são candidatos;
custos crescem aproximadamente com ocorrências e candidatos locais.
```

Um cross-encoder pode ser reservado para:

```text
posteriors ambíguos;
baixa margem;
alta massa UNK;
subconjunto auditado.
```

### Escalabilidade metodológica

O método é universal somente se:

1. inventário, encoder e calibração forem fixados antes do SemEval;
2. nenhum gloss for criado depois de observar erros individuais;
3. as mesmas regras de candidato, agregação e abstention valerem para todas as
   palavras;
4. resultados sem cobertura forem marcados como inconclusivos;
5. hiperparâmetros não forem escolhidos pela correlação com as 37 palavras.

### Limite estrutural

O método não descobre livremente todo sentido possível. Ele mede mudança em
relação a um atlas lexical externo, com uma saída residual para o que o atlas
não explica.

A reivindicação correta seria:

> Medição temporal de distribuições de sentidos lexicalmente ancorados sob uma
> régua externa fixa, probabilística, auditável e com abstention.

Não seria:

> Descoberta não supervisionada completa da evolução dos sentidos.

## Experimento mínimo em três portas

Não queremos implementar imediatamente o pipeline completo para as 37 palavras.
Propomos três portas de viabilidade.

### Porta 1: compatibilidade WSD

Usar um modelo externo de WSD sem ajuste no SemEval.

Avaliar em subconjuntos heurísticos predefinidos de `plane`:

```text
D0 geometry: 182 ocorrências
D0 tool:      19 ocorrências
D1 aircraft: 208 ocorrências
```

Os rótulos heurísticos não treinam o modelo. Servem apenas para testar se a
régua externa lê o corpus lematizado.

Critério inicial:

```text
accuracy macro claramente acima do baseline por sentido;
D0 geometry >= 0,75;
D1 aircraft >= 0,80;
D0 tool deve superar substantivamente o acaso, com intervalo reportado;
theta temporal não participa da decisão.
```

Como `D0 tool` tem apenas 19 exemplos, não exigiremos `0,70` como corte rígido
sem intervalo de confiança.

Também testaremos a mesma ocorrência histórica sob uma única régua:

```text
plate figure represent an inclined plane
```

Ela deve permanecer geométrica; não há mais um segundo checkpoint para
reinterpretá-la.

### Porta 2: contrato semântico dos quatro casos

Sem mudar encoder, inventário ou calibração:

```text
plane:
grande transferência para aircraft

chairman:
baixa mudança entre sentidos

graft:
mudança alta e diversificação

tree:
mudança baixa
```

Critério ordinal predefinido:

```text
Delta(plane) > Delta(chairman)
Delta(graft) > Delta(chairman)
Delta(graft) > Delta(tree)
Delta(plane) > Delta(tree)
```

Esse teste não valida o método global, mas detecta rapidamente uma arquitetura
incapaz de atravessar as paredes conhecidas.

### Porta 3: avaliação integral

Somente depois das duas primeiras portas:

1. gerar candidatos WordNet para todas as 37 palavras;
2. processar todas as ocorrências sem ajustes por alvo;
3. calcular `Delta_fine`, `Delta_coarse`, `Delta_adj` e `Novelty`;
4. avaliar Spearman graded, ROC-AUC e AP;
5. bootstrap por palavra e do Spearman;
6. auditar cobertura, massa `UNK`, POS, frequência e `|S(w)|`;
7. comparar com APD relacional `0,210` e APD balanceado `0,212`.

## Ablations obrigatórias

```text
1. hard argmax versus posterior soft;
2. synset fino versus nível agregado;
3. sem UNK versus com abstention;
4. contexto lematizado versus reconstrução aproximada ou corpus original;
5. bi-encoder versus reranking apenas nos casos ambíguos;
6. prior uniforme versus prior lexical externo;
7. score bruto versus score condicionado à cobertura.
8. score bruto versus correção pelo null de permutação.
```

O prior merece cuidado:

```text
q(s|w,o) proporcional a likelihood(contexto|s) * prior(s|w)
```

Um prior moderno de frequência pode apagar sentidos históricos raros. Nossa
preferência inicial é prior uniforme no score principal e prior lexical apenas
como ablation.

## Riscos que reconhecemos

1. WordNet moderna pode não cobrir sentidos históricos.
2. A granularidade de synsets pode não coincidir com o gold do SemEval.
3. O encoder externo pode falhar em inglês histórico lematizado.
4. Treinamento moderno de WSD pode favorecer sentidos e registros recentes.
5. `UNK` pode capturar erro de domínio, não novidade lexical.
6. Definições lexicográficas introduzem supervisão explícita e alteram a
   contribuição científica.
7. Um modelo WSD excelente em benchmarks modernos pode ser mal calibrado no
   domínio histórico.
8. A agregação média pode ocultar dependências entre sentido e gênero textual.
9. O inventário pode conter sentidos ausentes dos dois corpora e diluir massa.
10. O benchmark com 37 palavras é pequeno para selecionar arquitetura ou
    hiperparâmetros.

## Decisões para as quais queremos contribuição

1. A arquitetura realmente separa as quatro paredes ou apenas transfere o
   problema para um recurso externo?
2. Um bi-encoder gloss-contexto é o componente principal correto, ou um
   cross-encoder/contextual WSD seria necessário desde o início?
3. WordNet é o inventário mais defensável, ou devemos combinar WordNet,
   Wiktionary histórico e um nível de supersenses?
4. Como calibrar `UNK` sem usar gold temporal e sem confundir domínio histórico
   com sentido novo?
5. JSD fina + JSD agregada é suficiente, ou OT hierárquico é conceitualmente
   superior?
6. Devemos usar prior uniforme ou alguma frequência lexical externa?
7. Como tratar sentidos da WordNet que não existiam historicamente sem inserir
   informação futura de modo problemático?
8. O texto lematizado inviabiliza WSD externo a ponto de exigir reconstrução ou
   retorno ao corpus original?
9. As três portas de viabilidade são suficientemente severas?
10. Qual resultado obrigaria a abandonar essa arquitetura?

## Critérios objetivos para manter a direção

Continuar se:

```text
1. a régua externa superar claramente nossos checkpoints no teste gloss-contexto;
2. plane e graft superarem chairman e tree sem ajustes por palavra;
3. chairman deixar de ser falso positivo dominante;
4. cobertura e UNK forem aceitáveis e estáveis nos dois períodos;
5. o resultado global superar APD=0,212 com intervalo ou ganho qualitativo
   consistente;
6. o resultado não depender de um único prior, temperatura ou nível WordNet.
```

## Critérios objetivos para abandonar ou reformular

Abandonar a reivindicação de mudança de sentido lexical se:

```text
1. nenhum encoder externo ler de forma confiável os contextos históricos;
2. plane continuar abaixo de chairman após o atlas fixo;
3. a maior parte do sinal migrar para UNK sem interpretação;
4. mudanças pequenas de inventário ou granularidade inverterem o ranking;
5. a melhora global depender de tuning nas 37 palavras;
6. cobertura histórica exigir intervenção manual alvo por alvo.
```

Nesse caso, a contribuição honesta permanece:

```text
diagnóstico de mudança relacional/contextual;
decomposição de nuisance por campo;
evidência de régua semântica móvel em treinamento contínuo;
limites de identificabilidade de clustering não supervisionado.
```

## Formato solicitado

Por favor, responda com:

1. diagnóstico principal em uma frase;
2. falhas da arquitetura, ordenadas por severidade;
3. avaliação de escalabilidade computacional e metodológica;
4. correções precisas das fórmulas;
5. escolha recomendada de inventário e família de encoder;
6. desenho revisado das três portas;
7. critérios de sucesso e falsificação;
8. reivindicação publicável mais forte que os dados permitiriam.

## Arquivos relevantes

```text
outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/plane_gloss_atlas/report.md
outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/plane_checkpoint_semantic_audit/report.md
outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/paired_field_controls/report.md
outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/contextual_usage_clusters_relational/report.md
outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/seed_community_profiles/report.md
scripts/evaluate_plane_gloss_atlas.py
scripts/audit_plane_checkpoint_semantics.py
scripts/diagnose_paired_field_controls.py
scripts/evaluate_contextual_usage_clusters.py
docs/relational_profile_formalization.md
```
