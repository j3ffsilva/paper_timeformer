# Pedido de segunda opinião: clusters temporais não são necessariamente sentidos

Queremos uma avaliação independente e crítica do ponto a que chegamos na
medição de mudança semântica temporal. Implementamos a proposta anterior de
passar de comunidades de tokens para comunidades de ocorrências
contextualizadas. O resultado tornou nossa parede mais precisa, mas não a
removeu.

Não tente apenas melhorar nossas métricas atuais. Questione a identificabilidade
do problema e diga se a direção ainda é defensável.

## Objetivo

Treinamos continuamente um Transformer MLM:

```text
theta_0 = treino em D0 (1810-1860)
theta_1 = continuação de theta_0 em D1 (1960-2010)
```

O benchmark é o SemEval-2020 Task 1 em inglês lematizado, com 37 palavras-alvo.
Queremos estimar:

```text
Delta_sem(w) = D(P_0(z | w), P_1(z | w))
```

onde `z` deveria representar sentido ou tipo semântico de uso, e não apenas
qualquer variação contextual.

## Correções e resultados anteriores

Antes desta etapa, corrigimos problemas reais do treinamento:

1. impedimos janelas de atravessarem fronteiras documentais;
2. substituímos mascaramento central determinístico por MLM dinâmico 15%
   com política 80/10/10;
3. passamos a incluir as caudas dos documentos;
4. treinamos um modelo mais robusto, `d_model=128`, 3 camadas, 12 épocas em D0
   e 8 épocas em D1.

Depois dessas correções, o modelo passou a produzir estruturas semanticamente
coerentes. Por exemplo:

```text
plane_nn em D0:
line, angle, plate, column, canal, coast

plane_nn em D1:
boat, ship, rail, route, engine, machine
```

Mas os métodos de medição continuaram falhando:

```text
Cloze PMI Spearman             -0,070
PPMI-JSD                         0,042
APD relacional                   0,210
APD balanceado                   0,212
```

No APD relacional:

```text
plane_nn:    rank 35, gold graded 0,882, maior mudança do dataset
chairman_nn: rank 1,  gold graded 0,000, estável
graft_nn:    rank 9,  gold graded 0,554
```

O diagnóstico foi que APD mede mudança de relações individuais e dispersão,
não necessariamente transferência entre sentidos.

## Teste com comunidades semânticas manuais

Como prova de geometria, definimos seis campos por pequenas listas de palavras:

```text
geometria
transporte
liderança
botânica
medicina
corrupção
```

Cada campo foi representado localmente em cada checkpoint para não comparar
diretamente espaços rotacionados.

Com temperatura `0,05`:

```text
plane_nn:    rank 1
graft_nn:    rank 2
tree_nn:     rank 26
chairman_nn: rank 32
Spearman:    0,184
```

Exemplos das massas:

```text
plane_nn D0:
geometria 0,718
transporte 0,081

plane_nn D1:
geometria 0,053
transporte 0,934
```

```text
chairman_nn D0:
liderança 0,923

chairman_nn D1:
liderança 0,999
```

Esse teste mostra que o modelo contém sinal capaz de separar `plane` de
`chairman` quando fornecemos as dimensões semânticas corretas. Porém:

1. os campos foram definidos manualmente;
2. o resultado global continuou modesto;
3. houve sensibilidade à temperatura;
4. isso não é ainda um método não supervisionado de descoberta de sentidos.

## Tentativa 1: comunidades não supervisionadas de tokens

Usamos 3.216 tokens compartilhados, com frequência mínima de 100 em cada
período. Construímos grafos kNN centrados e detectamos comunidades.

O mutual-kNN gerou de 23% a 72% de comunidades singleton. Trocamos para
union-kNN.

A configuração escolhida exclusivamente por estrutura foi:

```text
k = 40
resolução = 1,0
11 comunidades
AMI médio = 0,917
nenhum singleton
maior comunidade = 16,6% dos tokens
```

Apesar da estabilidade estrutural, uma comunidade ampla de 535 tokens misturou
geometria, transporte, botânica e outros campos físicos. O melhor Spearman
selecionado ficou em aproximadamente `0,200`.

Resultado qualitativo típico:

```text
plane_nn:    aproximadamente rank 15-17
chairman_nn: aproximadamente rank 6-8
graft_nn:    alto
tree_nn:     baixo
```

Agregar 11 partições estáveis em múltiplas resoluções não resolveu:

```text
melhor Spearman aproximadamente 0,168
plane continuou aproximadamente 18-23
chairman aproximadamente 7-10
```

Conclusão provisória: um token não deve pertencer a uma única comunidade
global, porque tokens são polissêmicos e os campos são hierárquicos e
sobrepostos.

## Nova hipótese testada: comunidades de ocorrências

Passamos a modelar cada ocorrência contextualizada de uma palavra:

```text
h_t(w, o)
```

Para cada palavra e checkpoint congelado:

1. juntamos ocorrências de D0 e D1;
2. balanceamos os períodos;
3. ocultamos o rótulo temporal do algoritmo;
4. agrupamos as ocorrências da própria palavra;
5. calculamos a distribuição de clusters em cada período;
6. usamos:

```text
Delta_usage(w) =
JSD(P(cluster | w, D0), P(cluster | w, D1))
```

Fizemos os agrupamentos separadamente sob `theta_0` congelado e `theta_1`
congelado. Portanto, não há comparação direta entre coordenadas de checkpoints.
O score final é a média dos dois controles congelados.

Para evitar escolher uma configuração conveniente após olhar o gold:

```text
K = {2, 3, 4, 5, 6}
seeds = {0, 1, 2, 3, 4}
máximo de 500 ocorrências por período
score por checkpoint = mediana sobre K e seeds
score final = média entre theta_0 e theta_1 congelados
```

## Representação A: vetores contextuais crus

Usamos os vetores da camada 2.

```text
Spearman = -0,155
ROC-AUC  =  0,497
AP       =  0,492
```

Auditoria:

| alvo | JSD | theta0 congelado | theta1 congelado | rank |
|---|---:|---:|---:|---:|
| graft_nn | 0,243 | 0,242 | 0,244 | 1 |
| chairman_nn | 0,168 | 0,170 | 0,165 | 2 |
| plane_nn | 0,024 | 0,030 | 0,018 | 25 |
| tree_nn | 0,004 | 0,006 | 0,003 | 36 |

Há forte concordância entre os checkpoints. Portanto, o falso positivo de
`chairman` não parece vir de rotação ou forgetting entre `theta_0` e `theta_1`.
Os dois modelos leem uma separação temporal forte em seus contextos.

Com apenas `K=2`, `chairman` cai para rank 33, mas `plane` continua somente em
rank 22. Com granularidade maior, `chairman` volta ao topo. Isso sugere
fragmentação de um mesmo campo semântico por construções, tópicos ou registros.

## Representação B: perfis relacionais das ocorrências

Para reduzir a dependência das coordenadas ocultas, representamos cada
ocorrência por sua afinidade com os 3.216 tokens-âncora compartilhados, dentro
do mesmo checkpoint:

```text
r_theta(w,o)[v] =
cos(h_theta(w,o), centroid_theta(v))
```

Usamos centroides das âncoras combinando D0 e D1 dentro de cada checkpoint,
centralizamos os perfis e comprimimos para 32 dimensões por PCA. PCA, KMeans e
balanceamento não receberam rótulo temporal nem gold.

Resultados:

```text
Spearman = -0,089
ROC-AUC  =  0,503
AP       =  0,553
```

Auditoria:

| alvo | JSD | theta0 congelado | theta1 congelado | desacordo | rank |
|---|---:|---:|---:|---:|---:|
| graft_nn | 0,220 | 0,227 | 0,212 | 0,015 | 1 |
| chairman_nn | 0,121 | 0,126 | 0,117 | 0,009 | 4 |
| plane_nn | 0,075 | 0,035 | 0,115 | 0,080 | 11 |
| tree_nn | 0,027 | 0,028 | 0,027 | 0,001 | 23 |

Por número de clusters:

| K | Spearman | plane rank | chairman rank | graft rank | tree rank |
|---:|---:|---:|---:|---:|---:|
| 2 | -0,222 | 22 | 4 | 1 | 30 |
| 3 | -0,261 | 21 | 4 | 1 | 19 |
| 4 | -0,042 | 12 | 4 | 1 | 24 |
| 5 | -0,024 | 11 | 2 | 1 | 25 |
| 6 | -0,036 | 10 | 3 | 1 | 25 |

O perfil relacional ajuda `plane`, mas não resolve `chairman`, e o resultado
global continua sem correlação útil com o gold.

## Exemplos concretos da parede

### `graft_nn`

Gold:

```text
binary = 1
graded = 0,554
```

Historicamente esperamos mudança de enxertia botânica para usos adicionais de
corrupção e medicina. Todos os métodos comunitários ou de ocorrências colocam
`graft` alto. Aqui, separação temporal e mudança semântica coincidem.

### `chairman_nn`

Gold:

```text
binary = 0
graded = 0,000
```

Os vizinhos em ambos os períodos continuam no campo de liderança:

```text
D0:
secretary, editor, commander, director, president, committee, jury

D1:
secretary, director, commander, president, commissioner, governor, publisher
```

Mesmo assim, os clusters de ocorrências separam fortemente D0 e D1 sob ambos os
checkpoints. A hipótese é que estejam capturando mudança de:

```text
tipo de instituição
registro e gênero textual
nomes e cargos associados
construção sintática
concentração colocacional
composição dos corpora
```

Tudo isso é variação temporal contextual real, mas não necessariamente mudança
do significado lexical de `chairman`.

### `plane_nn`

Gold:

```text
binary = 1
graded = 0,882
```

Há uma troca lexicalmente interpretável:

```text
D0: line, angle, plate, column
D1: boat, ship, rail, route, engine, machine
```

Campos manuais capturam a transferência:

```text
geometria 0,718 -> 0,053
transporte 0,081 -> 0,934
```

Mas clusters de ocorrências crus colocam `plane` em 25º; perfis relacionais
melhoram para 11º, com forte desacordo entre checkpoints:

```text
theta0 congelado: JSD 0,035
theta1 congelado: JSD 0,115
```

Portanto, uma mudança semântica clara não é necessariamente a principal direção
de variância encontrada pelo clustering.

### `tree_nn`

Gold:

```text
binary = 0
graded = 0,071
```

Nos vetores crus fica em 36º, como desejado. No perfil relacional sobe para
23º, mas continua muito abaixo de `graft` e `chairman`. Serve como controle de
uma palavra frequente e relativamente estável.

## Nossa parede atual

A formulação conceitual:

```text
Delta_sem(w) = D(P_0(z | w), P_1(z | w))
```

continua parecendo correta. O problema é identificar `z`.

O experimento mostra:

```text
P_t(cluster | w) != necessariamente P_t(sense | w)
```

Clusters não supervisionados de ocorrências encontram variação temporal real,
mas não distinguem:

```text
mudança de sentido
mudança de tópico
mudança de gênero/registro
mudança sintática
mudança de entidades e colocações
mudança na composição amostral
```

Quando fornecemos eixos semânticos manuais, `plane` e `chairman` se comportam
como esperado. Quando exigimos que os eixos sejam descobertos sem supervisão,
o método privilegia fatores temporalmente separáveis, não necessariamente
sentidos.

Há um risco de não identificabilidade: usando apenas corpus, período e
representações de um MLM, talvez não exista critério não supervisionado capaz
de dizer se uma separação temporal é semântica ou apenas contextual. O período
é precisamente o fator cujo efeito queremos medir, mas ele também carrega todos
os nuisances históricos.

## Hipóteses para a próxima etapa

Estamos considerando, sem preferência fechada:

1. **Invariância adversarial ao período**  
   Aprender uma representação que preserve substituibilidade lexical, mas
   remova informação temporal facilmente previsível antes do clustering.
   Risco: remover justamente o sinal de mudança semântica.

2. **Contraste contextual controlado**  
   Comparar ocorrências pareadas ou reponderadas por gênero, sintaxe, tópico e
   colocações, estimando mudança condicional em vez de marginal.
   Risco: os metadados podem não existir e o controle pode apagar sentidos
   ligados a novos contextos.

3. **Representação por substitutos semanticamente filtrados**  
   Para cada ocorrência, usar uma distribuição de substitutos do MLM, mas
   remover function words e agrupar substitutos por relações paradigmáticas ou
   ontológicas.
   Risco: já observamos que cloze puro é fortemente sintático.

4. **Weak supervision ou âncoras semânticas automáticas**  
   Induzir eixos com recursos externos, glosses, embeddings previamente
   treinados ou LLMs, mantendo a avaliação temporal não supervisionada.
   Risco: muda a reivindicação científica e pode introduzir conhecimento
   histórico futuro.

5. **Modelagem causal/fatorial explícita**  
   Decompor a representação em fatores de identidade lexical, sentido, tópico,
   sintaxe e período.
   Risco: identificabilidade continua sem alguma supervisão ou hipótese forte.

6. **Aceitar outro estimando**  
   Reformular o artigo para medir mudança relacional/contextual temporal, sem
   afirmar equivalência direta com mudança de sentido lexical.

## Perguntas para sua avaliação

Por favor, responda com crítica, não com uma lista genérica de métodos.

1. Nossa leitura de que chegamos a um problema de identificabilidade é correta?
2. Os resultados de `chairman` provam um nuisance contextual ou ainda podem
   indicar um erro específico na representação/agregação?
3. Como explicar simultaneamente:

```text
campos manuais:
plane rank 1, chairman rank 32

clusters de ocorrências relacionais:
plane rank 11, chairman rank 4
```

4. Qual propriedade operacional poderia fazer um cluster representar sentido,
   e não apenas uma partição estável de contextos?
5. Existe um experimento negativo/controle que identifique quais nuisances
   explicam `chairman`?
6. Devemos tentar remover informação de período antes de medir mudança, ou isso
   é conceitualmente contraditório?
7. Qual das seis hipóteses acima é a próxima experiência mais informativa e
   falsificável?
8. É possível manter uma contribuição não supervisionada forte, ou alguma forma
   de weak supervision passou a ser necessária?
9. Há uma formulação melhor que `D(P_0(z|w),P_1(z|w))` para separar mudança de
   sentido de mudança do ambiente contextual?
10. Em que ponto devemos abandonar a meta SemEval e assumir explicitamente que
    nosso estimando é mudança relacional/contextual?

## Formato solicitado

Responda com:

1. diagnóstico principal em uma frase;
2. falhas conceituais ou estatísticas, ordenadas por severidade;
3. interpretação dos quatro exemplos concretos;
4. próximo experimento mínimo, com formulação matemática;
5. resultado que manteria a direção;
6. resultado que a falsificaria;
7. recomendação honesta sobre a reivindicação publicável.

## Arquivos relevantes

```text
outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/contextual_usage_clusters_relational/report.md
outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/contextual_usage_clusters_relational/summary.json
outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/contextual_usage_clusters_relational/runs.csv
outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/contextual_usage_clusters/summary.json
outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/seed_community_profiles/report.md
outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/unsupervised_community_profiles_union/report.md
outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/multiresolution_community_profiles/report.md
outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/hidden_relational_profiles/report.md
scripts/evaluate_contextual_usage_clusters.py
scripts/evaluate_seed_community_profiles.py
scripts/evaluate_unsupervised_community_profiles.py
scripts/evaluate_multiresolution_community_profiles.py
docs/relational_profile_formalization.md
```
