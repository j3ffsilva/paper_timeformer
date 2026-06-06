# Pedido de segunda opinião: resultados e parede atual

Queremos uma avaliação independente e crítica do estado atual do experimento de
mudança semântica temporal. Não presuma que nossa interpretação esteja correta.
Procure erros conceituais, estatísticos e de implementação e proponha o próximo
teste com maior poder de discriminação.

## Objetivo científico

Treinamos continuamente o mesmo Transformer em ordem cronológica:

```text
theta_0 = treino em D0 (1810-1860)
theta_1 = continuação de theta_0 em D1 (1960-2010)
```

Queremos medir mudança semântica como mudança do perfil de relações de uma
palavra com o restante do vocabulário dentro de cada checkpoint, evitando
comparar diretamente coordenadas ocultas entre checkpoints.

Dataset: SemEval-2020 Task 1, inglês lematizado, 37 palavras-alvo.

## Problemas já corrigidos

### 1. Fronteiras documentais

O leitor antigo concatenava todas as linhas de cada período. Como as linhas do
SemEval são sentenças embaralhadas, isso criava janelas que atravessavam
sentenças não relacionadas. Agora cada linha é um documento separado.

### 2. Mascaramento central determinístico

O dataset antigo mascarava somente o token central de cada janela, sempre na
mesma posição. Em D1, `graft_nn` aparecia 109 vezes, mas era resposta positiva
do MLM apenas 4 vezes. Mais épocas apenas repetiam esses quatro exemplos.

Implementamos MLM dinâmico:

```text
15% dos tokens selecionados a cada época
80% -> [MASK]
10% -> token aleatório
10% -> permanece visível
```

As máscaras mudam por época, mas são reproduzíveis pela seed.

Para `graft_nn`, isso produziu:

```text
D0: 269 apresentações positivas em 12 épocas
D1: 147 apresentações positivas em 8 épocas
```

### 3. Treinamento robusto

Configuração:

```text
d_model=128
3 camadas
4 cabeças
FFN=384
12 épocas em D0
8 épocas em D1
40.188 passos de gradiente
aproximadamente 34,7 milhões de alvos mascarados
```

Perdas:

```text
D0: 6,9368 -> 4,8794
D1: 5,5362 -> 5,0236
```

O treinamento levou 1h53m49s na GPU. Portanto, poucas máscaras, baixa
capacidade e subtreino grosseiro deixaram de ser explicações satisfatórias.

## Resultado da formulação cloze-PMI

Definimos:

```text
q_t(w) = média das distribuições MLM nas ocorrências reais de w mascarada
p_t = distribuição do probe neutro [CLS] [MASK] [SEP]
R_t(w)[v] = log(q_t(w)[v] / p_t[v])
Delta = distância entre R_0(w) e R_1(w)
```

Avaliação global:

| Score | Spearman graded | ROC-AUC |
|---|---:|---:|
| PMI cosseno | -0,070 | 0,509 |
| PPMI-JSD | 0,042 | 0,518 |

Ou seja, praticamente acaso.

### Exemplo concreto: `graft_nn`

O novo mascaramento melhorou fortemente D0:

```text
antes: theta_0@D0 rank aproximadamente 2930, p=5,41e-5
agora: theta_0@D0 rank 288, p=5,89e-4
```

Top predictions em `theta_0@D0` passaram a incluir:

```text
river, water, tree_nn, ground, bank, sea, wood, land_nn, hill
```

Há coerência com o domínio botânico, embora não sejam necessariamente
substitutos diretos.

Em `theta_1@D1`, contudo:

```text
rank de graft_nn = 3316
top: and, the, be, it, of, that, have, time, in, to
```

Exemplos:

```text
government [MASK] be really old news
-> have, it, will, would, there

burn victim [MASK] of brand new skin
-> out, instead, one, because
```

Conclusão provisória: o cloze responde principalmente "qual token completa
esta posição sintaticamente?", não "quais palavras são semanticamente próximas
de w?". PMI reduz frequência, mas não converte substituibilidade posicional em
proximidade semântica ampla.

## Nova tentativa: perfis relacionais em estados ocultos

Usamos os mesmos checkpoints, sem novo treinamento.

Para cada checkpoint e corpus extraímos estados ocultos sem mascarar o alvo:

```text
theta_0@D0
theta_0@D1
theta_1@D0
theta_1@D1
```

Cada ocorrência de uma palavra é transformada num perfil de similaridades com
o mesmo vocabulário de referência dentro do próprio checkpoint:

```text
r_t(w, ocorrencia)[v] =
    cos(h_t(w, ocorrencia), centroide_t(v))
```

Isso evita comparar diretamente coordenadas de `theta_0` com coordenadas de
`theta_1`. Testamos:

1. perfil da tabela de embeddings de entrada;
2. perfil de centroides contextuais;
3. APD entre perfis de ocorrências;
4. energy distance entre as distribuições;
5. versões centralizadas para reduzir anisotropia.

Vocabulário de referência principal:

```text
3.216 tokens não-alvo
frequência mínima de 100 em cada período
```

## Resultados dos estados ocultos

| Método | Camada | Spearman | ROC-AUC |
|---|---:|---:|---:|
| APD relacional | 2 | 0,210 | 0,542 |
| APD relacional | média últimas 2 | 0,199 | 0,542 |
| APD relacional | 1 | 0,185 | 0,565 |
| APD relacional centrado | 3 | 0,133 | 0,595 |
| centroide relacional centrado | 3 | 0,125 | 0,551 |
| perfil dos embeddings de entrada | embedding | 0,101 | 0,598 |
| energy distance relacional | 2 | 0,021 | 0,473 |

Nenhum resultado é estatisticamente significativo com 37 alvos.

### Sensibilidade ao vocabulário de referência

| Frequência mínima por período | Referências | Melhor Spearman APD |
|---:|---:|---:|
| 50 | 4.867 | 0,228 |
| 100 | 3.216 | 0,210 |
| 200 | 2.013 | 0,196 |

O sinal positivo do APD é modesto, mas não depende de um único limiar.

## Exemplo relacional concreto: `graft_nn`

Na camada 2:

```text
APD relacional = 0,194
posição no ranking = 9 de 37
D0 = 119 ocorrências
D1 = 109 ocorrências
```

Após centralizar o espaço, alguns vizinhos do centroide são:

```text
theta_0@D0:
soil, stock, vine, road, plate, boundary

theta_1@D1:
cell, compound, machinery, tool, commodity, exposure
```

Isso parece capturar reorganização botânica -> técnica/médica de maneira mais
plausível que o cloze.

O controle fatorial de `graft_nn` na camada 2 foi:

```text
mudança natural theta_0@D0 -> theta_1@D1: 0,1247
efeito de corpus theta_0@D0 -> theta_0@D1: 0,0356
efeito de checkpoint theta_0@D0 -> theta_1@D0: 0,0667
```

Não tratamos essas quantidades como aditivas.

## A parede atual

### 1. APD tem sinal, mas é confundido por frequência

Na camada 2:

```text
correlação(APD, frequência total do alvo) = -0,436
correlação(gold SemEval, frequência total) = -0,113
```

Alvos raros tendem a obter APD maior. Exemplo:

```text
chairman_nn: APD 0,247, rank 1, D0=147, D1=683
graft_nn:    APD 0,194, rank 9, D0=119, D1=109
tree_nn:     APD 0,115, rank 37, D0=2322, D1=1596
```

`chairman_nn` é gold estável, mas aparece como a palavra de maior mudança.

### 2. Descontar dispersão elimina o sinal

APD mede a distância média entre usos de D0 e D1, mas inclui a dificuldade e a
heterogeneidade ordinária dos contextos.

Quando usamos energy distance com estimador intraperíodo sem diagonal:

```text
Spearman na camada 2 = 0,021
ROC-AUC = 0,473
```

Uma razão de separação temporal normalizada pela dispersão também ficou
negativa. Portanto, o APD positivo não demonstra que as distribuições temporais
se separaram além de sua variabilidade interna.

### 3. Anisotropia

Sem centralização, as maiores similaridades dos centroides estavam comprimidas
entre aproximadamente 0,87 e 0,94, com vizinhos pouco discriminativos.

Centralização melhora a interpretabilidade de `graft_nn`, mas reduz o melhor
Spearman e não resolve a frequência.

### 4. Continuação do treinamento

O score natural mistura mudança dos contextos e mudança dos pesos:

```text
theta_0@D0 -> theta_1@D1
```

A matriz fatorial diagnostica corpus e checkpoint separadamente, mas ainda não
temos uma forma justificada de remover recalibração/forgetting sem também
remover sinal semântico.

## Nossa interpretação provisória

1. O mascaramento central era de fato um defeito grave.
2. Cloze-PMI não operacionaliza bem proximidade semântica relacional.
3. Perfis de estados ocultos capturam algum sinal e exemplos qualitativamente
   plausíveis.
4. Perfis por ocorrência são melhores que centroides para o ranking graded.
5. O sinal atual é pequeno e confundido por frequência e dispersão contextual.
6. Ainda não temos um estimador que isole mudança temporal de polissemia,
   heterogeneidade contextual e deriva global do checkpoint.

## Perguntas para sua avaliação

1. Há erro conceitual ou de implementação na construção dos perfis relacionais
   por ocorrência?
2. APD relacional está medindo algo aproveitável, ou o `rho=0,21` deve ser
   tratado apenas como artefato de frequência?
3. Qual normalização preservaria mudança entre distribuições sem ser dominada
   pelo número de ocorrências ou pela dispersão interna?
4. Devemos usar amostragem balanceada, por exemplo exatamente 100 ocorrências
   por palavra e período, repetida em várias seeds?
5. Seria melhor comparar distribuições por MMD, classifier two-sample test,
   optimal transport, clustering de usos ou outra estatística?
6. Como incorporar a matriz checkpoint x corpus sem assumir aditividade das
   distâncias?
7. O desenho de checkpoints contínuos é defensável como método principal, ou
   precisamos de um encoder congelado/modelos independentes como controle?
8. Qual é o próximo experimento mínimo que distingue entre:

```text
H1: existe sinal semântico, mas o estimador atual é inadequado;
H2: os estados ocultos do modelo pequeno não representam sentidos com qualidade;
H3: continuação do treinamento domina o sinal;
H4: o benchmark de 37 palavras é pequeno demais para esta arquitetura.
```

## Arquivos para auditoria

```text
scripts/evaluate_hidden_relational_profiles.py
outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/hidden_relational_profiles/report.md
outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/hidden_relational_profiles/metrics.csv
outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/hidden_relational_profiles/scores.csv
outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/hidden_relational_profiles/factorial_diagnostics.csv
outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/cloze_diagnostics/graft_nn/report.md
docs/relational_profile_formalization.md
```

Por favor, responda com:

1. erros ou fragilidades, por severidade;
2. interpretação alternativa dos resultados;
3. próximo experimento recomendado, com fórmula precisa;
4. critérios objetivos para decidir se a abordagem deve ser mantida,
   reformulada ou abandonada.
