# Realinhamento: contribuição real do TimeFormer

Este documento resume o estado dos experimentos e redefine o alvo da contribuição
científica. Leia completo antes de propor qualquer experimento novo.

---

## O que é o TimeFormer

Um único Transformer MLM treinado continuamente na ordem cronológica dos corpora:

```
theta_0 = treino em D0 (1810-1860)
theta_1 = continuação de theta_0 em D1 (1960-2010)
```

A palavra-chave é **único e contínuo**. Não são dois modelos treinados
independentemente e depois alinhados. É um modelo que **experimentou** a
transição temporal.

Configuração atual:

```
d_model = 128
3 camadas, 4 cabeças, FFN = 384
MLM dinâmico: 15% dos tokens, 80/10/10
Corpus: SemEval-2020 Task 1, inglês lematizado
```

---

## O que foi tentado e o que falhou

### 1. Cloze-PMI (descartado)

```
q_t(w) = média das predições MLM com w mascarado
R_t(w)[v] = log(q_t(w)[v] / p_t[v])
```

Falhou porque o MLM responde "qual token completa sintaticamente esta posição",
não "quais palavras são semanticamente próximas de w". PMI não converte
substituibilidade posicional em proximidade semântica ampla.

### 2. Clustering de ocorrências (descartado como estimador principal)

Clusters de perfis relacionais por ocorrência produziram Spearman = -0.089.

Razão formal: P_t(cluster|w) ≠ P_t(sentido|w). O algoritmo encontra
direções de maior variância — tópico, registro, gênero, instituição — não
sentido lexical. Esse é o muro de identificabilidade.

### 3. Perfis relacionais com APD (resultado atual, mas incompleto)

```
r_t(w, ocorrência)[v] = cos(h_t(w, ocorrência), centroide_t(v))
APD = distância média entre r_0(w, o_i) e r_1(w, o_j)
```

Resultado: Spearman = 0.210, ROC-AUC = 0.542. Sinal positivo mas:
- Não significativo com 37 alvos
- Confundido por frequência: rho(APD, log_freq) = -0.436
- Partial Spearman controlando frequência: 0.188 (sinal preservado)
- Falso positivo: chairman_nn (rank 1, gold=estável)
- Falso negativo: plane_nn (rank 11, gold=0.882)

### 4. Controle de campo semântico (resultado parcial, promissor)

Subtraindo a mediana do campo semântico do JSD observado:

| alvo | JSD observado | mediana do campo | resíduo |
|---|---:|---:|---:|
| chairman_nn | 0.121 | 0.102 | 0.019 |
| plane_nn | 0.075 | 0.045 | 0.030 |
| graft_nn | 0.220 | 0.032 | 0.187 |
| tree_nn | 0.027 | 0.032 | -0.005 |

chairman cai de falso positivo dominante para resíduo mínimo. Mas plane
continua baixo — o JSD observado (0.075) é pequeno antes mesmo do baseline.

### 5. Atlas externo de WSD (testado e rejeitado como arquitetura principal)

Tentamos usar os próprios checkpoints para classificar ocorrências contra
glosses do WordNet. Resultado:

| checkpoint | conjunto | acurácia |
|---|---|---:|
| theta_0 | D0 geometria | 23.6% |
| theta_0 | D0 ferramenta | 10.5% |
| theta_1 | D1 avião | 63.9% |

Hidden states de MLM genérico não são compatíveis com representações de
glosses sem treinamento conjunto explícito. Essa arquitetura exigiria encoder
externo de WSD (BEM/ConSeC), o que transfere a contribuição central para um
componente de terceiros.

**Decisão: o atlas externo é uma direção de paper separada. Não é o TimeFormer.**

---

## O muro da régua móvel: achado central, não problema a esconder

A mesma ocorrência histórica:

```
plate figure represent an inclined plane
```

projetada com campos semânticos {geometria, transporte}:

```
theta_0: geometria 0.841, transporte 0.008
theta_1: geometria 0.043, transporte 0.951
```

Em D0, 218 ocorrências passaram de argmax geométrico (theta_0) para argmax
transporte (theta_1). Inversamente, theta_0 interpreta contextos modernos
inequívocos de aviação como geometria.

Isso não é um defeito a corrigir. É um **achado sobre treinamento contínuo**:
o modelo que aprendeu D0 e depois D1 incorpora a transição em seus pesos.
A mudança da régua IS o dado.

---

## Onde a literatura está: o que existe e onde estamos

### Hamilton et al. (2016) — o baseline clássico

- Word2Vec treinado separadamente por período
- Alinhamento por Procrustes ortogonal entre os dois espaços
- Embeddings **estáticos**: `gay` tem UM vetor por período, média de todos
  os contextos
- Limitação central: polissemia é borrada. `bank` próximo a rio e `bank`
  próximo a dinheiro contribuem igualmente para um único vetor

### SemEval-2020 — sistemas BERT-based

- Aplicam BERT pré-treinado **sem fine-tuning** no corpus histórico
- Extraem embeddings contextuais de cada ocorrência em D0 e D1
- Comparam distribuições via APD, clustering, etc.
- Melhor Spearman: ≈ 0.42 para inglês
- Limitação central: BERT foi treinado em inglês moderno; aplicá-lo a textos
  de 1810 é sair do domínio de treinamento. O modelo não conhece a distribuição
  léxica histórica.

### TimeFormer — o que é diferente

| propriedade | Hamilton 2016 | APD+BERT (SemEval) | TimeFormer |
|---|---|---|---|
| embeddings | estáticos | contextuais | contextuais |
| modelos | 2 independentes | 1 fixo externo | 1 contínuo |
| domínio | in-domain | out-of-domain | in-domain |
| alinhamento | Procrustes | não necessário | não necessário (perfis relacionais) |
| resolução temporal | 2 pontos | 2 pontos | N checkpoints |

A contribuição não é "usar transformers em vez de word2vec". Sistemas BERT já
fazem isso. A contribuição é:

> Um único modelo que **aprendeu a transição** — não dois snapshots alinhados
> depois. O modelo viu D0 e depois D1 em ordem, e essa sequência está nos
> pesos.

---

## O que queremos demonstrar: o alvo real

Não é maximizar o Spearman de 37 palavras.

O alvo é demonstrar que TimeFormer produz **vizinhanças semânticas temporais
coerentes** sem nenhuma anotação externa:

```
plane_nn @ D0 → vizinhos: surface, geometry, angle, line, figure, incline
plane_nn @ D1 → vizinhos: aircraft, airline, pilot, flight, airport
```

```
graft_nn @ D0 → vizinhos: soil, vine, stock, branch, root, cultivar
graft_nn @ D1 → vizinhos: cell, tissue, compound, machinery, corruption
```

```
chairman_nn @ D0 → vizinhos: committee, board, member, society, assembly
chairman_nn @ D1 → vizinhos: committee, board, member, executive, director
(vizinhos estáveis → mudança semântica baixa)
```

Isso é **caracterização temporal de sentido**, não ranking de mudança. É a
pergunta "o que esta palavra significava neste período?" respondida pelo
modelo sem supervisão.

---

## Por que isso é mais rico que os sistemas BERT do SemEval

Os sistemas APD+BERT mostram que as distribuições de embeddings de D0 e D1
são diferentes. Eles não caracterizam **o que mudou**.

TimeFormer, via perfis relacionais, dá uma resposta interpretável:
- r_0(w) top-20: as palavras mais próximas de w no vocabulário de D0
- r_1(w) top-20: as palavras mais próximas de w no vocabulário de D1
- Δr(w) = r_1(w) - r_0(w): o que w ganhou e perdeu em proximidade

Esse Δr é uma caracterização semântica da mudança. Hamilton 2016 faz algo
análogo, mas com vetores estáticos. TimeFormer faz com representações
contextuais, domain-adapted, de um modelo que viveu a transição.

---

## Experimentos concluídos após o realinhamento

### Prioridade 1: demonstração qualitativa de vizinhança temporal — concluída

Implementamos:

```python
r_t(w) = {v: cos(mean_occ h_t(w, o), centroide_t(v))
           for v in vocabulário_sobreposição}
top20_D0 = sorted(r_0(w), reverse=True)[:20]
top20_D1 = sorted(r_1(w), reverse=True)[:20]
delta_r = {v: r_1(w)[v] - r_0(w)[v] for v in vocabulário_sobreposição}
top10_ganhos  = sorted(delta_r, reverse=True)[:10]
top10_perdas  = sorted(delta_r)[:10]
```

Na execução final, usamos a camada 2, geometria centrada e 3.216 referências
compartilhadas. Para tornar `delta_r` comparável entre períodos, também
calculamos:

```
z_t(w)[v] = padronização de r_t(w)[v] entre referências
Delta_z(w)[v] = z_1(w)[v] - z_0(w)[v]
```

Os ganhos e perdas exibidos exigem que `v` esteja entre as 50 referências mais
próximas em pelo menos um período. Isso evita que grandes movimentos entre
referências remotas dominem o relatório.

O arquivo completo contém todos os 37 alvos e todas as referências:

```
outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/
temporal_relational_neighborhoods/neighborhoods.csv
```

#### `plane_nn`: resultado qualitativo forte

```
D0:
line, angle, plate, column, stock, canal, building, coast, border, ridge

D1:
boat, ship, fence, rail, route, pole, building, road, flag, trail
```

Ganhos salientes:

```
station, boat, passenger, route, track, machine, wagon, cart, trail
```

Perdas salientes:

```
line, angle, root, vine, fruit, bark
```

O resultado confirma uma transição imediatamente reconhecível de um campo
geométrico/material para **transporte amplo**. Ele não sustenta ainda a versão
mais específica "geometria para aviação": `aircraft`, `airline`, `pilot` e
`airport` não dominam o top-20.

#### `chairman_nn`: núcleo semântico estável

```
D0:
secretary, editor, commander, director, president, committee, jury

D1:
secretary, director, commander, president, commissioner, governor, publisher
```

Os ganhos dizem respeito sobretudo à realização histórica moderna do campo:

```
bush, clinton, richard, board, republican, vice, staff, executive
```

E as perdas incluem formas antigas de autoridade:

```
emperor, prophet, preacher, merchant, king
```

Isso sustenta a distinção entre estabilidade do campo de liderança e mudança
do ambiente institucional. Também explica por que APD de ocorrências pode ser
alto sem implicar substituição do sentido lexical.

#### `tree_nn`: estabilidade ampla

```
D0:
rock, water, leaf, valley, grass, grove, stem, bud, bark, vine, soil

D1:
wood, fountain, stone, sand, garden, mountain, grass, forest,
valley, leaf, bird
```

Há reorganização interna, mas não transferência para um campo semanticamente
alheio.

#### `graft_nn`: evidência mista

```
D1:
compound, machinery, currency, commodity, mechanic, utility,
consumption, tool, facility, acid, organ, substance, cell
```

Ganhos incluem:

```
compound, sanction, exposure, fraud, industry, corruption, research
```

Porém, a vizinhança D0 é heterogênea:

```
globe, bee, planet, road, chapel, horizon, platform, frontier,
village, ship, town, vine
```

Portanto, `graft` apresenta forte transição relacional, mas o perfil centroide
atual não autoriza descrevê-la sozinho como uma trajetória limpa de botânica
para medicina/corrupção.

### Conclusão atual da prioridade 1

Os resultados sustentam a reivindicação estreita:

> TimeFormer consegue caracterizar algumas transições relacionais temporais
> por vizinhanças lexicais interpretáveis e locais a cada checkpoint, sem
> alinhamento geométrico post-hoc.

Ainda não sustentam:

> TimeFormer identifica consistentemente os sentidos lexicais de qualquer
> palavra.

O critério qualitativo foi satisfeito claramente por `plane` e pela estabilidade
de `chairman`; foi satisfeito parcialmente por `tree`; e permanece misto para
`graft`.

Artefatos:

```
scripts/report_temporal_relational_neighborhoods.py
outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/
temporal_relational_neighborhoods/report.md
outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/
temporal_relational_neighborhoods/interpretation.md
```

---

## Experimentos que ainda precisam ser feitos

### Próxima prioridade: comparação explícita com Hamilton 2016

Rodar word2vec separadamente em D0 e D1, alinhar os espaços por Procrustes
ortogonal e aplicar exatamente o mesmo protocolo de relatório:

```
top-20 D0
top-20 D1
ganhos e perdas entre referências salientes
Spearman graded
```

Esse teste passou à frente dos demais porque as vizinhanças TimeFormer, embora
promissoras, não estabelecem contribuição por si mesmas. Se word2vec produzir
a mesma caracterização com menor custo e maior clareza, a novidade atribuída ao
treinamento contínuo enfraquece substancialmente.

Critérios comparativos:

1. coerência histórica dos top-20 sob protocolo cego;
2. separação de `plane` e estabilidade de `chairman`;
3. qualidade dos ganhos e perdas;
4. desempenho quantitativo no SemEval;
5. capacidade exclusiva do TimeFormer de fornecer trajetórias com N
   checkpoints, que não pode ser demonstrada neste corpus de apenas dois
   períodos.

### Prioridade posterior: field-controlled APD para todas as 37 palavras

O controle de campo foi calculado apenas para os quatro casos manuais. Antes de
generalizá-lo, precisamos definir campos automáticos sem reintroduzir o
clustering como estimador de sentido:

```
Delta_adj(w) =
score_relacional(w) - mediana(score_relacional(controles_de_campo(w)))
```

Esse experimento deve ser apresentado como controle de nuisance compartilhado,
não como identificador automático de sentidos.

### Prioridade posterior: modelo maior como ablation

O modelo atual (d=128, 3 camadas) é pequeno. Testar d=256 ou d=512 mede
quanto do sinal baixo é limitação de capacidade versus limitação de
arquitetura. Se um modelo maior dobra o Spearman, a contribuição é o
treinamento contínuo; se não, há algo mais fundamental.

---

## O que não vamos fazer (decisões encerradas)

1. **Atlas WSD externo como arquitetura principal**: exige BEM/ConSeC como
   componente central. A contribuição vira do encoder externo, não do
   TimeFormer. É paper separado se os resultados justificarem.

2. **Clustering como estimador de sentido**: muro de identificabilidade é
   formal, não de implementação. Mais épocas ou melhores algoritmos não
   atravessam essa parede.

3. **Maximizar Spearman SemEval como objetivo único**: 37 palavras não dão
   poder estatístico suficiente para selecionar arquiteturas. O Spearman é
   métrica secundária de validação, não objetivo primário.

---

## A contribuição como seria descrita num paper

> Apresentamos o TimeFormer, uma arquitetura de Transformer treinado
> continuamente sobre corpora cronologicamente ordenados para detecção de
> mudança semântica temporal. Ao contrário de abordagens que treinam modelos
> independentes por período e os alinham post-hoc, ou que aplicam modelos
> pré-treinados fora do domínio histórico, o TimeFormer aprende a transição
> semântica diretamente durante o treinamento. Introduzimos perfis relacionais
> como mecanismo de comparação sem alinhamento entre checkpoints, e mostramos
> que o modelo produz vizinhanças semânticas temporalmente coerentes para
> palavras com mudança documentada — sem qualquer anotação externa. Avaliamos
> quantitativamente no benchmark SemEval-2020 Task 1 e comparamos com
> word2vec diacrônico e métodos baseados em BERT.

---

## Critérios para saber se estamos no caminho certo

**Continuar se:**
- r_0(plane) top-10 contém palavras do campo geométrico/físico — **satisfeito**
- r_1(plane) top-10 contém palavras de transporte — **satisfeito**
- o núcleo institucional de chairman permanece estável — **satisfeito**
- Delta_adj com controle de campo coloca plane acima de chairman
- Spearman com field-control supera 0.25

**Reformular se:**
- As vizinhanças top-20 forem incoerentes para múltiplos alvos — **risco
  observado em graft**
- O modelo maior (d=512) não melhorar substancialmente o sinal
- A comparação com Hamilton 2016 mostrar que word2vec estático produz
  vizinhanças igualmente coerentes com muito menos complexidade

**O critério qualitativo continua necessário, mas não é suficiente**. Um leitor
reconhece imediatamente a transição de `plane` e a estabilidade de `chairman`.
Agora precisamos demonstrar que essa caracterização é superior ou
complementar à obtida por baselines mais simples sob o mesmo protocolo.
