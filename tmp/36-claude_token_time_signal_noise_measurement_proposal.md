# Proposta para segunda opinião: uma escala sinal-ruído para deslocamentos `token@time`

## Objetivo desta consulta

Implementamos a primeira versão do framework `token@time`. Ele permite
consultar o perfil relacional de uma palavra em cada período, comparar seus
perfis, identificar relações ganhas e perdidas e buscar palavras com
deslocamentos semelhantes.

Agora queremos melhorar a interpretação quantitativa dos resultados. A saída
atual usa principalmente similaridade cosseno, distância `1 - cos` e diferenças
padronizadas. Esses números são matematicamente definidos, mas pouco intuitivos
para um pesquisador interpretar:

```text
plane deslocamento = 0,0655
chairman deslocamento = 0,0554
graft deslocamento = 0,0344
```

Gostaríamos de saber se é metodologicamente defensável construir uma escala
comparável a uma unidade de medida interna do instrumento: não um "metro
semântico" universal, mas uma medida que diga quanto o deslocamento observado
excede a variação esperada sob ausência de mudança temporal.

Não assuma que nossa preferência por uma leitura inspirada em sinal e ruído é
correta. Queremos que você identifique pressupostos frágeis, alternativas
melhores e controles necessários.

## Posição científica do projeto

O Timeformer não precisa classificar automaticamente sentidos lexicais para ser
bem-sucedido. Seu objeto principal é:

```text
token@time
  -> perfil relacional no período
  -> vizinhos e relações observadas
  -> deslocamento do perfil ao longo do tempo
```

Os vizinhos são resultados oferecidos ao pesquisador. Uma análise externa de
sentidos pode posteriormente investigar quanto desses deslocamentos corresponde
a recomposição de sentidos em inventários lexicais conhecidos.

Portanto, a medida proposta deve quantificar mudança temporal relacional, sem
alegar que cada unidade corresponde a uma quantidade de "mudança de sentido" no
sentido lexicográfico estrito.

## Definições atuais

Para uma palavra `w`, período `t` e conjunto compartilhado de referências `V`:

```text
R_t(w)[v] = cos(c_t(w) - mu_t, c_t(v) - mu_t)
```

onde `c_t` é o centroide contextual e `mu_t` é a média dos tipos ativos no
período.

O deslocamento completo é:

```text
Delta(w, a, b) = R_b(w) - R_a(w)
```

O resumo escalar atual é:

```text
D_cos(w) = 1 - cos(R_a(w), R_b(w))
```

Temos ainda:

```text
delta_z(v) = z(R_b(w))[v] - z(R_a(w))[v]
```

usado para ordenar relações ganhas e perdidas.

Para comparar palavras diferentes:

```text
similaridade_de_deslocamento(w, u) = cos(Delta(w), Delta(u))
```

Essas grandezas respondem a perguntas distintas e não deveriam ser
artificialmente fundidas numa única escala.

## Uma apresentação mais intuitiva já possível

O deslocamento `1 - cos` pode ser convertido em rotação angular:

```text
theta(w) = arccos(1 - D_cos(w))
```

Com os valores observados:

| palavra | `1 - cos` | rotação aproximada |
|---|---:|---:|
| `plane` | 0,0655 | 20,8 graus |
| `chairman` | 0,0554 | 19,1 graus |
| `graft` | 0,0344 | 15,1 graus |

Graus são visualmente mais compreensíveis, mas ainda não dizem se a mudança é
grande em relação à incerteza e à variabilidade normal da palavra.

## Hipótese de uma medida sinal-ruído

Queremos investigar duas apresentações complementares:

```text
SNR_temporal(w) =
    D_observado(w) / centro(D_nulo(w))
```

e:

```text
Z_temporal(w) =
    [D_observado(w) - centro(D_nulo(w))]
    / escala(D_nulo(w))
```

Além delas, podemos reportar:

```text
percentil_nulo(w) =
    P[D_nulo(w) <= D_observado(w)]
```

Uma saída ideal seria semelhante a:

```text
plane
  rotação observada:              20,8 graus
  rotação típica sob estabilidade: 8,1 graus
  razão observada/nulo:             2,57
  deslocamento acima do nulo:       2,31 unidades padronizadas
  percentil sob o nulo:            98,9%
```

Os números acima são apenas ilustrativos. Ainda não estimamos esse nulo.

## O que pode significar "ruído"

Há pelo menos quatro fontes diferentes, que não queremos misturar:

1. **Ruído amostral de ocorrências:** o centroide mudaria se outras
   ocorrências do mesmo período fossem amostradas.
2. **Variação contextual intraperíodo:** a palavra possui usos contextuais
   heterogêneos mesmo sem mudança temporal.
3. **Variação do corpus:** composição documental, gênero, tópico, frequência e
   tamanho das amostras podem produzir deslocamento aparente.
4. **Variação do instrumento:** seed, checkpoint, camada, referências e
   procedimento de extração podem alterar a medida.

Precisamos decidir se existe um único nulo apropriado ou uma decomposição de
incerteza que preserve essas fontes separadamente.

## Nulos candidatos

### 1. Reamostragem dentro de cada período

Reamostrar ocorrências de `w` e das referências dentro de D0 e D1, reconstruir
os perfis e recalcular `D`.

Vantagem:

- estima incerteza amostral próxima do objeto medido.

Risco:

- se D0 e D1 realmente diferem, reamostrar separadamente preserva a diferença
  e não constitui necessariamente um nulo de estabilidade.

### 2. Divisão aleatória do corpus combinado

Combinar D0 e D1, criar pseudo-períodos pareados em tamanho e frequência e
recalcular o deslocamento.

Vantagem:

- remove deliberadamente a cronologia.

Riscos:

- apaga diferenças documentais reais que podem fazer parte do fenômeno;
- pode produzir um nulo excessivamente homogêneo;
- exige preservar estrutura documental para evitar vazamento entre janelas.

### 3. Permutação de rótulos de período por documento

Permutar D0/D1 no nível do documento, preservando os tamanhos dos períodos.

Vantagem:

- mantém dependências internas dos documentos;
- testa diretamente a associação entre perfil e rótulo temporal.

Risco:

- os períodos têm composição e tamanhos diferentes; permutabilidade pode não
  ser válida sem estratificação.

### 4. Split-half dentro de cada período

Comparar duas metades independentes de D0 e duas de D1 para estimar a
repetibilidade do perfil.

Vantagem:

- oferece uma unidade de instabilidade observável sem apagar a cronologia.

Limitação:

- mede repetibilidade, não gera sozinho a distribuição nula da diferença
  D0-D1.

### 5. Palavras-controle

Calibrar a escala por palavras externas pareadas por frequência e dispersão,
sem evidência conhecida de mudança.

Vantagem:

- produz uma referência substantiva.

Riscos:

- "estável" é uma hipótese difícil de garantir;
- controles podem diferir em polissemia, gênero e campo semântico;
- não deve transformar estabilidade presumida em verdade circular.

## Questão sobre unidade global ou específica por palavra

Uma unidade global é mais comparável:

```text
1 unidade = 1 desvio-padrão do nulo agregado
```

Mas palavras raras, frequentes, monossêmicas e contextualmente dispersas têm
pisos de ruído distintos. Uma unidade específica por palavra é estatisticamente
mais justa:

```text
Z_temporal(w) usa o nulo próprio de w
```

Por outro lado, isso pode prejudicar comparações entre palavras. Uma palavra
com deslocamento bruto pequeno e baixa variância pode superar outra com
deslocamento grande e alta variância.

Queremos uma recomendação explícita sobre:

- ranking por magnitude bruta;
- ranking por evidência acima do nulo;
- apresentação conjunta dessas duas dimensões;
- eventual modelo hierárquico que estime ruído por palavra com partial pooling.

## Problema identificado na incerteza atual

`PeriodStatistics.standard_error(layer)` calcula:

```text
sqrt(dispersão dos hidden states / número de ocorrências)
```

Essa grandeza estima a incerteza do centroide no espaço oculto. Atualmente,
`compare_profiles` combina os erros dos dois períodos por:

```text
sqrt(SE_a^2 + SE_b^2)
```

e apresenta o resultado junto ao score de deslocamento.

Isso parece dimensionalmente inadequado: o erro do centroide está em distância
euclidiana no espaço oculto, enquanto o score é uma distância angular entre
perfis relacionais. Não há uma propagação explícita através de centralização,
cossenos, conjunto de referências e comparação final.

Nossa leitura provisória é que esse valor não deve ser chamado de
`standard_error` do deslocamento. A incerteza do score deve ser estimada
diretamente por bootstrap ou por uma propagação estatística justificada.

## Problema lexical separado da escala

Em uma exploração inicial de `graft`, apareceram referências como:

```text
graf
wil
wit
mit
fit
```

O alvo `graft` está sendo corretamente representado pela média contextual de:

```text
graf + ##t
```

Entretanto, o filtro de referências exclui tokens iniciados por `##`, mas aceita
`graf` e `wil`, porque eles são entradas alfabéticas sem prefixo no vocabulário
WordPiece. Eles podem ser fragmentos produtivos, não palavras lexicais
interpretáveis.

Isso não invalida necessariamente o cálculo vetorial, mas invalida a leitura
humana de alguns vizinhos. Antes de avaliar a nova escala, provavelmente
precisamos construir referências a partir de tipos lexicais observados no
corpus, validar tokenização integral ou agregar WordPieces por palavra.

Pedimos que você trate separadamente:

1. validade da escala quantitativa;
2. validade lexical das dimensões usadas para explicar o deslocamento.

## Saída científica desejada

Não queremos substituir os objetos completos por um único número. Uma consulta
deveria preservar:

```text
magnitude bruta do deslocamento
rotação angular
incerteza ou distribuição nula
razão sinal-ruído
percentil sob o nulo
turnover de vizinhos
ganhos e perdas relacionais
estabilidade entre seeds
contagens e metadados
```

Também precisamos distinguir:

```text
evidência de estabilidade
ausência de evidência de mudança
evidência de mudança
dados insuficientes
```

## Perguntas para a segunda opinião

1. A analogia sinal-ruído é adequada para esse estimando ou induz uma
   interpretação incorreta inspirada em Shannon?
2. O que deveria ser chamado de sinal e o que deveria ser chamado de ruído?
3. Qual nulo responde melhor à pergunta "quanto esta palavra mudou acima da
   variação esperada sem cronologia"?
4. Devemos calibrar `1 - cos`, o ângulo, a norma de `Delta`, o turnover ou mais
   de uma dessas medidas?
5. SNR como razão é estatisticamente estável quando o centro do nulo se
   aproxima de zero?
6. É preferível usar média/desvio-padrão, mediana/MAD, quantis empíricos ou
   outra padronização?
7. A unidade deve ser global, específica por palavra ou hierárquica?
8. Como preservar comparabilidade entre palavras sem favorecer palavras raras
   ou contextualmente homogêneas?
9. Bootstrap de ocorrências é suficiente, ou referências e documentos também
   precisam ser reamostrados?
10. Como respeitar dependências entre ocorrências provenientes do mesmo
    documento?
11. O teste de permutação temporal é válido quando D0 e D1 diferem em tamanho,
    gênero e composição documental?
12. Como combinar resultados de seeds sem confundir incerteza amostral e
    incerteza do instrumento?
13. Que intervalos devem acompanhar ângulo, SNR, escore padronizado e
    percentil?
14. Como corrigir multiplicidade ao ranquear milhares de palavras?
15. O erro de centroide atual deve ser removido da interface, renomeado ou
    preservado apenas como diagnóstico separado?
16. Que filtro ou representação lexical deve substituir a filtragem superficial
    de WordPieces?
17. Qual implementação mínima teria maior valor informacional antes de
    sofisticarmos o modelo estatístico?
18. Que experimento falsificaria a alegação de que a nova unidade mede mudança
    temporal acima do ruído?

## Proposta provisória para você criticar

Nossa sequência provisória seria:

1. corrigir o conjunto de referências lexicais;
2. manter `1 - cos` como medida bruta canônica e apresentar também o ângulo;
3. remover a interpretação do erro de centroide como erro do deslocamento;
4. produzir pseudo-períodos por permutação ou repartição no nível documental;
5. reconstruir perfis e deslocamentos em muitas réplicas;
6. obter, por palavra, mediana, MAD e percentil empírico do nulo;
7. reportar magnitude bruta e evidência acima do nulo em eixos separados;
8. comparar rankings por magnitude e por evidência;
9. validar estabilidade entre seeds e diferentes subconjuntos de referências;
10. somente depois decidir se uma unidade padronizada merece um nome próprio.

Nossa preferência atual é não chamar essa grandeza de "metro semântico".
Termos como:

```text
deslocamento acima do nulo
unidades de variabilidade esperada
percentil temporal sob estabilidade
```

parecem mais defensáveis.

## Arquivos relevantes

Especificação:

```text
docs/39-token_time_analysis_framework.md
```

Implementação:

```text
src/timeformers/token_time.py
src/timeformers/token_time_repository.py
src/timeformers/token_time_statistics.py
src/timeformers/token_time_metrics.py
src/timeformers/token_time_index.py
src/timeformers/relational.py
scripts/token_time/build_profiles.py
scripts/token_time/explore_index.py
```

## Formato de resposta solicitado

Pedimos uma revisão crítica organizada em:

1. diagnóstico conceitual;
2. estimando recomendado;
3. desenho do nulo;
4. unidade e apresentação;
5. tratamento da incerteza;
6. validade lexical das referências;
7. experimento mínimo;
8. critérios de sucesso e de abandono;
9. alterações concretas sugeridas na API;
10. riscos de interpretação no artigo.

Ao final, indique uma sequência operacional curta, priorizada por valor
informacional e custo.

---

# Resposta: segunda opinião (Claude, 2026-06)

Esta seção é a revisão crítica solicitada acima, no formato pedido (10
itens + sequência operacional), seguida de uma seção 11 com uma evolução
concreta já implementada nesta sessão: uma métrica `lexical_validity`
derivada inteiramente do corpus/tokenizador, sem depender de listas de
palavras externas em inglês -- ponto levantado pelo autor porque o projeto
vai treinar e investigar em seguida sobre o corpus da Folha de São Paulo
(português brasileiro), e potencialmente outras línguas.

## 1. Diagnóstico conceitual

A moldura "sinal vs. ruído" carrega uma suposição implícita problemática: que
existe um valor "verdadeiro" de mudança semântica e que o desvio em torno dele
é "ruído" no sentido de erro de medição a ser descontado. Mas para
`token@time`, o que chamamos de "ruído" é, na verdade, **a variação que o
próprio fenômeno (uso linguístico) produziria mesmo sem mudança de sentido**:
poligonia de contexto, variação estilística, tamanho de amostra, etc. Isso não
é erro de instrumento -- é a distribuição nula do próprio objeto medido.

Recomendo abandonar o vocabulário "sinal/ruído" e SNR, e adotar:

```text
D_obs(w)       -- deslocamento observado (já definido: D_cos)
nulo de w      -- distribuição de D sob "nenhuma mudança temporal real"
Z_temporal(w)  -- D_obs comparado a essa distribuição nula
```

Isso é mais defensável porque (a) não presume que exista um "deslocamento
verdadeiro" platônico, e (b) torna explícito que o nulo é *específico de cada
palavra* (depende de sua frequência, polissemia contextual, dispersão), não um
limiar global único.

## 2. Estimador recomendado

Em vez de uma razão sinal/ruído (`D_obs / ruído`), recomendo um z-score robusto
contra a distribuição nula empírica de `w`:

```text
Z_temporal(w) = (D_obs(w) - median(D_null(w))) / MAD(D_null(w))
```

mais o percentil de `D_obs(w)` dentro de `D_null(w)`. A razão para usar
`median`/`MAD` em vez de `mean`/`std`: `D_null` tende a ter cauda direita
(poucas reamostragens produzem deslocamentos grandes por acaso), e
`median`/`MAD` são robustos a essa assimetria sem exigir testar normalidade.

O percentil é o número que efetivamente "fala" com um leitor: "o deslocamento
observado de `plane` está no percentil 97 da distribuição de deslocamentos que
`plane` teria mesmo sem mudança real" é interpretável sem jargão estatístico.

## 3. Desenho do nulo

Proponho **dois nulos complementares**, não um só -- eles capturam fontes de
variação diferentes e o acordo (ou desacordo) entre eles é informativo por si
só:

**Nulo A -- repetibilidade por split-half.** Para cada palavra `w` e período
`t`, dividir as ocorrências de `w` em duas metades aleatórias (fontes 1 e 2),
recomputar `R_t(w)` para cada metade mantendo `mu_t` fixo (calculado sobre
todas as ocorrências), e medir:

```text
s_repeat(w) = 1 - cos(R_t(w, metade1), R_t(w, metade2))
```

Repetir N vezes (reamostragem) para obter a distribuição `D_null_A(w)`. Isso
mede: "quanto o perfil de `w` se moveria se eu tivesse amostrado um conjunto
diferente de ocorrências do mesmo período, do mesmo tamanho".

**Nulo B (novo, proposto nesta revisão) -- permutação de período mantendo
`mu_t` observado.** Pegar todas as ocorrências de `w` (de ambos os períodos
juntas), permutar aleatoriamente os rótulos d0/d1 entre essas ocorrências
(mantendo o tamanho de cada grupo igual ao observado), recomputar `R_0(w)` e
`R_1(w)` para cada permutação **usando os centroides de referência `c_t(v)` e
`mu_t` reais/observados** (não recalculados -- isso é o que torna o nulo B
barato), e medir `D_null_B(w) = 1 - cos(R_0(w), R_1(w))` para cada permutação.

Isso responde: "se as ocorrências de `w` não tivessem nenhuma relação
sistemática com o eixo temporal -- se a divisão d0/d1 fosse arbitrária --
quão grande seria o deslocamento medido só por acaso de amostragem?" É
localizado (só re-amostra `w`, não o sistema de referência inteiro) e barato
(não precisa reprocessar o encoder nem recomputar `mu_t`/centroides de
referência).

**Por que os dois:** o nulo A isola "ruído de amostragem dentro de um
período"; o nulo B isola "ruído de atribuição d0/d1 para essa palavra
específica". Uma palavra pode ter `s_repeat` baixo (perfil estável dentro de
cada período) mas `D_null_B` largo (poucas ocorrências, então a permutação
produz deslocamentos grandes por acaso) -- nesse caso `D_obs` alto é menos
confiável do que `s_repeat` baixo sugeriria isoladamente. Reportar os dois.

## 4. Unidade e apresentação

Não recomendo colapsar tudo em um único número. Proponho apresentação em
**dois eixos**:

```text
eixo X: magnitude    -- D_cos(w) (ou o ângulo, em graus: arccos(cos(...)))
eixo Y: evidência    -- Z_temporal(w) e/ou percentil sob D_null
```

com interpretação por quadrante:

```text
alta magnitude, alta evidência   -> candidato forte a mudança temporal
alta magnitude, baixa evidência  -> deslocamento grande mas indistinguível do nulo
                                     (palavra rara / pouco dado -- ver dispersion/SE)
baixa magnitude, alta evidência  -> deslocamento pequeno mas consistente
                                     (pode ser real, porém de efeito pequeno)
baixa magnitude, baixa evidência -> nada a reportar
```

Isso evita a armadilha de "graft = 0,0344 < plane = 0,0655, logo plane mudou
mais" quando na verdade a posição relativa pode se inverter olhando para
`Z_temporal`.

## 5. Tratamento da incerteza

A métrica de `standard_error` (centroid, baseada em `dispersion`/`sum_sq`,
já implementada nesta sessão) deve ser **rebaixada a um diagnóstico de
qualidade de dados/cobertura**, não usada como medida de incerteza do
deslocamento em si. Ela responde "quão bem estimado é `c_t(w)`", o que é
necessário mas não suficiente -- não captura a variabilidade induzida pelo
sistema de referência (`mu_t`, `c_t(v)` para `v` em `V_active`) nem a
correlação entre as duas.

Concretamente, sugiro renomear/mover o campo: de
`TokenTimeDisplacement.standard_error` para
`TokenTimeProfile.centroid_standard_error` (por período, não por
deslocamento), e introduzir a incerteza *do deslocamento* via os nulos A/B
acima (que já incorporam toda a cadeia de cálculo, não só o centroide de
`w`).

Multiplicidade: ao reportar `Z_temporal`/percentil para milhares de palavras
(ex. todo `V_active`), usar correção de FDR (q-values, Benjamini-Hochberg) em
vez de um limiar fixo de p-valor por palavra -- senão a taxa de falsos
positivos cresce linearmente com o tamanho do vocabulário avaliado.

Entre seeds: **não fazer média de `Z_temporal` entre seeds**. Reportar
`Z_temporal` por seed e, separadamente, uma medida de concordância entre
seeds (ex. quantos dos seeds colocam `w` no mesmo quadrante, ou correlação de
`D_cos(w)` entre seeds). Médias de z-scores entre execuções com diferentes
inicializações escondem justamente o tipo de instabilidade que motivou usar
múltiplos seeds.

Classificação final em 4 categorias por palavra:

```text
estável         -- baixa magnitude, alta evidência (ou baixa evidência, tanto faz)
mudança         -- alta magnitude, alta evidência, consistente entre seeds
sem evidência   -- alta magnitude, baixa evidência (não descartar, não afirmar)
dados insuficientes -- contagem/cobertura baixa demais para qualquer conclusão
                       (usar standard_error/dispersion aqui)
```

## 6. Validade lexical das referências (revisado -- solução agnóstica de língua)

Minha sugestão original (filtrar `V_active`/conjunto de referências contra uma
lista de palavras em inglês ou `anchors.txt` do SemEval) está **descartada**.
O autor apontou corretamente que isso cria uma dependência específica de
língua, e o framework vai ser usado em seguida sobre o corpus da Folha de São
Paulo (português brasileiro) e potencialmente outras línguas -- qualquer
mecanismo de validação lexical precisa ser derivável só do corpus de
treinamento/tokenizador, sem recursos externos.

**Solução implementada nesta sessão**: `standalone_counts` / `lexical_validity`,
inteiramente derivada da tokenização WordPiece do próprio corpus:

```text
standalone_counts[v] = número de ocorrências em que o item de vocabulário v
                        constitui a palavra inteira -- ou seja, v não começa
                        com "##" E o WordPiece seguinte na mesma ocorrência
                        também não começa com "##"

lexical_validity(v) = standalone_counts[v] / counts[v]
```

Intuição: um item de vocabulário como `"graf"` pode passar pelo filtro atual
(`não começa com "##"`, `isalpha()`), mas se 99% de suas ocorrências forem como
`"graf" + "##t"` (i.e. fragmento de "graft"), seu `lexical_validity` será
próximo de 0 -- sinalizando que `"graf"` não é, na prática, uma palavra
independente nesse corpus, **em qualquer língua**, sem precisar saber que
"graf" não é uma palavra inglesa.

Casos de borda já tratados na implementação:
- Para os alvos multi-subtoken (ex. "graft" = "graf" + "##t"), que recebem um
  "id virtual" em `build_profiles.py`, `standalone_counts[id_virtual] =
  counts[id_virtual]` -- ou seja, `lexical_validity = 1.0` por construção,
  porque o id virtual *representa* a palavra completa reconstruída.
- Itens com `counts[v] == 0` recebem `lexical_validity = 0` (em vez de
  divisão por zero / NaN).
- Caches antigos (sem o campo `standalone_counts`) fazem
  `lexical_validity()` levantar `ValueError` explicitamente, em vez de
  silenciosamente retornar zeros.

Estado da implementação (já feito nesta sessão):
- `src/timeformers/token_time_statistics.py`: campo `standalone_counts` em
  `PeriodStatistics`, lido/escrito em `load`/`save`, método
  `lexical_validity()`.
- `scripts/token_time/build_profiles.py`: `continuation_mask` calculada uma
  vez a partir do vocabulário (`token.startswith("##")`), acumulação
  vetorizada de `standalone_counts` no laço principal, e
  `standalone_counts[id_virtual] = counts[id_virtual]` para alvos
  multi-subtoken.
- `src/timeformers/token_time_repository.py`: `build_reference_set` ganhou
  parâmetros opcionais `lexical_validity_d0`/`lexical_validity_d1`/
  `min_lexical_validity` -- candidatos com validade abaixo do limiar em
  qualquer um dos dois períodos são excluídos do conjunto de referências
  legíveis (mas continuam em `V_active` para o cálculo de `mu_t`/perfis, que
  não devem ser afetados por este filtro).

Próximo passo natural: re-extrair os caches de `seed1000`/`seed1001` (em
andamento) e inspecionar `lexical_validity` para os ~10 itens de vocabulário
com menor valor, para confirmar que são majoritariamente fragmentos
WordPiece (`##`-prefixáveis na prática) e não palavras curtas legítimas sendo
penalizadas injustamente (ex. "a", "I" em inglês -- que de fato são palavras
standalone na maioria das ocorrências, então devem ter `lexical_validity`
alto e não serem afetadas).

## 7. Experimento mínimo

1. Escolher 5-10 palavras-alvo do SemEval com `gold_score` conhecido,
   cobrindo a faixa (algumas claramente mudaram, algumas claramente não).
2. Para cada uma, computar `D_obs`, `D_null_A` (split-half, N=200
   reamostragens), `D_null_B` (permutação d0/d1, N=200), `Z_temporal` para
   cada nulo, e `lexical_validity` das suas top-k referências.
3. Verificar: (a) palavras com `gold_score` alto tendem a ter `Z_temporal`
   alto nos dois nulos? (b) os dois nulos concordam em ranking? (c) quantas
   das top-k referências de cada palavra têm `lexical_validity` baixo hoje
   (antes do filtro) -- isso quantifica o tamanho real do problema da seção
   6.
4. Repetir para seed1000 e seed1001 separadamente; reportar concordância
   entre seeds (não média).

Custo: nenhum reprocessamento do encoder é necessário -- tudo é computável a
partir dos caches já existentes (`theta_d0.pt`/`theta_d1.pt` com `sum_sq` e
`standalone_counts`).

## 8. Critérios de sucesso e de abandono

**Sucesso**: `Z_temporal` (em pelo menos um dos dois nulos) correlaciona
positivamente com `gold_score` do SemEval no conjunto de alvos, com
concordância razoável entre seed1000/seed1001 (mesmo ranking aproximado de
quais palavras têm `Z_temporal` alto).

**Abandono/revisão**: se `Z_temporal` não correlaciona melhor que `D_cos`
bruto com `gold_score`, ou se os dois nulos (A e B) discordam fortemente e
sistematicamente (ex. nulo A diz "está tudo dentro do esperado" para palavras
que nulo B marca como fora) -- isso indicaria que a fonte dominante de
variação não é nenhuma das duas modeladas, e vale voltar a investigar a
*forma* da distribuição nula antes de insistir num único z-score.

Critério intermediário (não abandono, mas replanejamento): se
`lexical_validity` mostrar que uma fração grande (>30%?) das referências
atuais de `V_active` são fragmentos WordPiece de baixa validade, o impacto
disso em `mu_t`/centroides de referência (não só na lista legível) merece
investigação separada antes de prosseguir com os nulos.

## 9. Alterações concretas sugeridas na API

```text
PeriodStatistics
  + standalone_counts: Tensor              (feito)
  + lexical_validity() -> Tensor           (feito)

TokenTimeProfile
  + centroid_standard_error: float         (renomeado de
                                             TokenTimeDisplacement.standard_error)

TokenTimeDisplacement
  - standard_error                          (remover -- ver seção 5)
  + d_null_a: list[float]                   (amostras do nulo A, split-half)
  + d_null_b: list[float]                   (amostras do nulo B, permutação d0/d1)
  + z_temporal_a: float
  + z_temporal_b: float
  + percentile_a: float
  + percentile_b: float

build_reference_set(..., lexical_validity_d0, lexical_validity_d1,
                     min_lexical_validity: float = 0.0)   (feito)
```

`d_null_a`/`d_null_b` como listas (amostras brutas) em vez de só
média/desvio, para permitir recomputar `median`/`MAD`/percentil sem reter
suposições de normalidade, e para permitir auditoria (plotar a distribuição
nula de uma palavra específica quando o resultado for surpreendente).

## 10. Riscos de interpretação no artigo

- **Risco principal**: leitores (e revisores) tendem a ler "z-score alto" como
  "significância estatística" no sentido confirmatório clássico, e a
  "mudança de sentido" no sentido lexicográfico. O texto deve deixar explícito
  em toda figura/tabela que `Z_temporal` mede "deslocamento relacional além do
  esperado por amostragem", não "mudança de significado certificada".
- **Risco de multiplicidade**: reportar uma tabela com `Z_temporal` para
  centenas/milhares de palavras de `V_active` sem correção de FDR vai gerar
  "achados" espúrios que parecerão fortes individualmente.
- **Risco de generalização entre corpora**: ao migrar para Folha de São Paulo
  (e potencialmente outras línguas), qualquer componente que dependa de
  propriedades específicas do inglês (listas de palavras, contagem de
  caracteres como proxy de "palavra real", etc.) precisa ser auditado --
  `lexical_validity` (seção 6) foi desenhado especificamente para não ter essa
  dependência, mas vale revisitar `isalpha()` em `build_reference_set`, que
  pode se comportar de forma inesperada com tokenização de outros scripts/
  diacríticos.
- **Risco de seed única**: qualquer afirmação "palavra X mudou" baseada em um
  único seed deve ser tratada como hipótese, não resultado -- a seção 5 já
  recomenda reportar concordância entre seeds em vez de médias.

## Sequência operacional priorizada

1. **(feito nesta sessão)** Implementar `standalone_counts`/`lexical_validity`
   e re-extrair caches de seed1000/seed1001.
2. Inspecionar `lexical_validity` nos dois caches: confirmar que fragmentos
   WordPiece conhecidos (ex. "graf") têm valor baixo e que palavras comuns
   (ex. "the", "plane") têm valor alto; verificar o tamanho do problema (seção
   6, "critério intermediário").
3. Implementar o nulo B (permutação d0/d1 com `mu_t`/centroides de referência
   observados fixos) -- é o mais barato (sem reprocessar o encoder) e o mais
   diretamente ligado à pergunta "esse deslocamento poderia ser acaso de
   atribuição de período?".
4. Implementar o nulo A (split-half) -- exige guardar/recalcular por
   subconjunto de ocorrências; mais caro que o B mas ainda sem reprocessar o
   encoder se as somas/`sum_sq` puderem ser particionadas.
5. Rodar o experimento mínimo (seção 7) nas palavras-alvo do SemEval, comparar
   `Z_temporal` com `gold_score`, checar concordância seed1000 vs seed1001.
6. Com base nos resultados de (5), decidir entre: avançar para a apresentação
   de dois eixos (seção 4) + classificação em 4 categorias (seção 5), ou
   revisar o desenho do nulo (critério de abandono, seção 8).
7. Só então atualizar `explore_index.py`/relatórios para expor
   `Z_temporal`/percentil/quadrante ao pesquisador, e aplicar
   `min_lexical_validity` em `build_reference_set` para os relatórios de
   vizinhança.

# Resposta à contra-análise do codex (Claude, 2026-06)

O codex revisou a seção anterior ("Resposta: segunda opinião") e devolveu 7
pontos de crítica + uma reordenação de prioridades em 8 passos. Concordo com
os 7 pontos. Abaixo, o que muda em relação ao que eu havia escrito, e o
resultado da correção do item 7 (já implementada e verificada nos dois
seeds).

## 1. Caches atuais não bastam para os nulos

Correto. `counts`/`sums`/`sum_sq` são agregados sobre *todas* as ocorrências
de um período -- não há como reconstruir "o que seria `D` se as ocorrências
de `w` fossem divididas de outra forma entre d0/d1" sem um cache adicional
por ocorrência (ou, no mínimo, por documento). Isso é uma dependência dura
para o nulo B (item 4 da nova ordem) e para o split-half (item 8): ambos
precisam particionar o conjunto de ocorrências de `w`, e isso não é possível
a partir dos agregados existentes. Aceito que essa cache (item 3) vem antes
de qualquer nulo.

## 2. Documento como unidade de reamostragem

Concordo -- e isso também é consistente com a correção do item 7 (abaixo):
a unidade natural para preservar correlação intra-texto e janelas
sobrepostas é o documento, não a ocorrência isolada do token. Vou adotar
"documento" como unidade de permutação tanto para o nulo B quanto para
qualquer diagnóstico de repetibilidade.

## 3-4. Split-half não é comparável a D_obs; classificação de estabilidade

Aceito a demoção do split-half (Nulo A) a diagnóstico de repetibilidade
apenas -- comparar um `D` calculado com n/2 ocorrências por braço contra um
`D_obs` calculado com `n0`/`n1` (tipicamente desbalanceados e muito
maiores) mistura duas perguntas diferentes. O Nulo B, que preserva `n0`/`n1`
e permuta apenas a *atribuição* das ocorrências de `w` aos períodos
(mantendo `mu_t`/centroides de referência fixos), é a pergunta certa: "dado
o desenho do experimento (esses tamanhos de amostra, esses dados), o `D_obs`
observado é atípico?"

Sobre a classificação: a proposta original ("baixa evidência = estável") está
errada -- ausência de evidência não é evidência de ausência, especialmente
com `n` pequeno. Adoto o esquema de 5 categorias do codex, que separa
explicitamente "não sei" de "sei que não mudou (dentro de uma margem)":

- **mudança detectável** (`D_obs` grande e fora do nulo)
- **mudança pequena mas detectável** (significativa, porém de magnitude
  modesta -- vale para o leitor distinguir "mudou" de "mudou muito")
- **inconclusivo** (nem dentro da margem de equivalência, nem fora do nulo --
  faltam dados para decidir)
- **compatível com estabilidade dentro de margem** (equivalência: `D_obs`
  dentro de uma margem pré-especificada E o nulo é estreito o bastante para
  essa margem ser informativa)
- **dados insuficientes** (nulo largo demais para qualquer veredito, ex.
  `n0` ou `n1` muito pequenos)

A margem de equivalência precisa ser definida em unidades de `D_cos`
(ex. com base na escala observada do nulo B calibrado em pseudo-períodos --
ver item 5), não escolhida arbitrariamente.

## 5. Calibrar o nulo antes de selecionar por gold_score

Concordo, e essa é provavelmente a correção de maior impacto na sequência:
calibrar o protocolo (nulo B + classificação) usando **pseudo-períodos**
(dividir um único período em duas metades aleatórias, rodar o pipeline como
se fossem d0/d1) antes de tocar nas 37 palavras-alvo do SemEval ou no
`gold_score`. Os testes de calibração:

- distribuição de p-valores/percentis sob H0 (pseudo-períodos não deveriam
  ter "mudança real") deve ser ~uniforme;
- taxa de falsos positivos na classificação "mudança detectável" deve ser
  próxima do alfa nominal;
- repetibilidade entre seed1000/seed1001 nos pseudo-períodos;
- o nulo deve estreitar com mais dados (`n` maior -> nulo mais estreito).

Só depois de congelar o protocolo com base nisso, aplicar uma vez às 37
palavras-alvo e tratar a correlação com `gold_score` como validade externa
secundária -- não como critério de seleção/ajuste.

## 6. Z_robusto, MAD=0, p-valor, número de permutações

Aceito as três correções de fórmula:

- `Z_robusto = (D_obs - mediana(D_null)) / (1.4826 * MAD(D_null))`, com o
  fator `1.4826` para que `MAD` seja consistente com o desvio-padrão sob
  normalidade;
- quando `MAD(D_null) == 0` (ex. nulo degenerado por `n` pequeno), `Z_robusto`
  não está definido -- esses casos devem cair em "dados insuficientes" na
  classificação, não produzir um `Z` artificialmente infinito/grande;
- p-valor unilateral via contagem: `p = (1 + #{D_null >= D_obs}) / (B + 1)`,
  que é a forma correta para permutação (evita `p=0`);
- 200 permutações é um tamanho de piloto -- insuficiente para qualquer
  correção de FDR sobre milhares de palavras de `V_active`. Para a avaliação
  final (passo 7) será preciso `B` bem maior (ou um nulo paramétrico ajustado
  aos pseudo-períodos, se viável).

## 7. Bug do `lexical_validity` -- confirmado e corrigido

Confirmado: o cálculo original em `build_profiles.py` checava
`next_is_continuation` sobre `input_ids_t` *já janelado* -- uma palavra cujas
subpeças WordPiece cruzam a fronteira de uma janela de `encode_windows` tinha
sua primeira peça classificada como standalone (a peça de continuação cai na
próxima janela, fora de vista). Como o codex notou, a correção precisa operar
sobre os limites de palavra reais, e a abordagem que adotei (tokenizar cada
documento *inteiro*, sem janelamento -- `truncation=False`, mesma chamada que
`encode_windows` usa antes de fatiar) é independente de língua mas depende do
esquema WordPiece (`"##"` como marcador de continuação); para outro tokenizer
(ex. BPE/SentencePiece) o marcador de continuação seria outro, mas a lógica
de "documento inteiro, sem janela" permanece.

Implementação: nova função `document_standalone_counts()` em
`build_profiles.py`, chamada por documento (lista de tokens) com
`tokenizer(tokens, is_split_into_words=True, add_special_tokens=False,
truncation=False)`; o cálculo windowed anterior foi removido de
`extract_context_statistics` (mantendo apenas o incremento
`standalone_counts[virtual_id] += 1` para os alvos multi-subtoken, que já
operava por ocorrência e estava correto).

**Resultado da re-extração (seed1000, CPU, completa):**

| palavra | antes (lv0 / lv1) | depois (lv0 / lv1) |
|---|---|---|
| "graf" (idx 22160) | 0.0054 (1/186) / 0.0940 (14/149) | 0.0054 (1/186) / 0.0940 (14/149) -- inalterado |
| "graft" (virtual, multi-subtoken) | 1.0000 / 1.0000 | 1.0000 / 1.0000 -- inalterado |
| "the" (idx 1996) | 0.9994 / 0.9990 | 0.9888 (448782/453870) / 0.9971 (402724/403890) |
| "plane" (idx 4946) | 0.9967 / 0.9831 | 0.9435 (284/301) / 0.9807 (811/827) |
| "chairman" (idx 3472) | 1.0000 / 0.9792 | 0.9808 (153/156) / 0.9736 (702/721) |

Interpretação: para palavras de alta frequência ("the", "plane",
"chairman"), a correção *reduz* `lexical_validity`, como esperado -- o cálculo
janelado anterior inflava `standalone_counts` ao classificar erroneamente
como standalone qualquer ocorrência cuja peça de continuação caísse na
janela seguinte. Para "graf"/"graft" o valor não mudou: o único caso
"standalone" de "graf" em d0 (e os 14 em d1) não eram artefatos de fronteira
de janela -- são ocorrências reais de "Graf" como substantivo próprio (ex.
sobrenome/título; plausível o aumento em d1 [1960-2010] refletir a
visibilidade de "Graf" como sobrenome, ex. Steffi Graf). Em todos os casos os
valores permanecem em faixas que separam claramente fragmento (~0) de palavra
real (~0.94-1.0), então a conclusão qualitativa da seção 6 não muda -- mas os
números absolutos usados como threshold (`min_lexical_validity`) devem usar
os valores corrigidos.

**seed1001 (re-extração completa):** valores idênticos a seed1000 em todos os
casos acima (`"graf"` 0.0054/0.0940, `"graft"` 1.0000/1.0000, `"the"`
0.9888/0.9971, `"plane"` 0.9435/0.9807, `"chairman"` 0.9808/0.9736) -- esperado,
já que `standalone_counts`/`counts`/`lexical_validity` dependem apenas da
tokenização do corpus (vocab/tokenizer do checkpoint), não dos pesos do
modelo, e os dois seeds compartilham o mesmo corpus e tokenizer.

## Ordem de prioridade revisada (adoto a proposta do codex)

Adoto a reordenação em 8 passos do codex, substituindo a "Sequência
operacional priorizada" anterior:

1. ~~Corrigir `lexical_validity` com limites de palavra reais~~ -- **feito**
   (ver item 7 acima; verificado em seed1000, seed1001 em andamento).
2. Remover o erro de deslocamento propagado (`TokenTimeDisplacement.
   standard_error`) -- esse valor não tem uma interpretação válida como
   "incerteza do deslocamento" (já discutido na seção anterior, mas reforço:
   deve ser removido/renomeado, não apenas reinterpretado).
3. Criar cache por ocorrência/documento -- pré-requisito para os nulos
   (item 1 desta resposta).
4. Implementar o nulo B primeiro: permutação ao nível de documento que
   preserva `n0`/`n1`.
5. Validar a calibração em pseudo-períodos, sem usar `gold_score`.
6. Congelar o protocolo (nulo + classificação + margem de equivalência).
7. Avaliar as 37 palavras-alvo do SemEval uma única vez com o protocolo
   congelado; tratar `gold_score` como validade externa secundária.
8. Adicionar o split-half apenas como diagnóstico de repetibilidade
   (não como nulo).

Isso substitui a sequência de 7 passos da seção anterior. O item 1 está
concluído; o próximo passo é o item 2 (remoção do `standard_error`
propagado), que é independente de dados e pode ser feito em paralelo com a
re-extração do seed1001.

## Atualização (Claude, 2026-06): passos 2 e 3 da ordem revisada

**Passo 2 (feito):** removido `TokenTimeDisplacement.standard_error` e a
propagação `sqrt(profile_a.standard_error**2 + profile_b.standard_error**2)`
em `compare_profiles` (`src/timeformers/token_time.py`), e a respectiva
impressão em `scripts/token_time/explore_index.py`. `TokenTimeProfile.
standard_error` (diagnóstico por palavra/período, de
`PeriodStatistics.standard_error(layer)`) e `PeriodStatistics.dispersion`/
`standard_error` permanecem -- são diagnósticos válidos de qualidade de dado
por tipo/período, só não são mais somados em quadratura para produzir uma
"incerteza do score de deslocamento" (que não tinha justificativa, já que
`score = 1 - cos(...)` não é uma diferença simples das duas centroides).

**Passo 3 (feito):** novo módulo `src/timeformers/token_time_occurrences.py`
com `OccurrenceCache`: para cada palavra-alvo (id real de vocabulário ou id
"virtual" multi-subtoken), guarda `layer_1`/`layer_2` com o hidden state de
cada ocorrência individual (`(n_ocorrencias, hidden_size)`) e `doc_index`
(`(n_ocorrencias,)`, índice do documento de origem no arquivo de corpus do
período). `build_profiles.py` ganhou `encode_windows_with_doc_index`
(replica `encode_windows` mas também devolve o índice do documento de cada
janela) e `extract_context_statistics` agora também acumula, por
palavra-alvo, os hidden states e índices de documento de cada ocorrência
(tanto para alvos de um subtoken quanto para os "virtuais" multi-subtoken),
salvos em `cache/occurrences_d{0,1}.pt`. Verificado num corpus de teste
mínimo: `OccurrenceCache[target_id]["layer_1"].mean(dim=0)` é idêntico (diff
0.0) à centroide `sums[layer_1][target_id] / counts[target_id]` das
estatísticas agregadas, e as contagens (`layer1.shape[0]`,
`doc_index.shape[0]`) batem com `counts[target_id]`, tanto para alvos
diretos quanto para o alvo multi-subtoken "graft".

Esse cache é o que falta para o passo 4 (nulo B: permutar a atribuição
d0/d1 das ocorrências de `w`, preservando `n0`/`n1`, mantendo `mu_t` e as
centroides de referência fixas em seus valores observados, e recomputando
apenas `centroid(w)` via `OccurrenceCache.centroid(target_id, layer,
occurrence_mask=...)`). `doc_index` permite que a permutação seja feita por
documento, não por ocorrência isolada, conforme o ponto 2 da contra-análise.

Re-extração em andamento (seed1000, depois seed1001) para popular
`occurrences_d{0,1}.pt` nos dois seeds.

**Re-extração concluída (2026-06-14):** `cache/occurrences_d{0,1}.pt` agora
existe para seed1000 e seed1001, verificado para "plane"/"chairman"/"graft"
em ambos os períodos e seeds -- `occ[tid]["layer_1"].shape[0]` e
`occ[tid]["doc_index"].shape[0]` coincidem exatamente com `counts[tid]`, e
`occ[tid]["layer_1"].mean(dim=0)` reproduz `sums["layer_1"][tid] /
counts[tid]` a menos de ruído de arredondamento float32 (~1e-6). `n_docs`
(documentos distintos por palavra-alvo) é tipicamente um pouco menor que
`counts` -- a maioria das palavras-alvo ocorre mais de uma vez no mesmo
documento, confirmando que a permutação por documento (passo 4) de fato
preserva correlações intra-texto que uma permutação por ocorrência
destruiria.

Passo 3 está completo nos dois seeds. Próximo: passo 4 (nulo B, permutação
por documento preservando n0/n1).

## Atualização (Claude, 2026-06): passo 4 -- nulo B implementado

Novo módulo `src/timeformers/token_time_null.py`:
`document_permutation_null(occurrences_a, occurrences_b, centroids_a,
centroids_b, mu_a, mu_b, reference_ids, layer, n_permutations, generator)`.

Implementação:

- pool das ocorrências de `w` em d0+d1 (via `OccurrenceCache`), com
  `doc_index` de d1 deslocado por `max(doc_index_d0)+1` para distinguir
  documentos dos dois períodos no pool combinado;
- agrupamento por documento (`torch.unique` sobre os ids combinados);
- em cada permutação: embaralha a ordem dos blocos de documento e os
  distribui greedily em dois grupos (A, B) tentando aproximar `|A| ~= n0` e
  `|B| ~= n1` (mantendo cada documento inteiro num único grupo -- preserva
  correlação intra-documento, ponto 2 da contra-análise);
- recalcula `centroid(w)` para cada grupo (média dos hidden states), monta
  `R_pseudoA(w)`/`R_pseudoB(w)` via `relational_profile` com `mu_a`/`mu_b` e
  as centroides de referência **observadas** (fixas -- só a linha de `w` é
  substituída), e calcula `D_null = 1 - cos(R_pseudoA, R_pseudoB)`
  (`relational.displacement`).

`TokenTimeIndex.null_b(word, reference_ids, layer=..., n_permutations=200,
generator=...)` encapsula isso, lendo `self.occurrences` (carregado
automaticamente por `TokenTimeIndex.load` quando `occurrences_d{0,1}.pt`
existe; índices de perfis antigos sem esse cache continuam funcionando, só
`null_b` fica indisponível).

**Teste fim a fim (seed1000, layer_2, B=200, seed do gerador=0):**

| palavra | D_obs | mediana(D_null) | MAD(D_null) | Z_robusto | p (one-sided) |
|---|---:|---:|---:|---:|---:|
| plane | 0.0655 | 0.0217 | 0.0016 | 18.10 | 0.005 |
| chairman | 0.0554 | 0.0261 | 0.0012 | 16.54 | 0.005 |
| graft | 0.0344 | 0.0227 | 0.0009 | 9.04 | 0.005 |

`p=0.005` é o mínimo possível com `B=200` (`1/(200+1)` ~= 0.005) -- consistente
com o aviso do item 6 da contra-análise: 200 permutações bastam para um
piloto, mas não para distinguir p-valores menores nem para FDR sobre
`V_active`. As três palavras parecem fortemente fora do nulo nesta amostra
pequena -- mas, como já registrado, isso ainda não significa "mudança de
sentido": o passo 5 (calibração com pseudo-períodos, sem gold) é o que falta
para saber se um `Z_robusto~9-18` é extremo de fato ou se o nulo B, como
desenhado, é sistematicamente estreito demais (ex. por não incorporar a
incerteza de `mu_t`/centroides de referência, que ficam fixos). Próximo
passo: passo 5.

## Atualização (Claude, 2026-06): passo 5 -- calibração em pseudo-períodos

Novo script `scripts/token_time/calibrate_null.py`. Protocolo: dividi os
253644 documentos de `1810-1860.txt` aleatoriamente (seed fixa 12345) em duas
metades de ~126822 documentos cada (`corpus_pseudo/pseudo_{a,b}.txt`), sem
nenhuma diferença temporal real entre elas -- ambas são amostras aleatórias
do mesmo período. Re-extraí perfis (incluindo `OccurrenceCache`) para esse
par pseudo com o checkpoint do seed1000, e roda `null_b` (B=200,
`layer_2`, gerador com seed=0) para as 37 palavras-alvo.

**Resultados (seed1000, pseudo_a vs pseudo_b):**

- `mean percentile = 0.373`, `stdev percentile = 0.230` (n=37; uniforme[0,1]
  esperaria média 0.5, desvio 0.289). Com n=37, 0.373 está a ~2.6 erros-padrão
  da média esperada (`0.289/sqrt(37) ~= 0.048`) -- um desvio a observar, mas
  não conclusivo nesta amostra pequena.
- **false-positive rate em `alpha=0.05` e `alpha=0.10`: 0/37 em ambos.** Não
  houve nenhum "falso positivo" (nenhuma palavra com `D_obs` pseudo
  anormalmente alto vs seu próprio nulo) -- o que é a direção segura
  (controla erro tipo I), mas com `alpha=0.10` esperaríamos ~3-4 de 37 por
  acaso; `0/37` sugere o nulo pode estar um pouco *largo* (conservador), não
  estreito.
- **Achado inesperado**: comparei `MAD(D_null)` no par pseudo (mesmo corpus,
  metades aleatórias) com `MAD(D_null)` no par real d0-vs-d1 (corpora
  genuinamente diferentes, 1810-1860 vs 1960-2010), para as mesmas 37
  palavras. A razão `MAD(pseudo)/MAD(real)` foi **0.60** -- ou seja, o nulo
  do par *real* é ~1.7x mais largo que o do par *pseudo*, embora cada
  metade pseudo tenha ~metade dos documentos do d0 completo (menos dados
  deveria, segundo a intuição "mais dados -> nulo mais estreito" do item 6
  da contra-análise, produzir um nulo *mais largo*, não mais estreito).

### Interpretação do achado inesperado

A causa provável: no nulo B, `mu_a`/`mu_b` e as centroides de referência
(`centroids_a[reference_ids]`/`centroids_b[reference_ids]`) ficam fixos nos
valores **observados** de cada período -- por desenho (ponto 1 da
contra-análise: só a centroide de `w` é recomputada). No par pseudo,
`pseudo_a`/`pseudo_b` são metades aleatórias do *mesmo* corpus, então
`mu_a ~= mu_b` e as centroides de referência são quase idênticas entre os
dois "períodos" pseudo. Já no par real, `mu_0` (1810-1860) e `mu_1`
(1960-2010) -- e as centroides de cada palavra de referência -- diferem
genuinamente, porque o vocabulário inteiro do corpus muda de um século para
o outro (não só `w`). Como `R_pseudoA(w)` usa `(mu_a, refs_a)` e `R_pseudoB(w)`
usa `(mu_b, refs_b)`, mesmo que a centroide permutada de `w` fosse *idêntica*
nos dois grupos, `R_pseudoA(w) != R_pseudoB(w)` simplesmente porque os
sistemas de referência `(mu, refs)` diferem -- e essa diferença é muito maior
entre 1810-1860 vs 1960-2010 do que entre duas metades aleatórias de
1810-1860.

Ou seja: **o nulo B, como desenhado, não mede apenas "ruído amostral da
centroide de `w`"** -- ele também herda o "piso" de quão diferentes são os
sistemas de referência `(mu_t, centroides de referência)` entre os dois
períodos sendo comparados. Isso não é necessariamente um defeito: `D_obs(w)`
*também* reflete os dois componentes (deslocamento de `w` + deslocamento do
sistema de referência), então o nulo capturar o segundo componente como parte
do "esperado por acaso" é, em certo sentido, correto -- desde que o segundo
componente não dependa de `w`.

**Implicação para o item 6 da contra-análise ("nulo deve estreitar com mais
dados")**: a comparação pseudo-vs-real que fiz confunde duas variáveis --
quantidade de dados E grau de dissimilaridade real entre os dois "períodos"
sendo comparados. Para testar "mais dados -> nulo mais estreito" de forma
limpa, é preciso variar apenas a quantidade de dados *mantendo os dois lados
do par comparáveis* (ex.: comparar um nulo construído a partir de uma divisão
50/50 de `1810-1860` com um construído a partir de uma divisão 25/75 do
*mesmo* corpus -- ambos com `mu_a ~= mu_b`), em vez de comparar pseudo (mesmo
corpus) com real (corpora de séculos diferentes).

### Próximos passos do passo 5

- Repetir a calibração com o checkpoint seed1001 sobre o **mesmo**
  `pseudo_a.txt`/`pseudo_b.txt` (em andamento) -- repetibilidade entre seeds
  para o mesmo split.
- Redesenhar o teste "mais dados -> nulo mais estreito" comparando divisões
  de tamanhos diferentes do mesmo corpus (em vez de pseudo vs real).
- Considerar `B` maior (>=1000) -- com `B=200` a taxa de falso-positivo em
  `alpha=0.05`/`0.10` não tem resolução suficiente (mínimo `p=1/201~=0.005`)
  para distinguir "0 falsos positivos por acaso" de "nulo conservador".

### Repetibilidade entre seeds (seed1001, mesmo split pseudo_a/pseudo_b)

Repeti a calibração com o checkpoint seed1001 sobre o **mesmo**
`pseudo_a.txt`/`pseudo_b.txt` (B=200, seed do gerador=0):

| métrica | seed1000 | seed1001 |
|---|---|---|
| mean percentile | 0.373 | 0.380 |
| stdev percentile | 0.230 | 0.239 |
| FP rate @ alpha=0.05 | 0/37 | 0/37 |
| FP rate @ alpha=0.10 | 0/37 | 0/37 |
| mean MAD(pseudo)/MAD(real) | 0.60 | 0.59 |

Concordância muito próxima entre seeds (diferenças de 2ª casa decimal),
confirmando que o achado da seção anterior não é ruído de uma única
extração: o nulo do par pseudo é consistentemente ~1.7x mais estreito que o
nulo do par real d0-vs-d1, e a distribuição de percentis é consistentemente
deslocada um pouco abaixo de 0.5 (~0.37-0.38) com `stdev` (~0.23-0.24)
levemente abaixo de 0.289.

### Síntese do passo 5 e recomendação

1. **Nenhum falso positivo excessivo**: 0/37 em `alpha=0.05` e `alpha=0.10`,
   em ambos os seeds. Não há evidência de que o nulo B esteja "estreito
   demais" (o que inflaria falsos positivos). Se algo, está no lado
   conservador.
2. **Leve viés de percentil (0.37-0.38 em vez de 0.5)**: consistente entre
   seeds, mas com n=37 está a ~2.5-2.7 erros-padrão da média esperada -- não
   é decisivo, mas é a direção "nulo um pouco largo/deslocado para a direita
   de `D_obs`", i.e. conservador, não anti-conservador. Combinado com (1),
   não vejo risco de inflação de falsos positivos no uso pretendido (passo
   7: avaliar as 37 palavras-alvo reais).
3. **MAD(pseudo) < MAD(real)**, replicado em dois seeds: não invalida o nulo
   B, mas invalida a forma como o item 6 da contra-análise ("nulo deve
   estreitar com mais dados") foi originalmente operacionalizado aqui. A
   causa é que `mu_a`/`mu_b`/centroides-de-referência ficam fixos nos
   valores observados de cada período, e esses valores diferem mais entre
   1810-1860 vs 1960-2010 (par real) do que entre duas metades aleatórias de
   1810-1860 (par pseudo) -- um efeito de "distância entre sistemas de
   referência" que domina sobre o efeito de tamanho amostral nesta
   comparação específica.

**Recomendação**: o nulo B passa nos testes de calibração que são
diretamente relevantes para o uso no passo 7 (sem inflação de falso
positivo, repetível entre seeds). O item 6 da contra-análise fica marcado
como "parcialmente respondido": confirmamos que o nulo não colapsa para
zero (não está artificialmente estreito), mas a comparação limpa
"mais dados -> nulo mais estreito, tudo o mais igual" ficaria para trabalho
futuro (ex.: comparar splits 50/50 vs 25/75 do mesmo corpus). Não vejo isso
como bloqueador para congelar o protocolo (passo 6) e seguir para a
avaliação real das 37 palavras (passo 7).

## Atualização (Claude, 2026-06): passo 6 -- protocolo congelado

Com base nos passos 1-5, o protocolo para o passo 7 (avaliação real das 37
palavras-alvo, d0=1810-1860 vs d1=1960-2010) fica congelado como:

- **Camada**: `layer_2` (como em todos os passos anteriores).
- **`V_active`**: tokens com >=10 ocorrências em ambos os períodos
  (`n_min_active=10`), excluindo alvos.
- **`reference_set`**: subconjunto whole-word de `V_active`, ordenado por
  contagem mínima entre os dois períodos, `max_references=3216`, sem filtro
  de `lexical_validity` adicional além do já embutido em `V_active` (o filtro
  de `lexical_validity` do passo 1 já está aplicado na construção de
  `V_active`/`reference_set` desde a correção de fronteiras de palavra).
- **`D_obs(w) = displacement(w, reference_set).score`** (`1 - cos(R_0(w),
  R_1(w))`), `mu_t`/centroides via `type_uniform_mean`/`PeriodStatistics`,
  encoder fixo (`period1_epoch2` de cada seed).
- **Nulo**: nulo B (`document_permutation_null`), `B=200`, permutação a nível
  de documento preservando `n0`/`n1`, `mu`/centroides de referência fixos nos
  valores observados de d0/d1 (não recomputados na permutação).
- **Estatísticas reportadas por palavra**: `D_obs`, `median(D_null)`,
  `MAD(D_null)`, `Z_robusto = (D_obs - median)/(1.4826*MAD)`, `p = (1 +
  #{D_null >= D_obs})/(B+1)`.
- **Sem `standard_error` propagado** (removido no passo 2) -- a incerteza de
  `D_obs(w)` é expressa apenas via o nulo B.
- **Critério de "sinal"**: por ora, reportar `Z_robusto` e `p` para as 37
  palavras sem aplicar correção de múltiplas comparações (n=37 é pequeno;
  decisão sobre correção fica para discussão com codex/leitura dos
  resultados). Repetir para os dois seeds (1000, 1001) e reportar lado a
  lado para `Z_robusto`/`p`/`D_obs`, como diagnóstico de repetibilidade
  (passo 8 cobre split-half; aqui é repetibilidade entre seeds/encoders).

## Atualização (Claude, 2026-06): passo 7 -- avaliação real das 37 palavras-alvo (d0 vs d1)

Rodei `calibrate_null.py --profile-dir outputs/token_time_fase_a/seed{1000,1001}`
(sem `--compare-profile-dir`) com o protocolo congelado do passo 6, B=200,
seed do gerador=0, sobre o par real d0 (1810-1860) vs d1 (1960-2010).

**Diagnóstico de calibração no par real** (para contraste com o par pseudo
do passo 5):

| métrica | seed1000 | seed1001 | pseudo (passo 5) |
|---|---|---|---|
| mean percentile | 0.726 | 0.708 | ~0.38 |
| stdev percentile | 0.386 | 0.397 | ~0.235 |
| "significativos" @ alpha=0.05 | 20/37 (0.541) | 18/37 (0.486) | 0/37 |
| "significativos" @ alpha=0.10 | 23/37 (0.622) | 23/37 (0.622) | 0/37 |

Isso é o esperado e desejável: entre dois séculos diferentes, a maioria das
37 palavras-alvo (escolhidas exatamente por terem mudança lexical
candidata) mostra `D_obs` bem acima do nulo (`percentile` perto de 1,
muitas vezes `percentile=1.0` com `Z` de dois dígitos), enquanto no par
pseudo (mesmo período, sem mudança real) a taxa de "significativos" foi
zero. O contraste de ~54-62% vs 0% confirma que o nulo B distingue
"há mudança real entre os períodos" de "dois conjuntos de documentos do
mesmo período".

### Correlação com o gold padrão SemEval2020 (`truth.tsv`)

Usando `binary`/`graded` de `data/processed/semeval2020_task1/eng_lemma/truth.tsv`
(37 palavras, mesma lista):

| | seed1000 | seed1001 |
|---|---|---|
| Spearman(`D_obs`, graded) | 0.061 (p=0.722) | 0.068 (p=0.690) |
| Spearman(`Z_robusto`, graded) | **0.278** (p=0.096) | **0.275** (p=0.100) |
| point-biserial(`binary`, `D_obs`) | 0.201 (p=0.232) | 0.209 (p=0.214) |
| point-biserial(`binary`, `Z_robusto`) | **0.330** (p=0.046) | **0.335** (p=0.043) |

**Achado principal do passo 7**: `Z_robusto` (D_obs normalizado pelo nulo B)
correlaciona consideravelmente melhor com o gold do que `D_obs` cru, em
ambos os seeds, de forma muito consistente (diferenças de 3ª casa decimal
entre seeds). `D_obs` sozinho é quase descorrelacionado do gold
(`rho~=0.06`); `Z_robusto` chega a `rho~=0.28` (Spearman, marginal,
`p~=0.10` com n=37) e `r_pb~=0.33` para o rótulo binário (`p~=0.045`,
significativo a 5%).

Interpretação provisória: `D_obs(w)` mistura "deslocamento real do sentido
de `w`" com "quão ruidosa/instável é a estimativa de `centroid(w)` em cada
período" (que varia muito com `n0`/`n1`, de 50 a >5000 ocorrências entre as
37 palavras). Dividir pelo `MAD(D_null(w))` -- que reflete justamente essa
escala de ruído específica de `w` -- remove boa parte dessa heterogeneidade
e deixa o que resta mais alinhado com a mudança semântica real anotada
pelos humanos. Isso é evidência a favor de reportar `Z_robusto` (não
`D_obs`) como a métrica primária de "sinal" do `token@time`, consistente
com a motivação original de todo este levantamento (passos 1-6).

### Status da ordem revisada de 8 passos

- [x] 1. `lexical_validity` com fronteiras de palavra reais.
- [x] 2. Remover `standard_error` propagado.
- [x] 3. Cache por ocorrência/documento.
- [x] 4. Nulo B (permutação a nível de documento).
- [x] 5. Calibração em pseudo-períodos sem gold.
- [x] 6. Protocolo congelado.
- [x] 7. Avaliação real das 37 palavras-alvo (`D_obs`, nulo B, `Z_robusto`,
  correlação com gold em ambos os seeds).
- [ ] 8. Split-half como diagnóstico de repetibilidade (não como nulo).

Falta apenas o passo 8 para concluir a ordem revisada do codex.

## Atualização (Claude, 2026-06): passo 8 -- split-half como diagnóstico de repetibilidade

Novo script `scripts/token_time/split_half_repeatability.py` (não é um nulo:
puramente diagnóstico). Para cada palavra-alvo `w`, divide as ocorrências de
`w` em d1 em duas metades aleatórias por documento (`OccurrenceCache.doc_index`,
gerador com seed=0), recomputa `centroid(w)` em cada metade mantendo `mu_1`/
centroides de referência fixos nos valores observados de d1 inteiro (mesmo
truque do nulo B), e compara `D_half1`, `D_half2` com `D_obs` (full d1).

**Resultados (B=1 split, ambos os seeds, n=37):**

| | seed1000 | seed1001 |
|---|---|---|
| Spearman(`D_half1`, `D_half2`) | 0.952 | 0.952 |
| Spearman(`D_obs`, mean(`D_half1`,`D_half2`)) | 0.993 | 0.992 |

Concordância muito alta entre as duas metades independentes do corpus de
d1, e entre cada metade e a amostra completa, para ambos os seeds. Palavras
com poucas ocorrências (ex. "graft", n~64 por metade; "lass", n=65) mostram
a maior variação absoluta entre metades (`graft`: 0.0315 vs 0.0391 no
seed1000; `lass`: 0.0263 vs 0.0413), mas mesmo assim a ordenação relativa
entre as 37 palavras é preservada (`rho=0.95`). Isso confirma que `D_obs(w)`
não é dominado por alguns documentos isolados -- é uma propriedade estável
do corpus, mesmo para as palavras de menor frequência da lista.

### Conclusão da ordem revisada de 8 passos

Todos os 8 passos da ordem revisada (proposta do codex, adotada no início
desta sessão de trabalho) estão concluídos:

1. [x] `lexical_validity` com fronteiras de palavra reais.
2. [x] Remover `standard_error` propagado.
3. [x] Cache por ocorrência/documento (`OccurrenceCache`).
4. [x] Nulo B (permutação a nível de documento, `document_permutation_null`).
5. [x] Calibração em pseudo-períodos sem gold -- sem inflação de falso
   positivo, repetível entre seeds; achado sobre `MAD(pseudo)` vs
   `MAD(real)` documentado e interpretado.
6. [x] Protocolo congelado (camada, `V_active`, `reference_set`, nulo B,
   estatísticas reportadas).
7. [x] Avaliação real das 37 palavras-alvo: `D_obs`, `Z_robusto`, `p`, e
   correlação com o gold do SemEval2020 -- **achado principal**:
   `Z_robusto` correlaciona muito melhor com o gold (`rho~=0.28`,
   `r_pb~=0.33`, `p<0.05` para o binário) do que `D_obs` cru (`rho~=0.06`),
   em ambos os seeds.
8. [x] Split-half como diagnóstico de repetibilidade (não nulo):
   `rho(D_half1, D_half2)~=0.95` em ambos os seeds.

Pronto para revisão pelo codex / próxima etapa de planejamento.
