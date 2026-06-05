# Parecer: Deslocamento Relacional e Mudança Semântica Estrutural
**Data:** 2026-06-04  
**Escopo:** Avaliação conceitual, metodológica e experimental da reorientação para interpretação estrutural multiescala  
**Baseado em:** `relational_change_current_plan.md`, `relational_metrics.py`, `relational.py`, `run_relational_continual_sanity.py`, `summarize_relational_sensitivity.py`, outputs de `relational_sensitivity*`

---

## 1. Resumo executivo

A reorientação proposta é cientificamente defensável, mas a evidência disponível não distingue ainda entre suas duas interpretações possíveis — e essa distinção é o coração da questão. O risco de racionalização é real e precisa ser endereçado antes de qualquer comprometimento com a nova framing.

**O que está sólido:** `delta_rel` como vetor descritivo é correto, invariante às transformações erradas e interpretável. A separação entre camada descritiva e camada de significância estrutural é uma boa arquitetura conceitual.

**O que está em aberto:** a distinção entre microvariação e reorganização estrutural é teoricamente defensável, mas os dados atuais não permitem saber se o limiar de detecção (~alpha=0.75) é uma propriedade do fenômeno semântico ou uma limitação do método. Sem experimentos que distingam persistência coordenada de mudança grande mas incoerente, a interpretação continua ambígua.

**Veredito:** não adotar a nova framing como posição principal ainda. Projetar e rodar o experimento mínimo decisivo descrito na Seção 9 antes de qualquer comprometimento.

---

## 2. Compreensão do que estamos propondo

A proposta tem duas camadas separadas:

**Camada 1 (já implementada):** `delta_rel(w, a, b) = r_b(w) - r_a(w)` — vetor que registra aproximações e afastamentos de cada palavra em relação a todas as outras, medido sobre distribuições de contexto (Jensen-Shannon). Preserva direção e é invariante a reorganizações globais do espaço do Transformer.

**Camada 2 (em proposição):** uma função posterior `S(w, a, b)` que distingue flutuação normal de reorganização estruturalmente relevante. Os componentes propostos incluem magnitude normalizada, substituição de vizinhança, coerência direcional, persistência temporal e escala relativa ao ruído esperado.

A motivação empírica é o dado de que o método detecta mudanças com confiança em alpha≥0.75, mas não em alpha=0.25, e a proposta reinterpreta alpha=0.25 como "microvariação" em vez de "mudança semântica não detectada". A questão é se essa reinterpretação é descoberta ou convenção.

---

## 3. Estamos vendo o fenômeno na escala correta?

**Parcialmente sim, mas a resposta correta ainda é "não sabemos".**

A distinção entre escala de fenômeno e limite de resolução do método é uma das questões mais difíceis na ciência empírica. O fato de que o método não detecta mudanças pequenas não implica que essas mudanças não sejam semanticamente relevantes — mas também não implica que sejam.

**Argumento a favor da nova framing:** em outras ciências, existem distinções análogas bem estabelecidas entre flutuação e transição estrutural. Em física estatística, microvariações térmicas são ruído; mudanças de fase são estruturais. Em biologia de populações, deriva genética é flutuação; seleção dirigida é mudança estrutural. A distinção não é convenção — ela tem correlatos observacionais (magnitude relativa, persistência, coordenação entre dimensões). Há razão para acreditar que semântica tem análogo.

**Argumento contra a nova framing:** as trajetórias sintéticas com alpha=0.25 não são "flutuações aleatórias" — são mudanças semânticas reais e sistemáticas plantadas pelo gerador, que o método simplesmente não detecta. O método não tem resolução suficiente para vê-las, mas isso não as torna irrelevantes. Existem mudanças semânticas historicamente importantes que foram graduais e persistentes — a expansão do campo semântico de "computador" de 1950 a 2000, por exemplo, aconteceu por acúmulo de pequenas associações novas ao longo de décadas. Um método que declara esse processo "microvariação estruturalmente estável" cometeria um erro histórico real.

**Diagnóstico:** a questão está mal colocada. A pergunta não deveria ser "mudanças pequenas são semânticas ou são ruído?" mas "mudanças pequenas E PERSISTENTES são detectáveis, e mudanças grandes E INCOERENTES são ignoradas?". Essa pergunta distingue estrutura de magnitude — e é a questão que os experimentos atuais não respondem.

---

## 4. A distinção entre microvariação e reorganização estrutural é válida?

**Sim, em teoria. Mas precisa de definição independente dos resultados.**

A distinção é válida se e somente se puder ser definida formalmente sem referência ao limite de detecção do método. Uma definição circular — "reorganização estrutural é o que o método detecta" — não tem valor científico.

**Definição proposta independente dos resultados:**

> Reorganização estrutural é uma mudança no perfil relacional que satisfaz simultaneamente três critérios: (1) **magnitude** supera a variação nula esperada para aquela palavra específica; (2) **persistência** — a mudança não reverte nos períodos seguintes mas se mantém ou se intensifica; (3) **coerência** — as dimensões que mudam formam um conjunto semanticamente interpretável (e.g., os k vizinhos mais próximos são substituídos por outro conjunto coeso), não um padrão aleatório de dimensões isoladas.

Essa definição é independente dos resultados porque: (1) usa o nulo por palavra, não o limiar global; (2) usa propriedades da trajetória temporal, não apenas do snapshot; (3) exige estrutura semântica, não apenas magnitude.

**O problema com os critérios propostos no prompt:** magnitude normalizada + substituição de vizinhança + coerência direcional + persistência + escala são todos componentes legítimos, mas tratá-los como uma função `F` sem especificar como agregá-los é adiar o problema. O veredito científico vai depender de qual combinação se escolhe — e qualquer combinação calibrada nos dados atuais é circular.

**Recomendação:** definir os três critérios (magnitude, persistência, coerência) como condições SEPARADAS, cada uma com limiar independente derivado do nulo ressampleado. Uma palavra passa a ser "estruturalmente alterada" se e somente se satisfizer todos os três. Isso é mais honesto que uma métrica contínua composta.

---

## 5. Risco de racionalização ou mudança oportunista do objetivo

**O risco é real e precisa ser nomeado diretamente.**

A sequência observável é:
1. Método não detecta mudanças em alpha=0.25.
2. Proposta: "mudanças em alpha=0.25 são microvariação, não mudança semântica relevante."
3. Resultado: método agora "está correto" por definição.

Esse é o padrão clássico de HARKing (Hypothesizing After Results are Known) aplicado à definição do objeto, não da hipótese. É mais difícil de detectar que HARKing de resultado, mas igualmente problemático.

**Como distinguir HARKing de descoberta genuína:** a definição de reorganização estrutural deve fazer ao menos uma previsão nova que os dados atuais não testaram. Se a definição não prediz nada além do que já foi observado, ela não acrescenta poder explicativo — apenas reformula o mesmo resultado.

**Uma previsão nova que a framing estrutural deve fazer:**
> Mudanças pequenas mas PERSISTENTES e COORDENADAS (alpha=0.25 durante todos os 10 períodos, sem flutuação) devem ser detectáveis como reorganização estrutural mesmo que cada passo individual fique abaixo do limiar de magnitude. Se o método não detectar isso, o limiar é de método, não de fenômeno.

Esse experimento não foi rodado. Se a nova framing é uma descoberta e não racionalização, deve prever o resultado deste experimento antes de rodá-lo.

**Resposta a cada uma das 7 questões do prompt:**

1. A distinção é teoricamente defensável independentemente dos resultados — sim, mas com a definição independente descrita acima, não com uma definição circular.
2. Estamos provavelmente descobrindo a escala correta E racionalizando uma limitação — ambos, em proporções que os experimentos atuais não permitem separar.
3. Sim, existem mudanças semânticas pequenas, graduais e historicamente importantes. A expansão de campos semânticos tecnológicos (computador, internet, streaming) acontece por acúmulo de pequenas associações. Ignorar alpha=0.25 pode apagar exatamente o fenômeno que mais importa estudar historicamente.
4. Sim, mudança estrutural pode emergir por acúmulo de pequenos deslocamentos persistentes. Esse é o caso mais crítico e é o que precisa ser testado.
5. Para evitar que a exigência de grande magnitude favoreça mudanças abruptas: a condição de persistência é o que resolve isso. Uma mudança abrupta grande que não persiste ≠ reorganização estrutural.
6. Sim, precisamos preservar ambos. O vetor `delta_rel` é a camada fina. `S(w)` é a interpretação estrutural. Não substituir uma pela outra.
7. Para evitar movimentação oportunista: pré-registrar o experimento com mudanças pequenas e persistentes antes de rodá-lo, com critério de sucesso definido explicitamente (e.g., "se detecção > 50% com alpha=0.25 persistente, a framing estrutural é confirmada").

---

## 6. Formulação matemática recomendada

### 6.1 Manter: o vetor `delta_rel`

```
delta_rel(w, a, b)[v] = r_b(w)[v] - r_a(w)[v]
                      = [1 - JS(q_b(w), q_b(v))/log2] - [1 - JS(q_a(w), q_a(v))/log2]
```

Correto, invariante a reorganizações globais, interpretável. Não mudar.

### 6.2 Magnitude normalizada por palavra

Em vez de um limiar global, usar o percentil 95 do nulo ressampleado por faixa de distribuição inicial `p_0`:

```
z(w, a, b) = magnitude(w, a, b) / p95_null(p0(w))
```

Os dados já mostram que a heterogeneidade do nulo é forte: sujeitos com p_0 ∈ [0.50, 0.75) têm nulo médio 5x maior que sujeitos com p_0 ∈ [0.75, 1.00]. Um limiar global é conservador para a maioria e insuficiente para a minoria crítica. Em corpus real, substituir faixas de p_0 por faixas de entropia preditiva ou frequência observável.

### 6.3 Persistência

```
persistente(w, a, b) = True se
    sign(delta_rel(w, a, a+1)) == sign(delta_rel(w, a, b)) para t > a+1
    [ou: correlação de Spearman entre magnitude por período e índice de período > 0]
```

Uma palavra que muda em t=3 e reverte em t=4 não é reorganização estrutural. Uma palavra que cresce monotonicamente ao longo dos 10 períodos é.

### 6.4 Coerência direcional

```
coerencia(w, a, b) = concentração das dimensões alteradas
```

Possíveis operacionalizações:
- Fração das dimensões com |delta| > limiar que mudam na mesma direção
- Substituição dos top-k vizinhos por um conjunto igualmente coeso (Jaccard do novo grupo com vocabulário de referência)
- Entropy do vetor delta normalizado — alta entropia = mudança espalhada, baixa entropia = mudança concentrada

### 6.5 Julgamento estrutural como interseção, não como função contínua

```
estrutural(w, a, b) = z(w) > 1.0 AND persistente(w) AND coerente(w)
```

Isso é mais honesto que uma métrica contínua composta: cada critério tem interpretação independente e pode ser falsificado separadamente.

---

## 7. Como preservar o vetor relacional e medir significância estrutural

A arquitetura em duas camadas é a correta. A diferença entre as camadas é:

| Camada | Produto | Para que serve |
|---|---|---|
| Descritiva | `delta_rel(w)` como vetor | Explicar direção: de quem se afastou, de quem se aproximou |
| Estrutural | `S(w)` como julgamento | Decidir se a mudança importa semanticamente |

A camada estrutural não substitui a camada descritiva — ela usa a camada descritiva como insumo. Um paper que apresente apenas `S(w)` sem `delta_rel(w)` perderia o elemento mais interpretável da proposta.

**Erro a evitar:** reportar apenas "palavras detectadas como estruturalmente alteradas" sem reportar em que direção mudaram. A direção é o que transforma o método de detector binário em ferramenta de análise histórica.

**O que o cosseno direcional com o oráculo mede agora:** no sintético, mede se a direção do delta_rel aprendido aponta na direção certa. Isso é combinação de magnitude + direção. Um delta_rel muito pequeno pode ter cosseno alto com o oráculo mas magnitude abaixo do limiar — essa é a distinção entre "direção correta mas magnitude insuficiente" e "mudança detectada". Os dados já registram isso, mas a framing atual não articula bem.

---

## 8. Polissemia e mudanças graduais

### 8.1 O que o perfil relacional agregado consegue capturar

| Tipo de mudança | Capturo com um perfil? | Observação |
|---|---|---|
| Substituição de sentido (Drift) | Sim | O centro migra monotonicamente |
| Mudança abrupta (Abrupt) | Sim | O centro salta |
| Coexistência de sentidos (Bifurcating) | Parcialmente | O centro fica no meio — detecta bimodalidade mas não a separa |
| Alteração de frequência relativa de sentidos | Não | Requer distribuição de perfis, não ponto central |
| Reorganização dentro de um único sentido | Não | Requer comparação de subgrupos |

A taxa de detecção de Bifurcating (63.3% em alpha=0.75) versus Drift/Abrupt (90%) confirma a limitação teórica: um único perfil por palavra-período apaga a estrutura bimodal que a coexistência de sentidos produz.

### 8.2 O que precisaríamos para polissemia completa

Para representar coexistência de sentidos, o mínimo necessário é uma **distribuição de perfis** por palavra-período — ou seja, em vez de `r_t(w)` como ponto, `R_t(w)` como distribuição sobre pontos. Isso requer:

- Agrupamento das ocorrências de `w` em `t` por sentido (não supervisionado, e.g., GMM sobre os estados ocultos)
- Um perfil relacional por sentido, não por palavra

Isso está fora do escopo do método atual. A recomendação é documentar explicitamente que o método captura mudança de sentido dominante mas não bimodalidade de sentidos coexistentes — isso é uma limitação honesta, não uma falha.

### 8.3 Mudanças graduais acumulativas

Esse é o caso mais importante e mais não-testado. O argumento para a framing estrutural só se sustenta se pequenas mudanças graduais e persistentes forem detectáveis — caso contrário, o método é sistematicamente cego ao tipo mais comum de mudança semântica real (deriva tecnológica, deriva cultural, expansão de campo semântico).

O teste necessário: gerar trajetória com alpha=0.10 por período, mas completamente monotônica — sem ruído estocástico no sentido da mudança. A mudança acumulada de t0 a t9 seria 10×0.10 = 1.00 em termos relativos. Se o método detectar essa trajetória como estrutural, ele é sensível a acumulação. Se não, está medindo magnitude de passo individual, não estrutura temporal.

---

## 9. Experimentos de falsificação recomendados

### [INDISPENSÁVEL] Experimento P: persistência versus magnitude isolada

Gerar quatro condições com o mesmo orçamento computacional:

```
P1: trajetória monotônica small, alpha=0.10, persistente por 10 períodos
    → mudança acumulada t0→t9 equivalente a alpha=1.0

P2: trajetória aleatória large, alpha=1.00 por cada período, direções independentes
    → magnitude alta mas incoerente, sem acumulação direcional

P3: trajetória gradual medium, alpha=0.25 por 10 períodos, monotônica
    → o que o corpus COHA frequentemente produz

P4: trajetória intermitente, alpha=1.00 por 2 períodos, depois reverte
    → mudança abrupta não persistente
```

Critério de sucesso da framing estrutural: P1 deve ser detectado, P2 não deve ser detectado.
Critério de falsificação: se P2 for detectado tanto quanto P1, o método mede magnitude, não estrutura.

**Classificação:** indispensável antes de qualquer comprometimento com a framing estrutural.

### [EXPERIMENTO PRINCIPAL] Experimento N: calibração do nulo por palavra

Construir o nulo ressampleado com resolução suficiente para estimar o percentil 95 por faixa de p_0 (ou por entropia preditiva, para uso em corpus real). Verificar se a calibração por palavra reduz os falsos positivos na classe Stable sem reduzir a detecção em Drift/Abrupt.

### [ABLAÇÃO] Experimento C: coerência versus magnitude

Para palavras detectadas acima do limiar de magnitude, medir a fração das dimensões de delta_rel que mudam na mesma direção. Uma mudança estrutural genuína deve ter alta coerência direcional. Uma mudança por ruído de treinamento deve ter baixa coerência. Verificar se coerência e magnitude se correlacionam ou são independentes.

### [ABLAÇÃO] Experimento K: Rank-Biased Overlap versus Jaccard

Substituir Jaccard pelos top-k vizinhos por RBO (Rank-Biased Overlap), que pondera os vizinhos mais próximos mais fortemente. Verificar se muda o limiar de detecção para Bifurcating — se RBO for mais sensível a bimodalidade, justifica a adição da métrica.

### [ANÁLISE POSTERIOR] Experimento T: trajetória temporal do delta

Para palavras acima do limiar, plotar a sequência de magnitudes consecutivas ao longo dos períodos. Se a sequência for monotonicamente crescente, a mudança é acumulativa. Se tiver pico e reversão, é transitória. Classificar palavras por forma de trajetória relacional sem usar rótulos sintéticos.

### [DESNECESSÁRIO / PERIGOSO] Optimal transport entre perfis

Medir deslocamento via OT entre distribuições relacionais seria matematicamente refinado, mas: (a) requer estimativa de densidade no espaço relacional, que com 40 sujeitos é muito esparso; (b) o custo computacional é proibitivo para corpus real; (c) o que se ganha em relação ao cosseno direcional não está claro. Reservar para investigação futura se os resultados básicos forem robustos.

---

## 10. Auditoria do código e dos resultados atuais

### O que `delta_rel` preserva e o que não preserva

`relational_delta` em `relational.py` (linha 22–25) retorna a diferença entre matrizes de similaridade. Correto.

**Preserva:** direção de aproximação/afastamento, invariância a rotação ortogonal (confirmada em teste), interpretabilidade por dimensão.

**Não preserva / não mede ainda:**
- Persistência temporal: apenas snapshots de dois períodos, sem memória de trajetória
- Coerência direcional: a função retorna uma matriz, mas as funções de sumarização colapsam para média absoluta — perdendo a estrutura direcional
- Heterogeneidade do nulo por palavra: o limiar de detecção em `summarize_relational_sensitivity.py` é global (percentil 95 sobre todos os sujeitos do nulo), quando os dados mostram forte heterogeneidade por faixa de p_0

### O que as métricas atuais medem

| Métrica | O que mede | Limites |
|---|---|---|
| `mean_abs_similarity_delta` | magnitude média da mudança | Não captura direção nem coerência |
| `jaccard_change` | fração dos top-k vizinhos que mudaram | Não captura magnitude nem direção |
| `spearman_change` | quanto o ranking relacional mudou | Não captura direção |
| `oracle_direction_advantage` | quanto a direção real supera a do placebo | Requer oráculo externo |

**Gap principal:** nenhuma métrica atual mede persistência ou coerência. As métricas de magnitude (mean_abs_delta) e de vizinhança (jaccard, spearman) são agnósticas à estrutura temporal da mudança. O método detecta "quanto mudou" e "se mudou na direção certa", mas não "se a mudança foi estruturalmente coerente e persistente".

### Componentes a criar

1. **`persistent_change_score(profiles: list[Tensor]) -> Tensor`** — recebe a sequência de perfis ao longo dos períodos e retorna, para cada sujeito, o coeficiente de correlação de Spearman entre o índice de período e a magnitude acumulada. Alta correlação = mudança monotônica = estrutural.

2. **`directional_coherence(delta: Tensor) -> Tensor`** — para cada linha (sujeito) do delta_rel, mede a concentração das dimensões que mudaram: entropia do vetor normalizado, ou fração de dimensões que mudaram na mesma direção geral.

3. **`word_specific_null_threshold(null_magnitudes: Tensor, p0: Tensor) -> Tensor`** — para cada sujeito, interpola o percentil 95 do nulo ressampleado baseado em sua distribuição inicial p_0, usando regressão local ou faixas predefinidas.

### Deve-se alterar o `relational_change_current_plan.md` agora?

Não ainda. O plano atual está correto em termos do que foi executado. Antes de atualizar o plano com a framing estrutural, é necessário ter o resultado do Experimento P (persistência versus magnitude) para saber se a distinção é descoberta ou racionalização.

---

## 11. Mudanças necessárias antes de prosseguir

| Prioridade | Ação | Classificação |
|---|---|---|
| 1 | Rodar Experimento P (persistência monotônica vs. magnitude incoerente) | Indispensável |
| 2 | Pré-registrar o critério de sucesso e falsificação do Experimento P antes de rodar | Indispensável |
| 3 | Implementar `word_specific_null_threshold` com interpolação por p_0 | Indispensável |
| 4 | Implementar `persistent_change_score` sobre sequências de perfis | Experimento principal |
| 5 | Medir coerência direcional nas palavras já detectadas acima do limiar | Experimento principal |
| 6 | Verificar se Bifurcating melhora com RBO versus Jaccard | Ablação |
| 7 | Plotar trajetórias temporais de magnitude para palavras detectadas | Análise posterior |
| 8 | Adotar framing estrutural no `relational_change_current_plan.md` | Aguardar resultado do exp. P |

---

## 12. Veredito

### Estamos indo na direção científica correta?

**Sim**, com uma ressalva importante: a distinção entre microvariação e reorganização estrutural é o lugar certo para olhar, mas ainda não temos como saber se o método está na escala correta ou na escala da sua própria limitação. A direção está certa; o ritmo de comprometimento com a nova framing está rápido demais.

### O resultado atual demonstra deslocamento semântico relacional ou sensibilidade a grandes mudanças sintéticas?

**Ambos, mas não se sabe em que proporção.** A vantagem direcional de +0.31 a +0.41 em alpha=1.00 com 3 seeds é evidência de deslocamento relacional real. A taxa de 5% em alpha=0.25 é evidência de limitação real. O que não sabemos: é o limite de alpha=0.75 uma propriedade do método (resolução insuficiente) ou do fenômeno (mudanças abaixo disso são flutuação)?

### Devemos tratar a baixa detecção de perturbações pequenas como propriedade desejável, limitação ou questão aberta?

**Questão aberta** — e deve permanecer aberta até o Experimento P. Declarar como propriedade desejável antes do experimento é racionalização. Declarar como limitação antes do experimento é desistir prematuramente de uma hipótese que pode ser verdadeira.

### Qual é o menor experimento decisivo?

Rodar Experimento P: gerar trajetória monotônica pequena (alpha=0.10 por período, sem flutuação estocástica na direção da mudança) e verificar se o método a detecta como estrutural usando os critérios de magnitude normalizada por palavra + persistência. Se a mudança acumulada for detectada, o método é sensível a estrutura temporal. Se não for, o limiar é de método.

Este experimento cabe em dois dias de implementação e pode ser rodado no corpus sintético existente com ajuste mínimo ao gerador de trajetórias.

### O que provavelmente ainda não estamos enxergando?

Dois pontos que a análise atual não aborda:

**1. A heterogeneidade do nulo é mais grave do que parece.** A correlação de -0.769 entre p_0 e magnitude nula significa que as palavras que mais mudam (as com p_0 próximo a 0.5, i.e., Bifurcating e Stable no intervalo médio) são exatamente as que têm o maior ruído relacional natural. Um limiar global favorece palavras extremas (Drift, Abrupt) e desfavorece palavras intermediárias (Bifurcating) — o que pode explicar por que Bifurcating tem taxa de detecção de 63% enquanto Drift tem 90%. Isso não é necessariamente um problema do método; pode ser uma propriedade real (palavras em transição têm mais ruído relacional). Mas precisa ser investigado antes de concluir que Bifurcating é "mais difícil".

**2. O oráculo 2D ([p_n1, 1-p_n1]) é uma representação muito simples da mudança.** Ele mede apenas se dois sujeitos estão no mesmo "lado" do eixo N1/N2 em cada período. Num corpus real, o espaço semântico é multidimensional. O método pode ser muito mais poderoso (ou muito mais fraco) em espaços de alta dimensão do que o benchmark sintético sugere. O próximo benchmark sintético deveria ter pelo menos 2 eixos semânticos independentes, para testar se o método consegue distinguir mudança num eixo sem ser confundido pelo outro.

---

## 13. Resposta ao codex — o que o parecer acima errou

**Adicionado em 2026-06-04, após discussão com o autor.**

O parecer das Seções 1–12 avaliou a metodologia de avaliação — benchmarks sintéticos, limiares de detecção, métricas de magnitude e direção, risco de racionalização dos critérios. Tudo isso é relevante para a validade experimental. Mas não é o objeto central da pesquisa.

**O objeto central é a arquitetura Timeformer**: um único Transformer padrão treinado continuamente e em ordem cronológica, sem receber qualquer sinal explícito de período, cujos checkpoints sucessivos registram passivamente a evolução semântica do corpus. A pergunta que o paper responde é: *um Transformer treinado dessa forma captura mudança semântica de maneira mensurável e interpretável?*

O probe preditivo, o `delta_rel`, o Jensen-Shannon, o oráculo sintético — todos são ferramentas de avaliação da arquitetura, não a arquitetura em si. O parecer anterior inverteu figura e fundo: tratou a metodologia de avaliação como a contribuição principal e a arquitetura como pressuposto.

**O que deveria ter sido avaliado:**

**1. A aposta arquitetural é teoricamente defensável?**

Sim. A hipótese é que o processo de otimização contínua transforma o Transformer num registro passivo de mudança semântica: quando o corpus muda — quando os contextos de uma palavra se reorganizam entre períodos — os pesos ajustam para minimizar a loss de predição nos novos contextos, e esse ajuste *é* o deslocamento semântico. Não é necessário um módulo temporal separado; a temporalidade entra pela ordem dos dados.

Isso é uma hipótese empírica com fundamentação teórica razoável. Não é óbvia — a hipótese alternativa é que o otimizador simplesmente explora os dados de cada período da forma mais eficiente possível, e o rastro que deixa nos pesos é dominado por forgetting e pela distribuição de frequência, não por semântica. O valor do benchmark sintético é justamente decidir entre essas duas hipóteses em condições controladas.

**2. O que o Timeformer oferece que alternativas não oferecem?**

A alternativa mais direta é treinar modelos independentes por período e alinhar os espaços por rotação ortogonal (Procrustes). Esse método resolve comparabilidade explicitamente, mas perde continuidade: cada modelo começa do zero, sem memória de períodos anteriores, e palavras que não mudaram podem aparecer deslocadas simplesmente por variância de inicialização.

O Timeformer resolve continuidade nativamente: theta_1 é inicializado por theta_0, então palavras estáveis permanecem próximas das suas representações anteriores sem nenhum mecanismo de alinhamento explícito. O custo é que forgetting e deriva de domínio do corpus podem contaminar o sinal. Essa é a troca que a arquitetura faz, e ela precisa ser avaliada empiricamente no COHA — não apenas no sintético.

A contribuição não é apenas detecção de mudança (que Procrustes já faz). É a possibilidade de ler diretamente da arquitetura *em relação a quais outros conceitos* uma palavra se moveu, sem alinhamento pós-hoc. O `delta_rel` com JSD é interpretável por construção: cada dimensão corresponde a um outro conceito, e o sinal diz se a distribuição de contextos se aproximou ou se afastou daquele conceito. Esse nível de interpretabilidade não é oferecido por Procrustes.

**3. Onde a arquitetura falha para corpus real que o benchmark sintético não revelou:**

O sintético tem corpus de tamanho igual em todos os períodos. O COHA não: a década de 1810 tem ~1M de tokens, a de 1990 tem ~40M. Isso significa que o modelo absorve sinal de mudança de forma desigual por período — um período com corpus maior produz mais atualizações de gradiente, e portanto mais potencial de deslocamento relacional, independentemente de a mudança semântica real ser maior ou menor. A arquitetura atual não calibra por tamanho de corpus.

O sintético tem vocabulário fixo em todos os períodos. No COHA, palavras entram e saem do vocabulário. O `delta_rel` de uma palavra que não aparecia em theta_0 e passa a aparecer em theta_5 não é comparável ao de uma palavra que estava presente desde o início.

O sintético tem um único eixo semântico (N1 vs. N2). Em corpus real, uma palavra pode mudar em relação a um campo semântico mas se estabilizar em relação a outro simultaneamente. O `delta_rel` agrega isso num único vetor por palavra-período, que pode cancelar mudanças em direções opostas. Isso não é um problema da implementação atual — é uma limitação estrutural da formulação que precisa ser documentada.

**4. O que o Experimento P (persistência vs. magnitude) deveria responder sobre a arquitetura, não sobre a métrica:**

O parecer original enquadrou o Experimento P como teste de se a framing de "reorganização estrutural" é racionalização ou descoberta. O enquadramento correto é diferente: o Experimento P testa se a *arquitetura Timeformer* é sensível à estrutura temporal da mudança ou apenas à sua magnitude pontual. Isso é uma propriedade arquitetural, não métrica. Se o Timeformer detecta mudanças monotônicas pequenas mas não detecta mudanças grandes e incoerentes, é porque os checkpoints acumulam sinal temporal — o que justifica a arquitetura. Se detecta apenas magnitude, o treinamento contínuo não está acrescentando nada além do que um único modelo treinado em todos os dados faria.

**Em resumo:** o parecer anterior era tecnicamente correto sobre o que avaliou, mas avaliou o instrumento em vez do objeto. A pergunta central — *é o Timeformer uma arquitetura válida para capturar mudança semântica em corpus real?* — não foi respondida. A resposta provisória, baseada nos dados sintéticos existentes, é: sim, a arquitetura captura mudança quando o sinal é suficientemente forte, e o treinamento contínuo sem sinal temporal é uma simplificação arquitetural legítima com troca explícita (continuidade nativa vs. risco de contaminação por forgetting). A validação definitiva requer o COHA, e a preparação para o COHA requer resolver os três problemas identificados acima: desigualdade de tamanho de corpus por período, cobertura de vocabulário variável, e espaço semântico multidimensional.
