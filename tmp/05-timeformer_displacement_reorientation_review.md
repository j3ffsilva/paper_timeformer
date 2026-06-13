# Revisão Técnica e Científica Independente — Reorientação para Deslocamentos Semânticos Explícitos

**Data:** 2026-06-04
**Revisor:** Independente, com acesso ao código-fonte completo
**Projeto:** `/Users/jeff/Documents/trabalhos/papers/paper-timeformers`
**Escopo:** Avaliar a nova proposta de reorientação do Paper 2 para deslocamentos temporais explícitos sobre espaço base congelado; confrontar com o planejamento e código atuais; produzir recomendações para reescrita do planejamento e refatoração do código.

---

## 1. Resumo Executivo

A nova proposta representa uma mudança de paradigma genuína em relação ao pipeline atual. O pipeline atual (`token@time = concat(h_s(t), m_s(t))` com Transformer condicionado por tempo) opera dentro de um espaço semântico que se deforma com o tempo, tornando deslocamentos difíceis de interpretar. A nova proposta fixa esse espaço no período base `t0` e aprende um módulo de deslocamento externo `delta(w,t)`, de modo que `e(w@t) = b(w) + delta(w,t)`. Isso é uma mudança de "o Transformer representa a posição semântica em t" para "o Transformer define o sistema de coordenadas e um módulo separado representa o afastamento desse sistema ao longo do tempo".

**A mudança é motivada e aponta para um problema real.** O design atual não preserva comparabilidade entre períodos porque o espaço latente do Transformer também se adapta à distribuição do período t. Se o objetivo é deslocamentos interpretáveis e comparáveis, congelar o espaço base é metodologicamente defensável.

**Porém, a proposta como formulada contém ambiguidades sérias** que, se não resolvidas antes de uma reescrita completa, podem produzir um segundo pipeline igualmente incoerente. Os riscos centrais são: (1) o Transformer base congelado treinado apenas em `t0` não produz representações densas e de qualidade uniforme para textos de períodos posteriores, por razões tanto lexicais quanto de distribuição; (2) `delta(w,t)` sem restrições pode reaprender o espaço semântico inteiro, tornando a decomposição `b + delta` circular; (3) a diferença entre esta proposta e embeddings diacrônicos alinhados clássicos (familia Hamilton et al.) precisa ser articulada com precisão, ou o paper corre o risco de ser rejeitado como redescoberta com nomenclatura diferente; (4) a distinção entre `delta(w,t)` independente de contexto e `delta(w,c,t)` dependente de contexto não está resolvida, e a resposta tem implicações profundas para a formulação.

**Veredito provisório:** a direção é correta e deve ser seguida, mas requer um experimento mínimo bem definido antes de qualquer reescrita completa do planejamento. O código atual deve ser parcialmente mantido como baseline e infraestrutura. O experimento mínimo proposto nesta revisão cabe em uma semana e é capaz de responder as questões mais urgentes.

---

## 2. Sua Compreensão da Nova Proposta

A nova proposta pode ser decomposta em cinco afirmações operacionais:

**Afirmação 1 — Espaço base fixo:**
Treinar um Transformer padrão `b(w,c)` exclusivamente sobre o corpus do período base `t0` e congelá-lo permanentemente. Esse Transformer define o sistema de coordenadas semântico de referência. Por definição, `delta(w, t0) = 0` para toda palavra.

**Afirmação 2 — Deslocamento como módulo externo:**
Para períodos `t > t0`, aprender um módulo externo que produza `delta(w,t)` — um vetor no mesmo espaço que `b(w)` (ou em espaço relacionado). A representação queryável é `e(w@t) = b(w) + delta(w,t)`.

**Afirmação 3 — Consultas de deslocamento:**
As consultas desejadas são do tipo `displacement(w@t1, w@t2) = e(w@t2) - e(w@t1)` e `neighbors(w@t)`, onde os vizinhos são determinados por similaridade em `e(w@t)`.

**Afirmação 4 — Trajetória como análise posterior:**
A trajetória de uma palavra não é um objeto aprendido diretamente; ela emerge da sequência de deslocamentos `delta(w,t0), delta(w,t1), ..., delta(w,tn)`, derivada após o aprendizado.

**Afirmação 5 — Questão em aberto:**
Deve-se aprender `delta(w,t)` (word-level) ou `delta(w,c,t)` (occurrence-level, dependente de contexto)? A proposta não resolve essa questão explicitamente.

Esta compreensão é coerente com o texto da seção 3 da descrição do prompt. A leitura deste revisor é que a questão da contextualidade (Afirmação 5) é o ponto mais crítico e, possivelmente, o que diferencia a proposta de trabalhos anteriores — ou a torna equivalente a eles, dependendo da resposta.

---

## 3. A Proposta Resolve o Problema Pretendido?

### 3.1 O desvio identificado é real?

Sim. A análise do código confirma o desvio descrito. Em `models.py`, tanto `TokenTime` quanto `Additive` e `FiLM` injetam informação temporal diretamente na projeção de entrada do Transformer, antes da self-attention. Isso significa que todos os pesos do Transformer são otimizados com gradiente influenciado pelo sinal de período. O espaço de representação que emerge é um espaço conjunto que codifica simultaneamente semântica e período — não há separação clara entre "onde a palavra está" e "em que época ela está".

No `TrajectoryTeacher` e `TrajectoryStudent` (`trajectory_models.py`), o `TemporalEncoder` recebe como entrada as sequências `R_s(t)` — vetores que já são gerados pelo Transformer condicionado por tempo. Isso significa que `m_s(t)` é uma função de representações que já misturaram semântica e período. A argumentação do planejamento atual de que `h_s(t)` representa "onde S está" e `m_s(t)` representa "como S chegou lá" é conceitualmente atraente, mas operacionalmente o `h_s(t)` já carrega informação de período internamente — a distinção é mais retórica do que arquitetural.

**O desvio existe, é substancial, e a motivação para corrigi-lo é legítima.**

### 3.2 A nova proposta resolve o problema?

**Parcialmente.** Congelar o Transformer base resolve o problema da deformação do sistema de coordenadas — o espaço de `b(w,c)` não muda. Porém, surgem três novos problemas que precisam ser resolvidos antes de afirmar que a proposta é uma solução:

**Problema A — Cobertura do vocabulário base:**
O Transformer treinado em `t0 = 1950` não foi exposto a palavras que entram no vocabulário após 1950 (neologismos) nem a palavras raras que aparecem poucas vezes no corpus de `t0`. Para essas palavras, `b(w,c)` terá representações de baixa qualidade ou simplesmente não existirá. O módulo de deslocamento `delta(w,t)` não pode compensar um espaço base degradado — ele apenas adiciona um vetor a uma representação ruim.

**Problema B — Distribuição shift lexical e sintático:**
O Transformer base, ao ser aplicado a textos de períodos posteriores, enfrentará distribuições de sequência diferentes da distribuição de treinamento. Mesmo que o vocabulário coincida, a distribuição de contextos muda. Uma palavra que em 1950 aparecia principalmente em contextos formais pode, em 2000, aparecer principalmente em linguagem informal. O `b(w,c)` para essa palavra em 2000 será produzido por um modelo que nunca viu essa distribuição de contextos — o que pode introduzir viés sistemático que `delta(w,t)` precisará compensar. Isso cria um entrelaçamento entre "deslocamento semântico real" e "drift de distribuição de contexto".

**Problema C — Identidade do deslocamento:**
Sem restrições, `delta(w,t)` pode aprender a mover toda palavra para uma posição "correta" independentemente de `b(w)`, o que equivale a reaprender o espaço semântico inteiro. Se o objetivo de treinamento for apenas "prever contextos corretamente em t", `delta(w,t)` simplesmente aprenderá o complemento de `b(w)` até a representação ótima para t, e `b(w) + delta(w,t)` se tornará indistinguível de um embedding aprendido do zero para t. A decomposição seria matematicamente válida mas interpretacionalmente vazia.

### 3.3 Conclusão

A proposta move o pipeline na direção certa, mas não é uma solução completa tal como formulada. Ela precisa, minimamente, de: (a) uma estratégia para lidar com palavras ausentes ou de baixa frequência em `t0`; (b) uma restrição explícita sobre `delta(w,t)` para evitar colapso da decomposição; (c) uma definição precisa de se `delta` é word-level ou occurrence-level.

---

## 4. Principais Riscos e Ambiguidades

### 4.1 [RISCO CRÍTICO] O Transformer base em `t0` projeta textos de períodos posteriores de forma degradada

Quando o Transformer base, treinado em 1950, processa uma sentença de 1990, ele opera fora de distribuição. Dois efeitos concretos:

- Palavras novas (não vistas em `t0`) podem ser mapeadas para o token `[UNK]` ou para subwords que não capturam o significado completo.
- Contextos lexicais típicos de 1990 mas ausentes em 1950 produzirão ativações que o Transformer nunca aprendeu a associar ao significado correto.

Este não é um problema teórico — ele foi documentado extensivamente na literatura de diachronic NLP. A intensidade do efeito depende da distância temporal entre `t0` e o período mais distante avaliado, e da taxa de mudança léxico-sintática do corpus.

**Consequência direta:** `b(w,c)` para uma ocorrência de `w` em 1990 não representa "o significado de `w` no espaço semântico de 1950" — representa "o que o modelo de 1950 faz com uma entrada que ele nunca viu". Isso invalida parcialmente a interpretação de `delta(w,t)` como deslocamento semântico puro.

**Possíveis mitigações:** (i) usar o corpus de `t0` expandido para incluir os n-gramas de alta frequência de todos os períodos, mas apenas as relações semânticas de `t0`; (ii) usar adapter layers especializadas por período para ajustar a projeção de entrada sem alterar o espaço de saída; (iii) limitar a análise a palavras que são frequentes em `t0` e em todos os períodos avaliados (intersecção de vocabulário), aceitando explicitamente essa limitação no paper.

### 4.2 [RISCO CRÍTICO] `delta(w,t)` é não identificável sem restrições

A formulação `e(w@t) = b(w) + delta(w,t)` define `delta` como `e(w@t) - b(w)`. Mas `e(w@t)` não é observável diretamente — ele é o que o modelo aprende. Sem restrições externas, o problema de treinamento é: dado `b(w)` fixo, encontrar `delta(w,t)` que minimiza alguma perda sobre o corpus de t.

Se a perda for MLM, o gradiente de `delta(w,t)` empurrará o vetor `b(w) + delta(w,t)` para a posição ótima no espaço para predição de contextos em t. Essa posição ótima pode ser qualquer ponto no espaço — não há razão para ela ser próxima de `b(w)`. O único fator que manteria `delta` pequeno seria uma regularização explícita de norma.

Mas regularização de norma cria outro problema: para palavras que mudaram muito de significado, `delta` deveria ser grande; para palavras estáveis, pequeno. Uma regularização uniforme de norma penalizaria igualmente mudanças reais e artificiais.

**Isso não é um problema teórico insolúvel** — é uma restrição de design que precisa ser articulada explicitamente. As opções são: (i) L2 com peso adaptativo (maior λ para palavras que aparecem com alta frequência em `t0`); (ii) restrição de que `b(w) + delta(w,t)` seja próximo do centroide dos contextos de `w` em t segundo o modelo base — ancorando o deslocamento ao espaço de contextos observáveis; (iii) inicialização de `delta` em zero e aplicação de early stopping baseado em evidência de mudança.

### 4.3 [RISCO ALTO] Proposta pode ser equivalente a alinhamento de word embeddings diacrônicos

A literatura de embeddings diacrônicos (família Hamilton et al., família de alinhamento ortogonal como Procrustes, familia de treinamento conjunto como Kim et al.) resolve exatamente o problema de comparar representações de palavras entre períodos em um espaço compartilhado. A formulação `e(w@t) = b(w) + delta(w,t)` é matematicamente equivalente a:

```
e(w@t) = b(w) + (representação_em_t - b(w)) = representação_em_t
```

se `delta(w,t)` for aprendido sem restrições. Isso é idêntico a treinar um embedding separado por período e depois alinhá-los ao espaço de `t0` por Procrustes ou equivalente.

A diferença que a nova proposta deve articular com precisão é: o módulo `delta` opera sobre representações **contextuais** produzidas pelo Transformer base, não sobre embeddings estáticos. Isso significa que:
- A base `b(w,c)` é dependente de contexto — `b(gay, "Os militares gay servem bem")` difere de `b(gay, "um homem alegre e gay")`.
- `delta(w,t)` aplicado a uma representação contextual move a posição de uma ocorrência específica, não de um tipo.

**Se a proposta defender `delta(w,t)` independente de contexto** (word-level), ela é funcionalmente equivalente a embeddings diacrônicos estáticos com alinhamento, diferindo apenas na forma de obter o espaço base. A contribuição seria técnica (usar um Transformer como espaço base em vez de word2vec), não conceitual.

**Se a proposta defender `delta(w,c,t)` dependente de contexto** (occurrence-level), ela é conceitualmente nova e não tem equivalente direto na literatura clássica. Mas os desafios de aprendizado e interpretação aumentam substancialmente.

### 4.4 [RISCO MÉDIO] Definição de `neighbors(w@t)` com polissemia

Se `delta(w,t)` é word-level, `e(w@t)` é um único ponto e `neighbors(w@t)` é bem definida como os k vetores mais próximos de `e(w@t)`. Porém, se `w` tem dois sentidos coexistentes em t (como diagnosticado no Paper 1 para a classe Bifurcating), um único `delta` colapsa ambos os sentidos em um ponto médio — o mesmo problema identificado com mean-prototype no Paper 1.

A questão aberta "deve `delta` depender de contexto?" tem como resposta natural "sim" se queremos lidar com polissemia. Mas isso torna o design muito mais complexo.

### 4.5 [AMBIGUIDADE] Fronteira entre `b(w,c)` e `delta(w,t)`

A proposta descreve `b(w,c)` como "onde a palavra está no espaço semântico de referência" e `delta(w,t)` como "deslocamento temporal". Mas essa fronteira não é clara para palavras que existem em `t0` mas com distribuição de contextos muito diferente: o Transformer base `b(w,c)` já captura parte da informação de como `w` se comporta — qual deve ser a separação correta entre o que o Transformer base deve capturar e o que `delta` deve capturar?

Por exemplo, para a palavra `gay` em `t0 = 1950`, o Transformer base produziria representações próximas do contexto semântico de "alegre/despreocupado". Em 1980, o mesmo Transformer aplicado a contextos de uso identitário produziria representações desalinhadas. `delta(gay, 1980)` deveria ser grande. Mas quão grande deveria ser esse delta comparado ao "deslocamento que o Transformer mesmo produziria se treinado em 1980"? Sem esse baseline, não há referência para avaliar se `delta` está capturando o que se pretende.

---

## 5. Formulação Matemática Recomendada

### 5.1 Notação base

Seja `C(w,t)` o conjunto de ocorrências contextuais da palavra `w` no período `t`. Seja `b_θ` o Transformer base, treinado em `C(w, t0)` para todo `w` e congelado. Para uma ocorrência `c ∈ C(w,t)`, a representação base é:

```
b(w,c) = b_θ(w | contexto_em_c)   ∈ R^d
```

O espaço R^d definido por b_θ é o sistema de coordenadas de referência.

### 5.2 Deslocamento word-level (recomendado como ponto de partida)

```
delta_φ : (w, t) → R^d
e(w@t) = b̄(w) + delta_φ(w,t)
```

onde `b̄(w) = mean_{c ∈ C(w,t0)} b(w,c)` é o centroide das ocorrências de `w` no período base, e `delta_φ` é uma função aprendida parametrizada por φ.

**Por que usar o centroide base e não a representação contextual?** Porque `delta(w,t)` word-level precisa de um ponto de referência fixo e não dependente de contexto. O centroide `b̄(w)` cumpre esse papel: é a posição "canônica" de `w` no espaço de referência. `delta(w,t)` pode então ser interpretado como o afastamento dessa posição canônica em t.

**Constraintes obrigatórias:**
- `delta(w, t0) = 0` por construção ou como perda de ancoragem.
- Regularização de norma: `||delta(w,t)||_2 ≤ C` (hard ou soft), onde C deve ser proporcional a alguma evidência de mudança observável.
- Restrição de suavidade temporal: `||delta(w,t) - delta(w,t-1)||_2 ≤ S` (opcional, pode impedir detecção de mudanças abruptas).

### 5.3 Deslocamento occurrence-level (formulação mais rica, para segunda fase)

```
delta_ψ : (b(w,c), t) → R^d
e(w,c@t) = b(w,c) + delta_ψ(b(w,c), t)
```

Aqui `delta_ψ` condicionado na representação base da ocorrência pode diferenciar sentidos coexistentes: se `b(w,c_1)` e `b(w,c_2)` são representações de duas ocorrências com sentidos distintos, `delta_ψ(b(w,c_1), t)` pode diferir de `delta_ψ(b(w,c_2), t)`, preservando bimodalidade.

**Porém**, isso aumenta a complexidade e o risco de que `delta_ψ` aprenda a recodificar toda a informação contextual, tornando a separação `b + delta` redundante. Esta formulação deve ser considerada apenas após validar a versão word-level.

### 5.4 Consultas recomendadas

```
# Deslocamento entre períodos para uma palavra
displacement(w, t1, t2) = delta(w,t2) - delta(w,t1) = e(w@t2) - e(w@t1)

# Vizinhos no período t
neighbors(w@t) = topk_v { sim(e(w@t), b̄(v)) : v ∈ Vocabulário }
               (comparar contra o centroide base dos vizinhos)

# Magnitude de mudança
change(w) = ||delta(w, tn) - delta(w, t0)||_2
           = ||e(w@tn) - b̄(w)||_2
```

Nota importante: comparar `e(w@t)` com `b̄(v)` (centroide base de outra palavra) é mais estável do que comparar `e(w@t)` com `e(v@t)` quando `delta(v,t)` também está sendo aprendido simultaneamente — pois garante que todos os vizinhos pertencem ao espaço de referência fixo.

---

## 6. Arquitetura e Objetivo de Treinamento Recomendados

### 6.1 Componentes necessários

**Componente 1 — Transformer Base (b_θ):**
Arquitetura idêntica a `BaseModel` em `models.py`, mas treinado apenas em `t0`, sem qualquer `TimeEncoding`, sem `TokenTime`, sem `Additive`. Treinamento padrão MLM. Congelado após treinamento.

Implementação: `Static` em `models.py` já é isso — treinado sem `needs_time`. O que muda é o dado de treinamento: deve ser apenas `t0`, não todos os períodos.

**Componente 2 — Módulo de Deslocamento (delta_φ):**
Uma tabela de embeddings `E_delta[w, t] ∈ R^d`, indexada por word_id e period_id. Possíveis arquiteturas:

- **Tabela simples** (`nn.Embedding(vocab_size * n_periods, d)`): mais transparente, mais interpretável, mais parâmetros. Serve como baseline.
- **Bilinear temporal** (`delta(w,t) = U[w] * V[t]`): fatoração de baixo posto — impõe que o deslocamento seja produto de uma direção por palavra e uma intensidade por período. Compacto e regularizável via norma nuclear.
- **Adapter temporal** (small MLP aplicado ao centroide base): `delta(w,t) = MLP_t(b̄(w))` — o deslocamento é condicionado na representação base da palavra. Parâmetros compartilhados entre palavras por período.

**Recomendação para início:** tabela simples com regularização L2. É o mais direto para verificar se o sinal existe. Migrar para bilinear se os resultados justificarem.

### 6.2 Objetivos de treinamento — classificação por importância

**Obrigatório — MLM sobre corpus de t com base congelada:**
```
L_MLM(t) = -log P_θ_frozen+φ(w_masked | contexto, t)
```
Apenas `phi` (parâmetros do módulo delta) é treinado. O Transformer base processa os tokens; `delta(w,t)` modifica a posição do token-alvo (sujeito) antes ou depois da camada final, dependendo da arquitetura de injeção.

Questão de design crítica: **onde injetar `delta(w,t)`?** Opções:
- Antes do Transformer (embedding stage): `input = b_θ.token_emb(w) + delta(w,t)` — mais simples, mas confunde a representação de entrada.
- Após o Transformer, como correção do vetor de saída: `e = b_θ(w,c) + delta(w,t)` — mais limpo, `b_θ` recebe apenas input sem modificação temporal, e delta é somado na saída. Esta é a formulação arquiteturalmente mais coerente com a proposta.
- Como adapter residual em cada camada: mais expressivo, mas mais caro e mais difícil de interpretar.

**Recomendação:** injeção no vetor de saída (após o Transformer). O Transformer base opera completamente sem modificação; `delta(w,t)` é adicionado ao embedding do sujeito antes da comparação com contexto.

**Obrigatório — Ancoragem em t0:**
```
L_anchor = sum_w ||delta(w, t0)||_2^2
```
Garante que `delta(w,t0) = 0` por otimização (ou pode ser imposto por construção zerando os gradientes em t0).

**Recomendado — Regularização de norma:**
```
L_norm = sum_{w,t} ||delta(w,t)||_2^2  *  lambda
```
Incentiva deslocamentos pequenos a menos que o sinal de treinamento requeira deslocamentos grandes. `lambda` é hiperparâmetro crítico.

**Opcional e potencialmente perigoso — Suavidade temporal:**
```
L_smooth = sum_{w,t} ||delta(w,t) - delta(w,t-1)||_2^2
```
Incentiva trajetórias suaves, mas pode suprimir mudanças abruptas reais. **Não recomendado como padrão** — pode ser explorado como ablação para medir o custo de impor suavidade.

**Opcional — Contrastiva temporal:**
```
L_contrast = -log [sim(e(w@t), contexto_em_t) / sum_{v} sim(e(v@t), contexto_em_t)]
```
Força `e(w@t)` a ser mais similar ao contexto de `w` em t do que ao contexto de outras palavras. Isso operacionaliza a intuição de que `delta(w,t)` move `w` para a vizinhança contextual correta em t.

**Perigoso se mal aplicado — supervisão de ground truth de mudança:**
Usar anotações de mudança semântica (SemEval) para treinar `delta` e depois avaliar no SemEval seria circular. Só deve aparecer como ablação de teto.

### 6.3 Possíveis colapsos e como detectá-los

| Colapso | Sintoma | Diagnóstico |
|---|---|---|
| `delta` reaprendendo tudo | `||delta(w,t0)||_2 >> 0` para maioria das palavras | Verificar ancoragem em t0 |
| `delta` uniformemente zero | Todos deslocamentos nulos | `||delta(w,t)||_2` histograma — deve ter variância |
| `delta` capta período mas não semântica | Deslocamento similar para todas as palavras em t | PCA dos deltas por período — se primeiro PC separa apenas por período, colapso |
| `delta` capta apenas frequência | Palavras raras têm delta menor | Correlação de `||delta||` com frequência em t |

---

## 7. Tratamento de Contextualidade e Polissemia

### 7.1 O problema central

O Transformer base produz representações contextuais `b(w,c)` que são diferentes para cada ocorrência de `w`. Mas as consultas desejadas (`displacement(gay@1950, gay@1980)`, `neighbors(gay@1950)`) tratam `gay@t` como um objeto único, não como uma distribuição sobre ocorrências.

Há uma tensão irreducível aqui: o espaço semântico é contextual (cada ocorrência tem um ponto), mas as consultas são word-level (uma posição por palavra por período). A proposta precisa decidir explicitamente como resolver essa tensão.

### 7.2 Cenário 1: `delta(w,t)` word-level — solução pragmática

`delta(w,t)` não depende de contexto. `e(w@t) = b̄(w) + delta(w,t)` onde `b̄(w)` é o centroide base.

**Vantagens:** simples, consultável, comparável diretamente entre períodos.
**Limitação:** colapsa polissemia. Uma palavra com dois sentidos coexistentes em t terá um único delta que não representa bem nenhum dos dois sentidos — o mesmo problema do mean-prototype identificado no Paper 1.

**Para que cenários funciona bem?** Palavras com mudança de significado dominante (um sentido substitui o outro), não com proliferação de sentidos coexistentes. Para a classe Drift do benchmark sintético: funciona. Para a classe Bifurcating: colapsa.

### 7.3 Cenário 2: `delta(w,c,t)` occurrence-level — solução rica mas complexa

`delta(w,c,t)` depende da representação contextual específica. Isso permite que `delta(gay, contexto_militar, 1980)` difira de `delta(gay, contexto_alegre, 1980)`.

**Complicação 1:** para que `neighbors(gay@1980)` seja definido, precisaríamos agregar `e(gay,c,@1980)` sobre as ocorrências. Essa agregação é o mesmo problema do Set Transformer — e recai no pipeline atual.

**Complicação 2:** durante inferência em um novo texto de 1980, temos `b(gay, novo_contexto)` mas não temos `delta(gay, novo_contexto, 1980)` para contextos não vistos no treinamento — a menos que `delta` seja parametrizado como função de `b(w,c)` (adapter), não como tabela de lookup.

**Recomendação:** para o paper atual, começar com word-level e documentar explicitamente que polissemia com sentidos coexistentes é limitação. A classe Bifurcating pode ser usada para demonstrar onde o modelo falha, o que é um resultado científico válido.

### 7.4 Relação com os agregadores atuais (Mean, Set Transformer, Set Slots)

Na proposta atual, os agregadores transformam `{b(w,c_i)}` em `R_s(t)`, que depois alimenta o encoder temporal. Na nova proposta, o papel do agregador muda:

- Se `delta(w,t)` é word-level: o agregador continua sendo necessário para derivar `b̄(w)` (centroide base), mas essa é uma operação mais simples — apenas media sobre ocorrências em t0. O Set Transformer não acrescenta nada aqui.
- Se `delta(w,c,t)` é occurrence-level: o problema de polissemia é movido para dentro do módulo de deslocamento; os agregadores voltam a ser relevantes para produzir representações queryáveis.

**Conclusão:** os agregadores expressivos (Set Transformer, Set Slots) perdem relevância na versão word-level. Eles só se tornam relevantes novamente se a proposta for estendida para `delta` occurrence-level ou se o objetivo for detectar bimodalidade nas representações `e(w@t)` derivadas.

### 7.5 Indução de sentidos sem supervisão

Se decidir ir para `delta` sensível a sentidos, as abordagens para induzi-los sem supervisão são:

- **GMM sobre `b(w,c)` em t0:** ajustar um modelo de mistura sobre os contextos de `w` no período base, associando cada componente a um "sentido". `delta` separado por componente.
- **Clustering de contextos:** K-Means ou similar para criar pseudo-labels de sentido.
- **Soft assignment via atenção:** usar attention sobre os centroides dos sentidos, como em modelos de sense disambiguation adaptativos.

Todos esses métodos são não-supervisionados mas introduzem hiperparâmetros (número de sentidos K) e não garantem sentidos linguisticamente interpretáveis.

---

## 8. Avaliação Experimental Recomendada

### 8.1 Benchmark sintético — partes reusáveis e partes a criar

**Classes de trajetória reusáveis sem modificação:** Stable, Drift, Abrupt. Essas classes testam o que o módulo `delta` deve capturar: deslocamentos zero, monotônicos e abruptos.

**Classe Bifurcating — reinterpretar:** na nova proposta, Bifurcating é um caso em que `delta(w,t)` word-level deve falhar — o delta único não consegue representar os dois sentidos coexistentes. Isso é resultado esperado e deve ser apresentado como diagnóstico de limitação, não de falha.

**Nova classe recomendada — Returned:** `P(N1|w,t)` decresce de 1.0 para 0.0 entre t0 e t5, depois retorna de 0.0 para 0.5 entre t5 e t9. Testa se `delta(w,t)` consegue capturar trajetórias não-monotônicas sem suavidade forçada. Esta classe distingue abordagens que impõem suavidade das que não impõem.

### 8.2 Métricas primárias para a nova proposta

**M1 — Magnitude de deslocamento por classe:**
```
M1(w) = ||delta(w, tn) - delta(w, t0)||_2
```
Para Stable: M1 deve ser próximo de zero. Para Drift: M1 deve crescer monotonicamente com t. Métrica: correlação de Spearman de M1 com a mudança de `P(N1|w,t)` entre t0 e tn.

**M2 — Preservação de ordem temporal (ranking):**
Para palavras Drift, a sequência `delta(w,t0), delta(w,t1), ..., delta(w,tn)` deve ordenar os períodos. Métrica: tau de Kendall entre a ordenação por `||delta(w,t) - delta(w,t0)||` e o índice de período.

**M3 — Ancoragem em t0:**
```
M3(w) = ||delta(w, t0)||_2
```
Deve ser próximo de zero para todas as palavras. Falha aqui indica problema de otimização.

**M4 — Separabilidade de vizinhança em t:**
`neighbors(w@t)` para palavras Drift deve mudar de período para período. Métrica: proporção de vizinhos de N1 em `neighbors(w@t)` para t inicial vs. t final — equivalente ao D2 atual mas reformulado para a nova consulta.

**M5 — Invariância para palavras estáveis:**
Para palavras Stable, `neighbors(w@t)` deve ser estável ao longo de t. Métrica: variância média da composição da vizinhança entre períodos.

### 8.3 Comparação com baselines relevantes

**Baseline 1 — Centroide sem delta:** `e(w@t) = b̄_t(w)` (centroide das representações base aplicadas ao corpus de t, sem módulo delta). Testa se o simples centroide de `b_θ` em t já captura o deslocamento.

**Baseline 2 — Alinhamento Procrustes:** treinar embeddings word2vec por período e alinhar ao espaço de t0 por rotação ortogonal. Equivalente clássico da literatura — necessário para posicionar a contribuição.

**Baseline 3 — Transformer condicionado por tempo (pipeline atual):** `h_s(t)` do TokenTime atual, para comparar se o espaço deformado é ou não inferior ao espaço base + delta.

### 8.4 Experimento mínimo de validação da proposta (antes de qualquer reescrita maior)

**Objetivo:** Verificar se `delta(w,t)` word-level como tabela de embeddings treinada com MLM frozen captura sinal temporal no benchmark sintético.

**Configuração:**
- Corpus sintético com classes Stable, Drift, Abrupt (sem Bifurcating inicialmente)
- Transformer base `Static` treinado apenas em t0
- Módulo delta: tabela `nn.Embedding(vocab_size * n_periods, d_model)`, inicializada em zero
- Injeção: soma do delta ao vetor de saída do sujeito antes do MLM head
- Loss: MLM padrão + ancoragem L_anchor
- Avaliação: M1, M2, M3, M4 definidos acima

**Verificações de sanidade:**
1. M3 ≈ 0 para todas as palavras (ancoragem funciona)
2. M1 é maior para Drift do que para Stable por margem significativa
3. M1 para Stable não é zero — deve ser pequeno mas não nulo, pois algum delta pode emergir mesmo sem mudança real; verificar se está abaixo de um threshold razoável
4. M2 > 0.5 (ranking temporal parcialmente correto) para Drift

**Duração estimada:** 2-3 dias de implementação + 1 dia de experimentos = 3-4 dias antes de ter resposta definitiva.

---

## 9. Auditoria do Código Atual e Classificação por Arquivo

### 9.1 `src/timeformers/models.py`

**Conteúdo:** Define `BaseModel`, `Static`, `Additive`, `TokenTime`, `FiLM`.

**Classificação:** Manter sem mudança, com uso diferenciado.
- `Static`: torna-se o Transformer base `b_θ` — usa exatamente como está, treinado apenas em t0.
- `TokenTime`, `Additive`, `FiLM`: tornam-se baselines/ablação para comparação com a nova proposta. Devem continuar existindo mas não são mais o componente central.

**Ação necessária:** nenhuma modificação no arquivo. Apenas mudança na semântica de uso: `Static` é promovido a papel central; os modelos com condicionamento temporal são rebaixados a baseline.

### 9.2 `src/timeformers/representations.py`

**Conteúdo:** Extrai `h_s^i(t)` por ocorrência do modelo semântico.

**Classificação:** Reuso com adaptação menor.
- A função `extract_occurrence_representations` pode ser usada para extrair `b(w,c)` do `Static` base.
- A variável `context` (linha 33: `model.token_emb(context_ids).mean(dim=1)`) continua útil para o sinal SSL do agregador, mas precisa ser explicitamente documentada como "embeddings de co-ocorrentes do modelo base", não como representação contextual completa.
- Acrescentar função `compute_word_centroids(reps, period=t0)` para calcular `b̄(w)` por palavra no período base.

### 9.3 `src/timeformers/aggregators.py`

**Conteúdo:** Define `MeanAggregator`, `AttentionPoolingAggregator`, `SetTransformerAggregator`, `SetSlotsAggregator`.

**Classificação:** Manter como infraestrutura auxiliar; papel muda substancialmente.

Na nova proposta word-level: o agregador mais importante é `MeanAggregator`, pois `b̄(w)` é apenas uma média. `SetTransformerAggregator` e `SetSlotsAggregator` só voltam a ser relevantes se a proposta evoluir para `delta` occurrence-level ou para análise de bimodalidade das representações.

Há um bug não corrigido a documentar: `SetSlotsAggregator` retorna `R` com dimensão `num_slots * d_model` (linha 73), diferente de `SetTransformerAggregator` que retorna `d_model`. Isso deve ser documentado claramente ou corrigido se `set_slots` for mantido como opção.

**Ação necessária:** acrescentar função `compute_base_centroids(reps, t0)` que usa `MeanAggregator` exclusivamente sobre o período base.

### 9.4 `src/timeformers/aggregator_ssl.py`

**Conteúdo:** SSL do Set Transformer usando embeddings de co-ocorrentes como sinal.

**Classificação:** Manter como baseline/ablação; não é central na nova proposta word-level.

Se a proposta evoluir para `delta` occurrence-level, o sinal de co-ocorrência deste arquivo pode ser reaproveitado para distinguir sentidos dentro do módulo delta. O mecanismo `context_similarity_contrastive_loss` é reutilizável nesse contexto.

**Ação necessária:** nenhuma, exceto renomear `context_similarity_contrastive_loss` para `cooccurrence_contrastive_loss` para clareza semântica (esse era um ponto identificado na revisão anterior).

### 9.5 `src/timeformers/trajectory_models.py`

**Conteúdo:** Define `TemporalEncoder`, `TrajectoryTeacher`, `TrajectoryStudent`.

**Classificação:** Descontinuar da configuração principal; manter como ablação/comparação.

Na nova proposta, o encoder temporal (`m_s(t)`) não é mais um componente central — a trajetória emerge da sequência de deltas, não de um encoder explícito. No entanto, os resultados do D5a (bidirectional < causal < linear) devem ser reportados como contexto de trabalho anterior.

Se a proposta futura quiser comparar "trajetória derivada dos deltas" com "trajetória aprendida por encoder temporal", este arquivo torna-se necessário para o baseline. Manter, mas não desenvolver.

**Bug documentado anteriormente:** o construtor de `TrajectoryTeacher` usa `encoder_variant="linear"` como default — potencialmente enganoso. Não é necessário corrigir se o componente for descontinuado da configuração principal, mas deve ser documentado.

### 9.6 `src/timeformers/trajectory_train.py`

**Conteúdo:** Treinadores do teacher e do student; avaliação D5a.

**Classificação:** Descontinuar da configuração principal; manter como ablação.

Mesma lógica de `trajectory_models.py`. O pipeline de distillation pode ser mantido como ablação de comparação com a nova abordagem.

**Bug potencial documentado na revisão anterior:** desalinhamento entre `losses` e `class_ids` em `evaluate_all_masked_reconstruction` quando um batch tem posições completamente inválidas. Não precisa ser corrigido se o componente for descontinuado.

### 9.7 `src/timeformers/trajectory_metrics.py`

**Conteúdo:** Métricas D2, D5a, D6, probes lineares, CKA.

**Classificação:** Reuso com adaptação — parte das funções é diretamente reutilizável; parte precisa ser reimplementada para a nova proposta.

**Reutilizável diretamente:**
- `probe_p_n1_r2`: probe linear de `p_n1` — funciona sobre qualquer vetor de representação, incluindo `delta(w,t)`.
- `teacher_sanity_metrics` (parcialmente): CKA e probe podem ser reutilizados para verificar sanidade do módulo delta.
- `d6_bimodality_silhouette`: pode ser usado para verificar se `e(w@t)` para palavras Bifurcating colapsa sentidos — agora como diagnóstico de limitação esperada, não de falha.

**Necessita reimplementação:**
- `cosine_axis_scores` e `d2_context_drift_metrics`: os protótipos N1/N2 derivados de `p_n1` precisam ser repensados. Na nova proposta, a métrica natural é `||delta(w,t)||` comparado com a mudança esperada de `p_n1`. Isso é mais limpo e não usa os protótipos como leakage.
- Novas métricas M1–M5 definidas na Seção 8.2 precisam ser implementadas.

### 9.8 `src/timeformers/trajectories.py`

**Conteúdo:** Constrói `TrajectorySequences` com interpolação linear e máscaras de validade.

**Classificação:** Manter como infraestrutura auxiliar.

Na nova proposta, a "trajetória" é derivada dos deltas. A sequência `delta(w,t0), delta(w,t1), ..., delta(w,tn)` pode ser armazenada em estrutura análoga a `TrajectorySequences`. O código de interpolação e construção de máscaras é reutilizável. O conceito de `observed_mask` vs. `valid_mask` continua relevante para distinguir períodos observados de períodos interpolados.

### 9.9 `src/timeformers/trajectory_losses.py`

**Conteúdo:** `masked_mse`, `linear_cka`, `variance_regularizer`, `anti_identity_loss`.

**Classificação:** Manter, reutilizar.

- `masked_mse`: direto para calcular loss de ancoragem em t0 e loss MLM.
- `linear_cka`: útil para verificar se `delta(w,t)` e `b(w)` são ortogonais (diagnóstico de qualidade).
- `variance_regularizer`: pode ser aplicado sobre os deltas para evitar colapso.

### 9.10 Scripts

**`scripts/run_synthetic_pipeline.py`:**
Classificação: Descontinuar na forma atual; criar novo script análogo para a nova proposta.
O script atual orquestra o pipeline antigo (encoder temporal + student). Pode ser mantido para gerar resultados de baseline do pipeline atual, mas um novo script deve ser criado para o pipeline `base + delta`.

**`scripts/run_d5a_student_ablation.py`:**
Classificação: Manter para reproduzir resultados de D5a como baseline.
Bug documentado: `d_in=args.d_model` hardcoded é problema potencial com `set_slots`; não precisa ser corrigido se o script for mantido apenas para reprodução.

**`scripts/run_ssl_aggregator_sanity.py`:**
Classificação: Manter para sanidade do Set Transformer SSL.
Se o paper incluir comparação entre agregadores como ablação, este script continua sendo relevante.

### 9.11 Resumo tabular

| Arquivo | Classificação | Ação |
|---|---|---|
| `src/timeformers/models.py` | Manter sem mudança | `Static` vira componente central; `TokenTime/Additive/FiLM` viram baseline |
| `src/timeformers/representations.py` | Reuso com adaptação menor | Acrescentar `compute_word_centroids` |
| `src/timeformers/aggregators.py` | Manter como auxiliar | Documentar bug `SetSlotsAggregator`; papel reduzido |
| `src/timeformers/aggregator_ssl.py` | Manter como baseline/ablação | Renomear `context_similarity_contrastive_loss` |
| `src/timeformers/trajectory_models.py` | Descontinuar da config principal | Manter para ablação |
| `src/timeformers/trajectory_train.py` | Descontinuar da config principal | Manter para ablação |
| `src/timeformers/trajectory_metrics.py` | Reuso com adaptação | Reimplementar D2; acrescentar M1–M5 |
| `src/timeformers/trajectories.py` | Manter como auxiliar | Reutilizar estrutura de sequências |
| `src/timeformers/trajectory_losses.py` | Manter, reutilizar | Reutilizar direto |
| `scripts/run_synthetic_pipeline.py` | Descontinuar na forma atual | Criar novo script para pipeline base+delta |
| `scripts/run_d5a_student_ablation.py` | Manter para baseline | Não modificar |
| `scripts/run_ssl_aggregator_sanity.py` | Manter para ablação | Não modificar |

**Arquivos novos a criar:**
- `src/timeformers/displacement_module.py`: módulo `delta_φ(w,t)`, variantes tabela/bilinear/adapter
- `src/timeformers/displacement_metrics.py`: métricas M1–M5
- `scripts/run_displacement_pipeline.py`: pipeline end-to-end da nova proposta
- `scripts/run_displacement_sanity.py`: verificações de sanidade específicas

---

## 10. Plano de Migração em Etapas

### Etapa 0 — Preservação do estado atual (antes de qualquer mudança)

**Objetivo:** garantir que os resultados atuais sejam reproduzíveis independentemente das mudanças futuras.

**Ações:**
1. Criar tag git `v0-trajectory-pipeline` no commit atual.
2. Documentar formalmente os resultados do `set_ssl` em `docs/01-synthetic_results_current.md` (atualmente ausentes nas tabelas formais — identificado na revisão anterior).
3. Executar o pipeline atual com 10+ seeds e arquivar os resultados em `outputs/baseline_trajectory/`.

**Duração:** 1 dia.

### Etapa 1 — Experimento mínimo de validação da nova proposta (sem reescrita do planejamento)

**Objetivo:** verificar se `delta(w,t)` word-level como tabela de embeddings captura sinal temporal no sintético, antes de qualquer compromisso arquitetural maior.

**Implementação mínima necessária:**
- Criar `src/timeformers/displacement_module.py` com `DisplacementTable(vocab_size, n_periods, d_model)`.
- Modificar o treinamento para usar `Static` em t0 apenas, congelar, e treinar `DisplacementTable` com MLM + L_anchor sobre os demais períodos.
- Criar `scripts/run_displacement_sanity.py` para rodar as verificações de sanidade M1–M5 no sintético.

**Verificações de sanidade da Etapa 1:**
1. M3 ≈ 0: `||delta(w, t0)||_2 < 0.1` para ≥ 95% das palavras
2. Drift > Stable: `mean(M1, classe_Drift) > mean(M1, classe_Stable) + 1 std(M1, classe_Stable)`
3. Ranking temporal: `mean(M2, classe_Drift) > 0.6`
4. Vizinhança muda em Drift: proporção de vizinhos N1 em t0 > 0.7; proporção de vizinhos N1 em t9 < 0.3

**Se a Etapa 1 passar:** o sinal existe e a nova proposta é viável. Prosseguir com Etapa 2.
**Se a Etapa 1 falhar:** diagnosticar — o problema pode ser (a) a injeção de delta no lugar errado, (b) a regularização de ancoragem muito forte impedindo aprendizado, (c) o Transformer base não produzindo representações densas o suficiente para t0. Cada falha tem diagnóstico diferente e não justifica abandonar a proposta — apenas requer ajuste de design.

**Duração:** 3-4 dias.

### Etapa 2 — Comparação com baselines (simultânea ao ajuste do planejamento)

**Objetivo:** posicionar a nova proposta em relação a baselines relevantes.

**Implementação:**
- Baseline 1: centroide `b̄_t(w)` sem delta (aplicar `Static` ao corpus de t e fazer mean pooling)
- Baseline 2: `TokenTime` do pipeline atual como baseline de condicionamento interno
- Para cada baseline: métricas M1–M5 no sintético

**Por que agora:** a comparação com baselines é necessária para saber se a nova proposta acrescenta algo além do que já existe. Se o centroide `b̄_t(w)` sozinho já captura os deslocamentos tão bem quanto `delta(w,t)`, o módulo delta é desnecessário.

**Duração:** 3-4 dias (pipeline de baseline é simples).

### Etapa 3 — Reescrita do planejamento com base nas Etapas 1 e 2

**Objetivo:** reescrever `docs/02-novo_planejamento.md` com a nova proposta validada empiricamente.

**O que deve constar:**
- Formulação matemática precisa de `b_θ`, `delta_φ`, `e(w@t)` com decisão explícita sobre word-level vs. occurrence-level
- Discussão explícita do problema de cobertura do vocabulário base (Risco 4.1)
- Discussão explícita do problema de identidade de `delta` (Risco 4.2)
- Posicionamento em relação a embeddings diacrônicos clássicos (Risco 4.3)
- Benchmark sintético atualizado com métricas M1–M5
- Plano de ablação: tabela vs. bilinear vs. adapter; ancoragem forte vs. fraca; com vs. sem suavidade

**Duração:** 2-3 dias.

### Etapa 4 — Extensão para corpus natural (após reescrita)

**Objetivo:** aplicar o pipeline validado no COHA.

**Pré-condições para esta etapa:**
- Etapas 1–3 completas
- Decisão sobre como lidar com vocabulário fora de t0 no corpus natural
- Estratégia para `neighbors(w@t)` que funcione com o vocabulário completo de todos os períodos

**Duração:** 4–6 semanas (dominada pelo preprocessamento do COHA e pela escala computacional).

---

## 11. Veredito: Devemos Seguir Nessa Direção?

### 11.1 Estamos corrigindo um desvio real ou abandonando uma direção válida cedo demais?

**Estamos corrigindo um desvio real.** O código confirma que `TokenTime` injeta informação temporal dentro da self-attention, o que deforma o espaço semântico por período. A distinção `h_s(t)` vs. `m_s(t)` no planejamento atual é conceitualmente atraente mas operacionalmente frágil — `h_s(t)` não é comparável entre períodos porque o espaço foi treinado conjuntamente. O Paper 1 encontrou equivalência entre mecanismos de condicionamento, o que é consistente com essa análise: se o espaço se deforma de qualquer jeito, o mecanismo de condicionamento importa pouco.

Porém, o pipeline atual não é inútil — ele gera resultados válidos para D5a (bidirecional vs. causal vs. linear em classe Abrupt), o que é um resultado real sobre reconstrução de trajetórias. Abandonar todo o pipeline seria desperdiçar esses resultados. A abordagem correta é posicionar o pipeline atual como baseline e contexto, não como método principal.

### 11.2 A nova proposta é cientificamente coerente?

**Coerente em intenção; incompleta em formulação.** A intuição de congelar o espaço base e aprender deslocamentos externos é metodologicamente limpa e tem motivação sólida. As lacunas que tornam a formulação incompleta são:

1. Como lidar com palavras ausentes ou raras em t0 — a proposta não aborda.
2. Como prevenir que `delta` reescreva o espaço semântico inteiro — a proposta menciona isso como "questão aberta" mas não propõe solução.
3. Se `delta(w,t)` é word-level ou occurrence-level — a proposta deixa em aberto.

Essas não são falhas fatais — são questões de design que precisam ser respondidas antes da implementação. A revisão propõe respostas concretas para todas elas.

### 11.3 Qual o menor experimento que deve ser rodado antes de uma reescrita maior?

**A Etapa 1 descrita na Seção 10.** Em resumo:

1. Treinar `Static` em t0 apenas.
2. Congelar `Static`.
3. Criar `DisplacementTable(vocab_size, n_periods, d_model)` inicializada em zero.
4. Treinar com MLM padrão (apenas `DisplacementTable` tem gradiente) + L_anchor em t0.
5. Verificar: M3 ≈ 0; M1 maior para Drift do que para Stable; M2 > 0.6 para Drift.

Esse experimento cabe em 3-4 dias e responde a pergunta mais crítica: o módulo delta aprende sinal temporal a partir de MLM com base congelada?

Se sim: a proposta tem fundamento empírico e pode-se prosseguir com a reescrita completa do planejamento.

Se não: o problema pode ser de design (onde injetar o delta), não de premissa — investigar antes de desistir.

### 11.4 Quais das nossas premissas provavelmente estão erradas?

**Premissa provavelmente errada 1:**
> "O Transformer base congelado em t0 produz representações de qualidade comparável para textos de todos os períodos."

Esta premissa é quase certamente falsa para períodos distantes. A intensidade do problema depende de quão diferente a distribuição léxico-sintática de t_n é de t0. No sintético, onde o vocabulário é fixo e apenas as probabilidades de contexto mudam, o problema não aparece. No COHA (1810–2000), o problema pode ser substancial para os períodos mais distantes.

**Premissa provavelmente errada 2:**
> "A trajetória de uma palavra pode ser derivada posterior mente a partir dos deltas sem perda de informação relevante."

A trajetória derivada dos deltas `delta(w,t0), delta(w,t1), ..., delta(w,tn)` é equivalente a uma sequência de pontos em R^d. O `TrajectoryStudent` atual opera sobre sequências de representações e aprende `m_s(t)` como representação de estado da trajetória. Se não houver diferença entre "sequência de deltas" e "trajetória aprendida", o pipeline de distillation do Paper 1 já capturou isso. Se houver diferença, ela precisa ser articulada — e a nova proposta ainda não fez isso.

**Premissa possivelmente errada 3:**
> "delta(w,t) word-level é suficiente para o nível de granularidade semântica que o paper quer explorar."

Para detectar mudança de significado dominante (Drift, Abrupt), provavelmente sim. Para detectar polissemia coexistente (Bifurcating), certamente não. Se o paper quiser afirmar algo sobre bimodalidade ou polissemia temporal, precisa ir além do word-level.

**Premissa a verificar:**
> "A nova proposta é suficientemente diferente de Hamilton et al. (embeddings diacrônicos alinhados) para constituir uma contribuição nova."

Essa premissa é testável apenas após implementar e comparar. Se `delta(w,t)` word-level treinado com MLM produzir resultados equivalentes aos de word2vec alinhado por Procrustes, a contribuição técnica existe (uso de Transformer contextual como base) mas a contribuição conceitual é fraca. A contribuição forte emerge apenas se o espaço base contextual (`b(w,c)`) acrescentar algo que o espaço estático não tem — e isso é uma questão empírica.

---

*Esta revisão foi produzida com base na leitura integral de todos os arquivos de planejamento, documentação de resultados e código-fonte listados na especificação, sem modificação de qualquer arquivo existente.*

---

## 12. Contra-argumentação técnica para avaliação do revisor

Esta seção registra uma réplica ao parecer acima. O objetivo não é rejeitar
a recomendação geral de reorientação, com a qual concordamos, mas esclarecer
dois pontos técnicos que podem alterar a formulação do experimento mínimo.

Solicitamos que o revisor avalie esta contra-argumentação e acrescente sua
resposta ao final deste mesmo arquivo, na Seção 13 solicitada abaixo.

### 12.1 Comparabilidade do espaço produzido pelo TokenTime atual

O parecer afirma repetidamente que o `TokenTime` atual torna `h_s(t)` não
comparável entre períodos porque o espaço semântico se deforma com o tempo.

Concordamos que o `TokenTime` mistura informação semântica e temporal e não
oferece uma decomposição explícita e identificável entre:

```text
semântica-base + efeito temporal
```

Porém, discordamos da afirmação forte de que suas representações não são
comparáveis entre períodos.

No código atual existe um único modelo `TokenTime`, com os mesmos pesos,
mesma projeção e mesmo Transformer compartilhados para todos os períodos:

```python
t = self.time_enc(epoch_idx).unsqueeze(1).expand_as(tok)
return self.proj(torch.cat([tok, t], dim=-1))
```

Assim, `h(w,t0)` e `h(w,t1)` ainda pertencem ao mesmo espaço vetorial de
saída e podem ser comparados por distância ou similaridade.

Nossa leitura corrigida é:

> O problema do TokenTime atual não é ausência de comparabilidade geométrica
> entre períodos, mas ausência de identificabilidade e interpretabilidade do
> deslocamento. Não podemos separar quanto da diferença entre `h(w,t0)` e
> `h(w,t1)` veio da posição semântica-base, do condicionamento temporal ou
> das interações internas do Transformer.

Perguntas ao revisor:

1. O revisor concorda com essa distinção?
2. Se não concorda, qual propriedade matemática específica impede comparar
   vetores produzidos pelo mesmo modelo compartilhado em períodos distintos?
3. Devemos posicionar a nova proposta como melhoria de **interpretabilidade e
   identificabilidade do deslocamento**, em vez de afirmar que ela cria
   comparabilidade onde antes não existia?

### 12.2 Problema de gradiente no experimento mínimo proposto

O parecer recomenda o seguinte experimento mínimo:

```text
Static treinado apenas em t0 e congelado
delta(w,t) somado à saída do sujeito
treino de delta com MLM padrão
```

No entanto, a implementação atual do corpus e do MLM apresenta uma
incompatibilidade com essa proposta.

Em `src/timeformers/dataset.py`, o sujeito nunca é mascarado:

```python
mask_pos = rng.choice([POS_VERB, POS_OBJECT])
```

Somente verbo ou objeto recebem label de MLM.

Em `src/timeformers/models.py`, o `MLMHead` produz logits separadamente para
cada posição da saída do Transformer:

```python
"logits": self.mlm_head(hidden)
```

Se o Transformer-base já foi executado e está congelado, e adicionarmos
`delta(w,t)` apenas ao vetor de saída do sujeito **depois do Transformer**,
esse vetor modificado não participa da produção dos logits do verbo ou objeto
mascarado.

Portanto, com a arquitetura proposta:

```text
gradiente de L_MLM em relação a delta(w,t) = 0
```

O módulo de deslocamento permaneceria em zero, independentemente do sinal
temporal presente no corpus.

Há duas formas gerais de permitir que o MLM forneça gradiente para `delta`:

1. Injetar `delta` antes ou dentro do Transformer, permitindo que a
   self-attention propague seu efeito para verbo/objeto. Isso, porém, contradiz
   parcialmente a motivação de manter o Transformer completamente alheio ao
   componente temporal e torna o deslocamento menos diretamente interpretável.
2. Alterar a tarefa para que `b_bar(w) + delta(w,t)` prediga ou seja
   contrastado diretamente com o contexto observado no período.

Propomos como experimento mínimo corrigido:

```text
b_bar(w) = centroide-base congelado da palavra em t0
e(w@t)   = b_bar(w) + delta(w,t)
```

Treinar `delta` diretamente contra os contextos observáveis de `w` em `t`:

```text
L = L_context(e(w@t), contextos_observados(w,t))
    + lambda_norm * ||delta(w,t)||^2
```

Com:

```text
delta(w,t0) = 0
```

imposto por construção.

Possíveis formas de `L_context`:

- predição dos verbos e objetos coocorrentes a partir de `e(w@t)`;
- contrastiva entre `e(w@t)` e embeddings congelados dos contextos positivos,
  com contextos negativos de outras ocorrências;
- aproximação de `e(w@t)` ao centroide dos embeddings de contexto observados
  em t, acompanhada de negativos para evitar colapso.

Perguntas ao revisor:

1. O revisor concorda que o experimento MLM descrito na Seção 8.4 produziria
   gradiente nulo para `delta` se este for adicionado somente após o
   Transformer na posição do sujeito?
2. Qual objetivo direto de contexto recomenda para o experimento mínimo?
3. Seria metodologicamente melhor:
   - predizer coocorrentes observáveis;
   - usar uma perda contrastiva;
   - ou modificar o MLM para mascarar/predizer o sujeito?
4. Mascarar o sujeito e treiná-lo para ser previsto faria `delta(w,t)` chegar
   ao modelo de forma circular, já que o identificador de `w` mascarado não
   estaria disponível para selecionar seu próprio deslocamento?
5. Existe uma maneira de usar MLM com base congelada e delta externo sem
   reintroduzir condicionamento temporal dentro do Transformer?

### 12.3 Definição de vizinhança

O parecer recomenda:

```text
neighbors(w@t) = vizinhos de e(w@t) entre os centroides-base b_bar(v)
```

Essa consulta é útil porque responde:

> Para quais conceitos do espaço-base o deslocamento de w aponta no período t?

Entretanto, a intenção original também requer:

```text
neighbors_within_period(w@t)
    = vizinhos de e(w@t) entre todos os e(v@t)
```

Essa segunda consulta responde:

> Quais palavras eram semanticamente próximas naquele período?

Propomos manter as duas consultas como operações e métricas distintas:

```text
base_neighbors(w,t)   = top-k de e(w@t) contra b_bar(v)
period_neighbors(w,t) = top-k de e(w@t) contra e(v@t)
```

Perguntas ao revisor:

1. O revisor concorda que as duas consultas respondem perguntas diferentes e
   ambas devem ser avaliadas?
2. Quais riscos de colapso ou interpretação surgem ao comparar todos os
   `e(v@t)` treinados simultaneamente?
3. Como testar se a vizinhança dentro do período é semanticamente coerente,
   sem utilizar os labels sintéticos como supervisão de treinamento?

### 12.4 Questões que ainda precisamos resolver antes de prosseguir

Além dos pontos acima, solicitamos avaliação sobre as seguintes decisões:

1. O experimento mínimo deve começar com `delta(w,t)` word-level, aceitando
   explicitamente que Bifurcating é uma limitação esperada?
2. O espaço-base deve ser treinado somente em `t0`, ou devemos separar:
   - vocabulário/tokenização aprendidos em todos os períodos;
   - relações semânticas e pesos do Transformer aprendidos somente em `t0`?
3. Quais restrições mínimas tornam `delta` identificável sem impedir mudanças
   abruptas reais?
4. Precisamos comparar já no primeiro experimento:
   - `delta` aprendido;
   - centroide contextual congelado por período sem delta;
   - TokenTime atual;
   ou essa comparação pode vir depois da primeira sanidade?
5. Qual resultado falsificaria a nova proposta, em vez de apenas indicar que
   escolhemos uma loss inadequada?

---

## 13. Solicitação de tréplica do revisor

Reavalie o parecer original à luz da contra-argumentação da Seção 12.

Acrescente sua resposta **ao final deste mesmo arquivo**, sem alterar ou
remover as seções anteriores e sem modificar qualquer outro arquivo do
projeto.

Sua tréplica deve responder diretamente:

1. Você concorda com a correção sobre comparabilidade versus identificabilidade
   do espaço TokenTime?
2. Você concorda que o experimento MLM originalmente proposto gera gradiente
   nulo para `delta` na arquitetura descrita?
3. Qual é a formulação corrigida do menor experimento que devemos executar?
4. Qual objetivo de treinamento recomenda para aprender `delta` externo?
5. Devemos avaliar `base_neighbors`, `period_neighbors` ou ambos?
6. Há algum outro problema conceitual, matemático ou de implementação que
   ainda precisamos resolver antes de prosseguir?
7. Após essa revisão, seu veredito sobre seguir na nova direção muda?

---

## 14. Tréplica do Revisor

**Data:** 2026-06-04

A contra-argumentação da Seção 12 é tecnicamente precisa em dois pontos e levanta questões abertas legítimas. Respondo por item.

---

### 14.1 Comparabilidade versus identificabilidade do espaço TokenTime

**Concordo com a correção.** A afirmação forte do parecer original — de que representações do `TokenTime` "não são comparáveis entre períodos" — está imprecisa. A formulação correta é:

> `h(w,t0)` e `h(w,t1)` pertencem ao mesmo espaço vetorial de saída (mesmos pesos, mesma dimensão), portanto são **geometricamente comparáveis**: distâncias e similaridades cosine entre eles têm sentido matemático.

> O problema é de **identificabilidade e interpretabilidade**: não há como decompor a diferença `h(w,t1) - h(w,t0)` entre (a) mudança semântica da palavra, (b) efeito direto do condicionamento temporal e (c) interações não-lineares do Transformer que co-evoluíram com o sinal de período.

A distinção importa para o posicionamento do paper. A nova proposta não cria comparabilidade onde antes não existia — ela cria **interpretabilidade do deslocamento** onde o pipeline atual não tem. Essa é uma afirmação mais precisa e mais defensável perante revisores.

**Recomendação concreta:** reescrever a motivação no planejamento futuro substituindo "o espaço do TokenTime não é comparável entre períodos" por "o deslocamento produzido pelo TokenTime não é identificável nem interpretável separadamente da posição semântica-base". Isso é mais difícil de atacar e é mais correto.

---

### 14.2 Gradiente nulo no experimento MLM originalmente proposto

**Concordo que há um problema sério aqui, mas com uma ressalva importante.**

A análise da Seção 12.2 está correta para a arquitetura de injeção **após o Transformer**:

```python
e_subj = b_theta(input)[POS_SUBJECT] + delta(w, t)
# e_subj não alimenta o mlm_head que prediz verbo/objeto mascarado
# -> grad(L_MLM, delta) = 0
```

Nesse design, o delta é somado a um vetor que não participa do cálculo dos logits do verbo/objeto. Gradiente nulo confirmado.

**A ressalva:** isso só é verdade se a injeção for *após* o Transformer. Se a injeção for *antes* — no embedding de entrada do sujeito — o gradiente não é nulo, mesmo com o Transformer congelado:

```python
# embedding de entrada do sujeito com delta
emb_subj = token_emb(w) + delta(w, t)
# Transformer congelado processa todos os tokens
# self-attention propaga a influência de emb_subj para posições de verbo/objeto
h = frozen_transformer([emb_cls, emb_subj, emb_verb_ou_mask, emb_obj_ou_mask, emb_sep])
logits = mlm_head(h)  # mlm_head treinável
```

O ponto-chave: **congelar os pesos não bloqueia o backward pass.** Os gradientes ainda fluem através das operações, incluindo a self-attention, até `emb_subj` e portanto até `delta(w,t)`. A diferença é que os pesos do Transformer não são atualizados — `delta` sim.

Isso resolve o problema de gradiente sem alterar a tarefa de MLM. A self-attention do Transformer (cujas matrizes Q, K, V estão congeladas) funciona como um propagador de gradiente da posição mascarada de volta à posição do sujeito.

**Porém, há um custo conceitual:** injetar `delta` antes do Transformer significa que o Transformer "vê" a representação modificada temporalmente do sujeito durante o forward pass. Isso cria dependências cruzadas entre sujeito temporalmente deslocado e contexto, o que não é exatamente o que "espaço base congelado + deslocamento externo" promete. O Transformer nunca veria os dados de t0 dessa forma — o que ele processa em t1 já é o sujeito deslocado.

Essa não é uma contradição fatal, mas precisa ser documentada: o espaço de **saída** do Transformer permanece fixo (pesos congelados), mas o **input** foi modificado pelo delta. A interpretação correta é que o deslocamento acontece no espaço de embeddings de entrada, e o Transformer congelado projeta isso para o espaço de saída de forma não-temporal.

---

### 14.3 Experimento mínimo corrigido

**Formulação recomendada:**

Injetar `delta(w,t)` na camada de embedding (antes do Transformer), manter o Transformer base congelado com todos os pesos incluindo `mlm_head`, e treinar apenas `delta`:

```python
# durante treino do módulo de deslocamento
token_emb_modified = token_emb(input_ids).clone()
token_emb_modified[:, POS_SUBJECT, :] += delta(subject_ids, epoch_idx)

positions = torch.arange(SEQ_LEN, device=device)
x = token_emb_modified + pos_emb(positions)
hidden = frozen_transformer_encoder(x)       # pesos congelados, backward ativo
logits = frozen_mlm_head(hidden)             # mlm_head também pode ser congelado
loss_mlm = cross_entropy(logits, labels)     # label em POS_VERB ou POS_OBJECT

loss = loss_mlm + lambda_anchor * anchor_loss(delta, t0_mask)
# apenas delta.parameters() no optimizer
```

**Verificação de sanidade adicional específica para este design:** confirmar que o gradiente de `loss_mlm` em relação a `delta` é não-nulo antes de rodar o experimento completo:

```python
loss.backward()
assert delta_table.weight.grad is not None and delta_table.weight.grad.abs().max() > 1e-8
```

Se o gradiente for nulo, o design está errado — diagnóstico antes de gastar tempo em treinamento.

**Alternativa se a injeção pré-Transformer for considerada conceptualmente inadequada:**

Usar uma perda de predição de co-ocorrentes diretamente sobre `e(w@t) = b_bar(w) + delta(w,t)`:

```python
# b_bar(w) = centroide congelado das representações base de w em t0
# e_wt = b_bar[subject_ids] + delta(subject_ids, epoch_ids)
# L_context = contrastiva: e_wt próximo dos embeddings dos verbos/objetos observados em t
# negativos = verbos/objetos de outras ocorrências do mesmo período
```

Isso é conceitualmente mais limpo (delta opera completamente fora do Transformer) mas requer uma implementação separada do loop de treinamento atual e não reutiliza o `MLMDataset` existente diretamente.

**Recomendação:** começar com a injeção pré-Transformer (menor mudança de código, reusa todo o pipeline atual, gradiente confirmável). Se o resultado for problemático conceptualmente, migrar para o objetivo contrastivo de co-ocorrentes na Etapa 2.

---

### 14.4 Objetivo de treinamento recomendado para `delta` externo

Classificando por ordem de preferência para o experimento mínimo:

**1º — MLM com injeção pré-Transformer e base congelada** (Seção 14.3 acima)
- Reutiliza o dataset e o loop de treino atuais sem modificação
- Gradiente existe e é analiticamente verificável
- Único custo: semanticamente, o Transformer processa input temporalmente modificado

**2º — Contrastiva de co-ocorrentes sobre centroides congelados**
- Mais limpo conceitualmente: delta completamente externo ao Transformer
- Requer implementação nova do loop de treino
- Mais difícil de avaliar se o sinal de treinamento é suficientemente informativo no corpus sintético pequeno

**3º — Predição direta do centroide de contexto em t**
- `e(w@t)` deve ser próximo do centroide das representações do sujeito nas ocorrências de t
- Risco de colapso: se o centroide de t é exatamente o que queremos aprender como deslocamento, estamos definindo o target pelo próprio target
- Aceitável se o centroide for calculado com o Transformer **base** (não com delta), criando um "alvo fixo" não circulare

Das três, a **primeira** é a mais pragmática para o experimento mínimo. A segunda é a mais correta conceitualmente e deve ser a configuração principal do paper se a primeira funcionar.

---

### 14.5 `base_neighbors` e `period_neighbors`: avaliar ambas?

**Sim, ambas devem ser avaliadas, pois respondem perguntas fundamentalmente diferentes.**

- `base_neighbors(w,t)`: para quais conceitos-base (do espaço t0) o deslocamento de `w` aponta? Essa consulta é **interpretável mesmo sem corpus natural** — os vizinhos são palavras do vocabulário com posição conhecida no espaço de referência.

- `period_neighbors(w,t)`: quais palavras eram semanticamente próximas no período t? Essa consulta é a motivação original de aplicações como SemEval — ela descreve o mundo semântico de t.

**Risco específico de `period_neighbors` com deltas treinados em conjunto:** se delta(w,t) e delta(v,t) são treinados com o mesmo objetivo MLM, eles podem co-adaptar suas posições de forma que a vizinhança emerge de dependências do treinamento, não de proximidade semântica genuína. Para mitigar: avaliar `period_neighbors` com um subconjunto de palavras mantidas fora do treinamento do delta (heldout words com delta = 0), comparando com a vizinhança esperada por construção no sintético.

Para o **experimento mínimo**, avaliar apenas `base_neighbors` — é mais simples e suficiente para falsificar ou confirmar a proposta. `period_neighbors` entra nas métricas da Etapa 2.

---

### 14.6 Outros problemas a resolver antes de prosseguir

**Problema adicional não mencionado anteriormente: o delta do sujeito mascarado é circular no MLM?**

A Seção 12.2 pergunta se mascarar o sujeito tornaria o delta circular. A resposta é: **sim, se o delta for indexado por word_id do sujeito mascarado**, porque durante inferência o word_id de um token mascarado não está disponível para selecionar `delta(w,t)`.

Mas na configuração recomendada (mascarar apenas verbo/objeto, nunca o sujeito), isso **não ocorre** — o word_id do sujeito está disponível e o delta pode ser indexado corretamente. Isso reforça a recomendação de manter a estratégia atual de mascaramento (`mask_pos = rng.choice([POS_VERB, POS_OBJECT])`).

**Problema adicional: o delta deve ser treinável para palavras de Stable?**

No benchmark sintético, palavras Stable têm `p_n1 ≈ const` ao longo de t. Idealmente, `||delta(w,t)||` deveria ser próximo de zero para essas palavras. Mas o MLM não sabe quais palavras são Stable — ele simplesmente minimiza a perda de predição. Sem regularização de norma, o delta de palavras Stable pode ser não-nulo por razões estatísticas (variância de amostragem).

Isso não é um problema do design, mas é um parâmetro de calibração: `lambda_norm` deve ser alto o suficiente para suprimir deltas espúrios em palavras Stable, mas baixo o suficiente para não impedir deltas grandes em palavras Drift/Abrupt. A calibração desse hiperparâmetro é parte do experimento mínimo, não um problema bloqueante.

**Problema de vocabulário em t0 versus t_n no sintético:**

No corpus sintético atual, o vocabulário é totalmente fixo (40 sujeitos × 8 verbos × 8 objetos). Não há palavras novas em períodos posteriores. Isso significa que o Risco 4.1 (cobertura do vocabulário) **não aparece no sintético** — ele só aparecerá no COHA. O experimento mínimo no sintético não testará esse risco. Isso deve ser documentado explicitamente como limitação conhecida do experimento mínimo.

---

### 14.7 Veredito revisado após a tréplica

**O veredito geral não muda: seguir na direção, rodar o experimento mínimo antes de reescrever o planejamento.**

As correções da Seção 12 refinam o experimento, não invalidam a proposta:

| Ponto | Impacto |
|---|---|
| Comparabilidade vs. identificabilidade (12.1) | Muda o *posicionamento retórico* do paper, não o design |
| Gradiente nulo com injeção pós-Transformer (12.2) | Muda o *ponto de injeção* do delta: de pós para pré-Transformer |
| Duas consultas de vizinhança (12.3) | Acrescenta uma métrica ao conjunto de avaliação |

O experimento mínimo corrigido é:

1. Treinar `Static` apenas em t0, congelar todos os pesos incluindo `mlm_head`
2. Criar `DisplacementTable(n_subjects, n_periods, d_model)` inicializada em zero — apenas sujeitos por enquanto, pois verbos/objetos não são o foco do deslocamento semântico
3. Injetar `delta(subject_id, epoch_id)` no embedding do sujeito antes do Transformer
4. Treinar apenas a `DisplacementTable` com `L_MLM + lambda * L_anchor`
5. Verificar sanidade: grad não-nulo antes de começar; M3 ≈ 0; M1(Drift) > M1(Stable)

**O que falsificaria a proposta** (resposta à pergunta 12.4.5): `mean(||delta||, Stable) ≥ mean(||delta||, Drift)` após convergência com regularização calibrada. Isso indicaria que o delta não está capturando mudança semântica real, e sim variância estocástica do treinamento — um problema de sinal fraco, não de design incorreto. Nesse caso, o próximo passo seria o objetivo contrastivo de co-ocorrentes, não abandono da proposta.

**Uma suposição que pode estar errada e que o experimento mínimo deve testar explicitamente:** que o corpus sintético atual (12 ocorrências por sujeito-período, vocabulário de 8 verbos × 8 objetos) é denso o suficiente para que o MLM forneça sinal de deslocamento detectável. Com 12 ocorrências e 8 possíveis verbos/objetos, o sinal temporal em cada célula sujeito-período pode ser muito ruidoso para o MLM extrair. Se isso for um problema, será visto na M3: deltas não-nulos em t0 indicam que o ruído domina o sinal.
