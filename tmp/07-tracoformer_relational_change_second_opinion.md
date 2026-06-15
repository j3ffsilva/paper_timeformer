# Segunda Opinião Técnica e Científica: Proposta de Análise Relacional de Mudança Semântica com Treinamento Contínuo

**Data:** 2026-06-04
**Revisor:** Análise independente sobre os arquivos de código e o design da proposta
**Escopo:** Avaliação de validade científica, riscos metodológicos, código existente e plano de implementação

---

## 1. Resumo Executivo

A proposta é conceitualmente bem motivada e o instinto central — que relações entre palavras são mais estáveis frente a transformações globais do espaço do que coordenadas absolutas — é defensável e tem raízes sólidas na literatura de semântica distributiva. No entanto, a formulação atual tem três problemas críticos que precisam ser resolvidos antes de qualquer implementação:

1. **A invariância relacional não é total.** Perfis relacionais baseados em similaridade por cosseno são invariantes a rotações e escalamento uniforme, mas NÃO são invariantes a deformações anisotrópicas — e deformações anisotrópicas são exatamente o que o treinamento contínuo produz através de gradientes desbalanceados, mudanças de curvatura no espaço de loss, e LayerNorm. Isso compromete a premissa central.

2. **O treinamento contínuo sem proteção contra catastrophic forgetting confunde sinal com ruído.** O código atual (`train.py`, `MLMTrainer`) não implementa nenhum mecanismo de proteção: nenhum replay, nenhuma regularização EWC, nenhum learning rate por-camada. Qualquer delta relacional medido entre theta_(t-1) e theta_t é uma mistura inseparável de (a) mudança semântica genuína no corpus e (b) degradação de representações de períodos anteriores.

3. **O corpus sintético existente é insuficiente para validar a abordagem relacional.** O vocabulário tem apenas ~60 tokens em sentenças de três palavras (S V O), o que cria um espaço relacional trivialmente pequeno. Com 40 sujeitos, os perfis relacionais têm tamanho 40 (ou ~60), o que é pequeno demais para que distâncias topológicas sejam significativas.

**Veredito preliminar:** A proposta é promissora mas não está pronta para implementação. A validação matemática dos invariantes precisa ser feita explicitamente, e o experimento mínimo de falsificação precisa ser rodado antes de comprometer tempo em novos componentes.

---

## 2. Compreensão Precisa da Proposta

O pipeline proposto pode ser decomposto em quatro operações distintas:

**Passo 1 — Treinamento contínuo cronológico:**
```
theta_0 ← train_from_scratch(D_t0)
theta_t ← continue_training(theta_{t-1}, D_t)  para t = 1...n
```

**Passo 2 — Extração de representações por checkpoint:**
```
e_t(w) = mean { h_{theta_t}(w, c) : c ∈ D_t }
```
onde h é a representação contextual de w nos contextos do período t, processados pelo modelo theta_t.

**Passo 3 — Construção do perfil relacional:**
```
r_t(w)[v] = cos_sim(e_t(w), e_t(v)),  para todo v no vocabulário
```

**Passo 4 — Medida de mudança:**
```
delta_rel(w, t0, t1) = função(r_t0(w), r_t1(w))
```

A intuição é que se theta_t passou por uma rotação global ou rescalonamento uniforme em relação a theta_{t-1}, os cossenos internos são preservados e `delta_rel = 0` para todas as palavras — o que seria correto: nenhuma palavra mudou semanticamente por conta de uma reorganização global do espaço.

Isso é formulado como alternativa C (treinamento contínuo + análise relacional sem alinhamento explícito) versus as alternativas A (modelos independentes + alinhamento), B (treinamento contínuo + alinhamento), D (transformer condicionado temporalmente — pipeline anterior do projeto) e E (transformer congelado).

---

## 3. A Proposta Mede Mudança Semântica ou Apenas Instabilidade?

Esta é a questão central de validade, e a resposta honesta é: **em parte ambos, e a separação não é garantida pelo design proposto.**

### 3.1 O que o perfil relacional captura de fato

O perfil `r_t(w)` mede a posição de `w` em relação a todos os outros tokens no espaço de representação de theta_t. Se dois tokens `w` e `v` eram vizinhos em t0 e deixam de ser em t1, isso pode ser causado por:

- (a) `w` mudou de uso no corpus de t1 para t0 — mudança semântica genuína.
- (b) `v` mudou, afastando-se de `w` mesmo que `w` seja estável.
- (c) O treinamento no corpus de t1 introduziu novos tokens ou padrões que reorganizaram o espaço de forma não-uniforme, deslocando tanto `w` quanto `v` sem mudança semântica em nenhum dos dois.
- (d) Catastrophic forgetting degradou seletivamente as representações de palavras com baixa frequência em t1.

A proposta atual não oferece nenhum mecanismo para separar (a) de (b), (c) e (d).

### 3.2 O que "instabilidade" seria

Se definirmos instabilidade como variação de representação causada por fatores não-semânticos (volume de corpus, velocidade de aprendizado, perturbações de otimização), então o delta relacional medido é a soma de mudança semântica + instabilidade. Para que o método seja válido, é preciso mostrar que, para palavras do grupo "Stable" do corpus sintético, `delta_rel ≈ 0`. Se palavras sabidamente estáveis mostram delta relacional alto, o método está medindo principalmente instabilidade.

**Esse é o teste de falsificação primário e ainda não foi implementado.**

### 3.3 Distinguibilidade vs. instabilidade: um critério operacional

Uma forma de separar os dois é usar âncoras: um conjunto fixo de palavras de alta frequência com trajetória conhecidamente estável. Se o delta relacional de uma palavra alvo cresce, mas sua distância às âncoras também cresce de forma uniforme, o mais provável é reorganização global. Se cresce seletivamente em relação a algumas âncoras mas não a outras, é sinal de mudança relacional específica.

---

## 4. Invariâncias e Limites Matemáticos

### 4.1 O que é realmente invariante por cosseno

Dado um espaço de embeddings E com vetores {e_1,...,e_n}, o perfil relacional por cosseno é invariante sob:

- **Rotação:** R·e_i → R·e_j preserva ⟨e_i, e_j⟩ / (‖e_i‖‖e_j‖) ✓
- **Reflexão:** similar à rotação ✓
- **Escalamento uniforme:** α·e_i → α·e_j preserva cosseno ✓

O perfil relacional NÃO é invariante sob:

- **Escalamento anisotrópico:** diag(λ_1,...,λ_d)·e → os cossenos mudam. Esta é a deformação mais provável durante o treinamento contínuo, pois diferentes dimensões são atualizadas em velocidades distintas por Adam/AdamW através dos segundos momentos.

- **Translação:** e_i + b → cos(e_i+b, e_j+b) ≠ cos(e_i, e_j) em geral. Isso pode ocorrer se o token embedding do [CLS] ou do [PAD] drift de forma sistemática.

- **Normalização por LayerNorm:** o LayerNorm em `models.py` (`norm_first=True`) age por sequência, não por token. Mas ao longo do treinamento, os parâmetros γ e β do LayerNorm são atualizados, o que equivale a uma transformação afim direcionada por dimensão — não preserva o perfil relacional.

### 4.2 Implicação prática

A premissa central da proposta — que perfis relacionais dispensam alinhamento — é matematicamente incorreta para deformações anisotrópicas. O treinamento por gradient descent com Adam produz exatamente esse tipo de deformação. A questão empírica é se a magnitude dessas deformações anisotrópicas é pequena o suficiente para ser negligenciada na prática.

**Isso precisa ser medido, não assumido.**

Uma forma de quantificar: após dois checkpoints consecutivos theta_t e theta_{t+1}, computar a distância CKA entre os espaços de representação para o mesmo conjunto de sentenças. Se CKA ≈ 1.0, a estrutura geométrica foi quase preservada. Se CKA cair significativamente, a mudança anisotrópica é expressiva e os perfis relacionais não podem ser diretamente comparados.

A infraestrutura para medir CKA já existe no código: `trajectory_losses.py` exporta `linear_cka`, e `trajectory_metrics.py` implementa `cka_metric`. Esses componentes já poderiam ser usados para esta validação.

### 4.3 O papel do número de vizinhos k

As métricas baseadas em k-NN (Jaccard, RBO) são mais robustas a deformações anisotrópicas do que similaridades escalares brutas, desde que a deformação não mude a ordenação dos vizinhos. Isso é mais provável de ser satisfeito para k pequeno (os vizinhos mais próximos tendem a permanecer vizinhos mesmo sob deformações moderadas). Portanto, métricas de vizinhança são metodologicamente superiores a delta de coordenada por cosseno, mas não são completamente imunes.

---

## 5. Riscos do Treinamento Contínuo

### 5.1 Catastrophic forgetting

O `MLMTrainer` atual em `train.py` não tem nenhuma proteção contra forgetting. Ao continuar o treinamento em D_t1 a partir de theta_0, os gradientes de D_t1 sobrescrevem parâmetros que codificavam informação de D_t0. A magnitude desse efeito depende da similaridade entre D_t0 e D_t1, do número de passos de treinamento em t1, e do learning rate.

No corpus sintético atual (`corpus.py`), os exemplos de diferentes épocas usam o mesmo vocabulário (mesmos S, V, O), então o forgetting seria menos severo do que em corpora reais. Mas para corpora de texto natural, o forgetting pode ser substancial.

**Consequência para a validade:** Se uma palavra aparece com frequência em D_t0 e raramente em D_t1, sua representação em theta_1 pode ter degradado não por mudança semântica mas simplesmente por ausência de reforço. O delta relacional desta palavra seria espúrio.

### 5.2 Separar mudança de corpus de artefato de otimização

Mesmo que D_t0 e D_t1 sejam idênticos, rodar mais passos de treinamento pode mudar o espaço de representação (convergência mais profunda, saddle points diferentes). Qualquer delta relacional observado entre dois checkpoints treinados em corpora distintos é a soma de:
- efeito do conteúdo diferente do corpus
- efeito do número diferente de passos de treinamento
- efeito da ordem dos exemplos (curriculum implícito)
- forgetting seletivo

**Controle necessário:** rodar um checkpoint "placebo" onde theta_t é continuado no mesmo D_t0 (sem mudança de corpus) e medir o delta relacional resultante. Se delta_rel_placebo ≈ delta_rel_real, o método não está capturando mudança semântica.

### 5.3 Desbalanceamento de tamanho de corpus entre períodos

O corpus sintético usa `examples_per_subject_epoch` fixo (padrão: 12), o que garante tamanhos iguais por período. Em dados reais, corpora históricos têm tamanhos muito diferentes por período. Mais exemplos em um período significa mais passos de gradiente, o que equivale a mais "pressão" sobre o espaço de representação independentemente de mudança semântica.

**Controle necessário:** normalizar o número de passos de gradiente por período, ou usar um subconjunto de tamanho fixo por período.

### 5.4 Learning rate e scheduling

O `MLMTrainer` usa `CosineAnnealingLR` com `T_max=n_epochs`. Ao continuar o treinamento de theta_{t-1} para theta_t, o scheduler reiniciado do ponto 0 do cosseno. O learning rate inicial elevado de um scheduler recomeçado causará perturbações maiores no início de cada período do que ocorreria em treinamento genuinamente contínuo. Isso introduz um artefato sistemático: o delta relacional entre theta_0 e theta_1 será maior que entre theta_1 e theta_2, independentemente do conteúdo do corpus, simplesmente porque o scheduler sempre começa alto.

**Correção necessária:** o `ContinualPeriodTrainer` proposto deve usar um scheduler que continue do estado onde parou, ou usar um learning rate constante pequeno para os períodos de continuação.

---

## 6. Extração e Agregação das Representações

### 6.1 Confusão entre mudança de modelo e mudança de contexto

No código atual em `representations.py`, a função `extract_occurrence_representations` extrai `h_subj` (representação contextual do sujeito S na posição 1 da sequência [CLS S V O SEP]) para cada ocorrência do corpus. A representação é função tanto do modelo theta_t quanto dos contextos (V, O) presentes no corpus de t.

Se extrair as representações usando apenas os contextos do período t (como o código atual implicitamente faz), então `e_t(w)` é função de theta_t E de D_t. Isso significa que `delta_rel` pode ser positivo simplesmente porque os contextos presentes em D_t1 são diferentes dos de D_t0, mesmo que theta_t1 ≈ theta_t0 em termos de parâmetros.

**Solução:** O `CheckpointRepresentationExtractor` proposto deve suportar dois modos:
1. **Extração in-corpus:** usa os contextos reais de cada período — captura mudança de uso + mudança de modelo.
2. **Extração com sondas fixas:** usa um conjunto de sentenças-sonda idêntico para todos os checkpoints — captura apenas mudança de modelo.

A comparação entre os dois modos é informativa: se o delta relacional com sondas fixas é próximo de zero mas o delta com contextos in-corpus é alto, a mudança veio do corpus, não do modelo.

### 6.2 Agregação por média simples e polissemia

O código atual usa `mean(dim=0)` em `MeanAggregator`. Para palavras polissêmicas, a média colapsa os múltiplos sentidos em um vetor "fantasma" que não representa nenhum sentido real. O `SetTransformerAggregator` e `SetSlotsAggregator` em `aggregators.py` são alternativas superiores para este caso.

Entretanto, mesmo com `SetSlotsAggregator`, a agregação produz slots que não necessariamente correspondem aos mesmos sentidos entre checkpoints. O sentido A pode estar no slot 0 em t0 e no slot 1 em t1, tornando a comparação direta de slots incorreta. Isso precisa de um mecanismo de alinhamento de slots ou de comparação por conjuntos.

### 6.3 Disponibilidade de contextos ricos

O corpus sintético (`corpus.py`) é severamente limitado para testar a extração de representações relacionais. Cada sujeito S aparece em sentenças de 3 palavras com verbos e objetos do conjunto {V1..V8, O1..O8}. A "vizinhança" relacional de S é praticamente determinada pela distribuição de V e O escolhidos — o que é exatamente o que p_n1 controla. Isso faz com que o método pareça funcionar trivialmente no corpus sintético, mas não valida sua generalização para textos reais.

---

## 7. Métricas Relacionais Recomendadas

### 7.1 Classificação por robustez e interpretabilidade

As métricas propostas medem aspectos distintos do perfil relacional:

| Métrica | O que mede | Robustez a deformações anisotrópicas | Sensibilidade a k |
|---|---|---|---|
| Jaccard top-k | Sobreposição binária de vizinhança | Alta (ordinal) | Alta |
| Weighted Jaccard | Sobreposição ponderada por similaridade | Média | Média |
| RBO (Rank-Biased Overlap) | Concordância de ranking com decaimento | Alta (ordinal) | Baixa (padrão p=0.9) |
| Kendall tau | Concordância de ranking total | Alta (ordinal) | Baixa |
| Spearman rho | Correlação de ranking total | Alta (ordinal) | Baixa |
| Jensen-Shannon (JS) | Divergência de distribuição de similaridades | Baixa (escalar) | Baixa |
| k-NN graph spectral | Estrutura topológica local | Média | Alta |

**Recomendação de métricas primárias:** RBO (p=0.9) e Kendall tau são as mais defensáveis como primárias porque são ordinais (não dependem dos valores absolutos de similaridade), têm propriedades estatísticas conhecidas, e têm precedente na literatura de semantic change detection (Kim et al., 2014; Hamilton et al., 2016).

**JS Divergence como complemento:** A divergência JS mede mudança na distribuição de similaridades, não na topologia. É sensível a escalamento não-uniforme dos cossenos e portanto menos robusta. Útil como medida de segundo plano mas não deve ser a métrica primária.

**Jaccard top-k como sanity check:** Simples de interpretar, diretamente comparável com a literatura de second-order similarity methods (Schlechtweg et al., 2020), útil como baseline mínimo.

### 7.2 Como escolher k sem otimização em test set

A escolha de k deve ser justificada por critério de informação, não por performance:

- **k mínimo defensável:** k ≥ 10 para vocabulários pequenos, k ≥ 50 para vocabulários grandes. Abaixo de k=10, a vizinhança é dominada por tokens semanticamente idênticos (formas flexionadas, sinônimos exatos) e não reflete estrutura semântica.
- **k máximo defensável:** k ≤ √|V| onde |V| é o tamanho do vocabulário. Acima disso, a vizinhança começa a incluir palavras aleatórias.
- **No corpus sintético:** |V| ≈ 60, então k ∈ {5, 10} é o intervalo razoável. Mas com k=10 e |V|=60, a vizinhança já cobre 17% do vocabulário, o que torna Jaccard pouco informativo.

### 7.3 Estimativa de incerteza

Nenhuma métrica proposta inclui estimativa de significância. Alternativas:

- **Bootstrap sobre ocorrências:** amostrar subconjuntos do corpus de t para estimar a variância de `e_t(w)`, e por consequência a variância de `r_t(w)`. Delta relacional só é significativo se maior que 2× desvio padrão do bootstrap.
- **Permutation test:** permutar rótulos de período e medir a distribuição nula de delta relacional. Palavras com delta acima do percentil 95 da nula são candidatas a mudança real.
- **No código atual:** `trajectory_metrics.py` usa `spearmanr` com p-value de scipy, o que é o padrão mínimo aceitável.

---

## 8. Polissemia e Sentidos Coexistentes

### 8.1 O problema estrutural da polissemia para perfis relacionais

Seja w uma palavra com dois sentidos S1 e S2. Em t0, S1 é dominante (80%) e S2 é raro (20%). Em t1, S2 se torna dominante (60% S2, 40% S1). O centróide médio de w em t0 é fortemente influenciado por S1, e em t1 por S2. O delta relacional refletirá a mudança de sentido dominante — o que é exatamente o que se quer medir.

Porém, se a palavra w tem dois sentidos estáveis com frequências estáveis, mas o tamanho total do corpus varia entre períodos, a representação média pode derivar simplesmente por subamostragem, sem mudança semântica.

### 8.2 O SetSlotsAggregator como solução parcial

O `SetSlotsAggregator` em `aggregators.py` tenta capturar múltiplos sentidos através de `num_slots` queries aprendidas. Mas há dois problemas:
1. O número de slots é fixo e não corresponde necessariamente ao número real de sentidos.
2. Os slots não têm semântica garantida entre checkpoints diferentes — slot 0 em theta_0 e slot 0 em theta_1 podem corresponder a sentidos diferentes.

Para análise relacional de polissemia, uma abordagem mais robusta seria usar **Gaussian Mixture Models no espaço de ocorrências** para cada (w, t), identificando componentes de mistura como sentidos. A mudança semântica é então medida separadamente por componente.

### 8.3 Corpus sintético tem polissemia controlada

O corpus em `corpus.py` modela polissemia através de `p_n1`: a probabilidade de usar o contexto N1 vs N2. Sujeitos da classe `bifurcating` começam dominantemente em N1 e derivam para um estado misto (~0.5). Isso é exatamente polissemia emergente, e é o caso mais interessante para a análise relacional.

Para sujeitos `bifurcating`, a representação média colapsará em um vetor intermediário entre N1 e N2, enquanto a análise por slots deveria separar os dois sentidos. O `d6_bimodality_silhouette` em `trajectory_metrics.py` já mede isso via silhouette score — o que é reutilizável para a análise relacional.

---

## 9. Comparação das Alternativas A–E

### A. Modelos independentes + alinhamento post-hoc

**Vantagens:** Sem forgetting entre períodos; cada modelo é treinado de forma limpa; o espaço de representação de cada período é internamente coerente.

**Desvantagens:** O alinhamento (Procrustes ortogonal ou similar) assume transformação linear entre espaços, o que é só uma aproximação. Para modelos de linguagem, os espaços de representação podem ter topologias diferentes que não são linealmente mapeáveis.

**Quando preferir:** Quando os corpora de diferentes períodos são muito diferentes em tamanho ou domínio, tornando o treinamento contínuo proibitivo.

### B. Treinamento contínuo + alinhamento post-hoc

**Vantagens:** Os espaços de representação são mais similares entre checkpoints (foram inicializados sequencialmente), tornando o alinhamento por Procrustes mais preciso.

**Desvantagens:** Acumula forgetting sem mitigação; o alinhamento ainda introduz erro sistemático.

**Posição:** É melhor que A para corpus sintético, mas não resolve o problema de forgetting.

### C. Treinamento contínuo + análise relacional (proposta atual)

**Vantagens:** Não requer alinhamento explícito; invariante a rotações e escalamento uniforme; conceitualmente mais simples de justificar; computacionalmente mais barato que alinhamento.

**Desvantagens:** Não é invariante a deformações anisotrópicas (ver seção 4); confunde mudança semântica com instabilidade de treinamento (seção 5); a extração de representações mistura efeito do modelo com efeito do corpus (seção 6).

**Posição:** A mais promissora das três, mas precisa de validação experimental das premissas.

### D. Transformer condicionado temporalmente (pipeline anterior)

**Vantagens:** O tempo é uma variável explícita, permitindo interpolação e extrapolação temporal; um único modelo para todos os períodos.

**Desvantagens:** Confunde "o que o modelo aprendeu sobre o período t" com "o que aconteceu no período t"; o sinal temporal pode ser memorizado em vez de generalizado; difícil separar representação de período de representação de palavra.

**Posição do projeto:** O código existente (`models.py`) implementa `Additive`, `TokenTime` e `FiLM` como variantes de condicionamento temporal. Os resultados experimentais anteriores são a linha de base contra a qual a proposta C deve ser comparada.

### E. Transformer congelado aplicado a cada período

**Vantagens:** Elimina completamente a confusão entre mudança do modelo e mudança do corpus; o único efeito medido é a diferença na distribuição de contextos entre períodos.

**Desvantagens:** Usa um modelo pré-treinado que não viu os dados do período específico; pode não capturar neologismos ou mudanças de domínio; as representações são função apenas da distribuição de tokens de entrada.

**Posição:** É o controle mais forte para separar efeito de corpus de efeito de modelo. Deveria ser implementado como baseline obrigatório.

**Recomendação:** Antes de implementar o método C completo, implementar E como baseline e comparar os deltas relacionais produzidos por E e C. Se forem similares, C não está adicionando valor sobre um transformer fixo.

---

## 10. Benchmark Mínimo e Critérios de Falsificação

### 10.1 As quatro classes do corpus sintético

As quatro classes de sujeitos (`stable`, `drift`, `bifurcating`, `abrupt`) em `corpus.py` têm papéis diferentes para validação:

- **Stable (p_n1 fixo em 0.62-0.98):** Serve como controle de falsos positivos. Qualquer método correto deve produzir delta_rel ≈ 0 para sujeitos stable. Se delta_rel > 0 para stable, o método está detectando ruído.

- **Drift (p_n1 decresce monotonicamente de ~0.9 a ~0.1):** Serve como sinal forte. O método deve detectar mudança monotônica e ordenada.

- **Bifurcating (transição suave de um estado para um estado misto):** Testa sensibilidade a mudanças graduais e polissemia emergente.

- **Abrupt (mudança brusca em um único timestep):** Testa sensibilidade temporal e capacidade de localizar a mudança no tempo.

### 10.2 Experimento mínimo de falsificação

O experimento mínimo que pode falsificar ou corroborar a proposta é o seguinte:

**Configuração:**
1. Treinar um modelo `Static` (sem condicionamento temporal) em todos os dados de todas as épocas juntos → theta_base.
2. Extrair `e_base(w)` e perfil relacional `r_base(w)` para todos os sujeitos.
3. Treinar continuamente: theta_0 em D_t0, theta_1 continuado em D_t1, ..., theta_9 continuado em D_t9.
4. Para cada theta_t, extrair e_t(w) usando: (a) contextos do corpus D_t, (b) sondas fixas idênticas para todos os checkpoints.
5. Computar delta_rel entre theta_0 e theta_9 para cada sujeito nos dois modos (a) e (b).
6. Computar o mesmo delta_rel para a alternativa E (theta_base com contextos de D_t0 vs D_t9).

**Critério de corroboração:** O método C deve:
- Produzir delta_rel ≈ 0 para sujeitos `stable` (taxa de falso positivo < 10%).
- Produzir delta_rel > 0 para sujeitos `drift` e `abrupt` com separação estatisticamente significativa de `stable`.
- O modo (a) deve produzir delta_rel similar ao modo (b) para sujeitos `stable` (indicando que o modelo não mudou muito).
- O modo (a) deve produzir delta_rel maior que o modo (b) para sujeitos `drift` (indicando que parte da mudança é capturada pelo corpus e parte pelo modelo).

**Critério de falsificação:**
- Se delta_rel para `stable` é da mesma magnitude que para `drift`, o método não discrimina mudança real de ruído.
- Se delta_rel com sondas fixas (modo b) é igual a zero para todos os sujeitos independente de classe, o método está capturando apenas distribuição de contextos, não representações do modelo.
- Se delta_rel do método E (transformer congelado) é idêntico ao do método C, o treinamento contínuo não está adicionando informação.

### 10.3 Controle placebo

Rodar o treinamento contínuo usando o mesmo corpus D_t0 para todos os períodos (sem variação semântica). Se delta_rel > 0 neste caso, o método está detectando artefatos de treinamento.

---

## 11. Auditoria do Código e Plano de Implementação

### 11.1 Análise dos arquivos existentes

#### `corpus.py` — **Reutilizar com adaptação (2)**

O gerador de corpus é sólido e bem estruturado. As quatro classes de trajetória (stable, drift, bifurcating, abrupt) são adequadas para o benchmark mínimo.

Adaptações necessárias para a proposta C:
- Adicionar função `generate_corpus_by_period(examples)` que agrupa os exemplos por época para permitir treinamento por período separado.
- Adicionar suporte a `fidelity` variável por período (para simular mudança de domínio entre períodos).
- Adicionar geração de "sondas fixas" — sentenças de probe para extração de representações comparáveis entre checkpoints.

#### `dataset.py` — **Reutilizar com adaptação (2)**

`MLMDataset` funciona bem. Adaptação necessária: `MLMDataset(rows, epoch_filter=t)` para criar datasets filtrados por período. A lógica já existe implicitamente (os exemplos têm campo `epoch`), mas não há interface para filtrar por época na classe atual.

#### `models.py` — **Reutilizar como-está (1) para baseline; reutilizar com adaptação (2) para proposta C**

Para a proposta C, o modelo relevante é `Static` (sem condicionamento temporal). As variantes `Additive`, `TokenTime` e `FiLM` são a alternativa D e devem ser mantidas como baseline.

Nota importante: o código usa `norm_first=True` no `TransformerEncoderLayer`, o que é Pre-LN (Xiong et al., 2020). Pre-LN é mais estável para treinamento contínuo do que Post-LN porque gradientes não explodem. Isso é um ponto positivo para a proposta.

#### `train.py` — **Reutilizar como baseline apenas (3)**

O `MLMTrainer` atual treina o modelo em todos os dados de todas as épocas simultaneamente. Para a proposta C, é necessário um `ContinualPeriodTrainer` que:
1. Inicializa com theta_0 treinado em D_t0.
2. Continua de theta_{t-1} para theta_t usando D_t.
3. Salva checkpoints theta_t.
4. Suporta opção de replay de exemplos de períodos anteriores.
5. Mantém o scheduler de learning rate contínuo entre períodos (não reinicia).

#### `representations.py` — **Reutilizar com adaptação (2)**

`extract_occurrence_representations` extrai representações por ocorrência. Adaptações necessárias:
- Adicionar modo de extração com sondas fixas: `extract_probe_representations(model, probe_sentences)`.
- Suporte a extração por subconjunto de época: `extract_for_epoch(model, dataset, epoch=t)`.

#### `metrics.py` — **Manter como baseline (3)**

O módulo atual mede métricas absolutas (trajetória no espaço de representação, spearman com p_n1). Para a proposta C, essas métricas ainda são úteis como baseline de comparação — especialmente `trajectory_metrics` com `path_contrast_drift_minus_stable`.

#### `trajectory_metrics.py` — **Reutilizar com adaptação (2)**

`d6_bimodality_silhouette` mede bimodalidade dos sujeitos bifurcating usando silhouette score, que é diretamente relevante para detectar polissemia emergente na análise relacional. `linear_cka` é reutilizável para medir preservação de estrutura entre checkpoints.

#### `scripts/run_synthetic_pipeline.py` — **Manter como baseline (3)**

Este script implementa o pipeline D (transformer condicionado temporalmente). Deve ser mantido como linha de base para comparação com a proposta C, não adaptado.

#### `scripts/summarize_synthetic_results.py` — **Reutilizar com adaptação (2)**

O framework de sumarização é genérico e reutilizável. Adaptação: adicionar as métricas relacionais propostas na seção 7 ao dicionário `PIPELINE_METRICS`.

### 11.2 Avaliação dos novos componentes propostos

#### `ContinualPeriodTrainer` — **Necessário criar (4)**

Este é o componente mais crítico e mais arriscado. Implementação mínima necessária:
- Loop sobre períodos t=0..n.
- Para t=0: treinamento normal com scheduler cosine.
- Para t>0: `torch.optim.AdamW` com lr reduzido (1e-4 em vez de 1e-3) e sem reinício do scheduler.
- Salvamento de `theta_t.pt` após cada período.
- Suporte a replay: manter buffer de amostras de períodos anteriores.

#### `CheckpointRepresentationExtractor` — **Parcialmente necessário (2/4)**

A infraestrutura de extração já existe em `representations.py`. O que falta é:
- Modo de sondas fixas (novo).
- Interface que itera sobre todos os checkpoints salvos pelo `ContinualPeriodTrainer`.

Pode ser implementado como extensão de `extract_occurrence_representations` em vez de classe nova.

#### `RelationalProfileIndex` — **Necessário criar (4)**

Não existe infraestrutura para armazenar e consultar perfis relacionais `r_t(w)`. Precisa de:
- Armazenamento eficiente de matrizes de similaridade (n_vocab × n_vocab pode ser grande).
- Para vocabulários grandes: armazenar apenas top-k similaridades por palavra (k-NN sparse).
- Interface: `get_profile(word, epoch)` → vetor de similaridades.
- Para o corpus sintético atual com ~60 tokens, uma matriz densa é viável.

#### `RelationalChangeMetrics` — **Necessário criar (4)**

As métricas de Jaccard, RBO, JS divergence não existem no código atual. Precisam ser implementadas. Recomendação de implementação:
- Começar com Jaccard top-k e Spearman rho (mais simples, mais interpretáveis).
- Adicionar RBO como métrica principal.
- JS divergence como complemento.
- Todos devem retornar distribuição de valores por palavra (não apenas média) para permitir análise de casos.

#### `RelationalTrajectoryAnalyzer` — **Necessário criar, mas pode ser simples (4)**

Este componente analisa a sequência de deltas relacionais over time para detectar padrões (monotônico, abrupto, bifurcando). Pode ser implementado como função simples que aplica as métricas do `RelationalChangeMetrics` sobre pares de checkpoints consecutivos.

### 11.3 Dependências entre componentes (ordem de implementação recomendada)

```
1. ContinualPeriodTrainer (treinar e salvar theta_t para todos t)
2. CheckpointRepresentationExtractor (modo sondas fixas + modo in-corpus)
3. RelationalProfileIndex (armazenar r_t(w) para todos w, t)
4. RelationalChangeMetrics (Jaccard, RBO, Spearman)
5. Experimento de falsificação mínimo (seção 10.2)
6. RelationalTrajectoryAnalyzer (apenas se o experimento 5 for positivo)
```

**Não implementar 6 antes de validar 5.** O `RelationalTrajectoryAnalyzer` é prematuro se o experimento de falsificação revelar que o método não discrimina stable de drift.

---

## 12. Relação com Literatura e Contribuição Potencial

### 12.1 Métodos conhecidos de segunda ordem

A proposta reinventa, em parte, métodos de **second-order semantic change** que existem desde pelo menos 2015:

- **Hamilton et al. (2016)** — "Diachronic Word Embeddings Reveal Statistical Laws of Semantic Change" (ACL 2016): propõe alinhamento Procrustes de word2vec treinado por período e compara vizinhanças. A análise de vizinhança é essencialmente o que a proposta C chama de "perfil relacional".

- **Schlechtweg et al. (2019)** — "A Comparison of Semantic Change Detection Approaches" (SemEval 2020): avalia sistematicamente métricas de second-order similarity (JSD, Spearman, Jaccard sobre k-NN) em BERT alinhado. A proposta C, sem o alinhamento mas com treinamento contínuo, é uma variante não testada deste framework.

- **Schlechtweg et al. (2020)** — "SemEval-2020 Task 1: Unsupervised Lexical Semantic Change Detection": define o benchmark padrão, usando exatamente Spearman rho sobre Jaccard de k-NN de representações BERT. A métrica principal da proposta C já é a métrica padrão da área.

- **Rosenfeld & Erk (2018)** — "Deep Neural Models of Semantic Shift": treina modelos separados por período e os compara via segundo-ordem. Mais próximo de A do que C.

### 12.2 O que seria genuinamente novo

A proposta tem elementos de novidade real em dois aspectos:

1. **Treinamento contínuo como mecanismo de construção do espaço relacional:** Em vez de treinar modelos independentes (A) ou alinhar post-hoc (B), usar a continuidade do treinamento para criar um espaço de representação que evolui organicamente. Isso pode preservar melhor as relações de segunda ordem porque os parâmetros "lembram" de períodos anteriores.

2. **Análise relacional sem alinhamento como princípio explícito:** A literatura atual usa alinhamento por padrão. A proposta C questiona essa premissa e oferece uma alternativa baseada em invariância por design. Isso é uma contribuição conceitual mesmo que a implementação precise de qualificações (como documentado na seção 4).

3. **O conjunto de aggregators (SetTransformerAggregator, SetSlotsAggregator):** Não encontrei na literatura de semantic change detection o uso de set transformers para agregar ocorrências contextuais em um perfil de período. Essa é a contribuição técnica mais original do projeto como um todo.

### 12.3 O que precisa ser posicionado cuidadosamente

A proposta precisa demonstrar que:
- Treinamento contínuo C supera modelos independentes A em algum cenário (não apenas que C é mais conveniente).
- A análise relacional sem alinhamento é mais precisa do que análise relacional com alinhamento (B) — o que é contraintuitivo e precisa de justificativa empírica.

Sem esses resultados, a contribuição é principalmente metodológica (uma alternativa que funciona igualmente bem, não melhor) — o que pode ser aceitável para certos venues mas não para os mais competitivos.

---

## 13. Veredito e Próximos Passos

### 13.1 Resposta direta às questões do veredito

**A proposta está formulada de forma coerente?**

Parcialmente sim. A intuição é correta e bem motivada. A formulação matemática do perfil relacional é precisa. Mas a afirmação de que perfis relacionais "dispensam alinhamento" está superestimada — eles reduzem a necessidade de alinhamento para transformações específicas (rotações, escalamento uniforme), não eliminam essa necessidade para o tipo de deformação que o treinamento contínuo com Adam realmente produz (anisotrópica). A proposta precisa ser reformulada com esta qualificação explícita.

**Perfis relacionais realmente permitem evitar alinhamento?**

Parcialmente. Para rotações e reflexões: sim. Para escalamento uniforme: sim. Para deformações anisotrópicas produzidas por gradient descent: não garantido — precisa de verificação empírica (teste CKA entre checkpoints consecutivos). Para palavras de baixa frequência com forgetting seletivo: não — o perfil relacional dessas palavras se degrada por razões não-semânticas.

**Qual é o maior risco de validade?**

O maior risco é **confusão entre forgetting seletivo e mudança semântica.** Uma palavra que era frequente em t0 e rara em t1 terá sua representação degradada em theta_1, não por mudança semântica mas por ausência de gradiente. Seu delta relacional será alto, mas é um falso positivo. Esse risco é especialmente severo para corpora históricos reais (onde frequências variam enormemente entre períodos) e é impossível de mitigar sem alguma forma de replay ou regularização.

**Qual é o experimento mínimo a rodar?**

O experimento de falsificação descrito na seção 10.2:
1. Treinar continuamente sobre as 10 épocas do corpus sintético.
2. Medir delta_rel para cada sujeito.
3. Verificar se `mean(delta_rel[stable]) << mean(delta_rel[drift])` com separação estatisticamente significativa.
4. Verificar se `delta_rel[stable]` é próximo de zero (não apenas menor que drift).

Este experimento pode ser rodado em < 1 hora com o código existente adaptado.

**Quais controles são indispensáveis?**

1. **Controle placebo de corpus:** treinar continuamente com o mesmo corpus em todos os períodos e verificar que delta_rel ≈ 0.
2. **Controle de transformer congelado (alternativa E):** comparar delta_rel do método C com delta_rel usando o modelo sem atualizar parâmetros. Se forem iguais, o treinamento contínuo não agrega informação.
3. **Controle de sondas fixas vs. contextos in-corpus:** separar efeito do modelo do efeito da distribuição de contextos.
4. **Controle de forgetting:** verificar que palavras de alta frequência uniforme entre períodos têm delta_rel ≈ 0.

**Existe uma alternativa mais simples ou mais forte que deveria ser preferida?**

Sim: **modelos independentes por período com análise relacional de segunda ordem (alternativa A + análise de vizinhança)**. Este é o método de Hamilton et al. (2016) com BERT no lugar de word2vec. É mais simples de implementar, completamente isento de forgetting por construção, e tem todo o corpo de literatura de SemEval como baseline de validação. Se a proposta C não superar este método nos benchmarks, a justificativa para treinamento contínuo enfraquece significativamente.

### 13.2 Próximos passos recomendados (em ordem de prioridade)

**Prioridade 1 — Antes de qualquer implementação nova:**
- Medir CKA linear entre theta_0 e theta_1 usando o corpus sintético e o código existente (usa `linear_cka` de `trajectory_losses.py`). Se CKA > 0.95, deformações anisotrópicas são pequenas o suficiente para prosseguir. Se CKA < 0.90, a premissa central precisa ser qualificada fortemente.
- Rodar o experimento de falsificação mínimo da seção 10.2.

**Prioridade 2 — Implementação mínima:**
- Implementar `ContinualPeriodTrainer` em `train.py` (extensão do `MLMTrainer` existente) com salvamento de checkpoints por período.
- Implementar extração de perfis relacionais como função em `representations.py` (não como classe nova).
- Implementar Jaccard top-k, RBO e Spearman rho como funções em `metrics.py`.

**Prioridade 3 — Validação:**
- Rodar o experimento placebo (mesmo corpus em todos os períodos).
- Comparar com alternativa E (transformer congelado).
- Comparar com alternativa B (treinamento contínuo + Procrustes) para verificar se a proposta C é superior.

**Prioridade 4 — Apenas se as prioridades 1-3 forem positivas:**
- Implementar `RelationalProfileIndex` para vocabulários grandes.
- Implementar `RelationalTrajectoryAnalyzer`.
- Estender corpus sintético com vocabulário maior para validar escalabilidade.
- Aplicar a corpora reais.

### 13.3 Avaliação de risco global

| Risco | Severidade | Probabilidade | Mitigação |
|---|---|---|---|
| Deformações anisotrópicas invalidam invariância | Alta | Média | Medir CKA antes de implementar |
| Forgetting seletivo produz falsos positivos | Alta | Alta | Replay ou EWC obrigatório |
| Corpus sintético trivial demais para validar | Média | Alta | Aumentar vocabulário antes de escalar |
| Método idêntico a Hamilton et al. (2016) | Média | Média | Posicionar como extensão para LLMs, não como novo método |
| Scheduler reiniciado introduz artefato temporal | Média | Alta | Corrigir no ContinualPeriodTrainer |
| delta_rel confunde mudança de corpus com mudança de modelo | Alta | Alta | Implementar extração com sondas fixas |

**Conclusão geral:** A proposta é promissora e corretamente motivada, mas precisa de validação experimental de suas premissas antes de ser implementada em escala. O risco mais imediato e mais tratável é o scheduler reiniciado (correção de uma linha de código). O risco mais fundamental é a confusão entre efeito de modelo e efeito de corpus na extração de representações, que requer implementação de sondas fixas. Nenhum desses problemas é fatal, mas ambos precisam ser resolvidos para que os resultados sejam interpretáveis.
