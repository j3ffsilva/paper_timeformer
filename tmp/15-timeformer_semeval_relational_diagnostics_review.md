# Diagnóstico do Piloto SemEval Relacional
**Data:** 2026-06-05  
**Baseado em:** leitura integral de `real_corpus.py`, `real_models.py`, `run_diachronic_relational_experiment.py`, `evaluate_semeval2020_relational.py`, `diagnose_semeval2020_relational.py`, `relational.py` e os resultados dos pilotos 2k e 10k

---

## 1. Resumo executivo

Os pilotos SemEval estão produzindo resultados próximos de aleatório, mas não por falha conceitual da tese — por três problemas de implementação que, combinados, impedem qualquer sinal semântico real de chegar à métrica.

**Problema 1 (mais grave):** o probe `[CLS] word [MASK] [SEP]` é incompatível com o que o modelo aprendeu a fazer. O modelo foi treinado para prever tokens mascarados no MEIO de sequências longas, a partir de contexto completo. O probe consulta o modelo com uma sequência de 3 tokens onde o MASK está imediatamente após a palavra-alvo, sem nenhum contexto adicional. O modelo nunca viu esse padrão durante o treinamento.

**Problema 2:** o score escalar é calculado sobre uma matriz 37×37 (targets vs. targets), não sobre a distribuição de cada target em relação às 932 âncoras. A mudança que você mede é "quanto target_i se moveu em relação aos outros 36 targets do SemEval" — um espaço de referência minúsculo e heterogêneo.

**Problema 3:** o modelo é pequeno demais e treinado por poucas épocas para extrair qualquer representação semântica das 10k janelas de texto real disponíveis.

O diagnóstico de entropia (rho=-0.383) é real e aponta corretamente que o modelo está capturando ruído de calibração, não mudança semântica.

**Veredicto:** não mudar a tese. Mudar o probe, mudar o score, e aumentar a escala de treino, nessa ordem.

---

## 2. Compreensão da proposta

A tese: um Transformer treinado cronologicamente sem sinal temporal captura mudança semântica na forma de mudança de perfil relacional — como cada palavra se relaciona com as demais dentro de cada checkpoint. O score de mudança de uma palavra é derivado de como seu perfil relacional se transformou entre o checkpoint t0 e o checkpoint t1.

No benchmark sintético, isso funcionou porque: o modelo tinha muitas ocorrências por palavra, um probe bem projetado (mascaramento do par contextual), e a comparação era feita entre todos os 40 sujeitos — um espaço de referência coerente e bem amostrado.

No piloto SemEval, nenhuma dessas três condições é satisfeita.

---

## 3. O piloto SemEval está testando a hipótese correta?

**Não da forma atual.** O pipeline está medindo algo diferente do que a tese propõe.

A tese diz: `r_t(w)[v] = similaridade_t(w, v)` — a similaridade entre a palavra `w` e a palavra `v` dentro do checkpoint `t`. Para isso ser informativo, `v` deve ser um conjunto de referência semanticamente relevante e estável.

No piloto, as "palavras v" de referência são os outros 36 alvos do SemEval — palavras que também estão mudando, são heterogêneas entre si (substantivos e verbos de campos completamente diferentes), e em número pequeno demais para definir uma vizinhança significativa.

A proposta correta, mais próxima do design sintético, seria: cada target é comparado com as 932 âncoras. O perfil relacional `r_t(w)` seria um vetor de 932 dimensões representando as similaridades de `w` com as âncoras no checkpoint `t`. O score de mudança seria a magnitude de `delta_rel(w) = r_t1(w) - r_t0(w)`.

Mas há um problema mais fundo: as âncoras estão sendo usadas como espaço de referência VIA o probe — não via embedding direto. E o probe está quebrado, como detalho na Seção 4.

---

## 4. O probe está errado

Esta é a causa raiz dos resultados ruins.

### O que o modelo aprendeu

`RealMLMDataset._make_item()` mascara a posição do MEIO da janela:
```python
mask_pos = candidate_positions[len(candidate_positions) // 2]
```

O modelo foi treinado a prever tokens em posições centrais de janelas de 32 tokens, a partir de contexto bilateral completo (~15 tokens de cada lado). Isso é MLM padrão.

### O que o probe está fazendo

```python
ids = [self.cls_id, word_id, self.mask_id, self.sep_id]
```

O probe apresenta ao modelo a sequência `[CLS] word [MASK] [SEP]` — 4 tokens, com o MASK na posição 2 (imediatamente após a palavra). O modelo extrai os logits na posição 2:

```python
probabilities = torch.softmax(out["logits"][:, 2, :], dim=-1)
```

**Por que isso é problemático:**

1. O modelo nunca viu durante o treinamento uma sequência de 3 tokens onde o MASK está na posição 2. A distribuição de entrada é completamente fora de distribuição.

2. O único "contexto" é a palavra alvo em posição 1. Isso captura co-ocorrência bigrama direta (o que vem imediatamente após esta palavra), não semântica distribucional.

3. A informação semântica de um Transformer vem da interação entre tokens via self-attention. Com apenas 3 tokens, praticamente nenhuma interação contextual é possível — a posição do MASK vê apenas o [CLS] e o word_id à esquerda, e o [SEP] à direita.

4. O modelo não foi treinado para prever o que segue imediatamente uma palavra no início de uma sequência. Ele foi treinado para prever tokens no meio de sequências densas. A tarefa do probe é fundamentalmente diferente.

**O que o probe está medindo de fato:** a probabilidade que o modelo MLM, mal calibrado para esse padrão de input, atribui a cada âncora como "token que sucede diretamente esta palavra no início de uma sequência". Isso é ruído parametrizado — não semântica.

---

## 5. O score escalar está errado

Mesmo com um probe correto, o score atual seria inadequado.

### O que o código computa

`jensen_shannon_similarity_matrix(distributions)` recebe um tensor de shape `[n_targets, n_anchors]` e retorna uma matriz `[n_targets, n_targets]` — a similaridade entre TARGETS baseada em suas distribuições sobre âncoras.

`delta_rel[i]` = variação na linha i dessa matriz 37×37 = "quanto target_i mudou em relação aos outros 36 targets do SemEval".

### Por que isso é problemático

**Contaminação por mudanças nos outros targets:** se `bag_nn` tem alta `mean_abs_delta`, pode ser porque `bag_nn` mudou — ou porque os outros 36 targets ao redor mudaram, deslocando a posição relativa de `bag_nn` mesmo sem ela ter mudado. Com 37 palavras heterogêneas, você não tem controle sobre qual causa está operando.

**Espaço de referência inadequado:** os 36 outros targets são palavras como `graft_nn`, `tip_vb`, `prop_nn`, `chairman_nn` — semanticamente dispersas. "Quanto `attack_nn` se moveu em relação a `tree_nn`" não mede mudança semântica de `attack_nn`. Mede co-variação acidental.

**O score correto:** `JSD(q_t0(w), q_t1(w))` — a divergência Jensen-Shannon direta entre a distribuição do target sobre âncoras no checkpoint 0 e no checkpoint 1. Isso é uma medida direta de quanto o perfil preditivo da palavra mudou. Não depende de outros targets.

No `relational.py`, `jensen_shannon_similarity_matrix` computa pairwise JSD entre todas as linhas de uma matriz. Se em vez de passar a matriz `[n_targets, n_anchors]` e obter `[n_targets, n_targets]`, você computasse diretamente `JSD(q_t0[i], q_t1[i])` para cada target, teria o score correto.

---

## 6. Interpretação dos resultados 2k e 10k

### Spearman perto de zero

O Spearman graded próximo de zero (−0.030 e −0.151) confirma que o ranking produzido não tem correlação com o gold. Mas note que o Spearman levemente NEGATIVO não é evidência de que a proposta está errada — é evidência de que o score está medindo algo que ocasionalmente tem correlação inversa com mudança semântica. Com 37 amostras, qualquer Spearman com |rho| < 0.3 não é distinguível de ruído.

### AUC binário perto de 0.5

AUC=0.476 (2k) e 0.560 (10k) para `max_abs_delta`. O ganho de 0.476 para 0.560 ao passar de 2k para 10k janelas sugere que mais dados de treino ajudam levemente. Mas 0.560 ainda é quase chance.

### Top do ranking

```
bag_nn      binary=0 graded=0.100  ← falso positivo
graft_nn    binary=1 graded=0.554  ← acerto
stroke_vb   binary=0 graded=0.176  ← falso positivo
word_nn     binary=0 graded=0.179  ← falso positivo
risk_nn     binary=0 graded=0.000  ← falso positivo grave
```

Os falsos positivos têm algo em comum: são palavras frequentes e polissêmicas que aparecem em contextos sintáticos diferentes entre 1810-1860 e 1960-2010. O modelo está capturando mudança de padrão sintático, não mudança de significado semântico.

---

## 7. Interpretação do diagnóstico por frequência e entropia

### Correlação com entropia (rho=-0.383, p=0.019)

Esta é a descoberta mais informativa do diagnóstico. Significa: palavras cujo perfil preditivo inicial é menos espalhado (baixa entropia = modelo mais "confiante" na âncora que prevê) têm `max_abs_delta` maior.

**Explicação mecânica:** quando o modelo está em theta_0 e tem distribuição concentrada `q_t0(w)` (alta confiança numa âncora), qualquer perturbação após treinar em t1 causa grande variação no perfil relacional. Palavras com distribuição inicialmente uniforme (alta entropia) não podem mudar muito — já estavam distribuídas igualmente por todas as âncoras.

**Implicação:** o score está capturando "quanto o modelo estava inicialmente calibrado para essa palavra", não "quanto o significado da palavra mudou". Palavras raras ou semanticamente ambíguas, que o modelo nunca aprendeu bem, têm entropia alta e `delta_rel` baixo — não porque são estáveis, mas porque eram ruidosas.

### Correlação com frequência (`mean_abs_delta` vs freq_t0: rho=0.380)

O `mean_abs_delta` está positivamente correlacionado com a frequência em t0. Palavras mais frequentes em t0 têm perfis mais definidos → quando o modelo faz a transição para t1, esses perfis mudam mais (ou o ajuste de gradiente é mais forte para elas). Isso é artefato de treinamento, não sinal semântico.

O `max_abs_delta` não tem correlação significativa com frequência — isso sugere que pegar o ponto máximo da distribuição remove parte do viés de frequência, justificando a escolha de `max_abs_delta` como score principal.

---

## 8. O que provavelmente está errado

Em ordem de impacto:

**1. O probe (causa principal):** `[CLS] word [MASK] [SEP]` é incompatível com o treinamento. Solução: extrair representações das ocorrências reais da palavra no corpus de cada período.

**2. O score escalar (causa secundária):** delta_rel entre targets é o objeto errado. O score correto é `JSD(q_t0(w), q_t1(w))` diretamente.

**3. Escala de modelo insuficiente:** `d_model=32, 1 layer` não aprende representações semânticas de texto real. O próprio código default do `RealStaticMLM` especifica `d_model=96, n_layers=2` — os pilotos usaram configurações abaixo do mínimo funcional.

**4. Épocas insuficientes:** 1 época por período provavelmente não produz gradiente suficiente para atualizar os pesos de forma semanticamente relevante em 10k janelas.

**5. Contaminação por domain shift:** 1810-1860 vs. 1960-2010 é uma diferença de 150 anos. A mudança de gênero, vocabulário, estrutura sintática e domínio entre esses dois períodos é enorme — dominando qualquer sinal de mudança semântica específica de palavras-alvo.

---

## 9. Alternativas recomendadas

### 9.1 Probe correto — extração de ocorrências reais [PRIORITÁRIO]

Em vez de `[CLS] word [MASK] [SEP]`, usar ocorrências reais da palavra no corpus do período com a própria palavra mascarada:

```python
# Para cada ocorrência de "attack_nn" no corpus de 1810-1860:
# "the [MASK] on the town was swift" → extrai distribuição em posição de mask
# Média sobre todas as ocorrências → q_t0(attack_nn)
```

Isso é exatamente o que o modelo aprendeu a fazer durante o treinamento, e produz a distribuição contextual real da palavra. A diferença entre `q_t0(w)` e `q_t1(w)` medida por JSD é o score de mudança.

Este é o maior impacto possível por menor custo de implementação.

### 9.2 Score direto: JSD(q_t0, q_t1) [PRIORITÁRIO]

Não computar a matriz 37×37. Computar diretamente:

```python
score(w) = JSD(q_t0(w), q_t1(w))
```

Isso usa as 932 âncoras como vocabulário de referência e mede diretamente quanto a distribuição preditiva da palavra mudou entre os dois checkpoints. É invariante à mudança de outros targets. É o que `delta_rel` deveria produzir como scalar.

A função `jensen_shannon_similarity_matrix` em `relational.py` já computa JSD corretamente. Basta usar `1 - similarity_t0_t1[i, i]` — a auto-similaridade do target_i entre os dois períodos. Mas isso requer colocar as distribuições dos dois períodos na mesma chamada.

Mais simples: implementar diretamente `jsd_per_target(dists_t0, dists_t1)` que retorna o JSD linha a linha.

### 9.3 Modelo maior e mais épocas [NECESSÁRIO]

Os pilotos usaram `d_model=32, 1 layer, 1 época`. Isso é insuficiente para texto real. Usar:
- `d_model=96, n_layers=2` (o default do `RealStaticMLM`)
- pelo menos 3–5 épocas em t0, 2–3 em t1
- ou orçamento de passos fixo que garanta convergência da loss de MLM abaixo de algum limiar

### 9.4 Controle nulo por resampling [NECESSÁRIO antes de interpretar resultados]

Dividir o corpus de 1810-1860 aleatoriamente em dois subsets e rodar o pipeline como se fossem dois períodos diferentes. O JSD(q_null_a(w), q_null_b(w)) para cada target define o ruído esperado sem mudança semântica. Um score de mudança real deve superar o percentil 95 desse nulo.

Sem isso, não é possível saber se qualquer resultado positivo está acima do ruído.

### 9.5 Âncoras: reduzir e tornar mais estáveis

932 âncoras para um modelo de d_model=32 é desproporcional — o modelo tem mais "âncoras possíveis" do que capacidade de representação. Com d_model=96, 932 ainda é razoável. Mas âncoras selecionadas por frequência pura incluem palavras funcionais (preposições, determinantes) que têm distribuição uniforme sobre qualquer contexto e não discriminam.

Filtrar âncoras: usar apenas substantivos e verbos com alta cobertura nos dois períodos. 200–300 âncoras mais seletivas provavelmente produzem distribuições mais informativas.

### 9.6 Escala antes de mudar método [NÃO RECOMENDADO como primeiro passo]

Escalar o piloto atual (mais janelas, mais épocas) com o probe errado não vai resolver o problema raiz. O sinal que estamos ampliando é o errado. Corrigir o probe primeiro, depois escalar.

---

## 10. Mudanças necessárias no código

### Prioritário: novo modo de extração no probe

`real_corpus.py` precisa de um novo dataset que, para cada target, encontra suas ocorrências reais no corpus e cria janelas com a palavra mascarada:

```python
class RealTargetOccurrenceDataset(Dataset):
    """Mask target word in real occurrence windows."""
    # Para cada ocorrência de target_word no corpus,
    # criar janela com target_word mascarado na posição real
```

Isso requer modificar `run_diachronic_relational_experiment.py` para:
1. Indexar onde cada target aparece em cada corpus de período
2. Criar janelas centradas nas ocorrências
3. Mascarar a posição do target
4. Agregar as distribuições sobre âncoras por média

### Prioritário: novo score em `relational_rows`

Substituir o cálculo atual por:

```python
def target_jsd_scores(dists_t0: torch.Tensor, dists_t1: torch.Tensor) -> torch.Tensor:
    """JSD between each target's anchor distribution in t0 and t1."""
    # dists_t0, dists_t1: [n_targets, n_anchors]
    # retorna [n_targets] com JSD por target
```

Isso elimina a necessidade da matriz de similaridade entre targets para o score principal. A matriz pode permanecer como diagnóstico relacional, mas o score de mudança passa a ser direto.

---

## 11. Veredito

### Continuar com o paradigma de mudança semântica relacional?

**Sim.** O problema não está na tese — está na implementação do probe e do score.

### Abandonar `mean_abs_delta`/`max_abs_delta`?

**Sim, como score principal.** Substituir por `JSD(q_t0(w), q_t1(w))` direto, onde `q` é extraído de ocorrências reais.

`max_abs_delta` pode permanecer como diagnóstico auxiliar, mas não como métrica principal porque: (a) depende da heterogeneidade dos outros targets, (b) não é invariante à mudança alheia, (c) é sensível a âncoras individuais que podem ser ruído.

### Próximo experimento mínimo?

**Dois passos em sequência, não em paralelo:**

**Passo A (1–2 dias):** implementar extração de ocorrências reais para o probe e score JSD direto. Rodar no piloto 10k com `d_model=96, 2 layers, 3 épocas em t0, 2 em t1`. Verificar se o Spearman graded sobe acima de 0.2. Se sim, o pipeline está funcionando e o problema era de implementação.

**Passo B (somente após Passo A positivo):** rodar o controle nulo por resampling para calibrar o limiar de detecção. Rodar a escala completa do corpus.

### O que pode falsificar a direção?

Se após corrigir o probe e o score, o Spearman graded ainda ficar próximo de zero ou negativo com `d_model=96` e escala adequada, há um problema mais profundo: o treinamento contínuo com 150 anos de diferença entre períodos pode estar dominado por domain shift (não mudança semântica de palavras específicas), e o método precisaria de uma forma de isolar o sinal semântico do drift geral do corpus.

Esse seria o caso mais interessante: a tese seria parcialmente correta (captura distribuição shift) mas precisaria de normalização por word-level baseline para isolar mudança semântica de mudança de corpus.
