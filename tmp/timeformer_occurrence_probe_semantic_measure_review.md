# Parecer: Ainda estamos medindo o objeto errado?
**Data:** 2026-06-05  
**Baseado em:** `real_corpus.py`, `relational.py`, `run_diachronic_relational_experiment.py`, inspeção direta dos CSVs de saída do piloto 10k-occurrence

---

## 1. Resumo executivo

O probe por ocorrência real corrigiu o problema anterior de incompatibilidade com o treinamento. Mas o `direct_jsd` ainda não mede mudança semântica — e agora sabemos exatamente por quê.

Os valores de `direct_jsd` para os 37 alvos ficam entre 0.00302 e 0.00404, com desvio padrão de 0.00018. Isso é compressão total: a diferença entre o alvo mais alto e o mais baixo é 0.001. Nenhuma correlação com o gold é matematicamente possível nesse intervalo — o ranking é inteiramente determinado por ruído estocástico.

A causa é dupla e precisa ser corrigida em ordem:

**Causa 1 (mais crítica):** as âncoras incluem palavras funcionais. O primeiro elemento da lista é `'a'` — o artigo indefinido. Com 932 âncoras dominadas por palavras funcionais de alta frequência, todos os alvos têm a mesma distribuição preditiva (basicamente a distribuição marginal de frequência do vocabulário), tornando o JSD entre checkpoints indistinguível do ruído.

**Causa 2:** o modelo é pequeno demais (d_model=32, 1 camada, 1 época). Mesmo com âncoras corretas, esse modelo provavelmente não aprendeu representações contextuais específicas por palavra.

**Veredicto:** sim, ainda estamos medindo o objeto errado — mas por razões implementacionais corrigíveis, não por falha conceitual da tese. O próximo passo é filtrar âncoras para palavras de conteúdo, não escala.

---

## 2. Compreensão da tese relacional

A tese: mudança semântica de uma palavra é mudança em seu perfil relacional — como ela se relaciona com outras palavras no mesmo espaço semântico. O probe de ocorrência extrai `q_t(w) = média sobre ocorrências de P_t(âncora | contexto real com w mascarada)`, e `direct_jsd(w) = JSD(q_t0(w), q_t1(w))` mede quanto essa distribuição mudou entre checkpoints.

A formulação está correta. O objeto que queremos medir é legítimo. O problema está no que chamamos de "âncoras" — palavras que deveriam definir o espaço semântico de referência mas que na prática estão sendo selecionadas por frequência bruta.

---

## 3. O novo probe por ocorrência corrigiu o problema anterior?

**Sim, o probe em si está correto.** `RealTargetOccurrenceDataset` cria janelas com a palavra-alvo mascarada em seu contexto real — exatamente o que o modelo aprendeu a fazer durante o treinamento. A média das distribuições preditivas sobre ocorrências é o estimador correto de `q_t(w)`.

O que mudou positivamente:
- O viés por entropia desapareceu (rho=-0.064, não significativo). Antes havia rho=-0.383 — o probe artificial favorecia palavras com perfis de previsão menos incertos, independente de semântica.
- A extração está agora conceitualmente alinhada com a tarefa de treinamento MLM.

O probe NÃO é mais o problema. O problema está em outro lugar.

---

## 4. O `direct_jsd` mede mudança semântica estrutural?

**Conceitualmente sim. Empiricamente não, nos pilotos atuais.**

O JSD entre `q_t0(w)` e `q_t1(w)` é a medida direta correta: informa quanto a distribuição preditiva da palavra sobre âncoras mudou entre os dois checkpoints. É invariante a mudanças globais de coordenadas, usa o mesmo vocabulário de referência nos dois períodos e é interpretável (cada âncora é um conceito nomeado).

O problema: `q_t(w)` em ambos os checkpoints é dominada por âncoras funcionais de alta frequência. Se as top-10 âncoras por probabilidade são `'a', 'the', 'of', 'in', 'to', 'and',...` para QUALQUER alvo em QUALQUER contexto, então:

```
q_t0(attack_nn) ≈ q_t0(bag_nn) ≈ q_t0(graft_nn) ≈ [0.08, 0.07, 0.06, ...]
q_t1(attack_nn) ≈ q_t1(bag_nn) ≈ q_t1(graft_nn) ≈ [0.08, 0.07, 0.06, ...]
```

O JSD de ~0.003 que observamos é a variação estocástica de amostragem nesse regime — não mudança semântica.

Os dados confirmam: todos os 37 alvos têm `direct_jsd` entre 0.00302 e 0.00404, stdev=0.00018. A razão max/min é 1.34. Isso é ruído de amostragem, não sinal semântico.

---

## 5. Por que o sinal SemEval ainda não apareceu?

**Diagnóstico direto dos dados:**

```
direct_jsd: min=0.00302 max=0.00404 mean=0.00330 stdev=0.00018
```

Com stdev = 0.00018 sobre uma escala de [0,1], a amplitude de variação é 0.018% do espaço possível. Nenhuma métrica de ranking (Spearman, AUC) pode capturar correlação com qualquer gold label nessa amplitude.

**Causa raiz 1 — âncoras incluem palavras funcionais:**

Os primeiros itens da lista de âncoras são: `'a', 'able', 'about', 'above', 'according', 'account', 'across', 'act', ...`

`'a'` (artigo indefinido) está no vocabulário de referência. Palavras funcionais têm probabilidade alta e uniforme em qualquer contexto — não discriminam entre `attack_nn` e `bag_nn`, e não mudam de forma semanticamente informativa entre 1810 e 1960.

O código seleciona âncoras como os primeiros `max_anchors` elementos do vocabulário (que está ordenado por frequência) excluindo os alvos. Os tokens mais frequentes em texto lematizado em inglês são exatamente as palavras funcionais: artigos, preposições, conjunções, pronomes.

Com 932 âncoras dominadas por palavras funcionais, `q_t(w)` é essencialmente a distribuição marginal de frequência do corpus — a mesma para todos os alvos e muito parecida nos dois períodos (função words são estáveis entre 1810 e 1960 em termos de distribuição).

**Causa raiz 2 — modelo insuficiente:**

Mesmo com âncoras corretas, d_model=32, 1 camada, 1 época sobre 10k janelas não tem capacidade de aprender representações contextuais específicas por palavra. O modelo provavelmente aprendeu a prever o token mais frequente na posição mascarada independente do contexto.

---

## 6. Estamos medindo diferença local demais em vez de mudança estrutural?

Não — o problema não é granularidade da medida. JSD é uma medida global sobre a distribuição inteira, não local. O problema é que a distribuição inteira está errada porque o espaço de referência (âncoras) está errado.

Se âncoras fossem 300 substantivos e verbos semanticamente informativos, o `direct_jsd` entre `q_t0(graft_nn)` e `q_t1(graft_nn)` seria muito maior — porque `graft` em 1810 aparece com âncoras do campo agrícola e em 1960 com âncoras do campo médico/político. Com âncoras funcionais, essa distinção desaparece.

A tese de "mudança estrutural" faz mais sentido como restrição sobre o que contamos como mudança (não qualquer variação, só mudança no conjunto principal de relações), mas a falha atual é anterior a essa questão.

---

## 7. Riscos da média sobre ocorrências e perda de polissemia

A média sobre ocorrências é matematicamente correta como estimador de `q_t(w)`. Para palavras com sentido único ou com sentido dominante claro, ela funciona bem.

O problema de polissemia é real mas secundário: se `bank` tem 60% de ocorrências financeiras e 40% geográficas em t0, e 80%/20% em t1, a distribuição média captura essa mudança de frequência relativa. O que NÃO captura é se o contexto específico de cada sentido mudou internamente.

Para o SemEval, onde os alvos têm trajetórias de mudança relativamente limpas (um sentido substitui o outro ao longo de 150 anos), a média sobre ocorrências é adequada. Não é o que está causando os resultados ruins.

---

## 8. Riscos das âncoras atuais

Este é o problema dominante. Três riscos concretos:

**Risco 1 — Palavras funcionais:** como demonstrado, `'a'` está no topo da lista. Palavras funcionais têm alta probabilidade para qualquer contexto e são estáveis entre períodos. Elas aplanam `q_t(w)` para próximo da distribuição marginal.

**Risco 2 — Deriva de corpus domina sinal semântico:** mesmo que as âncoras fossem somente palavras de conteúdo, a diferença de 150 anos entre os corpora (1810-1860 vs. 1960-2010) produz forte deriva de domínio. Palavras como "science", "technology", "industry" mudaram de frequência não por mudança semântica das palavras-alvo, mas por mudança geral do corpus. Isso inflacionaria o JSD de qualquer alvo que apareça em contextos onde essas palavras são vizinhas.

**Risco 3 — 932 âncoras é excessivo:** com 932 dimensões e um modelo de d_model=32, há muito mais âncoras do que dimensões de representação. O modelo não tem como discriminar 932 âncoras de forma informativa. 200-300 âncoras de conteúdo bem selecionadas seriam mais informativos do que 932 incluindo funcionais.

---

## 9. O que vem primeiro: top-k, nulo, multimodalidade ou escala?

**Em ordem estrita de prioridade:**

### Passo 1 (hoje, sem retraining): filtrar âncoras para palavras de conteúdo

O corpus é lematizado com POS tags: `attack_nn`, `circle_vb`, etc. As âncoras podem ser filtradas para incluir apenas tokens com sufixo `_nn` ou `_vb`. Isso remove artigos, preposições, conjunções e pronomes imediatamente.

Adicionalmente: excluir âncoras com frequência > 5% do corpus (palavras muito frequentes são frequentemente funcionais mesmo com POS de conteúdo) e < 10 ocorrências por período (muito raras para estimativa estável).

Com âncoras de conteúdo, `q_t(w)` passa a capturar quais substantivos e verbos co-ocorrem com `w` — o sinal semântico que queremos.

Este é um ajuste de 1-2 horas no script de preparação, sem retraining.

### Passo 2 (após filtrar âncoras): diagnóstico manual dos top-5 anchors

Antes de reescalar o modelo, verificar manualmente:
- Para `graft_nn`: quais são os top-5 âncoras em q_t0 e q_t1 após filtrar?
- Para `tip_vb`: idem.
- Eles fazem sentido semântico? São diferentes entre períodos?

Se sim → o modelo é funcional e o problema era só as âncoras. Escala vai ajudar.
Se não → o modelo é fraco demais mesmo com âncoras corretas. Escala é necessária antes de qualquer outra decisão.

### Passo 3 (condicional): escalar o modelo

Se o diagnóstico manual mostrar que o modelo não produz âncoras semanticamente informativas mesmo após filtrar, treinar com `d_model=96, 2 layers, 3 épocas em t0, 2 em t1`. O default do `RealStaticMLM` já especifica esses valores — os pilotos usaram configurações abaixo do mínimo funcional.

### Passo 4 (após modelo funcional): top-k Jaccard como métrica complementar

`1 - Jaccard(top-k t0, top-k t1)` é mais alinhado com a ideia de "círculo social estrutural": ignora microperturbações na cauda da distribuição e foca na mudança dos vizinhos principais. Implementar como métrica paralela ao `direct_jsd`.

JSD captura mudança de massa total na distribuição. Top-k Jaccard captura mudança de vizinhança estrutural. Ambos devem ser reportados — cada um responde a uma pergunta diferente.

### Passo 5 (obrigatório antes de conclusões): nulo por resampling

Antes de qualquer afirmação sobre quais palavras mudaram, é necessário calibrar o limiar. Dividir cada corpus em dois subsets e medir `direct_jsd` e top-k Jaccard sem mudança temporal real. Isso define o piso de ruído por palavra e permite z-score ou percentil.

Sem o nulo, qualquer word_i com JSD maior que outra pode ser artefato de frequência diferente, cobertura de ocorrências diferente ou variação aleatória do modelo.

**Multimodalidade (preservar polissemia):** desnecessário agora. Para os 37 alvos do SemEval, a maioria tem trajetórias limpas. Voltar a isso quando o método básico funcionar.

---

## 10. Mudanças concretas recomendadas no código

### `scripts/prepare_semeval2020_task1.py`

Ao gerar `anchors.txt`, filtrar para:

```python
anchors = [
    word for word in vocabulary
    if (word.endswith("_nn") or word.endswith("_vb"))
    and min_freq <= counts[word] <= max_freq
    and word not in target_set
][:max_anchors]
```

Onde `max_freq` exclui palavras super-frequentes (> 0.5% do corpus) e `min_freq` garante estimativa estável (>= 50 ocorrências em cada período).

### `scripts/run_diachronic_relational_experiment.py`

Adicionar `--anchor-filter-pos` para filtrar âncoras por sufixo POS ao carregar a lista. Isso permite testar sem re-preparar o corpus.

### Novo diagnóstico imediato

Adicionar aos outputs:
```python
# top-5 anchors per target per period
for word, dist in zip(targets, distributions):
    top5 = [(anchors[i], float(v)) for i,v in sorted(enumerate(dist), key=lambda x: -x[1])[:5]]
    print(f"{period} {word}: {top5}")
```

Sem isso, não há como saber se o modelo produziu âncoras semanticamente informativas.

---

## 11. Experimento mínimo recomendado

**Objetivo:** verificar se âncoras de conteúdo produzem sinal semântico com o modelo atual (sem retraining).

**Configuração:**
- Reusar checkpoints do piloto 10k (`--reuse-checkpoints`)
- Gerar nova lista de âncoras filtradas para `_nn` e `_vb` com freq >= 50 por período
- Rodar extração de perfis e `direct_jsd` com as novas âncoras
- Inspecionar manualmente top-5 âncoras para `graft_nn`, `tip_vb`, `attack_nn`
- Medir Spearman graded

**Critério de sucesso:** `direct_jsd` para `graft_nn` > `direct_jsd` para `chairman_nn` (palavra sabidamente estável), com margem de pelo menos 3 stdev. Se isso acontecer com o modelo fraco, a arquitetura está capturando sinal e escala vai ajudar muito.

**Critério de falsificação:** se mesmo com âncoras de conteúdo os valores continuarem comprimidos e sem hierarquia semântica clara, o modelo é genuinamente fraco demais e precisa ser substituído antes de qualquer diagnóstico adicional.

**Custo:** 1-2 horas (sem retraining).

---

## 12. Veredito

### Ainda estamos medindo o objeto errado?

**Parcialmente.** O que medimos está correto na formulação (`direct_jsd` sobre ocorrências reais é o objeto certo). O que medimos está errado na implementação: as âncoras incluem palavras funcionais que eliminam toda discriminação semântica. Mudar as âncoras pode ser o suficiente para produzir sinal, sem mudar a formulação.

### Devemos abandonar `direct_jsd` como score principal?

**Não.** `JSD(q_t0(w), q_t1(w))` com âncoras de conteúdo é a medida mais diretamente alinhada com a tese. Deve permanecer como score principal. Top-k Jaccard deve ser adicionado como complemento estrutural.

### O próximo passo deve ser top-k/rank, nulo por resampling, mudança de âncoras, embeddings contextuais, preservação multimodal ou escala maior?

**Mudar âncoras primeiro.** É o diagnóstico mais barato e potencialmente a causa dominante. Escala depois, se necessário. Nulo por resampling somente depois que o sinal básico aparecer.

### Qual teste barato pode dizer se o problema é métrica ou modelo fraco?

Inspecionar manualmente os top-5 âncoras de `direct_jsd` para `graft_nn` e `chairman_nn` após filtrar âncoras por POS. Se `graft_nn` tiver âncoras como `land_nn, soil_nn, tree_nn` em t0 e `tissue_nn, surgery_nn, government_nn` em t1, e `chairman_nn` tiver âncoras similares em ambos os períodos — o modelo está funcionando e a métrica está certa. Se ambas tiverem as mesmas âncoras funcionais de alta frequência, o modelo é fraco demais independente das âncoras.
