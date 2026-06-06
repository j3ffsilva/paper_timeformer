# Parecer Técnico e Científico Independente: Falha do Perfil PMI Relacional no SemEval-2020

**Data:** 2026-06-06  
**Escopo:** Auditoria completa de implementação, resultados e diagnósticos do método de perfil log-PMI relacional para detecção de mudança semântica temporal  
**Checkpoints auditados:** `outputs/semeval2020_pmi_pilot` (3+2 épocas) e `outputs/semeval2020_pmi_long_epochs_12_8` (12+8 épocas)

---

## 1. Veredito Executivo

O método falha por razões múltiplas e sobrepostas, das quais o prior p_t é apenas a mais discutível — não necessariamente a mais importante. A evidência empírica indica que o modelo MLM, com apenas 2 camadas e d_model=96, não aprendeu representações semânticas suficientemente informativas para diferenciar sentidos de palavras a partir do contexto. Os perfis PPMI têm entropia normalizada em torno de 0.92–0.99 (em uma escala de 0 a 1, onde 1 é distribuição uniforme), o que significa que q_t(w) é quase uniforme sobre 27.311 tokens. Nesse regime, o log-PMI amplifica ruído de cauda e se torna dependente quase exclusivamente das diferenças de calibração entre checkpoints — que é exatamente o que o diagnóstico de entropia expõe: rho(score, variação de entropia) = 0.9443 com p < 10^-17.

Antes de reformular o prior ou o estimador de distância, é necessário verificar empiricamente se o modelo aprendeu qualquer estrutura semântica. Os experimentos discriminativos propostos na seção 8 podem ser executados com os checkpoints existentes em menos de 2 horas de CPU. Nenhuma nova rodada longa deve ser iniciada antes desses diagnósticos.

**Recomendação imediata:** executar as probes de seção 8 e 9 antes de qualquer nova decisão sobre arquitetura ou prior.

---

## 2. Auditoria da Implementação

### 2.1 Correspondência com a formalização matemática

A formalização em `docs/relational_profile_formalization.md` e a implementação em `src/timeformers/relational.py` são consistentes entre si nas fórmulas fundamentais. A função `log_pmi_profiles` (linha 73) implementa corretamente:

```
R_t(w)[v] = log(q_t(w)[v] + eps) - log(p_t[v] + eps)
```

que é a versão com smoothing aditivo da definição canônica `log(q/p)`. A formalização usa `log(q_t(w)[v] / p_t[v])` sem eps explícito, mas o pseudocódigo menciona eps=1e-9, de modo que há consistência interna.

**Divergência 1 — Normalização redundante pós-softmax em `run_diachronic_relational_experiment.py`:**
As funções `occurrence_prediction_distributions` (linha 175) e `full_vocab_occurrence_distributions` (linha 258–261) aplicam softmax sobre logits e depois renormalizam o resultado: `distributions / distributions.sum(...).clamp_min(eps)`. Softmax já produz uma distribuição que soma 1 (a menos de erros de ponto flutuante em float32). A renormalização é numericamente inócua na maioria dos casos mas introduz uma transformação não declarada: se o subset `anchor_ids` for usado (modo legado), a distribuição resultante já não é a distribuição marginal do modelo sobre o vocabulário inteiro, mas sim a restrição renormalizada ao subconjunto de âncoras. Isso não afeta o modo PMI (que usa vocabulário completo), mas é um bug latente no modo legado de âncoras.

**Divergência 2 — Mascaramento durante treinamento (RealMLMDataset._make_item, linha 111–129):**
O dataset de treinamento mascara SEMPRE a posição central de cada janela (`mask_pos = candidate_positions[len(candidate_positions) // 2]`). Isso é uma estratégia de mascaramento determinístico, não aleatório. O BERT canônico mascara 15% dos tokens aleatoriamente, com 80% substituição por [MASK], 10% manutenção, 10% token aleatório. A implementação atual mascara exatamente 1 posição por janela (a central) e ignora completamente as políticas de 10%/10%. Isso tem dois efeitos: (a) o modelo aprende a prever apenas tokens em posições centrais de janelas de 30 tokens, não qualquer token em qualquer posição; (b) o modelo nunca recebe o sinal de "manter o token ou substituir por aleatório", o que distorce as representações internas. Esta é uma divergência significativa em relação ao MLM padrão e pode produzir um modelo que memoriza padrões de posição em vez de contexto semântico.

**Divergência 3 — epoch_idx ignorado no modelo:**
`RealStaticMLM.embed` (linha 54–57) recebe `epoch_idx` como argumento e o descarta imediatamente (`del epoch_idx`). Isso é correto dado que o modelo é estático (sem embeddings temporais), mas o comentário e nome da classe (`RealStaticMLM`) deixam explícito que essa é uma escolha intencional. Contudo, o dataset injeta `period_idx` como `epoch_idx`, e ele chega ao modelo. O fato de ser descartado não é um bug, mas revela que a informação de período não participa da representação — o modelo aprende uma representação única, e a separação temporal é capturada apenas pelo estado dos pesos no checkpoint correspondente. Isso é coerente com a proposta, mas introduz a questão de forgetting catastrófico (ver seção 3).

**Divergência 4 — Prior com sequência de comprimento mínimo:**
A função `neutral_probe_distribution` (linha 179–204) constrói a sequência `[CLS] [MASK] [SEP] [PAD]...` com `seq_len=32`. O modelo foi treinado em janelas de 30 tokens de conteúdo (32 - 2 para CLS/SEP), então receber uma janela com apenas 1 token de conteúdo (o [MASK]) é genuinamente fora de distribuição. O modelo nunca viu, durante treinamento, uma janela com [MASK] na posição 1 imediatamente seguida de [SEP]. A distribuição resultante captura um padrão de ativação não aprendido, não o prior linguístico genuíno. **Essa é a divergência mais crítica entre formalização e implementação.**

**Divergência 5 — Posição do [MASK] na extração de q_t(w):**
Em `RealTargetOccurrenceDataset._make_item` (linha 207–224), o token-alvo é mascarado na posição real dentro da janela (`mask_pos = mask_pos_in_window + 1`). A extração em `full_vocab_occurrence_distributions` (linha 248) usa `out["logits"][batch_indices, batch["mask_pos"].to(device), :]`, o que é correto: extrai logits na posição dinâmica do mask. Não há bug aqui, mas importa verificar que o `mask_pos` está sendo propagado corretamente pelo DataLoader (verificado: sim, linha 222 salva o campo e linha 166 o usa).

**Divergência 6 — Ausência de limiar mínimo de ocorrências:**
A formalização menciona explicitamente (Questão aberta 2) sugestão de `|C_t(w)| >= 10` como limiar mínimo. A implementação não aplica esse limiar: palavras com zero ocorrências recebem distribuição uniforme (`sums.fill_(1.0 / vocab_size)`, linha 243), e palavras com 1 ocorrência são tratadas igualmente a palavras com centenas. Isso torna q_t(w) ruidoso para palavras raras e não há flag de qualidade no output.

**Divergência 7 — Tratamento de missing words:**
Quando `missing = counts == 0` (linha 171 e 253), a implementação preenche com distribuição uniforme E define `counts[missing] = 1`. Isso faz com que a função retorne uma distribuição uniforme normalizada como se houvesse 1 ocorrência, ocultando o fato de que a palavra não apareceu no corpus do período. Não há flag de aviso ou campo de metadados que permita identificar quais palavras não tiveram ocorrências.

### 2.2 Implementação de métricas

**`pmi_cosine_displacement` (linha 93–103):** correto. Computa `1 - cosine_similarity(R_t0, R_t1)` sobre vetores de 27.311 dimensões.

**`ppmi_jsd_displacement` (linha 106–125):** correto matematicamente. A normalização `ppmi / sum(ppmi)` transforma a parte positiva de R_t(w) em uma distribuição de probabilidade, e JSD é computado corretamente. O retorno é em nats (não dividido por log(2)), diferente do que a formalização chama de "interpretável em bits". Isso é inconsistência de unidade entre documentação e código, mas não afeta o ranking relativo dos scores.

**`jensen_shannon_divergence_rows` vs `ppmi_jsd_displacement`:** as duas funções computam JSD mas operam sobre objetos diferentes. A primeira recebe distribuições brutas; a segunda recebe perfis log-PMI e aplica PPMI antes do JSD. O script de avaliação usa `ppmi_jsd_displacement` corretamente.

---

## 3. Causa Mais Provável

A causa mais provável do fracasso não é um único defeito isolado, mas uma cadeia de problemas que se reforçam:

**Causa primária: capacidade insuficiente do MLM para o tamanho do corpus.**
O modelo tem d_model=96, 2 camadas, 4 heads, d_ff=192. Com vocabulário de 27.311 tokens e corpus de ~800.000 janelas de 32 tokens por período, esse modelo é dramaticamente subparametrizado. Um BERT-base tem d_model=768, 12 camadas, e é treinado por décadas de tokens. A loss ao final de 12 épocas no período 0 é 4.78 (equivalente a perplexidade ~119), e ao final de 8 épocas no período 1 é 4.88 (perplexidade ~131). Para referência, um BERT-base atinge perplexidade de ~4-6 em inglês. Com perplexidade 119-131, o modelo está essencialmente prevendo tokens quase aleatoriamente — a distribuição q_t(w) resultante é próxima da uniforme, o que explica diretamente a entropia normalizada de 0.92–0.99 observada nos diagnósticos.

**Causa secundária: prior p_t genuinamente fora de distribuição.**
A hipótese dos autores é válida e deve ser considerada como causa amplificadora, não primária. Com q_t(w) ≈ uniforme e p_t também próxima de uniforme mas com forma ligeiramente diferente (devido à sequência [CLS][MASK][SEP] anômala), o log-PMI amplifica diferenças de calibração minúsculas em tokens raros. A cauda de q_t(w) e p_t para tokens raros pode divergir por erros de amostragem ou por estrutura de atenção induzida pelo padrão de mascaramento de treinamento. O resultado é que tokens raros com probabilidade 1e-5 em q_t e 1e-7 em p_t (diferença de escala acidental) dominam o perfil PMI mesmo sem qualquer relevância semântica.

**Causa terciária: mascaramento determinístico central.**
O modelo aprende a prever especificamente o token central de janelas de 32 posições. Durante o probe, o token-alvo pode estar em qualquer posição da janela (não necessariamente central). O modelo nunca foi treinado para prever tokens em posições não centrais, então q_t(w) pode refletir viés de posição além de semântica.

**Causa quaternária: forgetting catastrófico parcial.**
A loss do período 1 começa em 6.02 (bem acima da loss final do período 0, que era 4.78), indica que o modelo "esqueceu" o período anterior antes de reconsolidar. Com apenas 8 épocas de fino-ajuste, o checkpoint do período 1 possivelmente aprendeu o novo corpus mas a custo de sobre-escrever representações do período 0. Isso torna a comparação R_t0 vs R_t1 não apenas uma medida de mudança semântica, mas de mudança de calibração entre checkpoints com históricos de treinamento distintos.

---

## 4. Hipóteses Alternativas, Ordenadas por Plausibilidade

### H1 — MLM não aprendeu semântica lexical suficiente (plausibilidade: muito alta)
Evidência direta: perplexidade 119-131, entropia PPMI normalizada 0.92–0.99. O modelo prevê tokens quase uniformemente, independente do contexto. Qualquer score derivado de q_t(w) mede principalmente ruído de cauda.

### H2 — Prior p_t amplifica artefatos de cauda (plausibilidade: alta)
Evidência: top-PMI incoerente para graft_nn. Mas essa hipótese só é relevante se H1 for parcialmente falsa (se q_t(w) tivesse algum sinal semântico, o prior ruidoso o distorceria). Com q_t(w) quase uniforme, substituir p_t não resolve o problema raiz.

### H3 — Forgetting catastrófico destrói a comparabilidade entre checkpoints (plausibilidade: média-alta)
Evidência: a loss do período 1 reinicia 1.24 nats acima da loss final do período 0. Se o modelo esqueceu as representações do período 0 ao aprender o período 1, então R_t0(w) e R_t1(w) foram gerados por modelos com espaços de ativação estruturalmente diferentes, e a diferença de cosseno mede principalmente essa deriva, não mudança semântica lexical. Isso explicaria por que o score correlaciona com variação de entropia (calibração global) em vez de graded semântico.

### H4 — Mascaramento determinístico central introduz viés de posição (plausibilidade: média)
O modelo aprende a prever o token na posição 16 (central de 32). Durante o probe de ocorrências reais, o token-alvo pode estar em posição 5 ou 25. Contextos com o alvo em posições não centrais receberão ativações de posição diferentes das vistas em treinamento. Isso adiciona ruído ao q_t(w) dependendo da posição do alvo no corpus real.

### H5 — q_t(w) mede concordância morfossintática, não semântica lexical (plausibilidade: média)
MLMs têm tendência a prever substituições morfossintáticas plausíveis (verbos no lugar de verbos, substantivos no lugar de substantivos) mais do que substitutos semânticos próximos. Com modelo pequeno e corpus histórico com grafia variável, é possível que q_t(w) capture padrões de colocação de POS em vez de sentido.

### H6 — Instabilidade estatística por número pequeno de alvos (plausibilidade: baixa como causa principal)
37 alvos é suficiente para correlações de Spearman robustas se o sinal existisse. A correlação quase perfeita com entropia (rho = -0.94 e +0.94) indica que o padrão encontrado é real, apenas o padrão errado.

### H7 — Overfitting ao corpus de treinamento (plausibilidade: baixa)
Com perplexidade >100 em um corpus de 800k exemplos por período, o problema é underfitting, não overfitting.

---

## 5. O Que os Resultados de 12+8 Épocas Realmente Demonstram

### 5.1 Melhora real mas insuficiente

De 3+2 para 12+8 épocas:
- pmi_cosine: Spearman 0.005 → 0.114 (+0.109), ROC-AUC 0.530 → 0.560 (+0.030)
- ppmi_jsd: Spearman -0.057 → 0.092 (+0.149), ROC-AUC 0.482 → 0.536 (+0.054)

A melhora é real e estatisticamente legível, mas os scores finais são próximos do acaso. Isso demonstra que mais épocas ajudam marginalmente mas não resolvem o problema fundamental.

### 5.2 A loss ainda decrescente NÃO invalida a conclusão sobre épocas

A loss ao final de 12 épocas no período 0 (4.78) ainda está claramente longe da convergência — o gráfico de loss mostra decréscimo monotônico sem platô visível. Isso significa que o modelo NÃO convergiu. Portanto, o experimento de 12+8 épocas não permite a afirmação "o modelo convergiu e mesmo assim o método falhou". Tecnicamente, "poucas épocas" não pode ainda ser descartado como explicação.

Contudo, a escala do gap entre a perplexidade atual (~125) e a perplexidade de modelos competentes (~5-20 em inglês) sugere que a quantidade de épocas necessária para convergência seria extraordinariamente grande para este modelo neste corpus — possivelmente 100+ épocas, ou seja, um problema de arquitetura/capacidade, não de cronograma de treinamento.

### 5.3 A correlação com entropia é estável entre rodadas

A dominância de entropia (rho ≈ ±0.94) é um sinal de que o score, em ambas as rodadas, mede principalmente diferença de calibração global entre checkpoints. A correlação com freq_t0 (rho = 0.39) indica um efeito adicional: palavras mais frequentes geram q_t(w) mais estável (lei de grandes números), produzindo perfis PMI com mais massa nas dimensões esperadas e portanto maior variação mensurável entre períodos. Esse efeito de frequência, combinado com o de calibração, produz o padrão observado sem nenhuma relação com mudança semântica real.

### 5.4 Os scores do piloto (3+2 épocas) são patológicos de modo diferente

No piloto, os scores de pmi_cosine para todos os 37 alvos variam entre 0.658 e 1.391 — amplitude enorme. No experimento 12+8, variam entre 0.155 e 0.452 — amplitude menor. Os verbos (circle_vb, pin_vb, stroke_vb, tip_vb) que tinham scores extremamente altos no piloto (1.37–1.39) passaram a scores medianos (0.23–0.31) no longo. Isso indica que o piloto produzia perfis extremamente instáveis (alta variância) enquanto o modelo mais treinado produz perfis mais compactos mas ainda sem sinal.

---

## 6. Análise do Caso `graft_nn`

`graft_nn` é o alvo com maior pmi_cosine no experimento 12+8 (0.452), e é classificado corretamente como changed (binary=1.0, graded=0.554). Contudo, o diagnóstico revela:

- freq_t0=119, freq_t1=109 (frequência estável, sem mudança de distribuição de ocorrências)
- entropy_t0=0.918, entropy_t1=0.982 (a entropia do perfil PPMI aumentou substancialmente de t0 para t1)
- entropy_abs_delta=0.064 (maior entre os top-10)

O alto score de pmi_cosine para graft_nn é portanto explicável pela alta variação de entropia (0.064), não por detecção de mudança semântica genuína. O mencionado top-PMI incoerente em t0 (steppe, reprobate, pus, baptist, ruff em vez de stock, bark, sap, scion, bud) confirma que:

1. O modelo não associa graft_nn aos termos de enxertia botânica que seriam semanticamente relevantes em 1810-1860.
2. Os tokens de alto PMI em t0 são tokens raros com probabilidade anormalmente alta em q_t(graft_nn) relativa ao prior p_t.
3. A diferença entre os perfis t0 e t1 reflete mudança de quais tokens raros dominam a cauda ruidosa, não mudança de campo semântico.

O fato de que graft_nn "acerta" por ser changed provavelmente é resultado de que palavras genuinamente polissêmicas como graft (enxerto + corrupção política) aparecem em contextos mais variados nos dois períodos, gerando q_t(w) com distribuição menos concentrada e portanto maior variação de entropia PPMI entre períodos — uma correlação espúria entre diversidade contextual e o score, não uma detecção de mudança de sentido.

---

## 7. Avaliação Crítica de q_t, p_t, Log-PMI e Métricas

### 7.1 q_t(w): distribuição condicional por ocorrência

A interpretação de q_t(w) como "perfil semântico" é válida em princípio, mas depende de que o MLM head produza saídas semanticamente informativas. A evidência empírica (perplexidade ~125, entropia normalizada ~0.95) indica que o modelo atual não satisfaz essa precondição. O MLM head prevê tokens principalmente com base em probabilidades de frequência geral (viés lexical) e padrões de posição (viés de posição central), não com base em contexto semântico. Nesse regime, q_t(w) é uma versão levemente perturbada da distribuição de frequência do corpus, modulada pelo contexto local de forma fraca.

A questão sobre o que o MLM head prevê (substitutos semânticos, reconstrução lexical, concordância morfossintática) não pode ser respondida diretamente pelos experimentos realizados. Um modelo com perplexidade ~125 não tem capacidade de fazer qualquer dessas coisas com precisão. BERT-base, com perplexidade ~4-6, mistura todas as três: prevê tokens que são lexicalmente plausíveis (reconstrução), gramaticalmente compatíveis (concordância morfossintática) e semanticamente relacionados (substitutos semânticos). A mistura é inevitável, mas o modelo treinado é demasiado ruidoso para exibi-la de forma detectável.

### 7.2 p_t: prior via [CLS][MASK][SEP]

O prior p_t = P(.|[CLS][MASK][SEP]) é matematicamente defensável como estimativa da distribuição marginal do modelo, mas tem dois problemas práticos severos:

**Problema 1 — Fora de distribuição:** O modelo foi treinado em janelas de 32 tokens com o token mascarado sempre na posição central (posição 16). A sequência [CLS][MASK][SEP][PAD]... coloca o [MASK] na posição 1, que o modelo nunca viu durante treinamento como posição de mascaramento. A distribuição resultante captura o comportamento do modelo em um input anômalo.

**Problema 2 — Não é marginalmente consistente com q_t(w):** Para que log(q_t(w)/p_t) meça PMI genuíno, p_t deve ser a marginal da mesma distribuição geradora que q_t(w). Mas q_t(w) é computada sobre ocorrências reais em janelas de 30 tokens, enquanto p_t é computada sobre uma janela de 1 token. São distribuições condicionadas a contextos estruturalmente incompatíveis.

### 7.3 Log-PMI como transformação

O log-PMI é matematicamente justificado quando q e p são estimativas consistentes das distribuições condicional e marginal. Quando q ≈ p (ambas próximas de uniforme), log(q/p) ≈ 0 na maior parte do vocabulário com exceção de tokens onde a estimativa de máxima verossimilhança é particularmente ruidosa. O eps=1e-9 afeta apenas tokens com probabilidade muito próxima de zero: com softmax sobre 27.311 tokens, tokens raros terão probabilidade da ordem de 1e-6 a 1e-8, e o eps entra na conta para esses tokens. Isso não introduz viés sistemático, mas também não resolve o problema de q ≈ p.

### 7.4 Métricas temporais: pmi_cosine e ppmi_jsd

**pmi_cosine (1 - cosseno):** mede mudança angular no espaço de 27.311 dimensões. Quando R_t(w) é dominado por ruído em dimensões de tokens raros, o cosseno mede semelhança entre padrões de ruído, não entre estruturas semânticas. O resultado é um score sensível a calibração diferencial entre checkpoints.

**ppmi_jsd:** mede divergência entre as distribuições de probabilidade induzidas pela parte positiva de R_t(w). Como PPMI retém apenas as dimensões onde q > p, e com q ≈ p quase uniforme, a parte positiva PPMI é determinada por quais tokens caem acidentalmente acima vs abaixo da linha p_t. Isso é essencialmente um comparador de ruído de estimação entre checkpoints.

**Conclusão sobre métricas:** ambas as métricas são matematicamente corretas para o objeto que medem, mas o objeto (perfil log-PMI de um modelo com perplexidade 125) não é um proxy válido para campo semântico de uma palavra. A culpa não está nas métricas, mas nos inputs que recebem.

---

## 8. Experimentos Discriminativos Usando Checkpoints Existentes

Os experimentos abaixo usam exclusivamente os checkpoints em `outputs/semeval2020_pmi_long_epochs_12_8/continual_real/checkpoint_t00.pt` e `checkpoint_t01.pt`. Nenhum novo treinamento é necessário.

---

### Exp-1: Probe de vizinhos semânticos conhecidos (sanity check fundamental)

**Hipótese testada:** O modelo associa palavras semanticamente relacionadas de forma detectável em q_t(w).

**Mudança exata:** Para palavras-alvo com mudança conhecida (graft_nn, record_nn, gas_nn), computar q_t(w) usando checkpoint_t00 e listar os 20 tokens com maior probabilidade. Comparar contra: (a) os substitutos semânticos esperados pelo sentido histórico, (b) tokens de alta frequência no corpus.

**Resultado esperado se hipótese correta:** os 20 tokens incluem substitutos semanticamente próximos do sentido histórico da palavra (para graft_nn em t0: bark, scion, bud, sap, stock; para gas_nn em t0: air, steam, coal, lamp, light).

**Resultado esperado se hipótese errada:** os 20 tokens são dominados por tokens de alta frequência geral (artigos, preposições, verbos auxiliares) ou por tokens raros sem relação semântica.

**Custo computacional:** 5-10 minutos de CPU. Apenas inferência sobre ~37 palavras com os checkpoints existentes.

**Critério de decisão:** se menos de 2 dos 20 tokens pertencem ao campo semântico esperado para qualquer palavra-alvo, o modelo não aprendeu representações semânticas úteis e todos os experimentos subsequentes de refinamento de prior/métrica são inúteis. Se 5+ tokens por palavra pertencem ao campo semântico, o problema está em p_t ou na métrica.

---

### Exp-2: Substituição de p_t por frequência empírica do corpus

**Hipótese testada:** O prior p_t = P(.|[CLS][MASK][SEP]) distorce os perfis PMI de forma mensurável.

**Mudança exata:** Substituir p_t pela distribuição de frequência empírica de unigrama: `p_freq[v] = count_t(v) / sum_t(v)`, computada diretamente do corpus. Recomputar todos os perfis R_t(w) com o novo prior e avaliar com o mesmo pipeline de evaluate_semeval2020_relational.py.

**Fórmula:** `R_t_freq(w)[v] = log(q_t(w)[v] + eps) - log(p_freq[v] + eps)`

O corpus de contagens já foi processado (o script `run_diachronic_relational_experiment.py` computa counts no setup, linhas 467-473). As distribuições q_t estão salvas em `outputs/semeval2020_pmi_long_epochs_12_8/profiles/prediction_anchor_js/t*.pt` como `full_distributions`.

**Resultado esperado se hipótese correta:** o Spearman graded aumenta de 0.114 para >0.25 e o padrão de entropia (rho ≈ -0.94) se dissipa.

**Resultado esperado se hipótese errada:** Spearman permanece em ~0.1 e o padrão de entropia persiste com mesma intensidade. Isso indicaria que o problema é em q_t, não em p_t.

**Custo computacional:** <5 minutos. Nenhum novo forward pass. Apenas recarregar `full_distributions` dos arquivos .pt salvos e recomputar log-PMI com o novo prior.

**Critério de decisão:** melhora de Spearman >0.1 pontos confirma que p_t é causa relevante. Melhora <0.05 pontos descarta p_t como causa principal.

---

### Exp-3: Substituição de p_t pela média de q_t sobre todos os alvos

**Hipótese testada:** O prior ideal é a média interna de q_t, que captura o comportamento médio do modelo nas posições reais de mascaramento, não em um input anômalo.

**Mudança exata:** `p_avg_t = (1/|W|) * sum_{w in W} q_t(w)`, computada sobre os 37 alvos. Recomputar R_t(w) usando p_avg_t como prior.

**Resultado esperado se hipótese correta:** os perfis R_t(w) passam a capturar desvios de q_t(w) em relação ao comportamento médio do modelo nas mesmas condições de mascaramento. Spearman aumenta.

**Resultado esperado se hipótese errada:** o padrão de entropia persiste porque q_t(w) já é quase uniforme para todos os alvos e a média deles também é quase uniforme — log(q_t(w)/p_avg_t) continua sendo ruído.

**Custo computacional:** <5 minutos. Mesmo raciocínio do Exp-2.

**Critério de decisão:** se Spearman não melhora com nenhum prior alternativo (Exp-2 e Exp-3), o problema está em q_t, não em p_t.

---

### Exp-4: Probe de invariância temporal com palavras estáveis

**Hipótese testada:** Se o método funciona, palavras sem mudança semântica (binary=0, graded≈0) devem ter pmi_cosine perto de 0. Se o método é dominated por calibração, mesmo palavras estáveis terão scores elevados.

**Mudança exata:** Inspecionar os scores das palavras estáveis no bottom-10 do diagnóstico (chairman_nn: graded=0, binary=0; risk_nn: graded=0, binary=0; lane_nn: graded=0.10, binary=0). Os scores são: chairman_nn=0.284, risk_nn=0.403, lane_nn=0.185.

Esses valores são substancialmente maiores que 0. A variância do score entre palavras estáveis é a baseline de ruído do método. Se o score de palavras estáveis é 0.15–0.40, então um score de 0.45 para graft_nn (genuinamente mudada) está dentro do ruído. Não é necessário novo experimento computacional — os dados existentes já permitem esta análise.

**Conclusão imediata:** a separação entre changed e unchanged é praticamente nula. O grupo changed tem score médio 0.334, o grupo unchanged 0.326. Diferença de 0.008, completamente dentro do ruído.

---

### Exp-5: Análise de entropia de q_t(w) como diagnóstico de capacidade do modelo

**Hipótese testada:** q_t(w) é semanticamente vazia (próxima da uniforme) para todos os alvos.

**Mudança exata:** Para cada alvo e período, computar a entropia de Shannon de q_t(w) (não do PPMI) e comparar com a entropia de uma distribuição uniforme de 27.311 tokens (ln(27311) ≈ 10.21 nats). Os arquivos `t*.pt` contêm `full_distributions` com q_t(w). Computar `H(q_t(w)) = -sum_v q_t(w)[v] * log(q_t(w)[v])` para cada palavra.

**Resultado esperado se hipótese correta:** H(q_t(w)) / ln(27311) > 0.95 para a maioria das palavras, confirmando que q_t(w) é quase uniforme.

**Resultado esperado se hipótese errada:** H(q_t(w)) / ln(27311) < 0.85 para palavras alteradas, indicando que o modelo concentra probabilidade em subconjuntos menores de tokens para essas palavras.

**Custo computacional:** <1 minuto. Apenas operações tensores sobre os perfis salvos.

**Critério de decisão:** se entropia normalizada de q_t(w) > 0.95 para todos os alvos, o modelo é definitivamente incapaz de produzir perfis semânticos e o problema é de capacidade. Nenhum ajuste de prior ou métrica resolverá isso.

**Nota:** A entropia normalizada do PPMI já está no diagnóstico e é 0.92–0.99. A entropia de q_t(w) diretamente (sem a transformação log-PMI) provavelmente é ainda mais próxima de 1.

---

### Exp-6: Comparação de perfis com modelo de referência pré-treinado

**Hipótese testada:** Um modelo com capacidade adequada produziria perfis semânticos coerentes para as mesmas palavras e corpus.

**Mudança exata:** Usar um modelo BERT pré-treinado disponível publicamente (bert-base-uncased, disponível via HuggingFace) para computar q_t(w) sobre o mesmo corpus SemEval-2020 para 5 palavras de referência. Comparar os top-20 tokens de q_t(graft_nn) entre o modelo local e o BERT-base.

**Resultado esperado:** BERT-base produziria tokens semanticamente coerentes (enxertia botânica em t0, corrupção política em t1). Se assim for, confirma-se que o problema é de capacidade do modelo treinado localmente, não do método em si.

**Custo computacional:** 30-60 minutos de CPU (ou 5-10 minutos de GPU). Requer download do BERT-base (~450MB).

**Critério de decisão:** se BERT-base produz top-20 coerentes mas o modelo local não, o problema é de capacidade. Se BERT-base também produz top-20 incoerentes para este corpus (que é pré-tokenizado por palavras simples, não subwords), há um problema adicional de compatibilidade de tokenização.

---

### Exp-7: Verificação de forgetting catastrófico via q_t do período 0 com checkpoint do período 1

**Hipótese testada:** O checkpoint do período 1 esqueceu as representações semânticas do período 0.

**Mudança exata:** Computar q_t0_from_theta1(w) = distribuição do modelo do período 1 aplicado a contextos do período 0. Comparar com q_t0(w) = distribuição do modelo do período 0 aplicado a contextos do período 0. Se o modelo esqueceu o período 0, q_t0_from_theta1(w) deve diferir substancialmente de q_t0(w) mesmo para palavras sem mudança semântica.

**Custo computacional:** 20-30 minutos de CPU. Reutiliza corpus e checkpoints existentes.

**Critério de decisão:** se KL(q_t0 || q_t0_from_theta1) para palavras estáveis é da mesma ordem que KL(q_t0 || q_t1) para palavras mudadas, então forgetting catastrófico explica uma fração significativa do score de mudança medido.

---

## 9. Testes Necessários Antes de Novo Treinamento

Antes de iniciar qualquer novo ciclo de treinamento (seja com mais épocas, modelo maior, ou prior diferente), os seguintes testes devem ser executados e documentados:

### T1 — Teste de substituição léxica em janelas conhecidas
Construir 10 frases de controle com contexto sintético claro (ex: "the surgeon performed a graft on the patient", "the politician was accused of graft and corruption"). Verificar se o modelo, com checkpoint_t00 e checkpoint_t01, prediz tokens semanticamente apropriados para cada contexto quando o alvo é mascarado. Este teste verifica se q_t(w) é sensível ao contexto local ou apenas ao tipo POS do token mascarado.

**Invariância exigida:** P_theta(surgeon | context_medical) > P_theta(surgeon | context_political) por pelo menos uma ordem de magnitude.

### T2 — Teste de simetria de PMI
Se R_t(w)[v] representa a associação de w com v, então o PMI deve ser aproximadamente simétrico: R_t(w)[v] ≈ R_t(v)[w] (até constantes de normalização). Para 5 pares de palavras conceitualmente associadas, verificar se a simetria se mantém. PMI altamente assimétrico indica que q_t é dominado por frequência, não por associação semântica.

**Invariância exigida:** |R_t(w)[v] - R_t(v)[w]| < 2 nats para pares semanticamente próximos.

### T3 — Teste de estabilidade de q_t com sub-amostras de contextos
Para palavras com muitas ocorrências (head_nn com 3599, word_nn com 4387), dividir os contextos aleatoriamente em duas metades e computar q_t(w) para cada metade separadamente. Medir KL(q_half1 || q_half2). Se essa variância intra-período é comparável à variância inter-período (o que o score de mudança mede), o estimador é instável para o número de ocorrências disponível mesmo com o modelo convergido.

**Invariância exigida:** KL(q_half1 || q_half2) < 0.1 * KL(q_t0 || q_t1) para palavras estáveis com >300 ocorrências.

### T4 — Teste de invariância a permutação de checkpoints
Para palavras genuinamente estáveis (chairman_nn, risk_nn), calcular pmi_cosine(R_t0, R_t1) com os checkpoints na ordem correta E na ordem invertida (R_t1 como "baseline"). Se o método funciona, as palavras estáveis devem ter scores baixos em ambas as direções. Se o score muda radicalmente quando os checkpoints são invertidos, o score mede assimetria de calibração, não mudança semântica.

### T5 — Teste de baseline aleatória
Substituir q_t(w) por amostras de Dirichlet(alpha * p_t) com alpha=100 (distribuição próxima de p_t mas com ruído controlado). Verificar se o Spearman resultante é comparável ao obtido com q_t(w) real. Se for, confirma-se que o "sinal" do método não supera o ruído de uma baseline aleatória.

### T6 — Documentar o limiar mínimo de ocorrências
Verificar quantos alvos têm menos de 10 ocorrências em algum período. A formalização menciona esse limiar como necessário; a implementação não o aplica. Palavras com <10 ocorrências devem ser filtradas e os resultados reavaliados.

---

## 10. Recomendação Final: Continuar, Reformular ou Abandonar

### Diagnóstico consolidado

O método log-PMI relacional tem fundamentos teóricos sólidos e é conceitualmente bem motivado. O problema atual não é o método, mas sua instância de implementação: um modelo com capacidade insuficiente (d_model=96, 2 camadas), treinado com mascaramento determinístico central, em um corpus de ~400k janelas por período, produz q_t(w) quase uniforme. Isso torna qualquer métrica derivada de q_t(w) uma medida de ruído de calibração, não de semântica.

### Recomendação estruturada

**Passo 1 (imediato, custo nulo):** Executar Exp-5 (entropia de q_t bruta) e a análise do Exp-4 (separação entre grupos changed/unchanged). Isso confirmará ou refutará em <1 hora se o problema é de capacidade do modelo.

**Passo 2 (se Exp-5 confirma capacidade insuficiente):** Substituir o modelo próprio por BERT-base-uncased (Exp-6). O método log-PMI relacional é agnóstico à arquitetura — pode ser aplicado a qualquer MLM pré-treinado usando os mesmos checkpoints do corpus para fine-tuning. Treinar BERT-base em DMLM (domain-adaptive MLM) no corpus SemEval por 1-3 épocas por período é computacionalmente viável em GPU moderada (2-4 horas).

**Passo 3 (se Exp-2/Exp-3 mostram que p_t importa):** Substituir prior por frequência empírica de corpus ou por média de q_t sobre alvos. Isso não requer novo treinamento.

**Passo 4 (se Exp-6 e novos priors ainda falham):** Revisar o mascaramento de treinamento para compatibilidade com MLM padrão (mascaramento aleatório 15%, com políticas de substituição padrão). Isso requer novo treinamento mas não mudança de arquitetura.

### Sobre abandono

Abandonar o método neste ponto seria prematuro: a falha observada é atribuível à capacidade do modelo, não ao método. Com um MLM competente (perplexidade <30 no corpus de domínio), o método tem probabilidade razoável de funcionar. A evidência para isso: quando o modelo tem mais épocas (12+8 vs 3+2), o Spearman melhora monotonicamente, sugerindo que com modelo mais capaz o sinal cresceria.

Contudo, **não se deve iniciar novo treinamento longo antes de confirmar com BERT-base (Exp-6) que o método produz perfis semanticamente coerentes com um modelo capaz**. Caso contrário, existe o risco de gastar recursos em treinamento de um modelo médio que ainda produz resultados marginais.

### Sobre manter pmi_cosine vs ppmi_jsd

Manter ambas por ora. pmi_cosine usa todo o perfil R_t(w), inclusive dimensões negativas (repulsão semântica), e é mais sensível a reorientação global do campo semântico. ppmi_jsd usa apenas associações positivas e é mais interpretável como mudança nas associações específicas da palavra. Com modelo de baixa qualidade, ppmi_jsd pode ser marginalmente mais robusto porque filtra dimensões com PMI negativo que são predominantemente ruidosas. Se com modelo melhor ambas melhoram, manter pmi_cosine como primária (mais informativa) e ppmi_jsd como secundária (mais interpretável).

---

## Apêndice: Invariâncias de Qualidade para o Pipeline

Antes de qualquer publicação, o pipeline deve satisfazer as seguintes invariâncias verificáveis:

1. **Invariância de frequência:** o Spearman(score, freq_t0) deve ser < 0.15. O atual 0.39 indica que o método mede frequência de corpus, não mudança semântica.
2. **Invariância de entropia:** o Spearman(score, entropy_abs_delta) deve ser < 0.30. O atual 0.944 indica colapso para medida de calibração.
3. **Sanidade de vizinhos:** para 90% dos alvos com >100 ocorrências, os 5 tokens com maior R_t0(w)[v] devem incluir pelo menos um sinônimo ou colocado semanticamente relacionado segundo WordNet ou outro recurso externo.
4. **Separação de grupos:** a diferença de score médio entre grupos binary=1 e binary=0 deve ser > 0.05 com p-value Wilcoxon < 0.05.
5. **Estabilidade intra-período:** KL entre q_t computado na primeira e segunda metade de cada corpus por período deve ser < 10% do KL inter-período para palavras estáveis com >200 ocorrências.

---

## 11. Tréplica para reavaliação pelo Claude Code

Esta seção registra uma contestação técnica ao parecer acima. Solicitamos que o
Claude Code reavalie o próprio diagnóstico, sem apagar ou reescrever o parecer
original, e acrescente sua resposta ao final deste arquivo.

Concordamos com pontos importantes do parecer:

- o probe `[CLS] [MASK] [SEP]` está fora da distribuição de treinamento;
- o mascaramento central determinístico pode gerar incompatibilidade com probes
  em posições variadas;
- não devemos iniciar outro treinamento longo antes de explorar os checkpoints
  existentes;
- priors alternativos, estabilidade por subamostragem e avaliação cruzada entre
  checkpoint e corpus são diagnósticos prioritários;
- o top-PMI incoerente mostra que o perfil atual não é semanticamente
  interpretável.

Contudo, algumas conclusões e testes propostos parecem incorretos ou mais fortes
do que a evidência permite.

### 11.1 A entropia de `q_t(w)` não é 0,92-0,99

O parecer afirma:

> Os perfis PPMI têm entropia normalizada em torno de 0.92-0.99, o que significa
> que q_t(w) é quase uniforme.

Essa inferência mistura dois objetos diferentes:

```text
q_t(w): distribuição bruta do MLM sobre o vocabulário

pi_t(w): distribuição obtida após max(0, log(q_t(w)/p_t)) e normalização L1
```

Os valores 0,92-0,99 citados pertencem à entropia normalizada de `pi_t(w)`, não
à entropia normalizada de `q_t(w)`.

As estatísticas já computadas para a rodada longa foram:

```text
H(q_t0) médio = 7.709 nats
H(q_t1) médio = 7.620 nats
log(|V|) = log(27311) ~= 10.215 nats
```

Logo:

```text
H(q_t0) / log(|V|) ~= 0.755
H(q_t1) / log(|V|) ~= 0.746
```

Isso não prova que `q_t(w)` seja semanticamente útil, mas contradiz a afirmação
de que ela é quase uniforme com entropia normalizada superior a 0,95.

**Pedido de reavaliação:** corrija a distinção entre entropia de `q_t`, entropia
de PPMI normalizado, massa positiva de PMI e concentração semântica. Diga quais
conclusões do parecer permanecem válidas após essa correção.

### 11.2 Perplexidade 119-131 não demonstra que o modelo é quase aleatório

O parecer converte a loss MLM em perplexidade e a compara com valores atribuídos
ao BERT-base:

```text
exp(4.78) ~= 119
```

Mas comparações de perplexidade entre modelos exigem, no mínimo, compatibilidade
de:

- vocabulário e tokenização;
- distribuição dos tokens mascarados;
- política de masking;
- corpus e domínio;
- denominador usado na loss;
- protocolo de validação;
- inclusão ou exclusão de tokens especiais e raros.

Nosso modelo usa vocabulário lexical de 27.311 tokens construído no corpus
histórico, palavras lematizadas com POS e exatamente um alvo mascarado por
janela. Compará-lo diretamente com uma suposta perplexidade 4-6 de BERT-base
parece tecnicamente injustificado.

Além disso, `exp(cross_entropy)` é uma perplexidade sobre a tarefa específica,
mas não mede diretamente qualidade semântica. Um modelo pode ter loss ainda alta
e conter informação contextual útil, ou ter loss baixa por explorar frequência
e sintaxe sem representar sentidos adequadamente.

**Pedido de reavaliação:** forneça uma base comparável para afirmar que loss
4,78 implica incapacidade semântica. Se não houver, reformule essa conclusão
como hipótese a ser testada por probes, não como causa primária estabelecida.

### 11.3 O top de `q_t(w)` não é uma lista de colocados contextuais

O Exp-1 espera que o top-20 de `q_t(graft_nn)` contenha:

```text
bark, scion, bud, sap, stock
```

Mas `q_t(w)` é a distribuição de preenchimento da posição ocupada por `w`.
Portanto, ela representa substituibilidade cloze média, não uma distribuição de
coocorrentes ou vizinhos de embedding.

Em uma frase como:

```text
the bark of the [MASK] and the stock precisely meet
```

`graft`, `scion` ou `cutting` podem preencher a lacuna. `sap` e `bark`,
embora pertençam ao campo botânico, provavelmente não são substitutos
gramaticalmente adequados em muitas ocorrências.

O mesmo vale para o sentido de corrupção: `corruption`, `bribery`, `fraud` e
`payoff` são candidatos mais defensáveis do que qualquer palavra apenas
associada ao contexto.

**Pedido de reavaliação:** defina critérios adequados para avaliar semanticidade
de uma distribuição cloze. Separe explicitamente:

- substitutos lexicais plausíveis;
- sinônimos;
- hiperônimos/hipônimos;
- colocados;
- palavras apenas pertencentes ao mesmo domínio.

Considere que a média sobre contextos heterogêneos pode não colocar nenhum
substituto específico no top-20, mesmo se distribuições por ocorrência forem
localmente informativas.

### 11.4 A loss inicial de t1 não demonstra forgetting

O parecer interpreta:

```text
loss final em D0 = 4.78
loss inicial durante treino em D1 = 6.02
```

como evidência de forgetting parcial. Essa diferença mostra principalmente que
`D1` tem distribuição diferente de `D0` e que o checkpoint treinado em `D0`
generaliza pior para `D1`. Isso é domain shift, não forgetting.

Forgetting exige medir desempenho anterior após aprender o novo período:

```text
loss(theta_0, D0)
versus
loss(theta_1, D0)
```

ou comparar:

```text
q_{D0, theta_0}(w)
versus
q_{D0, theta_1}(w)
```

O Exp-7 proposto no próprio parecer é apropriado para isso, mas a hipótese não
pode ser tratada como evidência observada antes de executá-lo.

**Pedido de reavaliação:** retire a loss inicial em `D1` como evidência direta de
forgetting e especifique a matriz mínima:

```text
theta_0 avaliado em D0
theta_0 avaliado em D1
theta_1 avaliado em D0
theta_1 avaliado em D1
```

necessária para separar adaptação, domain shift e forgetting.

### 11.5 A simetria proposta para o PMI não é uma invariância do nosso objeto

O teste T2 exige:

```text
R_t(w)[v] ~= R_t(v)[w]
```

Porém:

```text
R_t(w)[v] = log(q_t(w)[v] / p_t[v])
```

e `q_t(w)[v]` significa a probabilidade de o MLM preencher com `v` uma posição
real ocupada por `w`, média sobre os contextos de `w`.

Já `q_t(v)[w]` usa outro conjunto de contextos, outra distribuição sintática e
outra frequência. Isso não deriva de uma única distribuição conjunta simétrica
`P(w,v)`. Portanto, o nome "PMI" não torna essa construção automaticamente
simétrica.

Exemplo simples:

```text
q(car)[vehicle] pode ser alto
q(vehicle)[car] pode ser diluído entre muitos hipônimos
```

Essa assimetria pode ser semanticamente legítima.

**Pedido de reavaliação:** demonstre matematicamente sob quais hipóteses essa
simetria deveria valer para o estimador cloze atual. Caso não valha, retire T2
das invariâncias obrigatórias.

### 11.6 A inversão de checkpoints não testa assimetria

O teste T4 propõe inverter os checkpoints:

```text
d(R_t0, R_t1)
versus
d(R_t1, R_t0)
```

Mas as duas métricas atuais são simétricas:

```text
1 - cos(a,b) = 1 - cos(b,a)
JSD(a,b) = JSD(b,a)
```

Logo, inverter a ordem deve produzir exatamente o mesmo score, independentemente
de calibração ou semanticidade. Esse teste não diagnostica nada além da correta
implementação da simetria da métrica.

Uma alternativa informativa seria trocar quais contextos são aplicados a quais
checkpoints:

```text
q(theta_0, D0), q(theta_0, D1),
q(theta_1, D0), q(theta_1, D1)
```

**Pedido de reavaliação:** substitua T4 por um teste baseado nessa matriz
checkpoint por corpus.

### 11.7 Mais épocas ainda não estão formalmente eliminadas, mas perderam prioridade

Concordamos com a observação de que a loss ainda caía no epoch final. Portanto,
12+8 épocas não provam convergência e não eliminam logicamente a hipótese de
subtreino.

Entretanto, a rodada custou mais de duas horas, reduziu substancialmente a loss e
melhorou o Spearman apenas de 0,005 para 0,114. Os marcadores PMI continuaram
incoerentes e a correlação com variação de entropia permaneceu próxima de 0,94.

Assim, nossa conclusão é mais limitada:

> Mais épocas podem continuar melhorando o modelo, mas não são a próxima
> intervenção causalmente informativa. Antes disso, precisamos localizar se a
> falha está em q_t, p_t, log-PMI ou na comparação entre checkpoints.

**Pedido de reavaliação:** avalie essa formulação mais moderada. Não extrapole
que seriam necessárias 100+ épocas sem estimativa de curva de aprendizado ou
evidência experimental.

### 11.8 Sequência experimental que propomos

Propomos a seguinte ordem usando apenas checkpoints existentes:

1. **Inspecionar `q_t(w)` bruto**, globalmente e por ocorrência, com critérios de
   substituição cloze, sem passar por `p_t`.

2. **Medir concentração e sensibilidade contextual de `q_t`:**
   entropia normalizada, top-k mass, rank/probabilidade do token gold e mudança
   entre contextos semanticamente distintos.

3. **Split-half/bootstrap intra-período:** comparar a variabilidade de
   `q_t(w)` entre subconjuntos de ocorrências com a distância interperíodo.

4. **Matriz checkpoint por corpus:**

   ```text
   q(theta_0, D0), q(theta_0, D1),
   q(theta_1, D0), q(theta_1, D1)
   ```

   Isso separa efeito do contexto/corpus, efeito do checkpoint e interação entre
   ambos.

5. **Recalcular com priors alternativos sem novo forward pass:**

   - frequência empírica por período;
   - média de `q_t(w)` sobre alvos;
   - prior compartilhado entre períodos;
   - marginal MLM em posições mascaradas de contextos naturais.

6. **Aplicar shrinkage/truncamento**, por exemplo suporte mínimo em `p_t`,
   top-m por massa de `q_t`, ou combinação convexa com uma marginal robusta,
   avaliando estabilidade e não apenas Spearman.

7. **Somente depois**, comparar com um MLM pré-treinado ou retreinar com masking
   aleatório compatível com os probes.

### 11.9 Perguntas para a resposta do Claude Code

Acrescente uma resposta após esta tréplica, abordando explicitamente:

1. Quais conclusões originais você mantém após corrigir a entropia de `q_t`?
2. Você possui base comparável para usar perplexidade como evidência de ausência
   de semântica neste setup?
3. Qual é o teste correto de semanticidade para `q_t(w)` como distribuição
   cloze?
4. Você concorda que a loss inicial em `D1` mede domain shift, não forgetting?
5. Sob quais hipóteses a simetria `R(w)[v] ~= R(v)[w]` deveria valer?
6. Você concorda que inverter argumentos de cosseno/JSD é um teste vazio?
7. Qual experimento deve vir primeiro: inspeção de `q_t`, priors alternativos ou
   comparação com BERT? Justifique causalmente.
8. Após as correções, qual é sua ordenação revisada das causas prováveis?

Ao responder, diferencie claramente:

- fatos diretamente observados;
- inferências sustentadas pelos dados;
- hipóteses ainda não testadas;
- recomendações pragmáticas.
