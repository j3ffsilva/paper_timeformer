# Parecer Técnico e Científico: TimeFormer — Pós-Correção de Fronteiras de Documento

**Data:** 2026-06-06
**Escopo:** Auditoria independente do experimento `semeval2020_pmi_line_documents_3_2`
**Status:** NÃO foram feitas alterações em nenhum arquivo do projeto

---

## 1. Veredito Executivo

O método, na configuração atual, **não mede mudança semântica**. Os resultados são indistinguíveis de ruído: Spearman=-0.025 (p=0.88), ROC-AUC=0.494. A métrica `pmi_cosine` está rastreando mudança de entropia do corpus (rho=0.946 com `entropy_abs_delta`), não mudança semântica. Três causas concorrem e são verificáveis sem novo treinamento:

1. **Fato observado:** `q_t` bruto é quasi-uniforme (H/log|V|≈0.78 para graft_nn). O modelo não aprendeu distribuições cloze informativas para palavras-alvo.
2. **Fato observado com raiz identificada:** O mascaramento central determinístico no treino cria um desalinhamento de posição verificável no probe: 24% das ocorrências de `graft_nn` caem em posições de máscara (>16) que nunca foram vistas durante o treino.
3. **Fato observado:** O modelo é pequeno (d=96, 2 camadas, 4 cabeças) e subtreinado (3+2 épocas, loss final≈6.0), o que é incompatível com distribuições cloze informativas.

A correção de fronteiras (uma linha por documento) é **semanticamente correta** e os checkpoints antigos são **inválidos**. Mas a correção não é suficiente para fazer o método funcionar.

---

## 2. Validade da Correção de Fronteiras

**Pergunta:** A correção de tratar cada linha como documento separado é semanticamente correta para este dataset?

**Fato observado (lido no README do SemEval):** O README afirma explicitamente:

> "sentences randomly shuffled"
> "sentences are split at replacement tokens (10 x "@") and replacement tokens are removed"

O formato é **uma sentença por linha**, com sentido independente entre linhas. Cada linha é uma sentença proveniente de documentos históricos distintos (CCOHA/COHA), reordenadas aleatoriamente.

**Conclusão:** A correção está **correta**. Tratar o arquivo como um único documento conectado (o bug anterior) cria janelas MLM que conectam o fim de uma sentença com o início de outra, produzindo sequências semanticamente incoerentes que corrompem o sinal de treino. O treinamento com o bug ensinava o modelo a predizer continuações de fronteiras artificiais.

**Verificação adicional:** O corpus processado tem 253.644 linhas (corpus 1) e 353.692 linhas (corpus 2), confirmando o tratamento por sentença.

**Os checkpoints `semeval2020_pmi_pilot` e `semeval2020_pmi_long_epochs_12_8` devem ser considerados inválidos** para avaliar o método. O treino com fronteiras incorretas não pode servir como base de comparação. Esta conclusão é **fato**, não inferência.

---

## 3. Auditoria do Diagnóstico de q Bruto

**Pergunta:** O script `diagnose_cloze_semantics.py` mede corretamente `q_t` bruto? Há bugs, leakage, desalinhamento, ou problemas na classificação heurística de sentidos?

### 3.1 Correção do diagnóstico

**Fato observado (lido no código):**

A função `infer_occurrences` em `diagnose_cloze_semantics.py` (linhas 139-159):
```python
logits = out["logits"][batch_indices, batch["mask_pos"].to(device), :]
rows.append(torch.softmax(logits, dim=-1).cpu())
```
lê os logits na posição exata do [MASK] armazenada em `mask_pos` pelo dataset. O alinhamento entre contextos e tensores é garantido por:
1. A verificação explícita `if probabilities.size(0) != len(contexts): raise RuntimeError(...)` (linha 312-313).
2. O dataset `RealTargetOccurrenceDataset` percorre os documentos em ordem determinística (sem shuffle).
3. `occurrence_contexts()` usa exatamente o mesmo algoritmo de janelamento que o dataset.

**Conclusão:** Não há bug de leakage, desalinhamento de contextos/tensores, ou erro na extração de logits. O diagnóstico mede corretamente `q_t` bruto para cada ocorrência individual.

### 3.2 Problema identificado: desalinhamento de posição de máscara (H1)

**Fato observado (verificado por análise do código e dos dados):**

O dataset de treino `RealMLMDataset._make_item` usa mascaramento **central determinístico**:
```python
mask_pos = candidate_positions[len(candidate_positions) // 2]
```
Para uma janela de `content_len=30` tokens, a máscara cai sempre na posição central (posição 16 em `input_ids` para janelas completas).

O dataset de probe `RealTargetOccurrenceDataset._make_item` usa:
```python
mask_pos = mask_pos_in_window + 1  # posição real da ocorrência no contexto
```

**Quantificação empírica para `graft_nn` (corpus 1, 119 ocorrências):**
- Distribuição de posições de treino: média=12.0, desvio-padrão=4.1, range=[2, 16]
- Distribuição de posições de probe: média=14.2, desvio-padrão=8.0, range=[1, 30]
- **24.4% das ocorrências de `graft_nn` (29/119) caem em posições >16**, que nunca foram mascaradas durante o treino
- Apenas 33.6% das ocorrências caem dentro de ±2 posições da média de treino

A causa raiz: 70% das sentenças do corpus têm menos de 30 tokens (sentença média: 25.7 tokens). Para uma sentença de comprimento L < 30, o treino mascara a posição L//2+1 (centro da sentença), mas o probe mascara a posição real da palavra-alvo k+1 (que pode ser qualquer posição 1..L). O modelo tem embedding posicional absoluto (`nn.Embedding(seq_len, d_model)`): posições nunca vistas no treino produzem predições degeneradas.

**Isto é um bug de protocolo, não um bug de implementação do diagnóstico.**

### 3.3 Problema identificado: prior p_t fora de distribuição

**Fato observado (lido em `run_diachronic_relational_experiment.py`, linhas 179-204):**

A função `neutral_probe_distribution` usa o template `[CLS][MASK][SEP]` com MASK na posição 1. Esta é uma posição de baixa frequência no treino (treino concentra máscaras em posições 10-16). O prior p_t é estimado em uma configuração de inferência que o modelo também nunca viu de forma típica.

**Inferência:** O prior p_t reflete principalmente o bias posicional do modelo na posição 1, não uma distribuição marginal genuína. Isso contamina R_t(w) = log(q_t/p_t), mas é um problema secundário dado que q_t já é quasi-uniforme.

### 3.4 Classificação heurística de sentidos

A classificação de sentidos em `classify_context` é um conjunto de keywords simples sem stemming ou lematização extra. Para um corpus já lematizado, é razoável.

**Problema observado:** Para `graft_nn` no corpus 1 (1810-1860), a classificação retorna apenas "botanical" (n=77) e "other" (n=42). No corpus 2 (1960-2010), detecta "botanical" (n=4), "corruption" (n=24), "medical" (n=22), "other" (n=58), "ambiguous" (n=1). A classificação captura a mudança de sentido real de graft_nn (botânico → político/médico). Porém, a classe "other" engloba 58/109 ocorrências no corpus 2, o que significa que uma fração substancial das ocorrências de sentido moderno não é classificada.

**Conclusão:** A classificação é adequada para diagnóstico qualitativo, mas não exaustiva.

---

## 4. Interpretação dos Exemplos de graft_nn

**Pergunta:** Os exemplos concretos permitem concluir que `q_t` não é sensível ao sentido, ou o critério top-k/rank é inadequado?

**Fatos observados (lidos nos arquivos de diagnóstico):**

**Distribuições agregadas (`theta_0` em D0):**
- Top tokens: `[UNK]` (2.4%), `and` (2.4%), `the` (1.7%), `be` (1.7%), `of` (1.0%)
- Rank de `graft_nn`: 2930
- Probabilidade de `graft_nn`: 5.4×10⁻⁵
- Entropia normalizada: H/log|V| = 0.781

Nenhum dos substitutos semanticamente esperados (botanical: scion, branch, stock; corruption: bribery, scandal; medical: transplant) aparece no top-20 em qualquer combinação checkpoint×corpus.

**Análise por grupo de sentido (theta_0, D0):**
- Grupo "botanical" (n=77): topo é `[UNK]`, `and`, `the`, `be` — sem nenhum substituto botânico
- Grupo "other" (n=42): topo é `and`, `[UNK]`, `be`, `of` — idêntico ao agregado

Os exemplos individuais confirmam: mesmo em contextos inequivocamente botânicos como:
- "cut off and graft the top first give the [MASK] there the best possible chance while the necessary reduction of the top throw the sap into the remaining side branch"
  -> top: `[UNK]`, `and`, `man`, `own`, `time` (rank de `graft_nn`=2384)

E em contextos de corrupção no corpus 2 como:
- "but when the big [MASK] scandal come up there in waynesville fisher would n't prosecute"
  -> top: `and`, `be`, `to` (rank de `graft_nn`=6033)

**Conclusão:** q_t não é sensível ao sentido. Este é um **fato observado**, não uma inferência. As distribuições são dominadas por function words independentemente do contexto semântico. O critério top-k/rank é adequado para esta conclusão: se os substitutos esperados não aparecem nos top-2000 de 27.311 tokens, o modelo não está capturando o sentido.

**Ressalva:** O critério é correto mas é necessário distinguir entre duas causas:
- (a) O modelo não aprendeu nada sobre graft_nn especificamente (subtreino)
- (b) O modelo aprendeu mas o probe não consegue extrair (mismatch de posição)

A evidência aponta para **(a) e (b) simultâneos**: mesmo nos exemplos onde a máscara cai na posição central (pos=16, ~28% dos casos), os substitutos continuam sendo function words.

---

## 5. Análise da Matriz Checkpoint×Corpus

**Fatos observados (lidos em `report.md` para graft_nn):**

| Par | JSD | Tipo |
|-----|-----|------|
| theta_0@D0 vs theta_0@D1 | 0.022 | efeito de corpus (mesmo checkpoint) |
| theta_1@D0 vs theta_1@D1 | 0.039 | efeito de corpus (mesmo checkpoint) |
| theta_0@D0 vs theta_1@D0 | 0.109 | efeito de checkpoint (mesmo corpus) |
| theta_0@D1 vs theta_1@D1 | 0.086 | efeito de checkpoint (mesmo corpus) |

**Inferência:** O efeito de checkpoint (0.086–0.109) é 4-5× maior que o efeito de corpus (0.022–0.039). Isso indica que o treino continuado em D1 muda substancialmente as distribuições do modelo, independentemente do corpus de probe.

**Problema crítico:** A mudança de checkpoint não é devida a aprendizado de novos sentidos. Para `graft_nn`, o efeito de checkpoint (0.109) é a mudança de distribuições quasi-uniformes dominadas por function words entre theta_0 e theta_1. Com H/log|V|≈0.78, ambas as distribuições são quasi-uniformes, e JSD=0.109 entre duas distribuições quasi-uniformes indica uma mudança sutil no ruído, não em sinal semântico.

**Hipótese não testada:** A pergunta mais relevante seria: o efeito de checkpoint (0.109) é maior para palavras com mudança semântica real (binary=1) do que para palavras estáveis (binary=0)? Os dados do CSV de mudanças mostram que a resposta é **não** (ROC-AUC=0.494 implica que a ordenação por JSD é essencialmente aleatória com respeito ao label binário).

**O que tornaria a conclusão causal:** Seria necessário um controle com checkpoint aleatório (não treinado) para comparar os JSDs. Se JSD(random, random) ≈ 0.1, então o efeito de checkpoint observado é simplesmente ruído de inicialização/treino, não representação de mudança diacrônica. **Isso pode ser testado sem novo treinamento, usando pesos aleatórios.**

---

## 6. Causa Mais Provável, com Grau de Confiança

Listamos as hipóteses com grau de confiança e distinção fato/inferência/hipótese:

### H1: Mascaramento central determinístico (posição mismatch)
- **Grau de confiança: ALTO (confirmado como fato)**
- **Fato:** 24.4% das ocorrências de graft_nn no probe caem em posições nunca vistas no treino (>16). Distribuição de treino: range=[2,16]. Distribuição de probe: range=[1,30].
- **Fato:** 70% das sentenças têm <30 tokens; logo, em sentenças curtas, treino e probe mascaram posições diferentes na mesma sentença.
- **Inferência:** Com embedding posicional absoluto, predições em posições fora da distribuição de treino são degeneradas.
- **Limitação:** H1 explica parte da falha, mas não explica por que mesmo ocorrências com máscara na posição 16 (27% dos casos) produzem apenas function words.

### H2: Modelo subtreinado / pouco capaz
- **Grau de confiança: ALTO (fato)**
- **Fato:** Loss final = 5.96 (5 épocas totais). Entropia normalizada H/log|V|=0.78 para graft_nn (qualquer perfeito seria 0.0, modelo uniforme seria 1.0). O modelo mal se diferencia de uniforme.
- **Fato:** Arquitetura: d_model=96, 2 camadas, 4 cabeças. Para um vocabulário de 27.311 tokens, isso é muito pequeno. O rank médio de graft_nn é ~3.500 de 27.311, indicando que o modelo não tem preferência clara pelo target mesmo em contextos óbvios.
- **Inferência:** Mais épocas e/ou modelo maior são necessários, mas não suficientes se H1 não for corrigido.

### H3: Média cloze apaga multimodalidade
- **Grau de confiança: MÉDIO (plausível mas não separável do H2 no estado atual)**
- **Hipótese não testada:** Se q_t fosse informativa em cada ocorrência, a média poderia ainda perder o sinal multimodal. Mas como q_t individual já é quasi-uniforme (ver exemplos com H(q)/log|V|≈0.85 em exemplos individuais), o problema está na inferência individual, não só na média.
- **Nota:** A média é matematicamente legítima se as distribuições individuais fossem peaks em tokens diferentes por sentido. O diagnóstico mostra que não são peaks em nenhum token relevante.

### H4: Prior p_t fora de distribuição amplifica ruído
- **Grau de confiança: BAIXO (secundário)**
- **Fato:** p_t usa MASK na posição 1 (fora da distribuição de treino). Isso contamina R_t.
- **Inferência:** O efeito é secundário porque o problema principal é que q_t é quasi-uniforme. Mesmo com p_t perfeito, R_t ≈ log(uniforme/marginal) ≈ 0 para todos os tokens.

**Causa mais provável: H1 + H2 são co-primárias.** H1 pode ser corrigido sem novo treinamento. H2 requer novo treinamento com modelo maior e mais épocas, mas somente após H1 ser avaliado.

---

## 7. A Definição Cloze Ainda É Defensável?

**Pergunta:** A distribuição cloze média pode em princípio medir mudança de sentido multimodal?

**Análise:**

A definição `q_t(w) = media{P(v | c_i, theta_t) : c_i contém w}` é matematicamente uma mistura de distribuições condicionais. Em princípio:
- Se `graft_nn` tem dois sentidos (botânico e corrupção) com distribuições multimodais `q_botanical` e `q_corruption`, a média seria uma mistura com massa em ambos os conjuntos de tokens.
- Entre períodos, a proporção de sentidos muda (D0: 77/119 botânico; D1: 4/109 botânico).
- A média q_t mudaria refletindo essa proporção.

**Porém:** Para que isso funcione, o modelo precisaria produzir distribuições informativas por sentido. O diagnóstico mostra que **não produz**. Mesmo no grupo "botanical" (theta_0, D0, n=77), o top token é `[UNK]` e `scion` não aparece no top-20.

**Inferência:** A definição cloze é defensável **em teoria** para um modelo com capacidade adequada. Ela falha **na prática atual** porque:
1. O modelo não aprendeu substituições lexicais informativas
2. O mismatch de posição de máscara produz predições fora de distribuição

**Hipótese não testada:** Se um MLM pré-treinado de grande escala (BERT-base ou RoBERTa) fosse usado com o mesmo protocolo, as distribuições cloze seriam informativas? Isso separaria a questão da definição da questão do modelo.

**Comparação de mixtures vs médias:** Usar JSD entre distribuições individuais (não agregadas) ou modelagem de mistura (GMM sobre distribuições por ocorrência) poderia capturar multimodalidade que a média apaga. Esta é uma extensão válida, mas requer que o modelo subjacente produza distribuições informativas primeiro.

---

## 8. Experimentos Mínimos, Ordenados por Informação/Custo

Os experimentos abaixo usam **exclusivamente os checkpoints existentes** em `outputs/semeval2020_pmi_line_documents_3_2/continual_real/` (checkpoint_t00.pt e checkpoint_t01.pt). Nenhum novo treinamento até o Experimento 5.

### Experimento 1: Controle com pesos aleatórios (custo: minutos)

**Hipótese testada:** O JSD checkpoint×corpus observado é indistinguível de ruído de inicialização aleatória.

**Protocolo:**
- Instanciar 2 modelos com pesos aleatórios (mesma arquitetura, seeds diferentes)
- Executar `diagnose_cloze_semantics.py` com esses pesos no lugar dos checkpoints
- Comparar a matriz JSD resultante com a matriz observada

**Arquivos/checkpoints:** Nenhum checkpoint necessário — pesos aleatórios da mesma classe `RealStaticMLM`

**Custo:** < 10 minutos de CPU

**Resultado se hipótese correta:** JSD(random_0, random_1) ≈ JSD(theta_0, theta_1) ≈ 0.1. O efeito de checkpoint observado não seria evidência de aprendizado.

**Resultado se hipótese errada:** JSD(random) << 0.1. O treino produziu mudança real nas distribuições (mas ainda não semânticamente válida).

**Critério de decisão:** Se JSD(random) > 0.05 × JSD(theta), o resultado é compatível com ruído. Escrever 3 linhas de código, não um script completo.

---

### Experimento 2: Teste de posição — ocorrências recentralizadas (custo: horas de CPU)

**Hipótese testada:** O mismatch de posição de máscara é a causa principal da degeneração de q_t.

**Protocolo:**
- Modificar `RealTargetOccurrenceDataset._make_item` para forçar o alvo ao centro do contexto: ao construir o probe, usar sempre uma janela de 30 tokens centrada no alvo, preenchendo com padding se necessário, e mascarar sempre na posição 16.
- Alternativamente, filtrar as ocorrências de graft_nn para manter apenas as que já caem na posição 16 (n=33 de 119) e reexecutar o diagnóstico.
- Comparar H(q)/log|V|, rank do alvo, e top-tokens antes e após recentralização.

**Arquivos:** Modificação em `src/timeformers/real_corpus.py` (apenas para o probe, não o treino) ou filtro no CSV `occurrences.csv` já existente.

**Custo:** Modificação de 5-10 linhas de código. Inferência: < 30 minutos de CPU com os checkpoints existentes.

**Resultado se hipótese correta:** Ocorrências com máscara na posição 16 produzem distribuições com H/log|V| significativamente menor (mais concentradas) e ranks de substitutos relevantes < 500.

**Resultado se hipótese errada:** Mesmo com máscara centralizada, as distribuições permanecem quasi-uniformes. O problema é o modelo, não a posição.

**Critério de decisão:** Comparar H/log|V| médio e rank médio do alvo entre ocorrências centrais (pos=16) vs não-centrais (pos≠16). Se a diferença for < 5%, H1 não é a causa dominante.

**Nota crítica:** O arquivo `occurrences.csv` já contém `mask_pos` implícito nas posições das ocorrências. Pode-se calcular quais das 119 ocorrências caem na posição 16 sem rodar novamente o modelo.

---

### Experimento 3: Rank da palavra-alvo por posição de máscara (custo: trivial, dados já existentes)

**Hipótese testada:** O rank do alvo é sistematicamente melhor para ocorrências com máscara próxima da posição de treino.

**Protocolo:**
- No arquivo `occurrences.csv` existente, calcular a posição de máscara para cada ocorrência (usando o mesmo algoritmo de `_make_item`)
- Plotar rank do alvo vs posição de máscara
- Calcular correlação de Spearman entre posição de máscara e rank do alvo

**Arquivos:** `outputs/semeval2020_pmi_line_documents_3_2/cloze_diagnostics/graft_nn/occurrences.csv` (já existe)

**Custo:** 10 linhas de Python, sem GPU, sem I/O de modelos.

**Resultado se hipótese correta:** Ocorrências com máscara perto da posição 12-16 têm rank do alvo sistematicamente menor (melhor). Correlação Spearman significativa entre |mask_pos - 14| e rank.

**Resultado se hipótese errada:** Sem correlação entre posição e rank.

**Critério de decisão:** Se |rho| > 0.2 (p < 0.05), H1 é um fator contribuinte real.

---

### Experimento 4: Probe com frases idênticas deslocadas por padding (custo: horas de CPU)

**Hipótese testada:** O modelo tem forte dependência de posição absoluta para suas predições.

**Protocolo:**
- Pegar as 10 ocorrências de graft_nn com melhor contexto semântico
- Para cada ocorrência, criar variantes com diferentes quantidades de padding à esquerda: [PAD]×k + [CLS] + contexto + [MASK] + ...
- Medir como a distribuição muda com o deslocamento de posição
- Comparar com a linha de base (sem padding extra)

**Custo:** Código novo, mas trivial. Inferência com CPU, < 1 hora.

**Resultado se hipótese correta:** A distribuição muda drasticamente com deslocamento de posição. O rank do alvo muda em >1000 posições com deslocamento de 4 tokens.

**Resultado se hipótese errada:** A distribuição é estável ao longo de deslocamentos. O modelo é invariante à posição (improvável dado embedding posicional absoluto).

---

### Experimento 5: Recentralizar o probe e reexecutar a avaliação completa (custo: horas de CPU)

**Hipótese testada:** Corrigir H1 é suficiente para recuperar sinal semântico nos checkpoints existentes.

**Protocolo:**
- Modificar `RealTargetOccurrenceDataset._make_item` para sempre colocar o alvo na posição central (pos=16), com padding se necessário para sentenças curtas.
- Reexecutar `run_diachronic_relational_experiment.py` com os checkpoints existentes (apenas a etapa de probe, não o treino).
- Comparar métricas de avaliação (Spearman, ROC-AUC) com a baseline atual.

**Custo:** Modificação de < 10 linhas. Inferência: 1-2 horas de GPU com os checkpoints existentes.

**Resultado se hipótese correta:** Spearman melhora de -0.025 para > 0.2 e ROC-AUC melhora para > 0.6.

**Resultado se hipótese errada:** Métricas permanecem próximas de chance. H1 não é a causa dominante; H2 (capacidade/subtreino) é o gargalo.

**Critério de decisão:** Se Spearman > 0.15 e ROC-AUC > 0.6 após recentralização, H1 era a causa dominante e a solução é simples. Se não, passar ao Experimento 6.

---

### Experimento 6 (apenas se Exp.5 falhar): Masking aleatório (custo: dias de GPU)

**Hipótese testada:** O modelo aprenderia q_t informativo se treinado com masking aleatório (como BERT) em vez de central determinístico.

**Protocolo:**
- Modificar `RealMLMDataset._make_item` para mascarar uma posição aleatória (com probabilidade 15% para qualquer token, como BERT).
- Treinar novamente do zero: 3+2 épocas, mesma arquitetura.
- Comparar H(q)/log|V| e métricas finais.

**Custo:** 1-2 dias de GPU (estimativa baseada no tempo de treino anterior).

**Critério de decisão:** Se H/log|V| < 0.5 após treino com masking aleatório, H1 era resolvível. Se ainda > 0.7, H2 (capacidade) é o problema.

---

### Experimento 7 (apenas se Exp.6 falhar): MLM pré-treinado de grande escala

**Hipótese testada:** A definição cloze q_t é válida com um modelo com capacidade suficiente.

**Protocolo:**
- Substituir o modelo treinado por BERT-base ou RoBERTa-base (via HuggingFace).
- Usar o corpus lematizado do SemEval com o probe de ocorrência.
- Comparar com os resultados dos melhores sistemas do SemEval-2020 (baseline: SGNS, UCD, etc.).

**Custo:** Sem treinamento (usar checkpoint pré-treinado). Inferência: horas de GPU.

**Critério de decisão:** Se Spearman > 0.4 com BERT pré-treinado, a definição cloze é válida e o problema era capacidade. Se não, a definição tem limitações fundamentais.

---

## 9. Critérios Objetivos para Continuar ou Abandonar Cada Caminho

### Caminho A: Corrigir masking (H1)
- **Continuar:** Experimento 3 mostra |rho| > 0.2 entre posição de máscara e rank do alvo.
- **Abandonar:** Experimento 2 mostra que ocorrências centralizadas têm H/log|V| > 0.70 (ainda quasi-uniformes). Passa-se diretamente ao Experimento 6.

### Caminho B: Mais épocas / modelo maior (H2)
- **Pré-condição:** Executar Experimento 5 primeiro. Só escalar o modelo após confirmar que H1 não é suficiente.
- **Continuar:** Exp.5 melhorou métricas (ROC-AUC > 0.6) mas não atingiu o estado da arte (ROC-AUC > 0.7). Mais épocas podem ajudar.
- **Abandonar:** Exp.7 (BERT pré-treinado) também falha (ROC-AUC < 0.55). Isso indicaria que a definição cloze tem limitações fundamentais para este dataset.

### Caminho C: Mixtures vs médias (H3)
- **Pré-condição:** O modelo precisa produzir distribuições individuais com peaks em tokens relevantes (não function words).
- **Continuar:** Após Exp.6 ou Exp.7, se as distribuições individuais mostrarem bimodalidade por sentido mas a média as achatar.
- **Abandonar:** Se distribuições individuais continuarem quasi-uniformes mesmo após resolver H1 e H2.

### Caminho D: Abandonar q_t médio
- **Quando abandonar:** Se Experimento 7 (BERT pré-treinado com masking aleatório) produzir ROC-AUC < 0.55, a definição cloze média é inadequada para este task e dataset. Considerar abordagens de representação contextual (type embeddings por uso, WSD via clustering, ou modelos de mistura sobre usos individuais).

---

## 10. Recomendação Final

**Sequência obrigatória antes de qualquer novo treinamento:**

1. **Imediatamente** (sem código): Calcular correlação entre posição de máscara e rank do alvo no `occurrences.csv` existente (Experimento 3). Custo: 15 minutos. Resultado esperado: rho significativo, confirmando H1.

2. **Esta semana** (modificação de 5 linhas): Filtrar as ocorrências de graft_nn com máscara na posição central (pos=16) e reexecutar o diagnóstico com os checkpoints existentes (Experimento 2, versão por filtro). Custo: < 1 hora.

3. **Esta semana** (modificação de 10 linhas + inferência): Recentralizar o probe e reexecutar a avaliação completa com os checkpoints existentes (Experimento 5). Custo: 1-2 horas de GPU. **Este experimento decide se H1 é suficiente ou se H2 domina.**

4. **Somente se Experimento 5 falhar**: Treinar com masking aleatório (Experimento 6). Somente se isso também falhar: avaliar com MLM pré-treinado (Experimento 7).

**Sobre a definição cloze:** A definição é defensável em teoria e não deve ser abandonada com base apenas nos resultados atuais. Os resultados refletem um modelo inadequado e um protocolo de probe com mismatch de posição, não necessariamente uma falha da definição. A avaliação justa da definição requer um modelo competente.

**Sobre a abordagem geral:** A hipótese de que o modelo aprende um "círculo semântico" diacrônico é plausível mas não verificável com o modelo atual. O perfil relacional R_t(w) = log(q_t/p_t) é uma operacionalização razoável, mas requer que q_t seja informativo. A prioridade técnica imediata é verificar se H1 é resolvível sem novo treinamento.

**O que não se deve fazer:** Treinar modelos maiores ou por mais épocas antes de verificar se H1 (masking central) é a causa dominante. Esse experimento pode custar dias de GPU e confirmar algo que poderia ser verificado em horas com os checkpoints existentes.

---

## Apêndice: Resumo dos Fatos, Inferências e Hipóteses

| Afirmação | Tipo | Evidência |
|-----------|------|-----------|
| Cada linha é um documento independente no SemEval | Fato | README: "sentences randomly shuffled" |
| Checkpoints pmi_pilot e pmi_long_epochs_12_8 são inválidos | Fato | Bug de fronteiras confirmado + README |
| q_t(graft_nn) é quasi-uniforme (H/log\|V\|=0.78) | Fato | report.md, report.json |
| Top tokens de q_t são function words, não substitutos semânticos | Fato | report.md, todas as células |
| 24.4% das probes de graft_nn caem em posições nunca treinadas | Fato | Análise do código + contagem de posições |
| O mismatch de posição degrada predições do modelo | Inferência | Embedding posicional absoluto + análise de distribuição |
| Corrigir H1 melhorará Spearman/ROC-AUC substancialmente | Hipótese não testada | Aguarda Experimento 5 |
| O modelo é subtreinado para o task | Fato | Loss=6.0, entropia quasi-uniforme |
| pmi_cosine rastreia entropia em vez de semântica | Fato | rho=0.946 com entropy_abs_delta |
| A definição cloze média pode medir mudança multimodal com modelo adequado | Hipótese não testada | Requer Experimento 7 |
| JSD checkpoint×corpus (0.086-0.109) reflete mudança semântica real | Hipótese não testada | Requer controle com pesos aleatórios (Exp.1) |
