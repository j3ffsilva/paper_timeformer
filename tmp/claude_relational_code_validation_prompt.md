# Prompt para validação externa do código relacional

Você deve realizar uma auditoria técnica e científica independente do projeto:

`/Users/jeff/Documents/trabalhos/papers/paper-timeformers`

Sua tarefa é verificar se o código atual realmente implementa a pergunta
científica descrita abaixo, identificar riscos à validade e recomendar mudanças
antes da execução com múltiplas seeds.

Não altere arquivos de código, documentação ou resultados. Escreva somente seu
parecer final em:

`./tmp/timeformer_relational_code_validation_review.md`

O parecer deve ser autocontido, concreto e baseado na inspeção do código e dos
artefatos produzidos.

---

## 1. O que queremos investigar

Queremos observar mudança semântica durante o treinamento cronológico de um
Transformer padrão, sem fornecer ao modelo qualquer identificador de tempo.

O treinamento pretendido é:

```text
theta_0 = treinar um Transformer padrão nos textos do período D_0
theta_1 = continuar o treinamento de theta_0 nos textos de D_1
theta_2 = continuar o treinamento de theta_1 nos textos de D_2
...
theta_t = continuar o treinamento de theta_{t-1} nos textos de D_t
```

Não queremos:

- inserir embeddings de período;
- condicionar o Transformer explicitamente por tempo;
- aprender um vetor temporal adicional;
- aprender trajetórias com teacher/student;
- comparar diretamente coordenadas absolutas entre checkpoints;
- alinhar espaços vetoriais como requisito da formulação principal.

Queremos medir, depois de cada checkpoint, como as relações semânticas internas
de cada palavra mudaram.

---

## 2. Exemplo trivial e concreto da ideia

Suponha três palavras em `t0`:

```text
sim_t0(banco, dinheiro) = 0.90
sim_t0(banco, rio)      = 0.20
sim_t0(banco, cadeira)  = 0.10
```

O perfil relacional de `banco@t0` é aproximadamente:

```text
r_t0(banco) = [dinheiro: 0.90, rio: 0.20, cadeira: 0.10]
```

Depois de continuar o treinamento com textos de `t1`, obtemos:

```text
sim_t1(banco, dinheiro) = 0.40
sim_t1(banco, rio)      = 0.85
sim_t1(banco, cadeira)  = 0.10
```

Então:

```text
r_t1(banco) = [dinheiro: 0.40, rio: 0.85, cadeira: 0.10]

delta_rel(banco, t0, t1)
  = r_t1(banco) - r_t0(banco)
  = [dinheiro: -0.50, rio: +0.65, cadeira: 0.00]
```

Interpretamos isso como:

- `banco` afastou-se de `dinheiro`;
- `banco` aproximou-se de `rio`;
- a relação com `cadeira` permaneceu estável.

Não importa se todo o espaço em `t1` foi rotacionado, refletido ou reposicionado.
Se todas as relações internas permanecerem iguais, a mudança semântica
relacional deve ser zero.

Formalmente:

```text
r_t(w)[v] = similaridade_t(w, v)
delta_rel(w, a, b) = r_b(w) - r_a(w)
```

A trajetória não precisa ser aprendida. Ela pode ser analisada posteriormente
a partir da sequência:

```text
r_0(w), r_1(w), ..., r_t(w)
```

---

## 3. Nuance importante: palavra própria estável versus relação estável

No corpus sintético, uma palavra classificada como `stable` mantém constante sua
própria probabilidade de pertencer ao contexto N1.

Entretanto, seu perfil relacional ainda pode mudar porque outras palavras se
movem. Exemplo:

```text
t0:
  A = 0.8
  B = 0.8

t1:
  A = 0.8  # A permaneceu estável
  B = 0.2  # B mudou
```

Mesmo que `A` não tenha mudado sua propriedade própria, sua relação com `B`
mudou. Portanto, a auditoria deve distinguir:

1. estabilidade da propriedade plantada da palavra;
2. estabilidade de seu perfil relacional diante do movimento das demais.

Verifique se o código e as métricas atuais tratam essa distinção corretamente e
se o chamado `oracle` relacional está matematicamente adequado.

---

## 4. Como o experimento atual tenta operacionalizar a ideia

### 4.1 Corpus sintético

Cada sujeito `S1`, `S2`, etc. possui, em cada período, uma probabilidade
plantada `p_n1` de ocorrer com contextos do grupo N1 em vez de N2.

Existem quatro classes de trajetória:

- `stable`;
- `drift`;
- `bifurcating`;
- `abrupt`.

### 4.2 Treinamento

O modelo principal é um Transformer `Static`, sem condicionamento temporal.

No experimento relacional, o dataset `ContextPairMLMDataset` mascara verbo e
objeto simultaneamente:

```text
[CLS] sujeito [MASK] [MASK] [SEP]
```

O objetivo é obrigar o modelo a prever os contextos com base no sujeito, pois o
MLM anterior, que mascarava somente verbo ou objeto, permitia resolver a tarefa
usando o outro marcador contextual.

### 4.3 Representações avaliadas

O código compara quatro modos:

1. `subject_prediction_probes`
   - entrada: `[CLS] sujeito [MASK] [MASK] [SEP]`;
   - representação: média dos estados ocultos pós-Transformer nas duas posições
     mascaradas;
   - atualmente é o modo principal mais promissor.

2. `subject_only_probes`
   - mesma entrada neutra;
   - representação: estado oculto na posição do sujeito.

3. `fixed_probes`
   - conjunto fixo e balanceado de contextos concretos;
   - representação: centroide dos estados do sujeito.

4. `in_corpus`
   - ocorrências reais de cada período;
   - representação: centroide dos estados contextuais do sujeito.

Para cada modo, o código calcula uma matriz de similaridades entre sujeitos
dentro de cada checkpoint.

### 4.4 Controles

O experimento executa:

```text
continual_real:
  D_0 -> D_1 -> ... -> D_9

continual_placebo:
  D_0 -> D_0 -> ... -> D_0

frozen:
  theta_0 aplicado às consultas sem atualizar pesos
```

O placebo pretende estimar a deriva relacional causada apenas por continuar a
otimização, mesmo sem trocar o corpus.

Também existe validação por período, parada antecipada e restauração do melhor
checkpoint daquele período.

### 4.5 Mudança contrafactual

Além da mudança relacional bruta, calculamos:

```text
delta_excedente = delta_real - delta_placebo
```

Também comparamos a direção relacional aprendida com uma direção oráculo
construída a partir dos valores sintéticos `p_n1`.

O código registra separadamente:

- direção observada versus oráculo;
- direção placebo versus oráculo;
- direção excedente versus oráculo.

Não assuma que subtrair o placebo dessa forma é necessariamente válido.
Avalie criticamente se os deltas real e placebo são comparáveis, se a subtração
é identificável e se há controles ou alternativas melhores.

---

## 5. Resultado atual que precisa ser validado

O experimento validado mais recente está em:

`outputs/relational_continual_validated/seed_1000`

Configuração aproximada:

- seed: `1000`;
- 100 ocorrências por sujeito/período;
- até 60 épocas em `t0`;
- até 30 épocas nos períodos posteriores;
- validação e parada antecipada;
- Transformer estático;
- mascaramento simultâneo do par contextual.

No modo `subject_prediction_probes`, para `t0 -> t9`, foram observados:

| Classe | Direção observada | Direção placebo | Direção excedente |
|---|---:|---:|---:|
| abrupt | +0.960 | +0.559 | +0.789 |
| bifurcating | +0.877 | +0.677 | +0.290 |
| drift | +0.944 | +0.549 | +0.697 |
| stable | +0.962 | +0.663 | +0.799 |

Nos períodos iniciais, porém, a direção é fraca ou negativa. O placebo também
apresenta direção positiva em vários casos.

Precisamos saber se o resultado positivo representa mudança semântica relacional
recuperada corretamente ou se decorre de vazamento do gerador, desenho do probe,
forma do oráculo, treinamento prolongado, escolha de checkpoint ou outra
construção circular.

---

## 6. Arquivos que devem ser auditados

Inspecione pelo menos:

- `src/timeformers/corpus.py`
- `src/timeformers/dataset.py`
- `src/timeformers/models.py`
- `src/timeformers/train.py`
- `src/timeformers/representations.py`
- `src/timeformers/relational.py`
- `src/timeformers/relational_metrics.py`
- `scripts/run_relational_continual_sanity.py`
- `tests/test_relational.py`
- `docs/relational_change_current_plan.md`

Consulte também:

- `docs/novo_planejamento.md`, para entender a direção anterior;
- `tmp/timeformer_relational_change_second_opinion.md`;
- `outputs/relational_continual_validated/seed_1000/config.json`;
- `outputs/relational_continual_validated/seed_1000/relational_summary.csv`;
- `outputs/relational_continual_validated/seed_1000/counterfactual_summary.csv`;
- históricos e checkpoints desse diretório, quando necessário.

---

## 7. Questões obrigatórias da auditoria

### 7.1 Correspondência entre intenção e código

1. O código realmente treina um único Transformer cronologicamente, preservando
   modelo e estado do otimizador entre períodos?
2. Existe algum sinal temporal explícito ou indireto indevido chegando ao
   modelo?
3. Os checkpoints representam corretamente o melhor estado de cada período?
4. O `continual_placebo` começa exatamente no mesmo estado e recebe condições
   comparáveis ao `continual_real`?
5. O controle `frozen` mede o que afirma medir?

### 7.2 Construção das representações

1. O `ContextPairMLMDataset` mascara corretamente ambos os marcadores e produz
   labels corretos?
2. O probe preditivo é uma operacionalização defensável de `palavra@tempo`?
3. Usar a média dos estados ocultos nas posições mascaradas é justificável?
4. Esse probe mede semântica aprendida ou apenas comportamento do cabeçote MLM?
5. Deveríamos comparar estados ocultos, logits, probabilidades sobre contexto,
   embeddings de entrada ou mais de uma dessas opções?
6. Existe vazamento entre treinamento, validação, probes e oráculo?
7. Os centroides usados em `fixed_probes` e `in_corpus` são adequados?

### 7.3 Métricas relacionais

1. A matriz de similaridade por cosseno é apropriada?
2. As métricas são realmente invariantes a mudanças irrelevantes do sistema de
   coordenadas?
3. Jaccard, Spearman, média absoluta e CKA estão implementados corretamente?
4. O cálculo de direção relacional por sujeito está correto?
5. O oráculo baseado em vetores `[p_n1, 1-p_n1]` é válido?
6. Esse oráculo cria artificialmente o resultado positivo?
7. O tratamento da diagonal e dos demais sujeitos está correto?
8. Como devemos tratar palavras estáveis cuja vizinhança muda porque outras
   palavras mudaram?

### 7.4 Placebo e identificação causal

1. É válido calcular `delta_real - delta_placebo`?
2. Real e placebo permanecem comparáveis após trajetórias diferentes de
   otimização e parada antecipada?
3. Um placebo positivo na direção do oráculo indica problema metodológico?
4. Seria melhor usar:
   - múltiplos placebos;
   - várias seeds pareadas;
   - permutação dos períodos;
   - repetição de cada período;
   - bootstrap;
   - diferença-em-diferenças;
   - normalização pela distribuição nula;
   - comparação estatística em vez de subtração vetorial?

### 7.5 Validade experimental

1. O corpus sintético torna a tarefa trivial ou circular?
2. O modelo está apenas aprendendo a frequência N1/N2?
3. Mesmo nesse caso, o experimento constitui uma validação adequada da medida
   relacional?
4. Há confundimento entre número de atualizações, mudança de corpus,
   esquecimento catastrófico e mudança semântica?
5. A parada antecipada está correta e suficiente?
6. Quais resultados precisam ser replicados antes de prosseguir?
7. Quais critérios objetivos devem determinar sucesso ou falsificação?

### 7.6 Engenharia de software

1. Há bugs, problemas de reprodutibilidade ou estados compartilhados
   acidentalmente?
2. Os resultados e configurações salvos permitem reprodução completa?
3. Os testes atuais cobrem as propriedades científicas centrais?
4. Quais testes adicionais são obrigatórios?
5. O código relacional está modularizado de forma adequada?
6. O pipeline anterior interfere ou pode confundir a configuração principal?

---

## 8. Testes mentais que você deve aplicar

Avalie explicitamente o comportamento esperado nestes casos:

### Caso A: somente rotação global

```text
E_t1 = E_t0 @ Q
Q é ortogonal
```

Todas as métricas de mudança relacional por sujeito devem ser zero.

### Caso B: nenhuma troca de corpus, mas treinamento continua

```text
D_0 -> D_0 -> D_0
```

Qual mudança é aceitável? Como ela deve ser usada como distribuição nula?

### Caso C: apenas uma palavra muda

Se somente `S1` mudar, o perfil relacional de `S1` muda, mas também podem mudar
os perfis das demais palavras por causa da relação delas com `S1`. Verifique se
o oráculo e a interpretação capturam isso.

### Caso D: todas as palavras se movem juntas preservando relações

Mesmo que suas propriedades absolutas mudem, se todas as relações internas
permanecerem iguais, a mudança relacional deve ser zero.

### Caso E: permutação dos rótulos de período

Se os períodos forem embaralhados, quais métricas deveriam degradar? Esse teste
deve entrar no experimento?

### Caso F: probe diferente, mesma semântica

Se `h_subj`, estado mascarado e logits produzem conclusões divergentes, qual
representação possui justificativa científica mais forte?

---

## 9. Formato obrigatório do parecer

Escreva o parecer somente em:

`./tmp/timeformer_relational_code_validation_review.md`

Use esta estrutura:

1. **Resumo executivo e veredito**
2. **Sua compreensão do objetivo**
3. **O código implementa o objetivo?**
4. **Bugs ou inconsistências concretas**
5. **Riscos à validade científica**
6. **Avaliação do probe preditivo**
7. **Avaliação das métricas e do oráculo**
8. **Avaliação do placebo e da mudança contrafactual**
9. **Auditoria por arquivo**
10. **Testes adicionais obrigatórios**
11. **Mudanças necessárias antes de múltiplas seeds**
12. **Mudanças necessárias antes de corpus real**
13. **Plano recomendado em ordem de prioridade**

No veredito, responda diretamente:

- Estamos medindo a mudança semântica relacional que afirmamos medir?
- O resultado positivo atual pode ser considerado evidência preliminar válida?
- Existe algum bug ou circularidade capaz de explicar os resultados?
- Devemos executar múltiplas seeds agora ou corrigir algo antes?
- Quais são as três mudanças mais importantes antes de prosseguir?

Classifique cada recomendação como:

- **Bloqueadora**
- **Importante**
- **Desejável**

Sempre cite arquivos, funções e, quando possível, linhas concretas. Não altere
nenhum arquivo além do parecer solicitado.
