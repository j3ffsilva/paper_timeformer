# Reframing: TimeFormer como instrumento de consulta temporal

**Data:** 2026-06-06  
**Para:** codex  
**Assunto:** novo enquadramento da contribuição e próximos passos

---

## O que mudou no nosso entendimento

Estávamos tentando construir um *detector* de mudança semântica — um sistema
que produz um score escalar e compete com o Spearman do SemEval. Isso criou uma
barreira artificial e um objetivo errado.

O TimeFormer não é um detector. É um **instrumento de consulta temporal**.

A pergunta que ele responde é:

> Quais eram os vizinhos lexicais de `w` no corpus de D0, e quais são em D1?

Isso não pressupõe nenhuma teoria de sentido. O pesquisador recebe as
vizinhanças e interpreta. Pode ser mudança de sentido lexical, pode ser mudança
de registro, pode ser deriva de domínio, pode ser mudança social — o instrumento
fornece evidência linguística densa e temporalmente localizada sem precisar
categorizar o tipo de mudança.

---

## O que isso significa para os resultados que já temos

### `plane_nn` — resultado correto, mal enquadrado

Não é um falso negativo no SemEval. É uma demonstração bem-sucedida do
instrumento:

```
D0: line, angle, plate, column, canal, coast
D1: boat, ship, rail, route, engine, machine
```

O pesquisador vê a transição geométrico/material → transporte. Não precisamos
dizer que "detectamos mudança semântica". Dizemos que o instrumento produz
vizinhanças temporalmente interpretáveis para essa palavra.

### `chairman_nn` — resultado correto, interpretado errado

Não é um falso positivo. É um resultado informativo:

```
D0: secretary, editor, commander, director, president, committee, jury
D1: secretary, director, commander, president, commissioner, governor, publisher
```

O núcleo institucional permanece estável. O que muda é a realização concreta do
campo — de figuras históricas religiosas e militares para executivos corporativos
e políticos contemporâneos. Isso é mudança da *ecologia lexical* da palavra, não
do seu sentido. O instrumento capturou corretamente. O Spearman do SemEval não
mede isso porque SemEval pergunta sobre mudança de sentido lexical, não sobre
mudança de ecologia.

### O exemplo canônico que articula a contribuição

`negro@1950` vs `negro@2020` em português. Não há mudança de sentido
lexical — a palavra continua referindo-se a pessoas negras. Mas as vizinhanças
lexicais mudam profundamente: contextos de submissão, inferiorização e
discriminação dão lugar a contextos de resistência, afirmação cultural e
identidade. Um instrumento que mostre isso sem qualquer anotação de sentido
é genuinamente útil para sociolinguistas e historiadores da língua.

---

## O argumento central do paper

Um único Transformer treinado continuamente em ordem cronológica produz
representações contextuais in-domain que permitem consultar a **vizinhança
lexical** de qualquer palavra em qualquer checkpoint — sem alinhamento
post-hoc entre espaços, sem modelo externo pré-treinado, sem anotação de
sentido.

A contribuição tem três componentes separáveis:

1. **Arquitetura:** modelo único contínuo vs. dois modelos independentes
   alinhados (Hamilton 2016) ou modelo externo fora do domínio (APD+BERT).

2. **Representações contextuais in-domain:** cada ocorrência é representada
   no seu contexto real, treinada sobre o corpus histórico. Isso é diferente
   de word2vec (estático, frequência global) e de BERT aplicado fora do
   domínio.

3. **Consulta temporal sem alinhamento:** a pergunta `similares(w@t)` é
   respondida diretamente pelos vizinhos no checkpoint θ_t, sem projetar
   dois espaços distintos num referencial comum.

O Spearman do SemEval é validação secundária — mostra que o instrumento não
é aleatório. Não é o objetivo primário.

---

## O que o SemEval ainda serve para

Validação externa quantitativa. Um Spearman positivo e acima de zero com 37
palavras mostra que as vizinhanças que o instrumento produz têm correlação com
julgamentos humanos de "algo linguisticamente relevante mudou". Não precisamos
superar os 0.42 dos melhores sistemas do SemEval para publicar — precisamos
mostrar que o instrumento não é ruído.

O resultado atual com APD de estados ocultos (Spearman=0.21) já é suficiente
para essa função. Melhorar esse número é desejável mas não é bloqueador.

---

## O que precisamos fazer a partir de agora

### 1. Comparação com Hamilton 2016 — prioridade máxima

Treinar word2vec separadamente em D0 e D1, alinhar com Procrustes ortogonal,
e aplicar o mesmo protocolo de vizinhança:

```
top-20 D0, top-20 D1, ganhos e perdas
Spearman graded e AUC binário
```

Isso posiciona a contribuição concretamente: o TimeFormer produz vizinhanças
comparáveis ou melhores com representações contextuais in-domain e sem
alinhamento geométrico? Se produzir, a contribuição é clara. Se word2vec
produzir vizinhanças igualmente coerentes com muito menos custo, precisamos
articular o que o treinamento contínuo adiciona que não pode ser obtido com
dois modelos independentes.

### 2. Relatório qualitativo completo dos 37 alvos

Ampliar o relatório de vizinhança temporal (já feito para plane, chairman,
tree, graft) para todas as 37 palavras do SemEval. Isso é o resultado central
do paper — não tabelas de Spearman, mas vizinhanças interpretáveis.

Formato por palavra:

```
top-10 D0 | top-10 D1 | top-5 ganhos | top-5 perdas | campo estável?
```

Avaliação qualitativa cega: um pesquisador que não conhece o gold do SemEval
consegue distinguir palavras com vizinhança estável de palavras com vizinhança
claramente transformada?

### 3. Field-controlled APD para o ranking quantitativo

O APD bruto confunde deriva de campo com mudança específica da palavra.
Subtrair a mediana do APD de palavras do mesmo campo semântico produz um
estimador mais robusto. Isso já foi testado manualmente para 4 palavras —
aplicar sistematicamente às 37.

Definição de campos sem usar o modelo como classificador: agrupamento por
co-ocorrência de corpus (ou simplesmente categorias manuais baseadas nos
alvos disponíveis).

### 4. Narrativa do paper: instrumento, não detector

Reescrever a introdução e seção de contribuição para posicionar o TimeFormer
como instrumento de consulta temporal, não como sistema de detecção de mudança
semântica. A mudança é de framing, não de conteúdo técnico.

A reivindicação central passa a ser:

> TimeFormer permite que pesquisadores consultem vizinhanças lexicais de
> qualquer palavra em qualquer ponto do tempo, com representações contextuais
> treinadas in-domain e sem dependência de alinhamento geométrico post-hoc.

Exemplos como `negro@1950`/`negro@2020` e `plane@1810`/`plane@1960` são a
demonstração primária. O SemEval é validação de que o instrumento não é ruído.

---

## O que não vamos fazer

- Maximizar Spearman do SemEval como objetivo principal
- Tentar "resolver" o problema de chairman como falso positivo — é um resultado
  correto que requer interpretação diferente, não correção
- Usar clustering como estimador de sentido (muro de identificabilidade formal)
- Introduzir componente externo de WSD (BEM, ConSeC) — transfere a contribuição

---

## Critérios para saber que estamos no caminho certo

**Continuar se:**
- Comparação com Hamilton 2016 mostrar que word2vec produz vizinhanças menos
  interpretáveis ou menos coerentes para múltiplos alvos
- Relatório qualitativo dos 37 alvos for avaliado positivamente em teste cego
- Field-controlled APD separar claramente plane (alta mudança específica) de
  chairman (baixa mudança específica) e graft de tree

**Reformular se:**
- Hamilton 2016 produzir vizinhanças igualmente coerentes com mínimo esforço
- As vizinhanças dos 37 alvos forem incoerentes para a maioria das palavras
- O relatório qualitativo não passar em avaliação cega

---

## Estado do repositório

Código da arquitetura TimeFormer: estável, testado.  
Dataset SemEval processado: `data/processed/semeval2020_task1/eng_lemma/`  
Checkpoints: `outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/continual_real/`  
Vizinhanças qualitativas: `outputs/.../temporal_relational_neighborhoods/`  
Próximo experimento: Hamilton 2016 baseline.
