# O projeto em linguagem simples

---

## O problema que queremos resolver

Imagine que você tem um dicionário que foi atualizado ao longo de 100 anos. A palavra *gay* em 1920 significava alegre. Em 1980, a palavra tinha adquirido um segundo significado, relacionado à identidade sexual, e esse passou a ser o sentido dominante. Se você perguntar a um modelo de linguagem moderno "o que *gay* significa em 1920?", ele vai te dar uma resposta influenciada por tudo que o modelo aprendeu — incluindo os décadas de uso posterior que o modelo viu no treinamento.

O que queremos estudar é justamente isso: **como as palavras mudam de comportamento semântico ao longo do tempo**, e se um modelo de linguagem treinado cronologicamente capta essas mudanças de forma mensurável e interpretável.

Não estamos tentando fazer o modelo "saber" sobre 1920 e 1980 ao mesmo tempo. Estamos tentando treinar o modelo da mesma forma que um ser humano aprende: exposto primeiro ao vocabulário de uma época, depois ao da próxima, e depois ao da seguinte — em ordem cronológica — e perguntar: as representações internas do modelo mudaram de formas que refletem as mudanças semânticas reais que sabemos que ocorreram?

---

## A jornada das ideias: de onde viemos e por que mudamos

### Primeira tentativa: condicionar o Transformer por tempo

O Paper 1 comparou diferentes formas de "marcar" cada sentença com o período a que ela pertence antes de ela entrar no Transformer. A ideia era: se o modelo sabe que está lendo uma sentença de 1920, vai produzir representações diferentes de quando lê uma sentença de 1980.

Fizemos isso de três formas diferentes — adicionando o período ao embedding dos tokens, multiplicando os embeddings por pesos dependentes do período, e concatenando o período ao embedding antes de uma projeção. O Paper 1 mostrou que as três formas produziram resultados essencialmente equivalentes: o modelo conseguia usar o período como informação, mas os três mecanismos não se diferenciaram de forma relevante.

Isso foi um achado importante: **não é o mecanismo de condicionamento que importa, mas se faz sentido condicionar o Transformer por período desta forma**.

### Segundo planejamento: separar posição semântica de trajetória

No segundo planejamento, propusemos que a representação de uma palavra no tempo fosse um par: a posição semântica atual (o que o modelo de linguagem normal produziria) mais um vetor de trajetória separado que capturia "como a palavra chegou até aqui". Propusemos aprender esse vetor de trajetória através de um teacher e um student — com masked trajectory distillation.

O problema que identificamos: ao condicionar o Transformer por tempo (como no Paper 1), o próprio espaço semântico se deforma a cada período. Não existe um sistema de coordenadas fixo. *gay@1920* e *gay@1980* existem em espaços que foram co-treinados com a informação de período — você não pode subtrair um do outro e chamar o resultado de "deslocamento semântico" de forma confiável, porque o espaço em si também se moveu.

Analogy concreta: imagine que você quer medir quanto uma pessoa se deslocou entre dois pontos numa cidade. Se a cidade inteira se reorganizou entre as suas visitas (ruas mudaram de lugar, o centro se moveu), a diferença de coordenadas no mapa não reflete o deslocamento real da pessoa. O que queremos é uma medida que não dependa de o mapa permanecer fixo.

### Terceiro planejamento: espaço congelado em t0 com deslocamentos externos

Então pensamos: e se treinarmos o Transformer somente nos textos do período inicial, congelarmos os pesos, e aprendermos um módulo separado que "empurra" cada palavra para onde ela ficaria em períodos posteriores?

A ideia é limpa: espaço semântico de referência fixo em t0, deslocamento aprendido para cada período posterior. Mas identificamos um problema prático: se o Transformer foi treinado somente em textos de 1920 e você o aplica em textos de 1990, ele estará operando fora de distribuição — processando vocabulário, construções sintáticas e contextos que nunca viu. As representações que produz para textos de 1990 não têm a mesma qualidade de interpretação que as de textos de 1920. O deslocamento que você calcularia misturaria mudança semântica real com degradação da qualidade do modelo fora de distribuição.

### A proposta atual: treinamento contínuo com análise relacional

A ideia que estamos investigando agora é diferente de todas as anteriores em um aspecto fundamental: **o Transformer não recebe nenhuma informação temporal**. Ele é um modelo padrão, treinado da forma mais simples possível.

O que fazemos é treinar o mesmo modelo continuamente, em ordem cronológica — primeiro com os textos de D_0, depois continuando com os textos de D_1, depois D_2, e assim por diante. Depois de cada período, salvamos um snapshot do modelo (um *checkpoint*). No final, temos 10 checkpoints: theta_0, theta_1, ..., theta_9.

```
theta_0 = treinamento em D_0
theta_1 = continuação de theta_0 em D_1
theta_2 = continuação de theta_1 em D_2
...
```

A pergunta é: os checkpoints theta_0 e theta_9 representam as palavras de forma diferente, de maneiras que refletem as mudanças semânticas reais que aconteceram no corpus?

---

## A ideia central: mudança relacional, não mudança de coordenadas

Aqui está o ponto mais importante de toda a proposta, e vale gastar tempo para entendê-lo bem.

Quando um Transformer é treinado, ele organiza as palavras num espaço de alta dimensão. Palavras semanticamente relacionadas ficam próximas; palavras não relacionadas ficam distantes. Mas esse espaço não é fixo de um checkpoint para o outro — ele pode rodar, refletir, expandir ou reorganizar durante o treinamento.

Exemplo trivial: imagine um espaço bidimensional onde no checkpoint t0 temos:

```
gato:     [1, 0]
cachorro: [0, 1]
carro:    [-1, 0]
```

Depois de treinar mais, o espaço inteiro rotaciona 90 graus, e no checkpoint t1 temos:

```
gato:     [0, 1]
cachorro: [-1, 0]
carro:    [0, -1]
```

Se olharmos para os vetores absolutos, *gato* "se moveu" de `[1, 0]` para `[0, 1]` — uma mudança aparente. Mas a distância entre gato e cachorro é a mesma nos dois checkpoints. A distância entre gato e carro é a mesma. **Nenhuma relação semântica mudou**. A rotação é um artefato do treinamento, não uma mudança de significado.

Portanto: **não queremos medir mudança de coordenadas absolutas**. Queremos medir mudança de relações.

O perfil relacional de uma palavra num checkpoint é simplesmente o conjunto de similaridades dela com todas as outras palavras naquele checkpoint:

```
r_t(banco) = {
    dinheiro: 0.90,
    rio: 0.20,
    cadeira: 0.10,
    conta: 0.85,
    ...
}
```

Se no próximo checkpoint esse perfil for:

```
r_{t+1}(banco) = {
    dinheiro: 0.40,
    rio: 0.85,
    cadeira: 0.10,
    conta: 0.40,
    ...
}
```

Então houve mudança semântica: *banco* se afastou de *dinheiro* e *conta*, e se aproximou de *rio*. Não importa se o vetor absoluto de *banco* rotacionou, refletiu ou mudou de escala — o que importa é que suas vizinhanças mudaram.

A medida de mudança é a diferença entre os perfis relacionais:

```
delta_rel(banco, t0, t1) = r_{t1}(banco) - r_{t0}(banco)
= {
    dinheiro: -0.50,   ← afastou
    rio: +0.65,        ← aproximou
    cadeira: 0.00,     ← estável
    conta: -0.45,      ← afastou
}
```

Esse vetor de diferenças é diretamente interpretável: cada dimensão corresponde a uma palavra de referência e diz se a relação com ela aumentou ou diminuiu.

---

## Como extraímos as representações: o probe preditivo

Para construir o perfil relacional de uma palavra num checkpoint, precisamos de uma representação por palavra. O Transformer produz representações contextuais — cada ocorrência de uma palavra produz um vetor diferente dependendo da sentença em que aparece. Precisamos resumir isso num único ponto por palavra por checkpoint.

Experimentamos várias abordagens e encontramos uma que funciona muito melhor que as outras.

A abordagem que funciona: consultamos o checkpoint com uma **sentença-sonda**. Para cada palavra S, criamos a entrada:

```
[CLS] S [MASK] [MASK] [SEP]
```

O verbo e o objeto estão mascarados. O Transformer processa essa entrada e, nas posições mascaradas, produz estados ocultos que representam **o que o modelo prevê como contextos típicos de S**. Tomamos a média desses dois estados ocultos como a representação de S nesse checkpoint.

Por que isso funciona? Porque captura exatamente o que queremos: que contextos o modelo associa a esta palavra, segundo o que aprendeu até agora. Se o modelo aprendeu que *gay* aparece com *cheerful*, *merry*, *laughter*, os estados ocultos nas posições mascaradas vão refletir isso. Depois de treinar nos textos de décadas posteriores, os estados vão refletir os novos contextos.

Comparamos com outras abordagens:

| Representação | O que mede | Resultado |
|---|---|---|
| Estado na posição do sujeito (h_subj) | Como o modelo codifica a palavra internamente | Sinal fraco — não discrimina bem |
| Centroides de ocorrências reais do corpus | Média das representações do sujeito em contextos reais | Sinal muito fraco — praticamente zero |
| Probe preditivo (estados em [MASK][MASK]) | O que o modelo prevê como contextos típicos | Sinal forte — discrimina muito bem |

A diferença é grande: com o probe preditivo, o alinhamento entre a mudança relacional observada e a mudança esperada pelo gerador sintético é de +0.94 a +0.96. Com as outras abordagens, o valor fica próximo de zero. Isso nos diz que o sinal de mudança semântica está codificado principalmente no comportamento preditivo do modelo, não diretamente no vetor do sujeito.

---

## O corpus sintético: por que e como funciona

Para testar se a abordagem funciona, usamos um corpus onde sabemos exatamente quais palavras deveriam mudar e em que direção. Assim, podemos comparar o que o método detecta com o que de fato foi plantado.

O corpus é construído com sentenças muito simples: sujeito + verbo + objeto. Há 40 sujeitos (S1 a S40), 8 verbos (V1–V8) e 8 objetos (O1–O8). Os verbos e objetos se dividem em dois grupos: N1 (V1–V4, O1–O4) e N2 (V5–V8, O5–O8).

A "trajetória semântica" de cada sujeito é controlada pela probabilidade com que aparece em sentenças N1 versus N2 ao longo do tempo. Essa probabilidade é chamada de p_n1.

Os 40 sujeitos se dividem em quatro classes de 10:

**Stable** — p_n1 permanece constante ao longo dos 10 períodos. A palavra não muda de comportamento contextual. Como *cadeira* — sempre significa o mesmo, sempre aparece nos mesmos contextos.

**Drift** — p_n1 começa alto (≈ 0.93) e termina baixo (≈ 0.10). A palavra migra gradualmente de contextos N1 para contextos N2. Como *broadcast* migrando de "semear grãos" para "transmitir rádio" e depois "publicar nas redes".

**Bifurcating** — p_n1 começa alto e vai para um platô intermediário (≈ 0.50). A palavra desenvolve dois sentidos coexistentes. Como *gay* em um período de transição: metade das ocorrências em contextos do sentido antigo, metade em contextos do novo.

**Abrupt** — p_n1 permanece alto até um período t_k, depois cai abruptamente para quase zero. A mudança é súbita, não gradual. Como se uma palavra mudasse de sentido dominante de um período para o próximo sem transição.

Há um ruído intencional de 25%: mesmo que p_n1 seja 0.9, 10% das sentenças por acaso vão usar verbos e objetos de N2. Isso simula o fato de que no mundo real as palavras raramente aparecem exclusivamente em um tipo de contexto.

---

## O experimento: três regimes comparados

Para ter confiança nos resultados, não rodamos apenas o experimento principal. Rodamos três regimes em paralelo e comparamos os resultados.

**Regime 1 — continual_real:** o Transformer é treinado em D_0, depois continua em D_1, D_2, ..., D_9. Cada período tem o corpus com as probabilidades corretas para aquele momento. Este é o experimento que queremos validar.

**Regime 2 — continual_placebo:** o Transformer é treinado em D_0, depois continua treinando em D_0, D_0, D_0, ..., D_0 — o mesmo corpus repetido 10 vezes. O corpus nunca muda. Se o método detectar "mudança semântica" aqui, é sinal de alarme — a "mudança" seria um artefato do treinamento contínuo, não de mudança no corpus.

**Regime 3 — frozen:** usamos o checkpoint theta_0 (o modelo treinado somente em D_0) para extrair as representações de todos os períodos, sem nunca atualizar os pesos. Como esperado, as representações são idênticas em todos os períodos — o que confirma que a extração de probes é determinística e que as mudanças observadas nos outros regimes vêm de fato de mudanças no modelo.

A comparação entre real e placebo nos permite calcular o **sinal excedente** — a mudança que só existe quando o corpus muda, além da deriva que ocorre apenas por continuar otimizando:

```
excedente = mudança(real) - mudança(placebo)
```

Se o excedente for positivo e alinhar com a direção esperada, temos evidência de que o modelo está capturando mudança semântica real, não ruído de treinamento.

---

## O oráculo: como sabemos qual direção é a "correta"

No corpus sintético, sabemos exatamente o que cada sujeito deveria fazer. Se S5 tem p_n1 que vai de 0.92 para 0.08, esperamos que S5 se afaste de outros sujeitos que também têm p_n1 alto e se aproxime de sujeitos que têm p_n1 baixo.

Construímos um perfil relacional *oráculo* para cada período, baseado diretamente nos valores de p_n1 plantados. Dois sujeitos com p_n1 parecido deveriam ser semanticamente próximos; dois sujeitos com p_n1 muito diferente deveriam ser semanticamente distantes.

A métrica principal é o **cosseno direcional**: quão parecida é a *direção* da mudança relacional observada (excedente do real sobre o placebo) com a *direção* que o oráculo prediz?

Exemplo concreto: para o sujeito S5 (Drift, vai de p_n1=0.92 para p_n1=0.08), o oráculo diz que, de t0 para t9:
- S5 deveria se afastar dos sujeitos Stable que têm p_n1 ≈ 0.7 (pois S5 foi para 0.08)
- S5 deveria se afastar dos outros sujeitos Drift que também foram para p_n1 baixo (eles estão no mesmo "canto" que S5, e esse canto específico de p_n1 baixo era antes compartilhado com N1 — agora mudou)
- S5 deveria se aproximar de outros sujeitos que têm p_n1 baixo

Se o probe preditivo do modelo no checkpoint theta_9 mostrar exatamente esse padrão — S5 perdeu similaridade com quem tinha p_n1 alto e ganhou similaridade com quem tem p_n1 baixo — então o cosseno direcional entre o excedente e o oráculo é próximo de 1.

O resultado observado em uma seed foi +0.70 a +0.80 para todas as quatro classes. É promissor, mas ainda é apenas uma seed.

---

## Uma nuance importante: palavras "estáveis" também mudam seu perfil relacional

Há uma sutileza que vale entender: uma palavra classificada como *Stable* no gerador (p_n1 constante) pode ter seu perfil relacional alterado mesmo sem mudar seu próprio comportamento.

Imagine que em t0, tanto S_estavel quanto S_drift estão na "vizinhança N1" do espaço:

```
t0: S_estavel ↔ S_drift: muito similares (ambos têm p_n1 alto ≈ 0.9)
```

Em t9, S_drift foi para p_n1 = 0.1. S_estavel ainda tem p_n1 ≈ 0.8. Agora eles estão em pontas opostas:

```
t9: S_estavel ↔ S_drift: muito diferentes (um tem 0.8, outro tem 0.1)
```

O perfil relacional de S_estavel mudou — mas por causa do movimento de S_drift, não por causa de mudança em S_estavel. Isso é semanticamente correto: se *gay* começa a aparecer em contextos muito diferentes dos que aparecem com palavras como *cheerful*, a distância semântica entre *gay* e *cheerful* aumenta — mesmo que *cheerful* não tenha mudado nada.

O paper precisa ser explícito sobre isso: medir mudança relacional de uma palavra estável não é zero quando as vizinhanças se reorganizam ao redor dela. Isso é uma característica do método, não um defeito. É análogo a dizer "a posição geopolítica do Brasil mudou depois que os vizinhos redefiniram suas fronteiras" — mesmo sem o Brasil fazer nada.

---

## Controles e o que cada um testa

**Por que o placebo tem direção positiva mesmo sem mudar o corpus?**

Observamos que o regime placebo (D_0 repetido) também mostra algum alinhamento com o oráculo — em torno de +0.55 a +0.68. Isso é mais alto do que esperaríamos por acaso e precisa de explicação.

A hipótese mais provável: conforme o modelo continua treinando em D_0, ele melhora progressivamente sua estimativa das probabilidades p_n1 de cada sujeito. À medida que suas estimativas melhoram, a matriz de similaridade entre sujeitos passa a refletir melhor a estrutura real de p_n1. O oráculo de períodos posteriores ainda preserva parte da estrutura de t0 (as palavras Stable não mudaram, as Bifurcating convergiram para 0.5, e parte das relações persistem). Por isso, a melhora do placebo se alinha parcialmente com o oráculo de períodos posteriores.

Isso não invalida o resultado — o *excedente* (real menos placebo) ainda é positivo e substancial. Mas significa que a subtração placebo é uma correção parcial, não perfeita. Para aumentar a confiança, adicionaremos um terceiro controle:

**Permutação de períodos:** rodamos o regime com os períodos em ordem embaralhada — D_5, D_2, D_8, D_3, ..., em vez de D_0, D_1, D_2, D_3, .... Se o sinal do método depende da *ordem cronológica*, ele deve degradar quando a ordem for embaralhada. Se o sinal for robusto mesmo com ordem embaralhada, então não estamos capturando mudança temporal — estamos apenas capturando que o corpus de períodos diferentes é variado. Este controle ainda não foi rodado e é crítico antes de qualquer conclusão forte.

---

## O que os resultados atuais mostram e o que ainda precisamos verificar

**Com uma seed, observamos:**

- Para t0 → t9, o alinhamento direcional do excedente com o oráculo é forte (+0.70–0.79) nos quatro classes.
- O probe preditivo (estados em posições mascaradas) é claramente superior a todas as outras representações.
- O controle frozen confirma que a extração de probes é determinística — zero variação entre períodos com o mesmo modelo.
- Os períodos iniciais (t1, t2) mostram sinal fraco ou negativo nos consecutivos — possivelmente instabilidade na transição inicial.

**O que ainda precisamos verificar:**

1. **Múltiplas seeds** — um resultado com uma seed pode ser coincidência. Precisamos ver se o padrão se replica em pelo menos 5 seeds distintas.
2. **Permutação de períodos** — o controle mais crítico para confirmar que o sinal é cronologicamente específico.
3. **Estabilidade do placebo** — entender melhor por que o placebo alinha tanto com o oráculo e se isso é removido adequadamente pelo excedente.

---

## Como o novo pipeline se compara com a tentativa anterior

A tabela abaixo resume a diferença entre o planejamento anterior e o atual:

| Aspecto | Planejamento anterior | Planejamento atual |
|---|---|---|
| Informação temporal ao modelo | Sim (period embedding) | Não |
| Comparação de representações | Coordenadas absolutas | Perfis relacionais internos |
| Componentes necessários | Transformer + agregador + teacher + student | Transformer padrão + análise posterior |
| Como a trajetória é obtida | Aprendida via masked trajectory distillation | Derivada dos perfis relacionais de cada checkpoint |
| Problema central | Espaço não-comparável entre períodos condicionados | Deriva de otimização que pode parecer mudança semântica |
| Requisito de alinhamento | Implícito (todos treinados no mesmo espaço) | Nenhum (métricas são internas a cada checkpoint) |

A simplificação é substancial. O pipeline atual tem menos componentes, menos hiperparâmetros, e a pergunta científica é mais diretamente testável: "o Transformer captura mudança semântica sem precisar de sinal temporal explícito, desde que seja treinado cronologicamente?"

---

## Como tudo se encaixa numa linha

O Paper 1 mostrou que diferentes mecanismos de condicionamento temporal do Transformer são equivalentes entre si — o mecanismo não é o que importa.

O Paper 2 parte de um ângulo diferente: e se removermos o condicionamento temporal completamente? Treinamos o modelo exatamente como se fosse um Transformer padrão, mas cronologicamente. Medimos mudança semântica não por coordenadas absolutas — que mudam por artefatos de treinamento além da semântica — mas por mudança de vizinhança interna, que é invariante a reorganizações globais do espaço.

O resultado promissor de uma seed sugere que o modelo capta a estrutura semântica correta sem receber informação de período. O próximo passo é replicar em múltiplas seeds e rodar o controle de permutação de períodos. Se o sinal for robusto nesses dois testes, teremos evidência de que treinamento cronológico, por si só, é suficiente para rastrear mudança semântica relacional — sem period embeddings, sem teacher, sem student, sem alinhamento de espaços.
