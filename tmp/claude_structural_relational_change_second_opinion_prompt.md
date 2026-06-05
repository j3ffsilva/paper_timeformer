# Pedido de segunda opinião: deslocamento relacional e mudança semântica estrutural

Você deve realizar uma revisão técnica, conceitual e científica independente do
projeto:

`/Users/jeff/Documents/trabalhos/papers/paper-timeformers`

Não altere código, documentação ou resultados existentes. Sua tarefa é ler o
material relevante, compreender o ponto atual da pesquisa, avaliar criticamente
a nova interpretação proposta e escrever um parecer autocontido em:

`./tmp/timeformer_structural_relational_change_review.md`

Este pedido não busca apenas confirmação. Procure ativamente falhas conceituais,
riscos à validade, mudança oportunista de objetivo, alternativas melhores e
experimentos capazes de falsificar nossa interpretação.

---

## 1. Ponto atual da pesquisa

O objetivo atual é estudar mudança semântica ao longo do tempo sem inserir
informação temporal no Transformer e sem depender da comparação de coordenadas
absolutas entre checkpoints.

Treinamos um Transformer padrão cronologicamente:

```text
theta_0 = treino(D_0)
theta_1 = continua_treino(theta_0, D_1)
theta_2 = continua_treino(theta_1, D_2)
...
theta_t = continua_treino(theta_(t-1), D_t)
```

Nenhum identificador de período é fornecido ao modelo. Após cada período,
salvamos um checkpoint.

Não interpretamos diretamente:

```text
h_t1(w) - h_t0(w)
```

porque as coordenadas internas podem rotacionar, refletir, redimensionar ou
sofrer deriva de otimização sem que as relações semânticas tenham mudado.

Em vez disso, consultamos cada checkpoint com um probe preditivo
pós-Transformer:

```text
[CLS] palavra [MASK] [MASK] [SEP]
```

Extraímos a distribuição prevista de contextos válidos:

```text
q_t(w) = P_t(contexto | palavra)
```

Definimos a relação entre duas palavras no mesmo checkpoint por similaridade de
Jensen-Shannon:

```text
r_t(w)[v] = 1 - JS(q_t(w), q_t(v)) / log(2)
```

O perfil relacional de uma palavra é:

```text
r_t(w) = [
    relacao_t(w, palavra_1),
    relacao_t(w, palavra_2),
    ...,
    relacao_t(w, palavra_n)
]
```

E seu vetor de deslocamento relacional é:

```text
delta_rel(w, a, b) = r_b(w) - r_a(w)
```

Cada dimensão desse vetor informa de qual palavra ou conceito `w` se aproximou
ou se afastou. O vetor preserva direção e interpretabilidade, mas não depende
das coordenadas ocultas do Transformer.

---

## 2. O que já conseguimos demonstrar

O benchmark sintético planta trajetórias semânticas conhecidas. Ele permite
comparar o vetor relacional aprendido com um vetor-oráculo.

Os principais controles são:

- `continual_real`: treinamento cronológico em `D_0, D_1, ..., D_t`;
- `continual_placebo`: continua treinando repetidamente em `D_0`, diagnosticando
  deriva de otimização;
- `resampled_null`: recebe novos textos a cada período, mas sem mudança na
  distribuição semântica plantada;
- `frozen`: aplica o checkpoint inicial sem atualizar seus pesos.

Em três seeds e com orçamento fixo de `8250` atualizações, a mudança acumulada
de `t0` para `t9` apresentou:

```text
direção relacional observada versus oráculo: +0.913
direção placebo versus oráculo:             +0.603
vantagem direcional observada:              +0.310
```

A vantagem foi positiva em todas as seeds e classes sintéticas. Assim, o
sistema registra uma direção relacional associada à mudança real, embora a
deriva de otimização também produza sinal e precise permanecer controlada.

Escalamos artificialmente a intensidade das trajetórias:

```text
p_t(alpha) = p_0 + alpha * (p_t - p_0)
```

Usando o percentil 95 do nulo ressampleado e exigindo direção positiva com o
oráculo, obtivemos:

| Intensidade `alpha` | Magnitude média | Direção média | Detectadas |
|---:|---:|---:|---:|
| 0.25 | 0.0060 | +0.210 | 5.0% |
| 0.50 | 0.0112 | +0.577 | 22.5% |
| 0.75 | 0.0232 | +0.784 | 81.7% |
| 1.00 | 0.0416 | +0.893 | 100.0% |

Portanto, o método recupera com confiança mudanças relacionais suficientemente
fortes, mas não pequenas perturbações.

Também testamos mais exemplos mantendo o mesmo orçamento de atualizações:

| Exemplos | Atualizações | Escala | Direção | Detectadas |
|---:|---:|---:|---:|---:|
| 100 | 8250 | 0.50 | +0.577 | 22.5% |
| 300 | 8250 | 0.50 | +0.612 | 11.7% |
| 100 | 8250 | 0.75 | +0.784 | 81.7% |
| 300 | 8250 | 0.75 | +0.837 | 78.3% |

Mais exemplos melhoraram a direção média, mas não a taxa de detecção quando o
orçamento de atualizações permaneceu fixo.

O nulo também é heterogêneo: algumas palavras apresentam mais variação
relacional natural que outras. Isso mostrou que igualdade exata entre perfis
relacionais não é uma definição realista de estabilidade.

Consulte principalmente:

- `docs/relational_change_current_plan.md`
- `src/timeformers/relational.py`
- `src/timeformers/relational_change.py`
- `src/timeformers/corpus.py`
- `scripts/run_relational_continual_sanity.py`
- `scripts/summarize_relational_sensitivity.py`
- `tests/test_relational.py`
- resultados sob `outputs/relational_sensitivity*`

---

## 3. A reconsideração conceitual

Até agora, parte da análise tratava mudança semântica como qualquer diferença
relacional detectável acima do ruído.

Estamos reconsiderando esse objetivo. Talvez procurar detectar diferenças cada
vez menores seja observar o fenômeno na escala errada.

Nossa nova hipótese conceitual é:

> estabilidade semântica não é ausência de movimento relacional; é ausência de
> reorganização estrutural relevante das relações.

Queremos manter o vetor de deslocamento relacional:

```text
delta_rel(w, a, b) = r_b(w) - r_a(w)
```

pois ele explica direção, aproximações e afastamentos. Porém, não queremos
interpretar automaticamente todo `delta_rel != 0` como mudança semântica
significativa.

Precisamos distinguir:

```text
microvariação relacional normal
```

de:

```text
reorganização semântica estrutural
```

---

## 4. Exemplo concreto: círculo de amizades

Imagine que cada palavra é uma pessoa e suas relações semânticas são seu círculo
de amizades.

Em um ano normal:

- um colega muda de empresa;
- duas pessoas brigam e ficam um pouco mais distantes;
- uma amiga muda de cidade;
- surgem pequenas aproximações e afastamentos.

O círculo não permaneceu numericamente idêntico. Há um vetor de deslocamento
relacional diferente de zero. Ainda assim, podemos considerar o círculo
estruturalmente estável: os principais grupos, papéis e relações gerais
permanecem reconhecíveis.

Em outro caso, a pessoa abandona sua profissão, muda de país e passa a integrar
uma comunidade completamente diferente. Várias relações mudam de forma
coordenada, persistente e em escala maior. Aqui ocorreu uma reorganização
estrutural.

Em termos relacionais:

```text
microvariação:
  poucas dimensões de delta_rel mudam modestamente e sem persistência

reorganização estrutural:
  conjuntos coerentes de dimensões mudam, vizinhanças são substituídas,
  papéis relacionais se alteram e o novo estado persiste
```

Avalie se essa analogia corresponde a uma distinção formal defensável ou se ela
esconde ambiguidades importantes.

---

## 5. Exemplo trivial com vetores relacionais

Considere o perfil relacional simplificado de uma pessoa/palavra com quatro
grupos:

```text
r_t0(w) = [trabalho=0.9, família=0.8, música=0.2, esporte=0.1]
```

### Caso A: pequenas perturbações

```text
r_t1(w) = [0.85, 0.82, 0.25, 0.08]
delta_A  = [-0.05, +0.02, +0.05, -0.02]
```

Todas as relações variaram, mas a organização geral permaneceu igual.

### Caso B: reorganização estrutural

```text
r_t1(w) = [0.2, 0.3, 0.9, 0.8]
delta_B  = [-0.7, -0.5, +0.7, +0.7]
```

A palavra/pessoa mudou de região relacional: os grupos dominantes foram
substituídos de maneira coordenada.

O vetor `delta_rel` registra corretamente os dois casos. A questão é definir uma
função posterior que considere o Caso A estável e o Caso B estruturalmente
alterado, sem apagar a direção explicativa do vetor.

---

## 6. Analogia da ordem de grandeza

Dois furacões da mesma classe não são idênticos. Suas velocidades e trajetórias
variam, mas pertencem à mesma ordem de fenômeno. Uma mudança de classe representa
uma diferença estruturalmente mais relevante que uma pequena diferença
numérica.

Isso sugere que a interpretação da magnitude relacional talvez deva ser
multiescala ou relativa à variação normal, em vez de depender somente de uma
diferença absoluta.

Uma possibilidade ilustrativa seria:

```text
intensidade_estrutural(w) =
    magnitude_observada(w) / escala_de_variacao_normal(w)
```

e, opcionalmente:

```text
ordem_de_mudanca(w) = log(1 + intensidade_estrutural(w))
```

Não estamos comprometidos com escala logarítmica. Ela é apenas uma analogia
para expressar que diferenças da mesma ordem podem ser tratadas como
equivalentes, enquanto saltos de escala podem ser semanticamente importantes.

---

## 7. Possível formulação mais abstrata

Queremos avaliar uma interpretação em duas camadas:

### Camada descritiva

Mantém o vetor completo:

```text
delta_rel(w, a, b)
```

Ele responde:

- de quais conceitos a palavra se aproximou;
- de quais conceitos se afastou;
- em que direção relacional ocorreu a mudança;
- como a trajetória se desenvolveu ao longo dos períodos.

### Camada de significância estrutural

Uma função posterior avalia se o deslocamento representa apenas flutuação ou
uma reorganização semanticamente relevante:

```text
S(w, a, b) = F(
    magnitude_normalizada,
    substituicao_de_vizinhanca,
    coerencia_das_relacoes_alteradas,
    persistencia_temporal,
    escala_e_incerteza
)
```

Possíveis componentes:

- magnitude relativa à variação nula esperada;
- substituição dos principais vizinhos ou grupos relacionais;
- mudança de ranking;
- deslocamento coerente de conjuntos de relações;
- persistência em períodos posteriores;
- detecção de ponto de mudança;
- comparação multiescala;
- robustez a ressampleamento dos textos e probes.

Não sabemos ainda se esses componentes devem formar:

- uma métrica contínua;
- uma classificação ordinal, como estável/transição/reorganização;
- um teste estatístico;
- uma detecção de ponto de mudança;
- uma representação hierárquica ou topológica;
- alguma alternativa melhor.

---

## 8. A objeção que deve ser enfrentada diretamente

Existe um risco sério de estarmos mudando o objetivo depois de observar que o
método não detecta bem mudanças pequenas.

Talvez estejamos dizendo:

```text
"mudanças pequenas não importam"
```

apenas porque o método atual não consegue medi-las.

Avalie rigorosamente:

1. A distinção entre microvariação e reorganização estrutural é teoricamente
   defensável independentemente dos resultados atuais?
2. Estamos descobrindo a escala apropriada do fenômeno ou racionalizando uma
   limitação experimental?
3. Há mudanças semânticas pequenas, graduais e historicamente importantes que
   essa formulação apagaria?
4. Uma mudança estrutural pode ocorrer por acúmulo persistente de deslocamentos
   pequenos?
5. Como evitar que a exigência de grande magnitude favoreça apenas mudanças
   abruptas?
6. Precisamos preservar simultaneamente uma medida fina e uma interpretação
   estrutural em escala superior?
7. Quais hipóteses e critérios precisam ser definidos antes de novos resultados
   para evitar movimentação oportunista dos critérios?

---

## 9. Questões científicas para o parecer

### 9.1 Validade da formulação

1. `delta_rel` é uma representação adequada para direção de mudança semântica?
2. Faz sentido separar o vetor descritivo de uma medida de significância
   estrutural?
3. O que deveria definir formalmente uma reorganização estrutural?
4. Quais invariâncias uma boa medida deve possuir?
5. Como tratar mudanças causadas pelo movimento das demais palavras, mesmo
   quando a distribuição própria de `w` permanece estável?
6. Como distinguir mudança semântica, mudança de domínio, frequência, ruído de
   amostragem e deriva de treinamento?

### 9.2 Alternativas matemáticas

Avalie criticamente, sem assumir que sejam todas apropriadas:

- magnitude normalizada por uma distribuição nula individual ou local;
- escala logarítmica ou classificação por ordem de grandeza;
- Jaccard ou rank-biased overlap de vizinhos;
- divergência entre distribuições relacionais;
- coerência direcional entre grupos de dimensões;
- persistência e integração temporal do deslocamento;
- change-point detection;
- medidas multiescala em grafos semânticos;
- comparação de comunidades ou papéis em redes;
- optimal transport;
- métodos topológicos;
- modelos de estado ou processos estocásticos;
- qualquer alternativa mais simples e defensável.

Indique quais medidas preservam a direção interpretável do vetor relacional e
quais serviriam apenas como resumo escalar.

### 9.3 Polissemia

Avalie se uma representação relacional agregada consegue distinguir:

- substituição de um sentido por outro;
- surgimento de um novo sentido;
- coexistência de sentidos;
- alteração apenas na frequência relativa dos sentidos;
- reorganização estrutural dentro de apenas um sentido.

Explique se precisamos de perfis relacionais por sentido ou distribuições de
perfis, em vez de um único perfil por palavra-período.

### 9.4 Avaliação experimental

Proponha experimentos capazes de distinguir as seguintes hipóteses:

```text
H1: o método apenas não detecta mudanças pequenas.
H2: mudanças pequenas são detectáveis, mas pertencem a uma faixa de
    microvariação estruturalmente estável.
H3: mudanças estruturais são caracterizadas por coerência e persistência, não
    apenas por magnitude.
H4: a mudança estrutural pode emergir do acúmulo de deslocamentos pequenos.
```

O benchmark deve incluir, no mínimo:

- perturbações pequenas independentes e transitórias;
- perturbações pequenas coordenadas;
- perturbações pequenas persistentes e acumulativas;
- grandes mudanças incoerentes;
- mudanças abruptas estruturais;
- substituição de vizinhança mantendo densidade semelhante;
- surgimento e coexistência de sentidos;
- nulos com diferentes frequências e incertezas.

Defina critérios de sucesso e falsificação antes da execução.

### 9.5 Relação com literatura e contribuição

Avalie se a proposta pode ser posicionada legitimamente como:

> deslocamento semântico relacional vetorial com interpretação estrutural
> multiescala.

Indique famílias de literatura que precisam ser confrontadas, especialmente:

- semantic change detection;
- semantic shift versus semantic drift;
- dynamic embeddings;
- mudança em grafos e comunidades;
- concept drift;
- change-point detection;
- estabilidade estrutural e análise multiescala.

Não invente referências. Quando não tiver certeza de uma citação específica,
descreva apenas a família de trabalhos.

---

## 10. Auditoria do código e próximos passos

Inspecione se o código e os resultados atuais realmente sustentam as afirmações
descritas. Em particular:

- confirme se `delta_rel` e as métricas atuais preservam direção relacional;
- verifique se a métrica de magnitude atual mede apenas diferenças locais ou
  algo estrutural;
- identifique o que já pode ser reaproveitado;
- indique componentes e métricas que precisam ser criados;
- proponha o menor experimento capaz de decidir se devemos adotar essa
  reorientação;
- diga se devemos alterar agora `docs/relational_change_current_plan.md` ou
  esperar evidência adicional.

Classifique recomendações como:

1. indispensável antes de prosseguir;
2. experimento principal;
3. ablação;
4. análise posterior;
5. ideia desnecessária ou perigosa.

---

## 11. Formato obrigatório do parecer

Escreva o parecer em:

`./tmp/timeformer_structural_relational_change_review.md`

Use esta estrutura:

1. **Resumo executivo**
2. **Sua compreensão do que estamos propondo**
3. **Estamos vendo o fenômeno na escala correta?**
4. **A distinção entre microvariação e reorganização estrutural é válida?**
5. **Risco de racionalização ou mudança oportunista do objetivo**
6. **Formulação matemática recomendada**
7. **Como preservar o vetor relacional e medir significância estrutural**
8. **Polissemia e mudanças graduais**
9. **Experimentos de falsificação recomendados**
10. **Auditoria do código e dos resultados atuais**
11. **Mudanças necessárias antes de prosseguir**
12. **Veredito**

No veredito, responda diretamente:

- Estamos indo na direção científica correta?
- O resultado atual demonstra deslocamento semântico relacional ou apenas
  sensibilidade a grandes mudanças sintéticas?
- Devemos tratar a baixa detecção de perturbações pequenas como propriedade
  desejável, limitação ou questão ainda aberta?
- Qual é o menor experimento decisivo?
- O que provavelmente ainda não estamos enxergando?

Novamente: não altere nenhum outro arquivo. Escreva somente o parecer solicitado
em `./tmp/timeformer_structural_relational_change_review.md`.
