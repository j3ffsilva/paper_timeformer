# Pré-registração ConSeC-TimeFormer no nível da ocorrência

**Data:** 2026-06-14
**Estado:** anterior à extração dos vetores TimeFormer nas ocorrências ConSeC.

## Motivação

A integração por score agregado de palavra foi NO-GO. APD contextual e JSD de
mistura de sentidos não produziram rankings convergentes.

Agora os dois instrumentos serão aplicados às **mesmas ocorrências**:

```text
ocorrência histórica
  -> posterior ConSeC sobre sentidos
  -> vetor contextual TimeFormer
```

## Dados congelados

Serão usadas as 25 palavras confirmatórias e as 50 ocorrências balanceadas de
cada uma das três amostras ConSeC:

```text
seed 20260613
seed 20260614
seed 20260615
```

Ocorrências repetidas entre seeds serão codificadas uma única vez. Seus
posteriores ConSeC já foram verificados como idênticos.

Os encoders serão os dois checkpoints cronológicos completos previamente
selecionados sem gold:

```text
bert_tiny_option_d_full_seed1000/checkpoints/period1_epoch2
bert_tiny_option_d_full_seed1001/checkpoints/period1_epoch2
```

O gold SemEval não entra nesta análise.

## Codificação

O contexto de raio 20 preservado pelo Gate 3 será tokenizado como palavras. A
janela WordPiece de comprimento 32 será centrada no alvo. Não será permitido
substituir um alvo truncado por outra posição.

Serão extraídos:

```text
layer_1
layer_2, apenas como sensibilidade
```

Quando o alvo produzir vários WordPieces, seu vetor será a média das peças.

## Distâncias por par

Para cada palavra, todos os pares não ordenados entre suas 50 ocorrências
serão formados:

```text
distância semântica = JSD entre posteriores ConSeC
distância geométrica = 1 - cosseno entre vetores TimeFormer
cross_period = 1 se o par liga D0 a D1
```

## Métrica principal

Para cada palavra será calculada a correlação parcial de Spearman:

```text
rho(distância semântica, distância geométrica | cross_period)
```

Controlar `cross_period` impede que uma separação temporal compartilhada pelos
dois instrumentos seja confundida com alinhamento entre sentido e geometria.

O resultado será repetido nas seis combinações:

```text
3 seeds ConSeC × 2 seeds TimeFormer
```

Para cada combinação serão reportados mediana e média dos 25 coeficientes por
palavra. A inferência principal usará a média, por palavra, das seis
combinações.

## Regra de decisão

O alinhamento no nível da ocorrência será considerado estabelecido se:

1. a mediana dos coeficientes for positiva nas seis combinações;
2. a média agregada por palavra for positiva;
3. um teste de inversão aleatória de sinais sobre as 25 palavras, com 20.000
   permutações, produzir `p < 0,05`;
4. pelo menos 15 das 25 palavras tiverem coeficiente agregado positivo.

## Análises secundárias

Sem alterar a decisão principal:

1. correlação usando apenas pares dentro do mesmo período;
2. correlação usando apenas pares entre períodos;
3. repetição em `layer_2`;
4. associação residual entre distância geométrica e `cross_period`,
   controlando a distância semântica;
5. para sentidos previstos presentes nos dois períodos, diferença entre:

```text
distância entre períodos dentro do mesmo sentido
-
distância dentro dos períodos no mesmo sentido
```

O item 5 é diagnóstico, pois rótulos duros descartam incerteza do posterior.

## Interpretação

```text
Alinhamento positivo:
  a geometria contextual contém estrutura compatível com os sentidos ConSeC,
  mesmo que os scores agregados por palavra não coincidam.

Alinhamento nulo:
  o encoder organiza contexto por propriedades diferentes do inventário
  lexical externo.

Período residual positivo após controlar sentido:
  existe deriva contextual dentro ou além da mistura de sentidos.
```

## Saídas

```text
outputs/consec_timeformer_occurrence_alignment/embeddings.npz
outputs/consec_timeformer_occurrence_alignment/per_target.csv
outputs/consec_timeformer_occurrence_alignment/summary.json
```
