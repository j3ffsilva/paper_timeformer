# Pré-registração da integração ConSeC-TimeFormer

**Data:** 2026-06-14
**Estado:** anterior ao cálculo das correlações entre os dois instrumentos.

## Pergunta

Palavras cuja distribuição explícita de sentidos muda mais segundo o ConSeC
também apresentam maior separação contextual na régua `layer_1` do
`bert-tiny` treinado cronologicamente?

Esta análise testa **validade convergente entre instrumentos**. Ela não testa
se uma camada causa a mudança da outra e não reabre a hipótese, já descartada,
de redistribuição causal entre camadas.

## Dados congelados

### ConSeC

Serão usados os 25 alvos confirmatórios da Porta 3. Para cada palavra:

```text
consec_jsd = média da JSD bruta nas seeds 20260613, 20260614 e 20260615
consec_excess = média da JSD excedente nas mesmas três seeds
consec_z = média do z do nulo intrapalavra
```

### TimeFormer

A condição principal será o fine-tuning cronológico completo, já selecionado
sem gold:

```text
outputs/bert_tiny_option_d_full_seed1000/selected_apd_eval/
outputs/bert_tiny_option_d_full_seed1001/selected_apd_eval/
```

Para cada palavra:

```text
layer1_apd = média da APD de layer_1 nas seeds 1000 e 1001
layer2_apd = média da APD de layer_2 nas seeds 1000 e 1001
layer_delta = layer2_apd - layer1_apd
```

Os dois avaliadores usam ocorrências próprias. Portanto, a análise compara
scores agregados por palavra, não as mesmas frases.

## Hipóteses principais

Nos 25 alvos confirmatórios:

1. `Spearman(layer1_apd, consec_jsd) > 0`;
2. `Spearman(layer1_apd, consec_excess) > 0`.

Serão calculados p-valores por permutação de palavra com 20.000 permutações.
A análise será considerada convergente se as duas correlações forem positivas
e pelo menos a correlação com `consec_excess` tiver `p < 0,05`.

O excedente recebe o critério inferencial porque reduz o efeito do tamanho do
inventário. A JSD bruta permanece importante por ser o ranking mais estável.

## Controles pré-especificados

1. correlação parcial entre `layer1_apd` e cada score ConSeC controlando:
   - gold graduado;
   - número de sentidos WordNet;
2. estabilidade da correlação ao usar cada seed TimeFormer separadamente;
3. repetição no subconjunto de cobertura `high`;
4. comparação descritiva com:
   - `layer2_apd`;
   - `layer_delta`;
   - condição pseudo-período seed 1000;
   - condição de LR discriminativa seed 1000.

As análises de `layer_2`, diferenças entre camadas e condições alternativas
são exploratórias. Nenhuma será usada para substituir a análise principal se
produzir um número mais favorável.

## Interpretação

Resultados possíveis:

```text
Convergência principal:
  ConSeC e layer_1 ordenam palavras de forma semelhante.

Convergência apenas bruta:
  parte do alinhamento pode vir do inventário ou da amostragem.

Sem convergência:
  APD contextual e mudança explícita de sentidos capturam fenômenos distintos.

Parcial por gold próxima de zero:
  os instrumentos concordam principalmente porque ambos acompanham o benchmark.

Parcial por gold positiva:
  há concordância entre instrumentos além do ranking fornecido pelo gold.
```

## Saídas

```text
outputs/consec_timeformer_integration/per_target.csv
outputs/consec_timeformer_integration/summary.json
```
