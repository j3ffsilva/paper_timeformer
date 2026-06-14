# Integração ConSeC-TimeFormer

**Data:** 2026-06-14
**Decisão pré-registrada:** validade convergente não estabelecida.

## Pergunta

A APD contextual de `layer_1` ordena as palavras da mesma forma que a mudança
explícita de distribuição de sentidos prevista pelo ConSeC?

Foram comparados os 25 alvos confirmatórios da Porta 3. O score ConSeC foi a
média de três amostras; o score TimeFormer foi a média dos dois treinos
cronológicos completos.

## Resultado principal

| Comparação | Spearman | IC bootstrap 95% | p permutação |
|---|---:|---:|---:|
| `layer_1` × JSD bruta | -0,031 | [-0,437; 0,401] | 0,884 |
| `layer_1` × JSD excedente | -0,024 | [-0,407; 0,398] | 0,908 |

Nenhum critério pré-registrado passou:

```text
correlação bruta positiva       = não
correlação excedente positiva   = não
p excedente < 0,05              = não
```

Portanto:

```text
Validade convergente escalar = NO-GO
```

## Controles

O resultado permaneceu próximo de zero ao controlar o gold e o número de
sentidos:

| Score ConSeC | Parcial por gold | Parcial por nº sentidos | Parcial por ambos |
|---|---:|---:|---:|
| JSD bruta | -0,097 | -0,039 | -0,090 |
| JSD excedente | -0,060 | -0,025 | -0,058 |
| z nulo | -0,045 | -0,021 | -0,053 |

Também não dependeu de uma seed TimeFormer:

| Condição `layer_1` | × JSD bruta | × JSD excedente |
|---|---:|---:|
| full seed 1000 | -0,061 | -0,052 |
| full seed 1001 | -0,028 | -0,015 |
| pseudo seed 1000 | 0,008 | 0,032 |
| LR discriminativa seed 1000 | -0,052 | -0,017 |

Nos 21 alvos de alta confiança, as correlações ficaram positivas, mas pequenas:

```text
layer_1 × JSD bruta     = 0,108
layer_1 × JSD excedente = 0,082
```

## Contexto do benchmark

Esta análise usa um subconjunto diferente dos 37 alvos da avaliação original.
Nos 25 confirmatórios:

| Instrumento | Spearman com gold |
|---|---:|
| ConSeC JSD bruta | 0,604 |
| ConSeC JSD excedente | 0,409 |
| TimeFormer `layer_1` | 0,076 |
| TimeFormer `layer_2` | -0,025 |

Nos 21 alvos de alta confiança, `layer_1` sobe para `0,288`, mas sua
correlação com ConSeC continua baixa. Assim, o NO-GO não é explicado apenas
pela qualidade dos inventários.

## Análises exploratórias

`layer_2` apresentou correlações descritivas maiores com os scores corrigidos:

```text
layer_2 × JSD excedente = 0,211
layer_2 × z nulo        = 0,354
```

Esses valores não eram hipóteses principais, não receberam teste confirmatório
pré-registrado e não são estáveis o bastante para reabrir a narrativa de
redistribuição entre camadas.

O contraste cronológico menos pseudo-período foi negativo para JSD bruta
(`-0,339`) e excedente (`-0,175`). Isso também não sustenta uma resposta
adaptativa cronológica simples.

## Interpretação

A APD mede separação média entre duas nuvens contextuais. Ela pode crescer
quando gênero, tópico, sintaxe ou composição documental mudam, mesmo que a
mistura de sentidos permaneça estável.

A JSD do ConSeC mede mudança na massa atribuída a sentidos WordNet. Ela pode
detectar substituição ou diversificação de sentidos sem exigir grande
deslocamento médio na geometria contextual.

Portanto, os instrumentos capturam aspectos diferentes:

```text
APD       = mudança contextual agregada
ConSeC JSD = mudança explícita de mistura de sentidos
```

O resultado impede interpretar APD como substituto direto de mudança de
sentido.

## Próximo passo

A próxima análise deve ocorrer no nível da ocorrência, usando os mesmos
contextos nos dois instrumentos.

Para cada palavra e ocorrência:

```text
ConSeC     -> posterior sobre sentidos
TimeFormer -> vetor contextual de layer_1
```

Dentro de cada palavra, será testado se:

1. pares com posteriores de sentido mais diferentes também têm vetores mais
   distantes;
2. ocorrências atribuídas ao mesmo sentido permanecem geometricamente
   comparáveis entre D0 e D1;
3. a APD total pode ser separada em mudança de mistura de sentidos e deriva
   contextual dentro do sentido.

Esse desenho elimina a diferença de amostragem entre os instrumentos e evita
comparar escalas absolutas entre palavras.

## Artefatos

```text
outputs/consec_timeformer_integration/
scripts/integrate_consec_timeformer.py
docs/29-consec_timeformer_integration_preregistration.md
```
