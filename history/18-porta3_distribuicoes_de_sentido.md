# 18. A mudança aparece nas distribuições de sentido

Depois de validar contextos individuais, a Porta 3 voltou à pergunta central
do projeto:

```text
quanto a distribuição de sentidos de uma palavra muda entre D0 e D1?
```

## O desenho

O ConSeC foi mantido congelado. Para cada palavra, 25 ocorrências de cada
período produziram probabilidades sobre sentidos WordNet. A distância temporal
foi a Jensen-Shannon entre as duas distribuições médias.

Dos 37 alvos:

```text
25 entraram no teste confirmatório
9 ficaram como cobertura parcial
3 foram controles monossêmicos
```

## Resultado

Nos 25 alvos confirmatórios:

```text
Spearman = 0,586
IC 95% = [0,231; 0,818]
p por permutação = 0,0028
```

A análise de alta confiança produziu `0,600`. A Porta 3 passou.

`plane` foi o maior deslocamento. `tree` permaneceu baixo. `chairman` ficou
em zero por ser monossêmico. `graft`, mantido como diagnóstico, mostrou a
ascensão do sentido de corrupção, mas continuou limitado pela ausência do
objeto botânico no inventário.

## Nova cautela

O número de sentidos WordNet correlacionou com a JSD (`rho=0,533`). Parte do
score pode refletir mais graus de liberdade em inventários grandes.

Controlando esse número exploratoriamente, a associação com o gold permaneceu
positiva (`rho parcial=0,454`), mas esse controle precisa de replicação
pré-registrada.

## O que mudou no projeto

O projeto agora possui uma medida de mudança:

- explícita em sentidos;
- externa ao encoder temporal;
- auditada quanto à cobertura;
- estatisticamente associada ao benchmark.

O próximo passo deixa de ser procurar mais uma representação. Passa a ser
testar a estabilidade da medida entre amostras e calibrar o efeito do tamanho
do inventário.

Relatório:
[Resultados da Porta 3](../docs/24-consec_gate3_results.md).
