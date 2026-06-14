# 23. Composição de sentidos e componente não atribuída

O projeto já sabia duas coisas:

1. APD e JSD não produzem o mesmo ranking de palavras;
2. dentro de cada palavra, usos semanticamente diferentes ficam mais
   distantes na geometria.

Faltava conectar essas observações ao deslocamento temporal do centróide.

## A decomposição

Para cada período e sentido, os vetores foram ponderados pelas probabilidades
do ConSeC. O deslocamento total foi separado exatamente em:

```text
composição = efeito atribuído à mudança das proporções dos sentidos
restante   = componente algébrica não atribuída à composição
```

A fórmula simétrica garante:

```text
deslocamento total = composição + deriva
```

## Por que um nulo era necessário

Uma identidade algébrica sempre produz duas parcelas. Isso, sozinho, não prova
que a parcela chamada "composição" tem conteúdo semântico.

Os posteriores foram então embaralhados entre vetores dentro de cada período.
O procedimento preservou:

- a mistura de sentidos de D0 e D1;
- os vetores de D0 e D1;
- o deslocamento global.

Ele destruiu apenas a correspondência entre sentido e geometria.

## O resultado

Na `layer_2`, a composição respondeu por uma projeção média de `0,048` acima
do nulo. O intervalo foi `[0,024; 0,081]`, 23 das 25 palavras ficaram
positivas e `p=0,00005`.

A componente complementar respondeu pela parcela direcional restante,
aproximadamente `0,952`.

O resultado não diz que apenas 5% da semântica mudou. Ele diz:

> Cerca de 5% da direção do deslocamento médio pode ser ligada
> especificamente à troca da mistura de sentidos acima de uma associação
> aleatória. A natureza do restante não é identificada por esta análise.

Esse restante pode ser investigado por especialistas usando as vizinhanças e
os contextos recuperados pelo TimeFormer. A decomposição, sozinha, não permite
classificá-lo como mudança contextual, semântica, de gênero, domínio ou
qualquer combinação dessas possibilidades.

## O caso `plane`

`plane_nn` apresentou:

```text
JSD média = 0,428
excesso de composição = 0,341
```

A substituição de usos geométricos por aviação aparece simultaneamente:

- na distribuição explícita de sentidos;
- na organização local dos vetores;
- na direção do deslocamento temporal.

É o exemplo concreto mais completo da cadeia metodológica.

## A camada superior, reinterpretada

`layer_2` havia falhado como ranking temporal por APD. No entanto:

- separa diferenças locais de sentido melhor que `layer_1`;
- atribui uma parcela de composição maior;
- sua parcela de composição acompanha a JSD (`rho=0,615`).

Logo, a camada não era "semanticamente vazia". O resumo APD é que descartava
informação demais.

## Próxima etapa

A decomposição precisa agora de intervalos por palavra obtidos por bootstrap
estratificado de ocorrências. Isso mostrará quais casos individuais são
estáveis e quais dependem da amostra.

Detalhes: `docs/34-soft_sense_vector_decomposition_results.md`.
