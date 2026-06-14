# 22. A geometria contém sentidos localmente

O capítulo anterior mostrou que APD e JSD não ordenam palavras da mesma
forma. Isso poderia significar que os vetores do TimeFormer não contêm
estrutura de sentido. Mas a comparação havia reduzido cada palavra a um único
número.

O projeto então mudou a escala.

## As mesmas ocorrências

Cada frase da Porta 3 foi processada pelos dois instrumentos:

```text
ConSeC     -> distribuição sobre sentidos WordNet
TimeFormer -> vetor contextual
```

Para cada palavra, todos os pares de suas 50 ocorrências foram comparados. Um
par recebia:

- uma distância semântica entre posteriores;
- uma distância cosseno entre vetores;
- uma indicação de atravessar ou não os períodos.

## O achado principal

Na `layer_1`, a correlação parcial média foi `0,062`, com intervalo
`[0,047; 0,078]`. Ela foi positiva em 23 das 25 palavras e se repetiu nas seis
combinações de amostra e seed.

O efeito é modesto, mas responde claramente:

> usos que o ConSeC considera semanticamente mais diferentes tendem a ficar
> um pouco mais distantes no espaço contextual.

O alinhamento permaneceu ao comparar apenas ocorrências do mesmo período.
Portanto, não é apenas um eixo temporal compartilhado.

## A surpresa da camada superior

Na `layer_2`, análise secundária, o alinhamento subiu para `0,187` e foi
positivo nas 25 palavras.

Antes, essa camada parecia ruim porque sua APD entre períodos quase não
acompanhava o gold. Agora fica claro que duas propriedades são diferentes:

```text
separar sentidos localmente
resumir mudança temporal por uma média de distâncias
```

`layer_2` faz a primeira melhor que `layer_1`, mas a segunda continua fraca.

## O que permanece sem atribuição

Mesmo comparando usos atribuídos ao mesmo sentido, ocorrências de períodos
diferentes ficaram ligeiramente mais distantes. Isso estabelece uma componente
adicional, mas não identifica sua natureza:

```text
mudança da mistura de sentidos
+
componente não atribuída à mistura de sentidos
```

Tópico, sintaxe, gênero, domínio e mudança não coberta pelo inventário são
explicações possíveis. Nenhuma delas é determinada por este resultado.

## Próxima pergunta

O passo seguinte é transformar essa distinção numa decomposição vetorial:

```text
deslocamento total
= componente de composição dos sentidos
+ componente complementar não atribuída
```

Detalhes: `docs/32-occurrence_level_consec_timeformer_results.md`.
