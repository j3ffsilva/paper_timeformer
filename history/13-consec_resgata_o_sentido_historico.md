# 13. ConSeC resgata o sentido histórico

A adjudicação humana mostrou que a falha do LMMS em `plane` como ferramenta
era real. Restava saber se o problema era geral do WSD WordNet ou específico
do LMMS.

## Segundo modelo congelado

Foi executado o checkpoint oficial ConSeC SemCor+WNGT, sem treinamento,
calibração ou seleção de limiar nos exemplos do projeto.

Como os contextos históricos não possuem sentidos anotados para todas as
palavras vizinhas, usamos a interface de extração para o alvo sem o feedback
loop completo.

## Contraste decisivo

Nos 16 exemplos humanos de ferramenta:

```text
LMMS     2/16   12,5%
ConSeC  14/16   87,5%
```

Nos 18 exemplos com rótulo humano definido:

```text
LMMS     2/18   11,1%
ConSeC  14/18   77,8%
```

Portanto, o inglês histórico e o sentido raro não tornam a tarefa
intrinsecamente impossível. O fracasso anterior está ligado principalmente
ao método LMMS.

## Nova decisão

A regra de parada não foi acionada, porque os dois modelos não falharam.
Entretanto, o resultado ainda é pequeno e não justifica escalar diretamente
para todo o benchmark.

A próxima porta passa a ser:

1. rodar ConSeC em todas as ocorrências do Gate 1 de `plane`;
2. confirmar simultaneamente geometria, ferramenta e aviação;
3. testar depois 3 a 5 palavras com inventários claros;
4. somente então decidir sobre os 37 alvos.

O ConSeC torna-se o candidato principal para WSD externo. O LMMS permanece
como baseline negativo útil.

Relatório completo:
[resultado ConSeC](../docs/18-consec_plane_adjudicated_results.md).
