# ConSeC no subconjunto adjudicado de `plane`

## Pergunta

O LMMS errou o sentido histórico de ferramenta por uma limitação específica
de seus vetores de sentido ou porque WSD WordNet geral não funciona nesses
contextos?

Para responder, aplicamos um segundo modelo externo congelado às mesmas 19
ocorrências anotadas cegamente.

## Configuração

Usamos:

```text
modelo: ConSeC SemCor+WNGT
checkpoint oficial: 83,2 no benchmark WSD ALL
commit oficial: 9602b5fd69f57be08a186988d1df34fe4152b63f
checkpoint SHA-256:
92421ed245723964db09ce396f19a0d1e55fe4d6e10d5ecb83278d9bc70ce8af
```

O modelo, as definições candidatas e a ordem experimental permaneceram
congelados. Nenhum parâmetro ou limiar foi escolhido usando as 19
ocorrências.

O teste usa a interface oficial de extração para o alvo, sem o feedback loop
de sentidos vizinhos. Esse detalhe é importante: os contextos históricos não
possuem anotação lexical completa necessária para construir o loop de maneira
equivalente ao protocolo Raganato.

O inventário é o mesmo closed set WordNet usado no Gate 1:

```text
ferramenta manual
avião
ferramenta elétrica
plano geométrico
nível de existência
```

O sentido botânico não é candidato para o lema simples `plane` no WordNet;
ele aparece lexicalizado como `plane_tree`.

## Resultado

| Métrica | LMMS | ConSeC |
|---|---:|---:|
| Acurácia nos 16 exemplos humanos de ferramenta | 12,5% | **87,5%** |
| Acertos em ferramenta | 2/16 | **14/16** |
| Acurácia nos 18 itens não `unclear` | 11,1% | **77,8%** |
| Acertos nos itens não `unclear` | 2/18 | **14/18** |

Matriz resumida do ConSeC:

| Rótulo humano | Predição ConSeC | N |
|---|---|---:|
| `tool` | `tool` | 14 |
| `tool` | `geometry` | 2 |
| `geometry` | `aircraft` | 1 |
| `botanical` | `geometry` | 1 |
| `unclear` | `tool` | 1 |

Os dois erros de ferramenta ocorrem na enumeração que inclui `plough`,
`bead plane`, `snipe bill`, `compass plane` e `forkstaff plane`. O modelo
acertou outras posições da mesma enumeração, indicando sensibilidade à
posição e ao contexto local, não desconhecimento completo do domínio.

## Interpretação

O resultado rejeita a explicação mais forte:

```text
WSD WordNet geral é incapaz de reconhecer o sentido histórico de ferramenta
```

Uma leitura mais sustentada é:

1. a falha do LMMS é principalmente específica de sua representação por
   vetores de sentido e matching contextual;
2. um modelo supervisionado de extração baseado em glossas consegue ler a
   maioria dos contextos históricos;
3. o sentido raro não está ausente de toda representação externa;
4. cobertura lexical continua sendo um problema separado, exemplificado por
   `plane tree`;
5. a linha de WSD externo merece continuar, mas com ConSeC como instrumento
   preferido e LMMS apenas como baseline negativo.

## Limitações

- Há apenas um anotador humano e não há medida de concordância.
- O anotador usou Google Translate para compreender inglês histórico.
- O teste contém somente 19 itens, dos quais 16 são ferramentas.
- Não usamos o feedback loop completo do ConSeC.
- O closed set não cobre o sentido botânico.

## Decisão

Não escalar imediatamente para os 37 alvos.

O próximo passo de maior valor é executar o mesmo ConSeC congelado sobre
**todo o Gate 1 de `plane`**:

```text
D0 geometria: 182 ocorrências
D0 ferramenta: 19 ocorrências, reportando rótulo original e humano
D1 aviação: 208 ocorrências
```

Se o Gate 1 completo for aprovado, executar um piloto pequeno com 3 a 5
palavras adicionais que tenham inventários WordNet claros. Só depois desse
piloto decidir se um atlas WSD externo é sustentável para os 37 alvos.

## Artefatos

```text
scripts/evaluate_consec_plane_adjudicated.py
outputs/external_wsd/consec_plane_adjudicated/predictions.csv
outputs/external_wsd/consec_plane_adjudicated/summary.json
```

Fontes oficiais:

- https://github.com/SapienzaNLP/consec
- https://aclanthology.org/2021.emnlp-main.112/
