# Bootstrap estratificado da decomposição

**Data:** 2026-06-14
**Decisão pré-registrada:** conclusão global estável.

## Pergunta

A contribuição de composição permanece positiva quando as ocorrências de D0 e
D1 são reamostradas?

Foram executadas 2.000 réplicas estratificadas por palavra, período e amostra
ConSeC. Os mesmos índices foram usados nas duas seeds TimeFormer.

## Resultado agregado

Para `layer_2`:

```text
excesso médio observado       = 0,0484
mediana bootstrap da média    = 0,0421
IC bootstrap 95% da média     = [0,0317; 0,0525]
```

Todos os critérios pré-registrados passaram:

```text
IC agregado acima de zero               = sim
>= 15 palavras com P(excesso>0) > 0,5   = sim (25/25)
plane robustamente positivo             = sim
rho observado × mediana bootstrap > 0,8 = sim (0,919)
```

Portanto:

```text
Conclusão global da decomposição = ESTÁVEL
```

A mediana bootstrap é ligeiramente menor que a estimativa observada. Isso é
esperado para uma razão não linear de projeções e recomenda usar `0,042` como
descrição conservadora da contribuição média.

## Estabilidade individual

Dez palavras tiveram IC 95% inteiramente acima de zero:

| Palavra | Mediana bootstrap | IC 95% |
|---|---:|---:|
| `plane_nn` | 0,246 | [0,110; 0,363] |
| `multitude_nn` | 0,134 | [0,061; 0,212] |
| `gas_nn` | 0,108 | [0,041; 0,186] |
| `record_nn` | 0,061 | [0,008; 0,125] |
| `land_nn` | 0,046 | [0,005; 0,103] |
| `attack_nn` | 0,045 | [0,006; 0,101] |
| `bit_nn` | 0,044 | [0,001; 0,101] |
| `thump_nn` | 0,043 | [0,003; 0,098] |
| `risk_nn` | 0,038 | [0,012; 0,071] |
| `fiction_nn` | 0,023 | [0,001; 0,060] |

Nenhuma palavra foi robustamente negativa. As outras 15 tiveram intervalos
incluindo zero.

Consequentemente:

```text
evidência agregada forte
evidência individual robusta em 10/25
```

## Extremos pré-especificados

| Palavra | Observado | Mediana | IC 95% | Classificação |
|---|---:|---:|---:|---|
| `plane_nn` | 0,341 | 0,246 | [0,110; 0,363] | robusta positiva |
| `multitude_nn` | 0,177 | 0,134 | [0,061; 0,212] | robusta positiva |
| `gas_nn` | 0,147 | 0,108 | [0,041; 0,186] | robusta positiva |
| `record_nn` | 0,069 | 0,061 | [0,008; 0,125] | robusta positiva |
| `player_nn` | -0,007 | 0,004 | [-0,016; 0,033] | incerta |
| `donkey_nn` | -0,002 | 0,001 | [-0,008; 0,016] | incerta |
| `stab_nn` | 0,001 | 0,012 | [-0,049; 0,093] | incerta |

Os sinais negativos pontuais de `player` e `donkey` não sobrevivem à
reamostragem.

## Sensibilidade de camada

| Camada | Mediana bootstrap da média | IC 95% | robustas positivas |
|---|---:|---:|---:|
| `layer_1` | 0,029 | [0,014; 0,044] | 3 |
| `layer_2` | 0,042 | [0,032; 0,053] | 10 |

As duas camadas sustentam uma contribuição agregada positiva, mas `layer_2`
produz evidência individual mais estável.

## Auditoria por sentido

### `plane_nn`

| Sentido | Massa D0 → D1 | Contribuição |
|---|---:|---:|
| plano matemático | 0,913 → 0,130 | +0,521 |
| avião | 0,018 → 0,816 | -0,183 |

A soma líquida é `0,340`. O sinal de cada parcela depende da projeção do
centróide do sentido no deslocamento total; não é determinado apenas pelo
aumento ou redução da massa.

### `multitude_nn`

```text
"grande número indefinido": 0,407 -> 0,642
contribuição = +0,162
```

Esse sentido domina a contribuição robusta, apesar da JSD global moderada.
Como o synset WordNet associado tem rotulagem pouco intuitiva
(`battalion.n.02`), `multitude` deve receber inspeção textual antes de ser
usada como exemplo central.

### `gas_nn`

Os usos gerais de substância gasosa caem, enquanto combustível e gasolina
crescem. A contribuição líquida é positiva e robusta, mas as parcelas de
sentidos novos apontam parcialmente contra a direção total.

### `record_nn`

O sentido geral de registro/evidência cai de `0,583` para `0,236`; gravação
fonográfica e recorde esportivo crescem. A contribuição líquida permanece
positiva e robusta.

### Casos baixos

`player`, `donkey` e `stab` apresentam cancelamento entre sentidos e intervalos
amplos. Eles não sustentam conclusões individuais.

## Interpretação

O bootstrap modifica a força das afirmações:

- podemos afirmar que existe contribuição média de composição;
- podemos destacar dez palavras com evidência individual robusta;
- não podemos tratar todas as 23 estimativas pontualmente positivas como
  descobertas individuais;
- `plane` é o exemplo mais estável e semanticamente transparente;
- `multitude` é estatisticamente robusta, mas precisa de auditoria textual.

## Próximo passo desta linha

Esta linha de validação por sentidos já não precisa criar outra métrica. Deve:

1. auditar contextos e posteriores dos exemplos que entrarão no artigo;
2. selecionar casos ilustrativos por critérios metodológicos, não pelo maior
   número;
3. consolidar a seção de método e resultados;
4. preparar tabelas e figuras reproduzíveis a partir dos CSVs e caches.

No projeto mais amplo, permanece uma etapa principal: consolidar diretamente
as consultas `token@time`, com vizinhos por período, relações ganhas e perdidas
e estabilidade entre seeds. A análise ConSeC deve interpretar parte dessas
saídas, não substituir as vizinhanças como resultado.

## Artefatos

```text
outputs/consec_timeformer_soft_decomposition_bootstrap/
scripts/bootstrap_soft_sense_decomposition.py
scripts/audit_soft_decomposition_senses.py
docs/35-soft_decomposition_stratified_bootstrap_preregistration.md
```
