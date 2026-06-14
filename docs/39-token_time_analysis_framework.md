# Framework de análise `token@time`

**Status:** especificação canônica de produto científico e implementação.

## Objetivo

O framework deve permitir consultar e comparar palavras temporalmente sem
exigir uma classificação automática de sentidos:

```text
token@time
  -> perfil relacional
  -> vizinhos no período
  -> deslocamentos entre períodos
  -> trajetória em múltiplos períodos
  -> busca e ranking
```

As dimensões do perfil são referências lexicais compartilhadas. Portanto, uma
mudança pode ser descrita por relações concretas, não por coordenadas ocultas
sem nome.

## Objetos fundamentais

### `TokenTimeProfile`

```text
R_t(w)[v] = relação entre a palavra w e a referência v no período t
```

Deve preservar palavra, período, checkpoint, camada, vocabulário de
referência, vetor relacional completo, contagem de ocorrências, seed e
metadados de extração.

### `TokenTimeDisplacement`

```text
Delta(w, a, b) = R_b(w) - R_a(w)
```

O vetor informa quais relações aumentaram ou diminuíram. Um score escalar é
apenas um resumo desse objeto.

### `TokenTimeTrajectory`

```text
T(w) = [R_0(w), R_1(w), ..., R_n(w)]
```

Uma trajetória verdadeira exige pelo menos três períodos. Com apenas D0 e D1,
temos um deslocamento de dois pontos, não evidência sobre aceleração,
reversão, oscilação ou forma do caminho.

## Operações do framework

### 1. Consultar `token@time`

```text
query(w, t, k)
```

Retorna vizinhos, similaridades, contextos representativos, incerteza entre
seeds/amostras e metadados da régua.

Pergunta respondida:

> Quais relações caracterizam esta palavra neste período?

### 2. Comparar dois `token@time`

```text
compare(w@a, w@b)
compare(w@a, u@b)
```

Deve oferecer:

- vetor completo `Delta`;
- relações ganhas e perdidas;
- mudança de ranking;
- sobreposição do topo da vizinhança;
- distância entre perfis;
- estabilidade entre seeds.

Comparar a mesma palavra entre períodos descreve mudança temporal. Comparar
palavras diferentes responde proximidade relacional, não mudança.

### 3. Descrever a trajetória de uma palavra

```text
trajectory(w)
```

Para três ou mais períodos:

- magnitude acumulada;
- comprimento total do caminho;
- magnitude de cada passo;
- eficiência do deslocamento;
- recuperação;
- período de pico;
- concentração de evento;
- deriva antes e depois do evento;
- velocidade e aceleração relacionais.

As métricas `final_magnitude`, `path_length`, `displacement_efficiency`,
`recovery`, `peak_period` e métricas locais de evento já existem em
`src/timeformers/structural_metrics.py`, mas ainda precisam ser conectadas ao
pipeline real de `token@time`.

### 4. Comparar trajetórias

```text
compare_trajectories(T(w), T(u), mode)
```

Devem existir três modos, porque "trajetória similar" é ambíguo.

#### Similaridade de direção

Compara os vetores de deslocamento passo a passo. Encontra palavras que se
aproximam e se afastam de referências semelhantes.

#### Similaridade de forma

Compara curvas normalizadas de magnitude. Encontra, por exemplo, duas palavras
com mudança abrupta no mesmo momento, mesmo que caminhem para campos lexicais
diferentes.

#### Similaridade de assinatura

Compara um vetor resumido:

```text
[M_final, L, eficiência, recuperação, pico, concentração]
```

Encontra trajetórias funcionalmente parecidas. É útil para busca exploratória,
mas não substitui a comparação vetorial completa.

Com apenas dois períodos, só a similaridade de **direção do deslocamento** e de
magnitude está disponível. Chamar isso de similaridade de forma seria
incorreto.

### 5. Encontrar trajetórias similares

```text
nearest_trajectories(w, k, mode)
```

Fluxo:

1. construir a trajetória de cada palavra;
2. escolher explicitamente `direction`, `shape` ou `signature`;
3. normalizar apenas as propriedades pertinentes ao modo;
4. calcular matriz palavra-palavra;
5. retornar vizinhos de trajetória com explicação das dimensões concordantes.

O relatório deve mostrar por que duas trajetórias foram aproximadas, evitando
uma lista opaca de resultados.

### 6. Encontrar quem mais e quem menos mudou

Não deve existir um único ranking universal. O framework deve retornar
rankings distintos:

| Ranking | Pergunta |
|---|---|
| deslocamento final | quem terminou mais longe do estado inicial? |
| caminho total | quem apresentou mais atividade temporal? |
| turnover de vizinhos | quem mais substituiu o topo da vizinhança? |
| persistência | quem mudou e permaneceu no novo estado? |
| recuperação | quem retornou ao perfil inicial? |
| evento concentrado | quem teve uma ruptura localizada? |

Para "quem menos mudou", valores baixos não bastam. É preciso considerar
intervalo entre seeds ou bootstrap, comparação com nulo/placebo, frequência,
número de ocorrências e potência para detectar mudança.

A saída correta deve distinguir:

```text
evidência de estabilidade
ausência de evidência de mudança
dados insuficientes
```

## Isolamento do sistema de coordenadas

O framework compara relações internas sobre referências lexicais
compartilhadas. Centralização e cosseno removem translação e são invariantes a
rotação, reflexão e escala isotrópica comuns.

Isso evita comparar diretamente dimensões ocultas arbitrárias, mas não elimina
por definição deformações anisotrópicas ou não lineares. Por isso, toda
consulta deve registrar referências, centralização, checkpoint, corpus,
estabilidade entre seeds e controles de régua fixa quando aplicáveis.

## Estado atual da implementação

| Capacidade | Estado |
|---|---|
| matrizes relacionais e deltas | implementada em `src/timeformers/relational.py` |
| mudança de vizinhos e rankings | implementada parcialmente em `src/timeformers/relational_metrics.py` |
| relatório D0/D1 de vizinhanças | implementado para o encoder antigo |
| métricas multitemporais | implementadas no benchmark sintético |
| relatório no `bert-tiny` integral | faltante |
| objeto/API unificada `token@time` | faltante |
| comparação palavra-palavra de trajetórias | faltante |
| busca de trajetórias similares | faltante |
| rankings com incerteza e nulo | faltante no pipeline unificado |
| trajetória real com 3+ períodos | depende de corpus temporal mais granular |

## Arquitetura de implementação proposta

```text
src/timeformers/token_time.py
  TokenTimeProfile
  TokenTimeDisplacement
  TokenTimeTrajectory

src/timeformers/token_time_metrics.py
  compare_profiles
  compare_trajectories
  trajectory_signature
  rank_change

src/timeformers/token_time_index.py
  nearest_profiles
  nearest_trajectories

scripts/build_token_time_profiles.py
scripts/report_token_time_neighborhoods.py
scripts/rank_token_time_change.py
scripts/find_similar_trajectories.py
```

Os nomes são propostas; a implementação deve reutilizar
`relational.py`, `relational_metrics.py` e `structural_metrics.py`, sem
duplicar fórmulas.

## Ordem de implementação

### Fase A: dois períodos, melhor encoder

1. extrair perfis D0/D1 no `bert-tiny` integral;
2. produzir vizinhos, ganhos e perdas;
3. agregar as duas seeds;
4. reportar estabilidade;
5. gerar rankings separados de deslocamento e turnover.

### Fase B: comparação e busca

1. implementar comparação entre deslocamentos;
2. criar busca por deslocamentos de direção semelhante;
3. explicar cada resultado pelas referências que mais contribuíram.

No SemEval, essa fase procura **deslocamentos similares**, não trajetórias
multitemporais similares.

### Fase C: trajetórias reais

1. escolher ou construir corpus com três ou mais períodos;
2. gerar `R_t(w)` para todos os períodos;
3. conectar `structural_metrics.py`;
4. implementar similaridade de forma e assinatura;
5. buscar trajetórias semelhantes;
6. rankear persistência, recuperação, atividade e eventos.

## Critério de conclusão do framework

O framework estará completo quando uma consulta puder responder:

```text
Quais eram os vizinhos de w em t?
O que w ganhou e perdeu entre a e b?
Quanto e de que forma w mudou?
Quais palavras seguiram deslocamentos ou trajetórias semelhantes?
Quais mudaram mais, quais mudaram menos e com qual incerteza?
Quais evidências devem ser entregues ao pesquisador para interpretação?
```
