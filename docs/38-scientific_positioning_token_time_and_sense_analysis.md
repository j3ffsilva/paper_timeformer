# Posicionamento científico: `token@time`, perfis relacionais e sentidos

**Status:** orientação canônica para os próximos experimentos e para o artigo.

## Contribuição principal

O TimeFormer é um instrumento de investigação temporal. Para uma palavra `w`
e um período `t`, ele produz um objeto consultável:

```text
w@t -> R_t(w)
```

`R_t(w)` é um perfil relacional: descreve `w` pelas relações que mantém com
outras referências lexicais naquele período. As saídas diretamente
interpretáveis são:

```text
vizinhos de w@t
relações que se fortalecem
relações que enfraquecem
trajetória do perfil entre períodos
```

Essas relações são resultados, independentemente de poderem ser convertidas
automaticamente em rótulos de sentidos.

## Problema metodológico principal

O desafio necessário para interpretar o deslocamento é não confundi-lo com
uma mudança arbitrária do sistema de coordenadas entre checkpoints. O desenho
relacional compara similaridades internas sobre referências compartilhadas,
em vez de comparar coordenadas brutas.

A alegação deve permanecer qualificada: centralização e cosseno removem
translação e são invariantes a rotação, reflexão e escala isotrópica comuns.
Deformações anisotrópicas ou não lineares não desaparecem por definição e
devem ser tratadas por controles empíricos.

## Pergunta semântica secundária

Depois de estabelecer o deslocamento relacional, perguntamos:

> Quanto dessa reorganização temporal se associa a mudanças de sentidos
> lexicais no sentido mais estrito?

ConSeC e WordNet fornecem uma régua externa para essa pergunta. Eles permitem
relacionar:

```text
mudança de P_t(sentido | palavra)
com
mudança de R_t(palavra)
```

Essa análise oferece validade semântica adicional, mas não é a definição de
sucesso do TimeFormer.

## Interpretação dos resultados atuais

Os resultados sustentam que uma parcela positiva e replicada do deslocamento
se associa à recomposição de sentidos reconhecida pelo inventário. Na
decomposição principal da `layer_2`, o excesso médio dessa contribuição foi
`0,048`, com IC 95% `[0,024; 0,081]`.

O complemento algébrico não recebe uma interpretação automática. O estudo não
identifica se ele representa:

- mudança semântica fora do inventário;
- mudança contextual dentro de um sentido;
- tópico, gênero ou domínio;
- transformação histórica das próprias referências;
- ou uma combinação desses fatores.

Essas são hipóteses para investigação, não conclusões do método.

## Papel da análise especializada

O TimeFormer pode oferecer a pesquisadores:

1. listas de vizinhos em cada período;
2. vizinhos ganhos e perdidos;
3. contextos históricos que sustentam essas relações;
4. palavras e relações prioritárias para auditoria;
5. trajetórias em corpora com mais de dois períodos.

Especialistas podem combinar esses insumos com leitura histórica, fontes
documentais e teoria linguística para avaliar a natureza da mudança. O método
não pretende substituir essa inferência.

## Hierarquia para o artigo

1. **Resultado principal:** `token@time` e deslocamento de vizinhanças
   relacionais temporalmente específicas.
2. **Garantia metodológica:** comparação relacional que evita depender de
   alinhamento geométrico post-hoc, com invariâncias claramente delimitadas.
3. **Validação externa:** relação entre geometria contextual e posteriores
   ConSeC nas mesmas ocorrências.
4. **Análise de sentidos:** parcela do deslocamento associada à recomposição de
   sentidos WordNet.
5. **Uso científico:** insumos auditáveis para interpretação por especialistas.

## Próximo passo experimental

Implementar a primeira fase do framework de análise `token@time`, produzindo:

```text
vizinhos em D0
vizinhos em D1
ganhos e perdas relativos
estabilidade entre seeds
contextos ilustrativos
```

A análise ConSeC deve acompanhar esse relatório como camada interpretativa,
sem filtrar ou redefinir quais vizinhos contam como resultado.

A especificação de consultas, comparações, busca de trajetórias e rankings está
em `docs/39-token_time_analysis_framework.md`.

## Documentos relacionados

- `history/07-realinhamento_instrumento_de_consulta_temporal.md`
- `history/21-duas_reguas_que_nao_sao_equivalentes.md`
- `history/26-token_time_como_resultado_principal.md`
- `docs/32-occurrence_level_consec_timeformer_results.md`
- `docs/34-soft_sense_vector_decomposition_results.md`
- `docs/37-consec_timeformer_article_package.md`
- `docs/39-token_time_analysis_framework.md`
