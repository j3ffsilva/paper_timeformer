# 25. Consolidação da análise de sentidos

## O ponto de chegada

Depois do bootstrap estratificado, a linha de validação por inventário externo
deixou de precisar de novas métricas abertas. Ela já possuía:

1. uma medida explícita de mudança da distribuição de sentidos;
2. replicação em três amostras;
3. um nulo dentro de cada palavra;
4. alinhamento entre sentidos e vetores nas mesmas ocorrências;
5. decomposição entre composição e deriva;
6. incerteza agregada e individual.

Continuar procurando métricas de sentidos criaria flexibilidade analítica sem
responder a uma lacuna claramente definida. Por isso, essa linha entrou em
consolidação. Isso não encerrou a investigação principal de vizinhanças
`token@time`.

## O que foi produzido

Um único script passou a gerar três figuras, três tabelas e oito contextos
auditados. Isso transforma arquivos dispersos de resultados em uma narrativa
reproduzível:

```text
ConSeC detecta mudança de sentidos
  -> a geometria contextual contém estrutura local de sentidos
  -> parte do deslocamento temporal vem da troca de composição
  -> o efeito agregado é estável
  -> alguns casos individuais são interpretáveis e robustos
```

## O papel dos exemplos

`plane`, `multitude`, `gas` e `record` foram usados para tornar a análise
concreta. Eles não substituem a estatística:

- `plane` mostra substituição dominante;
- `gas` mostra diversificação;
- `record` mostra emergência de um sentido cultural/material;
- `multitude` mostra por que definições são mais confiáveis que nomes de
  synsets pouco intuitivos.

## A cautela que permanece

A decomposição associa cerca de 5% da direção do deslocamento à composição
explícita de sentidos. A natureza do restante permanece indeterminada pelo
método. Vizinhos e contextos podem orientar uma investigação especializada,
mas não autorizam uma classificação automática do residual.

## Próximo capítulo

O pacote está pronto como seção de validação semântica. O próximo trabalho
principal é recolocar `token@time` no centro e apresentar, de forma
reproduzível, as vizinhanças por período e os vizinhos ganhos e perdidos.
