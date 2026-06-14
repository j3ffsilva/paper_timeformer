# 14. Gate 1 completo com ConSeC

O teste inicial do ConSeC havia coberto apenas as 19 ocorrências auditadas de
`plane` como ferramenta. O passo seguinte foi repetir toda a porta
confirmatória originalmente desenhada:

```text
D0 geometria     182
D0 ferramenta     19
D1 aviação       208
```

## Dois resultados preservados

O projeto mantém dois fatos separados:

- LMMS falhou na porta original e permanece `NO-GO`;
- ConSeC foi executado posteriormente como segundo instrumento externo.

Isso evita reescrever a história experimental depois de observar a falha.

## Resultado do ConSeC

Com os rótulos originais:

```text
geometria  100,0%
ferramenta  78,9%
aviação     99,0%
macro       92,7%
```

Depois da adjudicação humana:

```text
geometria   99,5%
ferramenta  87,5%
aviação     99,0%
macro       95,3%
```

Todos os intervalos e cortes exigidos passaram. A frase histórica `inclined
plane` permaneceu geométrica.

## Mudança de decisão

O Gate 1 do ConSeC é `GO`. Isso não autoriza os 37 alvos, mas autoriza uma
Porta 2 pequena com `graft_nn`, `chairman_nn` e `tree_nn`.

O ponto científico agora é testar **cobertura entre palavras**, não continuar
comparando modelos em `plane`.

Relatório completo:
[Gate 1 completo](../docs/19-consec_plane_gate1_full_results.md).
