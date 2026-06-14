# 27. O framework de consultas temporais

## Da demonstração ao instrumento

Recolocar `token@time` no centro tornou necessário explicitar o que o
instrumento deve fazer. Não basta mostrar uma lista de vizinhos para quatro
palavras. O framework precisa oferecer operações reproduzíveis.

## As seis consultas

```text
1. consultar w@t
2. comparar w@t0 com w@t1
3. reconstruir a trajetória de w
4. comparar trajetórias de palavras
5. encontrar trajetórias semelhantes
6. rankear quem mais e menos mudou
```

Cada consulta preserva a saída vetorial ou relacional. Scores escalares servem
para ordenar e resumir, não para substituir a evidência concreta.

## Uma distinção temporal decisiva

O SemEval possui apenas dois períodos. Assim, ele permite medir:

```text
estado inicial -> estado final
```

Isso produz um deslocamento. Não permite observar sozinho:

```text
gradualidade
aceleração
ruptura intermediária
reversão
oscilação
```

Essas propriedades exigem pelo menos três períodos. No corpus atual, "palavras
com trajetórias semelhantes" significa rigorosamente "palavras com
deslocamentos semelhantes". O framework mantém os dois conceitos separados.

## Mais mudou e menos mudou

O projeto também abandonou a ideia de um ranking universal. Uma palavra pode:

- terminar longe, mas percorrer um caminho direto;
- mover-se muito e voltar;
- trocar quase todos os vizinhos;
- sofrer um único evento concentrado;
- parecer estável apenas porque há poucos dados.

Por isso, o framework produz rankings de deslocamento final, atividade,
turnover, persistência, recuperação e evento. "Menos mudou" só é uma conclusão
quando existe incerteza suficiente para distinguir estabilidade de falta de
potência.

## O que já existia

As peças estavam espalhadas:

- relações, deltas e top-k em `relational.py`;
- turnover e mudança de ranking em `relational_metrics.py`;
- caminho, eficiência, recuperação e eventos em `structural_metrics.py`;
- relatórios de vizinhança no pipeline antigo.

Faltavam objetos comuns, busca por trajetórias semelhantes e aplicação ao
melhor encoder. A especificação completa foi registrada em
`docs/39-token_time_analysis_framework.md`.

## Próximo passo

A primeira implementação deve permanecer no caso de dois períodos: perfis
`token@time` do `bert-tiny` integral, vizinhos, ganhos, perdas, estabilidade
entre seeds e rankings separados. Só depois o projeto deve avançar para um
corpus com resolução temporal suficiente para trajetórias completas.
