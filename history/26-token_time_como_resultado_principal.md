# 26. Recolocando `token@time` no centro

## A correção conceitual

Durante a validação com ConSeC, o projeto passou a falar como se seu sucesso
dependesse de inferir automaticamente sentidos WordNet. Isso invertia a
hierarquia científica.

O TimeFormer foi construído primeiro como instrumento de consulta temporal:

```text
palavra + período
  -> token@time
  -> perfil relacional naquele período
  -> vizinhos e relações características
```

Os vizinhos não são apenas um passo intermediário para chegar a sentidos. Eles
são o resultado principal: mostram como uma palavra se posiciona em relação ao
vocabulário em cada momento histórico.

## O que precisava ser isolado

Comparar coordenadas brutas de checkpoints diferentes confundiria mudança
lexical com rotação, reflexão, translação ou escala do espaço. Os perfis
relacionais foram introduzidos justamente para comparar relações internas,
com centralização e invariâncias explicitamente qualificadas.

Assim, a pergunta principal não é:

```text
o modelo descobriu sozinho o nome correto do sentido?
```

É:

```text
quais relações caracterizam w@t0 e w@t1,
e quais relações foram ganhas ou perdidas?
```

## Onde entram ConSeC e WordNet

Depois de obter o deslocamento temporal, surge uma segunda pergunta:

> Quanto desse deslocamento coincide com mudanças de sentidos discretos
> reconhecidos por um inventário lexical externo?

ConSeC mostrou que:

- diferenças de posteriores de sentido acompanham parcialmente distâncias
  entre vetores das mesmas ocorrências;
- uma parcela positiva da direção do deslocamento pode ser associada à mudança
  da mistura de sentidos;
- essa associação é robusta em dez palavras individuais.

Isso fornece validade semântica adicional. Não redefine o TimeFormer como um
classificador de sentidos.

## O papel do pesquisador

Quando a análise automática não atribui a natureza de um deslocamento, o
resultado continua útil:

```text
vizinhos de w@t0
vizinhos de w@t1
relações ganhas e perdidas
contextos históricos correspondentes
```

Esses elementos são insumos para linguistas históricos e outros especialistas.
Eles podem distinguir mudança de sentido, especialização, metáfora, mudança de
domínio, gênero textual ou transformação social usando conhecimento que o
modelo não possui.

## A trilha daqui por diante

1. consolidar as vizinhanças `token@time` como resultado principal;
2. demonstrar a remoção do problema de coordenadas com a formulação relacional;
3. apresentar ConSeC/WordNet como análise secundária de validade e
   interpretabilidade;
4. manter a componente não atribuída aberta à investigação especializada;
5. evitar transformar ausência de classificação automática em ausência de
   informação temporal.
