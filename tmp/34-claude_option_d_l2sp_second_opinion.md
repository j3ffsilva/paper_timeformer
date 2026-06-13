# Segunda opinião: redistribuição por camada, falha do L2-SP e próximo passo

## Diagnóstico em uma frase

Os dados são consistentes com "redistribuição entre camadas", mas também
com uma explicação mais aborrecida e mais provável — **ruído de n=37
amplificado por seleção de checkpoint sem intervalo de confiança, mais um
viés conhecido de que a última camada de um BERT pequeno é um péssimo
readout para similaridade semântica mesmo sem qualquer fine-tuning** —
e o desenho atual não separa essas duas hipóteses.

Antes de investir em distillation, eu pediria duas coisas baratas que
faltam: (1) intervalos de confiança/bootstrap em cada Spearman já
calculado, e (2) remover um confound de orçamento de treino que encontrei
no controle pseudo-período (abaixo, pergunta 2). Sem isso, não dá para
saber se há algo para "resgatar" na layer 2.

## Respostas às 10 perguntas

### 1. A evidência sustenta redistribuição entre camadas, ou há explicação alternativa mais provável?

Há pelo menos duas explicações alternativas que os números atuais não
excluem:

**(a) Ruído estatístico em n=37.** Com 37 pares, o erro padrão de um
Spearman gira em torno de `1/sqrt(n-3) ≈ 0,17`. Quase todas as diferenças
reportadas — `layer1`: `0,298 -> 0,325/0,322/0,340/0,341`; `layer2`:
`0,136 -> 0,030/0,038/0,012/0,014/0,116/0,153` — estão dentro dessa
margem. O salto mais dramático, `layer2` indo de `0,136` (congelado) para
`-0,019` já em `theta0` (seed 1000), pode ser causado por uma reordenação
de poucos pares de palavras com APD próximas. Sem bootstrap por palavra
em cada condição, "a layer 2 perdeu o sinal" e "a layer 2 sempre teve um
sinal instável que reordenou por acaso" são indistinguíveis.

**(b) Layer 2 nunca foi um bom readout, mesmo congelada.** Em BERT
pequenos, é bem documentado que a última camada tende a ser pior para
similaridade semântica do que camadas intermediárias (a representação
fica especializada para o objetivo de pré-treino, não para STS/sentido).
O baseline congelado já mostra `layer1=0,298 > layer2=0,136`. Se layer 2
já era marginal antes de qualquer treino temporal, então "a layer 2 caiu
para ~0,03" não é necessariamente "o fine-tuning destruiu algo que
funcionava" — pode ser "o fine-tuning empurrou um sinal fraco e instável
mais perto de zero", o que é um fenômeno bem menos interessante.

Eu não descartaria redistribuição — o padrão `layer1` estável/levemente
melhor e `layer2` degradando é consistente em quase todas as condições —,
mas a magnitude do efeito (poucos centésimos de Spearman) está no
território onde (a) e (b) são pelo menos tão prováveis quanto uma
história causal de "o sinal migrou de camada".

### 2. O contraste cronológico vs pseudo-período é interpretável com uma única seed pseudo?

Não, e há um problema adicional além do número de seeds: **a seleção de
checkpoint do controle pseudo-período usou um orçamento de treino
diferente do cronológico**. Pelos relatórios:

```text
cronológico: theta1 = D1@2 épocas (completo)
pseudo:      theta1 = pseudo-D1@0,5 época
```

Isso é um confound direto para a comparação de `layer2`
(`0,030–0,038` cronológico vs `0,153` pseudo, ou `0,014` vs `0,116` com
L2-SP). `theta1` pseudo viu 4x menos passos de treino no "segundo
período" que `theta1` cronológico. Menos treino tende, por si só, a
preservar mais a geometria inicial — exatamente o que `layer2` pseudo
mostra. Então a diferença observada pode ser inteiramente "quantidade de
gradiente aplicado", não "ordem cronológica vs aleatória".

Antes de adicionar uma segunda seed pseudo, eu corrigiria isso: ou (i)
forçar a mesma regra de seleção a escolher checkpoints com número de
passos comparável nas duas condições, ou (ii) reportar `layer2` em
*todos* os checkpoints salvos (0,25/0,5/1/2 épocas) para ambas as
condições, para visualizar a trajetória completa em vez de comparar dois
pontos escolhidos por critérios potencialmente desalinhados.

### 3. Distillation ponto a ponto, relacional, ou ambas — qual controle mínimo tem maior valor informacional?

Ponto a ponto (cosseno por âncora) é o controle mínimo certo para o
**Passo 1**, porque ele é o paralelo direto do L2-SP que já rodaram: L2-SP
restringiu parâmetros e não recuperou o ranking; distillation ponto a
ponto restringe a *saída* da camada 2 nas mesmas âncoras. É a comparação
mais limpa para a pergunta "proximidade paramétrica != preservação
funcional".

Dito isso, vale registrar a limitação que isso já compartilha com a
"Parede C" do outro documento: a métrica final (APD/Spearman) é
*relacional* por construção — ela compara distâncias entre ocorrências de
D0 e D1. Preservar cossenos ponto a ponto contra `theta_init` não garante
preservar a *matriz* de distâncias entre ocorrências do próprio modelo
temporal, que é o que a APD de fato lê. Por isso, se o ponto a ponto
"funcionar" (cosseno alto com `theta_init` preservado) mas o Spearman
continuar baixo, isso seria evidência forte de que o problema não é
geometria de representação, mas sim o que a "Parede A/C" já apontava:
mudança em geometria contextual não é o mesmo que mudança em sentido.
Eu rodaria ponto a ponto primeiro (é o controle causal do L2-SP), mas
guardaria o termo relacional (`||C_student - C_teacher||`) como
diagnóstico complementar barato — calculável no mesmo lote de âncoras,
mesmo que não entre na loss do piloto.

### 4. O teacher deve ser `theta_init`, o checkpoint anterior, ou uma média móvel?

`theta_init` para o **Passo 1**, pelo mesmo motivo da pergunta 3: cria o
paralelo direto com o L2-SP (que também ancorou em `theta_init`). Se
"distillation funcional para `theta_init`" não bater o que "L2-SP de
pesos para `theta_init`" já fez, a conclusão é direta: nem proximidade
paramétrica nem proximidade funcional a `theta_init` salvam o ranking de
`layer2`.

Teacher = checkpoint do período anterior (ex.: `theta0` como teacher
durante o treino de D1) responde a uma pergunta diferente —
"continual learning sem esquecer D0" — que é legítima, mas é um
experimento separado e introduz um teacher que já está parcialmente
contaminado pelo próprio regime de treino que se quer regularizar.
Média móvel (EMA) adiciona um hiperparâmetro (taxa de decaimento) que
vocês teriam que calibrar sem gold, e historicamente EMA-teacher é mais
útil para estabilizar treino (BYOL/mean teacher) do que para preservar
uma geometria de referência externa — não parece o encaixe certo aqui.
Eu manteria `theta_init` para o piloto e só consideraria checkpoint
anterior se o objetivo mudar explicitamente para "estabilidade entre
períodos" em vez de "preservar geometria pré-treinada".

### 5. Âncoras D0-apenas em D0 e replay D0+D1 em D1 evita vazamento e testa a hipótese certa?

Evita vazamento de rótulos (não usa `truth.tsv`, não usa as 37 palavras
para escolher âncoras — assumindo que isso seja garantido no código).
Mas há um ponto que o desenho não menciona explicitamente e que importa
muito: **as âncoras devem excluir ocorrências das 37 palavras-alvo (e
idealmente também dos 37 controles pareados por frequência)**. Se as
sentenças-âncora incluírem ocorrências de `plane`, `chairman`, etc., a
distillation estaria literalmente puxando a representação dessas
palavras de volta para `theta_init`, o que suprimiria diretamente o
sinal de mudança temporal que a APD tenta medir — uma forma de
vazamento que não passa por `truth.tsv`, mas que ainda assim contamina o
experimento de cima a baixo. Vale adicionar essa restrição explicitamente
ao protocolo e documentá-la.

Sobre o desenho "D0-apenas em D0, replay D0+D1 em D1": ele testa bem a
pergunta "o treino em D1 consegue aprender o domínio novo sem desviar as
representações de documentos antigos (e do `theta_init`) para sentenças
que não envolvem as palavras-alvo?". Isso é uma hipótese razoável, mas é
uma hipótese sobre *estabilidade geral do encoder*, não sobre
*preservação do sinal lexical específico das 37 palavras*. Como a seção
de controles pareados por frequência já mostrou que alvos e controles têm
magnitude de APD parecida, é plausível que estabilizar "o encoder em
geral" (via âncoras genéricas) não tenha efeito diferencial sobre o
ranking das 37 palavras — outro motivo para tratar isso como diagnóstico
de plasticidade do encoder, não como tentativa de "salvar" o resultado do
SemEval.

### 6. Como calibrar alpha/beta sem usar os 37 alvos?

A ideia de calibrar pela razão de normas de gradiente no piloto é
razoável, mas tem um risco: a razão de normas pode variar bastante com a
amostra de âncoras escolhida (lote pequeno, alta variância). Eu sugeriria:

1. fixar a amostra de âncoras com uma seed própria, separada da seed de
   treino, e medir a razão de normas em vários lotes (não um só) para
   obter uma faixa, não um ponto;
2. pré-registrar a regra (ex.: "`alpha` tal que
   `||grad_distill|| ≈ 0,1 * ||grad_MLM||` no passo 0") *antes* de rodar
   qualquer avaliação SemEval;
3. depois de fixado, rodar uma checagem de sensibilidade com `alpha/2` e
   `alpha*2` (decidido a priori, não escolhido a posteriori) — se o
   resultado mudar de "ajuda" para "não ajuda" dentro desse fator 2, isso
   por si só já é uma resposta (o efeito é frágil demais para ser
   defensável), e vocês já teriam o experimento de "duas variações" sem
   declarar isso uma grade.

`beta` (termo relacional) eu nem calibraria no piloto — começaria com
`beta=0`, já que a pergunta 3 sugere usar o termo relacional só como
diagnóstico, não como parte da loss inicial.

### 7. O controle de ordem invertida deve preceder a distillation?

Sim, e a pergunta 2 é o motivo concreto: o contraste cronológico vs
pseudo que motiva a distillation tem um confound de orçamento de treino
que ainda não foi isolado. Se esse confound explicar boa parte da
diferença em `layer2`, então "a layer 2 é mais danificada pela ordem
cronológica" — a premissa do experimento de distillation — pode estar
errada ou exagerada. Rodar distillation antes de resolver isso arrisca
gastar o experimento "bom" (distillation) para responder a uma pergunta
mal especificada (cronológico vs pseudo). Eu inverteria a ordem dos
Passos 1 e 3 do plano: primeiro reforçar o controle cronológico/pseudo
(orçamento comparável + ordem invertida D1->D0 + segunda seed pseudo),
e só então gastar o orçamento experimental em distillation, já sabendo se
há um efeito de ordem real para "resgatar".

### 8. Há razão científica para resgatar a layer 2, dado que layer 1 é mais estável e melhor correlacionada?

Marginal, e possivelmente nenhuma. Dois pontos:

- `layer1` já era a melhor camada **no baseline congelado**
  (`0,298` vs `0,136`), antes de qualquer treino temporal. O ganho do
  fine-tuning sobre `layer1` (`0,298 -> ~0,32–0,34`) é pequeno e, pela
  pergunta 1, possivelmente dentro do ruído. Ou seja, mesmo "resgatar"
  `layer2` para o nível atual de `layer1` não mudaria
  qualitativamente o resultado do SemEval — vocês já têm `layer1`
  fazendo esse papel.
- A intuição de que "a última camada deveria carregar mais semântica" é
  uma expectativa de arquitetura grande (BERT-base/large), não
  necessariamente de um `bert-tiny` de 2 camadas, onde a "última camada"
  é também "a segunda camada" — muito perto da saída lexical/MLM, com
  pouco espaço para abstração hierárquica. Pode ser que `layer2` em
  `bert-tiny` simplesmente não seja o lugar certo para procurar um
  readout semântico, independentemente do regime de treino.

Eu manteria `layer2` como objeto de análise de plasticidade (como o
próprio documento já sugere no "Passo 2"), mas não gastaria mais um ciclo
de experimentos tentando recuperá-la como régua, a menos que a
distillation ponto a ponto (Passo 1) produza um efeito que sobreviva às
checagens de ruído da pergunta 1.

### 9. Esses resultados fortalecem ou enfraquecem a motivação para a arquitetura externa de WSD/open set?

Fortalecem, mas por um motivo um pouco diferente do que a "Parede A/C"
original argumentava. A motivação original era de **identificabilidade**:
mesmo com um encoder perfeitamente estável, mudança em
`P(contexto|palavra)` não identifica mudança em `P(sentido|palavra)`.

Este novo conjunto de experimentos adiciona uma motivação de
**tratabilidade de engenharia**: mesmo conseguir uma régua contextual
*estável* via MLM continuado é difícil — depende de camada, é sensível à
seleção de checkpoint sem gold, o efeito cronológico vs pseudo tem
confounds de orçamento de treino, e os controles pareados por frequência
mostram que a APD absoluta não distingue alvos de controles. Ou seja,
mesmo abstraindo o problema de identificabilidade, o "eixo de
engenharia/diagnóstico" (régua contextual estável) está consumindo um
esforço considerável para produzir efeitos de poucos centésimos de
Spearman em n=37.

Isso não valida a arquitetura de WSD por si só (ela tem seus próprios
riscos, listados no outro documento), mas reduz o custo de oportunidade
de pausar a linha de regularização do MLM: o retorno marginal por
experimento nessa linha parece baixo, e o "eixo do estimando científico"
permanece inteiramente não testado.

### 10. Qual experimento único antes de encerrar a linha de regularização do MLM temporal?

Não seria a distillation. Seria, em ordem de custo:

1. **Bootstrap por palavra em todos os Spearman já calculados** (custo
   ~zero, reusa as APDs por palavra já produzidas em cada condição).
   Isso responde diretamente se `layer1 ≈ 0,30` vs `~0,32–0,34` e
   `layer2 ≈ 0,14` vs `~0,01–0,15` são distinguíveis entre si e de zero.
   Se os intervalos se sobrepõem amplamente (o que eu esperaria dado
   `n=37`), isso já é evidência forte para a regra de parada que vocês
   mesmos propuseram ("efeito não replicar..."), sem precisar de mais
   nenhum treino.
2. **Corrigir o confound de orçamento de treino no controle
   pseudo-período** (pergunta 2) — reaproveita checkpoints já salvos
   (basta avaliar `layer2` em todas as épocas salvas de ambas as
   condições), também sem novo treino.

Só depois disso eu rodaria a distillation ponto a ponto (Passo 1 do
documento), porque a essa altura vocês saberiam se há um efeito real
de ordem cronológica em `layer2` que vale a pena tentar preservar com
mais um experimento de treino.

## Observações adicionais sobre a narrativa

- **"Cosseno ~0,95 com a inicialização" como evidência de "sem colapso"**
  é uma barra baixa. Em espaços de alta dimensão, cosseno `0,95` ainda
  permite reorganização substancial no subespaço de baixa
  variância — que é exatamente onde distinções finas de sentido tendem a
  viver. Eu não usaria esse número para descartar reorganização
  semântica relevante; ele só descarta colapso grosseiro/numérico, que
  já era pouco provável de qualquer forma com `lr=3e-5` e 3+2 épocas.

- **Os controles pareados por frequência são o resultado mais importante
  desta atualização**, mais do que o L2-SP. Eles mostram que a APD
  absoluta não separa alvos de controles em nenhuma condição — o que
  significa que todo o sinal do SemEval vem de diferenças de *ranking*
  da ordem de `0,01–0,04` em APD absoluta. Isso é consistente com a
  leitura de "ruído de seleção" da pergunta 1 e deveria pesar mais na
  decisão de continuar ou não do que a história de redistribuição por
  camada.

## Resumo das recomendações

1. Não interpretar "redistribuição entre camadas" como conclusão até ter
   bootstrap por palavra nas condições já rodadas.
2. Corrigir o confound de orçamento de treino entre `theta1` cronológico
   (`D1@2`) e `theta1` pseudo (`pseudo-D1@0,5`) antes de comparar `layer2`
   entre as duas condições.
3. Se prosseguir com distillation: ponto a ponto, teacher=`theta_init`,
   âncoras explicitamente sem ocorrências das 37 palavras-alvo nem dos 37
   controles pareados, `beta=0` no piloto.
4. Tratar `layer2` como objeto de plasticidade, não como candidata a
   régua — mesmo no melhor cenário ela só alcançaria o patamar que
   `layer1` já tem.
5. Avançar em paralelo para a Porta 1 da arquitetura de WSD externa
   (compatibilidade gloss-contexto sem ajuste no SemEval), já que o
   retorno marginal da linha de regularização do MLM parece baixo mesmo
   antes de qualquer novo treino.
