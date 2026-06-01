**O problema que queremos resolver**

Imagine que você tem um dicionário que foi atualizado ao longo de 100 anos. A palavra *gay* em 1920 significava alegre. Em 1980 significava outra coisa. Se você perguntar a um modelo de linguagem moderno "o que *gay* significa em 1920?", ele vai te dar uma representação — um vetor de números — que provavelmente é quase idêntica à representação de *gay* em 1980. O modelo aprendeu a palavra, não a palavra num momento específico.

Traceabilidade temporal é a propriedade que queremos que um modelo tenha: *gay@1920* e *gay@1980* deveriam ocupar lugares diferentes no espaço geométrico das representações. Como se cada palavra tivesse um endereço que muda ao longo do tempo, e você pudesse consultar o endereço antigo diretamente.

O Paper 1 mostrou que modelos treinados com MLM (mascarar uma palavra e tentar adivinhá-la a partir do contexto) não desenvolvem essa propriedade de forma confiável. O objetivo deles é bom para entender contexto local — "qual palavra encaixa nesta sentença?" — mas não para posicionar palavras geometricamente de forma consistente ao longo do tempo. É como contratar alguém para organizar livros por assunto mas sem nunca pedir para eles manter um índice histórico de onde cada livro ficou em cada ano.

---

**A ideia central: dois tipos de representação**

Nossa proposta é que para rastrear uma palavra no tempo você precisa de dois tipos de informação, que são complementares.

O primeiro tipo é **onde a palavra está agora**, no período que você está consultando. Isso é o que os modelos atuais fazem bem. Se você encontra a palavra *vírus* numa sentença que fala de computadores e malware, o modelo sabe que estamos falando de vírus digital. Se a sentença fala de infecção e bactérias, é vírus biológico. O contexto da sentença informa a posição semântica no período. Chamamos essa representação de **h_s(t)** — o vetor que descreve onde o token S está no período t.

O segundo tipo é **como a palavra chegou até aqui** — a trajetória que ela percorreu. Imagine que você está tentando entender por que *broadcast* hoje significa postar nas redes sociais. A resposta exige saber de onde veio: semear sementes, depois transmitir rádio, depois televisão, e finalmente comunicação digital. Nenhuma sentença individual te dá isso. Você precisa de uma representação que olha para o arco inteiro da trajetória da palavra. Chamamos essa representação de **m_s(t)**.

A proposta é que token@tempo seja o par `(h_s(t), m_s(t))`. Um endereço atual mais uma descrição do caminho percorrido. Como se para localizar alguém você precisasse não só do endereço de hoje mas de um resumo de todas as cidades onde ela morou.

---

**Como construímos h_s(t)**

Essa parte já existe no Paper 1 e não muda. Você pega o modelo Token-Time — um transformer padrão onde cada token recebe, além do seu embedding normal, um vetor que codifica em qual período temporal a sentença está. Você treina esse modelo com MLM sobre um corpus fatiado por período. No final, para qualquer sentença no período t, você passa ela pelo modelo e pega o vetor que corresponde à posição da palavra que te interessa. Esse vetor é h_s(t).

Um exemplo concreto: você tem 1000 sentenças com a palavra *gay* dos anos 1920, e 1000 dos anos 1980. Cada sentença é processada com o rótulo do período correspondente. O modelo aprende que em 1920 *gay* aparece com palavras como *cheerful*, *merry*, *laughter*; em 1980 aparece com palavras muito diferentes. Mas como o Paper 1 mostrou, essa separação contextual não garante que os vetores de *gay@1920* e *gay@1980* fiquem em vizinhanças diferentes no espaço geométrico. O modelo sabe usar o contexto para desambiguar, mas não necessariamente posiciona os dois vetores longe um do outro.

---

**Como construímos m_s(t) — o agregador por período**

O primeiro passo para construir a representação de trajetória é resumir como a palavra se comportou em cada período. Para cada período t em que a palavra S aparece, você coleta todos os vetores h_s(t) de todas as ocorrências dessa palavra naquele período, e os resume num único vetor R_s(t).

A versão mais simples é tirar a média: R_s(t) = média de todos os h_s(t) daquele período. Isso é o que o Paper 1 chamou de mean-prototype, e funciona bem quando a palavra tem um sentido dominante no período. Mas falha quando a palavra tem dois sentidos coexistentes. Imagine *gay* em um período de transição — metade das ocorrências vêm do sentido antigo, metade do novo. A média dos dois vetores cai no meio, num ponto que não representa nenhum dos dois sentidos. Como se você tentasse descrever um casal misturando as características dos dois numa média — o resultado não se parece com nenhum deles.

Para lidar com isso, usamos um Set Transformer como agregador. Em vez de simplesmente fazer a média, o Set Transformer processa o conjunto inteiro de ocorrências — todas de uma vez, levando em conta as relações entre elas. Antes de comprimir para um único vetor, ele produz uma versão contextualizada de cada ocorrência, onde cada vetor foi ajustado à luz de todos os outros. Se metade das ocorrências está num cluster e metade está em outro, o Set Transformer tem a capacidade arquitetural de preservar essa estrutura ao comprimir, em vez de apagar com a média.

Imagine que você quer resumir uma turma de 30 alunos. A média das notas é uma informação — mas se há um grupo de alunos excelentes e outro de alunos que lutam, a média esconde isso. Um resumo que preserva "há dois grupos bem distintos" é mais informativo. O Set Transformer tenta fazer isso para os sentidos de uma palavra num período.

Um detalhe técnico importante: o Set Transformer produz, antes do vetor final R_s(t), um conjunto de vetores intermediários — um por ocorrência, que chamamos de u_s^i(t). Esses vetores intermediários são o que usamos para medir se a bimodalidade foi preservada. O vetor final R_s(t) é o resumo comprimido que entra na etapa seguinte.

---

**A sequência de representações e o que não fazer**

Com R_s(t) calculado para cada período em que S aparece, você tem uma sequência ao longo do tempo. Por exemplo, para *gay* no COHA (que cobre décadas de 1810 a 2000), você teria algo como R_gay(1810s), R_gay(1820s), ..., R_gay(2000s).

Dois princípios de construção importam aqui.

O primeiro é que a trajetória só existe onde a palavra existe. Se *internet* não aparece no corpus dos anos 1970, não tentamos extrapolar como ela "teria sido" naquele período. Extrapolar criaria trajetórias fictícias que o modelo depois tentaria modelar como se fossem reais — como inventar a história de alguém antes de ela nascer. A trajetória começa quando a palavra aparece pela primeira vez e termina na última ocorrência.

O segundo é que para décadas intermediárias onde a palavra tem poucas ou nenhuma ocorrência mas está "dentro" do período coberto, usamos interpolação linear entre as décadas vizinhas. Se temos R_s(1890s) e R_s(1910s) mas não R_s(1900s), assumimos que R_s(1900s) é o ponto médio. É o prior mais conservador possível — assume mudança gradual quando não temos evidência do contrário. Mudanças abruptas que coincidem com décadas sem ocorrência vão ser suavizadas, e documentamos isso como limitação.

Uma coisa que consideramos mas descartamos: calcular deltas — δ_s(t) = R_s(t) − R_s(t−1) — e passar os deltas para o encoder temporal em vez das representações diretamente. O problema é que delta pressupõe que subtrair dois vetores produz algo significativo como "deslocamento". Isso é verdade para médias simples (centroides vivem num espaço euclidiano bem comportado), mas não é garantido para saídas do Set Transformer, que podem codificar a estrutura do conjunto de formas não-lineares. Dois vetores de períodos diferentes do Set Transformer podem ser incomaráveis por subtração. Então passamos a sequência inteira de R_s(t) diretamente para o encoder temporal e deixamos ele aprender o que significa mudança.

---

**O encoder temporal e o objetivo de treinamento**

Agora temos a sequência Seq_s = (R_s(t_a), R_s(t_a+1), ..., R_s(t_b)) — a trajetória de S ao longo dos períodos em que aparece. Queremos aprender uma representação m_s(t) que capture o estado dessa trajetória em cada ponto.

Fazemos isso em dois passos, com um teacher e um student — uma estrutura clássica de knowledge distillation.

**O teacher** é treinado primeiro, sobre trajetórias completas, sem máscara. Ele recebe a sequência inteira e produz representações M_s(t) para cada período. O objetivo do teacher é duplo: reconstruir a sequência de entrada (como um autoencoder) e, ao mesmo tempo, ser penalizado se as suas representações forem muito parecidas com a entrada. Essa segunda parte — a regularização anti-identidade — é o que impede o teacher de "trapacear" simplesmente copiando a entrada. Sem ela, um autoencoder com capacidade suficiente aprende identidade (a saída é igual à entrada) e nada de útil foi comprimido. Com a regularização, o teacher é forçado a produzir uma representação que é diferente da sequência bruta mas ainda captura sua estrutura essencial.

Um exemplo simples de regularização anti-identidade: imagine que você pediu para alguém resumir um livro de 500 páginas, mas eles simplesmente te devolveram as 500 páginas. Isso satisfaz "incluir todas as informações" mas não cumpre o objetivo de resumo. A penalização mede o quanto a saída se parece com a entrada e aumenta o custo quando a similaridade é alta, forçando uma compressão real.

Depois de treinar, congelamos o teacher. As representações que ele produz — M_s^teacher(t) — viram os alvos fixos para o próximo passo.

**O student** aprende com a trajetória mascarada. Você pega a sequência Seq_s, esconde um período t_k (substitui por um token especial [MASK]), e pede para o student reconstruir M_s^teacher(t_k) — a representação que o teacher teria produzido para aquele período se visse a sequência completa. O student vê toda a trajetória exceto o período mascarado, e aprende a prever o que aquele período deveria ser.

Como o teacher está congelado, o alvo não muda durante o treinamento do student. Não há risco de colapso trivial onde teacher e student entram em conluio para produzir representações sem sentido — o teacher é uma referência fixa, como um gabarito.

A escolha mais importante: o student é **bidirecional**. Ele vê tanto o passado quanto o futuro da trajetória ao redor do período mascarado. Por quê isso importa? Considere uma palavra que muda de significado abruptamente numa única década — como se *gay* tivesse mudado completamente de 1940 para 1950, sem transição gradual. Se você só vê o passado, até 1940 parece que a trajetória vai continuar igual. Mas se você vê que a partir de 1950 o comportamento é radicalmente diferente, você consegue identificar que houve uma ruptura em 1940–1950 mesmo sem ver aquele período diretamente. A bidirecionalidade é necessária para capturar mudanças abruptas.

---

**O benchmark sintético e as quatro classes**

Para testar tudo isso de forma controlada, usamos um corpus sintético onde sabemos exatamente qual deveria ser a trajetória de cada palavra. O corpus é composto de sentenças sujeito-verbo-objeto muito simples, com dois grupos de verbos e objetos: N1 (verbos V1–V4, objetos O1–O4) e N2 (verbos V5–V8, objetos O5–O8).

A "trajetória semântica" de um sujeito é simplesmente com qual frequência ele aparece em sentenças do tipo N1 vs. N2 ao longo do tempo. Se um sujeito aparece quase sempre com verbos e objetos de N1 no início e quase sempre com N2 no final, sua trajetória é uma migração de N1 para N2.

Há um ruído intencional: 25% das vezes, o verbo ou objeto é sorteado da vizinhança errada. Isso simula o fato de que no mundo real uma palavra com um sentido pode às vezes aparecer em contextos inesperados.

As quatro classes de trajetória são:

**Stable** — a palavra fica no mesmo lugar o tempo todo. P(N1|s,t) é aproximadamente constante ao longo dos 10 períodos. Como *cadeira* — sempre significa o mesmo.

**Drift** — a palavra migra gradualmente de N1 para N2. Começa com quase 100% de ocorrências em N1 e termina com quase 100% em N2, numa rampa suave. Como *broadcast* ao longo de décadas.

**Bifurcating** — a palavra começa em N1 e vai para um estado misto, onde 50% das ocorrências são N1 e 50% são N2. Simula uma palavra que desenvolveu dois sentidos coexistentes, como *gay* num período de transição.

**Abrupt** — a palavra está em N1 até um certo período t_k, e depois muda completamente para N2 num salto único. Como se uma palavra mudasse de significado da noite para o dia, talvez por uma mudança cultural ou tecnológica repentina.

As classes Drift e Abrupt são as que testam a representação de trajetória de formas distintas. Drift é suave e previsível — até uma representação simples deve captá-la. Abrupt é o caso difícil, que só uma arquitetura bidirecional consegue reconstruir adequadamente quando o período da ruptura é mascarado.

---

**Os diagnósticos**

Temos seis formas de avaliar se o modelo está funcionando.

**D1** é o mais simples: treinamos um classificador linear sobre h_s(t) ou m_s(t) para prever se o sujeito estava numa sentença N1 ou N2. Se a representação separar os dois, o classificador tem boa acurácia. Mas aprendemos no Paper 1 que isso é necessário mas não suficiente — um modelo pode codificar o período sem posicionar os vetores nos lugares certos geometricamente.

**D2** é o principal: para cada período e cada sujeito, pega os 10 vetores mais parecidos com h_s(t) no espaço de representações e mede quantos são de sentença N1. Se a proporção acompanha P(N1|s,t) ao longo do tempo — começa alta para sujeitos Drift e cai até o final — então a representação está geometricamente correta. O resultado esperado é que m_s(t) tenha um D2 melhor do que h_s(t), especialmente em condições difíceis.

**D3** faz o seguinte: pega a mesma sentença, muda só o rótulo de período, e verifica se a representação muda de vizinhança. Modelos que ignoram o período dão taxa zero — sempre o mesmo resultado independente do período injetado.

**D4** testa generalização: treinamos o modelo nos períodos t0–t7, e avaliamos em t8–t9 que o modelo nunca viu. Um modelo que internalizou a *direção* da trajetória deve conseguir generalizar para períodos novos.

**D5a** é o diagnóstico do objetivo de treinamento do student. Você pega a trajetória completa de um sujeito, mascara um período, e verifica se o student consegue reconstruir o que o teacher produziria. Em sujeitos Abrupt, a ruptura abrupta em t_k deve ser reconstruível se o student bidirecional viu t_{k+1} e percebeu que a trajetória mudou completamente. Um student causal ou linear não conseguiria — ele só viu o passado e projetaria continuação.

**D6** testa a bimodalidade em sujeitos Bifurcating. Para sujeitos que têm dois sentidos coexistentes nos períodos tardios, os vetores intermediários do Set Transformer deveriam formar dois grupos distintos — um para as ocorrências N1 e outro para N2. Medimos isso com silhouette: se as ocorrências N1 estão mais perto entre si do que das ocorrências N2, e vice-versa, o silhouette é alto. Se tudo está misturado num centroide, o silhouette é baixo. A média simples falha porque produz um único vetor médio; o Set Transformer tem a capacidade de manter os dois grupos separados.

---

**As verificações de sanidade — as portas que precisam abrir**

Antes de qualquer experimento completo, há três verificações sequenciais que funcionam como portas de entrada. Se uma falhar, paramos para diagnosticar antes de avançar.

**Verificação 0 — o teacher é semanticamente informativo?** Treinamos o teacher com a regularização anti-identidade e verificamos duas coisas: as representações M_s(t) que ele produz são suficientemente diferentes das representações de entrada R_s(t)? E um probe linear sobre M_s(t) consegue prever P(N1|s,t) melhor do que um probe direto sobre R_s(t)? Se a segunda resposta for "não", o teacher não extraiu nada de útil da trajetória e o Passo 2 não tem base.

**Verificação 1 — as representações carregam sinal temporal?** A versão mais simples do sistema — encoder linear sem memória, agregador de média — consegue melhorar D2 sobre h_s(t) sozinho para sujeitos Drift? Se essa versão mínima já não captura a deriva gradual, nada de mais sofisticado vai ajudar. Também verificamos o complemento: para sujeitos Stable, a representação de trajetória deve ser estatisticamente indistinguível de h_s(t). Se o encoder está "inventando" trajetória onde não existe, está injetando ruído.

**Verificação 2 — o Set Transformer preserva bimodalidade?** Para sujeitos Bifurcating, os embeddings intermediários do Set Transformer nos períodos tardios formam dois grupos distintos (silhouette alto)? Se não, o argumento de que o centroide é o gargalo não se sustenta, e podemos usar mean pooling para tudo.

Somente após essas três verificações passarem avançamos para as ablações completas e para o corpus natural.

---

**O corpus natural e a avaliação extrínseca**

O COHA — Corpus of Historical American English — cobre décadas de 1810 a 2000 e é o mesmo corpus usado por Hamilton et al. nos experimentos seminais de LSCD. Usamos as palavras-alvo do SemEval-2020 que tenham ocorrências em pelo menos 3 décadas distintas como conjunto de avaliação.

Para cada palavra-alvo, construímos a sequência de representações por década, interpolando onde necessário e sem extrapolar além das datas de cobertura. Rodamos os diagnósticos D2, D3, D4 e D6 sobre h_s, m_s e o par completo.

A avaliação extrínseca no SemEval-2020 pergunta: dado um conjunto de palavras com mudança semântica anotada por humanos, o modelo consegue rankear as palavras da que mudou mais para a que mudou menos? Usamos a norma do deslocamento em m_s como medida de mudança e comparamos com Hamilton et al. como baseline.

Incluímos também uma análise qualitativa de trajetórias para palavras com histórico semântico conhecido — *gay*, *broadcast*, *mouse* — plotando as posições de h_s(t) e m_s(t) ao longo das décadas num espaço bidimensional (via PCA). Isso serve como salvaguarda: se a métrica de ranking for insensível a um ganho real, a visualização pode mostrar que as trajetórias de m_s são mais coerentes e suaves do que as de h_s.

---

**Uma nuance filosófica sobre m_s(t)**

Vale mencionar algo que pode parecer estranho à primeira leitura: porque o student é bidirecional, m_s@1900 para uma palavra específica depende do que o modelo sabe sobre os períodos posteriores. Se você treinar com dados até 1950, m_s@1900 é diferente do que se você treinar com dados até 2000 — porque o contexto futuro disponível muda.

Isso é incomum para um historiador: normalmente o "estado de algo em 1900" não deveria depender do que aconteceu em 1950. Mas m_s(t) não é uma representação causal — não está prevendo o futuro. É uma descrição retrospectiva de uma trajetória já observada. Como um biógrafo que escreve sobre a infância de alguém sabendo o que ela se tornaria — a descrição da infância é influenciada pelo conhecimento do arco completo, e isso é uma característica do método, não um defeito.

---

**Como tudo se encaixa numa linha**

O Paper 1 mostrou que o MLM cria representações que respondem ao período mas não posicionam os vetores geometricamente de forma consistente ao longo do tempo. O problema está no objetivo, não na arquitetura.

O Paper 2 propõe que token@tempo = (h_s(t), m_s(t)). h_s(t) vem do mesmo modelo do Paper 1 — boa desambiguação contextual, posição semântica no período. m_s(t) é novo — estado da trajetória, aprendido via masked trajectory distillation sobre sequências de representações por período.

O Set Transformer agrega as ocorrências por período sem perder bimodalidade. A sequência de representações é passada diretamente para o encoder temporal — sem calcular deltas, porque subtrair representações de conjunto não tem garantia semântica. O teacher self-supervised aprende a estrutura da trajetória sem rótulos externos. O student bidirecional reconstrói períodos mascarados usando contexto passado e futuro, o que é necessário para capturar rupturas abruptas.

As verificações de sanidade garantem que cada peça do pipeline funciona antes de avançar para a próxima. O cronograma é construído em torno dessas verificações como portas sequenciais: sem certeza de que o teacher é informativo, não há razão para treinar o student; sem certeza de que o student captura Abrupt melhor que linear, não há razão para ir para o COHA.

Se as verificações passarem e os resultados no COHA e SemEval mostrarem que token@time melhora traceabilidade sobre h_s(t) sozinho, o paper demonstra empiricamente a tese: representações explícitas de trajetória melhoram traceabilidade temporal além de snapshots.