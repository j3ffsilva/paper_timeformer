# Handoff v2 — próxima máquina (GPU)

**Data:** 2026-06-12
**Estado:** repositório limpo, pronto para rodar. Esta máquina (local) NÃO
tem GPU/MPS -- tudo que precisa de treino real do Timeformer fica para a
próxima máquina.

---

## 1. Onde paramos

Desde o handoff anterior (`docs/next_machine_handoff.md`, 2026-06-05) e o
adendo de 2026-06-06 em `docs/relational_change_current_plan.md`, fizemos
uma rodada inteira de diagnóstico **sem treinar nada novo** -- só medições
em cima do checkpoint já existente
(`outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/`, `theta0`/`theta1`,
d_model=128, 3 camadas, 4 heads, treinado do zero só com MLM contínuo).

Resumo do que descobrimos (detalhes completos em
`docs/perfil_relacional_v2_resultados_fase1.md` §7.9-7.25 e no resumo em
`docs/relational_change_current_plan.md`, seção "Adendos 2026-06-12"):

1. Boa parte da "separação por período" que víamos era **drift de
   checkpoint** (theta0 -> theta1), não sinal semântico real.
2. Corrigindo isso (encoder fixo, agrupar ocorrências antes do perfil), o
   sinal melhora um pouco (spearman APD ~0.13 -> ~0.20 contra o gold do
   SemEval, n=37), mas continua sem significância estatística.
3. **Teste decisivo**: rodamos a mesma métrica (APD entre ocorrências de
   t0 e t1) usando `bert-base-uncased` PRÉ-TREINADO e CONGELADO, nas mesmas
   frases. Resultado: spearman ~0.59 (p=0.0001) -- estatisticamente
   significativo e acima do teto da literatura para esta tarefa (~0.4-0.55).

**Conclusão**: o gargalo não é o desenho do perfil relacional, a
centralização ou a configuração diagonal (essas correções ajudam um pouco
e devem ser mantidas). O gargalo é a **qualidade/capacidade do encoder**:
um Transformer de 128 dimensões e 3 camadas, treinado do zero só com MLM
nos ~780k janelas do SemEval, não tem "conhecimento de mundo" suficiente
para separar sentidos de palavras (ex.: `plane_nn` = geometria vs avião).

---

## 2. O que fazer nesta máquina (GPU)

**Objetivo**: inicializar o Timeformer a partir de um encoder
pré-treinado pequeno, em vez de treinar do zero, e então continuar o
treino contínuo temporal normalmente (mesmo pipeline de sempre:
`theta_0 = treino(D_0)`, `theta_1 = continua_treino(theta_0, D_1)`).

### 2.1 Por que `bert-tiny` (e não `bert-base`)

A arquitetura atual (`RealStaticMLM` em `src/timeformers/real_models.py`)
é uma implementação própria (vocabulário próprio de ~27k tokens lematizados
do SemEval, embeddings próprios, `nn.TransformerEncoder` do PyTorch). Não
dá para simplesmente carregar os pesos do `bert-base-uncased` (768 dim, 12
camadas, vocabulário WordPiece de 30522) -- as dimensões não combinam.

Existe, porém, um modelo pré-treinado pequeno e compatível em escala:
**`prajjwal1/bert-tiny`** (Hugging Face) -- `d_model=128`, 2 camadas, 2
heads, `d_ff=512`. Isso é quase a mesma escala do nosso
`d_model=128, layers=3, heads=4, d_ff=384`.

Plano recomendado:

1. Ajustar a configuração do `RealStaticMLM` para `layers=2, heads=2,
   d_ff=512` (igual ao `bert-tiny`), mantendo `d_model=128`.
2. Carregar os pesos das 2 camadas do `TransformerEncoder` do `bert-tiny`
   diretamente no `nn.TransformerEncoder` do `RealStaticMLM` (mapeamento
   de nomes: `attention.self.{query,key,value}` -> `self_attn.{in_proj...}`,
   `intermediate`/`output` -> `linear1`/`linear2`, etc. -- checar
   `transformers.models.bert.modeling_bert.BertLayer` para o mapeamento
   exato).
3. **Embeddings de token**: o vocabulário é diferente (nosso vocab é
   lematizado e específico do SemEval, ~27k tokens; o do `bert-tiny` é
   WordPiece, 30522 tokens). Estratégia:
   - Para cada token do nosso vocabulário, tokenizar com o tokenizer do
     `bert-tiny` (pode gerar múltiplos sub-tokens) e inicializar nosso
     `token_emb.weight[i]` como a MÉDIA dos embeddings WordPiece
     correspondentes (do `bert-tiny`, que já tem `d_model=128` -- sem
     necessidade de projeção).
   - Tokens sem correspondência boa (raros, símbolos): manter
     inicialização aleatória padrão.
4. `pos_emb` e `mlm_head`: inicialização aleatória (não há correspondência
   direta de tamanho/posições -- `bert-tiny` usa `seq_len=512`, nós usamos
   `seq_len=32`; o MLM head depende do vocabulário, que é diferente).

Esse plano dá ao modelo conhecimento sintático/semântico geral do
`bert-tiny` nas camadas do encoder e nos embeddings de token, sem violar a
arquitetura/vocabulário já usados em todo o pipeline de medição (Tarefas
1-3, perfil relacional v2).

### 2.2 Onde implementar isso

Sugestão de novo script (a ser criado): `scripts/init_pretrained_encoder.py`
ou um novo flag `--init-from-pretrained bert-tiny` em
`scripts/run_diachronic_relational_experiment.py`, que:

1. Constrói o `RealStaticMLM` normalmente (com `layers=2, heads=2,
   d_ff=512`).
2. Carrega `prajjwal1/bert-tiny` via `transformers.AutoModel`.
3. Aplica o mapeamento de pesos do passo 2.1 acima.
4. Salva o `state_dict` resultante como um checkpoint "init" (ex.:
   `outputs/<run>/init_pretrained.pt`).
5. O `ContinualPeriodTrainer` (já existe, `src/timeformers/train.py`)
   carrega esse `state_dict` como ponto de partida em vez de pesos
   aleatórios, e prossegue normalmente com `theta_0 = treino(D_0)`,
   `theta_1 = continua_treino(theta_0, D_1)`.

Isso pode ser feito ANTES de subir para a máquina GPU (a parte de
carregar/mapear pesos do `bert-tiny` não precisa de GPU -- é só
manipulação de tensores pequenos). Recomendo implementar e testar essa
parte localmente (baixar `bert-tiny` é rápido, ~17MB) e só then subir o
checkpoint `init_pretrained.pt` + o comando de treino para a GPU.

### 2.3 Comando de treino (adaptar de `next_machine_handoff.md`)

```bash
python scripts/run_diachronic_relational_experiment.py \
    --input-dir data/processed/semeval2020_task1/eng_lemma/corpus \
    --targets data/processed/semeval2020_task1/eng_lemma/targets.txt \
    --profile-mode pmi \
    --probe-mode occurrence \
    --d-model 128 --layers 2 --heads 2 --d-ff 512 \
    --init-from-pretrained outputs/<run>/init_pretrained.pt \
    --base-epochs 12 --epochs-per-period 8 \
    --batch-size 192 --lr 0.0001 \
    --device cuda \
    --output-dir outputs/semeval2020_pmi_pretrained_init_d128
```

(O flag `--init-from-pretrained` ainda não existe -- precisa ser adicionado
ao script, junto com o suporte em `ContinualPeriodTrainer.__init__` ou
antes da chamada de treino, para `model.load_state_dict(torch.load(path),
strict=False)`.)

Os hiperparâmetros de treino (`base-epochs=12`, `epochs-per-period=8`,
`batch-size=192`, `lr=0.0001`) são os mesmos do run atual
(`outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/config.json`) -- ponto de
partida razoável, mas como o modelo agora começa de pesos pré-treinados
(não aleatórios), pode valer testar `lr` menor (ex. `3e-5` a `5e-5`,
comum em fine-tuning de BERT) e/ou menos épocas no `base-epochs`.

### 2.4 Depois do treino: reaplicar as medições

Depois de gerar os novos checkpoints (`theta0`/`theta1` do novo run), trazer
de volta para esta máquina (sem GPU) e rodar, sobre o novo `output-dir`:

1. `scripts/evaluate_fixed_encoder_v2.py` (Tarefa 1 -- encoder fixo,
   APD/Delta/NMI)
2. `scripts/prototype_modes_first_v2.py` (Tarefa 3 -- modos primeiro)
3. (Opcional, para comparação) `scripts/evaluate_pretrained_oracle_v2.py`
   já está feito e não precisa rodar de novo -- ele é o "teto" fixo
   (~0.59), serve de referência constante.

Comparar os novos números de Tarefa 1 (`fixed_encoder_metrics.json`) com
os atuais (`outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/fixed_encoder_eval/`,
spearman APD ~0.20 com `theta1` fixo). Se o spearman subir
significativamente em direção a ~0.4-0.59, a hipótese "gargalo = encoder"
está confirmada e essa passa a ser a configuração de referência do projeto.

---

## 3. O que NÃO fazer

- Não tunar hiperparâmetros (`gamma`, `tau`, `n_min`, `top_n`, k de
  clustering, etc.) contra `truth.tsv` -- n=37, overfitting garantido. Ver
  `docs/novo_perfil_relacional.md` e `docs/perfil_relacional_v2_resultados_fase1.md`
  para os valores já fixados.
- Não abandonar a configuração de "encoder fixo" (Tarefa 1) -- ela deve ser
  aplicada também ao novo encoder pré-treinado-inicializado.
- Não re-treinar do zero com `d_model` maior (256/512) -- essa ablação foi
  substituída pela inicialização pré-treinada (ver §7.24 do
  `docs/perfil_relacional_v2_resultados_fase1.md`).

---

## 4. Arquivos de referência

```
src/timeformers/real_models.py            # RealStaticMLM (arquitetura atual)
src/timeformers/train.py                  # ContinualPeriodTrainer
scripts/run_diachronic_relational_experiment.py   # pipeline de treino principal
scripts/evaluate_fixed_encoder_v2.py      # Tarefa 1 (encoder fixo)
scripts/prototype_modes_first_v2.py       # Tarefa 3 (modos primeiro)
scripts/evaluate_pretrained_oracle_v2.py  # Tarefa 2 (teto BERT, já rodado)
docs/perfil_relacional_v2_resultados_fase1.md     # resultados completos §7.9-7.25
docs/relational_change_current_plan.md            # log de decisões / conclusão
docs/novo_perfil_relacional.md                    # formalização do perfil relacional v2
outputs/semeval2020_pmi_dynamic_mlm_12_8_d128/    # checkpoint atual (theta0/theta1)
```
