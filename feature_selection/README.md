## Feature Engineering e Feature Selection — Predição de Carga Alta (EVCOMX)

Segue a aula de Feature Selection do bootcamp e o Guia de FE/FS da EVCOMX.

### Como rodar

1. `EDA/transforma_dataset.py`: gera o `EDA/base.csv` limpo.
2. `feature_selection/feature_selection.py` (células `#%%`) ou `feature_selection_carga_alta.ipynb` (mesmo conteúdo, já executado).

O código acha a raiz do projeto sozinho; pode rodar de dentro de `feature_selection/` ou da raiz.

### Arquivos gerados

| Arquivo | O que é |
|---|---|
| `dataset_modelagem_fs.csv` | **Conjunto final (9 features)** + `carga_alta` + metadados (`cd_corrida`, `nu_carregamento`, `conjunto`, `inicio`) |
| `dataset_features_completo.csv` | Todas as 80 candidatas, para a seleção de algoritmos testar outros conjuntos |
| `votos_feature_selection.csv` | Voto de cada método, estabilidade, vitórias no Boruta e correlação |
| `graficos/` | Figuras para a apresentação |

- **`conjunto`** traz o split 70/15/15 temporal (treino / validacao / teste), definido pelo professor. **Todo o grupo deve usar essa coluna.**
- **`inicio` não é feature.** É a data do cestão, guardada só como metadado.

### Decisões

- **Unidade de análise:** o cestão (carregamento 1 ou 2). É a pergunta operacional, o alvo já vem por cestão, dá ~3.500 exemplos e permite usar o cestão 1 como contexto do 2.
- **Split:** 70/15/15 em ordem de tempo, por corrida. A seleção roda só no treino. A validação escolhe quantas features ficam e o limiar de decisão. O teste é usado uma única vez.
- **Métrica de seleção:** PR-AUC, porque só ~16% dos cestões têm carga alta. No teste também reportamos acurácia, recall, precisão, F1 e ROC-AUC.
- **Multicolinearidade:**
  - saem os pares com |r| > 0,90, ficando a feature mais correlacionada com a carga alta;
  - depois, VIF < 10 aplicado em ordem de relevância. Remover sempre "a de maior VIF" jogava fora justamente as features mais preditivas.
- **Seleção:** seis métodos votam:
  - filtro (Mann-Whitney com Bonferroni ou informação mútua);
  - RFE;
  - L1;
  - Elastic Net;
  - permutação;
  - Boruta.
- **Quantas features:** as features são adicionadas uma a uma, por ordem de votos, e o desempenho é medido na validação. O "cotovelo" ficou em **9 features**.

### Conjunto final (9 features) e intuição física

| Feature | Sinal | Leitura |
|---|---|---|
| `prop_peso_pesado` | − | mais material pesado → cestão compacto → menos carga alta |
| `densidade_media` | − | cestão "fofo" (pouca tonelada por m³) → mais carga alta |
| `peso_por_ciclo_t` | + | muitas toneladas por ciclo de montagem = camadas grossas, pouco intercaladas → mais carga alta |
| `taxa_ca_ult5`, `taxa_ca_ult10` | + | carga alta nas últimas 5/10 corridas → o forno "está ruim" e continua ruim |
| `max_bloco_leve_m3` | + | um bloco grande de leve sem material pesado/misto intercalado → mais carga alta |
| `prop_leve_base` | + | muito leve nos 30% de baixo do cestão → mais carga alta |
| `prop_PACB`, `prop_PAEP` | + | pacotes de chaparia/estamparia → mais carga alta |

### Respostas às perguntas do Guia (§1.3)

1. **Quais features descrevem um cestão propenso a carga alta?** Pouco material pesado, baixa densidade, ciclos de montagem "grossos", blocos grandes de leve e carga alta recente no forno. As mais importantes na validação, medidas por permutação: `prop_peso_pesado`, `taxa_ca_ult5`, `peso_por_ciclo_t` e `densidade_media`.
2. **O volume estimado é a feature mais importante?** **Não.** O volume total quase não se relaciona com a carga alta (r = −0,04). Sozinho, ele dá PR-AUC de 0,17 na validação, igual a um chute. O sinal está na densidade e na proporção de pesado: os cestões são enchidos até volumes parecidos, e o que muda é quanto peso cabe nesse volume. A física do Guia se confirma pela densidade, não pelo volume total.
3. **Ordem e ciclos agregam além de peso e volume?** Um pouco. Sem `peso_por_ciclo_t` e `max_bloco_leve_m3`, a PR-AUC na validação cai de 0,43 para 0,41 (Random Forest). As features de ciclo são muito correlacionadas com a composição, então boa parte da informação delas já está em `prop_peso_pesado`.
4. **Qual método de seleção gera o melhor desempenho fora da amostra?**
   - Na validação, com Random Forest: cotovelo 0,43; filtro 0,43; RFE e Boruta 0,41; L1/Elastic Net 0,40.
   - As diferenças são pequenas. A votação entre métodos foi mais útil que qualquer método sozinho.
5. **Quantas features bastam?** **9 de 80.** No teste, o conjunto de 9 rende o mesmo que as 80:
   - regressão logística: PR-AUC 0,40 com 9 contra 0,41 com 80;
   - Random Forest: 0,34 contra 0,35.
   Um modelo com 9 features é mais simples de explicar e de manter.

### Desempenho no teste (jun–ago/2026)

| Conjunto | Algoritmo | Acurácia | Recall | Precisão | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| todas (80) | Reg. logística | 0,77 | 0,69 | 0,35 | 0,46 | 0,77 | 0,41 |
| **final (9)** | **Reg. logística** | 0,68 | **0,75** | 0,28 | 0,40 | 0,77 | 0,40 |
| final (9) | Random Forest | 0,81 | 0,39 | 0,36 | 0,37 | 0,76 | 0,34 |
| básico do Guia (3) | Random Forest | 0,48 | 0,71 | 0,18 | 0,28 | 0,62 | 0,22 |

Referências de um chute: PR-AUC = 0,15; quem sempre diz "normal" acerta 85%.

O limiar foi escolhido na validação. A escolha final entre recall e precisão fica para a análise de modelos, e depende do custo de um alarme falso versus o de uma carga alta não detectada (pergunta para a EVCOMX).

### Limitações e perguntas para a EVCOMX

- **Timing do histórico (crítico).** `taxa_ca_ult5/10` usa o resultado das corridas anteriores. Isso vale **se**, na hora de prever, já se sabe se a corrida anterior teve carga alta. A dependência é grande: sem o histórico, a PR-AUC na validação cai de 0,43 para 0,32 (Random Forest) e de 0,26 para 0,18 (regressão logística, nível de chute). **Confirmar com a EVCOMX.** Se não valer, use `dataset_features_completo.csv` sem as colunas `taxa_ca_*`.
- **`carga_alta_c1` como feature do cestão 2** melhora a validação (0,43 → 0,45), mas também depende do timing. Ficou fora do conjunto final.
- **Mudança de regime em maio/2026.** Desde maio há menos material pesado, menos ciclos por cestão e densidade menor. A taxa de carga alta continua igual, mas a relação "mais pesado → menos carga alta" enfraquece. O treino termina em 25/04, então validação e teste estão no regime novo. Perguntar o que mudou: regra de montagem, fornecedor, forno ou critério de registro.
- **Escolha da regressão logística.** Ela rende melhor com mais features (pico com ~23 na curva). Se a seleção de algoritmos escolher regressão logística, vale testar o consenso de ≥ 4 votos (22 features), disponível no `dataset_features_completo.csv`.
- **Variação entre versões.** Os números podem mudar um pouco com outra versão do scikit-learn.
