## Seleção de Algoritmos e Modelagem — Predição de Carga Alta (EVCOMX)

Segue a **Aula 05 do bootcamp** (Seleção de Algoritmos e Modelagem) e o **Guia de Modelagem Preditiva da EVCOMX** (§4), e continua a pasta `feature_selection/`.

> **Rodada de out/2026 (corte de 2.000 kg).** A EDA passou a remover as camadas abaixo de 2.000 kg (resposta da EVCOMX) e todas as etapas foram refeitas. A feature selection agora entrega **6 features** (antes, 9). O plano A continua o mesmo tipo de modelo (Random Forest de fábrica); o plano B passou a ser a Random Forest **ajustada** (`max_depth=4`), que agora ganha da de fábrica na validação. Os números abaixo são desta rodada.

### Como rodar

1. `EDA/transforma_dataset.py`: gera o `EDA/base.csv` limpo.
2. `feature_selection/feature_selection.py`: gera os CSVs de features.
3. `selecao_algoritmos/selecao_algoritmos.py` (células `#%%`) ou `selecao_algoritmos.ipynb` (mesmo conteúdo, já executado). Leva ~4 min por causa da busca de hiperparâmetros.

O código acha a raiz do projeto sozinho. Não recalcula nenhuma feature: lê `dataset_features_completo.csv`, `dataset_modelagem_fs.csv` e `votos_feature_selection.csv` da feature selection.

### Arquivos gerados

| Arquivo | O que é |
|---|---|
| `comparacao_algoritmos.csv` | Todos os modelos (os três baselines, de fábrica e ajustados): PR-AUC na CV temporal, na validação e IC 90% |
| `desbalanceamento.csv` | Pesos de classe × sem peso × subamostragem (seção 7b) |
| `limiares_validacao.csv` | Os três limiares de decisão, planos A e B, medidos na validação |
| `resultado_teste.csv` | A conferência única no teste, com a maioria e a regra de volume |
| `previsoes_validacao_teste.csv` | Chance de carga alta prevista para cada cestão da validação e do teste |
| `modelo_final_planoA.joblib` / `modelo_final_planoB.joblib` | Modelos para uso, re-treinados em treino + validação. Cada um guarda junto a lista de features (`features_`), os limiares sem arredondar (`limiares_`) e a versão do scikit-learn (`versao_sklearn_`) |
| `modelo_final.json` | Features, hiperparâmetros e limiares (sem arredondar) de cada plano |
| `graficos/` | 8 figuras para a apresentação |

### Passo a passo (slide 36 da aula)

1. **Tipo de pergunta: classificação.** "Este cestão vai dar carga alta?" é sim ou não. A regressão linear fica de fora.
2. **Métrica: PR-AUC** para comparar algoritmos. A acurácia engana: quem sempre diz "normal" acerta 82–85%. Recall e precisão vêm depois, com o limiar.
3. **Split e baselines.** O split 70/15/15 temporal vem da coluna `conjunto`. São três réguas (Guia de Modelagem §4.2):
   - **trivial:** `DummyClassifier`, maioria e sorteio, com PR-AUC de ~0,18 na validação;
   - **física, regra de volume:** alarme se o volume estimado passa de **56,4 m³**, limiar escolhido no treino (melhor F2, no máximo metade em alarme). É o que a operação faria hoje sem ML;
   - **estatística:** a regressão logística com as 6 features finais.
4. **Pipeline.** Mediana para vazios, padronização só na logística e o modelo. Quatro algoritmos: logística, árvore, Random Forest e Gradient Boosting (a trajetória recomendada pelo Guia, §4.3).
5. **Validação cruzada temporal.** Cinco provas em janela crescente dentro do treino, por corrida. O `cv=5` da aula sorteia linhas e vazaria o futuro.
6. **Hiperparâmetros.** `GridSearchCV` nas mesmas provas temporais, otimizando a PR-AUC (`average_precision`), como pede o Guia (§4.5). Todos os experimentos ficam em `comparacao_algoritmos.csv`.
6b. **Desbalanceamento** (Guia §4.4): pesos de classe, testados contra "sem peso" e subamostragem (seção 7b, abaixo).
7. **Escada (slide 33).** Fica o modelo mais simples a até 0,02 da melhor PR-AUC na validação.
8. **Limiar.** Escolhido na validação, em três cenários.
9. **Teste uma única vez.**

**Sobre o split do Guia (§4.1).** O Guia sugere `StratifiedGroupKFold` por corrida para comparar modelos. Usamos uma CV **agrupada por corrida e temporal**, que é mais rígida: além de nunca separar os cestões de uma corrida, nunca treina com o futuro. A verificação (seção 4) mostrou que as CVs embaralhadas, inclusive a agrupada, são as mais otimistas. O teste final temporal, nunca usado para escolher, também segue o Guia.

Conjuntos de features testados:
- **final (6)**, da feature selection;
- **consenso (18)**, as features com ≥ 4 votos;
- **final sem histórico (4)**, sem `taxa_ca_ult5/10`. É o plano B, caso o resultado das corridas anteriores não esteja disponível na hora de prever.

### Resultado

**Plano A (com histórico): Random Forest de fábrica com as 6 features finais.** A configuração é `n_estimators=300`, `min_samples_leaf=5` e `class_weight="balanced_subsample"`.

| Modelo (validação, abr–jun/2026) | PR-AUC | IC 90% |
|---|---|---|
| **Random Forest, fábrica, final (6)** | **0,427** | 0,35–0,52 |
| Random Forest, fábrica, consenso (18) | 0,428 | 0,35–0,52 |
| Gradient Boosting, ajustado, consenso (18) | 0,411 | 0,33–0,50 |
| Gradient Boosting, fábrica, final (6) | 0,395 | 0,32–0,49 |
| Árvore de decisão, ajustada, consenso (18) | 0,334 | 0,27–0,41 |
| Regressão logística, fábrica, consenso (18) | 0,302 | 0,24–0,38 |
| Regressão logística, final (6) | 0,286 | 0,23–0,36 |
| Baseline (maioria) | 0,179 | — |
| **Baseline físico: regra de volume** | **0,147** | — |

O Gradient Boosting fica a 0,02 da Random Forest. Pela regra da escada, fica a Random Forest: é um degrau abaixo e precisa de menos ajuste. Entre os dois conjuntos de mesma nota (0,427 e 0,428), fica o de 6 features.

**Plano B (sem histórico): Random Forest ajustada (`max_depth=4`) com 4 features.** PR-AUC na validação de 0,316, contra 0,179 do chute. A de fábrica ficou em 0,305; na rodada anterior (7 features) a de fábrica tinha vencido, com 0,318.

**A regra de volume não serve de régua: fica no nível do chute.** PR-AUC 0,159 na CV, 0,147 na validação (ROC-AUC 0,43, ou seja, ligeiramente invertida) e 0,149 no teste. Com o limiar do treino (56,4 m³; antes, 60,4 m³, porque os cestões perderam as camadas pequenas), ela dispara ~53 alarmes a cada 100 cestões e acerta 14–15% deles. O Guia avisa: "se o modelo final não superar o baseline de regra de volume por uma margem relevante, o argumento de ML fica fraco". O modelo supera com folga (0,43 contra 0,15 na validação). A régua que de fato aperta é outra, a **regra do histórico**, medida na análise de modelos.

### Desbalanceamento (seção 7b, Guia §4.4)

Random Forest de fábrica, 6 features, mesmas provas temporais:

| Estratégia | CV PR-AUC | Validação PR-AUC | Chance média na validação (taxa real 0,18) |
|---|---|---|---|
| sem peso | 0,414 | 0,410 | 0,25 |
| **pesos de classe** (`balanced_subsample`, a usada) | 0,413 | 0,427 | 0,41 |
| subamostragem 1:1, só no treino de cada prova | 0,419 | 0,411 | 0,58 |

- **Na ordem dos cestões, as três estratégias empatam** (diferenças de até 0,02, dentro do ruído; a PR-AUC não depende do limiar, e o limiar é ajustado depois). A subamostragem joga fora ~70% dos normais sem ganhar nada na validação.
- **O efeito dos pesos é na escala da chance:** quanto mais peso na classe rara, mais inflada a chance. É por isso que a análise calibra as chances antes de mostrá-las ao pátio.
- **Não usamos SMOTE:** ele cria cestões sintéticos interpolando outros, que podem ser receitas fisicamente impossíveis (Guia §4.4).
- A comparação **não muda a escolha** da seção 8: ela só documenta a estratégia.

### Descoberta principal: quem vai bem antes de maio não vai bem depois (gráfico 03)

- **A regressão logística é a melhor na CV temporal e a pior na validação.** Na CV (ago/2025–abr/2026) ela tem PR-AUC 0,45; na validação (abr–jun/2026), 0,29.
- **Ajustar hiperparâmetros na CV quase nunca ajudou a validação.** Na Random Forest com 6 features, a PR-AUC caiu de 0,427 (fábrica) para 0,377 (ajustada). A exceção é o plano B, em que a versão ajustada, mais rasa, ganhou por pouco (0,316 × 0,305).

As duas coisas têm a mesma causa: a **mudança de regime de maio/2026**. O treino é todo do regime antigo, então a CV dentro dele premia o que funciona naquele regime. Por isso:
- a validação decide entre os modelos;
- a versão de fábrica de cada algoritmo continua candidata.

A pergunta 1 do e-mail à EVCOMX é justamente sobre isso.

> **Transparência:** na primeira rodada, só os modelos ajustados eram candidatos. O escolhido foi a Random Forest ajustada com 22 features (validação 0,400; teste 0,369). Ao ver que o ajuste piorava a validação, a versão de fábrica entrou como candidata. Essa mudança foi motivada pela validação, mas o teste daquela rodada já tinha sido calculado. Fica registrado.

### Limiar de decisão (pergunta 3 do e-mail)

O limiar foi escolhido na validação, com no máximo 50% dos cestões em alarme. Esse teto é uma suposição nossa.

| Se a EVCOMX disser que… | Cenário | Limiar | Teste: recall | Teste: precisão | Alarmes / 100 cestões |
|---|---|---|---|---|---|
| carga alta não detectada é pior | não deixar passar (F2) | 0,41 | **76%** | 29% | 38 |
| os dois erros pesam igual | equilíbrio (F1) | 0,54 | 63% | 34% | 27 |
| alarme falso é pior | evitar alarme falso (F0,5) | 0,77 | 33% | 45% | 11 |

No teste há ~15 cargas altas a cada 100 cestões.

Como ler a primeira linha: a cada 100 cestões, o modelo dispara 38 alarmes. 11 deles são carga alta de verdade, e ele deixa passar 3 ou 4 das ~15.

### Conferência no teste (jun–ago/2026, uma única vez)

| | PR-AUC | ROC-AUC |
|---|---|---|
| **Plano A: Random Forest, 6 features** | **0,391** (IC 90%: 0,32–0,48) | 0,769 |
| Plano B: Random Forest ajustada, 4 features sem histórico | 0,228 | 0,611 |
| Baseline (maioria) | 0,146 | 0,500 |
| Baseline físico: regra de volume (> 56,4 m³) | 0,149 | 0,521 |

Na rodada anterior (corte só no peso zero, 9 features) o plano A tinha 0,354 no teste (IC 90%: 0,29–0,44). A melhora cabe no ruído: os dois intervalos se sobrepõem quase inteiros.

**Sobre a comparação com a feature selection:** no README da feature selection, a regressão logística com 6 features teve PR-AUC 0,42 no teste.
- **Isso não muda a escolha.** O teste não serve para escolher (slide 35), e 0,42 está dentro do intervalo do modelo escolhido.
- **Mas é mais um sinal de instabilidade.** A validação (abr–jun) e o teste (jun–ago) não concordam sobre o melhor algoritmo. Com ~75–90 cargas altas em cada conjunto, as diferenças entre os melhores modelos estão dentro do ruído.

### Explicação do modelo (gráficos 07 e 08)

- **Permutação na validação.** As features mais importantes são `taxa_ca_ult5`, `prop_peso_pesado`, `peso_por_ciclo_t`, `taxa_ca_ult10` e `densidade_media`, as mesmas da feature selection; `prop_PACB` contribui pouco.
- **Fluxograma para a apresentação.** O gráfico 08 traz uma árvore de 3 perguntas: densidade, proporção de pesado e carga alta recente. Ela é mais fraca que o modelo (PR-AUC 0,25 contra 0,43), mas mostra a lógica de um jeito que o pátio entende.

### O que depende das respostas da EVCOMX

| Pergunta | O que muda aqui |
|---|---|
| 1. O que mudou em maio/2026? | Se foi mudança permanente, re-treinar só com dados de maio em diante, ou dar mais peso a eles. Se foi critério de registro, rever o alvo. |
| ~~2. Camadas de 10 kg são "gancho que não pegou"?~~ | **Respondida:** sim, e o corte passa a ser 2.000 kg. Aplicado no `transforma_dataset.py`; todas as etapas rodaram de novo (esta rodada). |
| 3. Qual erro é pior? | Escolhe um dos três limiares da tabela acima. |
| **(não perguntada)** O resultado da corrida anterior está disponível na hora de prever? | Decide entre o plano A (PR-AUC 0,39 no teste) e o plano B (0,23). **É a decisão com mais impacto no desempenho; vale perguntar.** |

### Limitações

- **Validação e teste pequenos.** São ~515 cestões cada, com 75–90 cargas altas. Diferenças de PR-AUC menores que ~0,08 não são confiáveis (ver os IC 90%).
- **Modelo para uso.** Foi re-treinado em treino + validação; a análise de modelos o mede no teste (o teste vem depois dos dois). Os limiares vêm da validação e não se transferem bem para o modelo re-treinado (ver a análise): rever assim que houver dados novos.
- **Limiares sem arredondar.** O `modelo_final.json` guardava os limiares com 4 casas, e 3 cestões da validação ficavam entre o valor arredondado e o exato (verificação, item 1.5b). Agora os limiares são salvos inteiros, no `.json` e dentro do `.joblib`.
- **Versão do scikit-learn.** Os números podem mudar um pouco com outra versão (aqui: 1.9.1).
