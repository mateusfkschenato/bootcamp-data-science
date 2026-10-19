## Seleção de Algoritmos e Modelagem — Predição de Carga Alta (EVCOMX)

Segue a **Aula 05 do bootcamp** (Seleção de Algoritmos e Modelagem) e continua a pasta `feature_selection/`.

### Como rodar

1. `EDA/transforma_dataset.py`: gera o `EDA/base.csv` limpo.
2. `feature_selection/feature_selection.py`: gera os CSVs de features.
3. `selecao_algoritmos/selecao_algoritmos.py` (células `#%%`) ou `selecao_algoritmos.ipynb` (mesmo conteúdo, já executado). Leva ~4 min por causa da busca de hiperparâmetros.

O código acha a raiz do projeto sozinho. Não recalcula nenhuma feature: lê `dataset_features_completo.csv`, `dataset_modelagem_fs.csv` e `votos_feature_selection.csv` da feature selection.

### Arquivos gerados

| Arquivo | O que é |
|---|---|
| `comparacao_algoritmos.csv` | Todos os modelos (baseline, de fábrica e ajustados): PR-AUC na CV temporal, na validação e IC 90% |
| `limiares_validacao.csv` | Os três limiares de decisão, planos A e B, medidos na validação |
| `resultado_teste.csv` | A conferência única no teste |
| `previsoes_validacao_teste.csv` | Chance de carga alta prevista para cada cestão da validação e do teste |
| `modelo_final_planoA.joblib` / `modelo_final_planoB.joblib` | Modelos para uso, re-treinados em treino + validação |
| `modelo_final.json` | Features, hiperparâmetros e limiares de cada plano |
| `graficos/` | 8 figuras para a apresentação |

### Passo a passo (slide 36 da aula)

1. **Tipo de pergunta: classificação.** "Este cestão vai dar carga alta?" é sim ou não. A regressão linear fica de fora.
2. **Métrica: PR-AUC** para comparar algoritmos. A acurácia engana: quem sempre diz "normal" acerta 82–85%. Recall e precisão vêm depois, com o limiar.
3. **Split e baseline.** O split 70/15/15 temporal vem da coluna `conjunto`. Os baselines são `DummyClassifier`, maioria e sorteio, com PR-AUC de ~0,18 na validação.
4. **Pipeline.** Mediana para vazios, padronização só na logística e o modelo. Quatro algoritmos: logística, árvore, Random Forest e Gradient Boosting.
5. **Validação cruzada temporal.** Cinco provas em janela crescente dentro do treino, por corrida. O `cv=5` da aula sorteia linhas e vazaria o futuro.
6. **Hiperparâmetros.** `GridSearchCV` nas mesmas provas temporais.
7. **Escada (slide 33).** Fica o modelo mais simples a até 0,02 da melhor PR-AUC na validação.
8. **Limiar.** Escolhido na validação, em três cenários.
9. **Teste uma única vez.**

Conjuntos de features testados:
- **final (9)**, da feature selection;
- **consenso (22)**, as features com ≥ 4 votos;
- **final sem histórico (7)**, sem `taxa_ca_ult5/10`. É o plano B, caso o resultado das corridas anteriores não esteja disponível na hora de prever.

### Resultado

**Plano A (com histórico): Random Forest de fábrica com as 9 features finais.** A configuração é `n_estimators=300`, `min_samples_leaf=5` e `class_weight="balanced_subsample"`.

| Modelo (validação, abr–jun/2026) | PR-AUC | IC 90% |
|---|---|---|
| **Random Forest, fábrica, final (9)** | **0,425** | 0,35–0,51 |
| Random Forest, fábrica, consenso (22) | 0,425 | 0,35–0,51 |
| Gradient Boosting, ajustado, consenso (22) | 0,418 | 0,33–0,51 |
| Árvore de decisão, ajustada, final (9) | 0,326 | 0,25–0,40 |
| Regressão logística, fábrica, consenso (22) | 0,315 | 0,25–0,40 |
| Regressão logística, final (9) | 0,257 | 0,21–0,33 |
| Baseline (maioria) | 0,179 | — |

O Gradient Boosting empata com a Random Forest. Pela regra da escada, fica a Random Forest: é um degrau abaixo e precisa de menos ajuste. Entre os dois conjuntos de mesma nota, fica o de 9 features.

**Plano B (sem histórico): Random Forest de fábrica com 7 features.** PR-AUC na validação de 0,318, contra 0,179 do chute.

### Descoberta principal: quem vai bem antes de maio não vai bem depois (gráfico 03)

- **A regressão logística é a melhor na CV temporal e a pior na validação.** Na CV (ago/2025–abr/2026) ela tem PR-AUC 0,44; na validação (abr–jun/2026), 0,26.
- **Ajustar hiperparâmetros na CV quase nunca ajudou a validação.** Na Random Forest com 9 features, a PR-AUC caiu de 0,425 (fábrica) para 0,342 (ajustada).

As duas coisas têm a mesma causa: a **mudança de regime de maio/2026**. O treino é todo do regime antigo, então a CV dentro dele premia o que funciona naquele regime. Por isso:
- a validação decide entre os modelos;
- a versão de fábrica de cada algoritmo continua candidata.

A pergunta 1 do e-mail à EVCOMX é justamente sobre isso.

> **Transparência:** na primeira rodada, só os modelos ajustados eram candidatos. O escolhido foi a Random Forest ajustada com 22 features (validação 0,400; teste 0,369). Ao ver que o ajuste piorava a validação, a versão de fábrica entrou como candidata. Essa mudança foi motivada pela validação, mas o teste daquela rodada já tinha sido calculado. Fica registrado.

### Limiar de decisão (pergunta 3 do e-mail)

O limiar foi escolhido na validação, com no máximo 50% dos cestões em alarme. Esse teto é uma suposição nossa.

| Se a EVCOMX disser que… | Cenário | Limiar | Teste: recall | Teste: precisão | Alarmes / 100 cestões |
|---|---|---|---|---|---|
| carga alta não detectada é pior | não deixar passar (F2) | 0,35 | **80%** | 27% | 43 |
| os dois erros pesam igual | equilíbrio (F1) | 0,54 | 49% | 35% | 20 |
| alarme falso é pior | evitar alarme falso (F0,5) | 0,74 | 24% | 40% | 9 |

No teste há ~15 cargas altas a cada 100 cestões.

Como ler a primeira linha: a cada 100 cestões, o modelo dispara 43 alarmes. 12 deles são carga alta de verdade, e ele deixa passar 3 das ~15.

### Conferência no teste (jun–ago/2026, uma única vez)

| | PR-AUC | ROC-AUC |
|---|---|---|
| **Plano A: Random Forest, 9 features** | **0,354** (IC 90%: 0,29–0,44) | 0,767 |
| Plano B: Random Forest, 7 features sem histórico | 0,202 | 0,608 |
| Baseline (maioria) | 0,146 | 0,500 |

**Sobre a comparação com a feature selection:** no README da feature selection, a regressão logística com 9 features teve PR-AUC 0,40 no teste.
- **Isso não muda a escolha.** O teste não serve para escolher (slide 35), e 0,40 está dentro do intervalo do modelo escolhido.
- **Mas é mais um sinal de instabilidade.** A validação (abr–jun) e o teste (jun–ago) não concordam sobre o melhor algoritmo. Com ~75–90 cargas altas em cada conjunto, as diferenças entre os melhores modelos estão dentro do ruído.

### Explicação do modelo (gráficos 07 e 08)

- **Permutação na validação.** As features mais importantes são `prop_peso_pesado`, `taxa_ca_ult5`, `peso_por_ciclo_t` e `densidade_media`, as mesmas da feature selection.
- **Fluxograma para a apresentação.** O gráfico 08 traz uma árvore de 3 perguntas: densidade, proporção de pesado e carga alta recente. Ela é mais fraca que o modelo (PR-AUC 0,26 contra 0,42), mas mostra a lógica de um jeito que o pátio entende.

### O que depende das respostas da EVCOMX

| Pergunta | O que muda aqui |
|---|---|
| 1. O que mudou em maio/2026? | Se foi mudança permanente, re-treinar só com dados de maio em diante, ou dar mais peso a eles. Se foi critério de registro, rever o alvo. |
| 2. Camadas de 10 kg são "gancho que não pegou"? | Se forem, removê-las no `transforma_dataset.py` e rodar tudo de novo: as features de peso e ciclo mudam. |
| 3. Qual erro é pior? | Escolhe um dos três limiares da tabela acima. |
| **(não perguntada)** O resultado da corrida anterior está disponível na hora de prever? | Decide entre o plano A (PR-AUC 0,35 no teste) e o plano B (0,20). **É a decisão com mais impacto no desempenho; vale perguntar.** |

### Limitações

- **Validação e teste pequenos.** São ~515 cestões cada, com 75–90 cargas altas. Diferenças de PR-AUC menores que ~0,08 não são confiáveis (ver os IC 90%).
- **Modelo para uso.** Foi re-treinado em treino + validação e não tem nota própria; os limiares vêm da validação. Rever os limiares assim que houver dados novos.
- **Versão do scikit-learn.** Os números podem mudar um pouco com outra versão (aqui: 1.9.1).
