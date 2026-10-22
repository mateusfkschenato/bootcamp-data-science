## Análise de Modelos e Métricas — Predição de Carga Alta (EVCOMX)

Segue a **Aula 06 do bootcamp** (Análise de Modelos e Métricas de Desempenho) e o **Guia de Modelagem Preditiva da EVCOMX** (§5), e continua a pasta `selecao_algoritmos/`. A seleção escolheu o modelo. Aqui medimos **o que ele entrega**, com as métricas certas, e traduzimos os erros para a língua da operação.

> **Rodada de out/2026 (corte de 2.000 kg).** A EDA passou a remover as camadas abaixo de 2.000 kg (resposta da EVCOMX) e todas as etapas foram refeitas. O plano A agora tem **6 features** (antes, 9) e o plano B é a Random Forest ajustada (`max_depth=4`) com 4 features. O achado principal não mudou: **no teste, o modelo não se separa da regra simples do histórico do forno**. Os números abaixo são desta rodada.

### Como rodar

1. `EDA/transforma_dataset.py`: gera o `EDA/base.csv` limpo.
2. `feature_selection/feature_selection.py`: gera os CSVs de features.
3. `selecao_algoritmos/selecao_algoritmos.py`: gera o `modelo_final.json`, os `.joblib` e as previsões.
4. `analise_modelos/analise_modelos.py` (células `#%%`) ou `analise_modelos.ipynb` (mesmo conteúdo, já executado). Leva ~2 min.

O código acha a raiz do projeto sozinho. **Não escolhe nada de novo**: algoritmo, features, hiperparâmetros e limiares vêm do `modelo_final.json` (inclusive os do plano B, que pode sair "de fábrica" ou "ajustado" da seleção).

Duas conferências automáticas garantem que esta etapa mede o mesmo modelo da seleção:
- os modelos re-treinados aqui só no treino dão **chances idênticas** às de `previsoes_validacao_teste.csv`;
- recall, precisão e alarmes no teste são **iguais** aos de `resultado_teste.csv`.

Além dos arquivos da seleção, lê `original_dataset/VariavelResposta.csv` para a **duração da parada**. Ela não é feature (só existe depois da carga alta). Serve só para traduzir erros em minutos.

### Arquivos gerados

| Arquivo | O que é |
|---|---|
| `metricas_cv_temporal.csv` | 4 algoritmos × 7 métricas, média e desvio nas 5 provas temporais |
| `metricas_validacao.csv` | As mesmas métricas na validação, com IC 90% |
| `baselines_validacao.csv` | Maioria, sorteio e "regras de uma feature só" |
| `comparacao_pareada_validacao.csv` / `_teste.csv` | Bootstrap pareado: a Random Forest vence de verdade ou é sorte? |
| `custo_conforme_k_validacao.csv` | Custo de cada estratégia para cada razão de custo `k` |
| `faixas_de_risco.csv` | Taxa real de carga alta em cada faixa de chance prevista |
| `metricas_por_periodo.csv` | PR-AUC por período, com IC 90%; recall e precisão no corte F2 |
| `recall_em_minutos.csv` | Recall contado em eventos e em minutos de parada |
| `teste_traduzido.csv` / `custo_no_teste.csv` | O teste a cada 100 cestões, a cada 30 dias e em custo |
| `modelo_para_uso_*.csv` | O modelo re-treinado em treino + validação, medido no teste |
| `calibracao_aplicada.csv` | Brier antes e depois de calibrar (Platt e isotônica) |
| `tabela_unica_validacao.csv` / `_teste.csv` | **A tabela única do Guia (§5.5)**: modelos e baselines lado a lado |
| `curva_de_aprendizado.csv` | PR-AUC conforme a quantidade de treino |
| `desempenho_por_segmento.csv` | Carregamento 1 × 2, turno e faixa de volume |
| `analise_erros_fn_fp.csv` | O que as cargas altas perdidas e os alarmes falsos têm em comum |
| `casos_para_especialista.csv` | Os 10 alarmes falsos e as 10 cargas altas perdidas mais "certos", para revisar com o pátio |
| `dependencia_parcial_sentido.csv` | Para onde a chance anda quando cada feature cresce |
| `regras_para_o_patio.csv` | Taxa real por histórico do forno × densidade do cestão |
| `graficos/` | 10 figuras para a apresentação |

### Passo a passo (slide 37 da aula)

1. **Qual erro custa mais caro?** Uma carga alta não detectada (FN) para o forno **4 min em média** (mediana 3 min) e soma ~185 min a cada 30 dias. O custo de um alarme falso (FP) depende do que o pátio faz com o alarme: é a pergunta 3 do e-mail. Enquanto isso, usamos a razão `k = custo do FN ÷ custo do FP`.
2. **Métrica principal antes de testar.** PR-AUC para comparar modelos (não depende do corte e foca na classe rara), ROC-AUC como segunda opinião, recall e precisão depois do corte. A acurácia engana: na validação, quem sempre diz "normal" tem a **maior** acurácia (82%) e recall zero.
3. **Baseline na mesma métrica.** Além de maioria e sorteio, **regras de uma feature só**, sem treinar nada.
4. **Comparar com validação cruzada** temporal, em várias métricas, e na validação com IC 90% e bootstrap pareado.
5. **Matriz de confusão** do escolhido, nos três cortes.
6. **Corte pelo custo**, para cada `k`.
7. **Teste traduzido** em alarmes a cada 100 cestões, minutos de parada e custo.

Extras: **calibração** (a "chance" do modelo é chance de verdade?), **estabilidade no tempo** e **o modelo para uso** medido no teste.

**Como ler "vantagem real".** Em cada comparação, sorteamos 1.000 vezes as corridas (bootstrap pareado) e contamos em quantos sorteios a Random Forest vence. 95% ou mais = vantagem real; 80–95% = provável; 20–80% = empate; abaixo de 20% e de 5% = desvantagem provável e real.

### Achado principal: a régua honesta é o histórico do forno, não o chute

| Validação (abr–jun/2026) | PR-AUC | ROC-AUC | Recall com 20 alarmes por 100 |
|---|---|---|---|
| Baseline: maioria | 0,18 | 0,50 | 0,20 |
| Regra: menos `densidade_media` | 0,17 | 0,49 | 0,16 |
| Regra: menos `prop_peso_pesado` | 0,21 | 0,58 | 0,25 |
| Regra física do Guia: mais `volume_m3_total` | 0,15 | 0,43 | 0,12 |
| **Regra: mais `taxa_ca_ult5` (só histórico)** | **0,34** | 0,69 | **0,42** |
| **Random Forest (plano A)** | **0,43** | 0,73 | **0,42** |

| Random Forest contra a regra do histórico | PR-AUC | Recall com 20 alarmes por 100 |
|---|---|---|
| Validação, RF treinada só no treino | vantagem real (+0,09) | empate (+0,01) |
| Teste, RF treinada só no treino | empate (0,391 × 0,358) | **desvantagem real** (0,45 × 0,56) |
| Teste, **RF para uso** (treino + validação) | vantagem provável (0,404 × 0,358) | empate (0,53 × 0,56) |

- A **regra de volume do Guia** (§4.2) fica **abaixo** do chute: nestes dados, mais volume não é mais risco (no treino, o sinal da correlação é até negativo). O modelo a supera com folga; a régua que aperta é o histórico.
- As features de composição **sozinhas** ficam no nível do chute no regime pós-maio. Juntas, sem o histórico (plano B), ficam acima do chute: PR-AUC 0,23 no teste, contra 0,15.
- O plano A supera o plano B com folga (vantagem real no teste): o histórico é o que mais pesa.

Leitura: o sinal mais forte e mais estável é **"carga alta puxa carga alta"**. A composição do cestão contribui, mas, no regime novo, ainda **não se mostrou maior** do que a regra simples do histórico. Em PR-AUC o modelo fica um pouco à frente no teste (e o modelo para uso chega a "vantagem provável"), mas com o mesmo número de alarmes a regra pega tanto ou mais cargas altas. **Na apresentação, a régua do modelo deve ser a regra do histórico, não só o chute** (slide 36: "esquecer o baseline").

### Comparação entre algoritmos (gráficos 01 e 02)

| Validação | PR-AUC (IC 90%) | ROC-AUC | Recall c/ 20 alarmes por 100 | Random Forest contra ele (PR-AUC / recall) |
|---|---|---|---|---|
| Regressão logística | 0,29 (0,23–0,37) | 0,67 | 0,35 | vantagem real / provável |
| Árvore de decisão | 0,30 (0,24–0,37) | 0,67 | 0,37 | vantagem real / provável |
| **Random Forest** | **0,43** (0,35–0,52) | 0,73 | 0,42 | — |
| Gradient Boosting | 0,39 (0,32–0,49) | 0,73 | 0,42 | provável / empate |

- **Random Forest e Gradient Boosting empatam.** A seleção ficou com a Random Forest pela regra da escada (slide 31: na dúvida, o mais simples e estável). A análise confirma a escolha.
- **Cuidado com métricas no corte 0,5 na CV** (`metricas_cv_temporal.csv`). O Gradient Boosting tem recall 0,23 nesse corte, mas é efeito do corte: os outros três usam `class_weight="balanced"`, que empurra as chances para cima. Com o mesmo orçamento de alarmes, os quatro ficam entre 0,49 e 0,51 de recall na CV.
- A curva ROC parece boa (0,73). A de precisão-recall mostra o preço: para pegar 76% das cargas altas, só ~3 em cada 10 alarmes são verdadeiros.
- **Por que CV temporal e não o `cv=5` da aula:** o `cv=5` divide o treino em 5 partes e, em 4 das 5 provas, treina com meses posteriores para prever meses anteriores. O modelo "veria o futuro".

### O corte pelo custo (gráfico 03, pergunta 3 do e-mail)

Custo a cada 100 cestões = `alarmes falsos + k × cargas altas não detectadas`. Para cada `k`, o corte de menor custo foi escolhido na validação.

| Se 1 carga alta não detectada vale… | Corte mais barato da seleção | Teste: recall | Teste: precisão | Alarmes / 100 |
|---|---|---|---|---|
| até ~2,5 alarmes falsos | evitar alarme falso (0,77) | 33% | 45% | 11 |
| de ~2,75 a ~4,75 alarmes falsos | equilíbrio (0,54) | 63% | 34% | 27 |
| 5 alarmes falsos ou mais | não deixar passar (0,41) | 76% | 29% | 38 |

- **Na validação, a economia é otimista por construção.** O corte "ótimo" é procurado entre todos os cortes, inclusive "nunca alarmar", então o modelo nunca fica pior que as estratégias sem modelo nos mesmos dados. O gráfico 03 serve como **mapa** (qual corte para qual `k`), não como prova de ganho.
- **A prova é o teste**, com os cortes da validação: economia de 12% com `k = 2`, 20% com `k = 3`, 37% com `k = 5` e 35% com `k = 8`. **Com `k` pequeno, o ganho é pequeno.**
- **Como a EVCOMX chega no `k`:** se um alarme falso custar ~1 min de trabalho no pátio e um minuto de pátio valer o mesmo que um minuto de forno parado, `k ≈ 4` → corte "equilíbrio". Se o minuto de forno valer mais (energia, produção), `k` sobe e o corte vai para "não deixar passar".

### A "chance" do modelo não é chance de verdade (gráfico 04)

- A chance média prevista é **2,3 a 2,7 vezes** a taxa real: 0,41 contra 0,18 na validação; 0,39 contra 0,15 no teste. O `class_weight="balanced"` infla as chances.
- O Brier score do modelo (0,20 na validação) é **pior** que o de quem sempre diz a taxa média (0,15). O modelo ordena bem, mas a chance crua não deve ir para o pátio.
- **Proposta: faixas de risco com a taxa real.** As fronteiras são os três cortes da seleção:

| Faixa | Chance prevista | Taxa real (validação) | Taxa real (teste) | % dos cestões (teste) |
|---|---|---|---|---|
| baixo | 0,00–0,41 | 8% | 6% | 62% |
| atenção | 0,41–0,54 | 17% | 17% | 11% |
| alto | 0,54–0,77 | 27% | 26% | 17% |
| muito alto | 0,77–1,00 | 62% | 45% | 11% |

A ordem se mantém no teste. A faixa "muito alto" cai de 62% para 45%, por isso a taxa de cada faixa deve ser revista com dados novos.

### Calibração aplicada (seção 8b, Guia §6.2)

O Guia pede: se as chances não batem com as frequências reais, calibrar. Testamos Platt e isotônica de dois jeitos: com o calibrador ajustado na **validação** (sobre a floresta treinada no treino) e **dentro da CV temporal** (`CalibratedClassifierCV` com as 5 provas).

| Versão | Brier validação | Brier teste | Chance média no teste (taxa real 0,15) | PR-AUC teste |
|---|---|---|---|---|
| sem calibrar | 0,196 | 0,182 | 0,39 | 0,391 |
| **Platt, ajustada na validação** | — | **0,109** | 0,17 | 0,391 |
| isotônica, ajustada na validação | — | 0,110 | 0,18 | 0,363 |
| Platt, dentro da CV temporal | 0,138 | 0,115 | 0,21 | 0,359 |
| isotônica, dentro da CV temporal | 0,141 | 0,121 | 0,23 | 0,339 |
| referência: sempre a taxa do treino | 0,148 | 0,125 | 0,16 | 0,146 |

- **A calibração resolve a chance inflada.** Com Platt, o Brier cai de 0,182 para 0,109 no teste, abaixo da referência (0,125), e a chance média passa a ser parecida com a taxa real.
- **Platt não muda a ordem dos cestões** (a PR-AUC é a mesma), então não muda os alarmes por volume. A isotônica cria empates e perde um pouco de PR-AUC quando ajustada só na validação.
- **Proposta:** o alarme continua definido pelo volume de alarmes; o **nível de risco mostrado** ao pátio passa a ser a chance calibrada (Platt) ou a faixa de risco com a taxa real. O calibrador é re-ajustado junto com cada re-treino.

### Estabilidade no tempo (gráfico 05)

- O modelo fica acima do chute **em todos os 8 períodos**, mesmo no limite baixo do IC 90%.
- As notas oscilam de 0,25 a 0,62, mas os intervalos são largos (com 29 a 78 cargas altas por período). A queda de mar–mai (prova 5: 0,30; fim abr–mai: 0,25) coincide com a mudança de maio, mas o IC de fim abr–mai (0,18–0,39) se sobrepõe ao de junho (0,35–0,52). **Não dá para afirmar que a queda é real.**
- Os últimos dias de abril (49 cestões) foram somados a maio, e agosto (12 cestões) a julho.

### Minutos de parada

- O recall contado em **minutos de parada** é parecido com o recall em eventos (teste, corte "não deixar passar": 76% dos eventos, 84% dos minutos).
- Nas 167 cargas altas de validação + teste, a chance prevista quase não se relaciona com a duração (Spearman 0,09, p = 0,25), e a duração das cargas altas pegas e perdidas não difere (Mann-Whitney, p = 0,24). **O modelo não separa bem paradas longas de curtas.**

### Teste traduzido para a operação (gráfico 06)

Teste: 515 cestões em 41 dias (~380 cestões e ~185 min de parada a cada 30 dias). Modelo treinado só no treino; cortes vindos da validação.

| Plano | Corte | Alarmes / 100 | Cargas altas pegas / 100 | Perdidas / 100 | Alarmes falsos / 100 | Falsos por acerto | Min. de parada cobertos / 30 dias |
|---|---|---|---|---|---|---|---|
| A | não deixar passar | 38 | 11 | 3,5 | 27 | 2,5 | 154 de 185 |
| A | equilíbrio | 27 | 9 | 5 | 18 | 2,0 | 123 de 185 |
| A | evitar alarme falso | 11 | 5 | 10 | 6 | 1,2 | 72 de 185 |
| B | não deixar passar | 51 | 9 | 5 | 42 | 4,5 | 143 de 185 |
| B | equilíbrio | 15 | 4 | 10 | 10 | 2,5 | 75 de 185 |
| B | evitar alarme falso | 11 | 3 | 11 | 8 | 2,4 | 55 de 185 |

"Minutos cobertos" supõe que um alarme certo permite **evitar** a carga alta. É um teto do ganho, não uma promessa.

### O modelo para uso, medido no teste

Os `.joblib` da seleção foram re-treinados em treino + validação. O teste vem depois dos dois, então medir esses modelos nele é legítimo e não muda nenhuma escolha.

| Teste | PR-AUC | ROC-AUC | Recall c/ 20 alarmes por 100 |
|---|---|---|---|
| RF plano A, treinada só no treino | 0,391 | 0,77 | 0,45 |
| **RF plano A para uso** | **0,404** | 0,79 | **0,53** |
| RF plano B para uso | 0,209 | 0,61 | 0,32 |
| Regra: só histórico | 0,358 | 0,79 | 0,56 |

**Os limiares não se transferem.** O modelo para uso dá chances mais baixas (média 0,34 contra 0,39), então o mesmo limiar dispara menos alarmes:

| Corte (mesmo limiar) | Alarmes / 100: treinada só no treino | Alarmes / 100: para uso | Recall: só treino → para uso |
|---|---|---|---|
| não deixar passar (0,41) | 38 | 34 | 76% → 72% |
| equilíbrio (0,54) | 27 | 21 | 63% → 53% |
| evitar alarme falso (0,77) | 11 | 9 | 33% → 29% |

**Recomendação:** na operação, fixar o corte pelo **volume de alarmes** ("os N cestões de maior risco a cada 100") e recalcular o limiar equivalente a cada re-treino, com dados recentes.

### Revisão pelo Guia de Modelagem (§5.5 a §5.7)

#### Tabela única de comparação (seção 11b, Guia §5.5)

Validação (abr–jun/2026), todos treinados só no treino. O limiar de cada linha é o de melhor F2 na própria validação (no máximo metade em alarme), então recall, precisão, F2, MCC e acurácia balanceada são otimistas **para todas as linhas por igual**. PR-AUC, ROC-AUC e Brier não dependem do limiar. Regras não dão chance: o Brier não se aplica a elas.

| Modelo | Features | PR-AUC | ROC-AUC | Recall @ limiar | Precisão @ limiar | F2 | Brier | MCC | Acur. balanceada | Alarmes /100 |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline trivial: sempre "normal" | 0 | 0,18 | 0,50 | 0,00 | — | 0,00 | 0,18 | 0,00 | 0,50 | 0 |
| Baseline físico: regra de volume | 1 | 0,15 | 0,43 | 0,40 | 0,15 | 0,30 | — | −0,09 | 0,44 | 50 |
| Baseline: regra do histórico | 1 | 0,34 | 0,69 | 0,64 | 0,26 | 0,50 | — | 0,19 | 0,62 | 44 |
| Regressão logística | 6 | 0,29 | 0,67 | 0,70 | 0,26 | 0,52 | 0,32 | 0,20 | 0,63 | 48 |
| Árvore de decisão | 6 | 0,30 | 0,67 | 0,60 | 0,24 | 0,46 | 0,32 | 0,15 | 0,60 | 44 |
| **Random Forest (plano A)** | 6 | **0,43** | **0,73** | 0,76 | 0,29 | 0,57 | 0,20 | 0,27 | 0,68 | 47 |
| Gradient Boosting | 6 | 0,39 | 0,73 | **0,79** | 0,29 | **0,59** | **0,14** | **0,28** | **0,69** | 49 |
| Random Forest sem histórico (plano B) | 4 | 0,32 | 0,64 | 0,64 | 0,24 | 0,48 | 0,31 | 0,15 | 0,60 | 48 |

Teste (jun–ago/2026), só os escolhidos e os baselines, com o limiar vindo da validação:

| Modelo | PR-AUC | ROC-AUC | Recall @ limiar | Precisão @ limiar | F2 | Brier | MCC | Acur. balanceada | Alarmes /100 |
|---|---|---|---|---|---|---|---|---|---|
| Baseline trivial | 0,15 | 0,50 | 0,00 | — | 0,00 | 0,15 | 0,00 | 0,50 | 0 |
| Baseline físico: regra de volume | 0,15 | 0,52 | 0,49 | 0,14 | 0,33 | — | −0,00 | 0,50 | 50 |
| Baseline: regra do histórico | 0,36 | **0,79** | **0,76** | **0,34** | **0,61** | — | **0,38** | **0,76** | **32** |
| **Random Forest (plano A)** | **0,39** | 0,77 | **0,76** | 0,29 | 0,57 | 0,18 | 0,32 | 0,72 | 38 |
| Random Forest sem histórico (plano B) | 0,23 | 0,61 | 0,64 | 0,18 | 0,43 | 0,32 | 0,11 | 0,57 | 51 |

- Na validação, a Random Forest lidera nas métricas que não dependem do corte (PR-AUC e ROC-AUC). Nas métricas do corte, o Gradient Boosting empata e fica até um pouco à frente, e tem o melhor Brier (ele não usa pesos de classe, então a chance infla menos).
- **No teste, a regra do histórico, com o limiar da validação (0,4 = 2 cargas altas nas últimas 5 corridas), pega a mesma fração das cargas altas que o modelo (76%) com menos alarmes (32 contra 38 a cada 100), e por isso tem F2, MCC e acurácia balanceada maiores.** O modelo só fica à frente em PR-AUC (0,39 × 0,36), por uma diferença que o bootstrap pareado classifica como empate.

#### Curva de aprendizado (seção 11c, gráfico 07)

Mesmo modelo, treinado com frações crescentes do treino, medido na validação:

| Fração do treino | 20% | 30% | 50% | 70% | 90% | 100% |
|---|---|---|---|---|---|---|
| a partir do mais antigo | 0,35 | 0,37 | 0,39 | 0,40 | 0,40 | 0,43 |
| a partir do mais recente | 0,37 | 0,39 | 0,41 | 0,43 | 0,41 | 0,43 |

- O modelo acerta ~0,84 a 0,91 de PR-AUC no próprio treino em qualquer tamanho: o limite é de **variância** (poucos eventos), não de viés.
- **Os meses recentes valem mais que os antigos:** com os 60–70% mais recentes do treino o modelo já chega a 0,43, a nota do treino inteiro; começando do mais antigo, só chega lá com 100%. Com os 20% mais recentes (fev–abr/2026) ele tem 0,37, um pouco acima dos 20% mais antigos (0,35). Isso pesa a favor de **re-treinar acumulando dados, dando mais peso ao período recente**, se a EVCOMX confirmar que a mudança de maio é permanente.

#### Desempenho por segmento (seção 11c, gráfico 08)

Validação + teste (1.028 cestões, 167 cargas altas), modelo treinado só no treino, corte "não deixar passar":

| Segmento | Cestões | Cargas altas | Taxa (chute) | PR-AUC (IC 90%) | Recall | Precisão | Alarmes /100 |
|---|---|---|---|---|---|---|---|
| cestão 1 | 540 | 81 | 0,15 | 0,33 (0,26–0,42) | 67% | 28% | 36 |
| cestão 2 | 488 | 86 | 0,18 | 0,47 (0,39–0,56) | 85% | 30% | 51 |
| turno A (06–14h) | 364 | 48 | 0,13 | 0,39 (0,29–0,50) | 81% | 25% | 43 |
| turno B (14–22h) | 294 | 48 | 0,16 | 0,39 (0,30–0,51) | 71% | 30% | 38 |
| turno C (22–06h) | 370 | 71 | 0,19 | 0,45 (0,37–0,55) | 76% | 32% | 46 |
| volume baixo (< 47 m³) | 394 | 64 | 0,16 | 0,44 (0,35–0,55) | 84% | 28% | 49 |
| volume médio | 157 | 38 | 0,24 | 0,50 (0,38–0,64) | 76% | 37% | 50 |
| volume alto (> 64 m³) | 477 | 65 | 0,14 | 0,33 (0,25–0,43) | 68% | 26% | 35 |

- O modelo fica acima do chute em todos os segmentos, e os intervalos se sobrepõem: **não há segmento onde ele falhe**.
- **Os alarmes se concentram no cestão 2 e nos cestões de pouco volume** (51 e 49 a cada 100), onde o recall é alto. No **cestão 1 e nos cestões volumosos**, o modelo alarma menos e perde mais (recall 67–68%).

#### Falsos negativos e falsos positivos (seção 11c)

No corte "não deixar passar", validação + teste: 127 VP, 40 FN, 312 FP, 549 VN.

| | Cargas altas perdidas (FN) | Pegas (VP) | Alarmes falsos (FP) | Normais sem alarme (VN) |
|---|---|---|---|---|
| densidade média (t/m³, mediana) | 0,79 | 0,71 | 0,69 | 0,77 |
| proporção de pesado | 0,27 | 0,14 | 0,14 | 0,24 |
| toneladas por ciclo | 17,2 | 27,0 | 25,2 | 18,4 |
| ciclos de montagem | 3 | 1 | 1 | 3 |
| peso total (t) | 52,2 | 34,4 | 34,0 | 52,2 |
| carga alta nas últimas 5 corridas | 0,2 | 0,8 | 0,4 | 0,0 |
| % de cestões 2 | 33% | 57% | 56% | 42% |

(Todas as diferenças da tabela com p < 0,01, Mann-Whitney.)

- **As cargas altas perdidas têm "cara de cestão bom":** pesadas, densas, bem intercaladas (3 ciclos), mais vezes cestão 1 e com o forno calmo. Com as features de hoje, o modelo não tem como vê-las; elas sugerem uma causa que não está nos dados (umidade, forma física da sucata, estado do forno). São as primeiras candidatas para a conversa com o pátio (`casos_para_especialista.csv`).
- **Os alarmes falsos têm "cara de cestão arriscado":** leves, pouco densos, um ciclo só, forno com carga alta recente, mais cestão 2. É a hipótese do Guia ("talvez seja um cestão realmente arriscado que, por sorte, não gerou evento"): pelo perfil, eles se parecem com os acertos (VP), não com os normais.

#### Interpretabilidade (seção 11d, gráficos 09 e 10)

O pacote `shap` não faz parte do ambiente do projeto. Usamos a **decomposição por caminho das árvores** (método de Saabas, o do pacote `treeinterpreter`): em cada árvore, a mudança da chance a cada pergunta vai para a feature da pergunta. A chance de base (0,50, por causa dos pesos balanceados) mais as contribuições dá **exatamente** a chance prevista; o script confere isso nos 513 cestões da validação. É uma aproximação do SHAP: a ideia é a mesma, a divisão das interações é diferente.

- **Importância global** (contribuição média em módulo, validação): `prop_peso_pesado`, `taxa_ca_ult5`, `peso_por_ciclo_t`, `densidade_media` e `taxa_ca_ult10` na frente; `prop_PACB` quase não contribui. É quase a mesma ordem da permutação (seleção, gráfico 07).
- **Explicação local** (o cestão de maior risco do teste, O7144 c2, chance 0,97, deu carga alta): pouco pesado (10% do peso, +0,16), forno com 5 cargas altas nas últimas 5 corridas (+0,09), ciclos grossos (34,0 t por ciclo, +0,08), 9 cargas altas nas últimas 10 (+0,07) e cestão fofo (0,72 t/m³, +0,06). É o tipo de frase que o Guia pede: "este cestão foi sinalizado porque…".
- **Dependência parcial** (variação da chance do menor ao maior valor de cada feature, validação): `taxa_ca_ult5` +0,22 (sobe em 100% dos passos), `taxa_ca_ult10` +0,17, `peso_por_ciclo_t` +0,08, `prop_PACB` +0,01; `prop_peso_pesado` −0,21 e `densidade_media` −0,14. **Todos os sentidos batem com a física.** Com o corte de 2.000 kg, o efeito parcial da densidade ficou maior (era −0,04 com 9 features): sem `max_bloco_leve_m3` e `prop_leve_base`, ela carrega mais do "cestão fofo".

**Tradução em regras para o pátio** (taxa real na validação, abr–jun/2026; densidade pelos tercis do treino):

| Histórico do forno | fofo (< 0,80 t/m³) | médio | denso (> 0,90 t/m³) | todas as densidades |
|---|---|---|---|---|
| nenhuma carga alta nas últimas 5 corridas | 9% (n=120) | 0% (n=28) | 11% (n=9) | **8%** (n=157) |
| 1 carga alta nas últimas 5 | 17% (n=96) | 11% (n=19) | 20% (n=15) | **16%** (n=130) |
| 2 ou mais nas últimas 5 | 24% (n=172) | 37% (n=49) | 0% (n=5) | **26%** (n=226) |

Com 2 ou mais cargas altas nas últimas 5 corridas, cerca de 1 em cada 4 cestões dá carga alta (contra ~1 em 12 quando o forno está calmo). No regime novo quase não há cestões densos (29 na validação inteira), por isso a densidade separa pouco nesta tabela. Com poucas amostras em algumas células, a tabela é ilustrativa; ela serve para explicar a lógica, não para substituir o modelo.

### E as métricas de regressão?

MAE e RMSE (slides 24 a 29) medem erros em números. A pergunta da EVCOMX é sim ou não, então elas não se aplicam. Prever a duração da parada seria regressão, mas ela só existe depois da carga alta. O Brier score (calibração) é o parente mais próximo do RMSE na classificação.

### Conclusão: o desempenho é bom o suficiente para uso operacional? (Guia FE/FS §11.2)

A resposta depende de duas coisas que só a EVCOMX sabe: **quanto custa um alarme falso** (`k`) e **quantos alarmes o pátio aguenta**.

- **Contra a operação de hoje (sem alarme): ajuda, se `k` não for pequeno.** Com o corte "não deixar passar", o plano A pega 76% das cargas altas do teste e cobre ~154 dos ~185 min de parada a cada 30 dias, com ~38 alarmes a cada 100 cestões. No teste, a economia vai de ~12% (`k = 2`) a ~37% (`k = 5`).
- **Contra a regra simples do histórico: ainda não está provado.** O modelo treinado só no treino empata com ela em PR-AUC e perde com o mesmo número de alarmes; o modelo para uso chega a "vantagem provável" em PR-AUC, mas empata no orçamento de alarmes. Antes de dizer que a composição do cestão agrega, é preciso entender maio/2026 e medir de novo.
- **A chance crua não vai para o pátio.** Chance calibrada (Platt) ou faixas de risco com a taxa real de cada faixa.
- **Contra a regra de volume do Guia: sim, com folga.** A regra de volume fica no nível do chute.
- **O corte vem de `k` e se fixa pelo volume de alarmes**, não por um limiar fixo de chance.

A última célula do script monta essa conclusão com os números da rodada, para continuar certa se os dados mudarem.

### O que depende das respostas da EVCOMX

| Pergunta | O que muda aqui |
|---|---|
| 1. O que mudou em maio/2026? | Decide se re-treinamos só com o regime novo (ou dando mais peso a ele, como a curva de aprendizado sugere). É o caminho para saber se a composição do cestão supera a regra do histórico. |
| ~~2. Camadas de 10 kg são "gancho que não pegou"?~~ | **Respondida:** sim, e o corte passa a ser 2.000 kg. Todo o pipeline rodou de novo; esta é a rodada nova. |
| 3. Qual erro é pior? Quanto custa um alarme falso? | Define `k` e, com ele, o corte. |
| **(não perguntada)** O resultado da corrida anterior está disponível na hora de prever? | Ficou ainda mais importante: o histórico é o sinal mais forte. Sem ele, só o plano B (PR-AUC 0,23 no teste). |
| **(nova)** O pátio dá conta de quantos alarmes por 100 cestões? | O corte "não deixar passar" dispara ~34–38. Se o teto for menor, o corte sobe. |

### Limitações

- **Validação e teste pequenos.** 75–92 cargas altas em cada um. Por isso todas as comparações vêm com IC 90% e bootstrap pareado por corrida.
- **O teste já tinha sido usado** na seleção, e foi visto de novo nesta rodada (o pipeline inteiro foi refeito com o corte de 2.000 kg, que veio da EVCOMX e não do teste). Aqui ele não escolhe nada: os cortes e o modelo vêm da validação.
- **Custo do FP desconhecido.** Todo o custo está em "alarmes falsos equivalentes" até a EVCOMX responder.
- **"Minutos cobertos" é um teto**: supõe que o alarme permite evitar a carga alta.
- **Notebook.** O `.ipynb` foi gerado executando as células do `.py` em ordem; o conteúdo é o mesmo.
- **Versão do scikit-learn.** 1.9.1, a mesma da seleção (as conferências dependem disso).
- **SHAP.** O pacote `shap` não está no ambiente; as contribuições locais usam a decomposição de Saabas, que é exata para a chance prevista mas divide as interações de outro jeito. Se o grupo instalar o `shap`, vale comparar.
- **Segmentos e erros** usam validação + teste juntos para ter mais eventos. Nada ali escolhe ou ajusta o modelo.
