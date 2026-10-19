## Análise de Modelos e Métricas — Predição de Carga Alta (EVCOMX)

Segue a **Aula 06 do bootcamp** (Análise de Modelos e Métricas de Desempenho) e continua a pasta `selecao_algoritmos/`. A seleção escolheu o modelo. Aqui medimos **o que ele entrega**, com as métricas certas, e traduzimos os erros para a língua da operação.

### Como rodar

1. `EDA/transforma_dataset.py`: gera o `EDA/base.csv` limpo.
2. `feature_selection/feature_selection.py`: gera os CSVs de features.
3. `selecao_algoritmos/selecao_algoritmos.py`: gera o `modelo_final.json`, os `.joblib` e as previsões.
4. `analise_modelos/analise_modelos.py` (células `#%%`) ou `analise_modelos.ipynb` (mesmo conteúdo, já executado). Leva ~1 min.

O código acha a raiz do projeto sozinho. **Não escolhe nada de novo**: algoritmo, features e limiares vêm do `modelo_final.json`.

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
| `graficos/` | 6 figuras para a apresentação |

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
| Regra: menos `densidade_media` | 0,17 | 0,49 | 0,14 |
| Regra: menos `prop_peso_pesado` | 0,21 | 0,57 | 0,24 |
| **Regra: mais `taxa_ca_ult5` (só histórico)** | **0,34** | 0,69 | **0,42** |
| **Random Forest (plano A)** | **0,42** | 0,72 | **0,45** |

| Random Forest contra a regra do histórico | PR-AUC | Recall com 20 alarmes por 100 |
|---|---|---|
| Validação, RF treinada só no treino | vantagem real (+0,09) | empate (+0,03) |
| Teste, RF treinada só no treino | empate (0,354 × 0,358) | **desvantagem real** (0,48 × 0,56) |
| Teste, **RF para uso** (treino + validação) | empate (0,372 × 0,358) | empate (0,53 × 0,56) |

- As features de composição **sozinhas** ficam no nível do chute no regime pós-maio. Juntas, sem o histórico (plano B), ficam acima do chute: PR-AUC 0,20 no teste, contra 0,15.
- O plano A supera o plano B com folga (vantagem real no teste): o histórico é o que mais pesa.

Leitura: o sinal mais forte e mais estável é **"carga alta puxa carga alta"**. A composição do cestão contribui, mas, no regime novo, ainda **não se mostrou maior** do que a regra simples do histórico. Re-treinar com abril–junho melhorou o modelo, mas não o bastante para se separar da regra. **Na apresentação, a régua do modelo deve ser a regra do histórico, não só o chute** (slide 36: "esquecer o baseline").

### Comparação entre algoritmos (gráficos 01 e 02)

| Validação | PR-AUC (IC 90%) | ROC-AUC | Recall c/ 20 alarmes por 100 | Random Forest contra ele (PR-AUC / recall) |
|---|---|---|---|---|
| Regressão logística | 0,26 (0,21–0,32) | 0,65 | 0,34 | vantagem real / real |
| Árvore de decisão | 0,31 (0,25–0,38) | 0,69 | 0,38 | vantagem real / real |
| **Random Forest** | **0,42** (0,34–0,52) | 0,72 | 0,45 | — |
| Gradient Boosting | 0,39 (0,31–0,48) | 0,71 | 0,43 | provável / empate |

- **Random Forest e Gradient Boosting empatam.** A seleção ficou com a Random Forest pela regra da escada (slide 31: na dúvida, o mais simples e estável). A análise confirma a escolha.
- **Cuidado com métricas no corte 0,5 na CV** (`metricas_cv_temporal.csv`). O Gradient Boosting tem recall 0,17 nesse corte, mas é efeito do corte: os outros três usam `class_weight="balanced"`, que empurra as chances para cima. Com o mesmo orçamento de alarmes, os quatro ficam entre 0,46 e 0,52 de recall na CV.
- A curva ROC parece boa (0,72). A de precisão-recall mostra o preço: para pegar 74% das cargas altas, só ~1 em cada 4 alarmes é verdadeiro.
- **Por que CV temporal e não o `cv=5` da aula:** o `cv=5` divide o treino em 5 partes e, em 4 das 5 provas, treina com meses posteriores para prever meses anteriores. O modelo "veria o futuro".

### O corte pelo custo (gráfico 03, pergunta 3 do e-mail)

Custo a cada 100 cestões = `alarmes falsos + k × cargas altas não detectadas`. Para cada `k`, o corte de menor custo foi escolhido na validação.

| Se 1 carga alta não detectada vale… | Corte mais barato da seleção | Teste: recall | Teste: precisão | Alarmes / 100 |
|---|---|---|---|---|
| até ~2 alarmes falsos | evitar alarme falso (0,74) | 24% | 40% | 9 |
| de ~2,5 a ~5 alarmes falsos | equilíbrio (0,54) | 49% | 35% | 20 |
| mais de ~5,5 alarmes falsos | não deixar passar (0,35) | 80% | 27% | 43 |

- **Na validação, a economia é otimista por construção.** O corte "ótimo" é procurado entre todos os cortes, inclusive "nunca alarmar", então o modelo nunca fica pior que as estratégias sem modelo nos mesmos dados. O gráfico 03 serve como **mapa** (qual corte para qual `k`), não como prova de ganho.
- **A prova é o teste**, com os cortes da validação: economia de 1% com `k = 2`, 12% com `k = 3`, 33% com `k = 5` e 35% com `k = 8`. **Com `k` pequeno, o modelo quase não compensa.**
- **Como a EVCOMX chega no `k`:** se um alarme falso custar ~1 min de trabalho no pátio e um minuto de pátio valer o mesmo que um minuto de forno parado, `k ≈ 4` → corte "equilíbrio". Se o minuto de forno valer mais (energia, produção), `k` sobe e o corte vai para "não deixar passar".

### A "chance" do modelo não é chance de verdade (gráfico 04)

- A chance média prevista é **2,1 a 2,4 vezes** a taxa real: 0,38 contra 0,18 na validação; 0,35 contra 0,15 no teste. O `class_weight="balanced"` infla as chances.
- O Brier score do modelo (0,18) é **pior** que o de quem sempre diz a taxa média (0,15). O modelo ordena bem, mas a chance crua não deve ir para o pátio.
- **Proposta: faixas de risco com a taxa real.** As fronteiras são os três cortes da seleção:

| Faixa | Chance prevista | Taxa real (validação) | Taxa real (teste) | % dos cestões (teste) |
|---|---|---|---|---|
| baixo | 0,00–0,35 | 9% | 5% | 57% |
| atenção | 0,35–0,54 | 16% | 20% | 22% |
| alto | 0,54–0,74 | 30% | 32% | 12% |
| muito alto | 0,74–1,00 | 64% | 40% | 9% |

A ordem se mantém no teste. A faixa "muito alto" cai de 64% para 40%, por isso a taxa de cada faixa deve ser revista com dados novos.

### Estabilidade no tempo (gráfico 05)

- O modelo fica acima do chute **em todos os 8 períodos**, mesmo no limite baixo do IC 90%.
- As notas oscilam de 0,25 a 0,59, mas os intervalos são largos (com 29 a 78 cargas altas por período). A queda de mar–mai (prova 5: 0,25; fim abr–mai: 0,28) coincide com a mudança de maio, mas o IC de fim abr–mai (0,19–0,43) se sobrepõe ao de junho (0,33–0,50). **Não dá para afirmar que a queda é real.**
- Os últimos dias de abril (49 cestões) foram somados a maio, e agosto (12 cestões) a julho.

### Minutos de parada

- O recall contado em **minutos de parada** é praticamente igual ao recall em eventos (teste, corte "não deixar passar": 80% dos eventos, 84% dos minutos).
- Nas 167 cargas altas de validação + teste, a chance prevista quase não se relaciona com a duração (Spearman 0,09, p = 0,23), e a duração das cargas altas pegas e perdidas não difere (Mann-Whitney, p = 0,23). **O modelo não separa bem paradas longas de curtas.**

### Teste traduzido para a operação (gráfico 06)

Teste: 515 cestões em 41 dias (~380 cestões e ~185 min de parada a cada 30 dias). Modelo treinado só no treino; cortes vindos da validação.

| Plano | Corte | Alarmes / 100 | Cargas altas pegas / 100 | Perdidas / 100 | Alarmes falsos / 100 | Falsos por acerto | Min. de parada cobertos / 30 dias |
|---|---|---|---|---|---|---|---|
| A | não deixar passar | 43 | 12 | 3 | 31 | 2,6 | 155 de 185 |
| A | equilíbrio | 20 | 7 | 7 | 13 | 1,8 | 94 de 185 |
| A | evitar alarme falso | 9 | 3 | 11 | 5 | 1,5 | 60 de 185 |
| B | não deixar passar | 46 | 9 | 6 | 38 | 4,4 | 126 de 185 |
| B | equilíbrio | 26 | 6 | 9 | 20 | 3,6 | 95 de 185 |
| B | evitar alarme falso | 12 | 3 | 12 | 9 | 3,3 | 44 de 185 |

"Minutos cobertos" supõe que um alarme certo permite **evitar** a carga alta. É um teto do ganho, não uma promessa.

### O modelo para uso, medido no teste

Os `.joblib` da seleção foram re-treinados em treino + validação. O teste vem depois dos dois, então medir esses modelos nele é legítimo e não muda nenhuma escolha. (Corrige o README da seleção, que dizia que esse modelo "não tem nota própria".)

| Teste | PR-AUC | ROC-AUC | Recall c/ 20 alarmes por 100 |
|---|---|---|---|
| RF plano A, treinada só no treino | 0,354 | 0,77 | 0,48 |
| **RF plano A para uso** | **0,372** | 0,80 | **0,53** |
| RF plano B para uso | 0,198 | 0,62 | 0,27 |
| Regra: só histórico | 0,358 | 0,79 | 0,56 |

**Os limiares não se transferem.** O modelo para uso dá chances mais baixas (média 0,31 contra 0,35), então o mesmo limiar dispara menos alarmes:

| Corte (mesmo limiar) | Alarmes / 100: treinada só no treino | Alarmes / 100: para uso | Recall: só treino → para uso |
|---|---|---|---|
| não deixar passar (0,35) | 43 | 37 | 80% → 76% |
| equilíbrio (0,54) | 20 | 16 | 49% → 44% |
| evitar alarme falso (0,74) | 9 | 6 | 24% → 15% |

**Recomendação:** na operação, fixar o corte pelo **volume de alarmes** ("os N cestões de maior risco a cada 100") e recalcular o limiar equivalente a cada re-treino, com dados recentes.

### E as métricas de regressão?

MAE e RMSE (slides 24 a 29) medem erros em números. A pergunta da EVCOMX é sim ou não, então elas não se aplicam. Prever a duração da parada seria regressão, mas ela só existe depois da carga alta. O Brier score (calibração) é o parente mais próximo do RMSE na classificação.

### Conclusão: o desempenho é bom o suficiente para uso operacional? (Guia FE/FS §11.2)

A resposta depende de duas coisas que só a EVCOMX sabe: **quanto custa um alarme falso** (`k`) e **quantos alarmes o pátio aguenta**.

- **Contra a operação de hoje (sem alarme): ajuda, se `k` não for pequeno.** Com o corte "não deixar passar", o plano A pega 80% das cargas altas do teste e cobre ~155 dos ~185 min de parada a cada 30 dias, com ~43 alarmes a cada 100 cestões. No teste, a economia vai de ~1% (`k = 2`) a ~35% (`k = 8`).
- **Contra a regra simples do histórico: ainda não.** Nem o modelo treinado só no treino nem o modelo para uso se separam dela no teste. Antes de dizer que a composição do cestão agrega, é preciso entender maio/2026 e medir de novo.
- **A chance crua não vai para o pátio.** Faixas de risco com a taxa real de cada faixa.
- **O corte vem de `k` e se fixa pelo volume de alarmes**, não por um limiar fixo de chance.

A última célula do script monta essa conclusão com os números da rodada, para continuar certa se os dados mudarem.

### O que depende das respostas da EVCOMX

| Pergunta | O que muda aqui |
|---|---|
| 1. O que mudou em maio/2026? | Decide se re-treinamos só com o regime novo. É o caminho para saber se a composição do cestão supera a regra do histórico. |
| 2. Camadas de 10 kg são "gancho que não pegou"? | Se forem, todo o pipeline roda de novo; as features de peso e ciclo mudam. |
| 3. Qual erro é pior? Quanto custa um alarme falso? | Define `k` e, com ele, o corte. |
| **(não perguntada)** O resultado da corrida anterior está disponível na hora de prever? | Ficou ainda mais importante: o histórico é o sinal mais forte. Sem ele, só o plano B (PR-AUC 0,20 no teste). |
| **(nova)** O pátio dá conta de quantos alarmes por 100 cestões? | O corte "não deixar passar" dispara ~37–43. Se o teto for menor, o corte sobe. |

### Limitações

- **Validação e teste pequenos.** 75–92 cargas altas em cada um. Por isso todas as comparações vêm com IC 90% e bootstrap pareado por corrida.
- **O teste já tinha sido usado** na seleção. Aqui ele não escolhe nada: os cortes e o modelo vêm da validação.
- **Custo do FP desconhecido.** Todo o custo está em "alarmes falsos equivalentes" até a EVCOMX responder.
- **"Minutos cobertos" é um teto**: supõe que o alarme permite evitar a carga alta.
- **Notebook.** O `.ipynb` foi gerado executando as células do `.py` em ordem; o conteúdo é o mesmo.
- **Versão do scikit-learn.** 1.9.1, a mesma da seleção (as conferências dependem disso).
