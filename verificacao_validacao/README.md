## Verificação e Validação — Predição de Carga Alta (EVCOMX)

Segue a **Aula 07 do bootcamp** (Verificação e Validação de Modelos) e o **Guia de Modelagem Preditiva da EVCOMX** (§6), e fecha o caminho `EDA/ → feature_selection/ → selecao_algoritmos/ → analise_modelos/`.

> **Rodada de out/2026 (corte de 2.000 kg).** A EVCOMX respondeu a pergunta 2 do e-mail: camadas de 10 kg e 100 kg também podem ser desconsideradas, com corte em 2.000 kg. A EDA passou a remover todas as camadas abaixo de 2.000 kg e **todas as etapas rodaram de novo**. Esta verificação conferiu a rodada nova (31 conferências, nenhuma falha) e ganhou uma seção, a **5c-bis**, que mede o quanto o modelo depende do corte. Os números abaixo são desta rodada.

A aula separa duas perguntas:
- **Verificação:** o modelo foi construído e executado como dissemos?
- **Validação:** ele funciona em dados novos, e dá para confiar nele fora do conjunto original?

**Esta etapa não escolhe nada.** Algoritmo, features e limiares continuam os do `selecao_algoritmos/modelo_final.json`. Muito do que a aula pede já tinha sido feito nas etapas anteriores. Aqui isso é **conferido automaticamente**, sem refazer, e o que faltava é acrescentado.

### Como rodar

1. As quatro etapas anteriores, em ordem: `EDA/transforma_dataset.py`, `feature_selection/feature_selection.py`, `selecao_algoritmos/selecao_algoritmos.py` e `analise_modelos/analise_modelos.py`.
2. `verificacao_validacao/verificacao_validacao.py` (células `#%%`) ou `verificacao_validacao.ipynb` (mesmo conteúdo, já executado). Leva ~4 min.

O código acha a raiz do projeto sozinho. Ele **para com erro** se algum número das etapas anteriores não bater (rótulos, chances, contas do teste, split). Foi o que aconteceu com a resposta à pergunta 2 do e-mail: os dados mudaram, tudo rodou de novo e as conferências passaram.

### Arquivos gerados

| Arquivo | O que é |
|---|---|
| `checklist_verificacao.csv` | As 31 conferências (slide 8 + Guia de Modelagem §6.1), com resultado e detalhe |
| `contagens_por_etapa.csv` | Cestões e corridas em cada etapa (consistência dos joins) |
| `walk_forward_mensal.csv` | Backtesting: treina até o mês t, prevê o mês t+1 |
| `robustez_erro_pesagem.csv` | Quanto as previsões mudam com erro de pesagem de 2%, 5% e 10% |
| `sensibilidade_corte_peso.csv` | O modelo com as camadas cortadas em 0, 100, 1.000 e 2.000 kg (seção 5c-bis) |
| `validacao_dominio.csv` | As 5 perguntas de domínio do Guia, com evidência e veredito |
| `validacao_negocio.csv` | Ganho líquido em minutos de forno, por estratégia e cenário |
| `gap_treino_cv_validacao_teste.csv` | PR-AUC e ROC-AUC de cada modelo no treino, na CV temporal, na validação e no teste |
| `curva_complexidade.csv` | Treino × CV variando a profundidade da árvore e a folha da Random Forest |
| `sensibilidade_historico.csv` | O modelo com o histórico como está, com vazamento e pulando a corrida anterior |
| `variacao_por_semente.csv` | PR-AUC na validação com 20 sementes diferentes |
| `uso_validacao_e_teste.csv` | Cada vez que a validação e o teste foram olhados no projeto |
| `tipos_de_validacao_cruzada.csv` | Quatro tipos de CV contra a nota real na validação |
| `drift_das_features.csv` | PSI, KS e correlação com o alvo de cada feature, por partição |
| `criterios_aceitacao.csv` | Os critérios de aceitação propostos, aplicados à validação e ao teste |
| `monitoramento_simulado.csv` | O painel de monitoramento em janelas de 30 dias |
| `registro_experimento.json` | Dados (com hash SHA-256), partições, modelo, avaliação e ambiente |
| `requirements_rodada.txt` | Versões das bibliotecas da rodada (gerado de novo a cada execução) |
| `graficos/` | 8 figuras para a apresentação |

### O que já estava feito (e aqui só foi conferido)

| A aula pede | Onde já estava | Conferência |
|---|---|---|
| Treino, validação e teste (slides 14 a 16) | Split 70/15/15 temporal por corrida (feature selection §1) | 1.3e, 1.3f |
| Dados agrupados e séries temporais (slide 17) | Cestões da mesma corrida juntos; ordem de tempo | 1.3e, 1.3f, 1.3g |
| Pipeline ajustado só no treino (slides 21 e 22) | `Pipeline` com mediana e padronização em todas as etapas | 1.3a, 1.3b, 1.3c |
| Variável pós-evento (slide 20) | Duração da parada, `carga_alta_c1` e data fora do modelo; histórico com `shift(1)` | 1.4a, 1.4b, 1.4e |
| Validação cruzada com média e dispersão (slides 24 a 26) | CV temporal na seleção e na análise | seção 4 |
| Baseline (slide 31) | Maioria, sorteio e regras de uma feature (análise §3) | 1.7a, 1.7b |
| Métricas certas e o caso dos 99% de acurácia (slide 36) | PR-AUC; "sempre normal" tem 82% de acurácia (análise §2) | 1.5a a 1.5d |
| Custo dos erros (slide 35) | Razão `k` e os três cortes (análise §7) | não refeito |

---

### 1. Checklist técnico (slide 8)

**31 conferências: 27 OK, 4 com ressalva, nenhuma falha.** O modelo foi construído como as etapas anteriores dizem:
- os rótulos batem com o arquivo bruto da EVCOMX, cestão a cestão;
- as chances re-treinadas aqui são idênticas às salvas pela seleção;
- a mediana de preenchimento e a padronização vêm só do treino, e a conferência distingue isso de "todos os dados" (as medianas diferem em 3 das 6 features);
- o histórico recalculado do arquivo bruto com `shift(1)` é igual ao do dataset;
- as contas do teste (recall, precisão, alarmes, acurácia, PR-AUC, ROC-AUC) foram refeitas à mão, com a matriz de confusão montada à mão, e batem com `resultado_teste.csv`.

As quatro ressalvas (a quinta, 1.5b, foi corrigida na seleção: veja abaixo):

| Item | Ressalva | Tamanho |
|---|---|---|
| **1.3d** | A importância por **permutação**, 1 dos 6 votos da feature selection (§7), é medida na **validação**, não no treino. A validação também escolheu quantas features ficam (§10). | Não vaza para o teste, mas deixa a nota da validação otimista (seção 3). |
| **1.3g** | 2 de 3.501 cestões (O4931 c2, O6571 c2) caem depois do início da partição seguinte: o cestão 2 tem hora registrada muito depois do cestão 1. | Desprezível no modelo, mas aponta um problema de dado: 90 corridas com o cestão 2 antes do 1 e 18 com mais de 1 dia entre eles (mediana: 29 min). |
| **1.4c** | **Timing do histórico.** Se o resultado da corrida anterior não estiver disponível na hora de prever, o histórico precisa pular uma corrida. | Validação cai de 0,43 para 0,40 (CV de 0,41 para 0,39). Perda pequena: o modelo sobrevive à resposta "não". |
| **1.7b** | A Random Forest contra a regra só do histórico: vantagem real na validação, empate no teste (PR-AUC). | Já era o achado principal da análise; vai para a apresentação. |

**1.5b, corrigida.** O `modelo_final.json` guardava os limiares com 4 casas, e 3 cestões da validação ficavam entre o valor arredondado e o exato. A seleção agora salva os limiares inteiros (no `.json` e dentro do `.joblib`), e a conferência lê o CSV sem perder o último dígito: OK.

**Conferências do Guia de Modelagem (§6.1), itens 1.8:**

| Item | O que o Guia pede | Resultado |
|---|---|---|
| **1.8a** | Teste de sanidade: embaralhar o alvo | Com a carga alta do treino embaralhada (10 sorteios), a PR-AUC na validação cai para **0,19** em média (de 0,16 a 0,23), contra 0,18 do chute e 0,43 do modelo real. Na CV temporal, 0,16 (chute 0,16). Não há vazamento escondido no pipeline. |
| **1.8b** | Testes unitários da agregação | Peso total = soma das camadas e volume idem (diferença máxima 1e-14); leve + misto + pesado = 1; soma dos 18 materiais = 1; volume > 0 em todas as camadas; densidade = peso ÷ volume. |
| **1.8c** | Consistência dos joins | 3.501 cestões em 1.799 corridas no bruto, na base limpa e no dataset de modelagem. Nenhum cestão inteiro sai na limpeza (saem só camadas abaixo de 2.000 kg, peso zero incluído). O `VariavelResposta.csv` tem 97 cestões sem nenhuma camada registrada: sem receita, não há o que prever. |

A auditoria de vazamento (1.4), a reprodutibilidade (1.6) e a estabilidade por prova e por semente (1.6b e seção 2), também pedidas no §6.1, já estavam no checklist.

**O que acontece com vazamento de verdade** (1.4c): se o histórico incluísse a corrida atual, a PR-AUC na CV subiria de 0,41 para **0,50**. É o "resultado excelente que pode estar errado" do slide 4, e a conferência 1.4b garante que o projeto não faz isso.

**Reprodutibilidade** (1.6): a mesma semente dá as mesmas chances (diferença de 2e-16, só pela soma em paralelo). Com 20 sementes diferentes, a PR-AUC da Random Forest na validação vai de **0,41 a 0,43** (dp 0,004). A semente 42 (0,427) está na ponta de cima. Diferenças de um ou dois centésimos entre modelos estão dentro desse ruído.

### 2. Treino bom não garante desempenho futuro (gráficos 01 e 02)

| PR-AUC | treino | CV temporal | validação | teste |
|---|---|---|---|---|
| Regressão logística | 0,47 | 0,45 ± 0,16 | **0,29** | — |
| Árvore de decisão | 0,45 | 0,37 ± 0,09 | 0,30 | — |
| **Random Forest (plano A)** | **0,84** | 0,41 ± 0,11 | 0,43 | 0,39 |
| Gradient Boosting | 0,74 | 0,40 ± 0,08 | 0,39 | — |
| Random Forest (plano B, ajustada) | 0,51 | 0,35 ± 0,08 | 0,32 | 0,23 |
| *chute (taxa de carga alta)* | *0,16* | *0,15* | *0,18* | *0,15* |

O teste só aparece para os modelos escolhidos, cujas notas já tinham sido publicadas. Não há motivo para olhar o teste com modelos que não vão ser usados.

- **treino − CV** (o modelo decora?): a Random Forest acerta quase tudo do treino (0,84) e cai para 0,41 fora dele. É esperado em florestas; o que importa é a nota fora do treino. A logística, a árvore rasa e o plano B (agora uma floresta rasa, `max_depth=4`) quase não decoram.
- **CV − validação** (o regime novo derruba?): depende do algoritmo. A **logística, melhor na CV, vira a pior** na validação (0,45 → 0,29). Random Forest e Gradient Boosting aguentam a passagem. É por isso que a seleção decidiu pela validação, não pela CV.
- **validação − teste:** mistura o otimismo da validação (ela escolheu), um período diferente e o ruído de ~75–90 cargas altas por partição.

**Curva de complexidade (slide 11, gráfico 02).** As duas curvas têm a forma da aula: simples demais erra em tudo; complexo demais (folha 1) acerta 100% do treino sem ganhar nada fora dele. Os modelos escolhidos estão na região boa. Na CV, uma folha de 50 daria 0,44 contra 0,41 da folha 5. A diferença é menor que o desvio entre provas (0,11), e a seleção já mostrou que ajustar na CV não melhorou a validação. **Não trocamos nada**: seria escolher olhando de novo.

### 3. O teste não é uma segunda validação (slides 15 e 29)

O registro completo está em `uso_validacao_e_teste.csv`. Resumo:

- **A validação decidiu:** 1 voto da feature selection (permutação), o número de features, o algoritmo, a versão, o conjunto e os três cortes.
- **O teste foi olhado em 12 momentos** (feature selection §12 e §13, seleção na 1ª rodada e na conferência, análise §8 a §11, e, na revisão pelo Guia de Modelagem, a calibração, a tabela única, os segmentos e erros, o walk-forward, a robustez e as validações de domínio e de negócio; e agora a sensibilidade ao corte de peso e a rodada inteira refeita com o corte de 2.000 kg). **Nenhuma escolha foi feita olhando o teste diretamente.** Mas a 1ª rodada da seleção viu o teste antes de mudar a lista de candidatos (a nota de "Transparência" no README da seleção). E a rodada de out/2026 refez tudo depois de o teste já ser conhecido; o que mudou foi a limpeza (o corte veio da EVCOMX, não do teste), e as escolhas seguiram as mesmas regras automáticas (cotovelo, escada, limiares na validação).

| Plano | Validação (onde foi escolhido) | Teste | Queda |
|---|---|---|---|
| A (com histórico) | 0,43 | 0,39 | 0,04 |
| B (sem histórico) | 0,32 | 0,23 | 0,09 |

**Leitura:** as duas notas são **estimativas boas, não provas independentes**. A validação aninhada do slide 29 resolveria o otimismo da validação, mas com ~90 cargas altas não cabe. **O teste de verdade é o próximo lote**: pedir à EVCOMX os dados depois de 04/08/2026 e rodar o modelo para uso uma única vez, com os critérios da seção 6 fixados **antes**.

### 4. Validação cruzada: limites e cuidados (slide 27, gráfico 03)

Quatro jeitos de fazer a CV, no mesmo treino, com os mesmos modelos:

| CV | Otimismo médio (CV − validação) | Desvio médio entre provas | Onde foi usada |
|---|---|---|---|
| `cv=5` da aula (sem embaralhar) | +0,08 | 0,10 | slide 26 |
| embaralhada, por corrida | **+0,10** | **0,02** | feature selection (L1, Elastic Net, RFE) |
| embaralhada, sem agrupar | +0,10 | 0,06 | nenhuma (pior caso) |
| **temporal** | **+0,06** | 0,11 | seleção e análise |

- **Embaralhar dá falsa segurança:** as CVs embaralhadas são as mais otimistas **e** as de menor desvio. Parecem precisas e erram mais. O histórico dos cestões de treino de cada prova contém as cargas altas das corridas que estão sendo previstas.
- **A temporal é a menos otimista**, com desvio grande: cada prova é um período diferente.
- **Nenhuma CV ordena bem os algoritmos.** Em todas, a logística fica perto das melhores (0,45–0,47); na validação, é a pior (0,29). Nenhuma CV feita só no regime antigo vê uma mudança que acontece depois. É o que o slide 27 avisa: a CV "não substitui divisão temporal quando o futuro é o cenário de uso".
- **Na feature selection**, a CV embaralhada só escolheu a penalidade do L1/Elastic Net e o número de features do RFE. Ela é otimista para todos os candidatos ao mesmo tempo, então o efeito na ordem das features deve ser pequeno. Fica como limitação; não refizemos.

### 5. Os dados mudaram? (slides 33 e 34, gráficos 04 e 05)

**PSI** compara a distribuição de cada feature com a do treino: abaixo de 0,10 estável; 0,10–0,25 moderada; acima de 0,25 grande. A **correlação de Spearman** com a carga alta, em cada partição, mostra se a feature continua dizendo a mesma coisa (concept drift).

| Feature | PSI validação | PSI teste | Correlação com a carga alta (treino → validação → teste) | Relação |
|---|---|---|---|---|
| `prop_peso_pesado` | **0,96** | **1,53** | −0,31 → −0,11 → −0,07 | enfraquece |
| `densidade_media` | **1,09** | **1,30** | −0,31 → +0,01 → −0,10 | enfraquece |
| `peso_por_ciclo_t` | **0,40** | **0,69** | +0,25 → +0,13 → +0,08 | enfraquece (o script diz "se mantém" por 0,001: a correlação cai à metade na validação e a um terço no teste) |
| `taxa_ca_ult10` | 0,01 | **0,58** | +0,22 → +0,24 → +0,38 | se mantém |
| `taxa_ca_ult5` | 0,01 | 0,24 | +0,24 → +0,26 → +0,38 | se mantém |
| `prop_PACB` | 0,09 | 0,00 | +0,10 → +0,05 → +0,18 | fraca já no treino |

As três features que saíram com o corte de 2.000 kg (`max_bloco_leve_m3`, `prop_leve_base` e `prop_PAEP`) eram justamente as que mudavam de sinal ou eram fracas já no treino na rodada anterior.

- **Os cestões do regime novo são outros cestões:** menos pesado, menos densos e com mais toneladas por ciclo. É o que a feature selection (§13) viu nas médias, agora medido feature a feature.
- **E a composição deixou de explicar a carga alta:** no treino, mais pesado e mais denso = menos carga alta (≈ −0,3); depois de maio, essa relação quase some. Isso explica por que as regras de composição sozinhas ficam no nível do chute na análise.
- **O histórico é o sinal mais estável:** a relação com a carga alta se mantém e até fica mais forte no teste. A distribuição dele muda no teste porque há fases mais longas de forno "calmo": 35% dos cestões do teste não têm nenhuma carga alta nas 10 corridas anteriores, contra 15% no treino e na validação.
- **A taxa de carga alta quase não mudou** (0,16 / 0,18 / 0,15). Mudou **quem** dá carga alta, não **quanto**.

Isso reforça a pergunta 1 do e-mail: se a mudança de maio for permanente, é preciso re-treinar com o regime novo, e as features de composição precisam provar de novo que ajudam.

### 5b. Backtesting walk-forward mês a mês (Guia §6.2, gráfico 07)

"Treinar até o mês t, testar no mês t+1, avançar." A cada mês, o plano A é re-treinado com tudo o que já aconteceu e prevê o mês seguinte. Nada é escolhido aqui.

| Mês testado | Cargas altas | Chute | Modelo | Regra do histórico | Regra de volume | Recall 40/100: modelo × regra |
|---|---|---|---|---|---|---|
| jan/2026 | 43 | 0,15 | **0,35** | 0,21 | 0,17 | 0,74 × 0,58 |
| fev/2026 | 34 | 0,15 | **0,47** | 0,19 | 0,13 | 0,76 × 0,56 |
| mar/2026 | 30 | 0,16 | **0,31** | 0,23 | 0,20 | 0,70 × 0,63 |
| abr/2026 | 55 | 0,15 | **0,29** | 0,23 | 0,19 | 0,55 × 0,49 |
| mai/2026 | 23 | 0,16 | **0,25** | 0,18 | 0,16 | 0,57 × 0,49 |
| jun/2026 | 78 | 0,17 | 0,44 | 0,39 | 0,15 | 0,72 × 0,71 |
| jul–ago/2026 | 60 | 0,16 | **0,47** | 0,37 | 0,15 | 0,82 × 0,82 |

(PR-AUC no mês, salvo a última coluna.)

- **O modelo re-treinado todo mês fica acima do chute nos 7 meses** e, em PR-AUC, acima da regra do histórico nos 7.
- **Mas em junho e julho, no orçamento de 40 alarmes a cada 100, empata com a regra** (0,72 × 0,71 e 0,82 × 0,82). Nesses meses o histórico ficou muito mais forte (PR-AUC da regra subiu de ~0,2 para ~0,4), e a composição do cestão quase não acrescenta.
- A regra de volume do Guia fica no nível do chute em todos os meses.
- **Leitura:** re-treinar com frequência ajuda (o modelo treinado só até abril empatava com a regra no teste), mas no período mais recente a vantagem sobre a regra é pequena. Bate com a análise e com a seção 5.

### 5c. Robustez a erros de pesagem (Guia §6.2)

As 4 features de composição foram refeitas a partir das **camadas**, com ruído multiplicativo no peso de cada camada (o volume acompanha), 10 sorteios por nível. O recálculo sem ruído reproduz o dataset exatamente (diferença 0). Teste, modelo treinado só no treino, corte "não deixar passar":

| Erro de pesagem (desvio) | Mudança média na chance | Cestões que trocam de lado no corte (média / pior sorteio) | PR-AUC (média) |
|---|---|---|---|
| 2% | 0,012 | 1,5% / 2,5% | 0,389 |
| 5% | 0,023 | 3,0% / 3,7% | 0,391 |
| 10% | 0,037 | 5,5% / 6,6% | 0,402 |
| *sem ruído* | — | — | *0,391* |

**O modelo é robusto a erro de balança.** Mesmo com 10% de erro em cada camada, a chance muda ~0,04 e 5–7% dos cestões trocam de lado no corte (são os que já estavam colados no limiar). A PR-AUC não muda. Com só 4 features de composição (antes, 7), cada uma pesa um pouco mais, e a chance ficou um pouco mais sensível que na rodada anterior (0,027 com 10%).

### 5c-bis. Sensibilidade ao corte de peso da camada (resposta da EVCOMX)

A EVCOMX disse que camadas de 10 kg e 100 kg também podem ser desconsideradas e que "dificilmente temos camadas com menos de 2000kg". A EDA usa esse corte. Mas nem tudo abaixo de 2 t parece gancho vazio: entre 100 kg e 2 t a mediana é ~1,6 t. Então refizemos as 4 features de composição **direto do arquivo bruto** com quatro cortes, re-treinamos o plano A só no treino e medimos. O corte de 2.000 kg refeito do bruto reproduz o dataset exatamente (diferença 7e-15). **Nada é escolhido aqui:** as features e os hiperparâmetros são os da rodada com 2.000 kg.

| Corte (camadas removidas) | Camadas removidas (cestões 1 e 2) | % do peso removido | Cestões com alguma feature diferente | PR-AUC validação | PR-AUC teste | Recall 40/100 validação | Recall 40/100 teste |
|---|---|---|---|---|---|---|---|
| só peso zero (versão anterior) | 2.020 | 0,0% | 58% | 0,449 | 0,386 | 0,72 | 0,77 |
| < 100 kg | 2.346 | 0,0% | 53% | 0,450 | 0,380 | 0,73 | 0,80 |
| < 1.000 kg | 2.732 | 0,2% | 47% | 0,454 | 0,382 | 0,74 | 0,76 |
| **< 2.000 kg (EVCOMX, em uso)** | 5.164 | 2,5% | — | 0,427 | 0,391 | 0,68 | 0,77 |

- **O corte quase não muda o modelo.** A PR-AUC varia 0,03 na validação e 0,01 no teste entre os quatro cortes, dentro do ruído (±0,01 só pela semente; IC 90% da validação ±0,08), mesmo com mais da metade dos cestões mudando alguma feature.
- **O corte de 2.000 kg é o pior na validação e o melhor no teste**, por pouco nos dois casos. Não há sinal de que ele esteja jogando fora informação útil, nem de que esteja ajudando: a decisão de usá-lo é de domínio (a palavra da EVCOMX), não de desempenho.
- **Limitação:** a feature selection não foi refeita para cada corte. Com o corte antigo, ela escolhia 9 features (as 6 de hoje mais `max_bloco_leve_m3`, `prop_leve_base` e `prop_PAEP`), com PR-AUC de 0,425 na validação e 0,354 no teste.

### 5d. Calibração (Guia §6.2)

Medida e aplicada na análise (seções 8 e 8b): as chances cruas estão infladas (Brier 0,182 no teste, pior que dizer sempre a taxa média, 0,125). **Platt ajustado na validação corrige: Brier 0,109 no teste, sem mudar a ordem dos cestões.** Calibrar dentro da CV temporal dá resultado parecido (0,115).

### 6. Critérios de aceitação e monitoramento (slides 34 e 35)

#### Critérios de aceitação

O slide 35 pede que "bom o suficiente" seja definido **antes** de ver o resultado final. Isso nunca tinha sido escrito. Como já é tarde para o teste atual, os critérios valem para **o próximo lote**; aplicá-los aqui é só uma ilustração. São uma **proposta do grupo**: o orçamento de alarmes e o recall mínimo são parâmetros no topo da célula, a confirmar com a EVCOMX.

| Critério | Regra | Validação | Teste (treinada só no treino) | Teste (modelo para uso) |
|---|---|---|---|---|
| C1 acima do chute com folga | limite baixo do IC 90% da PR-AUC > taxa de carga alta | sim (0,35 > 0,18) | sim (0,31 > 0,15) | sim (0,33 > 0,15) |
| C2 não perde para a regra do histórico | RF vence em ≥ 20% dos sorteios do bootstrap | sim (99%) | sim (78%) | sim (88%) |
| C3 recall no orçamento | recall com 40 alarmes a cada 100 ≥ 70% | **não (68%)** | sim (77%) | sim (81%) |
| C4 faixas de risco em ordem | taxa real cresce de "baixo" para "muito alto" | sim | sim | sim |
| **Mínimo para usar (C1 a C4)** | | **não** | **sim** | **sim** |
| C5 justifica o modelo (desejável) | RF vence a regra com vantagem real (≥ 95%) | sim | não | não |

Os critérios não foram ajustados para passar; o que falhou fica registrado:
- **C3 na validação:** com 40 alarmes, o modelo pega 68%. Para chegar a 76% são precisos ~47 alarmes. **O recall mínimo e o orçamento de alarmes competem**, e a EVCOMX precisa dizer qual pesa mais.
- **C4** passa nos três lotes nesta rodada. Na rodada anterior (corte só no peso zero), o modelo para uso trocava a ordem de "alto" e "muito alto". O modelo re-treinado continua dando chances mais baixas, então **fronteiras fixas em chance não se transferem bem entre versões do modelo**: depois de cada re-treino, as faixas devem ser recalculadas pelo volume (ex.: os 10% de maior risco = "muito alto").
- **C5** só passa na validação, onde o modelo foi escolhido.

#### Monitoramento (gráfico 06)

Simulamos o modelo em operação desde o começo da validação (plano A, treinado só no treino, corte "não deixar passar"), com uma janela a cada 30 dias:

| Janela | Cestões | Cargas altas | PSI máx. (feature) | Alarmes / 100 | Recall | Precisão | PR-AUC × regra | Gatilhos |
|---|---|---|---|---|---|---|---|---|
| 25/04–25/05 | 158 | 21 | **0,81** (`prop_peso_pesado`) | 38 | **0,57** | 0,20 | 0,24 × 0,15 | G1, G3 |
| 26/05–24/06 | 360 | 74 | **1,96** (`densidade_media`) | 52 | 0,82 | 0,33 | 0,46 × 0,40 | G1 |
| 24/06–04/08 | 510 | 72 | **1,50** (`prop_peso_pesado`) | 38 | 0,75 | 0,28 | 0,39 × 0,35 | G1 |

- **O monitoramento teria avisado já em maio** que os cestões mudaram (G1). É exatamente o papel dele.
- **Na primeira janela, o recall também caiu** (57%, abaixo do mínimo de 70%: G3), coincidindo com a mudança de maio. Mas são só 21 cargas altas, e nas janelas seguintes o recall voltou a 75–82%. Pelo plano abaixo, um G3 isolado não dispara re-treino (são precisas 2 janelas seguidas).
- **Volume de alarmes e vantagem sobre a regra se seguraram** (G2 e G4 não dispararam): o histórico, estável, segura o modelo.
- **Um G1 sempre ligado perde a utilidade.** Depois do re-treino, a referência do PSI passa a ser o novo treino.

**Plano proposto.** O rótulo chega no mesmo dia, então dá para medir desempenho de verdade, não só a distribuição:

| O quê | Frequência | Gatilho | Ação |
|---|---|---|---|
| PSI das 6 features contra o treino do modelo em uso | semanal | > 0,25 em alguma feature | investigar com a EVCOMX: processo, receita ou coleta? |
| Alarmes a cada 100 cestões | semanal | fora de ±15 do esperado | revisar o limiar pelo volume de alarmes |
| Recall e precisão no corte | mensal (~380 cestões, ~55 cargas altas) | recall < mínimo combinado | reavaliar o corte e o modelo |
| PR-AUC do modelo × regra do histórico | mensal | modelo ≤ regra | o modelo não agregou no mês; se repetir, voltar à regra |
| Taxa real por faixa de risco | mensal | faixas fora de ordem | recalcular as fronteiras das faixas |
| Re-treino | G1 ou G3 por 2 janelas seguidas, ou a cada 3 meses | — | re-treinar com dados recentes, recalcular os limiares pelo volume e rodar esta verificação de novo |

### 6.3 Validação de domínio: o modelo faz sentido físico? (Guia §6.3)

| Pergunta do Guia | Evidência esperada | O que encontramos | Veredito |
|---|---|---|---|
| O risco aumenta com o volume? | curva crescente | O volume não entrou no modelo e não se relaciona com a carga alta (r = −0,04 no treino; taxa por tercil de volume 20% / 12% / 16%). O risco previsto até cai um pouco com o volume (Spearman −0,10). O que carrega a física é a densidade e o pesado: a dependência parcial cai com `prop_peso_pesado` (−0,21) e com `densidade_media` (−0,14). | o volume não diz nada; a física aparece pela densidade e pelo pesado |
| Leve em excesso eleva o risco? | contribuição positiva | Com o corte de 2.000 kg, a única feature de "leve/volumoso" que ficou no modelo é `prop_PACB`, com dependência parcial positiva mas quase nula (+0,01). O "cestão fofo" aparece pela `densidade_media` (−0,14) e pela `prop_peso_pesado` (−0,21). | coerente no sentido, fraco no tamanho |
| Violar a ordem das camadas eleva o risco? | efeito consistente com a regra | No treino, quem foge da regra (fundo ou topo não leve) tem **menos** carga alta (12% × 17%); depois de maio, **mais** (21% × 13%). Pesado no fundo ou no topo: 5% × 16% no treino. Nenhuma flag teve votos para entrar no modelo. | inverte entre regimes; rever com dados novos |
| Casos extremos se comportam bem? | cestões "impossíveis" com risco alto | Os 96 cestões mais extremos (5% menos densos ou 5% com mais toneladas por ciclo, validação + teste) recebem risco 0,59 contra 0,40 no geral, e 81% dão alarme (geral: 43%). A taxa real neles é 20%, um pouco acima da média (16%). | coerente, com folga pequena: no regime novo esses cestões têm só um pouco mais de carga alta |
| O modelo concorda com o especialista? | revisão com o pátio | Os 10 alarmes falsos mais "certos" e as 10 cargas altas com menor risco estão em `analise_modelos/casos_para_especialista.csv`. | pendente (EVCOMX) |

**Leitura.** O modelo **não contradiz a física**: todos os sentidos das features batem com o que se espera de um cestão alto. O que os dados recentes mostram é que **a física da composição explica menos depois de maio** (casos extremos com risco bem acima da média e taxa real só um pouco acima; violação da ordem trocando de sinal). É o mesmo concept drift da seção 5, visto pelo lado do domínio, e reforça a pergunta 1 do e-mail.

### 6.4 Validação de negócio: vale a pena? (Guia §6.4, gráfico 08)

A conta do Guia: `ganho líquido = VP × taxa de sucesso × duração média da parada − FP × custo do alarme falso`. A duração média sai dos dados (**4,0 min**, treino). A taxa de sucesso da intervenção, o custo do alarme falso e o valor do minuto **só a EVCOMX sabe**. Por isso o custo do alarme está em **minutos de forno equivalentes**, há três cenários e uma grade de sensibilidade, e `CUSTO_MINUTO_RS` fica vazio no topo da célula (preenchido, as tabelas saem em reais).

| Cenário | Taxa de sucesso | Custo de 1 alarme falso |
|---|---|---|
| conservador | 30% | 2 min de forno |
| base | 50% | 1 min |
| otimista | 80% | 0,5 min |

Ganho líquido no teste, em **minutos de forno a cada 30 dias** (sem nada, são ~223 min de parada a cada 30 dias):

| Estratégia | Alarmes /100 | Conservador | Base | Otimista |
|---|---|---|---|---|
| modelo, corte "não deixar passar" | 38 | −157 | −19 | +83 |
| modelo, corte "equilíbrio" | 27 | −95 | +1 | +77 |
| modelo, mesmo nº de alarmes da regra do histórico | 32 | −123 | −7 | +81 |
| regra de volume (> 56 m³) | 53 | −310 | −114 | +8 |
| regra do histórico (2+ cargas altas nas últimas 5) | 32 | −111 | +4 | **+95** |

Por ano, o melhor caso (regra do histórico, otimista) dá ~1.150 min; o modelo no corte "não deixar passar", ~1.015 min.

- **O ganho depende de dois números da EVCOMX.** No cenário conservador nenhuma estratégia compensa; no otimista, todas compensam (menos a regra de volume, que empata). A linha de empate no gráfico 08 mostra a fronteira.
- **Com muitos alarmes, o custo dos falsos come o ganho.** O corte "não deixar passar" só compensa se o alarme falso for barato (até ~0,8 min no cenário base). O corte "equilíbrio" é o mais seguro para o modelo.
- **Contra a regra do histórico, com o mesmo número de alarmes (32 a cada 100), a regra ganha em toda a grade:** ela acerta 57 cargas altas no teste e o modelo, 52. É o mesmo resultado da análise (recall com o mesmo orçamento: desvantagem real no teste).
- **Leitura para a EVCOMX:** no período de teste, um alarme simples baseado no histórico recente do forno entregaria o mesmo ou mais que o modelo. O modelo só se justifica se a composição do cestão voltar a explicar a carga alta (entender maio/2026) ou se o histórico não estiver disponível na hora de prever (aí o plano B é a única opção com modelo).

### 7. Registro do experimento (slide 30)

O que o slide 30 pede já existia, mas espalhado. Agora fica em **um arquivo**, gerado a cada rodada:

| Slide 30 | Onde |
|---|---|
| Dados: fonte, período, filtros, versão | `registro_experimento.json` → `dados`, com o **SHA-256** dos 4 CSVs brutos e do dataset de features. Se um CSV mudar, o hash muda. |
| Partições: estratégia e semente | `registro_experimento.json` → `particoes` |
| Modelo: algoritmo, atributos, hiperparâmetros | `registro_experimento.json` → `modelo` (cópia do `modelo_final.json`) |
| Avaliação: métricas, dispersão, baseline | `registro_experimento.json` → `avaliacao` |
| Ambiente | `requirements_rodada.txt` e `registro_experimento.json` → `ambiente` |

O `requirements.txt` na raiz do projeto tem as mesmas versões. As conferências 1.2a e 1.5a dependem do scikit-learn 1.9.1: com outra versão, as chances podem mudar um pouco e o script avisa.

---

### Conclusão: podemos confiar no modelo? (slides 37 e 38)

| Ideia da aula (slide 38) | Resposta |
|---|---|
| **Correção técnica:** foi construído como dissemos? | **Sim.** 31 conferências, nenhuma falha, incluindo o alvo embaralhado e os testes de agregação do Guia, também depois de refazer tudo com o corte de 2.000 kg. As 4 ressalvas são pequenas e estão documentadas. |
| **Generalização:** o treino prova o futuro? | **Não, e o projeto já sabia disso.** A RF decora o treino (0,84), mas fora dele fica em 0,39–0,43, sempre acima do chute. A mudança de maio derrubou a logística, não a RF. |
| **Integridade:** há vazamento? | **Não há vazamento para o teste.** Mas o teste foi olhado antes do fim e a validação escolheu quase tudo: as notas são estimativas, não provas. O ponto sensível é o timing do histórico, e a perda, se ele pular uma corrida, é pequena (0,43 → 0,40). |
| **Confiança:** protocolo, baseline, reprodutibilidade, monitoramento | Protocolo e baseline já existiam; **reprodutibilidade, critérios de aceitação e plano de monitoramento foram acrescentados aqui.** |
| **Robustez e domínio** (Guia §6.2 e §6.3) | Robusto a erro de pesagem e ao corte de peso da camada (0 a 2.000 kg: a PR-AUC muda menos de 0,03); walk-forward acima do chute em todos os meses; sentidos das features coerentes com a física, mas a composição explica menos depois de maio. |
| **Negócio** (Guia §6.4) | Depende do custo do alarme falso e da taxa de sucesso da intervenção. Com o mesmo número de alarmes, a regra do histórico entrega mais que o modelo no teste. |

**O que dizer à EVCOMX:** o modelo é tecnicamente correto, robusto a erro de balança e ao corte das camadas pequenas, e melhor que o chute em todos os períodos, mas os cestões mudaram em maio e a composição perdeu poder de explicação. Hoje, o modelo **empata com a regra simples do histórico** no teste (em PR-AUC; o modelo para uso chega a "vantagem provável") e perde para ela com o mesmo número de alarmes. Re-treinado todo mês, ele fica um pouco à frente em PR-AUC, mas empata no orçamento de alarmes nos meses mais recentes. Para confiar nele de verdade, falta: (1) entender maio, (2) fixar os critérios com a EVCOMX, e (3) medir uma única vez num lote novo, com o monitoramento ligado.

### O que depende da EVCOMX

| Pergunta | O que muda |
|---|---|
| 1. O que mudou em maio/2026? | Decide o re-treino com o regime novo e a referência do monitoramento (G1). |
| ~~2. Camadas de 10 kg são "gancho que não pegou"?~~ | **Respondida:** sim, com corte em 2.000 kg. Todas as etapas rodaram de novo e esta verificação conferiu a rodada (nenhuma falha); a seção 5c-bis mostra que o corte quase não muda o modelo. |
| 3. Qual erro é pior? Quantos alarmes o pátio aguenta? | Fixa o `ORCAMENTO` e o `RECALL_MIN` dos critérios de aceitação (C3), **antes** do próximo lote. |
| **(não perguntada)** O resultado da corrida anterior está disponível na hora de prever? | Se não estiver, o histórico pula uma corrida: validação 0,43 → 0,40. |
| **(nova)** Podemos receber os dados depois de 04/08/2026? | É o teste limpo que falta (seção 3). |
| **(nova)** Quanto vale um minuto de forno parado? Quanto custa um alarme falso? Que fração das cargas altas o pátio consegue evitar quando recebe o alarme? | Preenche `CUSTO_MINUTO_RS` e os cenários da validação de negócio (6.4). |
| **(nova)** Os cestões de `casos_para_especialista.csv` fazem sentido para o pátio? | É a validação com o especialista que o Guia pede (6.3). |

### Sugestões para as etapas anteriores (não alteradas aqui)

- ~~**Seleção:** salvar os limiares no `modelo_final.json` sem arredondar (1.5b).~~ Feito.
- ~~**Feature selection:** registrar no README que a permutação foi medida na validação (1.3d).~~ Feito (README da feature selection, "Armadilhas do Guia"). A CV interna embaralhada continua como limitação (seção 4).
- **EDA:** investigar as 18 corridas com mais de 1 dia entre o cestão 1 e o 2, e as 90 com o cestão 2 antes do 1 (1.3g). Pode ser efeito da hora estimada (`dt_hora_consumo_estimada`).

### Limitações

Do Guia de Modelagem (§6.6), que valem para todo o projeto:
- **Correlação não é causalidade.** O modelo prevê risco; não prova que mudar a camada elimina a carga alta.
- **As densidades do dicionário são valores médios.** A densidade real da sucata varia por lote; o volume estimado herda esse erro.
- **O registro do evento pode depender do operador.** Paradas curtas podem ser subnotificadas.
- **Período histórico limitado** (um ano) e uma mudança de processo no meio (maio/2026).
- **Poucos eventos por segmento.** Com 23 a 86 cargas altas por mês ou segmento, as métricas por fatia têm intervalos largos.
- **Faltam variáveis** que o pátio conhece: umidade, sucata molhada, forma física da sucata. As cargas altas perdidas, com "cara de cestão bom", sugerem algo fora dos dados.

Desta etapa:

- **Os critérios de aceitação foram escritos depois de ver os resultados.** Aplicados aqui, são ilustração; só valem como critério no próximo lote.
- **Validação e teste pequenos** (75–92 cargas altas cada). Por isso há IC 90%, bootstrap pareado e variação por semente.
- **PSI com features discretas.** Para `taxa_ca_ult5/10` (valores 0; 0,1; 0,2…), as faixas do PSI ficam nos próprios valores. O número é comparável entre partições, mas os limites 0,10/0,25 foram pensados para variáveis contínuas.
- **O monitoramento simulado** usa o modelo treinado só no treino e janelas de 30 dias. A primeira janela tem só 158 cestões e 21 cargas altas.
- **Ambiente.** A rodada salva foi executada em Linux, com as versões de `requirements_rodada.txt`. Ao rodar no Windows, o arquivo é regenerado com as versões da máquina; as conferências avisam se os números mudarem.
- **Notebook.** O `.ipynb` foi gerado executando as células do `.py` em ordem; o conteúdo é o mesmo.
