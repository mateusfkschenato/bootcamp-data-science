## Interpretação do Modelo (Explainable AI) — Predição de Carga Alta (EVCOMX)

Segue a **Aula 08 do bootcamp** (Interpretação de Modelos) e vem depois de `verificacao_validacao/`: primeiro se valida, depois se interpreta.

**Esta etapa não escolhe nada e não muda o modelo.** O modelo explicado é o que vai para uso: a Random Forest do plano A (`selecao_algoritmos/modelo_final_planoA.joblib`, treinada em treino + validação, 6 features). As explicações são feitas no **teste** (jun–ago/2026, 515 cestões, 75 cargas altas), que esse modelo nunca viu, como pede a aula. Olhar o teste aqui não decide nada.

Parte do que a aula pede já estava em `analise_modelos/` (seção 11d): dependência parcial + ICE, contribuições por caminho das árvores (Saabas) e as regras para o pátio. Esta pasta completa o que faltava.

### Como rodar

1. As cinco etapas anteriores, em ordem (a última é `verificacao_validacao/verificacao_validacao.py`).
2. `interpretacao_modelos/interpretacao_modelos.py` (células `#%%`) ou `interpretacao_modelos.ipynb` (mesmo conteúdo, já executado). Leva ~1,5 min.

**Sem pacotes novos.** `shap` e `lime` não estão no `requirements.txt`. Com só 6 features dá para calcular os **valores de Shapley exatos** testando as 2⁶ = 64 combinações de colunas (é o que o `shap` faz no modo "intervencional"), e o LIME é uma regressão linear ponderada em ~15 linhas. A ideia é a mesma dos pacotes, e o código mostra a conta.

O script **para com erro** se o `.joblib` não for a receita do plano A em treino + validação, ou se a soma dos valores de Shapley não fechar com a chance prevista.

### Arquivos gerados

| Arquivo | O que é |
|---|---|
| `importancia_global.csv` | Cada feature por cinco réguas: `feature_importances_`, permutação no teste, coeficiente da logística, SHAP e Saabas |
| `coeficientes_logistica.csv` | Coeficientes padronizados, `e^coef` (odds) e estabilidade em 300 reamostragens |
| `arvore_rasa_regras.txt` | Árvore de 3 níveis em texto (`export_text`) |
| `pdp_casos_impossiveis.csv` | Quantos pontos do PDP são combinações que não existem nos dados |
| `lime_estabilidade.csv` | LIME do falso negativo em 20 execuções, ao lado do SHAP do mesmo cestão |
| `contrafactuais_teste.csv` e `contrafactuais_resumo.csv` | Quanto pesado a mais tiraria cada cestão alarmado do alarme |
| `vies_cestao1_cestao2.csv` e `motivo_dos_alarmes.csv` | O modelo trata igual o cestão 1 e o 2? Quantos alarmes são "só histórico"? |
| `explicacoes_alarmes_teste.csv` | Os 175 alarmes do teste explicados em 3 frases, em linguagem do pátio |
| `graficos/` | 5 figuras para a apresentação |

### Roteiro da aula (slide 42) e onde está cada passo

| Passo | Onde |
|---|---|
| 1. Valide o modelo antes de interpretar | `verificacao_validacao/` (31 conferências) + seção 0 aqui |
| 2. Defina quem vai receber a explicação | tabela abaixo |
| 3. Importância global por permutação | seção 2, gráfico 01 |
| 4. Forma do efeito com PDP e ICE | seção 3, gráfico 02 (e gráfico 10 da análise) |
| 5. Explique casos com SHAP ou LIME | seções 4 e 5, gráficos 03 a 05 |
| (contrafactuais, slide 35) | seção 6 |
| 6. Verifique viés e casos estranhos | seção 7 |
| 7. Traduza para a linguagem do negócio | seção 8 e o fim deste README |

### Quem recebe a explicação (slide 9)

| Quem | O que quer saber | O que mostramos |
|---|---|---|
| O grupo e o professor (cientistas de dados) | O modelo aprendeu algo errado? | As cinco réguas de importância, PDP/ICE, SHAP × Saabas, LIME |
| Gestão da EVCOMX | Posso confiar e agir com base nisso? | Três fatores em linguagem de pátio, o que o modelo **não** sabe e o empate com a regra do histórico |
| Operador do pátio | Por que **este** cestão foi alarmado? O que eu mudo? | A explicação de 3 frases por alarme e o contrafactual em toneladas de pesado |
| Auditoria (aqui, a própria EVCOMX) | O modelo trata casos iguais de forma diferente? | Cestão 1 × cestão 2, alarmes "só histórico" |

---

### 1. Modelos transparentes (slides 16 a 20)

Logística com `StandardScaler` e árvore de 3 níveis, nas mesmas 6 features e nos mesmos dados (treino + validação).

| Feature | Coef. por desvio-padrão | Odds × a cada desvio-padrão | Sinal trocado nas reamostragens |
|---|---|---|---|
| `taxa_ca_ult5` | +0,55 | **× 1,73** | 0% |
| `prop_peso_pesado` | −0,51 | **× 0,60** | 0% |
| `densidade_media` | −0,26 | × 0,77 | 2% |
| `prop_PACB` | +0,22 | × 1,24 | 0% |
| `peso_por_ciclo_t` | +0,17 | × 1,18 | 1% |
| `taxa_ca_ult10` | +0,12 | × 1,12 | **16%** |

- **Leitura (slide 18):** a cada desvio-padrão a mais de histórico recente (~1,5 corrida com carga alta nas últimas 5), a razão de chances de carga alta é multiplicada por 1,7. A cada desvio-padrão a mais de pesado (~14 pontos percentuais do peso), ela cai 40%. Todos os sinais batem com a física e com a análise.
- **Colunas parecidas (slide 17):** `taxa_ca_ult10` troca de sinal em 16% das reamostragens. Ela anda junto com `taxa_ca_ult5` (correlação 0,85), e o coeficiente fica com "o que sobra". Não dá para ler a direção dela sozinha.
- **Árvore rasa** (`arvore_rasa_regras.txt`): a primeira pergunta é a densidade (≤ 0,90 t/m³), depois o histórico e a proporção de pesado (cortes em 16% e 20%). Todo cestão denso (> 0,90 t/m³) vai para "normal".
- **Desempenho × interpretabilidade (slide 14):** no teste, a logística tem PR-AUC **0,44**, a Random Forest de uso 0,40 e a árvore rasa 0,31. O teste não serve para escolher, e na validação a floresta ficou bem na frente (0,43 contra 0,29). Mas na CV temporal a logística também ia melhor (0,45 contra 0,41), e a feature selection já tinha visto 0,42 no teste: só a validação (abr–jun) favorece a floresta. Pela "escada" da Aula 05, vale levar a pergunta para a apresentação: se a logística empata, ela é mais fácil de explicar.

### 2. Importância global: três réguas, três ordens (slides 22, 23 e 38; gráfico 01)

| Feature | `feature_importances_` | Permutação no teste | Logística | SHAP |
|---|---|---|---|---|
| `taxa_ca_ult5` | 5º | **1º** | **1º** | 2º |
| `taxa_ca_ult10` | 4º | 2º | 6º | 3º |
| `densidade_media` | 2º | 3º | 3º | 4º |
| `peso_por_ciclo_t` | 3º | 4º | 5º | 5º |
| `prop_peso_pesado` | **1º** | 5º | 2º | **1º** |
| `prop_PACB` | 6º | 6º | 4º | 6º |

- **O erro do `feature_importances_` aparece ao vivo (slide 22).** Ele põe o histórico em 4º e 5º. As duas `taxa_ca` só têm 6 e 11 valores possíveis, contra ~2.900 da densidade e da proporção de pesado; a impureza favorece as colunas com muitos valores. A permutação, medida no teste, põe o histórico em 1º e 2º.
- **Os modelos concordam no que importa (exercício, item 2).** Histórico recente e proporção de pesado estão no topo da logística, do SHAP e (somando o par do histórico) da permutação. `prop_PACB` é a última em quase todas.
- **A permutação subestima `prop_peso_pesado`:** 5º lugar, com desvio maior que a média. Ela tem correlação 0,84 com a densidade e −0,72 com o peso por ciclo; embaralhada sozinha, as outras duas cobrem a falta.
- **Permutação em grupo (slide 38).** Embaralhando as duas `taxa_ca` juntas, a PR-AUC cai **0,21**; a soma das quedas individuais era 0,17. A receita inteira (densidade, pesado, peso por ciclo e PACB) cai 0,09, contra 0,07 somando. **O histórico do forno vale mais que o dobro da receita do cestão para este modelo.**

### 3. PDP, ICE e os "casos impossíveis" (slides 24 a 26 e 38; gráfico 02)

- **`prop_peso_pesado` tem um degrau:** a chance média fica em ~0,40 até ~16–20% de pesado e cai para ~0,25 acima de 20%. O mesmo corte de 20% aparece na árvore rasa. As linhas ICE mostram que o degrau é forte nos cestões de risco alto e quase some nos de risco baixo: o efeito depende das outras colunas.
- **`taxa_ca_ult5`** sobe de 0,26 para 0,52, e sobe em **100%** dos cestões (todas as linhas ICE sobem).
- **Casos impossíveis.** Para desenhar o PDP de `taxa_ca_ult5`, o scikit-learn cria cestões com 5 cargas altas nas últimas 5 corridas e **zero** nas últimas 10. Isso não pode acontecer. **57%** dos pontos desse PDP caem em combinações que não existem nos dados (célula da grade com menos de 3 cestões reais). Para `prop_peso_pesado` com a densidade fixa, são 10%. A forma da curva vale como ilustração; o tamanho do efeito, não.

### 4. SHAP (slides 28 a 31; gráficos 03 e 04)

Valores de Shapley exatos, com 200 cestões de referência do treino + validação. A chance média de referência é **0,25** (crua: os pesos balanceados inflam as chances; a calibração da análise corrige isso sem mudar a ordem). Em todos os 515 cestões, **referência + soma dos valores = chance prevista**, e o script confere.

- **Visão geral (beeswarm):** pouco pesado (azul) empurra para cima, até +0,28; histórico alto (vermelho) também, até +0,22. `prop_PACB` é zero na maioria dos cestões e quase não mexe.
- **SHAP × Saabas:** a decomposição usada na análise conta a mesma história (correlação de 0,82 a 0,97, cestão a cestão), mas dá mais crédito ao `peso_por_ciclo_t` e ao `taxa_ca_ult10`. É a diferença na divisão das interações.
- **O cestão de maior chance** (O7144 c2, 23/07, chance 0,98, deu carga alta): pouco pesado (10%, **+0,26**), 5 cargas altas nas últimas 5 corridas (+0,19), camadas grossas (34 t por ciclo, +0,10), cestão fofo (0,72 t/m³, +0,09).
- **O falso negativo mais "certo" (exercício, item 4)** (O6827 c2, 07/07, chance 0,06, deu carga alta). O modelo deixou passar porque **o forno estava calmo**: nenhuma carga alta nas últimas 10 corridas (−0,10) e nas últimas 5 (−0,05). A receita era normal, levemente boa (29% de pesado, 0,86 t/m³). Nada nas 6 features indicava risco. É o mesmo cestão que a análise já tinha mandado para o especialista (`analise_modelos/casos_para_especialista.csv`).

### 5. LIME (slides 32 a 34; gráfico 05)

No mesmo falso negativo, 20 execuções com sementes diferentes:
- **A direção não muda e as 3 colunas principais são as mesmas nas 20 execuções** (as duas `taxa_ca` e a densidade). O tamanho dos coeficientes varia ~10% de uma execução para outra, e `prop_PACB` troca de sinal em 5% delas. É a instabilidade do slide 34, aqui pequena.
- **LIME e SHAP respondem perguntas diferentes.** O LIME dá `taxa_ca_ult5` com coeficiente **+0,05**: "se o histórico subisse, a chance subiria". O SHAP dá **−0,05**: "o histórico deste cestão (zero) puxou a chance para baixo da média". As duas frases são verdadeiras ao mesmo tempo. Para explicar um caso ao pátio, o SHAP é o certo, porque a conta fecha; o LIME mostra para onde o modelo é sensível.

### 6. Contrafactuais: o que mudaria o alarme? (slide 35)

Para os 175 cestões alarmados no teste (corte F2, 0,41), procuramos o menor aumento na proporção de pesado que tiraria o cestão do alarme. O histórico não entra: ele não se muda na hora de montar o cestão.

| | Cestões alarmados | Pesado a mais, mediana | Em toneladas, mediana |
|---|---|---|---|
| Todos | 175 | +9 pontos percentuais | ~3,8 t |
| Forno calmo (< 2 cargas altas nas últimas 5) | 45 | +5 p.p. | ~1,4 t |
| Forno agitado (≥ 2 nas últimas 5) | 130 | +12 p.p. | ~5,1 t |

- **Versão coerente:** quando o pesado sobe, a densidade sobe e o peso por ciclo cai junto, pela relação observada nos dados (+10 p.p. de pesado ≈ +0,09 t/m³ e −4,3 t por ciclo).
- **Versão ingênua** (só o pesado sobe, densidade parada): em **22%** dos casos, a solução cai numa combinação de pesado e densidade que não existe nos dados. No forno agitado, 29%. É o "seja 10 anos mais jovem" do slide 35: o modelo responde, mas o cestão não existe.
- **Isto não é uma receita para o pátio.** É o que mudaria a resposta **do modelo**. Se trocar 4 t de leve por pesado evita a carga alta no forno de verdade é uma pergunta de causa (slide 37), que só um teste no pátio responde.

### 7. Viés e casos estranhos (slide 39)

Não há pessoas sendo decididas pelo modelo, então não há grupo protegido. O equivalente aqui é: o modelo trata igual situações iguais?

| | Cestões | Taxa de carga alta | Alarmes | Recall | Precisão | Peso da receita na decisão (\|SHAP\| médio) |
|---|---|---|---|---|---|---|
| Cestão 1 | 270 | 14% | 28% | 0,68 | 0,34 | 0,13 |
| Cestão 2 | 245 | 15% | **40%** | 0,76 | 0,28 | **0,22** |

- **O cestão 2 recebe 12 pontos a mais de alarmes com quase a mesma taxa de carga alta**, e a receita pesa ~65% mais na decisão dele. No teste, o cestão 2 tem em média 13% de pesado, contra 25% do cestão 1; o modelo lê "pouco pesado" como risco, mas a taxa real de carga alta é quase a mesma. A análise já tinha visto os alarmes falsos concentrados no cestão 2. **Pergunta para a EVCOMX:** o cestão 2 é montado de outro jeito (completa o forno, entra com o banho já líquido) a ponto de "fofo" ser normal para ele?
- **35% dos alarmes são "só histórico":** o histórico do forno empurra mais que toda a receita junta. Esses alarmes acertam mais (precisão 0,35, contra 0,28 dos alarmes puxados pela receita). Parte do modelo é, na prática, a regra "carga alta puxa carga alta", que já empatava com ele na análise e na verificação.

### 8. Traduzir para o negócio (slide 40 e exercício, item 5)

Cada alarme do teste ganhou uma explicação automática em 3 frases (`explicacoes_alarmes_teste.csv`), com os dois fatores que mais subiram a chance e o que mais segurou. Exemplos:

> **O7144 c2 (alarme, deu carga alta):** Risco alto: o modelo deu 98 pontos de chance, contra 25 de um cestão típico. O que mais aumentou o risco: pouco material pesado (10% do peso) e forno com carga alta recente (5 de 5 corridas). Nada segurou o risco.

> **O6827 c2 (falso negativo, explicado para a EVCOMX):** Risco baixo: o modelo deu 6 pontos de chance, contra 25 de um cestão típico. Nenhuma característica aumentou muito o risco. O que segurou o risco: forno calmo nas últimas 10 corridas (0 de 10 corridas).

E a frase do falso negativo para alguém sem formação em dados: *"O modelo não alarmou porque o forno vinha sem nenhuma carga alta havia 10 corridas e o cestão tinha uma receita normal. A carga alta veio de algo que o modelo não enxerga, como umidade ou formato da sucata. Casos assim são os que mais valem uma conversa com o pátio."*

**Antes e depois (slide 40):**

| Antes | Depois |
|---|---|
| "`prop_peso_pesado` tem SHAP médio absoluto de 0,08 e `taxa_ca_ult5` lidera a permutação." | "O que mais pesa é quanto pesado tem no cestão e se o forno teve carga alta nas últimas corridas. Com 2 ou mais cargas altas recentes, cerca de 1 em cada 4 cestões dá carga alta; com o forno calmo, 1 em 12." |
| "O contrafactual mediano é +0,09 de `prop_peso_pesado`." | "No alarme típico, o modelo deixaria de alarmar com uns 4 t de leve trocados por pesado. Isso é o que o modelo diz, não uma garantia do forno." |

### Os limites (slides 37 e 38)

- **Explicar o modelo não é explicar o forno.** O histórico é o maior sinal, mas ele é sintoma, não causa: uma fase ruim do forno ou da sucata faz várias cargas altas seguidas. "Esperar o forno acalmar" não é uma ação.
- **Colunas correlacionadas dividem o crédito.** O histórico em duas janelas e o trio densidade/pesado/peso por ciclo trocam importância entre si conforme a régua. Leia por grupo, não por coluna.
- **PDP e contrafactual ingênuo avaliam o modelo onde ele nunca esteve** (57% e 22% dos pontos). Por isso o contrafactual usado é o coerente.
- **As chances são cruas.** As explicações valem para a ordem dos cestões; para falar em "% de chance" com a EVCOMX, use as chances calibradas da análise.

### O que depende da EVCOMX

| Pergunta | Muda o quê |
|---|---|
| O cestão 2 é montado de outro jeito? "Fofo" é normal para ele? | Se sim, o alarme do cestão 2 precisa de outro corte ou de uma feature que diga "cestão 2" (seção 7). |
| Trocar leve por pesado é viável na montagem? Quanto custa? | Diz se o contrafactual (seção 6) vira uma ação possível ou só uma explicação. |
| As explicações de 3 frases fazem sentido para o pátio? | É o teste do "Para discutir" da aula: alguém sem formação em dados entende? |

### Limitações

- **O SHAP depende da referência.** Com outra amostra de 200 cestões, a chance de referência e os valores mudam um pouco (a ordem não). O modo "intervencional" trata as colunas como independentes, o que não é verdade aqui; a conta fecha, mas os pontos de referência também criam algumas combinações irreais.
- **O LIME aqui é uma versão simples** (ruído normal, sem a discretização em quartis do pacote `lime`). Os números não são iguais aos do pacote, mas a ideia e o comportamento são.
- **Uma explicação local por vez.** Os dois casos detalhados foram escolhidos por serem extremos (maior chance e falso negativo mais "certo"); não representam o cestão típico.
- **O teste foi olhado de novo**, só para descrever. Nada aqui muda o modelo, as features ou os limiares.
