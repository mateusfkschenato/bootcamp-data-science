## Feature Engineering e Feature Selection — Predição de Carga Alta (EVCOMX)

Segue a aula de Feature Selection do bootcamp, o Guia de FE/FS da EVCOMX e, na revisão, o **Guia de Modelagem Preditiva da EVCOMX** (§1 a §3).

> **Rodada de out/2026 (corte de 2.000 kg).** A EVCOMX respondeu que camadas de 10 kg e 100 kg também podem ser desconsideradas e deu 2.000 kg como corte. A EDA passou a remover todas as camadas abaixo de 2.000 kg (antes, só as de peso zero) e esta etapa foi refeita do zero. Mudou o conjunto final: de **9 para 6 features**. As 5 features com 6 votos são as mesmas de antes; das 4 com 5 votos, só `prop_PACB` continua. `max_bloco_leve_m3`, `prop_leve_base` e `prop_PAEP` caíram para 3 ou 4 votos. Os números abaixo são desta rodada.

### Como rodar

1. `EDA/transforma_dataset.py`: gera o `EDA/base.csv` limpo.
2. `feature_selection/feature_selection.py` (células `#%%`) ou `feature_selection_carga_alta.ipynb` (mesmo conteúdo, já executado).

O código acha a raiz do projeto sozinho; pode rodar de dentro de `feature_selection/` ou da raiz.

### Arquivos gerados

| Arquivo | O que é |
|---|---|
| `dataset_modelagem_fs.csv` | **Conjunto final (6 features)** + `carga_alta` + metadados (`cd_corrida`, `nu_carregamento`, `conjunto`, `inicio`) |
| `dataset_features_completo.csv` | Todas as 84 candidatas, para a seleção de algoritmos testar outros conjuntos |
| `votos_feature_selection.csv` | Voto de cada método, estabilidade, vitórias no Boruta e correlação |
| `tabela_features_finais.csv` | **Entregável do Guia de Modelagem (§3.2):** como cada feature final é calculada, por que entra e quando está disponível |
| `curva_n_features.csv` | PR-AUC na validação conforme o nº de features, na ordem dos votos |
| `features_guia_modelagem.csv` | As 3 features sugeridas pelo Guia de Modelagem e `violacao_regra`, por cestão (usadas na validação de domínio) |
| `features_guia_modelagem_avaliacao.csv` | O que essas 3 features acrescentam ao conjunto final (seção 11b) |
| `graficos/` | Figuras para a apresentação |

- **`conjunto`** traz o split 70/15/15 temporal (treino / validacao / teste), definido pelo professor. **Todo o grupo deve usar essa coluna.**
- **`inicio` não é feature.** É a data do cestão, guardada só como metadado.

### Decisões

- **Unidade de análise:** o cestão (carregamento 1 ou 2). É a pergunta operacional, o alvo já vem por cestão, dá ~3.500 exemplos e permite usar o cestão 1 como contexto do 2.
- **Camadas:** só as de 2.000 kg ou mais (decisão da EDA, com o corte dado pela EVCOMX). Nenhum cestão sai inteiro. Com menos camadas "fantasma", algumas flags de posição deixaram de ser quase constantes e voltaram ao filtro: são **84 candidatas** (antes, 80).
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
- **Quantas features:** as features são adicionadas uma a uma, por ordem de votos, e o desempenho é medido na validação. O "cotovelo" ficou em **6 features**. A 7ª da fila, `dia_semana` (5 votos), fica fora pelo cotovelo.

### Conjunto final (6 features): como se calcula, por que entra e quando está disponível

É o entregável que o Guia de Modelagem pede ao fim da feature selection (§3.2). Todas passam pela pergunta do Guia (§1.2): **"o pátio sabe isso antes de despejar o cestão?"**

| Feature | Como é calculada | Por que entra | Quando está disponível | Sinal | Votos |
|---|---|---|---|---|---|
| `densidade_media` | peso total ÷ volume estimado (volume da camada = peso ÷ densidade do material) | cestão "fofo" ocupa mais altura para o mesmo peso | após a montagem do cestão | − | 6 |
| `prop_peso_pesado` | peso de material pesado ÷ peso total | o pesado compacta a carga | após a montagem | − | 6 |
| `peso_por_ciclo_t` | peso total ÷ nº de ciclos de montagem (ciclo = bloco de pesado) | ciclos "grossos" = camadas pouco intercaladas | após a montagem | + | 6 |
| `taxa_ca_ult5` | % de corridas com carga alta entre as 5 anteriores (`shift(1)`, nunca a atual) | estado recente do forno: carga alta puxa carga alta | quando a corrida anterior termina (**confirmar com a EVCOMX**) | + | 6 |
| `taxa_ca_ult10` | idem, 10 corridas | mesma ideia, janela mais estável | idem (**confirmar**) | + | 6 |
| `prop_PACB` | peso de PACB ÷ peso total | pacote de chaparia: volumoso para o peso | após a montagem | + | 5 |

**O que saiu em relação à rodada anterior:** `max_bloco_leve_m3` (de 5 para 4 votos), `prop_leve_base` (de 5 para 3) e `prop_PAEP` (de 5 para 3). As três dependem da sequência e da posição das camadas, que é justamente o que muda quando as camadas pequenas saem: o maior bloco de leve, o que está nos 30% de baixo e a participação de um material específico.

**Features proibidas por vazamento** (Guia §1.2 e checklist §8.2): duração da parada (`qt_segundos_parada`), resultado do outro cestão da mesma corrida (`carga_alta_c1`, fora do conjunto final), data/hora de início como número, e qualquer dado de energia ou fusão da corrida. A `energia_estimada_mwh` das candidatas é só a soma da energia típica de cada material vezes o peso: é uma propriedade da receita, conhecida antes do despejo, não um dado medido da corrida.

### Respostas às perguntas do Guia (§1.3)

1. **Quais features descrevem um cestão propenso a carga alta?** Pouco material pesado, baixa densidade, ciclos de montagem "grossos" e carga alta recente no forno. As mais importantes na validação, medidas por permutação: `prop_peso_pesado`, `taxa_ca_ult5`, `peso_por_ciclo_t`, `taxa_ca_ult10` e `densidade_media`.
2. **O volume estimado é a feature mais importante?** **Não.** O volume total quase não se relaciona com a carga alta (r = −0,04). Sozinho, ele dá PR-AUC de 0,19 na validação, igual a um chute (0,18). O sinal está na densidade e na proporção de pesado: os cestões são enchidos até volumes parecidos, e o que muda é quanto peso cabe nesse volume. A física do Guia se confirma pela densidade, não pelo volume total.
3. **Ordem e ciclos agregam além de peso e volume?** **Nesta rodada, não de forma mensurável.** Sem `peso_por_ciclo_t`, a PR-AUC na validação fica igual (0,43 contra 0,42 com ela, Random Forest). As features de ciclo são muito correlacionadas com a composição, então a informação delas já está quase toda em `prop_peso_pesado`. Mesmo assim `peso_por_ciclo_t` teve 6 votos e é a 3ª mais importante por permutação: ela fica.
4. **Qual método de seleção gera o melhor desempenho fora da amostra?**
   - Na validação, com Random Forest: consenso 0,43; RFE 0,43; cotovelo 0,42; L1, Elastic Net, filtro e Boruta 0,41; todas as 84, 0,40.
   - As diferenças são pequenas (o IC 90% da validação é de ±0,08). A votação entre métodos foi mais útil que qualquer método sozinho.
5. **Quantas features bastam?** **6 de 84.** No teste, o conjunto de 6 rende o mesmo ou mais que as 84:
   - regressão logística: PR-AUC 0,42 com 6 contra 0,41 com 84;
   - Random Forest: 0,38 contra 0,35.
   Um modelo com 6 features é mais simples de explicar e de manter.

### Desempenho no teste (jun–ago/2026)

| Conjunto | Algoritmo | Acurácia | Recall | Precisão | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| todas (84) | Reg. logística | 0,81 | 0,55 | 0,39 | 0,45 | 0,75 | 0,41 |
| **final (6)** | **Reg. logística** | 0,58 | **0,88** | 0,24 | 0,38 | 0,77 | 0,42 |
| final (6) | Random Forest | 0,73 | 0,69 | 0,31 | 0,42 | 0,77 | 0,38 |
| básico do Guia (3) | Random Forest | 0,36 | 0,85 | 0,17 | 0,28 | 0,62 | 0,21 |

Referências de um chute: PR-AUC = 0,15; quem sempre diz "normal" acerta 85%.

O limiar foi escolhido na validação. A escolha final entre recall e precisão fica para a análise de modelos, e depende do custo de um alarme falso versus o de uma carga alta não detectada (pergunta para a EVCOMX).

### Revisão pelo Guia de Modelagem (§3)

**Passo a passo do Guia (§3.1) e onde cada passo está:**

| Passo do Guia | Onde |
|---|---|
| 1. Filtros simples (constantes, quase constantes, muitos nulos) | §3: `VarianceThreshold` feito à mão (moda > 99%) e > 50% nulos |
| 2. Redundância: \|r\| > 0,9 e VIF | §5: pares > 0,90 e VIF < 10, calculado à mão |
| 3. Relevância univariada: point-biserial, Mann-Whitney, informação mútua | §4 |
| 4. Importância via modelo: árvores, **permutação na validação**, L1 | §7 |
| 5. RFECV, dentro da validação cruzada | §8 |
| 6. Todas × selecionadas × **baseline mínimo (só volume)** | §11 e §12 |

**Passo 6, com o baseline mínimo do Guia** (PR-AUC na validação, Random Forest / logística):

| Conjunto | RF | Logística |
|---|---|---|
| todas as 84 candidatas | 0,40 | 0,34 |
| **final (6)** | **0,42** | 0,29 |
| só volume (baseline mínimo) | 0,19 | 0,20 |

Com a Random Forest, as 6 rendem tanto quanto as 84; o volume sozinho fica no nível do chute (0,18). A logística prefere mais features (já registrado nas limitações abaixo).

**Armadilhas do Guia (§3.2):**

| Armadilha | Como ficou |
|---|---|
| Selecionar com todo o dataset e só depois dividir | Toda a seleção roda no treino. A exceção documentada é a permutação, medida na validação, como o próprio Guia sugere no passo 4 ("permutation importance em conjunto de validação"). Ela não toca o teste, mas deixa a nota da validação otimista (verificação, item 1.3d). |
| Feature "boa demais" | Nenhuma feature sozinha passa de PR-AUC 0,34 na validação (a melhor é a regra do histórico). Nada sugere vazamento; o histórico foi recalculado do arquivo bruto com `shift(1)` na verificação (1.4b). |
| Descartar por correlação linear baixa | Informação mútua, Boruta e permutação enxergam relações não lineares; uma feature com correlação baixa ainda pode receber até 4 votos. |
| Escolher só pelo número | Toda feature final tem justificativa física (tabela acima). |

**Features sugeridas pelo Guia de Modelagem (§2.2) e testadas à parte (seção 11b).** Três features de estrutura ainda não estavam entre as candidatas: a violação da regra leve–pesado–leve (o cestão não começa ou não termina com leve), o "colchão" de leve no topo e a proporção do volume nas camadas centrais.

| Feature | Correlação no treino | Taxa / média com e sem carga alta (treino) | Conjunto final + ela: PR-AUC validação (RF / logística) |
|---|---|---|---|
| `violacao_fundo_topo` | −0,04 | com violação: 12% de carga alta; sem: 17% | 0,42 / 0,28 |
| `colchao_topo_m3` | +0,11 | 9,2 m³ com carga alta; 7,7 m³ sem | 0,40 / 0,28 |
| `prop_vol_centro` | −0,08 | 0,32 com carga alta; 0,34 sem | 0,43 / 0,29 |
| *referência: conjunto final* | | | *0,42 / 0,29* |

- **A violação da ordem não aumenta o risco no treino:** ali, os cestões que fogem da regra têm **menos** carga alta (12% contra 17%). Uma explicação possível: com vários ciclos leve → pesado → misto → leve por cestão (correção da EVCOMX), começar ou terminar com misto ou pesado não é, por si, erro de montagem. **Depois de maio a relação se inverte** (21% com violação contra 13% sem, validação + teste; verificação, seção 6.3). Com o regime novo, ela merece ser testada de novo quando houver mais dados.
- Nenhuma das três melhora o conjunto final além do ruído (a variação só pela semente é ~0,01; o IC 90% da validação, ±0,08). **O conjunto final não muda.**
- **Por que à parte e não como candidatas:** numa rodada de teste com as três na votação (feita ainda com o corte antigo, só peso zero), nenhuma passou de 4 votos, mas a presença delas mudou a ordem da multicolinearidade e do RFE, e o cotovelo caiu para as 5 features com 6 votos.

**O conjunto de 5 rende o mesmo que o de 6.** A curva (`curva_n_features.csv`) mostra a PR-AUC da Random Forest na validação: 0,422 com as 5 primeiras (as de 6 votos) e 0,416 com as 6. A diferença está dentro do ruído. Mantivemos as 6 porque a regra do cotovelo foi fixada antes (98% do máximo da curva suavizada) e trocar agora seria escolher olhando a validação de novo. Se a EVCOMX preferir um modelo mais simples, as 5 features (`densidade_media`, `prop_peso_pesado`, `peso_por_ciclo_t`, `taxa_ca_ult5`, `taxa_ca_ult10`) são a alternativa natural, e agora diferem do conjunto final só por `prop_PACB`.

### Limitações e perguntas para a EVCOMX

- **Timing do histórico (crítico).** `taxa_ca_ult5/10` usa o resultado das corridas anteriores. Isso vale **se**, na hora de prever, já se sabe se a corrida anterior teve carga alta. A dependência é grande: sem o histórico, a PR-AUC na validação cai de 0,42 para 0,30 (Random Forest) e de 0,29 para 0,20 (regressão logística, quase o chute). **Confirmar com a EVCOMX.** Se não valer, use `dataset_features_completo.csv` sem as colunas `taxa_ca_*`.
- **`carga_alta_c1` como feature do cestão 2** melhora pouco a validação (0,42 → 0,43) e depende do timing. Ficou fora do conjunto final.
- **Mudança de regime em maio/2026.** Desde maio há menos material pesado, menos ciclos por cestão e densidade menor. A taxa de carga alta continua igual, mas a relação "mais pesado → menos carga alta" enfraquece. O treino termina em 25/04, então validação e teste estão no regime novo. Perguntar o que mudou: regra de montagem, fornecedor, forno ou critério de registro.
- **Corte de 2.000 kg.** Nem tudo abaixo de 2 t parece gancho vazio (entre 100 kg e 2 t a mediana é ~1,6 t). A verificação (seção 5c-bis) refaz as features com cortes de 0, 100, 1.000 e 2.000 kg e mede o modelo: a PR-AUC muda menos de 0,03, dentro do ruído. A seleção de features **não** foi refeita para cada corte: com o corte antigo ela escolhia 9 features (as 6 de hoje mais `max_bloco_leve_m3`, `prop_leve_base` e `prop_PAEP`).
- **Escolha da regressão logística.** Ela rende melhor com mais features. Se a seleção de algoritmos escolher regressão logística, vale testar o consenso de ≥ 4 votos (18 features), disponível no `dataset_features_completo.csv`.
- **Variação entre versões.** Os números podem mudar um pouco com outra versão do scikit-learn.
