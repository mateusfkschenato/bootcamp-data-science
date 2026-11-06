## Verificação e Validação — Predição de Carga Alta (EVCOMX)

Segue a **Aula 07 do bootcamp** (Verificação e Validação de Modelos) e o **Guia de Modelagem Preditiva da EVCOMX** (§6). A aula separa duas perguntas:
- **Verificação:** o modelo foi construído e executado como dissemos?
- **Validação:** ele funciona em dados novos, e dá para confiar nele fora do conjunto original?

**Modelo verificado: o oficial**, a média de 5 Gradient Boosting com 121 features (120 + o resultado do cestão 1), de `selecao_algoritmos/modelo_final.joblib`.

> **Antes (out/2026), esta pasta verificava a Random Forest de 6 features** (31 conferências, critérios de aceitação em orçamento de alarmes).

**Esta etapa não escolhe nada.** O script **para com erro** se algum número das etapas anteriores não bater.

### Como rodar

1. As etapas anteriores, em ordem (a última é `analise_modelos/analise_modelos.py`).
2. `verificacao_validacao/verificacao_validacao.py` (células `#%%`) ou `verificacao_validacao.ipynb` (mesmo conteúdo, já executado). Leva ~10 min.

Para recalcular a receita com camadas alteradas (erro de balança, corte de peso), usa `comum/receita.py`, que reaproveita a conta do app (`app_streamlit/motor.py`, conferida contra o dataset).

### Arquivos gerados

| Arquivo | O que é |
|---|---|
| `checklist_verificacao.csv` | As 29 conferências, com status e detalhe |
| `gap_treino_wf_validacao_teste.csv`, `curva_complexidade.csv` | O gap e a curva de complexidade |
| `uso_validacao_e_teste.csv` | Cada vez que validação e teste foram usados no caminho do oficial |
| `tipos_de_validacao_cruzada.csv` | Quatro jeitos de validar, mesma receita |
| `drift_das_features.csv`, `relacao_feature_alvo_por_particao.csv` | Data drift (PSI) e concept drift |
| `walk_forward_mensal.csv`, `teste_retreino_mensal.csv` | Backtesting mês a mês; re-treino mensal dentro do teste |
| `robustez_erro_pesagem.csv`, `sensibilidade_corte_peso.csv`, `variacao_por_semente.csv` | Robustez |
| `criterios_aceitacao.csv`, `monitoramento_simulado.csv` | Critérios e monitoramento |
| `validacao_dominio.csv`, `validacao_negocio.csv` | Faz sentido físico? Vale a pena? |
| `registro_experimento.json`, `requirements_rodada.txt` | Registro (versões, hashes dos dados, receita, resultado) |
| `graficos/` | 6 figuras |

### 1. Checklist técnico (slide 8)

**29 conferências: 26 OK, 3 com ressalva, nenhuma falha.**

| Item | Conferência | Resultado |
|---|---|---|
| 1.1 | Rótulos = arquivo bruto, cestão a cestão; `carga_alta_c1` = alvo do cestão 1 (e 0 no cestão 1); parada > 0 só com carga alta | OK (3.501 cestões, 559 cargas altas) |
| 1.2 | O dataset de modelagem = o completo nas 121 features; o pacote de uso tem as 121 na mesma ordem | OK |
| 1.3 | Histórico recalculado do bruto com `shift(1)`; `rel_` usa só os 100 cestões **anteriores**; o walk-forward nunca treina com o futuro | OK |
| 1.3e | Split temporal sem sobreposição | **Ressalva:** 2 cestões 2 com hora registrada depois do início da partição seguinte (dado, não modelo) |
| 1.4 | Duração da parada e alvo não são features; nenhuma feature "boa demais" (a melhor sozinha: 0,35 de PR-AUC) | OK |
| 1.4c | Timing do resultado do cestão 1 | **Ressalva:** confirmado pela EVCOMX; se faltar, o app usa a reserva |
| 1.5 | Minutos, recall, precisão e PR-AUC do teste **refeitos à mão** (131 min = 18 FN + 91 FP) | OK |
| 1.6 | Re-treinar dá exatamente as mesmas chances (seleção, `modelo_dev`, `modelo_final`); versão do scikit-learn | OK |
| 1.7b | Oficial × regra do histórico no teste | **Ressalva:** −17 min, IC 90% de −59 a +21 (dentro do ruído); PR-AUC +0,21 (fora do ruído) |
| 1.8a | **Alvo embaralhado** (Guia §6.1): PR-AUC na validação cai a 0,17 (chute 0,18; modelo real 0,45) | OK: sem vazamento escondido |
| 1.8b–c | Testes da agregação (peso, volume, proporções somam 1, densidade) e dos joins (3.501 → 3.501 → 3.501 cestões) | OK |

**O que um vazamento de verdade faria:** se o histórico incluísse a corrida **atual**, a PR-AUC na validação saltaria de 0,45 para **0,88**. É o "resultado excelente que pode estar errado" do slide 4, e a conferência 1.3 garante que o projeto não faz isso.

### 2. Treino bom não garante desempenho futuro (slides 10 a 12; gráfico 01)

| Onde | PR-AUC |
|---|---|
| No próprio treino | 0,94 |
| Walk-forward (dev, nov–jun) | 0,50 |
| Validação (treino → abr–jun) | 0,45 |
| Teste (dev → jun–ago) | 0,57 |

O gap do treino é grande (é um modelo de árvores), mas o que importa é o fora da amostra, e ele é estável entre 0,45 e 0,57. **Complexidade:** com 300 árvores (o usado), a validação está no topo (0,44); com 600 e 1.200, o treino vai a 1,00 e a validação começa a cair. O modelo está no ponto, não sobreajustado.

### 4. Validação cruzada: limites e cuidados (slide 27; gráfico 02)

| Validação (mesma receita, dev) | PR-AUC |
|---|---|
| k-fold aleatório estratificado | **0,61** |
| k-fold agrupado por corrida | 0,60 |
| **Walk-forward mensal (o usado)** | **0,50** |
| Holdout temporal (treino → validação) | 0,45 |

Sortear linhas promete 0,61; no tempo, o modelo entrega 0,50. A diferença é o futuro vazando para o treino (e a mudança de maio). Por isso toda escolha usou o walk-forward.

### 5. Os dados mudaram? (slides 33 e 34; gráfico 03)

**Data drift (PSI contra o treino; > 0,25 = mudou):** a receita mudou muito depois de maio (`prop_peso_pesado` 1,5 no teste, `densidade_media` 1,3, `n_ciclos` 0,7) e há mais gusa (0,8). **As features `rel_` foram feitas para isso e mudam bem menos** (0,27–0,30). O histórico (`h_ewm2` 0,40 no teste) e o resultado do cestão 1 (0,00) são estáveis.

**Concept drift (a relação com o alvo mudou?):**
- **Densidade ≥ 0,8 t/m³:** 9% de carga alta no treino, **19% na validação**, 10% no teste. A relação "denso = seguro" ficou mais fraca em abr–jun.
- **Histórico alto:** 31% → 36% → **41%**; histórico baixo: 11% → 12% → 6%. O histórico ficou mais forte no teste.
- **Cestão 2 quando o cestão 1 deu carga alta:** 9% no treino, 17% na validação, 9% no teste; quando não deu, 19%, 21% e 16%. O efeito vale nas três partições, mas oscila (são só 33–144 casos).

### 5b. Backtesting walk-forward mês a mês (Guia §6.2; gráfico 04)

O oficial economiza minutos em **todos os 6 meses do dev** (de 13% em abril a 50% em fevereiro) e nos dois meses cheios do teste (48% e 50%). Agosto (12 cestões) não conta.

**Re-treinar todo mês dentro do teste** (com os dados do teste que já passaram): PR-AUC 0,59 (contra 0,57), mas 140 min (contra 131). Dentro do ruído; não há sinal de que o modelo "envelheça" em 40 dias.

### 5c. Robustez a erros de pesagem (Guia §6.2)

A receita dos cestões do teste foi recalculada com cada camada pesando `peso × (1 + ruído)`, 20 sorteios (antes, o script confere que **sem ruído** o recálculo reproduz exatamente as features):

| Erro da balança | Δ chance média | Alarmes que mudam | PR-AUC teste | Minutos teste |
|---|---|---|---|---|
| 5% | 0,009 | 2,7% | 0,563 (0,55–0,58) | 139 (128–153) |
| 10% | 0,015 | 4,1% | 0,560 (0,54–0,58) | 140 (127–158) |

Um erro de 10% na balança muda a chance em 1,5 ponto e 1 de cada 25 alarmes. O modelo é robusto.

### 5c-bis. Sensibilidade ao corte de peso da camada

As features foram refeitas a partir do arquivo bruto com cada corte, e o oficial foi **re-treinado** no dev:

| Corte | Camadas usadas | PR-AUC teste | Minutos teste |
|---|---|---|---|
| 0 (só peso zero sai) | 39.777 | 0,569 | 145 |
| 100 kg | 39.451 | 0,580 | 149 |
| 1.000 kg | 39.065 | 0,567 | 140 |
| **2.000 kg (o usado)** | 36.633 | 0,570 | 133 |

A PR-AUC fica em 0,57–0,58 com qualquer corte; os minutos variam dentro do ruído. Partindo do bruto, o corte de 2.000 kg não reproduz exatamente o oficial (0,570 × 0,567; o bruto não passa por toda a limpeza da EDA); a comparação justa é entre as linhas.

### 6.1 Critérios de aceitação (slide 35)

Proposta do grupo, em minutos, para **o próximo lote** (o teste atual já foi visto; aqui é ilustração). O orçamento de alarmes (4 por dia) é um parâmetro a confirmar com a EVCOMX.

| Critério | Regra | Dev | Teste |
|---|---|---|---|
| C1 economiza minutos | ≥ 20% de economia | sim (35%) | sim (48%) |
| C2 ordena melhor que o chute | PR-AUC ≥ 2 × taxa | sim (0,50 × 0,16) | sim (0,57 × 0,15) |
| C3 não perde para a regra do histórico | minutos ≤ regra + 10% | sim (616 × 816) | sim (131 × 147) |
| C4 faixas de risco em ordem | taxa real cresce com a faixa | **não** (3% < 13% = 13% < 26% < 55%) | sim (3% < 4% < 10% < 29% < 59%) |
| C5 cabe no pátio | ≤ 4 alarmes por dia | sim (2,3) | sim (3,7) |
| C6 (desejável) supera a RF antiga além do ruído | IC 90% dos minutos < 0 | — | **sim** (−55; −105 a −14) |
| C7 (desejável) supera a regra além do ruído | IC 90% dos minutos < 0 | — | **não** (−17; −59 a +21) |

- **C4 no dev:** as faixas 5–10% e 10–20% empatam (13%). As fronteiras das faixas devem ser revistas a cada re-treino, pelo volume (ex.: os 10% de maior risco = "muito alto").
- **C7** fica registrado como não atingido: no teste, a economia de minutos sobre a regra do histórico ainda cabe no ruído.

### 6.3 Validação de domínio: faz sentido físico? (Guia §6.3)

Mudando uma coisa de cada vez nos cestões do teste (a troca de material recalcula a receita inteira):

| Mudança | Esperado | Δ chance média | Cestões no sentido esperado | Bate? |
|---|---|---|---|---|
| +5 t de pesado no lugar de 5 t de leve | − | −0,037 | 76% | sim |
| +1 corrida com carga alta no histórico recente | + | +0,024 | 73% | sim |
| Com gusa líquida (+34 t) × sem | + | +0,102 | 100% | sim |
| Cestão 1 deu carga alta (só cestão 2) | − | −0,151 | 96% | sim |

Os 4 sentidos batem com a física e com o que os dados mostram (EDA, item 9).

### 6.4 Validação de negócio: vale a pena? (Guia §6.4; gráfico 06)

"Minutos evitados" supõe que **todo alarme certo evita a parada inteira**. E se não?

| O alarme certo evita… | Alarme certo custa 0 min | Alarme certo custa 1 min |
|---|---|---|
| 100% da parada | **48%** de economia (98 min/30 dias) | 25% |
| 75% | 27% | 4% |
| 50% | 6% | −17% |
| 25% | −15% | −38% |

(Sem modelo: 188 min de parada a cada 30 dias.)

**O ganho depende do que o pátio faz com o alarme.** Se o reajuste evita a parada inteira, o modelo economiza metade dos minutos. Se evita só metade da parada, o ganho praticamente some; se além disso o alarme certo custar 1 min, o modelo dá prejuízo. **É a pergunta mais importante para a EVCOMX agora**, mais que qualquer ajuste no modelo.

### 8. Conclusão: podemos confiar no modelo? (slides 37 e 38)

- **Foi construído como dissemos:** 29 conferências, nenhuma falha; alvo embaralhado vai ao chute; contas refeitas à mão.
- **Funciona fora da amostra, de forma estável:** walk-forward 0,50, validação 0,45, teste 0,57; economia em todos os meses; robusto a erro de balança, ao corte de peso e à semente.
- **Faz sentido físico** nos 4 sentidos testados.
- **O que ainda não está provado:** que ele economiza mais minutos que a regra simples do histórico (dentro do ruído no teste), e **quanto da parada o reajuste realmente evita** (seção 6.4). As duas respostas vêm do próximo lote e de um piloto no pátio.
