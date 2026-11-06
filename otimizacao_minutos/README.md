## Otimização por minutos perdidos: registro de onde veio o modelo oficial (out/2026)

> **Esta pasta é o registro da exploração, não o caminho oficial.** O modelo que nasceu aqui (média de 5 Gradient Boosting com 120 features + o resultado do cestão 1) virou o **modelo oficial do projeto** e foi distribuído pelas pastas de cada aula. Os scripts dos experimentos saíram da pasta e estão no histórico do Git (commit `f849472`).

Aqui ficam três coisas que não cabem numa aula: **como o modelo foi encontrado** (168 configurações), **a prova de que o teste foi aberto uma vez só**, para finalistas declarados antes, e **os cenários operacionais** (o que mudaria se a operação mudasse).

### Por que esta etapa existiu

Depois da interpretação da Random Forest, a EVCOMX deixou claro o critério de negócio: **o melhor modelo é o que faz perder menos minutos de forno**, e não o que acha mais cargas altas. Alarme falso = 1 min (reajustar a montagem); carga alta não detectada = a duração da parada. Em 31/out/2026, a EVCOMX confirmou também que, quando o cestão 2 ainda pode ser mexido, **já se sabe se o cestão 1 deu carga alta**.

### O protocolo (para os números serem honestos)

1. **Toda escolha usou só treino + validação** (o "dev", ago/25 a jun/26): walk-forward mensal (para cada mês de nov/25 a jun/26, treina com tudo antes e prevê o mês) e a prova treino → validação.
2. **O limiar de cada mês veio só dos meses anteriores** (o de menor minutos). Os minutos do dev contam de jan a jun/26: 950 min de parada sem modelo.
3. **Os finalistas foram registrados antes de abrir o teste**, em `finalistas_dev_pre_registro.csv`. Critério declarado: menos minutos no dev, com PR-AUC no walk-forward pelo menos igual à da RF do grupo.
4. **O teste (24/jun a 4/ago/26) foi aberto uma única vez** (`resultado_teste_final.csv`, `bootstrap_teste_final.csv`). Transparência: o grupo já tinha visto esse teste nas rodadas da RF; por isso nada foi escolhido olhando para ele.

### O que foi tentado (168 configurações em `experimentos_log.csv`)

| Ideia | Ajudou? |
|---|---|
| Histórico mais rico (últimas 3/5/10/20/50 corridas, médias exponenciais, última corrida, corridas desde a última carga alta, taxa nas últimas 8 h e 24 h) | Sim, um pouco |
| Gusa líquida da corrida (`gusl_t`, carregamento 3) | Sim, um pouco |
| Gradient Boosting com todas as features (120) | **Sim, nos minutos**: é melhor na faixa de recall 60–80%, onde cai o limiar de menor custo |
| Resultado do cestão 1 no cestão 2 (`carga_alta_c1`) | **Muito**: no teste, a PR-AUC vai de 0,44 a 0,57 |
| Risco por material e por baia; peso, volume e densidade do topo | Não |
| Pesos por recência e janelas de 3 ou 6 meses | Não (dentro do ruído) |
| Dar mais peso às cargas altas longas; modelar a duração da parada | Não. A duração não é previsível com estes dados (\|Spearman\| < 0,1) |
| Random Forest, Extra Trees, logística com splines, HGB com monotonia, combinações | Gradient Boosting foi o melhor |
| Rodada 2: tirar grupos de features, histórico por cestão, modelo separado por cestão, combinação de 10 HGB | Tudo dentro do ruído; a combinação perdeu na "segunda olhada" no teste |

Conclusão daquela rodada: com estes dados, o modelo chegou ao platô. Novas variações mexem a PR-AUC em ±0,02, menos que o ruído.

### Resultado no teste (aberto uma vez, 31/out/2026)

| Modelo | PR-AUC teste | Min. perdidos | Economia |
|---|---|---|---|
| **Escolhido (hoje: OFICIAL): 5 HGB + resultado do cestão 1** | **0,567** | **131** | **48%** |
| Principal sem o cestão 1 (hoje: reserva) | 0,443 | 165 | 34% |
| Histórico "seguro" (lag 2) | 0,436 | 148 | 41% |
| RF 6 features (o antigo plano A) | 0,404 | 186 | 26% |
| Regra: só o histórico | 0,358 | 147 | 41% |
| Sem modelo | — | 251 | — |

A análise completa desses números está em `analise_modelos/`.

### E se a operação mudasse? Três cenários (`cenarios_operacionais.py`)

O modelo fica igual. A pergunta é que mudança barata na regra de negócio reduziria os minutos. Simulação de nov/25 a ago/26. Os cenários 1 e 2 usam associações observadas: **não provam causa**, servem para propor um piloto.

- **Cenário 1, pausa preventiva antes do cestão 2** (`cenario1_*.csv`): quando o cestão 1 parou por carga alta, o cestão 2 da mesma corrida deu carga alta 10% das vezes, contra 19% quando não parou. Como substituta do reajuste, a pausa perde (economia de 14–19%, contra 41% do reajuste suposto), mas é a única ação com efeito **medido** nos dados.
- **Cenário 2, depois de uma carga alta, montar a corrida seguinte mais densa** (`cenario2_regra_pos_carga_alta.csv`): depois de uma carga alta, a seguinte dá carga alta 54% das vezes, contra 19%, e hoje o pátio não reage. Uma densidade mínima de 0,85 t/m³ depois de uma carga alta evitaria 26–33% das paradas, mas desde maio quase não há corridas tão densas e a sucata pesada custa mais. A regra se paga se (preço do pesado − preço do leve) por tonelada < ~0,18–0,22 × o valor de 1 minuto de forno.
- **Cenário 3, teto de alarmes por dia** (`cenario3_teto_alarmes.csv`): a média é ~2,5 alarmes por dia; um teto de 4 por dia já perde 11 pontos de economia.

### Arquivos

| Arquivo | O que é |
|---|---|
| `finalistas_dev_pre_registro.csv` | **Números do dev salvos antes de abrir o teste** (a prova do protocolo) |
| `resultado_teste_final.csv`, `bootstrap_teste_final.csv` | O teste, aberto uma vez, e a incerteza (bootstrap por dia) |
| `experimentos_log.csv` | O log das 168 configurações testadas |
| `cenarios_operacionais.py`, `cenario*.csv` | Os três cenários operacionais |
