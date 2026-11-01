# Feature selection da rodada RF (6 features) (histórico)

**Esta pasta é registro, não é o resultado oficial.** Ela guarda a votação entre seis métodos (filtro, RFE, L1, Elastic Net, permutação, Boruta) que escolheu **6 features** para a Random Forest: `densidade_media`, `prop_peso_pesado`, `peso_por_ciclo_t`, `taxa_ca_ult5`, `taxa_ca_ult10`, `prop_PACB`, como estava em out/2026, antes da troca do modelo.

**Por que mudou.** Depois disso, a EVCOMX deu o critério de negócio (o melhor modelo é o que faz **perder menos minutos de forno**; alarme falso = 1 min) e confirmou que o resultado do cestão 1 é conhecido a tempo de mexer no cestão 2. Com isso, o conjunto oficial tem **121 features** (as 84 candidatas + 36 de histórico, gusa, ritmo e regime + o resultado do cestão 1). No walk-forward, as 6 features perdem ~130 min a mais que o conjunto oficial (seção 9 do script oficial). No teste (24/jun a 4/ago/2026): RF de 6 features, PR-AUC 0,40 e 186 min perdidos; modelo oficial, PR-AUC 0,57 e 131 min (sem modelo: 251 min).

**O que tem aqui:**
- `README_rodada_rf6.md`: o README da pasta naquela rodada, com todos os números e decisões;
- o script e o notebook já executado daquela rodada. **O script tem uma trava no começo e não roda daqui:** ele grava nas pastas oficiais e sobrescreveria as saídas do modelo oficial. Para reproduzir a rodada, use o commit do git anterior à troca;
- os CSVs, modelos e gráficos que ele gerou.

Os números de comparação "RF 6 features" que aparecem nas tabelas oficiais vêm do mesmo protocolo do modelo oficial (walk-forward e treino no dev). Por isso podem diferir um pouco dos daqui: por exemplo, PR-AUC 0,404 no teste com treino no dev, contra 0,391 aqui, com treino só no treino.
