## Nesta pasta, há a Exploratory Data Analysis (EDA) acerca do dataset fornecido pela EVCOMX.

Projeto final do bootcamp de ciência de dados, em parceria com a EVCOMX: um modelo de classificação que prevê se uma receita de cestão vai gerar **carga alta** no forno elétrico a arco (FEA) durante o carregamento de sucata.

> **Atualização (nov/2026): a EDA e o modelo oficial.** O modelo oficial do projeto é a **média de 5 Gradient Boosting com 121 features** (ver `src/selecao_algoritmos/`). Ele se apoia muito em três informações que não estão na receita do cestão. A EDA ganhou o **ITEM 9** para mostrar, nos dados brutos, que elas existem e o tamanho delas (gráfico `14_sinais_do_modelo_oficial.png`). Os itens 1 a 8 não mudaram: a limpeza e o `base.csv` são os mesmos que alimentam todas as etapas. Antes, o modelo era uma Random Forest com 6 features.
>
> | Sinal (ITEM 9) | O que os dados mostram |
> |---|---|
> | **Histórico do forno** | Com 0 cargas altas nas 5 corridas anteriores, 14% das corridas dão carga alta; com 5 de 5, **77%**. Depois de uma corrida com carga alta, 55%; depois de uma sem, 19% |
> | **Gusa líquida** (carregamento 3) | Corridas **sem gusa: 4%** de carga alta; com 30–35 t, 34%; com 35–40 t, 45% |
> | **Cestão 1 → cestão 2** | Quando o cestão 1 deu carga alta, o cestão 2 da mesma corrida dá **menos**: 10%, contra 18% quando o cestão 1 não deu |

### Como rodar

O pipeline está dividido em dois arquivos, nessa ordem obrigatória:

1. **`transforma_dataset.py`** — Parte A (pipeline). Lê os 4 arquivos brutos do projeto, limpa e transforma os dados, e salva `base.csv` e `dataset_modelagem.csv` em disco. Rode do início ao fim antes do passo 2.
2. **`EDA.py`** — Parte B (análise exploratória). Lê os arquivos que a Parte A gerou e produz os gráficos (salvos em `graficos/`) e as investigações da EDA. **Não roda sem antes rodar o `transforma_dataset.py`** — ele depende de `base.csv` (já limpo) e de `dataset_modelagem.csv`.

Os dois arquivos usam marcadores `#%%`, então também dá pra rodar célula por célula no VS Code / Jupyter, na ordem em que aparecem. O `EDA.ipynb` junta as duas partes numa ordem só, já executado, com tabelas e gráficos.

### Arquivos de entrada (brutos, em `original_dataset/`)

- `infos_camadas.csv` — uma linha por corrida + carregamento + camada + material (peso, horário, baia).
- `infos_materiais.csv` — propriedades por material (densidade, rendimento metálico, energia elétrica). A coluna `Metálico` é, na prática, o código do material, apesar do nome.
- `dicionario_tipo_material.csv` — classificação de cada material em Leve / Misto / Pesado.
- `VariavelResposta.csv` — a variável resposta: se cada corrida/carregamento teve carga alta, e a duração da parada.

### Arquivos gerados pelo pipeline (`transforma_dataset.py`)

- `base.csv` — `infos_camadas` + `infos_materiais` + `dicionario_tipo_material` unidos (bloco A.0), já limpo (cd_baia, dt_hora_consumo, camadas abaixo de 2.000 kg removidas, peso zero incluído, Densidade/Rendimento como número) e com as colunas `peso_t` e `volume_m3`.
- `dataset_modelagem.csv` — uma linha por cestão (corrida + carregamento 1 ou 2), com as features agregadas (peso, volume, número de camadas, proporção de material leve) e a variável resposta (`carga_alta`). É o dataset pronto pra modelagem.
- `cestoes_acima_do_limite.csv` — gerado pela Parte B (ITEM 1), lista os cestões que passam da capacidade física (70 t ou 78 m³). Ideia do Lucas; o grupo decidiu **manter** — serve como lista operacional, propósito diferente do gráfico-resumo (Gráfico 10, ITEM 2).
- `graficos/` — todos os PNGs gerados pela Parte B.

### Estrutura da `EDA.py` (Parte B)

Organizada pelos itens do checklist de EDA do bootcamp:

| Item | Conteúdo |
|---|---|
| 1 — Entender os dados | tamanho, período, ligação entre tabelas, peso/volume por cestão, cestões acima da capacidade |
| 2 — Qualidade dos dados | nulos, duplicatas, peso zerado, consistência da variável resposta, gráficos-resumo dos problemas |
| 3 — Análise univariada | taxa de carga alta, carregamentos por tipo, duração das paradas, propriedades por material |
| 4 — Outliers | regra de negócio (capacidade física) x critério estatístico (IQR) |
| 5 — Features criadas | num_camadas, num_materiais, peso por classe, prop_leve |
| 6 — Análise bivariada | point-biserial, heatmap de correlação, peso x volume, posição da camada, **ciclos de montagem** (gráficos 12 e 13), comparação carregamento 1 x 2 |
| 7 — Padrões temporais | taxa de carga alta por turno e dia da semana |
| 8 — Dicionário de materiais | scatter densidade x energia, verificação da regra leve/pesado |
| 9 — Sinais do modelo oficial | histórico do forno, gusa líquida e cestão 1 → cestão 2 (gráfico 14) |

### Decisões já tomadas (documentadas no código)

- **Camadas abaixo de 2.000 kg são removidas, peso zero incluído** (Parte A, bloco A.3). A regra foi revista duas vezes com a EVCOMX:
  1. *Após a apresentação da EDA:* antes, o peso zero era imputado pela mediana do material. Mas peso zero significa que a camada estava na receita e o gancho não pegou o material: a camada nunca entrou no cestão. Passou a ser removido.
  2. *Resposta por e-mail (21/out/2026)* sobre as camadas de 10 kg e 100 kg: elas "também podem ser desconsideradas na montagem, dificilmente temos camadas com menos de 2000kg. Podem usar esse valor de corte." O corte passou de "peso = 0" para "peso < 2.000 kg".

  O corte remove **5.180 camadas (12% do bruto), mas só 2,5% do peso dos cestões 1 e 2**: 2.030 com peso zero, 326 entre 1 e 99 kg (249 delas com exatamente 10 kg) e 2.824 entre 100 kg e 2 t. Saem camadas de 2.663 cestões, mas **nenhum cestão inteiro** sai (o script para com erro se isso acontecer). Nem tudo abaixo de 2 t parece gancho vazio: entre 100 kg e 2 t a mediana é ~1,6 t. Por isso a verificação mede o quanto o modelo depende do corte (`src/verificacao_validacao/README.md`, seção 5c-bis): a PR-AUC muda menos de 0,03 entre os cortes de 0, 100, 1.000 e 2.000 kg, dentro do ruído.
- **A ordem leve → pesado → misto → leve se repete em vários ciclos dentro do mesmo cestão** (correção da EVCOMX). Ela não acontece uma vez só, com leve no fundo e no topo. Os gráficos 07 e 08 (fundo/meio/topo) só descrevem as pontas. A regra de montagem é analisada pelos gráficos 12 (sequência real das camadas) e 13 (carga alta por número de ciclos). Cestões com mais ciclos têm bem menos carga alta (cestão 1: 42% com 0 ou 1 ciclo, 7% com 5 ou mais; cestão 2: 26% contra 1%).

- **Outliers são mantidos**, não removidos — ficam marcados em duas flags (`acima_capacidade`, regra de negócio; `outlier_estatistico`, critério IQR), porque a maioria dos outliers estatísticos no carregamento 2 são cestões pesados legítimos, não erro de medição.
- **`num_materiais`** (quantos códigos de material diferentes entram no cestão) **foi incluída** como feature, como o Guia sugere (seção 8.2) — entra no `dataset_modelagem.csv` e na análise bivariada (point-biserial, boxplot, heatmap).
- O turno usado (madrugada/manhã/tarde/noite) é **diferente** do turno oficial da EVCOMX (Turno A/B/C) — decidir se troca ou se mantém e justifica na apresentação.

### Pontos ainda em aberto

- Nos dados, o ciclo aparece tanto como L→P→M quanto como L→M→P. Confirmar se as duas ordens contam como ciclo.

- Propriedades de RECC/RECG (sem densidade/energia/rendimento no dicionário): aguardando resposta do cliente.
- 28 linhas com chave corrida+carregamento+camada repetida: precisa investigar antes de decidir o que fazer.