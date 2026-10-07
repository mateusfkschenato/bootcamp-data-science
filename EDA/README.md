## Nesta pasta, há a Exploratory Data Analysis (EDA) acerca do dataset fornecido pela EVCOMX.

Projeto final do bootcamp de ciência de dados, em parceria com a EVCOMX: um modelo de classificação que prevê se uma receita de cestão vai gerar **carga alta** no forno elétrico a arco (FEA) durante o carregamento de sucata.

### Como rodar

O pipeline está dividido em dois arquivos, nessa ordem obrigatória:

1. **`transforma_dataset.py`** — Parte A (pipeline). Lê os 4 arquivos brutos do projeto, limpa e transforma os dados, e salva `base.csv` e `dataset_modelagem.csv` em disco. Rode do início ao fim antes do passo 2.
2. **`EDA.py`** — Parte B (análise exploratória). Lê os arquivos que a Parte A gerou e produz os gráficos (salvos em `graficos/`) e as investigações da EDA. **Não roda sem antes rodar o `transforma_dataset.py`** — ele depende de `base.csv` (já limpo) e de `dataset_modelagem.csv`.

Os dois arquivos usam marcadores `#%%`, então também dá pra rodar célula por célula no VS Code / Jupyter, na ordem em que aparecem.

### Arquivos de entrada (brutos, fornecidos pela EVCOMX)

- `infos_camadas.csv` — uma linha por corrida + carregamento + camada + material (peso, horário, baia).
- `infos_materiais.csv` — propriedades por material (densidade, rendimento metálico, energia elétrica). A coluna `Metálico` é, na prática, o código do material, apesar do nome.
- `dicionario_tipo_material.csv` — classificação de cada material em Leve / Misto / Pesado.
- `VariavelResposta.csv` — a variável resposta: se cada corrida/carregamento teve carga alta, e a duração da parada.

### Arquivos gerados pelo pipeline (`transforma_dataset.py`)

- `base.csv` — `infos_camadas` + `infos_materiais` + `dicionario_tipo_material` unidos (bloco A.0), já limpo (cd_baia, dt_hora_consumo, peso zero, Densidade/Rendimento como número) e com as colunas `peso_t` e `volume_m3`.
- `dataset_modelagem.csv` — uma linha por cestão (corrida + carregamento 1 ou 2), com as features agregadas (peso, volume, número de camadas, proporção de material leve) e a variável resposta (`carga_alta`). É o dataset pronto pra modelagem.
- `cestoes_acima_do_limite.csv` — gerado pela Parte B (ITEM 1), lista os cestões que passam da capacidade física (70 t ou 78 m³). **Ainda em avaliação** — ver nota abaixo.
- `graficos/` — todos os PNGs gerados pela Parte B.

### Estrutura da `EDA.py` (Parte B)

Organizada pelos itens do checklist de EDA do bootcamp:

| Item | Conteúdo |
|---|---|
| 1 — Entender os dados | tamanho, período, ligação entre tabelas, peso/volume por cestão, cestões acima da capacidade |
| 2 — Qualidade dos dados | nulos, duplicatas, peso zerado, consistência da variável resposta, gráficos-resumo dos problemas |
| 3 — Análise univariada | taxa de carga alta, carregamentos por tipo, duração das paradas, propriedades por material |
| 4 — Outliers | regra de negócio (capacidade física) x critério estatístico (IQR) |
| 5 — Features criadas | num_camadas, peso por classe, prop_leve |
| 6 — Análise bivariada | point-biserial, heatmap de correlação, peso x volume, posição da camada, comparação carregamento 1 x 2 |
| 7 — Padrões temporais | taxa de carga alta por turno e dia da semana |
| 8 — Dicionário de materiais | scatter densidade x energia, verificação da regra leve/pesado |
| 9 — Fechamento | conclusões (a escrever) |

### Decisões já tomadas (documentadas no código)

- **Outliers MANTIDOS!**, não removidos — ficam marcados em duas flags (`acima_capacidade`, regra de negócio; `outlier_estatistico`, critério IQR), porque a maioria dos outliers estatísticos no carregamento 2 são cestões pesados legítimos, não erro de medição.
- O turno usado (madrugada/manhã/tarde/noite) é **diferente** do turno oficial da EVCOMX (Turno A/B/C) — decidir se troca ou se mantém e justifica na apresentação.

### Pontos ainda em aberto

- `cestoes_acima_do_limite.csv`: ideia do Lucas, uso ainda não confirmado (lista operacional x redundante com o gráfico de resumo) — decidir com ele antes de cortar ou manter de vez.
- Propriedades de RECC/RECG (sem densidade/energia/rendimento no dicionário): aguardando resposta do cliente.
- 28 linhas com chave corrida+carregamento+camada repetida: precisa investigar antes de decidir o que fazer.
- Escrever o fechamento (ITEM 9): principais conclusões, o que confirma ou contradiz o que o Guia da EVCOMX já descreve, e limitações dos dados a reportar pra EVCOMX.