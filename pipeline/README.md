# Pipeline com Docker

Três serviços, uma imagem:

| Serviço | O que faz | Tempo |
|---|---|---|
| `treino` | Vai do dado bruto até o modelo, com todas as etapas e conferências | ~40 min |
| `previsao` | Prevê corridas novas, com o histórico do forno calculado | ~20 s |
| `app` | Simulador em Streamlit, em http://localhost:8501 | sobe em segundos |

## Como rodar

Precisa do Docker Desktop. Na raiz do repositório:

```bash
docker compose build                  # monta a imagem (só na primeira vez ou se o código mudar)
docker compose run --rm treino        # do bruto até o modelo
docker compose run --rm previsao      # prevê o que estiver em previsao/entrada/
docker compose up app                 # abre o simulador
```

Outras opções do treino:

```bash
docker compose run --rm treino --listar
docker compose run --rm treino --ate selecao
docker compose run --rm treino --de analise
docker compose run --rm treino --apenas testes
```

Sem Docker, os mesmos comandos funcionam em Python 3.13: `pip install -r requirements.txt`, depois `python pipeline/treinar.py` ou `python pipeline/prever.py`.

## O treino

Roda, em ordem, os scripts que já existem em cada pasta, sem alterá-los. Para na primeira etapa que falhar. O log de cada etapa fica em `pipeline/logs/`.

| Etapa | O que faz |
|---|---|
| `dados` | Confere os 4 arquivos de `original_dataset/` |
| `limpeza` | `EDA/transforma_dataset.py` |
| `eda` | `EDA/EDA.py` |
| `features` | `feature_selection/feature_selection.py` |
| `selecao` | `selecao_algoritmos/selecao_algoritmos.py`: gera `modelo_final.joblib` |
| `analise` | `analise_modelos/analise_modelos.py` |
| `verificacao` | `verificacao_validacao/verificacao_validacao.py` |
| `interpretacao` | `interpretacao_modelos/interpretacao_modelos.py` |
| `app` | `app_streamlit/preparar_dados_app.py` |
| `conferencia` | Compara o modelo com `pipeline/referencia.json` |
| `testes` | `app_streamlit/test_motor.py` e `pipeline/test_previsao.py` |

## A previsão

**Entrada**, em `previsao/entrada/`:

- `camadas.csv`, no formato do `infos_camadas.csv`, com **todas** as corridas depois do fim do histórico (4/ago/2026): carregamentos 1 e 2 e, se houver, o 3 com a gusa líquida;
- `resultados.csv` (opcional), no formato do `VariavelResposta.csv`, com o que já se sabe dessas corridas. Campo vazio significa "ainda não se sabe".

Os dois arquivos são **acumulativos**: a corrida de ontem continua lá, porque ela é o histórico da corrida de hoje. A pasta `previsao/exemplo/` tem corridas inventadas nesse formato (`docker compose run --rm previsao --entrada previsao/exemplo`).

**Saída**, em `previsao/saida/previsoes.csv`, uma linha por cestão ainda sem resultado:

| Coluna | O que é |
|---|---|
| `chance`, `faixa_de_risco` | Chance de carga alta e a faixa para mostrar ao pátio |
| `limiar`, `alarme` | Limiar de minutos e se o cestão alarma |
| `modelo` | `principal`, ou `reserva` se o cestão 2 não tiver o resultado do cestão 1 |
| `avisos` | Quando a previsão merece cuidado |

`--conservador` usa o limiar em que o alarme certo também custa 1 min.

**Como o histórico é calculado.** `pipeline/features_corrida.py` limpa as camadas com as regras da EDA, calcula a receita com a conta do app e refaz histórico, ritmo e regime recente sobre a série inteira: corridas conhecidas + corridas novas.

**Avisos que a previsão pode dar:**

| Aviso | Significado |
|---|---|
| `BLOQUEADA` | Material sem cadastro: a corrida não é prevista |
| `histórico desatualizado` | Mais de 24 h desde a última corrida com resultado |
| `N das últimas 5 corridas sem resultado` | O histórico recente está incompleto, e esse é o maior sinal do modelo |
| `sem gusa informada` | O modelo trata como corrida sem gusa, e a chance cai muito |
| `modelo reserva` | O resultado do cestão 1 não foi informado |

## Conferências

`pipeline/test_previsao.py` finge que as últimas 300 corridas ainda não aconteceram, passa as camadas brutas pela previsão e exige as **mesmas 121 features** do `dataset_features_completo.csv` (diferença máxima de 1e-9) e os **mesmos alarmes**. As chances batem em quase todos os cestões. Em 10 de 568, a chance muda mais de 0,1 ponto, no máximo 1,5, porque o último dígito do número cai num corte de árvore; nenhum alarme muda. Também confere que o resultado da própria corrida não entra na previsão dela.

`pipeline/conferir_modelo.py` compara o modelo gerado com `pipeline/referencia.json`: com os mesmos dados e versões, os números têm de ser iguais.

## Dados e sigilo

A imagem leva **só código e bibliotecas**: o `.dockerignore` é uma lista branca. Dados e modelo entram pelos volumes do `docker-compose.yml`, então a imagem pode ser publicada sem levar dados da EVCOMX. Quem recebe a imagem precisa receber os dados e o modelo à parte.

## Limites

- **Lote novo de dados:** o treino está preparado para os dados atuais. Os meses do walk-forward estão fixos em `comum/protocolo.py`, as conferências esperam estes dados, e `pipeline/referencia.json` precisa ser atualizado depois de revisar os resultados (`python pipeline/conferir_modelo.py --atualizar`).
- **A previsão não se atualiza sozinha:** alguém precisa colocar as corridas e os resultados em `previsao/entrada/`. Ligar isso ao sistema da aciaria é um passo com a EVCOMX.
- **O histórico assume que o resultado da corrida anterior já é conhecido.** A EVCOMX confirmou isso para o cestão 1 da mesma corrida; para a corrida anterior, vale confirmar. Sem esse resultado, a previsão sai com aviso.
- **Previsão por lote:** roda sob demanda e grava um CSV. Ainda não é uma API.
