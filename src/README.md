## Etapas do projeto

Aqui está o trabalho de cada aula, da análise dos dados à interpretação do modelo. Para ver como cada decisão foi tomada, comece por esta pasta. Para rodar tudo de uma vez, use a [`pipeline/`](../pipeline/README.md).

| Ordem | Pasta | Aula | O que faz | Para ler | Entrega principal |
|---|---|---|---|---|---|
| 1 | [`EDA/`](EDA/README.md) | EDA | Junta os dados brutos, limpa e investiga | [`EDA.ipynb`](EDA/EDA.ipynb) | `base.csv` |
| 2 | [`feature_selection/`](feature_selection/README.md) | Feature selection | Cria as 121 features e testa os grupos | [`feature_selection_carga_alta.ipynb`](feature_selection/feature_selection_carga_alta.ipynb) | `dataset_modelagem_fs.csv` |
| 3 | [`selecao_algoritmos/`](selecao_algoritmos/README.md) | Aula 05 | Compara os algoritmos, escolhe o limiar e treina o modelo | [`selecao_algoritmos.ipynb`](selecao_algoritmos/selecao_algoritmos.ipynb) | `modelo_final.joblib` |
| 4 | [`analise_modelos/`](analise_modelos/README.md) | Aula 06 | Mede o desempenho em minutos, PR-AUC e calibração | [`analise_modelos.ipynb`](analise_modelos/analise_modelos.ipynb) | `tabela_unica_teste.csv` |
| 5 | [`verificacao_validacao/`](verificacao_validacao/README.md) | Aula 07 | Confere os números, o drift e a robustez | [`verificacao_validacao.ipynb`](verificacao_validacao/verificacao_validacao.ipynb) | `checklist_verificacao.csv` |
| 6 | [`interpretacao_modelos/`](interpretacao_modelos/README.md) | Aula 08 | Explica o modelo e cada alarme | [`interpretacao_modelos.ipynb`](interpretacao_modelos/interpretacao_modelos.ipynb) | `explicacoes_alarmes_teste.csv` |
| — | [`otimizacao_minutos/`](otimizacao_minutos/README.md) | Registro | Mostra como o modelo oficial foi encontrado e simula três cenários operacionais | [`README.md`](otimizacao_minutos/README.md) | `finalistas_dev_pre_registro.csv` |

Cada etapa lê o que a anterior gerou, então a ordem importa. A `otimizacao_minutos/` é só registro e não entra no pipeline.

### O que há em cada etapa

- **`README.md`:** o resumo da etapa, com as decisões e os resultados.
- **Notebook `.ipynb`:** o código já executado, com as saídas ao lado.
- **Script `.py`:** o mesmo código, que é o que o pipeline roda. A EDA tem dois: `transforma_dataset.py` e `EDA.py`.
- **CSVs e `graficos/`:** os resultados gerados.

As regras usadas por todas as etapas (régua de minutos, limiar, walk-forward e a receita dos 5 Gradient Boosting) ficam em [`comum/protocolo.py`](../comum/protocolo.py).

### Como rodar

Da raiz do repositório:

```bash
python pipeline/treinar.py --listar        # mostra as etapas
python pipeline/treinar.py --apenas eda    # roda uma etapa
python pipeline/treinar.py --de analise    # recomeça de uma etapa até o fim
```

Os notebooks também rodam sozinhos, de dentro da própria pasta. Os resultados das etapas anteriores já estão no repositório.
