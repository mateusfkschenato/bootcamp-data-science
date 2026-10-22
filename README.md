# Previsão de carga alta no forno elétrico a arco

Projeto final do bootcamp de Ciência de Dados do Fundo Amanhã, feito com dados reais da **EVCOMX**.

## Equipe

| | |
|---|---|
| Geórgia Muniz | Ciência da Computação |
| Lucas Talamini | Estatística |
| Mateus Schenato | Ciência da Computação |
| Teresa Bauer | Engenharia Elétrica |

**Professor:** Fabrício Leal (Fundo Amanhã, Diretor Executivo e Vice-presidente)

---

## O problema

Numa aciaria, a sucata entra no forno elétrico a arco (FEA) em cestões, montados em camadas de material leve, misto e pesado. Às vezes o cestão fica "alto" demais e o forno precisa parar para acomodar a carga. A pergunta do projeto é simples:

> Olhando só para a receita do cestão, dá para saber antes se ele vai dar carga alta?

É um problema de classificação (sim ou não), com uma classe rara: só cerca de **15 a 18% dos cestões** têm carga alta.

<p align="center">
  <img src="EDA/graficos/11_carga_alta_por_faixa_de_peso.png" width="760" alt="Porcentagem de carga alta por faixa de peso do cestão">
</p>
<p align="center"><sub>Um dos primeiros achados da EDA: quanto mais pesado o cestão, menos carga alta. Até 60 t, a taxa fica acima da média; entre 65 e 70 t, cai para 1%.</sub></p>

---

## O caminho

> **Atualização (out/2026).** A EVCOMX respondeu que camadas de 10 kg e 100 kg também podem ser desconsideradas e deu **2.000 kg** como corte. A limpeza passou a remover todas as camadas abaixo de 2.000 kg (antes, só as de peso zero) e as cinco etapas rodaram de novo. O conjunto final caiu de 9 para 6 features e as conclusões se mantêm. Cada README tem o detalhe.

O projeto foi feito em cinco etapas, uma pasta para cada. Cada etapa lê o que a anterior gerou, então a ordem importa.

```
original_dataset/  →  EDA/  →  feature_selection/  →  selecao_algoritmos/  →  analise_modelos/  →  verificacao_validacao/
  dados brutos       limpeza     84 candidatas          4 algoritmos            métricas, custo       confere tudo, mede o
                     e gráficos  → 6 features           → Random Forest         e tradução para       drift, critérios de
                                                                                 a operação            aceitação e monitoramento
```

As bibliotecas e versões usadas estão em [`requirements.txt`](requirements.txt).

| Etapa | O que se fez | Detalhes |
|---|---|---|
| **EDA** | Junta as 4 tabelas, limpa (camadas abaixo de 2.000 kg saem: peso zero é o gancho que não pegou o material, e a EVCOMX deu 2 t como corte para as camadas muito leves), cria as primeiras features e investiga os dados. | [`EDA/README.md`](EDA/README.md) |
| **Feature selection** | Gera 84 features candidatas por cestão e escolhe 6 por votação entre seis métodos (filtro, RFE, L1, Elastic Net, permutação, Boruta). Entrega a tabela das features finais: como se calcula, por que entra e quando está disponível. | [`feature_selection/README.md`](feature_selection/README.md) |
| **Seleção de algoritmos** | Compara regressão logística, árvore, Random Forest e Gradient Boosting com validação cruzada temporal, contra três baselines (trivial, regra de volume e logística). Fica a Random Forest. | [`selecao_algoritmos/README.md`](selecao_algoritmos/README.md) |
| **Análise de modelos** | Mede o modelo com as métricas certas (tabela única com PR-AUC, recall, precisão, F2, Brier, MCC), compara com regras simples, calibra as chances, analisa erros e segmentos, explica o modelo e traduz os erros em alarmes e minutos de parada. | [`analise_modelos/README.md`](analise_modelos/README.md) |
| **Verificação e validação** | Confere automaticamente as etapas anteriores (31 conferências, com alvo embaralhado e testes da agregação), faz o backtesting mês a mês, testa erro de pesagem e o corte de peso das camadas, mede o quanto os dados mudaram depois de maio, valida domínio e negócio, propõe critérios de aceitação e um plano de monitoramento. Não escolhe nada de novo. | [`verificacao_validacao/README.md`](verificacao_validacao/README.md) |

Cada README de etapa tem as decisões, os números e as limitações com mais detalhe. Este aqui é o resumo.

As etapas de feature selection em diante seguem as aulas 05 a 07 do bootcamp e o **Guia de Modelagem Preditiva da EVCOMX**. Cada README tem uma seção dizendo onde está cada item do Guia.

---

## Os dados

São quatro arquivos fornecidos pela EVCOMX, cobrindo cerca de um ano de operação (ago/2025 a ago/2026):

- `infos_camadas.csv`: uma linha por camada de material em cada cestão (peso, horário, baia);
- `infos_materiais.csv`: densidade, rendimento metálico e energia de cada material;
- `dicionario_tipo_material.csv`: se cada material é leve, misto ou pesado;
- `VariavelResposta.csv`: se o cestão teve carga alta e quanto durou a parada.

A unidade de análise é o **cestão** (carregamento 1 ou 2 de cada corrida), o que dá uns 3.500 exemplos. A divisão é **70% treino / 15% validação / 15% teste, em ordem de tempo**, para que o modelo nunca aprenda com o futuro.

---

## O que encontramos

**As 6 features finais** têm leitura física direta: proporção de material pesado, densidade média do cestão, toneladas por ciclo de montagem, um tipo de pacote de chaparia (PACB) e a taxa de carga alta nas últimas 5 e 10 corridas.

**O volume total não é o que importa.** Os cestões são enchidos até volumes parecidos; o que muda é quanto peso cabe nesse volume. O sinal está na densidade e na proporção de pesado.

**Carga alta puxa carga alta.** O histórico recente do forno é o sinal mais forte de todos. Uma regra que só olha esse histórico, sem modelo nenhum, já fica muito acima do chute.

**Maio de 2026 mudou o jogo.** A partir de maio há menos material pesado e menos ciclos por cestão. Modelos que iam bem antes disso pioram depois, e isso aparece em todas as etapas.

**O modelo empata com a regra do histórico.** No teste (jun–ago/2026), a Random Forest tem PR-AUC 0,39, contra 0,36 de uma regra que só olha as últimas 5 corridas: a diferença está dentro do ruído. Com o mesmo número de alarmes, a regra acerta mais. Re-treinado todo mês (backtesting walk-forward), o modelo fica um pouco à frente em PR-AUC, mas empata nos meses mais recentes no orçamento de alarmes.

**A regra de volume do Guia não funciona nestes dados.** Alarmar quando o volume passa de um limiar fica no nível do chute (PR-AUC 0,15 no teste). O modelo a supera com folga.

**O modelo é tecnicamente sólido.** Com o alvo embaralhado, a nota cai ao acaso (sem vazamento escondido); um erro de 10% na balança muda a chance em ~0,04; trocar o corte das camadas leves (0, 100, 1.000 ou 2.000 kg) muda a PR-AUC em menos de 0,03; os sentidos das features batem com a física do forno. A chance crua vem inflada pelos pesos de classe, e a calibração de Platt corrige isso sem mudar a ordem dos cestões.

**Se vale a pena depende da EVCOMX.** O ganho em minutos de forno depende de quanto custa um alarme falso e de quantas cargas altas o pátio consegue evitar com o alarme. As tabelas de cenários (verificação, seção 6.4) estão prontas para receber esses números.
