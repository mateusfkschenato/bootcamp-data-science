<img src="static/marca/arcanum_logo.png" alt="Arcanum" width="220">

# Arcanum: simulador de carga alta (app Streamlit)

App da Aula 09 (Deploy e Visualização de Dados): o modelo escolhido do projeto sai do notebook e vira uma tela. Monte uma corrida camada por camada (cestão 1, cestão 2 e a gusa líquida) e o modelo diz se ela deve dar carga alta.

## Como rodar

O Streamlit não é um `.exe`. É uma biblioteca de Python: o `app.py` é um script comum, e o `streamlit run` liga um servidorzinho e abre a página no navegador (`http://localhost:8501`). O app fica no ar enquanto o terminal estiver aberto.

```bash
# na raiz do repositório (C:\projetos\bootcamp-data-science)
pip install -r app_streamlit/requirements.txt

cd app_streamlit
python -m streamlit run app.py
```

No Windows, se `streamlit run` der "não é reconhecido como nome de cmdlet", é porque a pasta `Scripts` do Python não está no PATH. O `python -m streamlit` funciona do mesmo jeito sem mexer no PATH. Rode de dentro de `app_streamlit/` para o Streamlit achar o tema em `.streamlit/config.toml`.

## O que dá para fazer

| Controle | O que acontece |
|---|---|
| **Quando alarmar?** (barra lateral) | Os dois limiares de minutos do modelo: o critério da EVCOMX (18%) e o mais conservador, em que o alarme certo também custa 1 min (30%). |
| **Nova corrida** | Abre uma corrida vazia na ponte rolante, com a panela de gusa líquida descendo. |
| **Novo cestão** | Um cestão desce pendurado na ponte rolante (até 2 por corrida). |
| **Inserir no cestão N** | A camada cai dentro do cestão. A altura dela é o **volume** (peso ÷ densidade do material), na mesma escala para todos; a linha tracejada é a boca do cestão (78 m³). |
| **Gusa líquida** | Liga ou desliga o 3º carregamento e a tonelagem. **Entra no modelo.** |
| **Enviar corrida** | Cada cestão ganha um selo com a chance de carga alta, e o painel de resultado diz se a corrida deve dar carga alta (alarma se qualquer cestão passar do limiar). |
| **Testar essa troca** | Quando há alarme, o app procura a troca de **uma** camada (não pesada → pesada, mesmo peso) que mais baixa a chance e deixa testar. |

Cada material tem uma cor sólida própria, com legenda embaixo do desenho: tons claros para leve, médios para misto e escuros para pesado. O laranja fica reservado para o alarme. Camadas de pesado ficam finas porque ocupam pouco volume, então o rótulo delas vai para o lado, com uma linha guia.

## Identidade visual

Laranja forte, preto e branco, como a EVCOMX. O laranja está em dois lugares: `LARANJA` em `desenho.py` e `primaryColor` em `.streamlit/config.toml`.

O logo do **Arcanum** fica em `static/marca/`: no cabeçalho, fixo no topo da página (logo inteiro na barra lateral, símbolo com ela fechada), no rodapé e no ícone da aba. As versões `_escuro` são para fundo preto.

A fonte é a **Bai Jamjuree**, carregada do Google Fonts (precisa de internet; sem ela, o navegador usa a fonte do sistema).

## Que modelo é esse

É o **modelo oficial do projeto**: `selecao_algoritmos/modelo_final.joblib`, a média das probabilidades de 5 `HistGradientBoostingClassifier`, com 120 features + o resultado do cestão 1. No teste (24/jun a 4/ago/2026): PR-AUC 0,57 e 131 min de forno perdidos, contra 251 min sem modelo. O pacote é carregado uma vez só, com `@st.cache_resource`.

| Grupo de features | Quantas | De onde o app tira |
|---|---|---|
| Receita do cestão (peso, volume, densidade, classes, materiais, ciclos, topo e fundo...) | 67 | Calculadas das camadas que você monta, com as regras de `feature_selection/feature_selection.py` |
| Cestão 1 → cestão 2 (`volume_c1_m3`, `delta_peso_12_t`, `mudou_composicao`...) | 6 | Calculadas da corrida montada |
| Gusa líquida (`gusl_t`, `tem_gusl`, `carga3_solida_t`) | 3 | Do controle de gusa (o app não tem carga sólida no 3º carregamento) |
| Histórico, ritmo e horário (`h_*`, `taxa_ca_ult*`, `gap_*`, `hora`, `turno`...) | 26 | Copiadas do **momento do forno** escolhido na tela (uma corrida real: típico, calmo ou agitado) |
| Receita comparada com os cestões anteriores (`rel_*`, `rel2_*`) | 14 | Feature da receita − média móvel naquele momento (a média sai de `X − rel_X` na corrida real) |
| Execução da montagem (`num_baias`, `tempo_montagem_min`, `intervalo_medio_min`) | 3 | **Estimadas**: baias = nº de materiais (vale em 3.497 de 3.501 cestões) e tempo = intervalo típico entre camadas × (camadas − 1) |
| `carga_alta_c1` | 1 | 0 no cestão 1. O cestão 2 usa o modelo reserva, que não tem essa feature, porque o app não pergunta o resultado do cestão 1 |

No app:

- **Cestão 1:** usa o modelo principal com `carga_alta_c1 = 0`, como ele foi treinado.
- **Cestão 2:** usa o **modelo reserva** do pacote (as mesmas 120 features, sem `carga_alta_c1`), porque o app não pergunta o resultado do cestão 1.
- **Momento do forno:** histórico, ritmo, horário e as médias móveis vêm de uma corrida real, escolhida na tela (típico, calmo ou agitado). As corridas simuladas no app não entram nesse histórico.
- **Corridas de exemplo:** o campo "Corrida de exemplo" abre receitas reais do período de teste e uma receita simulada sem materiais pesados (`dados_app/exemplos_corrida.json`) para a demonstração. O resultado aparece ao clicar em "Enviar corrida".
- **Recombinação:** em cada cestão com alarme, o "E agora?" procura até 3 mudanças (material mais denso com o mesmo peso ou inversão de camadas vizinhas de classes diferentes), mantendo o peso total, o volume até 78 m³, sem pesado no fundo ou no topo e dentro do p99 dos cestões reais. A auditoria com um segundo modelo, treinado em semanas alternadas, mostrou queda em 87% dos casos, mas de 15 pontos, contra 35 no modelo da busca: é uma sugestão a confirmar com o pátio e com um piloto.

O alarme dispara quando a chance passa do **limiar de minutos** que o pacote guarda (`limiar_minutos_TP0` ou `TP1`). Não é 50%: com parada média de ~3,9 min e alarme de 1 min, alarmar compensa bem antes disso.

**Chance da corrida.** Depois de enviar, o resultado mostra a chance de a corrida dar carga alta em algum cestão: `1 − (1 − P₁) × (1 − P₂)`, com P₂ = chance do cestão 2 se o cestão 1 não der. É a regra da cadeia, sem supor cestões independentes. Nas 245 corridas do teste com dois cestões, com o `modelo_dev`, a média prevista foi 31% e a taxa real 27%. O alarme não usa esse número: continua no limiar de minutos de cada cestão.

**Conferência.** O `test_motor.py` monta, para 400 corridas reais, a receita e o momento da própria corrida, e confere que as 120 features batem com o `feature_selection/dataset_features_completo.csv`. A única diferença é `num_camadas` em poucos cestões, em que os dados têm duas camadas com o mesmo número; no app, cada camada conta. Rodando em todas as 3.372 corridas com momento completo, também bate. Estimar baias e tempo de montagem muda a chance em 1,1 ponto percentual em média e troca o alarme em 2% dos cestões.

## Entradas recusadas

- Camada abaixo de 2 t (corte da EVCOMX: gancho que não pegou o material) ou acima de 15 t.
- Cestão acima de 80 t, 110 m³ estimados ou 20 camadas. Nos dados reais, o máximo é 78 t, 106 m³ e 18 camadas.

## Gusa líquida

O carregamento 3 **não é sempre** de gusa líquida:

- 1.436 das 1.799 corridas (80%) têm gusa líquida no 3º carregamento, em geral entre 30 e 42 t.
- 363 corridas (20%) têm só os cestões 1 e 2.
- 97 corridas têm só o cestão 1 e a gusa.

Nas corridas **sem** gusa líquida, só 4% deram carga alta, contra 37% nas corridas com gusa, e isso se repete em todos os meses. O modelo de minutos usa `gusl_t` e `tem_gusl`, então ligar ou desligar a gusa muda a previsão.

## Arquivos

```
app_streamlit/
├── app.py                  # a tela (Streamlit)
├── motor.py                # receita + momento → 120 features → chance (sem Streamlit; testável)
├── desenho.py              # SVG animado dos cestões e painel de resultado
├── preparar_dados_app.py   # gera dados_app/ a partir das etapas anteriores
├── test_motor.py           # confere as features contra o dataset de modelagem
├── dados_app/
│   ├── materiais_app.csv   # 22 materiais: classe, densidade, rendimento, energia, camada típica
│   ├── momentos.csv        # 1.686 momentos reais do forno (o app usa um; o teste usa 400)
│   └── config_app.json     # limiares de minutos, momento típico, referência do treino, estimativas
├── .streamlit/config.toml  # cores da EVCOMX
└── requirements.txt
```

Se o modelo ou as features mudarem, rode `python app_streamlit/preparar_dados_app.py` e `python -m pytest app_streamlit/test_motor.py -q` de novo.

## Publicar (bônus do exercício)

O jeito mais simples é o **Streamlit Community Cloud**: suba o repositório para o GitHub, crie um app apontando para `app_streamlit/app.py` e ele usa o `requirements.txt` desta pasta. Para rodar, o app só precisa de `selecao_algoritmos/modelo_final.joblib` e da pasta `dados_app/`.

Antes de publicar num link aberto, confirme com a EVCOMX: `dados_app/` vem dos dados reais deles.
