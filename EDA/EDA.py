# =====================================================================
# [NOTA DE ORGANIZAÇÃO — não faz parte do código original]
# Este arquivo é a PARTE B (FORA DO PIPELINE) do script, agora separada
# em um .py próprio. Antes, Parte A e Parte B viviam no mesmo arquivo e
# a Parte B reaproveitava, em memória, o DataFrame "cestao" que a Parte
# A deixava pronto. Com os arquivos separados isso não existe mais.
#
# >>> RODE O transforma_dataset.py (Parte A) ANTES DESTE ARQUIVO. <<<
# Ele é quem gera base.csv (limpo) e dataset_modelagem.csv, que este
# arquivo lê do disco.
#
# A Parte B está organizada pelos itens do checklist da EDA, nesta ordem:
#   ITEM 1 - Entender os dados
#   ITEM 2 - Qualidade dos dados
#   ITEM 3 - Análise univariada
#   ITEM 4 - Outliers
#   ITEM 5 - Features criadas
#   ITEM 6 - Análise bivariada
#   ITEM 7 - Padrões temporais
#   ITEM 8 - Dicionário de materiais
#   ITEM 9 - Fechamento
# Cada gráfico/célula foi reclassificado pra debaixo do item a que ele
# realmente responde, não pela ordem histórica em que foi escrito. Como
# a seção "GRÁFICOS 8 A 18" sempre foi UM script contínuo (sem #%% entre
# os gráficos individuais), reordenar os gráficos dentro dela é só
# reordenar linhas de um mesmo bloco - não quebra nenhuma célula do
# Jupyter. A configuração comum desses gráficos (leitura de
# base/resposta, funções mostrar/br/pct, cores, pasta "graficos/") teve
# que ficar num único lugar (logo no início do ITEM 2, que é quem
# primeiro precisa dela) porque é usada também pelos itens 3 e 6 - isso
# está anotado onde acontece. Nenhum conteúdo foi cortado nesta
# passada, só reclassificado; os cortes já feitos na versão anterior
# (Gráficos 1, 2, 4, 5, 8, Passo 13, Passo 15) continuam fora.
#
# [AJUSTE P/ ARQUIVO SEPARADO] A única mudança de conteúdo nesta
# separação está no início do ITEM 6 (Análise bivariada): em vez de
# reaproveitar o "cestao" que a Parte A deixava em memória, agora se lê
# o dataset_modelagem.csv que a Parte A já salva em disco - é a mesma
# informação (cestao filtrado pros carregamentos 1 e 2, já com a coluna
# carga_alta), só que lida do arquivo em vez de herdada do kernel.
# =====================================================================
#%%
import os
assert os.path.exists("dataset_modelagem.csv"), "Rode o transforma_dataset.py primeiro!"
# =====================================================================
# ITEM 1 — ENTENDER OS DADOS
# Tamanho do dataset, período coberto, se as tabelas se ligam
# corretamente pela corrida, peso/volume de cada cestão e quantos
# passam da capacidade física.
# =====================================================================

#%%
#import shutil
#shutil.copy("base_original.csv", "base.csv")
#print("base.csv restaurado a partir do backup.")
# %% PASSO 1 - Importar as bibliotecas
import pandas as pd                 # tabelas
import matplotlib.pyplot as plt     # gráficos

# %% PASSO 2 - Ler os arquivos
base = pd.read_csv("base.csv")
resposta = pd.read_csv("VariavelResposta.csv")

print("base:", base.shape)          # (linhas, colunas)
print("resposta:", resposta.shape)
base.head()                         # mostra as 5 primeiras linhas

#%% PASSO 3
for col in ["Densidade t/m3", "Rendimento Metálico %"]:
    if not pd.api.types.is_numeric_dtype(base[col]):
        base[col] = base[col].str.replace(",", ".").astype(float)

base["dt_hora_consumo"] = pd.to_datetime(base["dt_hora_consumo"], utc=True, format="ISO8601")
base["peso_t"] = base["qt_peso_carregamento"] / 1000

# %% PASSO 4 - Tamanho dos dados e período
print("Número de corridas:", base["cd_corrida"].nunique())
print("Primeira data:", base["dt_hora_consumo"].min())
print("Última data:  ", base["dt_hora_consumo"].max())

# Quantos carregamentos? Um carregamento = uma combinação corrida + número.
carregamentos = base[["cd_corrida", "nu_carregamento"]].drop_duplicates()
print("Número de carregamentos:", len(carregamentos))
print(carregamentos["nu_carregamento"].value_counts())

# %% PASSO 5 - As duas tabelas se ligam pela corrida?
corridas_base = set(base["cd_corrida"])
corridas_resposta = set(resposta["cd_corrida"])
print("Corridas só na base:", len(corridas_base - corridas_resposta))
print("Corridas só na resposta:", len(corridas_resposta - corridas_base))
# 0 e 0 significa que as duas tabelas têm exatamente as mesmas corridas.

# %% PASSO 6 - Peso e volume de cada cestão
# Só os carregamentos 1 e 2 são cestões. O 3 é gusa líquido.
cestoes = base[base["nu_carregamento"].isin([1, 2])].copy()

# Volume de cada camada = peso (t) dividido pela densidade (t/m³)
cestoes["volume_m3"] = cestoes["peso_t"] / cestoes["Densidade t/m3"]

# Somamos as camadas de cada cestão (corrida + carregamento)
total = cestoes.groupby(["cd_corrida", "nu_carregamento"])[["peso_t", "volume_m3"]].sum()
total = total.reset_index()
print(total.groupby("nu_carregamento").describe().T)

# %% PASSO 7 - Quais cestões passam da capacidade? (70 t ou 78 m³)
# [EM AVALIAÇÃO] o cestoes_acima_do_limite.csv foi ideia do Lucas.
# O Gráfico 10 (ITEM 2 - QUALIDADE DOS DADOS, mais abaixo) já mostra a
# mesma informação de forma agregada (quantos cestões, por qual
# motivo) — então, pra fins de EDA, esse CSV não é estritamente
# necessário. Mas se a ideia for usar como lista operacional (ex:
# entregar pra EVCOMX conferir manualmente quais corridas excederam),
# aí tem um propósito diferente do gráfico e vale manter. Decidir com
# o Lucas antes de cortar.
passou = total[(total["peso_t"] > 70) | (total["volume_m3"] > 78)]
print("Cestões acima do limite:", len(passou))
print("  acima de 70 t (peso):", (total["peso_t"] > 70).sum())
print("  acima de 78 m³ (volume):", (total["volume_m3"] > 78).sum())
passou.to_csv("cestoes_acima_do_limite.csv", index=False)


# =====================================================================
# ITEM 2 — QUALIDADE DOS DADOS
# Identificação de nulos, duplicatas, peso zerado, consistência da
# variável resposta, e os gráficos/resumos que sintetizam esses
# problemas. As decisões de limpeza em si (o que fazer com cada caso)
# já estão implementadas na Parte A; aqui fica só a investigação que
# levou a cada decisão.
# =====================================================================

# %% PASSO 8 - Valores ausentes (nulos)
porcentagem_nulos = base.isnull().mean() * 100
print(porcentagem_nulos.round(2).sort_values(ascending=False))

# %% PASSO 9 - Por que o cd_baia tem nulos?
# Vamos ver em que material e em que carregamento os nulos aparecem.
sem_baia = base[base["cd_baia"].isnull()]
print("Nulos por material:")
print(sem_baia["cd_codigo_material"].value_counts())
print("\nNulos por carregamento:")
print(sem_baia["nu_carregamento"].value_counts())
# Resultado: todos são GUSL (gusa líquido), que não vem de baia.
# Como dá para explicar o nulo por outra coluna, é MAR (ausência estrutural).

# %% PASSO 10 - Materiais sem densidade (sem correspondência no dicionário)
sem_densidade = base[base["Densidade t/m3"].isnull()]
print(sem_densidade[["cd_corrida", "cd_codigo_material", "tp_material"]])

# %% PASSO 11 - Duplicatas
print("Linhas totalmente iguais:", base.duplicated().sum())

colunas_chave = ["cd_corrida", "nu_carregamento", "nu_camada"]
print("Mesma corrida/carregamento/camada repetida:", base.duplicated(colunas_chave).sum())

# %% PASSO 12 - Peso zerado
peso_zero = base[base["qt_peso_carregamento"] == 0]
print("Linhas com peso 0:", len(peso_zero))
print(peso_zero["nu_carregamento"].value_counts())

# %% PASSO 14 - A variável resposta está consistente?
# Carga alta geral deve ser 1 quando carregamento 1 OU 2 teve carga alta.
c1 = resposta["is_carga_alta_carregamento_1"]
c2 = resposta["is_carga_alta_carregamento_2"]
geral = resposta["is_carga_alta"]
print("Geral diferente de (c1 ou c2):", ((c1 | c2) != geral).sum())

# Duração total deve ser a soma das duas durações.
d1 = resposta["qt_segundos_duracao_parada_carregamento_1"]
d2 = resposta["qt_segundos_duracao_parada_carregamento_2"]
dtotal = resposta["qt_segundos_parada_carga_alta"]
print("Duração total diferente de d1 + d2:", ((d1 + d2) != dtotal).sum())

# Flag sem duração, ou duração sem flag
print("Flag 1 sem duração:", ((c1 == 1) & (d1 == 0)).sum())
print("Duração 1 sem flag:", ((c1 == 0) & (d1 > 0)).sum())

# Carga alta no carregamento 2, mas a corrida não tem carregamento 2 na base
corridas_com_c2 = set(base[base["nu_carregamento"] == 2]["cd_corrida"])
estranhas = resposta[(c2 == 1) & (~resposta["cd_corrida"].isin(corridas_com_c2))]
print("Carga alta no c2 sem c2 na base:", len(estranhas))

# [REMOVIDO] Passo 13 (montagem: material Leve fora do lugar) -
# duplicado pelos Gráficos 14 e 15 (ITEM 6 - ANÁLISE BIVARIADA, mais
# abaixo). Passo 15 (print da taxa de carga alta) - duplicado pelo
# Gráfico 3 (ITEM 3 - ANÁLISE UNIVARIADA).

# -----------------------------------------------------------------
# Identificação coluna a coluna (qual linha está nula em cada coluna)
# -----------------------------------------------------------------
#%%
base[base["pk_aci_corrida"].isnull()]

# %%
base[base["nu_carregamento"].isnull()]

# %%
base[base["nu_camada"].isnull()]

# %%
base[base["cd_codigo_material"].isnull()]
# %%
base[base["qt_peso_carregamento"].isnull()]

# %%
base[base["ds_descricao_material_x"].isnull()]
# %%
base[base["cd_baia"].isnull()]
# %%
base[base["dt_hora_consumo"].isnull()]
# %%
base[base["ds_descricao_material_y"].isnull()]
# %%
base[base["tp_material"].isnull()]
# %%
base[base["Metálico"].isnull()]
# %%
base[base["Energia Elétrica"].isnull()]

# %%
base[base["Densidade t/m3"].isnull()]
# %%
base[base["Rendimento Metálico %"].isnull()]

# Como é possivel ver, as colunas com linhas nulas são: cd_baia, dt_hora_consumo, Metálico, Energia Elétrica, Densidade t/m3 e Rendimento Metálico %.

# -----------------------------------------------------------------
# 1) LIMPEZA DE FATO - decisões sobre cada nulo/duplicata/peso zero
# -----------------------------------------------------------------
# [ ] cd_baia e dt_hora_consumo nulos: já sabemos que é 100% GUSL
#     (gusa líquido, não vem de baia nem tem hora de "consumo" de
#     sucata). Decidir: mantém o nulo (é estrutural, não é erro) ou
#     preenche com algum marcador tipo "N/A - gusa líquido"? Documentar
#     a decisão e o porquê.

# [ORGANIZAÇÃO] A célula de limpeza de cd_baia/dt_hora_consumo que
# estava aqui foi movida pra Parte A, bloco A.1.

# [ ] Metálico, Energia Elétrica, Densidade t/m3, Rendimento Metálico %
#     nulos: isso acontece pra linhas cujo cd_codigo_material não bate
#     com nenhum material do dicionário (RECG, RECC, e também os
#     insumos CAL/COQUE, que têm código mas não têm essas propriedades
#     por não serem sucata metálica). Decidir, pra cada caso:
#       - RECG/RECC: dá pra conseguir a densidade/energia/rendimento
#         de outra forma (perguntar pro cliente, assumir valor de um
#         material parecido)? Ou essas linhas ficam de fora de contas
#         que dependem de volume_m3 (já que volume = peso/densidade
#         não dá pra calcular sem densidade)?
#       - CAL/COQUE (insumos): confirmar se eles devem ENTRAR na conta
#         de peso/volume do cestão ou se devem ser removidos antes de
#         calcular peso_t e volume_m3 por cestão (o guia oficial trata
#         eles como "insumos", não como sucata metálica - então
#         provavelmente não deveriam contar no peso que define carga
#         alta, mas isso precisa ser confirmado e registrado).
#
# [ ] As 2 linhas 100% duplicadas: decidir se remove ou investiga se
#     são duas corridas reais que coincidem em tudo.

#%%
dup_total = base[base.duplicated(keep=False)]
print("Índice no DataFrame:", dup_total.index.tolist())
print("Linha no arquivo CSV:", [i + 2 for i in dup_total.index.tolist()])

# [ORGANIZAÇÃO] A célula que remove as duplicatas (drop_duplicates) foi
# movida pra Parte A, bloco A.2.
#
# [ ] As 28 linhas com mesma chave corrida+carregamento+camada
#     repetida: isso é mais grave que duplicata simples - pode ser
#     erro de sistema (mesma camada registrada duas vezes com pesos
#     diferentes?). Precisa investigar essas 28 linhas especificamente
#     antes de decidir o que fazer.
#
# [ ] As 2030 linhas com peso = 0: decidir se é erro de registro
#     (remover), camada cancelada (remover ou manter como categoria
#     própria) ou outra coisa. Isso é MUITA linha pra simplesmente
#     ignorar sem entender a causa.

# [ORGANIZAÇÃO] A célula que imputa o peso zero pela mediana do
# material foi movida pra Parte A, bloco A.3.
#
# [ ] Depois de decidir tudo isso, documentar num resumo tipo "decisão
#     de limpeza": coluna / problema / decisão / justificativa. É isso
#     que os avaliadores querem ver (Guia, seção 10.1: "Visão crítica -
#     identificar limitações dos dados").

# -----------------------------------------------------------------
# Configuração comum dos gráficos (usada por este item e pelos itens
# 3 e 6 mais abaixo - leitura de base/resposta, funções mostrar/br/
# pct, cores e pasta de saída). Só precisa rodar uma vez.
# -----------------------------------------------------------------
#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MOSTRAR_NA_TELA = False     # mude para False para só salvar os PNGs, sem abrir janelas

os.makedirs("graficos", exist_ok=True)

# Cores com significado fixo em todos os gráficos
AZUL, VERMELHO, CINZA, LARANJA = "#2b6cb0", "#c0392b", "#7f8c8d", "#dd8a2e"

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.axisbelow": True,
})


def mostrar(nome):
    """Ajusta o layout, salva o PNG e (opcionalmente) mostra o gráfico."""
    plt.tight_layout()
    plt.savefig(f"graficos/{nome}.png", dpi=130)
    if MOSTRAR_NA_TELA:
        plt.show()
    else:
        plt.close()


def br(numero):
    """1234 -> '1.234' (milhar com ponto, como no Brasil)."""
    return f"{numero:,}".replace(",", ".")


def pct(valor, casas=1):
    """14.3 -> '14,3%' (vírgula decimal)."""
    return f"{valor:.{casas}f}".replace(".", ",") + "%"


# ---------- Ler e preparar os dados (base pra todos os gráficos) ----------
base = pd.read_csv("base.csv")
resposta = pd.read_csv("VariavelResposta.csv")

if not pd.api.types.is_numeric_dtype(base["Densidade t/m3"]):
    base["Densidade t/m3"] = base["Densidade t/m3"].str.replace(",", ".").astype(float)

base["peso_t"] = base["qt_peso_carregamento"] / 1000

# Só os cestões (carregamentos 1 e 2) e o volume de cada camada
cestoes = base[base["nu_carregamento"].isin([1, 2])].copy()
cestoes["volume_m3"] = cestoes["peso_t"] / cestoes["Densidade t/m3"]

# Peso e volume de cada cestão (uma linha por corrida + carregamento)
total = cestoes.groupby(["cd_corrida", "nu_carregamento"])[["peso_t", "volume_m3"]].sum()
total = total.reset_index()

# Junta com a resposta e cria a coluna "teve carga alta neste cestão?"
juntos = total.merge(resposta, on="cd_corrida")
juntos["carga_alta"] = np.where(
    juntos["nu_carregamento"] == 1,
    juntos["is_carga_alta_carregamento_1"],
    juntos["is_carga_alta_carregamento_2"],
)

# [REMOVIDO] Gráfico 8 (corridas por mês) - já estava inteiro
# comentado/desativado no código original, não rodava nada. Removido
# de vez.

# ---------- GRÁFICO 13 - cd_baia ausente: só no gusa líquido
eh_gusl = base["cd_codigo_material"] == "GUSL"
baia_nula = base["cd_baia"].isnull()
taxa_gusl = baia_nula[eh_gusl].mean() * 100
taxa_outros = baia_nula[~eh_gusl].mean() * 100
# Isso mostra que se, e somente se, o material for GUSL, o cd_baia é nulo.

fig, ax = plt.subplots(figsize=(6.5, 4.5))
nomes = ["GUSL\n(gusa líquido)", "Os outros 22\nmateriais"]
barras = ax.bar(nomes, [taxa_gusl, taxa_outros], color=[VERMELHO, AZUL])
ax.bar_label(barras, labels=[
    f"{pct(taxa_gusl, 0)}\n({br(baia_nula[eh_gusl].sum())} de {br(eh_gusl.sum())})",
    f"{pct(taxa_outros, 0)}\n({br(baia_nula[~eh_gusl].sum())} de {br((~eh_gusl).sum())})",
], padding=3)
ax.set_ylim(0, 125)
ax.set_title(
    "Ausência de cd_baia",
    loc="left", fontweight="bold", fontsize=11,
)
ax.set_ylabel("% de cd_baia ausente")
mostrar("13_cd_baia_ausente")

# ---------- GRÁFICO 10 - Cestões que excedem o limite, por motivo
acima_peso = total["peso_t"] > 70
acima_volume = total["volume_m3"] > 78
so_volume = (acima_volume & ~acima_peso).sum()
so_peso = (acima_peso & ~acima_volume).sum()
os_dois = (acima_peso & acima_volume).sum()
excedem = so_volume + so_peso + os_dois

fig, ax = plt.subplots(figsize=(7, 4.5))
nomes = ["Só volume\n(> 78 m³)", "Só peso\n(> 70 t)", "Peso e volume"]
valores = [so_volume, so_peso, os_dois]
barras = ax.bar(nomes, valores, color=VERMELHO)
ax.bar_label(barras, labels=[br(v) for v in valores], padding=3)
ax.set_title(
    "Cestões que excederam o limite",
    loc="left", fontweight="bold", fontsize=11,
)
ax.set_ylim(0, max(valores) * 1.12)
ax.set_ylabel("cestões")
mostrar("10_cestoes_que_excedem")

# ---------- GRÁFICO 16 - Resumo dos problemas de qualidade (cada barra tem sua unidade)
flag_c2_sem_c2 = (
    (resposta["is_carga_alta_carregamento_2"] == 1)
    & (~resposta["cd_corrida"].isin(corridas_com_c2))
).sum()

problemas = {
    "Peso = 0 (linhas)": (base["qt_peso_carregamento"] == 0).sum(),
    "Cestões acima de 70 t ou 78 m³ (cestões)": excedem,
    "Sem horário, fora do GUSL (linhas)": (base["dt_hora_consumo"].isnull() & ~eh_gusl).sum(),
    "Carreg. 3 com sucata, não GUSL (linhas)": ((base["nu_carregamento"] == 3) & ~eh_gusl).sum(),
    "Chave de camada repetida (linhas)": base.duplicated(colunas_chave).sum(),
    "Sem densidade: RECG e RECC (linhas)": base["Densidade t/m3"].isnull().sum(),
    "Carga alta no carreg. 2 sem ele na base (corridas)": flag_c2_sem_c2,
    "Linhas 100% duplicadas (linhas)": base.duplicated().sum(),
}
problemas = pd.Series(problemas).sort_values()

fig, ax = plt.subplots(figsize=(10, 4.8))
barras = ax.barh(problemas.index, problemas.values, color=VERMELHO, alpha=0.85)
ax.bar_label(barras, labels=[br(int(v)) for v in problemas.values], padding=3)
ax.set_xlim(0, problemas.max() * 1.12)
fig.suptitle(
    '"Problemas"',
    x=0.01, ha="left", fontweight="bold", fontsize=11,
)
ax.set_xlabel("ocorrências (a unidade está no nome de cada barra)")
mostrar("16_resumo_de_problemas")

# -----------------------------------------------------------------
# Investigação: as 152 linhas "sem_hora_suspeito" (dt_hora_consumo
# nulo, fora do GUSL) - resumo da investigação que levou à decisão já
# implementada na Parte A, bloco A.1.
# -----------------------------------------------------------------
# [CONDENSADO] A investigação original (7 células de código) confirmou:
# 152 linhas sem dt_hora_consumo, fora do GUSL, concentradas em 13
# corridas - cada uma com só uma fração das suas linhas sólidas
# afetada (nunca 100% de uma corrida), ou seja, falha pontual de
# registro, não um padrão estrutural. O peso dessas linhas é normal
# (não é zero nem nulo), reforçando que é só a hora que falhou, não o
# resto do registro. Decisão: estimar a hora pela mediana do
# carregamento irmão (1 ou 2) da mesma corrida, marcando
# dt_hora_consumo_estimada = True (implementado na Parte A, bloco A.1).


# =====================================================================
# ITEM 3 — ANÁLISE UNIVARIADA
# Cada variável olhada sozinha: taxa de carga alta, carregamentos por
# tipo, duração das paradas, propriedades dos materiais e número de
# camadas por cestão.
# =====================================================================

# %% GRÁFICO 3 - Taxa de carga alta
nomes = ["Geral", "Carreg. 1", "Carreg. 2"]
taxas = [geral.mean() * 100, c1.mean() * 100, c2.mean() * 100]
plt.bar(nomes, taxas, color=["red", "blue", "orange"])
plt.title("Taxa de carga alta (%)")
plt.ylabel("% das corridas")
plt.show()

# ---------- GRÁFICO 9 - Carregamentos por tipo
carregamentos = base[["cd_corrida", "nu_carregamento"]].drop_duplicates()
contagem = carregamentos["nu_carregamento"].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(7, 4.5))
nomes = ["Carregamento 1\n(cestão)", "Carregamento 2\n(cestão)", "Carregamento 3\n(gusa líquido)"]
barras = ax.bar(nomes, contagem.values, color=[AZUL, LARANJA, CINZA])
ax.bar_label(barras, labels=[br(v) for v in contagem.values], padding=3)
ax.set_title(
    "Quantidade de carregamentos",
    loc="left", fontweight="bold",
)
ax.set_ylim(0, contagem.max() * 1.12)
ax.set_ylabel("carregamentos")
mostrar("09_carregamentos_por_tipo")

# ---------- GRÁFICO 12 - Duração das paradas por carga alta
paradas = resposta[resposta["is_carga_alta"] == 1]["qt_segundos_parada_carga_alta"]
mediana = paradas.median()
media = paradas.mean()
minutos_total = paradas.sum() / 60

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(paradas, bins=40, color=VERMELHO, alpha=0.85)
ax.axvline(mediana, color="black", linestyle="--", label=f"mediana: {mediana:.0f} s")
ax.axvline(media, color=AZUL, linestyle="--", label=f"média: {media:.0f} s")
ax.set_title(
    "Tempo de parada",
    loc="left", fontweight="bold", fontsize=11,
)
ax.set_xlabel("duração da parada (segundos)")
ax.set_ylabel("corridas com carga alta")
ax.legend()
mostrar("12_duracao_das_paradas")

# -----------------------------------------------------------------
# 2) ANÁLISE UNIVARIADA - dicionário de materiais e num_camadas
# -----------------------------------------------------------------
# Falta:
# [ ] Univariada de Energia Elétrica, Densidade e Rendimento Metálico
#     - essas são propriedades por MATERIAL (só 24 materiais), não por
#     corrida, então não precisa ser histograma - uma tabela/describe()
#     já responde: qual material tem a menor/maior densidade, qual
#     consome mais energia, etc. (Guia, seção 3.1)
# [ ] Univariada de num_camadas por carregamento (quantas camadas, em
#     média, cada cestão tem).

# [ORGANIZAÇÃO] A célula que converte e salva Densidade/Rendimento de
# forma permanente foi movida pra Parte A, bloco A.4. A partir daqui
# é só leitura/investigação, não salva nada.

#%%
import pandas as pd

base = pd.read_csv("base.csv")

# =====================================================================
# Univariada de Energia Elétrica, Densidade e Rendimento Metálico
# Essas 3 colunas são propriedades POR MATERIAL (vêm do dicionário),
# não por linha/corrida - então a análise certa é 1 linha por material,
# não um histograma de 43 mil linhas repetidas.
# =====================================================================
propriedades_material = (
    base[["cd_codigo_material", "Energia Elétrica", "Densidade t/m3", "Rendimento Metálico %"]]
    .drop_duplicates(subset="cd_codigo_material")
    .sort_values("Densidade t/m3")
    .reset_index(drop=True)
)

print("=== Propriedades por material (ordenado por densidade) ===")
print(propriedades_material)

print("\n=== describe() das 3 propriedades ===")
print(propriedades_material[["Energia Elétrica", "Densidade t/m3", "Rendimento Metálico %"]].describe())

props_validas = propriedades_material.dropna(subset=["Densidade t/m3", "Energia Elétrica", "Rendimento Metálico %"])
print("\nMenor densidade:", props_validas.loc[props_validas["Densidade t/m3"].idxmin(), "cd_codigo_material"])
print("Maior densidade:", props_validas.loc[props_validas["Densidade t/m3"].idxmax(), "cd_codigo_material"])
print("Maior consumo de energia:", props_validas.loc[props_validas["Energia Elétrica"].idxmax(), "cd_codigo_material"])
print("Menor consumo de energia:", props_validas.loc[props_validas["Energia Elétrica"].idxmin(), "cd_codigo_material"])
print("Maior rendimento metálico:", props_validas.loc[props_validas["Rendimento Metálico %"].idxmax(), "cd_codigo_material"])
print("Menor rendimento metálico:", props_validas.loc[props_validas["Rendimento Metálico %"].idxmin(), "cd_codigo_material"])

# =====================================================================
# num_camadas por carregamento (feature nova + univariada)
# =====================================================================
num_camadas = (
    base.groupby(["cd_corrida", "nu_carregamento"])["nu_camada"]
    .nunique()
    .reset_index(name="num_camadas")
)

print("\n=== num_camadas - describe geral ===")
print(num_camadas["num_camadas"].describe())

print("\n=== num_camadas - describe por carregamento (1, 2 ou 3) ===")
print(num_camadas.groupby("nu_carregamento")["num_camadas"].describe())


# =====================================================================
# ITEM 4 — OUTLIERS
# Regra de negócio (capacidade física: 70t / 78m³) versus critério
# estatístico (IQR), cruzados entre si.
# =====================================================================
# A regra de negócio (70t / 78m³) continua sendo o critério OFICIAL
# de "cestão acima da capacidade" - isso não muda, é limite físico.
# Mas, DENTRO da capacidade, ainda pode ter cestão com peso/volume
# muito fora do padrão (ex: muito mais leve que a média) - isso é
# candidato a outlier ESTATÍSTICO, não é a mesma coisa.
#
# OBS: o item "peso = 0" não é tratado aqui - já foi resolvido na
# seção de limpeza de fato (ITEM 2), com imputação pela mediana do
# material (coluna peso_estimado). Como esses valores já foram
# substituídos, eles não aparecem mais como outlier extremo baixo.

#%%
import pandas as pd
import numpy as np

base = pd.read_csv("base.csv")
for col in ["Densidade t/m3", "Rendimento Metálico %"]:
    if not pd.api.types.is_numeric_dtype(base[col]):
        base[col] = base[col].str.replace(",", ".").astype(float)

base["peso_t"] = base["qt_peso_carregamento"] / 1000
cestoes = base[base["nu_carregamento"].isin([1, 2])].copy()
cestoes["volume_m3"] = cestoes["peso_t"] / cestoes["Densidade t/m3"]

total = cestoes.groupby(["cd_corrida", "nu_carregamento"])[["peso_t", "volume_m3"]].sum().reset_index()

# ---------- Critério de negócio (já existia) ----------
total["acima_capacidade"] = (total["peso_t"] > 70) | (total["volume_m3"] > 78)

# ---------- Critério estatístico (IQR), separado por carregamento ----------
def flag_iqr(grupo, coluna):
    q1, q3 = grupo[coluna].quantile([0.25, 0.75])
    iqr = q3 - q1
    limite_baixo = q1 - 1.5 * iqr
    limite_alto = q3 + 1.5 * iqr
    return (grupo[coluna] < limite_baixo) | (grupo[coluna] > limite_alto)

total["outlier_peso_iqr"] = total.groupby("nu_carregamento", group_keys=False)["peso_t"].apply(
    lambda s: flag_iqr(total.loc[s.index], "peso_t")
)
total["outlier_volume_iqr"] = total.groupby("nu_carregamento", group_keys=False)["volume_m3"].apply(
    lambda s: flag_iqr(total.loc[s.index], "volume_m3")
)
total["outlier_estatistico"] = total["outlier_peso_iqr"] | total["outlier_volume_iqr"]

# ---------- Conferência: os dois critérios concordam? ----------
print("Acima da capacidade (negócio):", total["acima_capacidade"].sum())
print("Outlier estatístico (IQR):", total["outlier_estatistico"].sum())
print("\nCruzamento (negócio x estatístico):")
print(pd.crosstab(total["acima_capacidade"], total["outlier_estatistico"]))

# Casos interessantes: outlier estatístico MAS dentro da capacidade
# (pode ser erro de registro - cestão com peso muito baixo comparado aos outros)
so_estatistico = total[total["outlier_estatistico"] & ~total["acima_capacidade"]]
print("\nOutlier estatístico, mas dentro da capacidade (investigar):", len(so_estatistico))
print(so_estatistico.groupby("nu_carregamento")[["peso_t", "volume_m3"]].describe().T)

#%%
mediana_peso = total.groupby("nu_carregamento")["peso_t"].transform("median")

so_estatistico = total[total["outlier_estatistico"] & ~total["acima_capacidade"]].copy()
so_estatistico["tipo"] = np.where(
    so_estatistico["peso_t"] < mediana_peso.loc[so_estatistico.index],
    "muito abaixo da mediana (leve demais)",
    "muito acima da mediana (mas ainda dentro da capacidade)",
)

print(so_estatistico.groupby(["nu_carregamento", "tipo"]).size())
# [DECISÃO JÁ TOMADA, documentada na conversa com o Claude]: outliers
# são MANTIDOS, não removidos. acima_capacidade e outlier_estatistico
# ficam como flags; a maioria dos outliers estatísticos do carregamento
# 2 (259 de 263) são cestões pesados legítimos, dentro da capacidade,
# não erro de medição - remover jogaria fora justamente os casos mais
# próximos do limite, que são os mais relevantes pra carga alta.


# =====================================================================
# ITEM 5 — FEATURES CRIADAS
# As features agregadas por cestão que alimentam a análise bivariada e
# o dataset de modelagem: num_camadas, num_materiais, peso_leve/misto/
# pesado, prop_leve.
# =====================================================================
# O Guia oficial (seção 8.2) espera comparar várias features entre
# carga_alta = 0 e carga_alta = 1. Por corrida+carregamento:
# [x] num_camadas - quantas camadas tem o cestão
# [x] num_materiais - quantos materiais DIFERENTES entram no cestão
# [ ] peso_leve, peso_misto, peso_pesado - peso somado por classe de
#     material dentro do cestão
# [ ] prop_leve - proporção do peso total que é material Leve (o Guia
#     cita isso especificamente como feature candidata)

# [ORGANIZAÇÃO] A célula que cria e salva volume_m3 no base.csv foi
# movida pra Parte A, bloco A.5. O que ficou aqui é só a prévia do
# features_cestao (não é salvo separadamente - foi consolidado dentro
# do dataset_modelagem.csv, ver ITEM 9 - FECHAMENTO).

#%%
num_camadas = base.groupby(["cd_corrida", "nu_carregamento"])["nu_camada"].nunique().rename("num_camadas")

# num_materiais: quantos códigos de material DIFERENTES entram no cestão
num_materiais = base.groupby(["cd_corrida", "nu_carregamento"])["cd_codigo_material"].nunique().rename("num_materiais")

peso_por_classe = (
    base.groupby(["cd_corrida", "nu_carregamento", "tp_material"])["peso_t"]
    .sum()
    .unstack("tp_material", fill_value=0)
    .rename(columns={"Leve": "peso_leve", "Misto": "peso_misto", "Pesado": "peso_pesado"})
)

peso_total = base.groupby(["cd_corrida", "nu_carregamento"])["peso_t"].sum().rename("peso_t_total")

features_cestao = pd.concat([num_camadas, num_materiais, peso_por_classe, peso_total], axis=1).reset_index()
features_cestao["prop_leve"] = features_cestao["peso_leve"] / features_cestao["peso_t_total"]


print(features_cestao.head())


# =====================================================================
# ITEM 6 — ANÁLISE BIVARIADA
# Cada feature contra is_carga_alta: boxplot + point-biserial, mapa de
# correlação (multicolinearidade), peso x volume, posição da camada x
# tipo de material, volume do cestão 2 com/sem carga alta, carga alta
# por faixa de peso, e comparação carregamento 1 x 2.
# =====================================================================
# [ ] Fazer o mesmo comparativo (com/sem carga alta) pras features
#     novas do ITEM 5: num_camadas, num_materiais, prop_leve, etc.
# [ ] Calcular a correlação point-biserial entre cada feature numérica
#     e is_carga_alta (o Guia pede isso explicitamente, seção 8.3,
#     usando scipy.stats.pointbiserialr) - isso dá um número pra cada
#     feature dizendo o quão forte é a relação com carga alta, e se
#     isso é estatisticamente significativo (p < 0.05).
# [ ] Montar o mapa de correlação completo (heatmap) entre todas as
#     features numéricas + is_carga_alta, pra ver não só a relação com
#     a resposta, mas também quais features são muito correlacionadas
#     ENTRE SI (multicolinearidade - ex: peso total e volume estimado
#     provavelmente são bem correlacionados entre si, e isso precisa
#     ser discutido antes de meter os dois num modelo).
# [ ] Comparar explicitamente carregamento 1 x carregamento 2 (o Guia
#     pergunta: a taxa de carga alta é igual nos dois? Se não, por
#     quê?).

# [ORGANIZAÇÃO] A célula que monta o dataframe "cestao" (features
# agregadas por corrida+carregamento) está na Parte A, bloco A.6, e é
# usada no fechamento (A.7) pra gerar dataset_modelagem.csv.
# [AJUSTE P/ ARQUIVO SEPARADO] Como "cestao" não existe mais em memória
# aqui (Parte A e Parte B agora são arquivos separados), lemos abaixo o
# dataset_modelagem.csv que a Parte A já salva em disco - é a mesma
# informação de "cestao" filtrada pros carregamentos 1 e 2, já com a
# coluna carga_alta pronta (rode o transforma_dataset.py antes deste
# arquivo).

#%%
from scipy.stats import pointbiserialr
import matplotlib.pyplot as plt

bivariada = pd.read_csv("dataset_modelagem.csv")
# [AJUSTE P/ ARQUIVO SEPARADO] dt_inicio volta do CSV como texto, não
# como datetime (era datetime só enquanto "cestao" vivia em memória, na
# Parte A) - precisa reconverter pra usar .dt no ITEM 7 (padrões temporais).
bivariada["dt_inicio"] = pd.to_datetime(bivariada["dt_inicio"], utc=True)

features_numericas = ["peso_t_total", "volume_m3_total", "num_camadas", "num_materiais", "peso_leve", "peso_misto", "peso_pesado", "prop_leve"]

# =====================================================================
# Boxplot + point-biserial, pra cada feature nova
# =====================================================================
resultados = []
for feat in features_numericas:
    r, p = pointbiserialr(bivariada["carga_alta"], bivariada[feat])
    resultados.append({"feature": feat, "r": round(r, 3), "p_valor": round(p, 4), "significativo": p < 0.05})

    plt.figure(figsize=(5, 4))
    plt.boxplot(
        [bivariada[bivariada["carga_alta"] == 0][feat], bivariada[bivariada["carga_alta"] == 1][feat]],
        tick_labels=["sem carga alta", "com carga alta"], showfliers=False,
    )
    plt.title(f"{feat} (r={r:.2f}, p={p:.4f})")
    plt.tight_layout()
    plt.savefig(f"graficos/bivariada_{feat}.png", dpi=130)
    plt.close()

tabela_pb = pd.DataFrame(resultados).sort_values("r", key=abs, ascending=False)
print("=== Point-biserial: feature x carga_alta ===")
print(tabela_pb)

# =====================================================================
# Heatmap de correlação (multicolinearidade)
# =====================================================================
corr = bivariada[features_numericas + ["carga_alta"]].corr()
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticks(range(len(corr.columns))); ax.set_yticklabels(corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
plt.colorbar(im)
plt.title("Correlação entre features (+ carga_alta)")
plt.tight_layout()
plt.savefig("graficos/heatmap_correlacao.png", dpi=130)
plt.close()

print("\n=== Pares de features com |correlação| > 0.6 (multicolinearidade) ===")
pares = corr.abs().unstack().sort_values(ascending=False)
pares = pares[(pares < 1.0) & (pares > 0.6)]
pares = pares[~pares.index.duplicated()]
print(pares)

# ---------- GRÁFICO 11 - Peso contra volume (carregamento 1)
c1_pv = total[total["nu_carregamento"] == 1]
correlacao = c1_pv["peso_t"].corr(c1_pv["volume_m3"])
passou = (c1_pv["peso_t"] > 70) | (c1_pv["volume_m3"] > 78)

fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.scatter(c1_pv[~passou]["peso_t"], c1_pv[~passou]["volume_m3"], s=10, alpha=0.4, color=AZUL, label="dentro do limite")
ax.scatter(c1_pv[passou]["peso_t"], c1_pv[passou]["volume_m3"], s=10, alpha=0.6, color=VERMELHO, label="excede peso ou volume")
ax.axvline(70, color="black", linestyle="--", linewidth=1)
ax.axhline(78, color="black", linestyle="--", linewidth=1)
ax.set_xlabel("peso do cestão (t)")
ax.set_ylabel("volume estimado (m³)")
ax.set_title(
    f"Peso X Volume (correlação {correlacao:.2f})".replace(".", ","),
    loc="left", fontweight="bold", fontsize=11,
)
ax.legend()
mostrar("11_peso_contra_volume")

# ---------- Posição relativa de cada camada (usado nos gráficos 14 e 15)
cestoes = cestoes.sort_values(["cd_corrida", "nu_carregamento", "nu_camada"])
ultima_camada = cestoes.groupby(["cd_corrida", "nu_carregamento"])["nu_camada"].transform("max")
cestoes["posicao"] = cestoes["nu_camada"] / ultima_camada
cestoes["faixa"] = pd.cut(cestoes["posicao"], [0, 0.3, 0.7, 1.0], labels=["fundo", "meio", "topo"])

# ---------- GRÁFICO 14 - Onde cada classe de material aparece no cestão
tabela = pd.crosstab(cestoes["tp_material"], cestoes["faixa"], normalize="index") * 100
tabela = tabela.reindex(["Pesado", "Misto", "Leve"])        # Leve fica embaixo no gráfico
leve_meio = tabela.loc["Leve", "meio"]

fig, ax = plt.subplots(figsize=(9, 4.2))
esquerda = np.zeros(len(tabela))
for faixa, cor in [("fundo", AZUL), ("meio", CINZA), ("topo", LARANJA)]:
    ax.barh(tabela.index, tabela[faixa], left=esquerda, color=cor, label=faixa)
    for i, valor in enumerate(tabela[faixa]):
        ax.text(
            esquerda[i] + valor / 2, i, f"{valor:.0f}%".replace(".", ","),
            ha="center", va="center", color="white", fontweight="bold",
        )
    esquerda = esquerda + tabela[faixa].values
ax.set_title(
    "% dos materiais",
    loc="left", fontweight="bold", fontsize=11,
)
ax.set_xlabel("% das linhas de cada classe")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
ax.grid(False)
mostrar("14_onde_cada_classe_aparece")

# ---------- GRÁFICO 15 - Classe da camada de fundo e de topo
classes_cestao = cestoes.groupby(["cd_corrida", "nu_carregamento"])["tp_material"]
fundo = classes_cestao.first().value_counts(normalize=True) * 100
topo = classes_cestao.last().value_counts(normalize=True) * 100
resumo = pd.DataFrame({"Fundo": fundo, "Topo": topo}).fillna(0).T
resumo = resumo.reindex(columns=["Leve", "Misto", "Pesado"]).fillna(0)

fig, ax = plt.subplots(figsize=(7.5, 4.5))
resumo.plot(kind="bar", ax=ax, color=[AZUL, CINZA, VERMELHO], rot=0, width=0.7)
for barra in ax.patches:
    if barra.get_height() > 0:
        ax.text(
            barra.get_x() + barra.get_width() / 2, barra.get_height() + 1.5,
            f"{barra.get_height():.0f}%", ha="center",
        )
ax.set_title(
    "Análise do material leve",
    loc="left", fontweight="bold", fontsize=11,
)
ax.set_ylabel("% dos cestões")
ax.set_ylim(0, 108)
ax.legend(title="Classe da camada", frameon=False)
mostrar("15_fundo_e_topo")

# ---------- GRÁFICO 17 - Volume do cestão 2: com e sem carga alta
c2_box = juntos[juntos["nu_carregamento"] == 2]
sem = c2_box[c2_box["carga_alta"] == 0]["volume_m3"]
com = c2_box[c2_box["carga_alta"] == 1]["volume_m3"]

fig, ax = plt.subplots(figsize=(6.5, 4.8))
caixas = ax.boxplot(
    [sem, com], tick_labels=["sem carga alta", "com carga alta"],
    patch_artist=True, showfliers=False,
)
for caixa, cor in zip(caixas["boxes"], [AZUL, VERMELHO]):
    caixa.set_facecolor(cor)
    caixa.set_alpha(0.6)
ax.axhline(78, color="black", linestyle="--", linewidth=1)
ax.text(1.5, 78.8, "78 m³", ha="center")
ax.set_title(
    "Cestão 2: volume médio",
    loc="left", fontweight="bold", fontsize=11,
)
ax.set_ylabel("volume estimado (m³)")
mostrar("17_volume_cestao2_com_e_sem_carga_alta")

# ---------- GRÁFICO 18 - Carga alta por faixa de peso (carregamento 1) - PRÉVIA
c1j = juntos[juntos["nu_carregamento"] == 1].copy()
c1j["faixa_peso"] = pd.cut(c1j["peso_t"], [0, 50, 55, 60, 65, 70, np.inf], labels=["até 50 t", "50 a 55 t", "55 a 60 t", "60 a 65 t", "65 a 70 t", "acima de 70 t"])
por_faixa = c1j.groupby("faixa_peso", observed=True)["carga_alta"].agg(["size", "mean"])
media_geral = c1j["carga_alta"].mean() * 100

fig, ax = plt.subplots(figsize=(9, 4.8))
barras = ax.bar(por_faixa.index.astype(str), por_faixa["mean"] * 100, color=VERMELHO, alpha=0.85)
rotulos = []
for n, m in zip(por_faixa["size"], por_faixa["mean"]):
    aviso = "\npoucos casos" if n < 100 else ""
    rotulos.append(f"{pct(m * 100)}\n(n={br(n)}){aviso}")
ax.bar_label(barras, labels=rotulos, padding=3, fontsize=9)
ax.axhline(media_geral, color="black", linestyle="--", linewidth=1)
ax.text(len(por_faixa) - 0.5, media_geral + 1, f"média: {pct(media_geral)}", ha="right")
ax.set_ylim(0, por_faixa["mean"].max() * 100 * 1.25)
ax.set_title(
    "Carregamento 1: % de carga alta pelo peso do cestão",
    loc="left", fontweight="bold", fontsize=11,
)
ax.set_xlabel("peso do cestão")
ax.set_ylabel("% com carga alta")
mostrar("18_carga_alta_por_faixa_de_peso")

#%%
# =====================================================================
# Carregamento 1 x 2
# =====================================================================
print("\n=== Taxa de carga alta por carregamento ===")
print(bivariada.groupby("nu_carregamento")["carga_alta"].agg(["mean", "count"]))


# =====================================================================
# ITEM 7 — PADRÕES TEMPORAIS
# Taxa de carga alta por turno e por dia da semana.
# =====================================================================
# [NOTA] O turno usado abaixo (madrugada/manhã/tarde/noite, pd.cut) é
# DIFERENTE do turno oficial do Guia da EVCOMX (Turno A 06h-14h, B
# 14h-22h, C 22h-06h) - decidir se troca pra bater com o guia ou se
# mantém e justifica a escolha na apresentação.

#%%
dt_local = bivariada["dt_inicio"].dt.tz_convert("America/Sao_Paulo")
bivariada["turno"] = pd.cut(dt_local.dt.hour, [-1, 6, 12, 18, 24], labels=["madrugada", "manha", "tarde", "noite"])
bivariada["dia_semana"] = dt_local.dt.day_name()

print("\n=== Taxa de carga alta por turno ===")
print(bivariada.groupby("turno")["carga_alta"].agg(["mean", "count"]))
print("\n=== Taxa de carga alta por dia da semana ===")
print(bivariada.groupby("dia_semana")["carga_alta"].agg(["mean", "count"]))


# =====================================================================
# ITEM 8 — DICIONÁRIO DE MATERIAIS
# Scatter Densidade x Energia Elétrica (colorido por Rendimento
# Metálico) e verificação numérica da regra leve/pesado do guia.
# =====================================================================
# [ ] Scatter plot de Densidade (eixo X) x Energia Elétrica (eixo Y),
#     colorido por Rendimento Metálico - o Guia (seção 3.2) pede esse
#     gráfico especificamente pra "revelar a personalidade de cada
#     tipo de sucata".
# [ ] Confirmar com números a relação: materiais leves têm densidade
#     baixa (< 0,55 t/m³) e rendimento alto (> 0,90); materiais pesados
#     têm densidade alta (> 2,50 t/m³). O Guia já antecipa esse
#     resultado (seção 3.3) - vale conferir se os dados batem com isso
#     ou se tem exceção.
# [ ] Separar explicitamente insumos (CAL, COQUE) dos materiais
#     metálicos de verdade antes de qualquer conta de peso/volume (ver
#     ITEM 2).

#%%
import pandas as pd
import matplotlib.pyplot as plt

base = pd.read_csv("base.csv")
for col in ["Densidade t/m3", "Rendimento Metálico %"]:
    if not pd.api.types.is_numeric_dtype(base[col]):
        base[col] = base[col].str.replace(",", ".").astype(float)

# ---------- Propriedades únicas por material (exclui RECG/RECC, sem dados) ----------
materiais = (
    base[["cd_codigo_material", "tp_material", "Densidade t/m3", "Energia Elétrica", "Rendimento Metálico %"]]
    .drop_duplicates(subset="cd_codigo_material")
    .dropna(subset=["Densidade t/m3", "Energia Elétrica", "Rendimento Metálico %"])
)

# ---------- Scatter: Densidade x Energia, cor = Rendimento (Guia, seção 3.2) ----------
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(materiais["Densidade t/m3"], materiais["Energia Elétrica"],
                 c=materiais["Rendimento Metálico %"], cmap="viridis", s=120, edgecolor="black")
for _, row in materiais.iterrows():
    ax.annotate(row["cd_codigo_material"], (row["Densidade t/m3"], row["Energia Elétrica"]),
                textcoords="offset points", xytext=(5, 5), fontsize=8)
plt.colorbar(sc, label="Rendimento Metálico %")
ax.set_xlabel("Densidade (t/m³)")
ax.set_ylabel("Energia Elétrica")
ax.set_title("Personalidade de cada tipo de sucata")
plt.tight_layout()
plt.savefig("graficos/scatter_materiais.png", dpi=130)
plt.close()

# ---------- Checando a regra do Guia (seção 3.3) ----------
leve = materiais[materiais["tp_material"] == "Leve"]
pesado = materiais[materiais["tp_material"] == "Pesado"]

print("Leve: exceções com densidade >= 0.55 ->", (leve["Densidade t/m3"] >= 0.55).sum())
print(leve[leve["Densidade t/m3"] >= 0.55][["cd_codigo_material", "Densidade t/m3"]])
print("\nLeve: exceções com rendimento <= 0.90 ->", (leve["Rendimento Metálico %"] <= 0.90).sum())
print(leve[leve["Rendimento Metálico %"] <= 0.90][["cd_codigo_material", "Rendimento Metálico %"]])

print("\nPesado: exceções com densidade <= 2.50 ->", (pesado["Densidade t/m3"] <= 2.50).sum())
print(pesado[pesado["Densidade t/m3"] <= 2.50][["cd_codigo_material", "Densidade t/m3"]])

# ---------- Insumos CAL/COQUE: já resolvido, só documentando ----------
print("\nOcorrências de CAL/COQUE no base.csv:",
      base["cd_codigo_material"].isin(["7071955", "8800022", "7071940"]).sum())
# Esperado: 0 - esses códigos existem só no dicionário de materiais, nunca
# foram usados em nenhuma corrida. O base.csv (join a partir de infos_camadas)
# já exclui eles automaticamente, não precisa de ação adicional.


# =====================================================================
# ITEM 9 — FECHAMENTO
# =====================================================================
# [ORGANIZAÇÃO] A célula que monta resposta_long, faz o merge final e
# salva dataset_modelagem.csv já está na Parte A, bloco A.7.
#
# [ ] Escrever, em texto corrido (não só código/gráfico), as
#     conclusões principais: qual(is) feature(s) mais se relaciona(m)
#     com carga alta, o que os dados confirmam ou contradizem do que
#     o Guia da EVCOMX já descreve (Efeito Mola, volume > peso como
#     preditor, etc.), e quais limitações/vieses dos dados valem a
#     pena mencionar pra EVCOMX.