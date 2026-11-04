"""
Simulador de carga alta · EVCOMX
Monte uma corrida (cestão 1, cestão 2 e a gusa líquida), camada por camada, e o
modelo escolhido do projeto (média de 5 Gradient Boosting,
selecao_algoritmos/modelo_final.joblib) diz se ela deve dar carga alta.

Rodar (de dentro desta pasta):
    python -m streamlit run app.py
"""
import importlib
import math

import streamlit as st
import streamlit.components.v1 as components

import desenho
import motor

# O Streamlit roda o app.py de novo a cada clique, mas guarda na memória os módulos
# importados. Recarregar garante que mudanças em motor.py e desenho.py apareçam
# sem precisar reiniciar o servidor.
importlib.reload(motor)
importlib.reload(desenho)

from desenho import BRANCO, CINZA, CINZA_CLARO, CINZA_LINHA, FONTE, FONT_FACE, LARANJA, LARANJA_ESCURO, PRETO

st.set_page_config(page_title="Simulador de Carga Alta | EVCOMX", page_icon="🔥", layout="wide",
                   initial_sidebar_state="collapsed")


# ====================================================================== carga (uma vez só)
@st.cache_resource
def carregar():
    return motor.carregar_modelo(), motor.carregar_materiais(), motor.carregar_config(), motor.carregar_momentos()


PACOTE, MATS, CFG, MOMENTOS = carregar()
GUSA = CFG["gusa"]
GUSA_PADRAO_T = 35.0      # número redondo perto da mediana real (34,3 t)
MODOS = list(CFG["limiares"]["principal"])
# Histórico, ritmo e horário do forno ficam fixos num momento real típico (ver README).
MOMENTO_ID = CFG["presets_momento"]["Forno típico"]


def _camadas(receita):
    return [[motor.Camada(c, p) for c, p in cam] for cam in receita]


@st.cache_data(show_spinner=False, max_entries=200)
def prever(receita, gusa_t, modo):
    """receita = tupla de tuplas (código, peso) por cestão; tudo hashable para o cache."""
    return motor.prever_corrida(_camadas(receita), MOMENTOS.loc[MOMENTO_ID], gusa_t, None, modo,
                                PACOTE, MATS, CFG)


@st.cache_data(show_spinner="Procurando uma troca que baixe o risco...", max_entries=100)
def sugerir(receita, k, gusa_t, modo):
    return motor.sugerir_troca(_camadas(receita), k, MOMENTOS.loc[MOMENTO_ID], gusa_t, None, modo,
                               PACOTE, MATS, CFG)


# ====================================================================== estado
def _iniciar_estado():
    padrao = {"corrida": None, "ativo": 1, "anim": None, "erro": None, "n_corridas": 0, "modo": MODOS[0]}
    for k, v in padrao.items():
        if k not in st.session_state:
            st.session_state[k] = v


_iniciar_estado()
ss = st.session_state


def _mudou_receita():
    """Qualquer mudança na receita invalida o resultado anterior."""
    if ss.corrida:
        ss.corrida["enviada"] = False
    ss.erro = None


# ====================================================================== ações (callbacks)
def nova_corrida():
    ss.n_corridas += 1
    ss.corrida = {"codigo": f"SIM-{ss.n_corridas:03d}", "cestoes": [], "gusa_t": GUSA_PADRAO_T,
                  "enviada": False}
    ss.ativo = 1
    ss.anim = {"tipo": "gusa"}
    ss.erro = None


def novo_cestao():
    if len(ss.corrida["cestoes"]) >= motor.MAX_CESTOES:
        return
    ss.corrida["cestoes"].append([])
    ss.ativo = len(ss.corrida["cestoes"])
    ss.anim = {"tipo": "cestao", "cestao": ss.ativo}
    _mudou_receita()


def remover_cestao():
    if ss.corrida["cestoes"]:
        ss.corrida["cestoes"].pop()
        ss.ativo = max(1, min(ss.ativo, len(ss.corrida["cestoes"])))
        ss.anim = None
        _mudou_receita()


def inserir_camada():
    cod = ss.material
    peso = ss.get(f"peso_{cod}")
    camadas = ss.corrida["cestoes"][ss.ativo - 1]
    nova = motor.Camada(cod, float(peso) if peso is not None else float("nan"))
    erro = motor.validar_insercao([motor.Camada(**c) for c in camadas], nova, MATS)
    if erro:
        ss.erro = erro
        ss.anim = None          # o desenho não muda: nada de repetir a animação atrás do aviso
        return
    camadas.append({"codigo": cod, "peso_t": round(float(peso), 2)})
    ss.anim = {"tipo": "camada", "cestao": ss.ativo, "camada": len(camadas) - 1}
    _mudou_receita()


def desfazer_camada():
    camadas = ss.corrida["cestoes"][ss.ativo - 1]
    if camadas:
        camadas.pop()
        ss.anim = None
        _mudou_receita()


def esvaziar_cestao():
    ss.corrida["cestoes"][ss.ativo - 1] = []
    ss.anim = None
    _mudou_receita()


def trocar_ativo(chave):
    ss.ativo = ss[chave]
    ss.anim = None


def mudar_gusa(chave_on, chave_t):
    ligada = ss.get(chave_on, False)
    ss.corrida["gusa_t"] = float(ss.get(chave_t) or GUSA_PADRAO_T) if ligada else None
    ss.anim = {"tipo": "gusa"} if ligada else None


def mudar_gusa_t(chave_t):
    if ss.corrida["gusa_t"] is not None:
        ss.corrida["gusa_t"] = float(ss[chave_t])
    ss.anim = None


def enviar_corrida():
    ss.corrida["enviada"] = True
    ss.anim = {"tipo": "resultado"}


def mudou_controle():
    """O modo do alarme mudou: o desenho muda sem animar de novo."""
    ss.anim = None


def aplicar_sugestao(cestao, j, para):
    ss.corrida["cestoes"][cestao - 1][j]["codigo"] = para
    ss.ativo = cestao
    ss.anim = {"tipo": "camada", "cestao": cestao, "camada": j}
    ss.corrida["enviada"] = True        # mostra o resultado novo direto


def descartar_corrida():
    ss.corrida = None
    ss.anim = None
    ss.erro = None


# ====================================================================== estilo (identidade EVCOMX)
CSS = f"""
<style>
{FONT_FACE}
/* fonte da EVCOMX em tudo (o tema já aplica; isto cobre o que sobrar), menos os ícones do Streamlit */
html, body, .stApp, .stApp *:not([data-testid="stIconMaterial"]):not(code):not(pre),
div[data-baseweb="popover"] * {{ font-family: {FONTE}; }}
/* sem o botão de link (âncora) ao lado dos títulos */
[data-testid="stHeaderActionElements"], .stApp h1 a[href^="#"], .stApp h2 a[href^="#"],
.stApp h3 a[href^="#"], .stApp h4 a[href^="#"] {{ display: none !important; }}
/* espaço para a barra do Streamlit não cortar o cabeçalho */
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 3.6rem; max-width: 1400px; }}

h1, h2, h3, h4 {{ color: {PRETO}; font-weight: 700 !important; letter-spacing: .3px; }}
.stApp a {{ color: {LARANJA_ESCURO}; }}

.cabecalho {{ background: {PRETO}; border-radius: 18px; padding: 22px 28px 20px; color: {BRANCO};
             border-bottom: 6px solid {LARANJA}; margin: 0 0 18px; overflow: visible; }}
.cabecalho .marca {{ letter-spacing: 4px; font-weight: 700; font-size: 13px; color: {LARANJA}; font-family: {FONTE}; }}
.cabecalho h1 {{ color: {BRANCO} !important; margin: 6px 0 0; font-size: 32px; line-height: 1.2; padding: 0; font-family: {FONTE}; }}
.cabecalho p {{ margin: 8px 0 0; color: #CFCFCF; font-size: 15px; font-family: {FONTE}; }}
.cabecalho b {{ color: {LARANJA}; }}

/* botões */
.stButton > button, button[data-testid^="stBaseButton"] {{
    border-radius: 999px; font-weight: 700; letter-spacing: .3px; padding: .5rem 1.15rem;
    border: 2px solid {PRETO}; background: {BRANCO}; color: {PRETO}; transition: all .15s ease; }}
.stButton > button:hover, button[data-testid^="stBaseButton"]:hover {{
    border-color: {LARANJA}; color: {LARANJA}; background: {BRANCO}; }}
.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] {{
    background: {LARANJA}; border-color: {LARANJA}; color: {BRANCO}; }}
.stButton > button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {{
    background: {LARANJA_ESCURO}; border-color: {LARANJA_ESCURO}; color: {BRANCO}; }}
.stButton > button:disabled, button[data-testid^="stBaseButton"]:disabled {{
    background: {CINZA_CLARO}; border-color: {CINZA_LINHA}; color: #9A9A9A; }}
.stButton > button p {{ font-weight: 700; }}

/* barra lateral */
section[data-testid="stSidebar"] {{ background: {PRETO}; }}
section[data-testid="stSidebar"] h3 {{ color: {LARANJA} !important; }}
section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] li, section[data-testid="stSidebar"] span {{ color: #EDEDED; }}
section[data-testid="stSidebar"] div[data-testid="stCaptionContainer"] p {{ color: #B5B5B5; }}
section[data-testid="stSidebar"] code {{ color: {LARANJA}; background: #222; }}
section[data-testid="stSidebar"] details {{ border-color: #333; }}

.mat-info {{ display: flex; gap: 10px; align-items: center; font-size: 13px; color: #333; margin: -4px 0 8px;
             font-family: {FONTE}; }}
.mat-info b {{ color: {PRETO}; }}
div[data-testid="stMetricValue"] {{ font-size: 1.2rem; color: {PRETO}; font-weight: 700; }}
div[role="dialog"] {{ background: rgba(198, 40, 40, .82) !important; backdrop-filter: blur(4px); border: 4px solid #8E0000 !important; border-top-width: 10px !important; }}
div[role="dialog"] h1, div[role="dialog"] h2, div[role="dialog"] h3 {{ color: {BRANCO} !important; font-weight: 800 !important; }}
div[role="dialog"] svg {{ color: {BRANCO}; }}
.aviso-grave {{ font-family: {FONTE}; font-size: 21px; font-weight: 800; line-height: 1.4; color: {BRANCO}; margin: 4px 0 8px; }}
.aviso-grave small {{ display: block; margin-top: 10px; font-size: 15px; font-weight: 700; color: #FFE3E3; }}
div[role="dialog"] .stButton > button, div[role="dialog"] button[data-testid^="stBaseButton"] {{
    background: {BRANCO} !important; border-color: {BRANCO} !important; color: #B71C1C !important; font-weight: 800; }}
div[role="dialog"] .stButton > button:hover, div[role="dialog"] button[data-testid^="stBaseButton"]:hover {{
    background: #FFE3E3 !important; border-color: #FFE3E3 !important; color: #8E0000 !important; }}
.rodape {{ color: {CINZA}; font-size: 12px; margin-top: 18px; border-top: 1px solid {CINZA_LINHA}; padding-top: 10px;
           font-family: {FONTE}; }}
</style>
<div class="cabecalho">
  <div class="marca">EVCOMX · FUNDO AMANHÃ · BOOTCAMP DATA SCIENCE 2026</div>
  <h1>Simulador de carga alta no forno elétrico a arco</h1>
  <p>Monte a receita da corrida, camada por camada, e veja se o modelo do projeto prevê <b>carga alta</b>.</p>
</div>
"""
# Sem linhas em branco: no Markdown do Streamlit, uma linha em branco encerra o bloco HTML
# e o resto do CSS apareceria como texto na página.
st.markdown("\n".join(linha for linha in CSS.splitlines() if linha.strip()), unsafe_allow_html=True)


# ====================================================================== aviso central
def abrir_aviso(mensagem):
    """Aviso no meio da tela: a página fica bloqueada até o usuário clicar em OK."""
    titulo = "⚠️ Não cabe no cestão" if mensagem.startswith("Não cabe") else "⚠️ Camada não inserida"

    def corpo():
        st.markdown(f'<div class="aviso-grave">{mensagem}<small>A camada não foi inserida.</small></div>',
                    unsafe_allow_html=True)
        if st.button("OK", type="primary", key="ok_aviso", width="stretch"):
            ss.erro = None
            st.rerun()

    try:
        janela = st.dialog(titulo, dismissible=False)      # sem X e sem fechar clicando fora
    except TypeError:
        janela = st.dialog(titulo)                         # versões antigas do Streamlit
    janela(corpo)()


# ====================================================================== barra lateral
with st.sidebar:
    st.subheader("Quando alarmar?", anchor=False)
    explica = {MODOS[0]: "o critério da EVCOMX: alarme falso custa 1 min, carga alta custa a parada",
               MODOS[1]: "mais conservador: o reajuste da montagem também custa 1 min quando acerta"}
    st.radio("Limiar", MODOS, key="modo", on_change=mudou_controle, label_visibility="collapsed",
             captions=[f"{explica[m]} (limiar {CFG['limiares']['principal'][m]:.0%})" for m in MODOS])
    modo = ss.modo

    with st.expander("Sobre o modelo"):
        mom = motor.resumo_momento(MOMENTOS.loc[MOMENTO_ID])
        st.markdown(
            f"""
**Média de 5 Gradient Boosting**, lida de `{CFG['modelo']['arquivo']}`: o modelo escolhido
na etapa de otimização por minutos, com {CFG['modelo']['n_features']} features.

No teste (24/jun a 4/ago/2026): PR-AUC **0,57** e **131 min** de forno perdidos, contra 251 min sem modelo.

O limiar não é 50%: é a chance a partir da qual alarmar dá menos minutos perdidos.
A **gusa líquida** entra no modelo.

O cestão 2 usa a versão do modelo **sem o resultado do cestão 1** (a reserva do pacote).

O histórico do forno fica fixo num momento real típico ({mom['inicio'][8:10]}/{mom['inicio'][5:7]}/{mom['inicio'][:4]},
{mom['ult10']*10:.0f} carga alta nas 10 corridas anteriores). O número de baias e o tempo de montagem
usam o valor típico.
""")


# ====================================================================== tela inicial
corrida = ss.corrida
if corrida is None:
    col_a, _ = st.columns([1.3, 1])
    with col_a:
        st.subheader("Como funciona", anchor=False)
        st.markdown(
            """
1. **Nova corrida** abre uma corrida vazia, já com a panela de gusa líquida.
2. **Novo cestão** pendura um cestão (até dois de sucata).
3. **Inserir camada**: escolha o material e o peso; cada camada cai no cestão, com altura proporcional ao volume.
4. **Enviar corrida**: o modelo calcula a chance de cada cestão e diz se a corrida deve dar carga alta.
""")
        st.button("➕  Nova corrida", type="primary", on_click=nova_corrida)
    st.stop()


# ====================================================================== corrida aberta
cestoes = corrida["cestoes"]
n_cestoes = len(cestoes)
ss.ativo = max(1, min(ss.ativo, n_cestoes))

# ---------- barra de ações ----------
b1, b2, b3, b4 = st.columns([1, 1, 1.5, 1])
with b1:
    st.button("➕  Novo cestão", type="primary", on_click=novo_cestao, disabled=n_cestoes >= motor.MAX_CESTOES,
              help="Uma corrida tem até 2 cestões de sucata.")
with b2:
    st.button(f"➖  Tirar cestão {max(n_cestoes, 1)}", on_click=remover_cestao, disabled=n_cestoes == 0)
with b3:
    chave_on, chave_t = f"gusa_on_{corrida['codigo']}", f"gusa_t_{corrida['codigo']}"
    g1, g2 = st.columns([1.2, 1])
    with g1:
        st.toggle("Gusa líquida", value=corrida["gusa_t"] is not None, key=chave_on,
                  on_change=mudar_gusa, args=(chave_on, chave_t),
                  help="3º carregamento da corrida. Entra no modelo (gusl_t e tem_gusl).")
    with g2:
        st.number_input("t de gusa", min_value=15.0, max_value=60.0, step=1.0, format="%.0f",
                        value=float(corrida["gusa_t"] or GUSA_PADRAO_T), key=chave_t,
                        on_change=mudar_gusa_t, args=(chave_t,), label_visibility="collapsed",
                        disabled=corrida["gusa_t"] is None)
with b4:
    st.button("🗑️  Descartar corrida", on_click=descartar_corrida)

# ---------- previsão (se a corrida já foi enviada) ----------
pronta = n_cestoes > 0 and all(len(c) > 0 for c in cestoes)
receita = tuple(tuple((c["codigo"], c["peso_t"]) for c in cam) for cam in cestoes)
previsao = prever(receita, corrida["gusa_t"], modo) if corrida["enviada"] and pronta else None

# ---------- desenho (esquerda) + inserir camada e painel (direita) ----------
camadas_obj = [[motor.Camada(**c) for c in cam] for cam in cestoes]
col_cena, col_lado = st.columns([1.55, 1], gap="large")

with col_cena:
    html_cena = desenho.desenhar_corrida(cestoes, MATS, ss.ativo, corrida["gusa_t"], ss.anim, previsao,
                                         corrida["codigo"])
    n_mat = len({c["codigo"] for cam in cestoes for c in cam})
    components.html(html_cena, height=560 + 30 * math.ceil(n_mat / 3), scrolling=False)

with col_lado:
    st.subheader("Inserir camada", anchor=False)
    if n_cestoes == 0:
        st.info("Clique em **Novo cestão** para pendurar o primeiro cestão.")
    else:
        if n_cestoes > 1:
            chave_radio = f"radio_ativo_{n_cestoes}_{ss.ativo}"
            st.radio("Cestão que recebe a camada", list(range(1, n_cestoes + 1)), index=ss.ativo - 1,
                     format_func=lambda i: f"Cestão {i}", horizontal=True, key=chave_radio,
                     on_change=trocar_ativo, args=(chave_radio,))
        else:
            st.caption("Camadas vão para o **cestão 1**.")
        st.selectbox("Material", list(MATS.index), key="material",
                     format_func=lambda c: f"{c} – {MATS.loc[c, 'descricao'].title()} ({MATS.loc[c, 'classe'].lower()})")
        cod = ss.material
        m = MATS.loc[cod]
        st.markdown(
            f'<div class="mat-info">{desenho.amostra_material(cod, 30)}<div><b>{cod}</b> · {m["classe"]} · '
            f'densidade {desenho._fmt(m["densidade"], 2)} t/m³<br>camada típica: {desenho._fmt(m["peso_p10_t"])} a '
            f'{desenho._fmt(m["peso_p90_t"])} t · aparece em {m["pct_cestoes"]:.0%} dos cestões</div></div>',
            unsafe_allow_html=True)
        st.number_input("Peso da camada (t)", min_value=motor.PESO_MIN_CAMADA_T, max_value=motor.PESO_MAX_CAMADA_T,
                        value=float(round(m["peso_tipico_t"], 1)), step=0.1, format="%.1f", key=f"peso_{cod}",
                        help="Abaixo de 2 t a EVCOMX considera que o gancho não pegou o material.")
        st.button(f"⬇️  Inserir no cestão {ss.ativo}", type="primary", on_click=inserir_camada,
                  width="stretch")
        a2, a3 = st.columns(2)
        with a2:
            st.button("↩️  Desfazer", on_click=desfazer_camada, disabled=not cestoes[ss.ativo - 1],
                      help="Tira a última camada do cestão selecionado.", width="stretch")
        with a3:
            st.button("🧹  Esvaziar", on_click=esvaziar_cestao, disabled=not cestoes[ss.ativo - 1],
                      width="stretch")

    st.subheader("Como estão os cestões", anchor=False)
    if n_cestoes == 0:
        st.caption("Nenhum cestão ainda.")
    ref = CFG["referencia"]
    for i, cam in enumerate(camadas_obj, start=1):
        r = motor.resumo_cestao(cam, MATS)
        with st.container(border=True):
            st.markdown(f"**Cestão {i}** · {r['n_camadas']} camadas · {r['n_ciclos']} ciclos")
            if not cam:
                st.caption("Vazio.")
                continue
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Peso", f"{desenho._fmt(r['peso_t_total'])} t",
                      help=f"Típico: {desenho._fmt(ref['peso_t_total']['p10'])} a {desenho._fmt(ref['peso_t_total']['p90'])} t")
            k2.metric("Volume", f"{r['ocupacao']:.0%}",
                      help=f"{desenho._fmt(r['volume_m3_total'])} m³ de {motor.CAPACIDADE_M3:.0f} m³ (volume estimado pela densidade)")
            k3.metric("Densid.", desenho._fmt(r["densidade_media"], 2),
                      help=f"Densidade média em t/m³. Metade dos cestões fica entre "
                           f"{desenho._fmt(ref['densidade_media']['p25'], 2)} e {desenho._fmt(ref['densidade_media']['p75'], 2)}")
            k4.metric("Pesado", f"{r['prop_peso_pesado']:.0%}",
                      help=f"do peso. Metade dos cestões fica entre {ref['prop_peso_pesado']['p25']:.0%} e "
                           f"{ref['prop_peso_pesado']['p75']:.0%}")

    # ---------- enviar ----------
    motivo = None
    if n_cestoes == 0:
        motivo = "Crie pelo menos um cestão."
    elif not pronta:
        motivo = "Todo cestão precisa de pelo menos uma camada."
    st.button("🔥  Enviar corrida", type="primary", on_click=enviar_corrida, disabled=motivo is not None,
              width="stretch")
    if motivo:
        st.caption(motivo)
    elif not corrida["enviada"]:
        st.caption("Receita pronta. Envie para o modelo avaliar.")

# ---------- segurar + / − nos campos numéricos acelera ----------
# O campo numérico do Streamlit só anda um passo por clique. Este script (num iframe invisível,
# da mesma origem) escuta a página: segurando o botão, repete o clique, cada vez mais rápido.
components.html("""<script>
(function () {
  const doc = window.parent.document;
  if (window.parent.__segurarRepete) return;          // instala uma vez só
  window.parent.__segurarRepete = true;
  let timer = null;
  const parar = () => { clearTimeout(timer); timer = null; };
  const acharBotao = (idx, sobe) => {
    const caixa = doc.querySelectorAll('[data-testid="stNumberInput"]')[idx];
    if (!caixa) return null;
    return caixa.querySelector(sobe ? '[data-testid="stNumberInputStepUp"]' : '[data-testid="stNumberInputStepDown"]');
  };
  doc.addEventListener('mousedown', (ev) => {
    const btn = ev.target.closest('[data-testid="stNumberInputStepUp"], [data-testid="stNumberInputStepDown"]');
    if (!btn || ev.button !== 0) return;
    const caixa = btn.closest('[data-testid="stNumberInput"]');
    const idx = Array.from(doc.querySelectorAll('[data-testid="stNumberInput"]')).indexOf(caixa);
    const sobe = btn.getAttribute('data-testid') === 'stNumberInputStepUp';
    let espera = 120, n = 0;
    const repetir = () => {
      const b = acharBotao(idx, sobe);                 // o Streamlit pode recriar o botão a cada clique
      if (!b || b.disabled) { parar(); return; }
      b.click(); n += 1;
      espera = Math.max(35, 120 - n * 6);              // acelera aos poucos
      timer = setTimeout(repetir, espera);
    };
    parar();
    timer = setTimeout(repetir, 400);                  // começa a repetir depois de 0,4 s segurando
  }, true);
  ['mouseup', 'mouseleave', 'blur'].forEach(e => doc.addEventListener(e, parar, true));
  window.parent.addEventListener('blur', parar);
})();
</script>""", height=0)

# ---------- resultado ----------
if previsao is not None:
    st.header("Resultado", anchor=False)
    html_res, altura = desenho.painel_resultado_html(previsao, modo, CFG["referencia"], CFG["parada_media_min"])
    components.html(html_res, height=altura, scrolling=False)

    st.subheader("E agora?", anchor=False)
    alvo = [r for r in previsao["cestoes"] if r["alarme"]]
    if not alvo:
        st.markdown("- Nenhum cestão alarmou neste modo: pelo modelo, a receita pode seguir como está.")
    for r in alvo:
        sug = sugerir(receita, r["cestao"], corrida["gusa_t"], modo)
        if not sug or sug["chance_antes"] - sug["chance_depois"] < 0.02:
            st.markdown(f"- **Cestão {r['cestao']}**: nenhuma troca simples de uma camada por pesado baixa a chance "
                        f"de forma visível.")
            continue
        passa = "fica abaixo do limiar" if sug["chance_depois"] <= sug["limiar"] else "ainda fica acima do limiar"
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(
                f"- **Cestão {r['cestao']}**: trocar a camada {sug['camada'] + 1} "
                f"(**{sug['de']}**, {desenho._fmt(sug['peso_t'])} t) por **{sug['para']}** "
                f"({MATS.loc[sug['para'], 'descricao'].title()}) com o mesmo peso leva a chance de "
                f"**{sug['chance_antes']:.0%} para {sug['chance_depois']:.0%}** ({passa}).")
        with c2:
            st.button("Testar essa troca", key=f"sug_{r['cestao']}", on_click=aplicar_sugestao,
                      args=(r["cestao"], sug["camada"], sug["para"]))
    if alvo:
        st.caption("A sugestão é uma pergunta feita ao modelo (o que muda se essa camada fosse de pesado?), "
                   "não uma regra de operação: confira com a equipe do pátio se o material está disponível.")

with st.expander("Receita em tabela"):
    linhas = [{"Cestão": i, "Camada": j + 1, "Material": c["codigo"],
               "Descrição": MATS.loc[c["codigo"], "descricao"].title(), "Classe": MATS.loc[c["codigo"], "classe"],
               "Peso (t)": c["peso_t"], "Volume (m³)": round(c["peso_t"] / MATS.loc[c["codigo"], "densidade"], 2)}
              for i, cam in enumerate(cestoes, start=1) for j, c in enumerate(cam)]
    if linhas:
        st.dataframe(linhas, hide_index=True)
    else:
        st.caption("Nenhuma camada ainda.")

st.markdown('<div class="rodape">Modelo: média de 5 Gradient Boosting</div>', unsafe_allow_html=True)

# ---------- aviso central (por último, com a página já desenhada atrás) ----------
if ss.erro:
    abrir_aviso(ss.erro)
