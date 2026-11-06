"""
Desenho dos cestões em SVG (com animação em CSS), para mostrar no Streamlit
com streamlit.components.v1.html, e o painel de resultado.

- Cada cestão (e a panela de gusa) desce de cima quando é criado.
- A altura de cada camada é proporcional ao VOLUME dela (peso ÷ densidade),
  na mesma escala para todos: a linha tracejada é a boca do cestão (78 m³).
- Cada material tem uma cor sólida própria (tabela COR_MATERIAL), com legenda.
- A gusa líquida (carregamento 3) aparece como uma panela à direita.

Identidade visual da EVCOMX: laranja forte, preto e branco, fonte Bai Jamjuree.
Para trocar o laranja, mude LARANJA aqui e primaryColor em .streamlit/config.toml.
"""
from __future__ import annotations

import html

import pandas as pd

# ---------------------------------------------------------------- identidade EVCOMX
LARANJA = "#FF6B00"
LARANJA_ESCURO = "#D95A00"
PRETO = "#111111"
BRANCO = "#FFFFFF"
CINZA = "#6B6B6B"
CINZA_CLARO = "#ECECEC"
CINZA_LINHA = "#D6D6D6"

# Fonte da EVCOMX: Bai Jamjuree (Google Fonts). Se não carregar, cai para a do sistema.
FONTE = "'Bai Jamjuree', 'Segoe UI', Arial, sans-serif"
URL_FONTE = "https://fonts.googleapis.com/css2?family=Bai+Jamjuree:wght@400;500;600;700&display=swap"
FONT_FACE = f"@import url('{URL_FONTE}');"
LINK_FONTE = f'<link href="{URL_FONTE}" rel="stylesheet">'

# ---------------------------------------------------------------- uma cor por material
# Tons claros para leve, médios para misto e escuros para pesado, para a classe
# continuar legível; nenhuma cor perto do laranja, que fica reservado para o alarme.
COR_MATERIAL = {
    # leve
    "SHRE": "#7FC8F8", "ESPS": "#B5E48C", "CAVA": "#FFE066", "RETL": "#F4A6C6",
    "PACB": "#C3AED6", "ESBS": "#9BE7D8", "MARP": "#E9EDC9",
    # misto
    "OXIC": "#3A86FF", "MISM": "#2A9D8F", "PAEP": "#6A994E", "MISB": "#9B5DE5",
    "ESPR": "#00B4D8", "CAFO": "#D4A017", "PAEB": "#C9184A",
    # pesado
    "GUSS": "#1D3557", "194032": "#6A040F", "RECA": "#2D6A4F", "SUGU": "#3C096C",
    "118441": "#7F5539", "FOFO": "#283618", "RECC": "#005F73", "RECG": "#8D99AE",
}

# geometria (unidades do viewBox)
LARGURA_SVG, ALTURA_SVG = 750, 540   # largura enxuta: o desenho fica maior na coluna
FUNDO_Y = 400            # fundo interno do cestão
ALTURA_BOCA = 220        # 78 m³ ocupam 220 px
CAPACIDADE_M3 = 78.0
PX_POR_M3 = ALTURA_BOCA / CAPACIDADE_M3
LARG_INTERNA = 150
TETO_DESENHO_Y = 86      # ~110 m³; acima disso a pilha é cortada (seta)
TOPO_VISTA = 60          # sem a ponte rolante, o desenho começa mais embaixo


def _fmt(x: float, casas: int = 1) -> str:
    return f"{x:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def cor_material(cod: str) -> str:
    return COR_MATERIAL.get(cod, "#999999")


def cor_texto(fundo: str) -> str:
    """Preto ou branco, o que contrastar mais com a cor de fundo."""
    r, g, b = (int(fundo[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lin = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    lum = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)
    return PRETO if lum > 0.32 else BRANCO


def amostra_material(cod: str, tamanho: int = 26) -> str:
    """Quadradinho com a cor do material (para legendas e para o seletor)."""
    return (f'<svg width="{tamanho}" height="{tamanho}" viewBox="0 0 {tamanho} {tamanho}" style="flex:none">'
            f'<rect x="0.5" y="0.5" width="{tamanho-1}" height="{tamanho-1}" rx="5" fill="{cor_material(cod)}" '
            f'stroke="{PRETO}" stroke-opacity=".25"/></svg>')


def _cestao_svg(i: int, x0: float, camadas: list[dict], materiais: pd.DataFrame,
                ativo: bool, anim: dict | None, res: dict | None) -> str:
    """Um cestão com as camadas. x0 = canto esquerdo da área interna."""
    xm = x0 + LARG_INTERNA / 2
    boca_y = FUNDO_Y - ALTURA_BOCA
    novo_cestao = bool(anim and anim.get("tipo") == "cestao" and anim.get("cestao") in (i, "todos"))
    camada_nova = anim.get("camada") if anim and anim.get("tipo") == "camada" and anim.get("cestao") == i else None
    resultado_anim = bool(anim and anim.get("tipo") == "resultado")

    # ---------- camadas ----------
    partes, y, volume = [], FUNDO_Y, 0.0
    rotulos_lado, ultimo_rotulo_lado = [], FUNDO_Y + 20
    for j, cam in enumerate(camadas):
        mat = materiais.loc[cam["codigo"]]
        vol = cam["peso_t"] / mat["densidade"]
        volume += vol
        h = vol * PX_POR_M3
        y_top = y - h
        fundo = cor_material(cam["codigo"])
        if y_top >= TETO_DESENHO_Y:
            classe_anim = ' class="camada-nova"' if j == camada_nova else ""
            texto = f'{html.escape(cam["codigo"])} · {_fmt(cam["peso_t"])} t'
            rotulo = ""
            if h < 12:
                # camada fina (pesado ocupa pouco volume): rótulo do lado, com linha guia
                y_rot = min(y_top + h / 2 + 3.5, ultimo_rotulo_lado - 11)
                ultimo_rotulo_lado = y_rot
                rotulos_lado.append(
                    f'<line x1="{x0 - 22}" y1="{y_rot - 3.5:.1f}" x2="{x0 + 6}" y2="{y_top + h/2:.1f}" '
                    f'stroke="{PRETO}" stroke-width=".8" stroke-opacity=".5"/>'
                    f'<text x="{x0 - 25}" y="{y_rot:.1f}" text-anchor="end" class="rot-lado" fill="{PRETO}">{texto}</text>')
            else:
                rotulo = (f'<text x="{xm}" y="{y_top + h/2 + 3.6:.1f}" text-anchor="middle" class="rot-camada" '
                          f'fill="{cor_texto(fundo)}">{texto}</text>')
            partes.append(
                f'<g{classe_anim} style="--queda:{-(y_top - 30):.0f}px">'
                f'<rect x="{x0}" y="{y_top:.2f}" width="{LARG_INTERNA}" height="{h:.2f}" fill="{fundo}" '
                f'stroke="{BRANCO}" stroke-opacity=".8" stroke-width=".8"/>{rotulo}'
                f'<title>Camada {j+1}: {html.escape(cam["codigo"])} – {html.escape(mat["descricao"])} '
                f'({mat["classe"]}) · {_fmt(cam["peso_t"])} t · {_fmt(vol)} m³</title></g>')
        y = y_top
    transbordou = volume > CAPACIDADE_M3
    camadas_svg = "".join(partes)
    if y < TETO_DESENHO_Y:
        camadas_svg += (f'<path d="M{xm-12},{TETO_DESENHO_Y+2} L{xm},{TETO_DESENHO_Y-12} L{xm+12},{TETO_DESENHO_Y+2} Z" '
                        f'fill="{LARANJA}"/>')

    # ---------- corpo de aço, alça, gancho e cabo ----------
    cor_borda, classe_corpo = PRETO, ""
    if res is not None and res["alarme"]:
        cor_borda, classe_corpo = LARANJA, ' class="pulsa"'
    x1, x2 = x0 - 7, x0 + LARG_INTERNA + 7
    corpo = (
        f'<path d="M{x1+6},{boca_y+6} C{x1+10},{boca_y-58} {x2-10},{boca_y-58} {x2-6},{boca_y+6}" '
        f'fill="none" stroke="{PRETO}" stroke-width="4" stroke-linecap="round"/>'
        f'<g{classe_corpo}><path class="contorno" d="M{x1},{boca_y-2} L{x1},{FUNDO_Y-6} Q{x1},{FUNDO_Y+8} {x1+14},{FUNDO_Y+8} '
        f'L{x2-14},{FUNDO_Y+8} Q{x2},{FUNDO_Y+8} {x2},{FUNDO_Y-6} L{x2},{boca_y-2}" '
        f'fill="none" stroke="{cor_borda}" stroke-width="7" stroke-linejoin="round" stroke-linecap="round"/></g>'
        f'<rect x="{x1-12}" y="{boca_y+40}" width="12" height="20" rx="3" fill="{PRETO}"/>'
        f'<rect x="{x2}" y="{boca_y+40}" width="12" height="20" rx="3" fill="{PRETO}"/>'
    )
    boca = (f'<line x1="{x1-18}" y1="{boca_y}" x2="{x2+18}" y2="{boca_y}" stroke="{CINZA}" '
            f'stroke-width="1.4" stroke-dasharray="5 4"/>'
            f'<text x="{x2+20}" y="{boca_y+4}" class="mini" fill="{CINZA}">78 m³</text>')
    if transbordou:
        boca += (f'<text x="{x2+20}" y="{boca_y+18}" class="mini" fill="{LARANJA}">'
                 f'+{_fmt(volume - CAPACIDADE_M3)} m³</text>')

    peso = sum(c["peso_t"] for c in camadas)
    ocup = volume / CAPACIDADE_M3
    titulo = (f'<text x="{xm}" y="{FUNDO_Y+34}" text-anchor="middle" class="titulo-cestao" '
              f'fill="{LARANJA if ativo and res is None else PRETO}">CESTÃO {i}</text>')
    if ativo and res is None:
        titulo += f'<rect x="{xm-50}" y="{FUNDO_Y+41}" width="100" height="3" rx="1.5" fill="{LARANJA}"/>'
    if camadas:
        info = (f'<text x="{xm}" y="{FUNDO_Y+60}" text-anchor="middle" class="info" '
                f'fill="{LARANJA if transbordou else CINZA}">{_fmt(peso)} t · {_fmt(volume)} m³ · {ocup:.0%}</text>')
    else:
        info = (f'<text x="{xm}" y="{FUNDO_Y-ALTURA_BOCA/2}" text-anchor="middle" class="info" fill="{CINZA}">'
                f'vazio</text><text x="{xm}" y="{FUNDO_Y-ALTURA_BOCA/2+16}" text-anchor="middle" class="mini" '
                f'fill="{CINZA}">insira as camadas</text>')

    # ---------- selo do resultado: só a chance ----------
    selo = ""
    if res is not None:
        cor = LARANJA if res["alarme"] else PRETO
        selo_cls = ' class="selo"' if resultado_anim else ""
        selo = (f'<g{selo_cls} style="transform-origin:{xm}px {FUNDO_Y+95}px">'
                f'<rect x="{xm-100}" y="{FUNDO_Y+72}" width="200" height="46" rx="12" fill="{cor}"/>'
                f'<text x="{xm-86}" y="{FUNDO_Y+104}" class="selo-num">{res["chance"]:.0%}</text>'
                f'<text x="{xm-22}" y="{FUNDO_Y+92}" class="selo-txt">de chance de</text>'
                f'<text x="{xm-22}" y="{FUNDO_Y+107}" class="selo-txt">carga alta</text></g>')

    grupo_cls = ' class="cestao-novo"' if novo_cestao else ""
    return (f'<g{grupo_cls}>{corpo}<g clip-path="url(#clip-c{i})">{camadas_svg}</g>{"".join(rotulos_lado)}{boca}</g>'
            f'<clipPath id="clip-c{i}"><rect x="{x0}" y="{TETO_DESENHO_Y-20}" width="{LARG_INTERNA}" '
            f'height="{FUNDO_Y - TETO_DESENHO_Y + 20}"/></clipPath>{titulo}{info}{selo}')


def _panela_svg(x0: float, gusa_t: float, anim: dict | None) -> str:
    """Panela de gusa líquida (carregamento 3)."""
    nova = bool(anim and (anim.get("tipo") == "gusa" or anim.get("cestao") == "todos"))
    larg_topo, larg_fundo, alt = 150, 112, 150
    topo_y = FUNDO_Y - alt
    xm = x0 + larg_topo / 2
    nivel = min(max(gusa_t / 60.0, 0.12), 1.0)        # 60 t enche a panela no desenho
    liq_y = FUNDO_Y - alt * nivel * 0.92
    dx = (larg_topo - larg_fundo) / 2
    contorno = f'M{x0},{topo_y} L{x0+dx},{FUNDO_Y} L{x0+larg_topo-dx},{FUNDO_Y} L{x0+larg_topo},{topo_y}'
    return (
        f'<g class="{"cestao-novo" if nova else ""}">'
        f'<path d="M{x0+6},{topo_y+4} C{x0+14},{topo_y-60} {x0+larg_topo-14},{topo_y-60} {x0+larg_topo-6},{topo_y+4}" '
        f'fill="none" stroke="{PRETO}" stroke-width="4" stroke-linecap="round"/>'
        f'<clipPath id="clip-panela"><path d="{contorno} Z"/></clipPath>'
        f'<g clip-path="url(#clip-panela)">'
        f'<rect x="{x0}" y="{liq_y:.1f}" width="{larg_topo}" height="{FUNDO_Y-liq_y:.1f}" fill="url(#gusa)"/>'
        f'<g class="onda"><path d="M{x0-60},{liq_y:.1f} q15,-6 30,0 t30,0 t30,0 t30,0 t30,0 t30,0 t30,0 t30,0 v8 h-240 Z" '
        f'fill="#ffd27a" opacity=".75"/></g></g>'
        f'<path d="{contorno}" fill="none" stroke="{PRETO}" stroke-width="7" stroke-linejoin="round"/>'
        f'<circle class="faisca f1" cx="{xm-20}" cy="{liq_y-4:.1f}" r="2" fill="#ffb347"/>'
        f'<circle class="faisca f2" cx="{xm+14}" cy="{liq_y-4:.1f}" r="1.6" fill="#ffd27a"/>'
        f'<circle class="faisca f3" cx="{xm+30}" cy="{liq_y-4:.1f}" r="1.8" fill="#ff8c32"/>'
        f'</g>'
        f'<text x="{xm}" y="{FUNDO_Y+34}" text-anchor="middle" class="titulo-cestao" fill="{PRETO}">CARREGAMENTO 3</text>'
        f'<text x="{xm}" y="{FUNDO_Y+60}" text-anchor="middle" class="info" fill="{LARANJA_ESCURO}">Gusa líquida · {_fmt(gusa_t, 0 if float(gusa_t).is_integer() else 1)} t</text>')


def desenhar_corrida(cestoes: list[list[dict]], materiais: pd.DataFrame, ativo: int | None,
                     gusa_t: float | None, anim: dict | None = None,
                     resultado: dict | None = None, codigo_corrida: str = "") -> str:
    """HTML completo (para components.html) com a corrida desenhada."""
    codigos = sorted({c["codigo"] for camadas in cestoes for c in camadas})
    defs = ('<linearGradient id="gusa" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#ffb347"/>'
            '<stop offset=".45" stop-color="#ff7a1a"/><stop offset="1" stop-color="#b8360b"/></linearGradient>')

    slots = [110, 360]
    corpo = []
    for i, camadas in enumerate(cestoes, start=1):
        res = resultado["cestoes"][i - 1] if resultado else None
        corpo.append(_cestao_svg(i, slots[i - 1], camadas, materiais, ativo == i, anim, res))
    if gusa_t:
        corpo.append(_panela_svg(585, gusa_t, anim))
    if not cestoes:
        corpo.append(f'<text x="{LARGURA_SVG/2}" y="230" text-anchor="middle" class="vazio-txt">'
                     f'Corrida aberta. Clique em <tspan font-weight="700" fill="{LARANJA}">Novo cestão</tspan> '
                     f'para começar.</text>')

    legenda = ""
    if codigos:
        itens = []
        for cod in codigos:
            m = materiais.loc[cod]
            itens.append(f'<div class="leg-item">{amostra_material(cod, 20)}'
                         f'<span><b>{html.escape(cod)}</b> – {html.escape(m["descricao"].title())} '
                         f'<i>({m["classe"].lower()})</i></span></div>')
        legenda = f'<div class="legenda">{"".join(itens)}</div>'

    return f"""<!doctype html><html><head><meta charset="utf-8">{LINK_FONTE}
<style>
  html,body{{margin:0;background:transparent;font-family:{FONTE};}}
  .quadro{{background:{BRANCO};border:1px solid {CINZA_LINHA};border-radius:16px;padding:6px 8px 10px}}
  svg.cena{{width:100%;height:auto;display:block}}
  svg text{{font-family:{FONTE}}}
  .titulo-cestao{{font-size:17px;font-weight:700;letter-spacing:.5px}}
  .info{{font-size:13px;font-weight:600}}
  .mini{{font-size:11px;font-weight:600}}
  .rot-camada{{font-size:10.5px;font-weight:700}}
  .vazio-txt{{font-size:17px;fill:{CINZA}}}
  .selo-num{{font-size:26px;font-weight:800;fill:{BRANCO}}}
  .selo-txt{{font-size:12px;font-weight:700;fill:{BRANCO}}}
  .rot-lado{{font-size:10px;font-weight:700}}
  .legenda{{display:flex;flex-wrap:wrap;gap:6px 18px;padding:8px 10px 0;color:{PRETO};font-size:12.5px}}
  .leg-item{{display:flex;align-items:center;gap:7px}}
  .leg-item i{{color:{CINZA};font-style:normal}}

  /* cestão novo: desce de cima */
  .cestao-novo{{animation:descer 1.1s cubic-bezier(.3,1.25,.5,1) both}}
  @keyframes descer{{0%{{transform:translateY(-470px)}}100%{{transform:translateY(0)}}}}
  .cestao-novo .contorno{{stroke-dasharray:900;stroke-dashoffset:900;animation:tracar 1.2s .35s ease-out forwards}}
  @keyframes tracar{{to{{stroke-dashoffset:0}}}}

  /* camada nova: cai de cima e assenta */
  .camada-nova{{animation:cair .75s cubic-bezier(.45,0,.75,1.25) both}}
  @keyframes cair{{0%{{transform:translateY(var(--queda));opacity:.2}}75%{{transform:translateY(3px);opacity:1}}100%{{transform:translateY(0)}}}}

  /* resultado */
  .selo{{animation:aparecer .6s .2s cubic-bezier(.3,1.6,.5,1) both}}
  @keyframes aparecer{{0%{{transform:scale(0)}}100%{{transform:scale(1)}}}}
  .pulsa{{animation:pulso 1.4s ease-in-out infinite}}
  @keyframes pulso{{0%,100%{{filter:drop-shadow(0 0 0 rgba(255,107,0,0))}}50%{{filter:drop-shadow(0 0 10px rgba(255,107,0,.9))}}}}

  /* gusa líquida */
  .onda{{animation:ondular 2.4s linear infinite}}
  @keyframes ondular{{to{{transform:translateX(60px)}}}}
  .faisca{{animation:faisca 1.6s ease-out infinite;opacity:0}}
  .f2{{animation-delay:.5s}} .f3{{animation-delay:1s}}
  @keyframes faisca{{0%{{transform:translateY(0);opacity:1}}100%{{transform:translateY(-34px);opacity:0}}}}

  @media (prefers-reduced-motion: reduce){{*{{animation:none!important}}.contorno{{stroke-dashoffset:0!important}}}}
</style></head><body>
<div class="quadro">
<svg class="cena" viewBox="0 {TOPO_VISTA} {LARGURA_SVG} {ALTURA_SVG - TOPO_VISTA}" role="img" aria-label="Desenho da corrida com os cestões">
<defs>{defs}</defs>
{''.join(corpo)}
</svg>
{legenda}
</div></body></html>"""


# ====================================================================== resultado
FORMATO_FEATURE = {
    "densidade_media": lambda v: f"{_fmt(v, 2)} t/m³",
    "prop_peso_pesado": lambda v: f"{v:.0%}",
    "peso_por_ciclo_t": lambda v: f"{_fmt(v)} t",
    "prop_PACB": lambda v: f"{v:.0%}",
}
EXPLICA_FEATURE = {
    "densidade_media": "cestão fofo (pouca tonelada por m³) sobe mais para o mesmo peso",
    "prop_peso_pesado": "o pesado compacta a carga; pouco pesado = cestão volumoso",
    "peso_por_ciclo_t": "ciclos grossos = leve acumulado sem pesado intercalado",
    "prop_PACB": "pacote de chaparia: volumoso para o peso",
}
NOME_FEATURE_HTML = {
    "densidade_media": "Densidade média", "prop_peso_pesado": "Pesado no peso",
    "peso_por_ciclo_t": "Toneladas por ciclo", "prop_PACB": "PACB no peso",
}


def _barra_chance(r: dict, parada_media: float) -> str:
    w = 420
    cor = LARANJA if r["alarme"] else PRETO
    xl = r["limiar"] * w
    esperado = r["chance"] * parada_media
    situacao = (f'<b class="tag-alarme">ALARME</b>: passou do limiar de {r["limiar"]:.0%}.' if r["alarme"]
                else f'<b>Sem alarme</b>: abaixo do limiar de {r["limiar"]:.0%}.')
    return f"""
<div class="linha-chance">
  <div class="nome-c">CESTÃO {r['cestao']}</div>
  <div>
    <svg width="{w+60}" height="44" viewBox="0 0 {w+60} 44" class="barra">
      <rect x="0" y="10" width="{w}" height="18" rx="4" fill="{CINZA_CLARO}"/>
      <rect x="0" y="10" width="{max(r['chance']*w, 2):.1f}" height="18" rx="4" fill="{cor}" class="cresce"/>
      <line x1="{xl:.1f}" y1="4" x2="{xl:.1f}" y2="34" stroke="{PRETO}" stroke-width="2" stroke-dasharray="3 2"/>
      <text x="{xl:.1f}" y="44" text-anchor="middle" class="eixo">limiar {r['limiar']:.0%}</text>
      <text x="0" y="44" class="eixo">0%</text><text x="{w}" y="44" text-anchor="end" class="eixo">100%</text>
      <text x="{w+8}" y="25" class="pct" fill="{cor}">{r['chance']:.0%}</text>
    </svg>
    <div class="leitura">{situacao} Sem fazer nada, a perda esperada é de <b>{_fmt(esperado)} min</b> de forno
      ({r['chance']:.0%} × {_fmt(parada_media)} min, a parada média). Um alarme custa 1 min, então alarmar
      compensa acima do limiar.</div>
  </div>
</div>"""


def _faixa_feature(nome: str, ref: dict, valores: list[tuple[int, float, str]]) -> str:
    w, h = 470, 46
    lo = min([0.0 if nome == "prop_PACB" else ref["p10"] * 0.8] + [v for _, v, _ in valores])
    hi = max([max(ref["p90"] * 1.25, 0.12 if nome == "prop_PACB" else ref["p90"] * 1.25)] + [v * 1.05 for _, v, _ in valores])
    X = lambda v: 10 + (v - lo) / (hi - lo) * (w - 20)
    fmt = FORMATO_FEATURE[nome]
    marcas = []
    for i, (c, v, leitura) in enumerate(valores):
        cor = LARANJA if leitura == "mais risco" else PRETO
        dy = -9 if len(valores) > 1 and i == 0 else (9 if len(valores) > 1 else 0)
        marcas.append(f'<circle cx="{X(v):.1f}" cy="{22+dy}" r="8.5" fill="{cor}" stroke="{BRANCO}" stroke-width="1.5"/>'
                      f'<text x="{X(v):.1f}" y="{26+dy}" text-anchor="middle" class="marca">{c}</text>')
    textos = " · ".join(f'C{c}: <b>{fmt(v)}</b> <span class="{l.replace(" ", "-")}">({l})</span>' for c, v, l in valores)
    return f"""
<div class="linha-feat">
  <div class="nome-f">{NOME_FEATURE_HTML[nome]}<div class="expl">{EXPLICA_FEATURE[nome]}</div></div>
  <div>
    <svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">
      <rect x="{X(ref['p10']):.1f}" y="15" width="{max(X(ref['p90'])-X(ref['p10']), 2):.1f}" height="14" rx="3" fill="{CINZA_CLARO}"/>
      <rect x="{X(ref['p25']):.1f}" y="15" width="{max(X(ref['p75'])-X(ref['p25']), 2):.1f}" height="14" rx="3" fill="#C4C4C4"/>
      <line x1="{X(ref['p50']):.1f}" y1="12" x2="{X(ref['p50']):.1f}" y2="32" stroke="{CINZA}" stroke-width="1.5"/>
      {''.join(marcas)}
      <text x="{X(ref['p10']):.1f}" y="44" class="eixo" text-anchor="middle">{fmt(ref['p10'])}</text>
      <text x="{X(ref['p90']):.1f}" y="44" class="eixo" text-anchor="middle">{fmt(ref['p90'])}</text>
    </svg>
    <div class="valores">{textos}</div>
  </div>
</div>"""


def _chance_corrida_html(c: dict) -> str:
    """Cartão com a chance da corrida e o (?) que mostra a conta ao passar o mouse."""
    if c["caso"] == "um_cestao":
        conta = f"P(corrida) = P₁ = {c['p1']:.0%}"
        texto = "Com um cestão só, a chance da corrida é a chance dele."
    elif c["caso"] == "c1_deu":
        conta = "P(corrida) = 100%"
        texto = "O cestão 1 já deu carga alta, então a corrida já deu."
    elif c["caso"] == "c1_nao_deu":
        conta = f"P(corrida) = P₂ = {c['p2']:.0%}"
        texto = "O cestão 1 não deu carga alta, então só falta o risco do cestão 2."
    else:
        conta = (f"P(corrida) = 1 − (1 − P₁) × (1 − P₂)<br>"
                 f"= 1 − (1 − {c['p1']:.0%}) × (1 − {c['p2']:.0%}) = {c['chance']:.0%}")
        texto = ("A corrida só escapa se nenhum cestão der carga alta. P₁ é a chance do cestão 1. P₂ é a chance do "
                 "cestão 2 se o cestão 1 não der, por isso pode diferir da barra do cestão 2.")
    return f"""<div class="corrida">
 <div class="c-rot">Chance de carga alta na corrida
  <span class="ajuda" tabindex="0" role="button" aria-label="Como este número é calculado" aria-describedby="conta">?
   <span class="dica" id="conta" role="tooltip"><b class="conta">{conta}</b>{texto}
   <em>O alarme não usa este número: ele vem do limiar de minutos de cada cestão, e pode disparar abaixo de 50%
   porque um alarme falso custa menos que a parada.</em></span>
  </span>
 </div>
 <div class="c-num">{c['chance']:.0%}</div>
</div>"""


def painel_resultado_html(prev: dict, modo: str, referencia: dict, parada_media: float) -> tuple[str, int]:
    """HTML do resultado e a altura sugerida do iframe."""
    from motor import ler_feature  # import local para desenho.py não depender do modelo
    alarmados = [r for r in prev["cestoes"] if r["alarme"]]
    if prev["alarme_corrida"]:
        quais = " e ".join(f"cestão {r['cestao']} ({r['chance']:.0%})" for r in alarmados)
        titulo = "Esta corrida deve dar carga alta"
        sub = f"Passou do limiar de minutos (modo <b>{modo}</b>): {quais}."
        classe, icone = "veredito alarme", "!"
    else:
        maior = max(prev["cestoes"], key=lambda r: r["chance"])
        titulo = "Sem carga alta prevista para esta corrida"
        sub = (f"Nenhum cestão passou do limiar (modo <b>{modo}</b>). "
               f"O de maior chance é o cestão {maior['cestao']}, com {maior['chance']:.0%}.")
        classe, icone = "veredito ok", "✓"

    barras = "".join(_barra_chance(r, parada_media) for r in prev["cestoes"])
    feats = []
    for nome in ["prop_peso_pesado", "densidade_media", "peso_por_ciclo_t", "prop_PACB"]:
        valores = [(r["cestao"], r["x"][nome], ler_feature(nome, r["x"][nome], referencia)) for r in prev["cestoes"]]
        feats.append(_faixa_feature(nome, referencia[nome], valores))

    altura = 160 + 96 + len(prev["cestoes"]) * 95 + 70 + 4 * 78 + 40
    html_out = f"""<!doctype html><html><head><meta charset="utf-8">{LINK_FONTE}
<style>
 body{{margin:0;font-family:{FONTE};color:{PRETO};background:transparent}}
 svg text{{font-family:{FONTE}}}
 .veredito{{border-radius:16px;padding:18px 22px;display:flex;gap:16px;align-items:center;color:{BRANCO};animation:entra .5s ease-out both}}
 .veredito.alarme{{background:{LARANJA}}}
 .veredito.ok{{background:{PRETO};border-left:8px solid {LARANJA}}}
 .veredito .ic{{font-size:30px;font-weight:800;width:52px;height:52px;border-radius:50%;background:rgba(255,255,255,.2);display:flex;align-items:center;justify-content:center;flex:none}}
 .veredito h2{{margin:0;font-size:23px;font-weight:700;letter-spacing:.3px}} .veredito p{{margin:5px 0 0;font-size:14px}}
 @keyframes entra{{from{{opacity:0;transform:translateY(-8px)}}to{{opacity:1;transform:none}}}}
 .corrida{{position:relative;display:flex;justify-content:space-between;align-items:center;gap:16px;margin-top:12px;padding:12px 22px;
   border:2px solid {PRETO};border-radius:16px}}
 .c-rot{{font-weight:700;font-size:15px;display:flex;align-items:center;gap:8px}}
 .c-num{{font-size:34px;font-weight:800;line-height:1}}
 .ajuda{{width:20px;height:20px;border-radius:50%;background:{PRETO};color:{BRANCO};font-size:13px;
   font-weight:700;display:inline-flex;align-items:center;justify-content:center;cursor:help;flex:none}}
 .ajuda:focus-visible{{outline:2px solid {LARANJA};outline-offset:2px}}
 .dica{{display:none;position:absolute;top:calc(100% + 6px);left:16px;right:16px;max-width:420px;z-index:5;box-sizing:border-box;padding:14px 16px;
   background:{PRETO};color:{BRANCO};border-radius:10px;font-size:13px;font-weight:500;line-height:1.45;cursor:default}}
 .ajuda:hover .dica,.ajuda:focus .dica{{display:block}}
 .dica .conta{{display:block;color:{LARANJA};font-size:14px;margin-bottom:8px}}
 .dica em{{display:block;font-style:normal;color:#CFCFCF;margin-top:8px}}
 h3{{font-size:16px;color:{PRETO};margin:22px 0 8px;font-weight:700;letter-spacing:.3px}}
 h3 span{{font-weight:500;color:{CINZA};font-size:12.5px;letter-spacing:0}}
 .linha-chance,.linha-feat{{display:grid;grid-template-columns:150px 1fr;gap:12px;align-items:start;padding:8px 0;border-bottom:1px solid {CINZA_CLARO}}}
 .nome-c{{font-weight:700;color:{PRETO};padding-top:12px;letter-spacing:.5px}}
 .nome-f{{font-weight:700;color:{PRETO};font-size:13.5px;padding-top:6px}}
 .expl{{font-weight:500;color:{CINZA};font-size:11.5px;margin-top:2px}}
 .eixo{{font-size:10.5px;fill:{CINZA}}} .pct{{font-size:18px;font-weight:800}}
 .marca{{font-size:10px;font-weight:800;fill:{BRANCO}}}
 .leitura,.valores{{font-size:12.5px;color:#333}} .tag-alarme{{color:{LARANJA_ESCURO}}}
 .mais-risco{{color:{LARANJA_ESCURO};font-weight:700}} .menos-risco{{color:{PRETO};font-weight:600}} .normal{{color:{CINZA}}}
 .cresce{{transform-origin:left;animation:cresce .9s ease-out both}}
 @keyframes cresce{{from{{transform:scaleX(0)}}to{{transform:scaleX(1)}}}}
 @media (prefers-reduced-motion: reduce){{*{{animation:none!important}}}}
</style></head><body>
<div class="{classe}"><div class="ic">{icone}</div><div><h2>{titulo}</h2><p>{sub}</p></div></div>
{_chance_corrida_html(prev["corrida"])}
<h3>CHANCE DE CARGA ALTA POR CESTÃO <span>· a corrida alarma se qualquer cestão passar do limiar de minutos</span></h3>
{barras}
<h3>ONDE A RECEITA FOGE DO NORMAL <span>· faixa clara = 80% dos cestões do treino, faixa escura = metade do meio, traço = mediana</span></h3>
{''.join(feats)}
</body></html>"""
    return html_out, altura
