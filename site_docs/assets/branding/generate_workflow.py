"""Generate RustScenic's input-to-output product overview."""
from pathlib import Path
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parent


def render(dark=False):
    ink = '#e6edf3' if dark else '#24292f'
    muted = '#9da7b3' if dark else '#57616c'
    surface = '#111820' if dark else '#f6f8fa'
    border = '#303b47' if dark else '#d8dfe5'
    orange, teal = '#d46f40', '#449f95'
    logo = ET.parse(OUT/'rustscenic-logo.svg').getroot().find('{http://www.w3.org/2000/svg}path').attrib['d']
    parts = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="410" viewBox="0 0 1120 410" role="img" aria-labelledby="title desc">
  <title id="title">Inputs → RustScenic → outputs</title>
  <desc id="desc">Single-cell RNA expression, a TF list and motif databases enter RustScenic: one Python package with a Rust backend running on CPUs. It infers gene networks, refines regulons with motifs and scores cell activity. Outputs are regulatory networks, regulons and cell-by-regulon activity scores. Optional ATAC and gene coordinates enable enhancer-gene links and eRegulons. Illustrations are conceptual.</desc>
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{orange}"/></marker>
  </defs>
  <style>
    text {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; fill: {ink}; }}
    .label {{ font-size: 21px; font-weight: 650; }}
    .caption {{ font-size: 16px; fill: {muted}; }}
    .small {{ font-size: 14px; fill: {muted}; }}
    .panel {{ fill: {surface}; stroke: {border}; stroke-width: 1.3; }}
  </style>''']

    def text(x,y,value,cls='caption',fill=None,size=None,anchor='middle',weight=None):
        style=(f'fill:{fill};' if fill else '')+(f'font-size:{size}px;' if size else '')+(f'font-weight:{weight};' if weight else '')
        parts.append(f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}" style="{style}">{escape(value)}</text>')

    def rect(x,y,w,h,fill,rx=0,extra=''):
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" {extra}/>')

    def path(d,stroke,extra=''):
        parts.append(f'<path d="{d}" fill="none" stroke="{stroke}" {extra}/>')

    def circle(x,y,r,fill):
        parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}"/>')

    text(140,24,'INPUTS','small',size=16,weight=650)
    text(560,24,'ONE PACKAGE','small',fill=orange,size=16,weight=650)
    text(980,24,'OUTPUTS','small',size=16,weight=650)
    for x in (16,856):
        parts.append(f'<rect x="{x}" y="44" width="248" height="348" rx="16" class="panel"/>')
    parts.append(f'<rect x="320" y="44" width="480" height="348" rx="20" fill="{surface}" stroke="{orange}" stroke-width="1.8"/>')
    for x in (274,810):
        path(f'M {x} 210 H {x+34}',orange,'stroke-width="2.8" marker-end="url(#arrow)"')

    # Inputs: ready expression matrices and the references used for refinement.
    text(140,79,'Single-cell RNA','label')
    expression = [[.12,.6,.3,.85,.15,.5,.2], [.3,.2,.8,.15,.55,.1,.65],
                  [.85,.15,.25,.65,.3,.6,.15], [.4,.7,.2,.1,.85,.35,.6]]
    for row,values in enumerate(expression):
        for col,value in enumerate(values):
            rect(63+col*23,99+row*20,19,16,teal,3,f'opacity="{value+.08}"')
    text(140,193,'Expression: cells × genes','caption')
    path('M 40 212 H 240',border,'stroke-width="1"')
    text(140,241,'Optional ATAC','label',size=18)
    path('M 65 291 h 18 l 7 -22 l 7 22 h 15 l 9 -34 l 9 34 h 14 l 6 -15 l 6 15 h 14 l 9 -28 l 9 28 h 17',teal,'stroke-width="2.2" stroke-linejoin="round"')
    text(140,316,'Accessibility: cells × peaks','caption',size=15)
    text(140,352,'TF list + motif databases','caption',size=15)
    text(140,375,'Gene coordinates for enhancer links','small',size=12)

    # A visible product container packages the analysis stages together.
    parts.append(f'<path transform="translate(382 77) scale(.072)" fill="{orange}" fill-rule="evenodd" d="{logo}"/>')
    text(590,132,'RustScenic','label',size=42,weight=700)
    text(560,174,'One Python package','label',size=24)
    text(560,204,'Python API · Rust backend · CPU','caption',size=17)
    path('M 346 228 H 774',border,'stroke-width="1"')
    for x,title,caption in [(342,'Infer','gene networks'),(496,'Refine','with motifs'),(650,'Score','cell activity')]:
        rect(x,249,128,67,surface,10,f'stroke="{border}" stroke-width="1.2"')
        text(x+64,275,title,'label',size=20)
        text(x+64,300,caption,'small',size=14)
    for x in (478,632):
        path(f'M {x} 282 h 10',muted,'stroke-width="1.5"')
        path(f'M {x+6} 278 l 4 4 l -4 4',muted,'stroke-width="1.5"')
    rect(410,340,300,34,surface,8,f'stroke="{border}" stroke-width="1"')
    text(560,363,'pip install rustscenic','caption',fill=ink,size=18)

    # Outputs: interpretable network objects and per-cell activity matrices.
    text(980,79,'Networks + regulons','label',size=20)
    for y,weight in [(113,2),(149,3.2),(185,1.3)]:
        path(f'M 937 149 L 1021 {y}',orange,f'stroke-width="{weight}" marker-end="url(#arrow)"')
        circle(1034,y,14,border)
        text(1034,y+5,'g'+str([113,149,185].index(y)+1),'small',fill=ink,size=12)
    circle(918,149,23,orange)
    text(918,155,'TF','label',fill='#ffffff',size=18)
    text(980,216,'eRegulons with optional ATAC','small',size=14)
    path('M 880 231 H 1080',border,'stroke-width="1"')
    text(980,262,'Cell activity scores','label',size=20)
    activity = [[.12,.25,.8,.72,.18,.35], [.2,.15,.9,.6,.28,.15],
                [.78,.6,.15,.22,.8,.55], [.9,.72,.25,.1,.65,.7]]
    for row,values in enumerate(activity):
        for col,value in enumerate(values):
            rect(898+col*27,277+row*18,23,14,orange,2,f'opacity="{value+.06}"')
    text(980,375,'Activity: cells × regulons','caption',size=15)
    parts.append('</svg>')
    return '\n'.join(parts)+'\n'


if __name__ == '__main__':
    for mode in ('light','dark'):
        (OUT/f'rustscenic-workflow-{mode}.svg').write_text(render(mode=='dark'),encoding='utf-8')
