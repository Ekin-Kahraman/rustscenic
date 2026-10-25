"""Generate the theme-aware, conceptual RustScenic README workflow."""
from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parent


def render(dark=False):
    ink = '#e6edf3' if dark else '#24292f'
    muted = '#9da7b3' if dark else '#57616c'
    surface = '#111820' if dark else '#f6f8fa'
    border = '#303b47' if dark else '#d8dfe5'
    orange, teal = '#d46f40', '#449f95'
    lines = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="370" viewBox="0 0 1120 370" role="img" aria-labelledby="title desc">
  <title id="title">RustScenic: from single-cell data to regulatory networks and activity</title>
  <desc id="desc">RNA expression and a TF list feed TF-to-gene inference. Motif rankings and annotations refine candidate networks into motif-supported regulons, which are scored across cells. Adding ATAC and gene coordinates enables topics, enhancer-to-gene links and TF-enhancer-gene networks called eRegulons. This is a conceptual workflow, not measured output.</desc>
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{muted}"/></marker>
  </defs>
  <style>
    text {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; fill: {ink}; }}
    .label {{ font-size: 22px; font-weight: 650; }}
    .caption {{ font-size: 17px; fill: {muted}; }}
    .small {{ font-size: 14px; fill: {muted}; }}
    .panel {{ fill: {surface}; stroke: {border}; stroke-width: 1.3; }}
    .flow {{ fill: none; stroke: {muted}; stroke-width: 2; marker-end: url(#arrow); }}
  </style>''']

    def text(x, y, value, cls='caption', anchor='middle', fill=None, size=None):
        attrs = f'x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}"'
        if fill or size:
            attrs += ' style="' + (f'fill:{fill};' if fill else '') + (f'font-size:{size}px;' if size else '') + '"'
        lines.append(f'  <text {attrs}>{escape(value)}</text>')

    def rect(x,y,w,h,fill,rx=0,extra=''):
        lines.append(f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" {extra}/>')

    def circle(x,y,r,fill,extra=''):
        lines.append(f'  <circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" {extra}/>')

    def path(d,stroke,extra=''):
        lines.append(f'  <path d="{d}" fill="none" stroke="{stroke}" {extra}/>')

    # Four steps are readable independently; the illustrations show output types.
    xs = [16,296,576,856]
    for x in xs:
        lines.append(f'  <rect x="{x}" y="30" width="248" height="212" rx="14" class="panel"/>')
    labels = ['RNA expression', 'Infer TF–gene links', 'Refine with motifs', 'Score cell activity']
    captions = ['Cells × genes', 'Candidate regulatory links', 'Motif-supported regulons', 'Cells × regulons']
    for x, label, caption in zip(xs, labels, captions):
        text(x+124,64,label,'label')
        text(x+124,221,caption)
    text(420,17,'+ TF list','small')
    text(700,17,'+ motif rankings / annotations','small')
    for x in (264,544,824):
        lines.append(f'  <path d="M {x+6} 136 H {x+23}" class="flow"/>')

    # RNA expression matrix: heterogeneous expression across genes and cells.
    rna = [[.12,.6,.3,.85,.15,.5,.2], [.3,.2,.8,.15,.55,.1,.65],
           [.85,.15,.25,.65,.3,.6,.15], [.4,.7,.2,.1,.85,.35,.6],
           [.15,.35,.65,.5,.2,.9,.3]]
    for row,values in enumerate(rna):
        for col,value in enumerate(values):
            rect(63+col*23,87+row*21,19,17,teal,3,f'opacity="{value+.08}"')

    # A candidate TF-to-target network, with weighted directed edges.
    for y,weight in [(98,2),(138,3.2),(178,1.3)]:
        path(f'M 374 138 L 454 {y}',orange,f'stroke-width="{weight}" marker-end="url(#arrow)" opacity=".85"')
        circle(467,y,15,border)
        text(467,y+5,'g'+str([98,138,178].index(y)+1),'small',fill=ink,size=13)
    circle(358,138,24,orange)
    text(358,144,'TF','label',fill='#ffffff',size=18)

    # Motif evidence retains supported targets and removes unsupported links.
    text(613,114,'A','label',fill=teal,size=24)
    text(633,114,'C','label',fill=orange,size=31)
    text(655,114,'G','label',fill=teal,size=40)
    text(678,114,'T','label',fill=orange,size=26)
    path('M 601 120 H 690',border,'stroke-width="1.3"')
    text(647,143,'motif evidence','small')
    for index,(y,keep) in enumerate([(101,True),(143,True),(181,False)],1):
        rect(724,y-16,66,25,border,5,'opacity=".8"' if keep else 'opacity=".3"')
        text(757,y+1,f'g{index}','small',fill=ink if keep else muted,size=14)
        if keep:
            path(f'M 705 {y-5} l 5 5 l 9 -12',teal,'stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round"')
        else:
            path(f'M 707 {y-11} l 10 10 M 717 {y-11} l -10 10',muted,'stroke-width="1.5" opacity=".45"')

    # Per-cell regulon activity heatmap, intentionally schematic.
    values = [[.12,.25,.8,.72,.18,.35], [.2,.15,.9,.6,.28,.15],
              [.78,.6,.15,.22,.8,.55], [.9,.72,.25,.1,.65,.7],
              [.25,.4,.55,.85,.3,.15]]
    for row,items in enumerate(values):
        for col,value in enumerate(items):
            rect(898+col*27,87+row*21,23,17,orange,2,f'opacity="{value+.06}"')

    # Optional multiome branch. Upper GRN and motif evidence also feed eRegulons.
    path('M 420 242 V 259 H 980 V 277',muted,'stroke-width="1.3" stroke-dasharray="4 5" marker-end="url(#arrow)" opacity=".65"')
    path('M 700 242 V 259',muted,'stroke-width="1.3" stroke-dasharray="4 5" opacity=".65"')
    lines.append('  <rect x="16" y="280" width="248" height="65" rx="12" class="panel" stroke-dasharray="5 4"/>')
    lines.append('  <rect x="296" y="280" width="528" height="65" rx="12" class="panel"/>')
    lines.append('  <rect x="856" y="280" width="248" height="65" rx="12" class="panel"/>')
    text(140,307,'Optional ATAC','caption',fill=ink,size=20)
    text(140,331,'+ gene coordinates','small',fill=muted,size=15)
    for x in (264,824):
        lines.append(f'  <path d="M {x+6} 312 H {x+23}" class="flow"/>')
    # Accessibility profile icon.
    path('M 316 328 h 9 l 5 -12 l 4 12 h 7 l 7 -25 l 7 25 h 6 l 5 -17 l 5 17 h 9',teal,'stroke-width="2.2" stroke-linejoin="round"')
    text(597,320,'Topics + enhancer–gene links','label',size=22)
    text(980,307,'TF–enhancer–gene','caption',fill=ink,size=19)
    text(980,330,'networks (eRegulons)','caption',size=17)
    lines.append('</svg>')
    return '\n'.join(lines)+'\n'


if __name__ == '__main__':
    for mode in ('light','dark'):
        (OUT/f'rustscenic-workflow-{mode}.svg').write_text(render(mode=='dark'))
