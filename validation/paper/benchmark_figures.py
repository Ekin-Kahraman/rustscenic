"""Create branded RNA/paired benchmark figures from verified individual runs."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
E = ROOT / 'docs/paper/evidence'
OUT = ROOT / 'docs/benchmarks/atlas-workflows-20261109'
OUT.mkdir(exist_ok=True)
for entry in json.loads((E/'manifest.json').read_text())['files']:
    if entry['path'] == 'scaling/consolidated-runs.csv':
        assert hashlib.sha256((E/entry['path']).read_bytes()).hexdigest() == entry['sha256']
runs = pd.read_csv(E/'scaling/consolidated-runs.csv')
# Layout, palette and typography match the live README's make_figure.py.
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'path','svg.hashsalt':'rustscenic-atlas-20261109',
    'pdf.fonttype':42,'font.size':11,'axes.spines.top':False,'axes.spines.right':False})


def draw(dataset, dark=False, mobile=False):
    paired = dataset == 'hdma'
    comparator = 'SCENIC+' if paired else 'pySCENIC'
    rows = runs[runs.dataset.eq(dataset)]
    final = rows[rows.cells.eq(rows.cells.max())]
    med = final.groupby('method')[['input_to_export_s','peak_pss_gb']].median()
    gain = med.loc['reference','input_to_export_s']/med.loc['rustscenic','input_to_export_s']
    memory = 100*(1-med.loc['rustscenic','peak_pss_gb']/med.loc['reference','peak_pss_gb'])
    bg, fg, muted, grid = ('#141414','#E6EDF3','#9DA7B3','#303B47') if dark else ('#FFFFFF','#24292F','#57616C','#D8DFE5')
    orange, teal = '#D46F40', '#449F95'
    fig = plt.figure(figsize=(8.4,12.4) if mobile else (15.4,8.6), facecolor=bg)
    title = 'RustScenic: scaling to 817,740 paired RNA + ATAC cells' if paired else 'RustScenic: scaling to 1.3 million real RNA cells'
    if mobile:
        title = 'RustScenic: scaling to\n817,740 paired cells' if paired else 'RustScenic: scaling to\n1.3 million real RNA cells'
    fig.text(.07 if mobile else .055,.97 if mobile else .955,title,color=fg,fontsize=21 if mobile else 26,weight='bold',va='top')
    panel = '5,000 genes · 20,000 regions · 256 TFs · 16 CPU cores' if paired else 'Mouse-brain RNA · 3,000 genes · 256 TFs · 16 CPU cores'
    if mobile:
        panel = 'RNA + ATAC · 5,000 genes · 20,000 regions · 16 CPUs' if paired else 'Mouse-brain RNA · 3,000 genes · 256 TFs · 16 CPUs'
    fig.text(.07 if mobile else .055,.895 if mobile else .887,panel,color=muted,fontsize=11 if mobile else 13)
    styles = [('rustscenic',orange,'o','-',3.5 if dark else 2.8),
              ('reference',muted,'s','--',2.1 if dark else 1.8)]
    handles = [Line2D([0],[0],color=color,marker=marker,ls=style,lw=lw,markersize=6,
                     label='RustScenic' if method=='rustscenic' else comparator)
               for method,color,marker,style,lw in styles]
    legend=fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.055,.875 if mobile else .844),
        ncol=2,frameon=False,fontsize=11.2 if mobile else 12,handlelength=2.8,columnspacing=2.2)
    for t in legend.get_texts(): t.set_color(fg)
    positions = [(.13,.52,.82,.265),(.13,.16,.82,.265)] if mobile else [(.075,.25,.385,.47),(.58,.25,.385,.47)]
    metrics = [('input_to_export_s',3600 if paired else 60,'A  Runtime',f'{gain:.1f}× faster','Workflow time (hours)' if paired else 'Workflow time (min)'),
               ('peak_pss_gb',1,'B  Memory',f'{memory:.1f}% lower memory','Peak memory (GB, PSS)')]
    for pos,(metric,divisor,label,benefit,ylabel) in zip(positions,metrics):
        ax=fig.add_axes(pos,facecolor=bg)
        for method,color,marker,style,lw in styles:
            s=rows[rows.method.eq(method)]
            grouped=s.groupby('cells')[metric].median()/divisor
            ax.plot(grouped.index/1e6,grouped.values,color=color,marker=marker,ls=style,lw=lw,
                markersize=6.5 if dark else 5.5,zorder=4 if method=='rustscenic' else 3)
            ax.scatter(s.cells/1e6,s[metric]/divisor,s=31 if dark else 24,facecolor=color,
                edgecolor=bg,linewidth=.65,zorder=6)
            ranges=s.groupby('cells')[metric].agg(['min','max'])/divisor
            ax.vlines(ranges.index/1e6,ranges['min'],ranges['max'],color=color,linewidth=1.2,zorder=5)
            end=grouped.iloc[-1]
            ax.annotate(f'{end:.2f}' if paired and metric=='input_to_export_s' else f'{end:.1f}',
                (grouped.index[-1]/1e6,end),xytext=(12,0),textcoords='offset points',va='center',color=color,
                fontsize=10.8 if mobile and dark else 10.2 if mobile else 12.2 if dark else 11.4,weight='bold' if method=='rustscenic' else 'normal')
        ax.set(xlim=(0,rows.cells.max()/1e6*1.2556),ylim=(0,rows[metric].max()/divisor*1.15),xlabel='Cells (millions)',ylabel=ylabel)
        ticks = [0,.25,.5,.817740] if paired else [0,.5,1,1.306127]
        labels = ['0','0.25','0.5','0.82'] if paired else ['0','0.5','1.0','1.3']
        ax.set_xticks(ticks,labels)
        ax.set_title(label,loc='left',pad=20,color=fg,fontsize=14 if mobile else 15.5,weight='bold')
        ax.text(1,1.065,benefit,transform=ax.transAxes,ha='right',color=teal,weight='bold',fontsize=15 if mobile else 17)
        ax.grid(axis='y',color=grid,lw=.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=fg,labelsize=10.5,length=3.5,pad=6)
        ax.set_xlabel('Cells (millions)',fontsize=12,labelpad=11,color=fg)
        ax.set_ylabel(ylabel,fontsize=12,labelpad=11,color=fg)
        for spine in ax.spines.values():
            spine.set_color(muted);spine.set_linewidth(.8)
    largest = '817,740' if paired else '1.3M'
    if mobile:
        foot = [f'Gains vs {comparator} at {largest}. Three seeds; lines: medians.',
                'Prepared inputs through export · different algorithms · PSS']
    else:
        foot = [f'Gains vs {comparator} at {largest}. Three seeds per size. Points: runs; lines: medians.',
                'Prepared inputs through export; raw preparation excluded. Algorithms and outputs differ. CPU models vary across sizes.']
    for y,line in zip((.075,.045) if mobile else (.125,.080),foot):
        fig.text(.07 if mobile else .075,y,line,color=muted,fontsize=10 if mobile else 11.5)
    stem=('scenicplus' if paired else 'pyscenic')+('-mobile' if mobile else '')+('-dark' if dark else '')
    for ext in ['svg','png']+(['pdf'] if not dark and not mobile else []):
        metadata={'Date':None} if ext=='svg' else ({'CreationDate':None,'ModDate':None} if ext=='pdf' else None)
        fig.savefig(OUT/f'{stem}.{ext}',dpi=180,facecolor=bg,metadata=metadata)
        if ext == 'svg':
            path=OUT/f'{stem}.{ext}'
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close(fig)


if __name__ == '__main__':
    for dataset in ['hdma','atlas']:
        for dark in [False,True]:
            for mobile in [False,True]: draw(dataset,dark,mobile)
    print(OUT)
