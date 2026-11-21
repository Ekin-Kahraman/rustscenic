"""Post hoc: tabulate the queue.sh fits (time, peak RSS, trees, edges) into queue-summary.csv."""
import json
from pathlib import Path
import pandas as pd

D = Path(__file__).resolve().parent
rss = pd.read_csv(D / 'queue-fits.tsv', sep='\t').set_index('name').max_rss_bytes
rows = []
for name, value in rss.items():
    info = D / 'native-bins' / name / 'info.json'
    if not info.exists():
        continue
    i = json.loads(info.read_text())
    rows.append(dict(name=name, dataset=i.get('dataset', 'bmmc'), max_bins=int(i['max_bins']), mode=i['mode'], seed=i['seed'],
                     inference_s=round(i['inference_s'], 1), peak_rss_gb=round(value / 1e9, 2),
                     trees_mean=round(i['trees_mean'], 2), edges=i['edges']))
table = pd.DataFrame(rows)
table.to_csv(D / 'queue-summary.csv', index=False)
print(table.to_string(index=False))
