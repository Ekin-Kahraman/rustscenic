"""Projection adapters retain genuine ranks, all motifs and original universe size."""
from pathlib import Path
import json
import pandas as pd
import pyarrow.feather as feather
from ctxcore.rnkdb import RankingDatabase


def load_projection(path):
    path = Path(path)
    frame = feather.read_feather(path).set_index('motifs')
    meta = json.loads(path.with_suffix('.json').read_text())
    return frame, int(meta['original_rank_universe'])


class ProjectedRankingDatabase(RankingDatabase):
    def __init__(self, path, name):
        super().__init__(name)
        self.frame, self.universe = load_projection(path)
        self.frame.columns.name = 'genes'

    @property
    def total_genes(self):
        return self.universe

    @property
    def genes(self):
        return tuple(self.frame.columns)

    def load_full(self):
        return self.frame

    def load(self, genesig):
        return self.frame.loc[:, sorted(set(genesig.genes) & set(self.frame.columns))]


def native_annotations(path):
    frame = pd.read_csv(path, sep='\t')
    keep = (frame.motif_similarity_qvalue <= .001) & (frame.orthologous_identity >= 0)
    return frame.loc[keep, ['#motif_id', 'gene_name']].rename(columns={'#motif_id': 'motif', 'gene_name': 'TF'}).drop_duplicates()
