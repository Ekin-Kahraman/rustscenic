"""Published native analysis stages. Algorithm differences are retained and recorded."""
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from common import read_json
from rankings import load_projection, native_annotations, ProjectedRankingDatabase


class Clock:
    def __init__(self):
        self.stages = {}
        self.fit_summary = {}

    def run(self, name, fn):
        print('STAGE', name, flush=True)
        start = time.perf_counter()
        result = fn()
        self.stages[name] = time.perf_counter() - start
        return result


def ann(x, features, cells):
    return ad.AnnData(x, obs=pd.DataFrame(index=cells), var=pd.DataFrame(index=features))


def infer_rna(method, rna, tfs, settings, seed, threads, clock):
    if method == 'rustscenic':
        import rustscenic.grn
        grn = clock.run('gene_network', lambda: rustscenic.grn.infer(rna, tf_names=tfs,
            **settings, early_stop_mode='arboreto', seed=seed))
        fit = grn.attrs.get('grn_fit', {})
        keep = ['target_count', 'fitted_trees_min', 'fitted_trees_max', 'fitted_trees_mean',
                'fitted_trees_median', 'n_estimators_ceiling', 'early_stop_mode', 'early_stop_window', 'subsample']
        clock.fit_summary = {key: value.item() if isinstance(value, np.generic) else value for key, value in fit.items() if key in keep}
        clock.fit_summary['metadata_source'] = 'RustScenic native grn_fit attributes'
    else:
        from grn_reference import infer_multiprocessing
        result = clock.run('gene_network', lambda: infer_multiprocessing(rna.X, rna.var_names.tolist(), tfs,
                            settings, seed, threads))
        grn = result[0]
        clock.fit_summary = result[2]
    return grn


def canonical_name(name):
    name = str(name)
    if name.endswith('(+)'):
        return name[:-3] + '_activator'
    if name.endswith('(-)'):
        return name[:-3] + '_repressor'
    return name


def activity(method, rna, regulons, seed, threads):
    if not regulons:
        return pd.DataFrame(index=rna.obs_names)
    if method == 'rustscenic':
        from bounded import aucell
        return aucell(rna, regulons, top_frac=.05)
    from pyscenic.aucell import aucell
    from ctxcore.genesig import GeneSignature
    from process_guard import guard_aucell_workers
    guard_aucell_workers()
    signatures = [GeneSignature(name, genes) for name, genes in regulons.items()]
    result = aucell(rna.to_df(), signatures, auc_threshold=.05, noweights=True,
                   normalize=False, seed=seed, num_workers=threads)
    if set(result.index) != set(rna.obs_names):
        raise ValueError('Reference AUCell changed the cell identities')
    return result.reindex(index=rna.obs_names)


def rna_workflow(method, rna, tfs, motifs, config, seed, clock):
    threads = config['threads']
    grn = infer_rna(method, rna, tfs, config['grn'], seed, threads, clock)
    if method == 'rustscenic':
        import rustscenic.grn
        import rustscenic.cistarget
        grn = clock.run('gene_correlation', lambda: rustscenic.grn.add_correlation(grn, rna, rho_threshold=.03))
        candidates = clock.run('candidate_modules', lambda: rustscenic.grn.build_regulons(
            grn, top_targets_per_tf=50, min_targets=10, include_repressors=True))
        rankings, universe = clock.run('motif_database_load', lambda: load_projection(motifs / 'mouse-gene.feather'))
        annotation = native_annotations(motifs / 'mouse-gene-annotations.tbl')
        enriched = clock.run('motif_enrichment', lambda: rustscenic.cistarget.enrich(rankings, candidates,
            top_frac=.05, auc_threshold=0., nes_threshold=3., rank_universe_size=universe))
        regulons = clock.run('motif_pruning', lambda: rustscenic.cistarget.prune_regulons(enriched, candidates,
            annotation, rankings=rankings, top_frac=.05, auc_threshold=0., nes_threshold=3.,
            min_genes=5, rank_universe_size=universe))
        motif_rows = enriched
    else:
        from pyscenic.utils import modules_from_adjacencies
        from pyscenic.prune import prune2df
        from pyscenic.transform import df2regulons
        expr = clock.run('dense_expression', rna.to_df)
        candidates = clock.run('candidate_modules_and_correlation', lambda: list(modules_from_adjacencies(grn, expr,
            thresholds=(), top_n_targets=(50,), top_n_regulators=(), min_genes=10,
            keep_only_activating=False, rho_dichotomize=True, rho_threshold=.03, rho_mask_dropouts=False)))
        database = clock.run('motif_database_load', lambda: ProjectedRankingDatabase(motifs / 'mouse-gene.feather', 'mouse-v10'))
        enriched = clock.run('motif_enrichment_and_pruning', lambda: prune2df([database], candidates,
            str(motifs / 'mouse-gene-annotations.tbl'), rank_threshold=1500, auc_threshold=.05,
            nes_threshold=3., weighted_recovery=False, client_or_address='custom_multiprocessing',
            num_workers=threads, filter_for_annotation=True))
        refs = df2regulons(enriched) if not enriched.empty else []
        regulons = {canonical_name(r.name): sorted(r.genes) for r in refs if len(r.genes) >= 5}
        motif_rows = pd.DataFrame([{'regulon': str(index[0]), 'motif': str(index[1]),
            'auc': float(row[('Enrichment', 'AUC')]), 'nes': float(row[('Enrichment', 'NES')]),
            'target_genes': json.dumps(row[('Enrichment', 'TargetGenes')])} for index, row in enriched.iterrows()])
    regulons = {str(k): sorted(set(v)) for k, v in regulons.items()}
    gene_activity = clock.run('cell_activity', lambda: activity(method, rna, regulons, seed, threads))
    return {'grn': grn, 'regulons': regulons, 'gene_activity': gene_activity,
            'motif_evidence': motif_rows, 'n_candidates': len(candidates),
            'scope': 'RNA matrix -> native signed top-50 modules -> motif refinement -> unweighted AUCell. GRN stopping/recipe/resources shared; pruning algorithms differ.'}


def topic_fit(method, atac_counts, config, seed, out, mallet):
    binary = atac_counts.X.copy().astype(np.float32)
    binary.data[:] = 1
    active = np.asarray(binary.sum(axis=0)).ravel() > 0
    binary = binary[:, active]
    feature_names = atac_counts.var_names[active]
    k, iterations = config['topics']['n_topics'], config['topics']['iterations']
    if method == 'rustscenic':
        import rustscenic.topics
        fitted = rustscenic.topics.fit_gibbs(ann(binary, feature_names, atac_counts.obs_names),
            n_topics=k, alpha=50. / k, eta=.1, n_iters=iterations, seed=seed,
            n_threads=config['threads'], verbose=False)
        cell_topic, topic_peak = fitted.cell_topic, fitted.topic_peak
        cell_topic.index = atac_counts.obs_names
        if list(topic_peak.columns) != list(feature_names):
            topic_peak = topic_peak.T
        def impute():
            # Same posterior-product/1e6/int32 operation as pycisTopic; BLAS executes the product.
            values = np.empty((atac_counts.n_obs, len(feature_names)), dtype=np.int32)
            left = cell_topic.to_numpy(dtype=np.float32)
            right = topic_peak.to_numpy(dtype=np.float32)
            for start in range(0, len(feature_names), 2000):
                values[:, start:start+2000] = (left @ right[:, start:start+2000] * np.float32(1000000)).astype(np.int32)
            keep = np.any(values != 0, axis=0)
            return ann(values[:, keep], feature_names[keep], atac_counts.obs_names)
    else:
        from pycisTopic.cistopic_class import create_cistopic_object
        from pycisTopic.lda_models import run_cgs_models_mallet
        cistopic = create_cistopic_object(binary.T.tocsr(), cell_names=atac_counts.obs_names.tolist(),
            region_names=feature_names.tolist(), min_frag=1, min_cell=1, tag_cells=False)
        temp = out / 'mallet'
        temp.mkdir(exist_ok=True)
        model = run_cgs_models_mallet(cistopic, n_topics=[k], n_cpu=config['threads'], n_iter=iterations,
            random_state=seed, alpha=50., alpha_by_topic=True, eta=.1, eta_by_topic=False,
            tmp_path=str(temp), mallet_path=str(mallet), reuse_corpus=False)[0]
        cell_topic = model.cell_topic.T.reindex(atac_counts.obs_names)
        topic_peak = model.topic_region.T.reindex(columns=feature_names)
        cistopic.selected_model = model
        def impute():
            from pycisTopic.diff_features import impute_accessibility
            result = impute_accessibility(cistopic, scale_factor=1000000, chunk_size=2000)
            return ann(result.mtx.T, result.feature_names, result.cell_names)
    assert cell_topic.shape == (atac_counts.n_obs, k)
    assert topic_peak.shape == (k, len(feature_names))
    assert np.isfinite(cell_topic.to_numpy()).all() and np.isfinite(topic_peak.to_numpy()).all()
    cell_topic.columns = [f'Topic{i+1}' for i in range(k)]
    topic_peak.index = cell_topic.columns
    # Fixed recipe, no labels: top 1,000 accessible regions per fitted topic.
    sets = {name: sorted(topic_peak.loc[name].nlargest(min(1000, len(feature_names))).index.tolist())
            for name in topic_peak.index}
    return cell_topic, topic_peak, sets, impute


def region_motif_support(method, region_sets, tfs, motifs, clock):
    ranks, universe = clock.run('region_database_load', lambda: load_projection(motifs / 'screen-projected.feather'))
    mapping = pd.read_csv(motifs / 'screen-overlap-mapping.tsv', sep='\t')
    annotations = pd.read_csv(motifs / 'human-gene-annotations.tbl', sep='\t')
    annotations = annotations.loc[(annotations.description == 'gene is directly annotated') |
        (annotations.description.str.contains('orthologous') & ~annotations.description.str.contains('similar'))]
    annotation_map = {m: sorted(set(g.gene_name) & set(tfs)) for m, g in annotations.groupby('#motif_id')}
    evidence, all_scores = [], []
    if method == 'rustscenic':
        import rustscenic.cistarget
        query_sets = {name: sorted(set(mapping.loc[mapping.Target.isin(peaks), 'Query']))
                      for name, peaks in region_sets.items()}
        query_sets = {k: v for k, v in query_sets.items() if v}
        enriched = rustscenic.cistarget.enrich(ranks, query_sets, top_frac=.005,
                    auc_threshold=0., nes_threshold=3., rank_universe_size=universe)
        for row in enriched.itertuples():
            all_scores.append({'region_set': row.regulon, 'motif': row.motif, 'auc': row.auc, 'nes': row.nes})
            factors = annotation_map.get(row.motif, [])
            if not factors:
                continue
            queries = query_sets[row.regulon]
            # Native Rust pruning retains motif-supported regions within its AUC cutoff.
            supported = set(ranks.loc[row.motif, queries].index[ranks.loc[row.motif, queries] < round(.005 * universe)])
            peaks = sorted(set(mapping.loc[mapping.Query.isin(supported) & mapping.Target.isin(region_sets[row.regulon]), 'Target']))
            for tf in factors:
                evidence.extend({'TF': tf, 'peak_id': peak, 'motif': row.motif, 'auc': float(row.auc), 'nes': float(row.nes)} for peak in peaks)
    else:
        import pyranges as pr
        from pycistarget.motif_enrichment_cistarget import cisTarget, cisTargetDatabase
        database = cisTargetDatabase.__new__(cisTargetDatabase)
        database.db_rankings = ranks
        database.total_regions = universe
        database.regions_to_db = {name: mapping.loc[mapping.Target.isin(peaks)].copy() for name, peaks in region_sets.items()}
        for name, peaks in region_sets.items():
            if database.regions_to_db[name].empty:
                continue
            records = []
            for peak in peaks:
                chrom, tail = peak.split(':'); start, end = map(int, tail.split('-'))
                records.append((chrom, start, end))
            regions = pr.PyRanges(pd.DataFrame(records, columns=['Chromosome', 'Start', 'End']))
            analysis = cisTarget(regions, name=name, species='homo_sapiens', auc_threshold=.005,
                nes_threshold=3., rank_threshold=.05, path_to_motif_annotations=str(motifs / 'human-gene-annotations.tbl'),
                annotation_version='v10nr_clust', annotation_to_use=['Direct_annot', 'Orthology_annot'])
            analysis.run_ctx(database)
            for motif, row in analysis.motif_enrichment.iterrows():
                all_scores.append({'region_set': name, 'motif': motif, 'auc': float(row.AUC), 'nes': float(row.NES)})
                factors = annotation_map.get(motif, [])
                for tf in factors:
                    evidence.extend({'TF': tf, 'peak_id': peak, 'motif': motif, 'auc': float(row.AUC), 'nes': float(row.NES)}
                                    for peak in analysis.motif_hits['Region_set'].get(motif, []))
    return (pd.DataFrame(evidence, columns=['TF', 'peak_id', 'motif', 'auc', 'nes']).drop_duplicates(),
            pd.DataFrame(all_scores, columns=['region_set', 'motif', 'auc', 'nes']))


def search_space(peaks, coords, max_distance=150000):
    rows = []
    for gene in coords.itertuples():
        same = peaks.loc[peaks.chrom == gene.chrom]
        distance = ((same.start + same.end) // 2).to_numpy() - int(gene.tss)
        rows.extend({'Name': peak, 'Gene': gene.gene, 'Distance': int(dist)}
                    for peak, dist in zip(same.peak, distance) if abs(dist) <= max_distance)
    return pd.DataFrame(rows)


def multiome_workflow(method, rna, atac_counts, atac_log, tfs, inputs, motifs, config, seed, out, mallet, clock):
    threads = config['threads']
    cell_topic, topic_peak, region_sets, impute = clock.run('topic_modelling', lambda: topic_fit(method, atac_counts, config, seed, out, mallet))
    atac_log = clock.run('accessibility_imputation', impute)
    support, motif_scores = clock.run('region_motif_enrichment', lambda: region_motif_support(method, region_sets, tfs, motifs, clock))
    coords = pd.read_csv(inputs / 'gene-coordinates.tsv', sep='\t')
    peaks = pd.read_csv(inputs / 'peaks.tsv', sep='\t')
    peaks = peaks.loc[peaks.peak.isin(atac_log.var_names)]
    support = support.loc[support.peak_id.isin(atac_log.var_names)]
    import rustscenic.grn
    if method == 'rustscenic':
        import rustscenic.enhancer
        import rustscenic.eregulon
        grn = infer_rna(method, rna, tfs, config['grn'], seed, threads, clock)
        grn = clock.run('gene_correlation', lambda: rustscenic.grn.add_correlation(grn, rna, rho_threshold=.03))
        from bounded import link_peaks
        links = clock.run('region_to_gene', lambda: link_peaks(rna, atac_log, coords,
                    max_distance=150000, min_abs_corr=.03))
        motifs_native = support.rename(columns={'TF': 'regulon'})[['regulon', 'peak_id', 'motif', 'auc']]
        motifs_native = pd.concat([motifs_native.assign(regulon=motifs_native.regulon + '_activator'),
                                   motifs_native.assign(regulon=motifs_native.regulon + '_repressor')], ignore_index=True)
        eregs = clock.run('regulon_assembly', lambda: rustscenic.eregulon.build_eregulons(grn, motifs_native, links,
                    min_target_genes=5, min_enhancer_links=1, cistarget_auc_threshold=0.))
        edges = pd.DataFrame([{'tf': er.tf.removesuffix('_activator').removesuffix('_repressor'),
            'region': str(peak), 'target': str(target), 'context': er.tf + '; positive region-gene links'}
            for er in eregs for target, target_peaks in er.target_to_peaks.items() for peak in target_peaks],
            columns=['tf', 'region', 'target', 'context'])
        links = links.rename(columns={'peak_id': 'region', 'gene': 'target', 'correlation': 'rho'})
        gsea_stats = pd.DataFrame()
        n_default = len(eregs)
    else:
        from scenicplus.TF_to_gene import calculate_TFs_to_genes_relationships
        from scenicplus.enhancer_to_gene import calculate_regions_to_genes_relationships
        from scenicplus.grn_builder.gsea_approach import build_grn
        from scenicplus.grn_builder.modules import merge_emodules
        expr, accessibility = clock.run('dense_expression', lambda: (rna.to_df(), atac_log.to_df()))
        grn = clock.run('gene_network', lambda: calculate_TFs_to_genes_relationships(expr, tfs,
                    temp_dir=out, method='GBM', n_cpu=threads, seed=seed))
        clock.fit_summary = {'available': False, 'metadata_source': 'Published SCENIC+ API does not expose per-target fitted tree counts.'}
        space = clock.run('search_space', lambda: search_space(peaks, coords))
        links = clock.run('region_to_gene', lambda: calculate_regions_to_genes_relationships(expr, accessibility,
                    space, temp_dir=out, importance_scoring_method='GBM', correlation_scoring_method='PR', n_cpu=threads))
        cistromes = pd.DataFrame(False, index=atac_log.var_names, columns=tfs)
        for row in support[['TF', 'peak_id']].drop_duplicates().itertuples(index=False):
            cistromes.at[row.peak_id, row.TF] = True
        motif_data = ann(cistromes.to_numpy(dtype=bool), tfs, atac_log.var_names)
        unmerged = clock.run('regulon_assembly', lambda: build_grn(grn, links, motif_data, is_extended=False,
            temp_dir=str(out), gsea_n_perm=1000, quantiles=(.85, .90), top_n_regionTogenes_per_gene=(5, 10, 15),
            top_n_regionTogenes_per_region=(), rho_dichotomize_tf2g=True, rho_dichotomize_r2g=True,
            rho_dichotomize_eregulon=True, rho_threshold=.03, NES_thr=0., adj_pval_thr=1.,
            min_target_genes=5, n_cpu=threads, merge_eRegulons=False, disable_tqdm=True, seed=seed))
        gsea_stats = pd.DataFrame([{'TF': str(er.transcription_factor), 'context': str(sorted(map(str, er.context))),
            'NES': float(er.gsea_enrichment_score), 'pvalue': float(er.gsea_pval), 'adjusted_pvalue': float(er.gsea_adj_pval),
            'target_genes': int(er.n_target_genes), 'target_regions': int(er.n_target_regions)} for er in unmerged])
        n_default = len(unmerged)
        strict = [er for er in unmerged if er.gsea_adj_pval < .05]
        eregs = merge_emodules(e_modules=strict, inplace=False, rho_dichotomize=True) if strict else []
        eregs = [er for er in eregs if not isinstance(er, list)]
        edges = pd.DataFrame([{'tf': str(er.transcription_factor), 'region': str(link.region),
            'target': str(link.target), 'context': str(sorted(map(str, er.context)))}
            for er in eregs for link in er.regions2genes], columns=['tf', 'region', 'target', 'context'])
    gene_sets = {tf: sorted(set(group.target)) for tf, group in edges.groupby('tf')}
    region_gene_sets = {tf: sorted(set(group.region)) for tf, group in edges.groupby('tf')}
    if method == 'rustscenic':
        gene_activity = clock.run('gene_activity', lambda: activity(method, rna, gene_sets, seed, threads))
        region_activity = clock.run('region_activity', lambda: activity(method, atac_log, region_gene_sets, seed, threads))
    else:
        from scenicplus.eregulon_enrichment import score_eRegulons
        from process_guard import guard_aucell_workers
        guard_aucell_workers()
        if edges.empty:
            gene_activity, region_activity = pd.DataFrame(index=rna.obs_names), pd.DataFrame(index=rna.obs_names)
        else:
            meta = edges.rename(columns={'tf': 'TF', 'region': 'Region', 'target': 'Gene'})[['TF', 'Region', 'Gene']].drop_duplicates()
            meta['Region_signature_name'] = meta.TF + '__regions'
            meta['Gene_signature_name'] = meta.TF + '__genes'
            scored = clock.run('gene_and_region_activity', lambda: score_eRegulons(meta, expr, accessibility,
                    auc_threshold=.05, normalize=False, n_cpu=threads))
            gene_activity, region_activity = scored['Gene_based'], scored['Region_based']
            gene_activity = gene_activity.reindex(index=rna.obs_names)
            region_activity = region_activity.reindex(index=rna.obs_names)
            gene_activity.columns = [str(c).removesuffix('__genes') for c in gene_activity.columns]
            region_activity.columns = [str(c).removesuffix('__regions') for c in region_activity.columns]
    return {'grn': grn, 'regulons': gene_sets, 'region_regulons': region_gene_sets,
        'gene_activity': gene_activity, 'region_activity': region_activity, 'eregulon_edges': edges,
        'region_to_gene': links, 'cell_topic': cell_topic, 'topic_peak': topic_peak,
        'motif_evidence': support, 'region_motif_scores': motif_scores, 'gsea_statistics': gsea_stats,
        'n_default_modules': n_default, 'n_eregulons': len(eregs),
        'scope': 'Paired matrices -> native topics -> real cisTarget motif enrichment -> native regulatory networks -> TF-union activity. Includes topic fitting; excludes raw reads/peak calling and database construction. Native linking, pruning and assembly algorithms differ; SCENIC+ outputs use adjusted p<0.05.'}
