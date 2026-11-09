"""Check directional recipe outputs and cost on cached real BMMC fits.

One mode/repeat per fresh process. Reuses the existing paired audit input
loader; includes all downstream processing, but not inference or input I/O.
"""
import argparse
import gc
import json
from pathlib import Path
import resource
import sys
import threading
import time

import numpy as np
import pandas as pd
import psutil
from rustscenic import _audited_multiome as workflow, aucell

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "validation/validated_workflow"))
from check_paired import load_case, digest
sys.path.insert(0, str(REPO / "validation/biological_fidelity"))
from polarity import signed_gene_sets


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "saved", "rankings", "output", "baseline"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--mode", choices=["union", "directional"], required=True)
    p.add_argument("--cells", type=int, default=5000)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    case = load_case(args, use_threads=False)
    frames, inputs = case["frames"], args.inputs
    process = psutil.Process()
    gc.collect()
    samples = [process.memory_info().rss]
    stop = threading.Event()
    def monitor():
        while not stop.wait(.005):
            samples.append(process.memory_info().rss)
    worker = threading.Thread(target=monitor)
    worker.start()
    started = time.perf_counter()
    try:
        result = workflow.run(case["rna"], case["atac"], case["tfs"],
            gene_coords=frames[str(inputs / "gene-coordinates.tsv")],
            region_rankings=case["ranks"],
            region_mapping=frames[str(inputs / "motifs/screen-overlap-mapping.tsv")],
            motif_annotations=frames[str(inputs / "motifs/human-gene-annotations.tbl")],
            output_dir=args.output, threads=4, activity_mode=args.mode, cached_stages=case["cached"])
    finally:
        elapsed = time.perf_counter() - started
        samples.append(process.memory_info().rss)
        stop.set()
        worker.join()
    receipt = dict(mode=args.mode, seconds=elapsed, starting_rss_bytes=samples[0],
        peak_sampled_rss_bytes=max(samples), sampling_seconds=.005,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        stages=result["elapsed"], checks={},
        output_matrix_bytes=sum(result[k].to_numpy(copy=False).nbytes for k in ("gene_activity", "region_activity")),
        signature_count=len(result["regulons"]), module_sha256=digest(workflow.__file__),
        input_sha256={str(path): digest(path) for path in
            (args.inputs / "rna.h5", args.inputs / "atac.h5", args.rankings,
             *(args.saved / f"{name}.parquet" for name in ("cell-topic", "topic-peak", "grn")))},
        scope="5000 real BMMC cells; topics and GRN reused; downstream stages recomputed. RSS, not HPC PSS. Validation below is outside timing.")
    old_edges = pd.read_parquet(args.saved / "eregulon-edges.parquet")
    cols = ["tf", "region", "target", "context"]
    pd.testing.assert_frame_equal(result["eregulon_edges"][cols].sort_values(cols).reset_index(drop=True),
        old_edges[cols].sort_values(cols).reset_index(drop=True), check_dtype=False, check_exact=True)
    receipt["checks"]["edges_unchanged"] = True
    for key, feature in (("regulons", "target"), ("region_regulons", "region")):
        if args.mode == "directional":
            edges = old_edges if feature == "target" else old_edges.drop(columns="target").rename(columns={"region": "target"})
            expected = signed_gene_sets(edges, "rustscenic")
        else:
            expected = json.loads((args.saved / ("regulons.json" if key == "regulons" else "region-regulons.json")).read_text())
        assert {k: sorted(v) for k, v in result[key].items()} == {k: sorted(v) for k, v in expected.items()}
        receipt["checks"][key + "_exact"] = True
    for key in ("gene_activity", "region_activity"):
        value = result[key]
        assert np.isfinite(value.to_numpy()).all() and value.dtypes.eq(np.float32).all()
        if args.mode == "union":
            expected = pd.read_parquet(args.baseline / f"{key}.parquet")
            pd.testing.assert_frame_equal(value.sort_index(axis=1), expected.sort_index(axis=1), check_exact=True)
            receipt["checks"][key + "_historical_exact"] = True
        elif key == "gene_activity":
            expected = aucell.score(case["rna"], result["regulons"], seed=20261106, top_frac=.05)
            pd.testing.assert_frame_equal(value.sort_index(axis=1), expected.sort_index(axis=1), check_exact=True)
            receipt["checks"]["directional_scores_exact"] = True
        value.to_parquet(args.output / f"{key}.parquet")
    receipt["checks"]["metadata"] = result["metadata"]["activity_mode"] == args.mode
    assert all(receipt["checks"].values())
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
