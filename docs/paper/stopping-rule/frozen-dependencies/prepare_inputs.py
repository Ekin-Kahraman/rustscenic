"""Rebuild the packaged real inputs from two checksum-pinned BEELINE archives."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from scipy import sparse

SOURCES = {
    "BEELINE-data.zip": "3e67582fb8a4ba5c2189f6a0bfb14abb1d61f21921459bdeb4a2a9f2b652d816",
    "BEELINE-Networks.zip": "d54fc6ed6529141e85f94803d2c730a8173a3e867026227528ff1ae832765607",
}
SPECS = {
    "mESC": ("mouse", ["mESC-ChIP-seq-network.csv", "mESC-lofgof-network.csv"]),
    "hESC": ("human", ["hESC-ChIP-seq-network.csv"]),
    "mDC": ("mouse", ["mDC-ChIP-seq-network.csv"]),
}


def sha(path):
    return hashlib.file_digest(Path(path).open("rb"), "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for name, expected in SOURCES.items():
        if sha(args.cache / name) != expected:
            raise ValueError(f"Archive checksum failed: {name}")
    args.out.mkdir(parents=True, exist_ok=True)
    provenance = {
        "source_record": "https://zenodo.org/records/19009603",
        "source_doi": "10.5281/zenodo.19009603",
        "source_license": "CC-BY-NC-4.0",
        "paper": "https://doi.org/10.1038/s41592-019-0690-6",
        "archive_sha256": SOURCES, "datasets": {}, "archive_members": {},
        "selection": "Bonferroni VGAM p < 0.01 / tested genes; all eligible TFs plus up to 500 eligible non-TFs by source variance; ties by gene symbol. No reference links used in selection.",
        "expression": "Source preprocessed expression, unchanged apart from cell-by-gene transpose and float32 conversion. No second normalisation or log transform.",
        "identifiers": "Uppercase to match the source network/TF naming convention; ambiguous case-colliding expression names are excluded, with no aliases or orthologue mapping.",
    }
    with zipfile.ZipFile(args.cache / "BEELINE-data.zip") as data_zip, zipfile.ZipFile(args.cache / "BEELINE-Networks.zip") as net_zip:
        def read(z, member, **kwargs):
            raw = z.read(member)
            provenance["archive_members"][member] = hashlib.sha256(raw).hexdigest()
            return pd.read_csv(io.BytesIO(raw), **kwargs)
        for name, (species, refs) in SPECS.items():
            base = f"BEELINE-data/inputs/scRNA-Seq/{name}"
            expression = read(data_zip, f"{base}/ExpressionData.csv", index_col=0)
            order = read(data_zip, f"{base}/GeneOrdering.csv", index_col=0)
            tf_file = f"{species}-tfs.csv"
            tf_set = set(read(net_zip, tf_file).TF.astype(str).str.upper())
            original = expression.index.astype(str)
            uppercase = original.str.upper()
            ambiguous = set(uppercase[uppercase.duplicated(keep=False)])
            valid_names = ~uppercase.isin(ambiguous)
            expression = expression.loc[valid_names].copy()
            expression.index = uppercase[valid_names]
            order.index = order.index.astype(str).str.upper()
            order = order.loc[~order.index.isin(ambiguous)].copy()
            if order.index.duplicated().any():
                raise ValueError(f"Ambiguous ordering names: {name}")
            order = order.loc[order.index.isin(expression.index)].copy()
            order["symbol"] = order.index
            order = order.loc[(order.VGAMpValue < 0.01 / len(order)) & (order.Variance > 0)]
            order = order.sort_values(["Variance", "symbol"], ascending=[False, True], kind="stable")
            tfs = sorted(set(order.index) & tf_set)
            non_tfs = [g for g in order.index if g not in tf_set][:500]
            genes = sorted(tfs + non_tfs)
            if len(tfs) < 5 or len(non_tfs) < 100:
                raise ValueError(f"Insufficient eligible input: {name}")
            matrix = expression.loc[genes].T.to_numpy(dtype=np.float32)
            if not np.isfinite(matrix).all() or np.any(matrix < 0):
                raise ValueError(f"Invalid expression: {name}")
            if np.any(np.ptp(matrix, axis=0) == 0):
                raise ValueError(f"Selected constant gene: {name}")
            dest = args.out / name
            dest.mkdir(parents=True, exist_ok=True)
            sparse.save_npz(dest / "matrix.npz", sparse.csr_matrix(matrix))
            pd.DataFrame({"source_symbol": [str(x) for x in original if str(x).upper() in genes], "symbol": [str(x).upper() for x in original if str(x).upper() in genes]}).sort_values("symbol").to_csv(dest / "gene-symbol-map.csv", index=False)
            order.loc[genes].drop(columns="symbol").to_csv(dest / "selection.csv")
            ref_infos = []
            for ref in refs:
                member = f"Networks/{species}/{ref}"
                frame = read(net_zip, member).iloc[:, :2].copy()
                frame.columns = ["TF", "target"]
                frame = frame.astype(str).apply(lambda col: col.str.upper())
                frame = frame.loc[(frame.TF != frame.target)].drop_duplicates()
                original_n = len(frame)
                frame = frame.loc[frame.TF.isin(tfs) & frame.target.isin(genes)].sort_values(["TF", "target"])
                if len(frame) < 20 or frame.TF.nunique() < 2:
                    raise ValueError(f"Insufficient eligible reference: {name}, {ref}")
                output = ref.replace(".csv", ".tsv")
                frame.to_csv(dest / output, sep="\t", index=False)
                ref_infos.append({"file": output, "source_member": member, "source_nonself_links": original_n, "eligible_links": len(frame), "supported_tfs": frame.TF.nunique(), "reference_type": "perturbation" if "lofgof" in ref else "ChIP-seq"})
            info = {"name": name, "species": species, "shape": list(matrix.shape), "genes": genes, "tfs": tfs,
                    "references": ref_infos, "ambiguous_uppercase_names_excluded": sorted(ambiguous),
                    "files": {f.name: sha(f) for f in sorted(dest.iterdir()) if f.name != "input.json"}}
            (dest / "input.json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
            provenance["datasets"][name] = {k: v for k, v in info.items() if k not in ["genes", "tfs", "files"]}
            print(name, matrix.shape, len(tfs), "TFs", [(r['eligible_links'], r['supported_tfs']) for r in ref_infos], flush=True)
    (args.out / "SOURCE-PROVENANCE.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
