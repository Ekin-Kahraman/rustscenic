"""Explore a precomputed endothelial-cell regulon. Python standard library only."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tf", default="ETS1", help="Transcription factor symbol; default: ETS1")
    parser.add_argument("--list", action="store_true", help="List available motif-supported regulons")
    parser.add_argument("--verify", action="store_true", help="Verify all delivered files before reading")
    args = parser.parse_args()
    if args.verify:
        hashes = json.loads((ROOT / "checksums.json").read_text())
        for name, expected in hashes.items():
            actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            if actual != expected:
                parser.exit(1, f"Checksum mismatch: {name}\n")
        print(f"Verified {len(hashes)} files.")
    data = json.loads((ROOT / "preview.json").read_text())
    if args.list:
        print("\n".join(data["regulons"]))
        return
    matches = [r for r in data["regulons"].values() if r["tf"] == args.tf.upper()]
    if not matches:
        parser.error(f"No motif-supported regulon for {args.tf!r}; use --list to see available choices.")
    print("Endothelial cell (CL:0000115) | 5,000 sampled cells | 234 source datasets")
    for row in matches:
        print(f"\n{row['tf']}: {row['target_count']} targets in the {row['polarity']} regulon")
        print("Top 10 targets by model importance within this motif-supported gene set:")
        print(f"{'Target':<14}{'Importance':>12}{'Correlation':>14}")
        for edge in row["targets"][:10]:
            print(f"{edge['target']:<14}{edge['importance']:>12.4f}{edge['rho']:>14.4f}")
        print(f"Mean cohort AUCell: {row['activity']['mean']:.4f}")
    print("\nExploratory predictions. Motif support applies to the gene set, not independent proof of each edge.")
    print("Importance is not a probability; correlation is not causation.")
    print("Full original network: data/artifacts/grn_signed.parquet")
    print("Study attribution: source-citations.json")

if __name__ == "__main__":
    main()
