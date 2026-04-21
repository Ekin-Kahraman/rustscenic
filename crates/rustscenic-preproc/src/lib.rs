//! rustscenic-preproc — scATAC fragment preprocessing.
//!
//! Absorbs the pycisTopic preprocessing surface so the SCENIC+ install
//! story collapses to one `pip install rustscenic`.
//!
//! Scope:
//! - Fragment I/O: parse 10x cellranger `fragments.tsv.gz` to an in-memory
//!   fragment table.
//! - Cell QC: fragments-per-barcode, insert-size distribution, (eventually)
//!   TSS enrichment, FRiP.
//! - Matrix build: intersect fragments against a peak BED and produce a
//!   cells × peaks sparse matrix.
//!
//! See `docs/atac-preprocessing-scope.md` for the full scope + validation
//! plan vs pycisTopic.

pub mod fragments;

pub use fragments::{read_fragments, Fragment, FragmentTable};
