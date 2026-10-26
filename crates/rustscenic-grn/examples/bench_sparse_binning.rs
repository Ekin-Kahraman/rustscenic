//! Time TF binning from a fixed CSC input without timing I/O or GBM fitting.
//!
//! Input: little-endian u64 [rows, columns, nnz], then u64 indptr,
//! i32 row indices and f32 data. All columns are selected in input order.
//! Usage: cargo run --release -p rustscenic-grn --example bench_sparse_binning -- input.csc 7

use rustscenic_grn::histogram::BinnedMatrix;
use std::{env, fs, hint::black_box, time::Instant};

fn main() {
    let args: Vec<String> = env::args().collect();
    assert!(
        args.len() >= 2,
        "expected input.csc and optional repeat count"
    );
    let repeats: usize = args.get(2).map_or(7, |s| s.parse().expect("repeat count"));
    assert!(repeats > 0);
    let bytes = fs::read(&args[1]).expect("read CSC input");
    let mut offset = 0;
    let mut take_u64 = || {
        let value = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
        offset += 8;
        usize::try_from(value).expect("integer fits usize")
    };
    let rows = take_u64();
    let columns = take_u64();
    let nnz = take_u64();
    let indptr: Vec<usize> = (0..=columns).map(|_| take_u64()).collect();
    let indices: Vec<i32> = bytes[offset..offset + nnz * 4]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| i32::from_le_bytes(*b))
        .collect();
    offset += nnz * 4;
    let data: Vec<f32> = bytes[offset..offset + nnz * 4]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    assert_eq!(bytes.len(), offset + nnz * 4, "unexpected trailing data");
    let selected: Vec<usize> = (0..columns).collect();
    let mut timings = Vec::new();
    let mut checksum = 0;
    // Warm the worker pool before taking measurements.
    drop(BinnedMatrix::from_csc_columns(
        &indptr, &indices, &data, rows, columns, &selected,
    ));
    for _ in 0..repeats {
        let start = Instant::now();
        let binned =
            BinnedMatrix::from_csc_columns(&indptr, &indices, &data, rows, columns, &selected);
        black_box(&binned);
        timings.push(start.elapsed().as_secs_f64());
        checksum = binned.bins.iter().fold(0xcbf29ce484222325_u64, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
        });
    }
    timings.sort_by(f64::total_cmp);
    println!(
        "rows={rows} columns={columns} nnz={nnz} repeats={repeats} median_seconds={} min_seconds={} max_seconds={} bins_checksum={checksum:016x}",
        timings[repeats / 2], timings[0], timings[repeats - 1],
    );
}
