# Barcode Sets

This directory contains iBAR sequence pools used for CROPseq-multi library design.
Two categories of files are provided: **designed sets** (root directory) and **subsets** (`subsets/`).

---

## Designed Sets

Files matching `barcodes_n*_k*_<metric>.noBsmBI.csv`

These are independently designed barcode pools, each optimized directly for a specific
combination of barcode length (`n`) and minimum pairwise edit distance (`k`). The
`noBsmBI` suffix indicates that any sequence containing a BsmBI recognition site has
been removed.

**File naming:** `barcodes_n{length}_k{distance}_{metric}.noBsmBI.csv`

| Column | Description |
|---|---|
| `barcode` | Full barcode sequence (length `n`) |
| `gc` | GC fraction |
| `n` | Barcode length (nt) |
| `k` | Minimum pairwise edit distance |
| `homopolymer` | Maximum homopolymer run length |
| `gc_min` / `gc_max` | GC fraction bounds applied during design |

**Distance metrics:**

- `Levenshtein` — edit distance accounting for substitutions, insertions, and deletions.
  Recommended for sequencing applications where indel errors are expected.
- `Hamming` — substitution-only distance. Provided for reference; suitable only when
  barcode length is fixed and indel errors are not a concern.

**Available designed sets:**

| File | Length | Min distance | Metric | Barcodes |
|---|---|---|---|---|
| `barcodes_n6_k2_Levenshtein.noBsmBI.csv` | 6 | 2 | Levenshtein | 883 |
| `barcodes_n6_k3_Levenshtein.noBsmBI.csv` | 6 | 3 | Levenshtein | 87 |
| `barcodes_n7_k2_Levenshtein.noBsmBI.csv` | 7 | 2 | Levenshtein | 1,991 |
| `barcodes_n7_k3_Levenshtein.noBsmBI.csv` | 7 | 3 | Levenshtein | 230 |
| `barcodes_n8_k2_Levenshtein.noBsmBI.csv` | 8 | 2 | Levenshtein | 12,606 |
| `barcodes_n8_k3_Levenshtein.noBsmBI.csv` | 8 | 3 | Levenshtein | 835 |
| `barcodes_n9_k2_Levenshtein.noBsmBI.csv` | 9 | 2 | Levenshtein | 42,534 |
| `barcodes_n9_k3_Levenshtein.noBsmBI.csv` | 9 | 3 | Levenshtein | 3,054 |
| `barcodes_n10_k2_Levenshtein.noBsmBI.csv` | 10 | 2 | Levenshtein | 178,183 |
| `barcodes_n10_k3_Levenshtein.noBsmBI.csv` | 10 | 3 | Levenshtein | 10,644 |
| `barcodes_n12_k3_Levenshtein.noBsmBI.csv` | 12 | 3 | Levenshtein | 62,679 |
| `barcodes_n12_k3_Hamming.noBsmBI.csv` | 12 | 3 | Hamming | 78,784 |

---

## Subsets (`subsets/`)

Files matching `subsets/barcodes_subset_n*_k*_Levenshtein.csv`

These are derived from the `barcodes_n12_k3_Levenshtein.noBsmBI.csv` parent pool.
Each subset selects the largest collection of barcodes from that parent whose
**first `n` nucleotides (the prefix)** satisfy the minimum pairwise Levenshtein
distance `k`. All barcodes retain the full 12-nt sequence from the parent, ensuring
compatibility with synthesis and sequencing protocols designed for 12-nt iBARs.

This approach has two practical advantages over the independently designed sets:

1. **Consistency** — all subsets draw from a single parent pool. Barcodes selected for
   a short-prefix experiment are valid full-length 12-nt sequences that can be used
   directly if sequencing depth or read length increases.

2. **Cross-library exclusion** — because all subsets share a common sequence space,
   previously used barcodes can be reliably excluded when designing a second library,
   preventing barcode collision across experiments.

**File naming:** `barcodes_subset_n{prefix_length}_k{distance}_Levenshtein.csv`

| Column | Description |
|---|---|
| `barcode` | Full 12-nt barcode sequence (from parent pool) |
| `prefix` | First `n` nucleotides evaluated for distance |
| `cycles` | Prefix length (`n`) |
| `min_levenshtein` | Minimum pairwise Levenshtein distance guaranteed over the prefix |

**Available subsets:**

| File | Prefix length | Min distance | Barcodes |
|---|---|---|---|
| `barcodes_subset_n6_k2_Levenshtein.csv` | 6 | 2 | 721 |
| `barcodes_subset_n6_k3_Levenshtein.csv` | 6 | 3 | 99 |
| `barcodes_subset_n7_k2_Levenshtein.csv` | 7 | 2 | 2,796 |
| `barcodes_subset_n7_k3_Levenshtein.csv` | 7 | 3 | 308 |
| `barcodes_subset_n8_k2_Levenshtein.csv` | 8 | 2 | 8,164 |
| `barcodes_subset_n8_k3_Levenshtein.csv` | 8 | 3 | 937 |
| `barcodes_subset_n9_k2_Levenshtein.csv` | 9 | 2 | 20,385 |
| `barcodes_subset_n9_k3_Levenshtein.csv` | 9 | 3 | 2,589 |
| `barcodes_subset_n10_k2_Levenshtein.csv` | 10 | 2 | 41,445 |
| `barcodes_subset_n10_k3_Levenshtein.csv` | 10 | 3 | 6,660 |
| `barcodes_subset_n11_k3_Levenshtein.csv` | 11 | 3 | 17,002 |
| `barcodes_subset_n12_k3_Levenshtein.csv` | 12 | 3 | 62,679 |

Note that the n12_k3 subset is identical to the parent designed set, as no prefix
truncation is applied.

---

## Which Set to Use

The library design notebooks use the **subsets** by default. Users specify a desired
edit distance (2 or 3), and the shortest prefix length that provides a sufficient
number of barcodes for the library is selected automatically, iterating to longer
prefixes if upstream filtering (BsmBI removal, GC content, homopolymer) reduces
the available pool below the required threshold.

The independently designed sets are retained for reference and for cases where a
fully optimized pool at a specific length is preferred over a subset of the n12
parent.

---

## Generation and QC Scripts

| Script | Purpose |
|---|---|
| `filter_barcode_subset.py` | Generate a subset from the n12_k3 parent at a given prefix length and edit distance |
| `run_all_subsets.sh` | Batch wrapper that runs `filter_barcode_subset.py` for all prefix/distance combinations |
| `qc_barcode_set.py` | All-pairs Levenshtein verification for any barcode set CSV |
| `plot_barcode_set_sizes.py` | Plot barcode pool sizes across lengths, comparing designed sets and subsets |
