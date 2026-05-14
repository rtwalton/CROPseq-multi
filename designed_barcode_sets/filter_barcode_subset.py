#!/usr/bin/env python3
"""
Find the largest subset of barcodes guaranteeing a minimum Levenshtein distance
over a given prefix length (number of cycles).

Two strategies are chosen automatically based on k-hash window size:

  1. k-hash + conflict graph (Feldman 2020, github.com/feldman4/dna-barcodes):
     Used when the k-hash window is >= 3 bases (longer barcodes, smaller k).
     Barcodes are bucketed by substrings — any pair with distance < k must share
     a bucket — so only within-bucket pairs need Levenshtein evaluation.
     sparse_dist runs in parallel across buckets. maxy_clique_groups then does
     greedy independent-set selection on the conflict graph.

  2. Greedy sequential scan:
     Used when k-hash degenerates (short prefixes or large k, window < 3 bases).
     Each candidate is accepted only if it is >= min_dist from all accepted so
     far. python-Levenshtein makes this fast in practice (~4s for 42k barcodes).
"""

import argparse
import csv
import multiprocessing
import os
import sys
from collections import defaultdict, Counter

import numpy as np
import scipy.sparse

try:
    import Levenshtein as _lev_mod
    _lev_distance = _lev_mod.distance
except ImportError:
    try:
        import editdistance as _ed
        _lev_distance = _ed.eval
    except ImportError:
        def _lev_distance(a, b):
            m, n = len(a), len(b)
            dp = list(range(n + 1))
            for ca in a:
                prev, dp[0] = dp[0], dp[0] + 1
                for j, cb in enumerate(b, 1):
                    prev, dp[j] = dp[j], prev if ca == cb else 1 + min(prev, dp[j], dp[j-1])
            return dp[n]


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(SCRIPT_DIR, "designed", "barcodes_n12_k3_Levenshtein.noBsmBI.csv")

_KHASH_MIN_WINDOW = 3


# ---------------------------------------------------------------------------
# k-hashing
# ---------------------------------------------------------------------------

def _khash_window(n, k):
    return int(np.ceil((n - k) / float(k)))


def khash(s, k):
    n = len(s)
    window = _khash_window(n, k)
    s2 = s + s
    arr = []
    for i in range(n):
        for j in (0, 1):
            arr.append(((i + j) % n, s2[i:i + window]))
    return arr


def build_khash(xs, k):
    D = defaultdict(list)
    for x in xs:
        for h in khash(x, k):
            D[h].append(x)
    return [sorted(set(v)) for v in D.values()]


# ---------------------------------------------------------------------------
# Sparse conflict distances — parallelized across hash buckets
# ---------------------------------------------------------------------------

def _sparse_dist_bucket(bucket, k):
    result = {}
    for i, a in enumerate(bucket):
        for b in bucket[i + 1:]:
            d = _lev_distance(a, b)
            if d < k:
                result[tuple(sorted((a, b)))] = d
    return result


def sparse_dist(hash_buckets, k, cores=1):
    if cores > 1:
        with multiprocessing.Pool(processes=cores) as pool:
            results = pool.starmap(
                _sparse_dist_bucket,
                [(bucket, k) for bucket in hash_buckets],
            )
        D = {}
        for r in results:
            D.update(r)
    else:
        D = {}
        for bucket in hash_buckets:
            D.update(_sparse_dist_bucket(bucket, k))
    return D


def sparse_view(xs, D):
    mapper = {x: i for i, x in enumerate(xs)}
    if D:
        pairs = list(D.keys())
        i_arr = np.array([mapper[a] for a, b in pairs])
        j_arr = np.array([mapper[b] for a, b in pairs])
        data = np.ones(len(pairs), dtype=bool)
    else:
        i_arr, j_arr, data = [], [], []
    n = len(xs)
    cm = scipy.sparse.coo_matrix((data, (i_arr, j_arr)), shape=(n, n))
    return (cm + cm.T).tocsr()


# ---------------------------------------------------------------------------
# Greedy independent-set selection on conflict graph (Feldman 2020)
# ---------------------------------------------------------------------------

def maxy_clique_groups(cm, group_ids):
    d1 = defaultdict(set)
    for id_, count in Counter(group_ids).items():
        d1[count] |= {id_}

    d2 = defaultdict(list)
    for i, id_ in enumerate(group_ids):
        d2[id_].append(i)
    d2 = {k: v[::-1] for k, v in d2.items()}

    selected = []
    available = np.arange(len(group_ids))

    while d1:
        count = min(d1)
        id_ = d1[count].pop()
        if not d1[count]:
            del d1[count]

        index = None
        while d2[id_]:
            idx = d2[id_].pop()
            if idx in available:
                index = idx
                break

        if index is not None:
            selected.append(index)
            available = available[available != index]
            remove = cm[index, available].indices
            mask = np.ones(len(available), dtype=bool)
            mask[remove] = False
            available = available[mask]

        n = len(d2[id_])
        if n > 0:
            d1[n] |= {id_}

    return selected


# ---------------------------------------------------------------------------
# Greedy sequential scan (fallback for small k-hash windows)
# ---------------------------------------------------------------------------

def greedy_scan(prefixes, min_dist):
    accepted = []
    for i, prefix in enumerate(prefixes):
        if i % 5000 == 0:
            print(f"  scanning {i}/{len(prefixes)}, accepted {len(accepted)} so far...",
                  file=sys.stderr)
        if all(_lev_distance(prefix, a) >= min_dist for a in accepted):
            accepted.append(prefix)
    return set(accepted)


# ---------------------------------------------------------------------------
# Main selection logic
# ---------------------------------------------------------------------------

def select_subset(barcodes, cycles, min_dist, cores=1):
    prefixes = [b[:cycles] for b in barcodes]
    unique_prefixes = list(dict.fromkeys(prefixes))

    window = _khash_window(cycles, min_dist)
    print(f"  k-hash window size: {window} (min useful: {_KHASH_MIN_WINDOW})", file=sys.stderr)

    if window >= _KHASH_MIN_WINDOW:
        print(f"  strategy: k-hash + conflict graph ({len(unique_prefixes)} unique prefixes)...",
              file=sys.stderr)
        hash_buckets = build_khash(unique_prefixes, min_dist)
        print(f"  {len(hash_buckets)} buckets; computing distances (cores={cores})...",
              file=sys.stderr)
        D = sparse_dist(hash_buckets, min_dist, cores=cores)
        print(f"  {len(D)} conflict pairs; selecting via greedy clique...", file=sys.stderr)
        cm = sparse_view(unique_prefixes, D)
        selected_idx = maxy_clique_groups(cm, [0] * len(unique_prefixes))
        selected_prefix_set = {unique_prefixes[i] for i in selected_idx}
    else:
        print(f"  strategy: greedy scan ({len(unique_prefixes)} unique prefixes)...",
              file=sys.stderr)
        selected_prefix_set = greedy_scan(unique_prefixes, min_dist)

    # Map back to full barcodes (first occurrence of each selected prefix)
    seen = set()
    result = []
    for bc, pfx in zip(barcodes, prefixes):
        if pfx in selected_prefix_set and pfx not in seen:
            seen.add(pfx)
            result.append(bc)
    return result


# ---------------------------------------------------------------------------
# QC: all-pairs verification
# ---------------------------------------------------------------------------

def check_barcode_set(sequences, min_dist):
    """Check all pairs of sequences for distance violations.

    Returns a list of (a, b, distance) tuples where distance < min_dist.
    sequences are evaluated as-is (pass prefixes if checking a prefix length).
    """
    violations = []
    seqs = list(sequences)
    n = len(seqs)
    for i in range(n):
        for j in range(i + 1, n):
            d = _lev_distance(seqs[i], seqs[j])
            if d < min_dist:
                violations.append((seqs[i], seqs[j], d))
    return violations


def report_qc(violations, min_dist, n_barcodes, file=sys.stderr):
    n_pairs = n_barcodes * (n_barcodes - 1) // 2
    print(f"  QC: checked {n_pairs:,} pairs across {n_barcodes} barcodes", file=file)
    if violations:
        print(f"  QC: FAILED — {len(violations)} violation(s) (dist < {min_dist}):", file=file)
        for a, b, d in violations[:10]:
            print(f"    {a}  {b}  dist={d}", file=file)
        if len(violations) > 10:
            print(f"    ... and {len(violations) - 10} more", file=file)
    else:
        print(f"  QC: PASSED — all pairs have Levenshtein distance >= {min_dist}", file=file)
    return len(violations) == 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_barcodes(path):
    with open(path, newline="") as fh:
        return [row["barcode"] for row in csv.DictReader(fh)]


def main():
    parser = argparse.ArgumentParser(
        description="Find largest barcode subset with given Levenshtein distance in N cycles."
    )
    parser.add_argument(
        "--input", "-i", default=DEFAULT_INPUT,
        help="CSV file with 'barcode' column (default: barcodes_n12_k3_Levenshtein.noBsmBI.csv)"
    )
    parser.add_argument(
        "--cycles", "-n", type=int, required=True,
        help="Number of sequencing cycles (prefix length to evaluate)"
    )
    parser.add_argument(
        "--min-dist", "-k", type=int, required=True,
        help="Minimum pairwise Levenshtein distance required"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output CSV path (default: barcodes_subset_n{cycles}_k{min_dist}_Levenshtein.csv next to input)"
    )
    parser.add_argument(
        "--cores", "-p", type=int, default=max(1, multiprocessing.cpu_count() - 1),
        help="Number of parallel worker processes (default: nCPU - 1)"
    )
    parser.add_argument(
        "--max-candidates", "-m", type=int, default=None,
        help="Limit input to first N barcodes (for quick testing)"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="After selection, verify all pairs satisfy the distance guarantee"
    )
    args = parser.parse_args()

    output_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(args.input)),
        f"barcodes_subset_n{args.cycles}_k{args.min_dist}_Levenshtein.csv",
    )

    print(f"Loading barcodes from {args.input} ...", file=sys.stderr)
    barcodes = load_barcodes(args.input)
    if args.max_candidates:
        barcodes = barcodes[:args.max_candidates]
    print(f"Loaded {len(barcodes)} barcodes.", file=sys.stderr)

    print(f"Selecting subset: cycles={args.cycles}, min_dist={args.min_dist}, cores={args.cores} ...",
          file=sys.stderr)
    subset = select_subset(barcodes, args.cycles, args.min_dist, cores=args.cores)
    print(f"Selected {len(subset)} barcodes.", file=sys.stderr)

    if args.verify:
        print("Verifying output (all-pairs check on prefixes)...", file=sys.stderr)
        prefixes = [bc[:args.cycles] for bc in subset]
        violations = check_barcode_set(prefixes, args.min_dist)
        passed = report_qc(violations, args.min_dist, len(subset))
        if not passed:
            sys.exit(1)

    with open(output_path, "w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["barcode", "prefix", "cycles", "min_levenshtein"])
        for bc in subset:
            writer.writerow([bc, bc[:args.cycles], args.cycles, args.min_dist])
    print(f"Written to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
