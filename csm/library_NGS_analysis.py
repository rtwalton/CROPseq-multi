import os
import shutil
import subprocess
import tempfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from Bio.Seq import Seq
import gzip
from tqdm import tqdm
import constants
import warnings
import itertools

def lorenz_curve(X, library_name='library'):
    X_sorted = X.copy()
    X_sorted.sort()
    X_lorenz = X_sorted.cumsum() / X_sorted.sum()
    X_lorenz = np.insert(X_lorenz, 0, 0) 
    X_lorenz[0], X_lorenz[-1]
    fig, ax = plt.subplots(figsize=[3,3])
    ## scatter plot of Lorenz curve
    ax.plot(np.arange(X_lorenz.size)/(X_lorenz.size-1), X_lorenz, 
               color='dodgerblue', label=library_name)
    ## line plot of equality
    ax.plot([0,1], [0,1], color='k', label='line of equality')
    plt.xlabel('fraction of reads')
    plt.ylabel('fraction of elements')
    plt.legend()
    
def gini(arr):
    ## first sort
    sorted_arr = arr.copy()
    sorted_arr.sort()
    n = arr.size
    coef_ = 2. / n
    const_ = (n + 1.) / n
    weighted_sum = sum([(i+1)*yi for i, yi in enumerate(sorted_arr)])
    return coef_*weighted_sum/(sorted_arr.sum()) - const_

def ratio_9010(arr, percentile=10):
    sorted_arr = arr.copy()
    sorted_arr.sort()
    n_percentile = int(arr.size/percentile)
    return sorted_arr[-n_percentile]/sorted_arr[n_percentile]


def generate_unique_construct_identifiers(df, use_tRNA=False, iBAR2_UMI=False, custom=None):
    """
    return df with added unique construct identifier (combination of spacers, iBARs, tRNA)
    """
    # if custom list of columns is provided, use that to generate unique identifier
    # this overrides the other parameters
    if custom is not None:
      df['unique_barcode_combo'] = df[custom].agg('-'.join, axis=1)
      return df
    
    if use_tRNA:
        if iBAR2_UMI:
            df['unique_barcode_combo'] =\
                df['spacer_1'] + '-' + \
                df['iBAR_1'] + '-' + \
                df['tRNA'] + '-' + \
                df['spacer_2']
        else:
            df['unique_barcode_combo'] =\
                df['spacer_1'] + '-' + \
                df['iBAR_1'] + '-' + \
                df['tRNA'] + '-' + \
                df['spacer_2'] + '-' +\
                df['iBAR_2']
    else:
        if iBAR2_UMI:
            df['unique_barcode_combo'] =\
                df['spacer_1'] + '-' + \
                df['iBAR_1'] + '-' + \
                df['spacer_2']
        else:
            df['unique_barcode_combo'] =\
                df['spacer_1'] + '-' + \
                df['iBAR_1'] + '-' + \
                df['spacer_2']+ '-' +\
                df['iBAR_2']

    return df

def extract_barcodes_from_fastq_pair(
    fastq_pair, 
    use_tRNA=False, 
    iBAR2_UMI=False,
    custom_read_primers = True,
    single_fastq = False,
    max_reads=1e7,
    **kwargs
    ):
    """
    parse pair of read 1 and read 2 fastq files and extract sequences
    """

    if custom_read_primers:
        read_1_ref_pos = 0
        read_2_ref_pos = 0
    else:
        read_1_ref_pos = len(constants.read_1_primer_seq)
        read_2_ref_pos = len(constants.read_2_primer_seq)

    # first sgRNA spacer and iBAR in read 1
    spacer1_coords = (read_1_ref_pos,read_1_ref_pos+20)
    iBar1_coords = (read_1_ref_pos+39,read_1_ref_pos+51)

    # second sgRNA spacer and iBAR in read 2
    spacer2_coords = (read_2_ref_pos+31,read_2_ref_pos+51)
    iBar2_coords = (read_2_ref_pos,read_2_ref_pos+12)
    # middle tRNA in read 2, use 4 nt prefix to map
    tRNA_2_coords = (read_2_ref_pos+51,read_2_ref_pos+51+4)

    infileR1 = gzip.open(fastq_pair[0], 'rt')
    if single_fastq:
        infileR2 = gzip.open(fastq_pair[0], 'rt')
    else:
        infileR2 = gzip.open(fastq_pair[1], 'rt')

    fastq_data = []
    total_reads = 0
    # this is slow so probably don't want to run this on tens of million of reads

    while infileR1.readline() and infileR2.readline():

        if not single_fastq:
            # paired reads are corresponding rows in separate files
            read_sequenceR1 = infileR1.readline().strip()
            infileR1.readline()
            infileR1.readline()
            read_sequenceR2 = infileR2.readline().strip()
            infileR2.readline()
            infileR2.readline()
        else:
            # paried reads are alternating entries in single fastq
            read_sequence_1 = infileR1.readline().strip()
            infileR1.readline()
            infileR1.readline()
            infileR1.readline()
            # paired read 1 is subsequent entry
            read_sequence_2 = infileR1.readline().strip()
            infileR1.readline()
            infileR1.readline()
            # use first 8 nt of read to assign to read 1 or read 2
            if (read_sequence_1[:8] == constants.read_1_primer_seq[:8]) or\
                (read_sequence_2[:8] == constants.read_2_primer_seq[:8]):
                read_sequenceR1 = read_sequence_1
                read_sequenceR2 = read_sequence_2
            elif (read_sequence_2[:8] == constants.read_1_primer_seq[:8]) or\
                 (read_sequence_1[:8] == constants.read_2_primer_seq[:8]):
                read_sequenceR1 = read_sequence_2
                read_sequenceR2 = read_sequence_1
            else: # for now, just defaulting to this option so that reads will be counted as 'not mapped'
                read_sequenceR1 = read_sequence_1
                read_sequenceR2 = read_sequence_2

        sg1_seq = read_sequenceR1[spacer1_coords[0]:spacer1_coords[1]]
        # iBARs in SBS orientation (reverse complement of sgRNA orientation)
        iBar1_seq = str(Seq(read_sequenceR1[iBar1_coords[0]:iBar1_coords[1]]).reverse_complement())

        sg2_seq = str(Seq(read_sequenceR2[spacer2_coords[0]:spacer2_coords[1]]).reverse_complement())
        # iBARs in SBS orientation (reverse complement of sgRNA orientation)
        iBar2_seq = read_sequenceR2[iBar2_coords[0]:iBar2_coords[1]]
        
        tRNA_2_seq = str(Seq(read_sequenceR2[tRNA_2_coords[0]:tRNA_2_coords[1]]).reverse_complement())

        fastq_data.append([sg1_seq, iBar1_seq, sg2_seq, iBar2_seq, tRNA_2_seq])
        total_reads += 1
        if total_reads >= max_reads:
            break

    df_barcodes = pd.DataFrame(
        columns =  ['spacer_1', 'iBAR_1', 'spacer_2', 'iBAR_2', 'tRNA_short'], 
        data=fastq_data)

    df_barcodes['tRNA']= df_barcodes['tRNA_short'].map(constants.tRNA_map_by_prefix)

    df_barcodes = generate_unique_construct_identifiers(df_barcodes, use_tRNA, iBAR2_UMI, **kwargs)

    return df_barcodes

def extract_barcodes_from_fastq_pair_align(
    fastq_pair, 
    use_tRNA=False, 
    iBAR2_UMI=False,
    single_fastq = False,
    spacer_len = 20,
    iBAR_len = 12,
    max_reads=1e7,
    **kwargs
    ):
    """
    parse pair of read 1 and read 2 fastq files and extract sequences
    """

    infileR1 = gzip.open(fastq_pair[0], 'rt')
    if single_fastq:
        infileR2 = gzip.open(fastq_pair[0], 'rt')
    else:
        infileR2 = gzip.open(fastq_pair[1], 'rt')

    fastq_data = []
    total_reads = 0
    # this is slow so probably don't want to run this on tens of million of reads

    while infileR1.readline() and infileR2.readline():

        if not single_fastq:
            # paired reads are corresponding rows in separate files
            read_sequenceR1 = infileR1.readline().strip()
            infileR1.readline()
            infileR1.readline()
            read_sequenceR2 = infileR2.readline().strip()
            infileR2.readline()
            infileR2.readline()
        else:
            # paried reads are alternating entries in single fastq
            read_sequence_1 = infileR1.readline().strip()
            infileR1.readline()
            infileR1.readline()
            infileR1.readline()
            # paired read 1 is subsequent entry
            read_sequence_2 = infileR1.readline().strip()
            infileR1.readline()
            infileR1.readline()
            # identify which read is which
            if (read_sequence_1.find(constants.CSM_stem_1)!=-1) or\
                (read_sequence_2.find(str(Seq(constants.CSM_stem_2).reverse_complement()))!=-1):
                read_sequenceR1 = read_sequence_1
                read_sequenceR2 = read_sequence_2
            elif (read_sequence_1.find(str(Seq(constants.CSM_stem_2).reverse_complement()))!=-1) or\
                (read_sequence_2.find(constants.CSM_stem_1)!=-1):
                read_sequenceR1 = read_sequence_2
                read_sequenceR2 = read_sequence_1
            else: # for now, just defaulting to this option so that reads will be counted as 'not mapped'
                read_sequenceR1 = read_sequence_1
                read_sequenceR2 = read_sequence_2

        read_1_ref_pos = read_sequenceR1.find(constants.CSM_stem_1)
        read_2_ref_pos = read_sequenceR2.find(str(Seq(constants.CSM_stem_2).reverse_complement()))
    
        # first sgRNA spacer and iBAR in read 1
        spacer1_coords = (read_1_ref_pos-spacer_len, read_1_ref_pos)
        iBar1_coords = (read_1_ref_pos+len(constants.CSM_stem_1), read_1_ref_pos+len(constants.CSM_stem_1)+iBAR_len)
        
        # second sgRNA spacer and iBAR in read 2
        spacer2_coords = (read_2_ref_pos+len(constants.CSM_stem_2), read_2_ref_pos+len(constants.CSM_stem_2)+spacer_len)
        iBar2_coords = (read_2_ref_pos-iBAR_len, read_2_ref_pos)
        # middle tRNA in read 2, use 4 nt prefix to map
        tRNA_2_coords = (spacer2_coords[1], spacer2_coords[1]+4)

        def extract_seq(read, cooridnates):
            try:
                return read[cooridnates[0]:cooridnates[1]]
            except:
                return 'N'*int(cooridnates[1]-cooridnates[0])
        
        sg1_seq = extract_seq(read_sequenceR1, spacer1_coords)
        iBar1_seq = str(Seq(extract_seq(read_sequenceR1, iBar1_coords)).reverse_complement())
        sg2_seq = str(Seq(extract_seq(read_sequenceR2, spacer2_coords)).reverse_complement())
        iBar2_seq = extract_seq(read_sequenceR2, iBar2_coords)
        tRNA_2_seq = str(Seq(extract_seq(read_sequenceR2, tRNA_2_coords)).reverse_complement())
        
        fastq_data.append([sg1_seq, iBar1_seq, sg2_seq, iBar2_seq, tRNA_2_seq])
        total_reads += 1
        if total_reads >= max_reads:
            break

    df_barcodes = pd.DataFrame(
        columns =  ['spacer_1', 'iBAR_1', 'spacer_2', 'iBAR_2', 'tRNA_short'], 
        data=fastq_data)

    df_barcodes['tRNA']= df_barcodes['tRNA_short'].map(constants.tRNA_map_by_prefix)

    df_barcodes = generate_unique_construct_identifiers(df_barcodes, use_tRNA, iBAR2_UMI, **kwargs)

    return df_barcodes


# ---------------------------------------------------------------------------
# awk helper code embedded as a Python string.
# Shared by extract_barcode_counts_subprocess for both extraction methods.
# ---------------------------------------------------------------------------
_AWK_HELPERS = r"""
function rc(seq,    i, c, r) {
    r = ""
    for (i = length(seq); i >= 1; i--) {
        c = substr(seq, i, 1)
        if      (c == "A") r = r "T"
        else if (c == "T") r = r "A"
        else if (c == "C") r = r "G"
        else if (c == "G") r = r "C"
        else               r = r "N"
    }
    return r
}
function safe_sub(seq, s, n,    out, j) {
    if (s < 1 || s + n - 1 > length(seq)) {
        out = ""; for (j = 0; j < n; j++) out = out "N"; return out
    }
    return substr(seq, s, n)
}
"""

# Complement translation table shared by the fast-Python extraction path
_DNA_COMP = str.maketrans('ACGTacgtNn', 'TGCAtgcaNn')


def _extract_barcode_counts_python(
    fastq_pair,
    method='align',
    custom_read_primers=True,
    single_fastq=False,
    max_reads=1e7,
    spacer_len=20,
    iBAR_len=12,
):
    """
    Pure-Python barcode extraction with dict-based aggregation.

    Replaces the slow per-read-append pattern in extract_barcodes_from_fastq_pair* with:
      - str.translate() for reverse-complement (3× faster than Biopython)
      - a dict counter so the final DataFrame has ~library_size rows, not ~total_reads rows

    Returns the same compact DataFrame as extract_barcode_counts_subprocess.
    Called automatically by extract_barcode_counts_subprocess when gawk/mawk are absent.
    """
    def rc(s):
        return s.translate(_DNA_COMP)[::-1]

    def safe_sub(seq, start, length):
        if start < 0 or start + length > len(seq):
            return 'N' * length
        return seq[start:start + length]

    stem1    = constants.CSM_stem_1
    stem2_rc = str(Seq(constants.CSM_stem_2).reverse_complement())
    stem_len = len(stem1)

    r1_off = 0 if custom_read_primers else len(constants.read_1_primer_seq)
    r2_off = 0 if custom_read_primers else len(constants.read_2_primer_seq)

    def _open(f):
        return gzip.open(f, 'rt') if f.endswith(('.gz', '.gzip')) else open(f, 'rt')

    counts = {}   # (sp1, ib1, sp2, ib2, tRNA_short) → int
    total  = 0

    f1 = _open(fastq_pair[0])
    f2 = _open(fastq_pair[0] if single_fastq else fastq_pair[1])

    try:
        while True:
            # Read one record from each file (4 lines each)
            if single_fastq:
                h1 = f1.readline()
                if not h1: break
                s1 = f1.readline().strip(); f1.readline(); f1.readline()
                f1.readline()
                s2 = f1.readline().strip(); f1.readline(); f1.readline()
                # Identify R1/R2 by stem presence
                if (stem1 in s1) or (stem2_rc not in s2):
                    r1s, r2s = s1, s2
                else:
                    r1s, r2s = s2, s1
            else:
                h1 = f1.readline()
                if not h1: break
                r1s = f1.readline().strip(); f1.readline(); f1.readline()
                f2.readline()
                r2s = f2.readline().strip(); f2.readline(); f2.readline()

            # --- Barcode extraction ---
            if method == 'align':
                p1 = r1s.find(stem1)
                p2 = r2s.find(stem2_rc)
                if p1 >= 0:
                    sp1 = safe_sub(r1s, p1 - spacer_len, spacer_len)
                    ib1 = rc(safe_sub(r1s, p1 + stem_len, iBAR_len))
                else:
                    sp1, ib1 = 'N' * spacer_len, 'N' * iBAR_len
                if p2 >= 0:
                    sp2 = rc(safe_sub(r2s, p2 + stem_len, spacer_len))
                    ib2 = safe_sub(r2s, p2 - iBAR_len, iBAR_len)
                    ts  = rc(safe_sub(r2s, p2 + stem_len + spacer_len, 4))
                else:
                    sp2, ib2, ts = 'N' * spacer_len, 'N' * iBAR_len, 'NNNN'
            else:  # 'position'
                sp1 = safe_sub(r1s, r1_off,       spacer_len)
                ib1 = rc(safe_sub(r1s, r1_off + 39, iBAR_len))
                sp2 = rc(safe_sub(r2s, r2_off + 31, spacer_len))
                ib2 = safe_sub(r2s, r2_off,         iBAR_len)
                ts  = rc(safe_sub(r2s, r2_off + 51, 4))

            key = (sp1, ib1, sp2, ib2, ts)
            counts[key] = counts.get(key, 0) + 1
            total += 1
            if total >= max_reads:
                break
    finally:
        f1.close()
        if not single_fastq:
            f2.close()

    if not counts:
        return pd.DataFrame(
            columns=['spacer_1', 'iBAR_1', 'spacer_2', 'iBAR_2', 'tRNA_short', 'tRNA', 'count'])

    df = pd.DataFrame(
        [(k[0], k[1], k[2], k[3], k[4], v) for k, v in counts.items()],
        columns=['spacer_1', 'iBAR_1', 'spacer_2', 'iBAR_2', 'tRNA_short', 'count'],
    )
    df['tRNA'] = df['tRNA_short'].map(constants.tRNA_map_by_prefix)
    return df


def extract_barcode_counts_subprocess(
    fastq_pair,
    method='align',
    custom_read_primers=True,
    single_fastq=False,
    max_reads=1e7,
    spacer_len=20,
    iBAR_len=12,
):
    """
    Extract and aggregate barcode combinations from a FASTQ pair using an awk subprocess.

    This is the fast replacement for the pure-Python extraction functions.  Instead of
    loading ~10 million reads into a Python DataFrame (one row per read), awk extracts
    barcodes at C speed, aggregates identical combinations in an associative array, and
    streams back only the unique combos with their counts.  For a typical CROPseq-multi
    library (~1,000 constructs, ~10M reads) this reduces the data handed to Python from
    ~10,000,000 rows to ~1,000–5,000 rows.

    Parameters
    ----------
    fastq_pair : list of str
        [path_R1, path_R2] — gzip or plain FASTQ.  For single_fastq mode pass the
        same path twice (or any second path; it is ignored).
    method : str
        'align' (default) — finds CSM stem sequences dynamically.
        'position' — uses fixed offsets from the read start.
    custom_read_primers : bool
        If True (default) the read starts at the dialout primer (offset = 0).
        If False the read starts with the Illumina sequencing primer; primer length
        is subtracted before extracting barcodes (position method only).
    single_fastq : bool
        If True, R1 and R2 are interleaved in a single file (alternating records).
    max_reads : int
        Stop after this many read pairs.
    spacer_len, iBAR_len : int
        Override default spacer/iBAR lengths (20 and 12 nt).

    Returns
    -------
    pandas.DataFrame
        Columns: spacer_1, iBAR_1, spacer_2, iBAR_2, tRNA_short, tRNA, count
        One row per unique barcode combination observed in the data.

    Notes
    -----
    Requires ``gawk`` or ``mawk`` on PATH.  If neither is found, falls back
    automatically to _extract_barcode_counts_python (dict aggregation with
    str.translate RC — still much faster than the legacy per-read DataFrame path).
    """
    awk_exe = shutil.which('gawk') or shutil.which('mawk')
    if awk_exe is None:
        return _extract_barcode_counts_python(
            fastq_pair, method=method,
            custom_read_primers=custom_read_primers,
            single_fastq=single_fastq,
            max_reads=max_reads,
            spacer_len=spacer_len,
            iBAR_len=iBAR_len,
        )

    from io import StringIO

    r1 = fastq_pair[0]
    r2 = fastq_pair[1] if not single_fastq else fastq_pair[0]

    stem1    = constants.CSM_stem_1
    stem2_rc = str(Seq(constants.CSM_stem_2).reverse_complement())
    stem_len = len(stem1)   # 19

    r1_off = 0 if custom_read_primers else len(constants.read_1_primer_seq)
    r2_off = 0 if custom_read_primers else len(constants.read_2_primer_seq)

    # Build the tRNA lookup table for awk's BEGIN block
    trna_init = '\n    '.join(
        f'trna_map["{k}"] = "{v}"'
        for k, v in constants.tRNA_map_by_prefix.items()
    )

    # --- Extraction logic: differs by method ---
    if method == 'align':
        extraction_block = f"""\
    p1 = index(r1, STEM1)   - 1
    p2 = index(r2, STEM2RC) - 1
    if (p1 >= 0) {{
        sp1 = safe_sub(r1, p1 - SL + 1, SL)
        ib1 = rc(safe_sub(r1, p1 + STEMLEN + 1, IL))
    }} else {{ sp1 = NSP; ib1 = NIB }}
    if (p2 >= 0) {{
        sp2 = rc(safe_sub(r2, p2 + STEMLEN + 1, SL))
        ib2 = safe_sub(r2, p2 - IL + 1, IL)
        ts  = rc(safe_sub(r2, p2 + STEMLEN + SL + 1, 4))
    }} else {{ sp2 = NSP; ib2 = NIB; ts = "NNNN" }}"""
    else:  # 'position'
        extraction_block = f"""\
    sp1 = safe_sub(r1, {r1_off} + 1,         SL)
    ib1 = rc(safe_sub(r1, {r1_off} + 39 + 1, IL))
    sp2 = rc(safe_sub(r2, {r2_off} + 31 + 1, SL))
    ib2 = safe_sub(r2,    {r2_off} + 1,       IL)
    ts  = rc(safe_sub(r2, {r2_off} + 51 + 1,  4))"""

    # --- awk program: two modes selected by SINGLE_FASTQ variable ---
    # Paired mode  (SINGLE_FASTQ=0): input is paste output → tab-sep lines,
    #   NR%4==2 picks sequence lines, $1=R1, $2=R2.
    # Single-fastq (SINGLE_FASTQ=1): input is one stream of 8 lines per pair,
    #   NR%8==2 → first fragment seq, NR%8==6 → second fragment seq.
    awk_prog = f"""\
{_AWK_HELPERS}
BEGIN {{
    FS = "\\t"
    {trna_init}
    STEM1   = "{stem1}"
    STEM2RC = "{stem2_rc}"
    STEMLEN = {stem_len}
    SL = {spacer_len}
    IL = {iBAR_len}
    NSP = ""; for (j = 0; j < SL; j++) NSP = NSP "N"
    NIB = ""; for (j = 0; j < IL; j++) NIB = NIB "N"
    reads = 0
}}

# ---- paired-end mode (paste output, tab-separated) ----
SINGLE_FASTQ == 0 && NR % 4 == 2 {{
    reads++; if (reads > MAX_READS) exit
    r1 = $1; r2 = $2
{extraction_block}
    counts[sp1 "\\t" ib1 "\\t" sp2 "\\t" ib2 "\\t" ts]++
}}

# ---- single-fastq mode (interleaved, 8 lines per pair) ----
SINGLE_FASTQ == 1 && NR % 8 == 2 {{ seq1 = $0 }}
SINGLE_FASTQ == 1 && NR % 8 == 6 {{
    seq2 = $0
    reads++; if (reads > MAX_READS) exit
    if (index(seq1, STEM1) > 0 || index(seq2, STEM2RC) == 0) {{
        r1 = seq1; r2 = seq2
    }} else {{
        r1 = seq2; r2 = seq1
    }}
{extraction_block}
    counts[sp1 "\\t" ib1 "\\t" sp2 "\\t" ib2 "\\t" ts]++
}}

END {{
    for (k in counts) print k "\\t" counts[k]
}}
"""

    def _open_cmd(f):
        return f"gzip -dc '{f}'" if f.endswith(('.gz', '.gzip')) else f"cat '{f}'"

    if single_fastq:
        fastq_cmd = _open_cmd(r1)
    else:
        fastq_cmd = f"paste <({_open_cmd(r1)}) <({_open_cmd(r2)})"

    # Write the awk program to a temp file to avoid shell-quoting issues
    with tempfile.NamedTemporaryFile(mode='w', suffix='.awk', delete=False) as awk_f:
        awk_f.write(awk_prog)
        awk_path = awk_f.name
    try:
        bash_cmd = (
            f"{fastq_cmd} | "
            f"{awk_exe} -v SINGLE_FASTQ={int(single_fastq)} "
            f"           -v MAX_READS={int(max_reads)} "
            f"           -f '{awk_path}'"
        )
        result = subprocess.run(
            ['bash', '-c', bash_cmd],
            capture_output=True, text=True, check=True,
        )
    finally:
        os.unlink(awk_path)

    df = pd.read_csv(
        StringIO(result.stdout),
        sep='\t',
        names=['spacer_1', 'iBAR_1', 'spacer_2', 'iBAR_2', 'tRNA_short', 'count'],
        dtype={'count': int},
    )
    df['tRNA'] = df['tRNA_short'].map(constants.tRNA_map_by_prefix)
    return df


def _count_constructs_bash(
    lib_info_df,
    lib_design_input_df,
    use_tRNA=False,
    iBAR2_UMI=False,
    custom_read_primers=True,
    single_fastq=False,
    return_raw_barcodes=False,
    method='align',
    max_reads=1e7,
    custom_mapping_columns=None,
):
    """
    Bash-accelerated backend for count_constructs.

    Uses extract_barcode_counts_subprocess() to run FASTQ extraction in awk,
    then performs all library mapping and statistics in Python on the resulting
    compact table (~library_size rows, not ~total_reads rows).

    Because statistics are computed as count-weighted sums over the aggregated
    table, all features of the Python backend are fully supported including
    iBAR2_UMI, custom_mapping_columns, and return_raw_barcodes.
    """
    df_total = pd.DataFrame()
    df_summary = pd.DataFrame()
    lib_design_df = generate_unique_construct_identifiers(
        lib_design_input_df, use_tRNA, iBAR2_UMI, custom=custom_mapping_columns)

    for index, row in tqdm(lib_info_df.iterrows(), total=lib_info_df.shape[0]):
        sample_id = row['sample_ID']
        fastq_pair = [row['fastq_R1'], row['fastq_R2']]

        if 'dialout' in lib_info_df.columns:
            lib_design_index = {
                k: v for v, k in lib_design_df[
                    lib_design_df['dialout'] == row['dialout']
                ]['unique_barcode_combo'].to_dict().items()
            }
        else:
            lib_design_index = {
                k: v for v, k in lib_design_df['unique_barcode_combo'].to_dict().items()
            }

        # Fast awk extraction: ~library_size rows instead of ~total_reads rows
        df_barcodes = extract_barcode_counts_subprocess(
            fastq_pair,
            method=method,
            custom_read_primers=custom_read_primers,
            single_fastq=single_fastq,
            max_reads=max_reads,
        )
        total_reads = int(df_barcodes['count'].sum())

        df_barcodes = generate_unique_construct_identifiers(
            df_barcodes, use_tRNA, iBAR2_UMI, custom=custom_mapping_columns)

        # Map individual elements
        df_barcodes['spacer_1_map'] = df_barcodes['spacer_1'].isin(lib_design_df['spacer_1'])
        df_barcodes['iBAR_1_map']   = df_barcodes['iBAR_1'].isin(lib_design_df['iBAR_1'])
        df_barcodes['spacer_2_map'] = df_barcodes['spacer_2'].isin(lib_design_df['spacer_2'])
        df_barcodes['iBAR_2_map']   = df_barcodes['iBAR_2'].isin(lib_design_df['iBAR_2'])
        df_barcodes['tRNA_map']     = df_barcodes['tRNA'].notna()

        # Map full constructs
        df_barcodes['design_index'] = df_barcodes['unique_barcode_combo'].map(lib_design_index)
        df_barcodes['sample']       = sample_id
        df_barcodes['timepoint']    = row['timepoint']
        df_barcodes['replicate']    = row['replicate']

        if return_raw_barcodes:
            df_total = pd.concat([df_total, df_barcodes])

        # Per-construct counts: weighted sum of 'count' grouped by design_index
        construct_counts = df_barcodes.groupby('design_index', dropna=True)['count'].sum()

        if index == 0:
            lib_design_counts_df = pd.merge(
                lib_design_input_df,
                pd.Series(construct_counts, name=sample_id),
                left_index=True, right_index=True, how='outer').fillna(0)
        else:
            lib_design_counts_df = pd.merge(
                lib_design_counts_df,
                pd.Series(construct_counts, name=sample_id),
                left_index=True, right_index=True, how='outer').fillna(0)

        if iBAR2_UMI:
            # Each row in df_barcodes is a unique barcode combo, so nunique(iBAR_2) per
            # design_index equals the number of distinct iBAR_2 sequences observed —
            # identical to the Python backend's groupby on the full per-read table.
            umi_counts = df_barcodes.groupby('design_index', dropna=True)['iBAR_2'].nunique()
            lib_design_counts_df = pd.merge(
                lib_design_counts_df,
                pd.Series(umi_counts, name=sample_id + '_UMI'),
                left_index=True, right_index=True, how='outer').fillna(0)

        if 'dialout' in lib_info_df.columns:
            sublib_design_counts_df = lib_design_counts_df[
                lib_design_counts_df['dialout'] == row['dialout']]
        else:
            sublib_design_counts_df = lib_design_counts_df

        # Count-weighted statistics
        # wsum(mask) = number of *reads* (not rows) satisfying mask
        def wsum(mask):
            return int((mask * df_barcodes['count']).sum())

        df_barcodes_mapped = df_barcodes[df_barcodes['design_index'].notna()]
        mapped_constructs_count = int(df_barcodes_mapped['count'].sum())

        df_summary.loc[sample_id, 'timepoint']    = row['timepoint']
        df_summary.loc[sample_id, 'replicate']    = row['replicate']
        df_summary.loc[sample_id, 'n_constructs'] = len(lib_design_df)
        df_summary.loc[sample_id, 'total_read_count'] = total_reads
        df_summary.loc[sample_id, 'NGS_coverage'] = total_reads / len(lib_design_df)

        df_summary.loc[sample_id, 'spacer_1_map'] = wsum(df_barcodes['spacer_1_map']) / total_reads
        df_summary.loc[sample_id, 'iBAR_1_map']   = wsum(df_barcodes['iBAR_1_map'])   / total_reads
        df_summary.loc[sample_id, 'spacer_2_map'] = wsum(df_barcodes['spacer_2_map']) / total_reads
        df_summary.loc[sample_id, 'iBAR_2_map']   = wsum(df_barcodes['iBAR_2_map'])   / total_reads
        df_summary.loc[sample_id, 'tRNA_map']     = wsum(df_barcodes['tRNA_map'])     / total_reads

        if iBAR2_UMI:
            df_summary.loc[sample_id, 'iBAR_2_UMI_mean_count'] = \
                sublib_design_counts_df[sample_id + '_UMI'].mean()

        if use_tRNA:
            all_map = (df_barcodes['spacer_1_map'] & df_barcodes['iBAR_1_map'] &
                       df_barcodes['tRNA_map'] & df_barcodes['spacer_2_map'])
            if not iBAR2_UMI:
                all_map = all_map & df_barcodes['iBAR_2_map']
        else:
            all_map = (df_barcodes['spacer_1_map'] & df_barcodes['iBAR_1_map'] &
                       df_barcodes['spacer_2_map'])
            if not iBAR2_UMI:
                all_map = all_map & df_barcodes['iBAR_2_map']

        if custom_mapping_columns is not None:
            mapping_cols = [c + '_map' for c in custom_mapping_columns]
            all_map = df_barcodes[mapping_cols].all(axis=1)

        all_elements_mapped_count = wsum(all_map)

        df_summary.loc[sample_id, 'all_elements_mapped_count'] = all_elements_mapped_count
        df_summary.loc[sample_id, 'mapped_constructs_count']   = mapped_constructs_count
        df_summary.loc[sample_id, 'fraction_mapped']           = \
            mapped_constructs_count / total_reads
        df_summary.loc[sample_id, 'fraction_recombined']       = \
            1 - mapped_constructs_count / all_elements_mapped_count \
            if all_elements_mapped_count > 0 else 0
        df_summary.loc[sample_id, 'dropout_count'] = \
            (sublib_design_counts_df[sample_id] == 0).sum()
        df_summary.loc[sample_id, 'gini_coefficient'] = \
            gini(sublib_design_counts_df[sample_id].values)
        df_summary.loc[sample_id, 'ratio_90_10'] = \
            ratio_9010(sublib_design_counts_df[sample_id].values)

    df_summary.index.rename('sample_ID', inplace=True)

    if return_raw_barcodes:
        return lib_design_counts_df, df_summary, df_total
    return lib_design_counts_df, df_summary


def count_constructs(
    lib_info_df,
    lib_design_input_df,
    use_tRNA=False,
    iBAR2_UMI=False,
    custom_read_primers=True,
    single_fastq=False,
    return_raw_barcodes=False,
    method='align',
    max_reads=1e7,
    custom_mapping_columns=None,
    backend='python',
    bash_script=None,
    ):
    
    """
    Count constructs from NGS data based on library design and sample information.
    This function processes NGS data to count occurrences of designed constructs, mapping barcodes
    and calculating quality metrics.
    Parameters
    ----------
    lib_info_df : pandas.DataFrame
        DataFrame containing sample information including fastq file paths and sample metadata
    lib_design_input_df : pandas.DataFrame
        DataFrame containing the library design with barcode sequences.
    use_tRNA : bool, optional (default=False)
        Whether or not to use tRNA sequence as a mapping requirement. 
    iBAR2_UMI : bool, optional (default=False)
        Whether or not iBAR2 is used as UMI, rather than encoding the construct ID.
        If iBAR2_UMI = True, iBAR2 will not be used for mapping and instead, the mean
        number of unique iBAR2 members per construct will be returned a a proxy for
        UMI complexity.
    custom_read_primers : bool, optional (default=True)
        Whether or not custom read primers were used for NGS (can influence amplicon alignment)
    single_fastq : bool, optional (default=False)
        For fastq files that have merged read 1 and read 2 on alternating lines, rather than
        read 1 and read 2 in separate files.
    return_raw_barcodes : bool, optional (default=False)
        Whether to return raw barcode data in addition to counts. Mostly for troubleshooting.
    method : str, optional (default='align')
        Method for extracting barcodes - either by aboslute amplicon 'position' or 'align' against
        a constant region of the sgRNA.
    max_reads : int, optional (default=1e7)
        Maximum number of reads to process. This only exists because the code is slow and 
        1e7 reads is sufficient for most libraries.
    custom_mapping_columns : list, optional (default=None)
        List of columns to use for mapping barcodes. Must uniquely identify constructs.
    backend : str, optional (default='python')
        Execution backend.  'python' uses the pure-Python per-read loop.  'bash' runs
        barcode extraction in awk (much faster for large FASTQ files: awk aggregates
        ~10M reads to ~library_size rows before any data enters Python), then performs
        library mapping and all statistics in Python.  Both backends produce identical
        results and support all arguments including iBAR2_UMI, custom_mapping_columns,
        and return_raw_barcodes.
    bash_script : str, optional (default=None)
        Unused; retained for API compatibility.
    Returns
    -------
    tuple
        If return_raw_barcodes=False:
            - lib_design_counts_df : pandas.DataFrame
                Library design with counts for each sample
            - df_summary : pandas.DataFrame
                Summary statistics for each sample
        If return_raw_barcodes=True (python backend only):
            - lib_design_counts_df : pandas.DataFrame
                Library design with counts for each sample
            - df_summary : pandas.DataFrame
                Summary statistics for each sample
            - df_total : pandas.DataFrame
                Raw barcode data for all samples
    Notes
    -----
    Summary statistics include:
    - Mapping rates for each barcode element
    - Total reads and mapped constructs
    - Fraction of reads mapped and recombined
    - Dropout counts and distribution metrics (Gini coefficient, 90/10 ratio)
    """

    if backend == 'bash':
        return _count_constructs_bash(
            lib_info_df, lib_design_input_df,
            use_tRNA=use_tRNA,
            iBAR2_UMI=iBAR2_UMI,
            custom_read_primers=custom_read_primers,
            single_fastq=single_fastq,
            return_raw_barcodes=return_raw_barcodes,
            method=method,
            max_reads=max_reads,
            custom_mapping_columns=custom_mapping_columns,
        )

    df_total = pd.DataFrame()
    df_summary = pd.DataFrame()
    lib_design_df = generate_unique_construct_identifiers(
        lib_design_input_df, use_tRNA, iBAR2_UMI, custom=custom_mapping_columns)
    
    for index, row in tqdm(lib_info_df.iterrows(), total=lib_info_df.shape[0]):

        if 'dialout' in lib_info_df.columns:
            lib_design_index = {
                k:v for (v,k) in lib_design_df[ lib_design_df['dialout'] == row['dialout'] 
                ]['unique_barcode_combo'].to_dict().items()
            }
        else:
            lib_design_index = {
                k:v for (v,k) in lib_design_df['unique_barcode_combo'].to_dict().items()
            }

        fastq_pair = [row['fastq_R1'], row['fastq_R2']]

        if method == 'position':
            df_barcodes = extract_barcodes_from_fastq_pair(fastq_pair, 
                use_tRNA=use_tRNA, iBAR2_UMI=iBAR2_UMI, 
                custom_read_primers = custom_read_primers,
                single_fastq = single_fastq,
                max_reads=max_reads,
                custom=custom_mapping_columns,
                )
        elif method == 'align':
            df_barcodes = extract_barcodes_from_fastq_pair_align(fastq_pair, 
                            use_tRNA=use_tRNA, iBAR2_UMI=iBAR2_UMI, 
                            single_fastq = single_fastq,
                            max_reads=max_reads,
                            custom=custom_mapping_columns,
                            )
        else:
            warnings.warn("'method' not recognized - will attempt to count constructs by method 'align'")
            df_barcodes = extract_barcodes_from_fastq_pair_align(fastq_pair, 
                use_tRNA=use_tRNA, iBAR2_UMI=iBAR2_UMI, 
                single_fastq = single_fastq,
                max_reads=max_reads,
                custom=custom_mapping_columns,
                )
        
        # map individual elements
        df_barcodes['spacer_1_map'] = df_barcodes['spacer_1'].isin(lib_design_df['spacer_1'])
        df_barcodes['iBAR_1_map'] = df_barcodes['iBAR_1'].isin(lib_design_df['iBAR_1'])
        df_barcodes['spacer_2_map'] = df_barcodes['spacer_2'].isin(lib_design_df['spacer_2'])
        df_barcodes['iBAR_2_map'] = df_barcodes['iBAR_2'].isin(lib_design_df['iBAR_2'])
        df_barcodes['tRNA_map'] = df_barcodes['tRNA'].notna()

        # map full constructs
        df_barcodes['design_index'] = df_barcodes['unique_barcode_combo'].map(lib_design_index)

        df_barcodes['sample']=row['sample_ID']
        df_barcodes['timepoint']=row['timepoint']
        df_barcodes['replicate']=row['replicate']

        if return_raw_barcodes:     
            df_total = pd.concat([df_total,df_barcodes])

        # add oligo counts to library design dataframe
        if index==0:
            lib_design_counts_df = pd.merge(
                lib_design_input_df, 
                pd.Series(df_barcodes['design_index'].value_counts(),
                    name=row['sample_ID']),
                    left_index=True, right_index=True, how='outer').fillna(0)

        else: 
            lib_design_counts_df = pd.merge(
                lib_design_counts_df, 
                pd.Series(df_barcodes['design_index'].value_counts(),
                    name=row['sample_ID']),
                left_index=True, right_index=True, how='outer').fillna(0)

        if iBAR2_UMI:
            lib_design_counts_df = pd.merge(
                lib_design_counts_df, 
                pd.Series(df_barcodes.groupby('design_index')['iBAR_2'].nunique(),
                    name=row['sample_ID']+'_UMI'),
                    left_index=True, right_index=True, how='outer').fillna(0)

        if 'dialout' in lib_info_df.columns:
            sublib_design_counts_df = lib_design_counts_df[lib_design_counts_df['dialout']==row['dialout']]
        else:
            sublib_design_counts_df = lib_design_counts_df

        # record summary statistics
        df_summary.loc[row['sample_ID'], 'timepoint'] = row['timepoint']
        df_summary.loc[row['sample_ID'], 'replicate'] = row['replicate']
        df_summary.loc[row['sample_ID'], 'n_constructs'] = len(lib_design_df)
        df_summary.loc[row['sample_ID'], 'total_read_count'] = len(df_barcodes)
        df_summary.loc[row['sample_ID'], 'NGS_coverage'] = len(df_barcodes)/len(lib_design_df)
        df_barcodes_mapped = df_barcodes[df_barcodes['design_index'].notna()]

        df_summary.loc[row['sample_ID'], 'spacer_1_map'] = df_barcodes['spacer_1_map'].sum()/len(df_barcodes)
        df_summary.loc[row['sample_ID'], 'iBAR_1_map'] = df_barcodes['iBAR_1_map'].sum()/len(df_barcodes)
        df_summary.loc[row['sample_ID'], 'spacer_2_map'] = df_barcodes['spacer_2_map'].sum()/len(df_barcodes)
        df_summary.loc[row['sample_ID'], 'iBAR_2_map'] = df_barcodes['iBAR_2_map'].sum()/len(df_barcodes)
        df_summary.loc[row['sample_ID'], 'tRNA_map'] = df_barcodes['tRNA_map'].sum()/len(df_barcodes)
        
        if iBAR2_UMI:
            df_summary.loc[row['sample_ID'], 'iBAR_2_UMI_mean_count'] = sublib_design_counts_df[row['sample_ID']+'_UMI'].mean()

        if use_tRNA:
            if iBAR2_UMI:
                df_summary.loc[row['sample_ID'], 'all_elements_mapped_count'] =  (
                    df_barcodes['spacer_1_map'] & df_barcodes['iBAR_1_map'] &\
                    df_barcodes['tRNA_map'] &\
                    df_barcodes['spacer_2_map']
                    ).sum()
            else:
                df_summary.loc[row['sample_ID'], 'all_elements_mapped_count'] =  (
                    df_barcodes['spacer_1_map'] & df_barcodes['iBAR_1_map'] &\
                    df_barcodes['tRNA_map'] &\
                    df_barcodes['spacer_2_map'] & df_barcodes['iBAR_2_map']
                    ).sum()
        else:
            if iBAR2_UMI:
                df_summary.loc[row['sample_ID'], 'all_elements_mapped_count'] =  (
                    df_barcodes['spacer_1_map'] & df_barcodes['iBAR_1_map'] &\
                    df_barcodes['spacer_2_map']
                    ).sum()
            else:
                df_summary.loc[row['sample_ID'], 'all_elements_mapped_count'] =  (
                    df_barcodes['spacer_1_map'] & df_barcodes['iBAR_1_map'] &\
                    df_barcodes['spacer_2_map'] & df_barcodes['iBAR_2_map']
                    ).sum()
                
        if custom_mapping_columns is not None:
                mapping_columns = [c+'_map' for c in custom_mapping_columns]
                df_summary.loc[row['sample_ID'], 'all_elements_mapped_count'] = df_barcodes[mapping_columns].all(axis=1).sum()

        df_summary.loc[row['sample_ID'], 'mapped_constructs_count'] = len(df_barcodes_mapped)
        # frequencies of mapping and recombination
        df_summary.loc[row['sample_ID'], 'fraction_mapped'] = df_summary.loc[row['sample_ID'], 'mapped_constructs_count']/\
            df_summary.loc[row['sample_ID'], 'total_read_count']
        df_summary.loc[row['sample_ID'], 'fraction_recombined'] = 1 - \
            df_summary.loc[row['sample_ID'], 'mapped_constructs_count'] / df_summary.loc[row['sample_ID'], 'all_elements_mapped_count']
        # dropout and uniformity
        df_summary.loc[row['sample_ID'], 'dropout_count'] = (sublib_design_counts_df[row['sample_ID']]==0).sum()
        df_summary.loc[row['sample_ID'], 'gini_coefficient'] =  gini(sublib_design_counts_df[row['sample_ID']].values)
        df_summary.loc[row['sample_ID'], 'ratio_90_10'] = ratio_9010(sublib_design_counts_df[row['sample_ID']].values)
    
    df_summary.index.rename('sample_ID', inplace=True)

    if return_raw_barcodes:
        return lib_design_counts_df, df_summary, df_total
    else:
        return lib_design_counts_df, df_summary


def display_summary_table(df):
    """
    Nice display of summary metrics.
    """
    summary_df = df.copy()
    count_columns = ['total_read_count','all_elements_mapped_count', 'mapped_constructs_count', 
                     'n_constructs', 'dropout_count', 'iBAR_2_UMI_mean_count']
    percent_columns = ['spacer_1_map', 'iBAR_1_map', 'spacer_2_map', 'iBAR_2_map', 'fraction_mapped','fraction_recombined']
    decimal_columns = [ 'gini_coefficient', 'ratio_90_10', 'NGS_coverage']
    count_columns = [c for c in count_columns if c in summary_df.columns]
    percent_columns = [c for c in percent_columns if c in summary_df.columns]
    decimal_columns = [c for c in decimal_columns if c in summary_df.columns]

    summary_df[count_columns] = summary_df[count_columns].map(lambda x: "{0:,.0f}".format(x))
    summary_df[percent_columns] = summary_df[percent_columns].map(lambda x: "{0:.02%}".format(x))
    summary_df[decimal_columns] = summary_df[decimal_columns].map(lambda x: "{0:.02f}".format(x))

    return summary_df[count_columns + percent_columns + decimal_columns].T
    

def plot_summary_metrics(summary_df):

    plt.figure(figsize=(1.5*len(summary_df),3))
    df_summary_melt = pd.melt(
        summary_df.reset_index(), 
        id_vars='sample_ID', 
        value_vars=['spacer_1_map','iBAR_1_map','spacer_2_map','iBAR_2_map','fraction_mapped']
        )
    sns.barplot(df_summary_melt, x='sample_ID', y='value', hue='variable', palette='tab20')
    plt.ylabel('fraction of reads')
    plt.ylim(0,1)
    plt.xticks(rotation=90)
    plt.legend(bbox_to_anchor=(1,1))
    plt.show()

    f, (a1, a2, a3) = plt.subplots( 1, 3, figsize=(3*len(summary_df),5))
    plt.subplot(1,3,1)
    sns.barplot(summary_df, ax=a1, x=summary_df.index, y='fraction_recombined')
    plt.ylim(0.001,1)
    plt.yscale('log')
    plt.xticks(rotation=90)

    plt.subplot(1,3,2)
    sns.barplot(summary_df, ax=a2, x=summary_df.index, y='gini_coefficient')
    plt.ylim(0,1)
    plt.xticks(rotation=90)

    plt.subplot(1,3,3)
    sns.barplot(summary_df, ax=a3, x=summary_df.index, y='ratio_90_10')
    a3.set_yscale('log', base=2)
    plt.ylim(2**-0,2**6)
    plt.xticks(rotation=90)
    
    plt.tight_layout(pad=5)
    plt.show()


def plot_lorenz_curves(samples_df, counts_df):
    """
    plot lorenz curves for all samples in summary_df
    """
    for sample in samples_df['sample_ID'].values:
        lorenz_curve(counts_df[sample].values, library_name=sample)
        plt.title(sample)

def plot_uniformity_histograms(samples_df, counts_df):
    """
    plot histogram of counts for all samples in summary_df
    """
    for sample in samples_df['sample_ID'].values:
        plt.figure(figsize=(3,3))
        sns.histplot(
                counts_df[sample], kde=True, 
                kde_kws={'bw_adjust': 2, 'cut': 0},
                color='dodgerblue',
                )
        plt.title(sample)
        plt.xlabel('reads per construct')
        plt.ylabel('constructs')
        plt.show()

def plot_sample_scatterplots(samples_df, counts_df):

    for sample_1, sample_2 in itertools.combinations(samples_df['sample_ID'].values, 2):
        sns.jointplot(
            data = counts_df,
            x=sample_1, 
            y=sample_2,
            color='black', linewidth=0, alpha=0.33, s=5,
            height=3,
            )
        plt.xlim(-10)
        plt.ylim(-10)
        plt.grid(alpha=0.2)
        plt.show()