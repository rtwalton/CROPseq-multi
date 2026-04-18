#!/usr/bin/env bash
# csm_count_constructs.sh
# Bash/awk implementation of the CROPseq-multi NGS FASTQ processing pipeline.
# Mirrors the behavior of csm/library_NGS_analysis.py: extract_barcodes_from_fastq_pair*
# and count_constructs() for a single sample.
#
# Usage:
#   csm_count_constructs.sh -1 R1.fastq.gz -2 R2.fastq.gz \
#       -l library_design.csv -o output_prefix [options]
#
# Required:
#   -1, --r1 FILE          R1 FASTQ (gzip or plain)
#   -2, --r2 FILE          R2 FASTQ (gzip or plain; same as -1 for --single-fastq)
#   -l, --library FILE     Library design CSV with columns:
#                            spacer_1, iBAR_1, spacer_2, iBAR_2 (at minimum)
#   -o, --output PREFIX    Output prefix; writes:
#                            PREFIX_counts.csv   — per-construct read counts
#                            PREFIX_summary.csv  — mapping quality metrics
#
# Options:
#   -m, --method METHOD    'align' (default) or 'position'
#   -s, --sample-id ID     Sample label in output (default: basename of R1)
#   --max-reads N          Stop after N reads (default: 10000000)
#   --use-tRNA             Include tRNA in the barcode combo key
#   --custom-primers       R1/R2 start with custom read primers (offsets spacer/iBAR
#                          extraction by primer length; only used with --method position)
#   --single-fastq         R1 and R2 are interleaved in a single file; pass the file
#                          as -1 and omit -2
#   -h, --help             Show this help and exit
#
# Output columns (PREFIX_summary.csv):
#   sample_ID, total_read_count, mapped_constructs_count, fraction_mapped,
#   spacer_1_map, iBAR_1_map, spacer_2_map, iBAR_2_map, tRNA_map,
#   all_elements_mapped_count, fraction_recombined, dropout_count,
#   gini_coefficient, ratio_90_10, NGS_coverage, n_constructs
#
# Dependencies: bash >=3.2, awk (BSD or GNU), coreutils (sort, paste, etc.)

set -uo pipefail

# ---------------------------------------------------------------------------
# Constants (from csm/constants.py)
# ---------------------------------------------------------------------------
readonly CSM_STEM_1="GTTTGAGAGCTAAGCAGGA"          # 19 nt; found in R1
readonly CSM_STEM_2="GTTTCAGAGCTATGCTGGA"          # 19 nt; RC is found in R2
readonly CSM_STEM_2_RC="TCCAGCATAGCTCTGAAAC"       # RC(CSM_STEM_2)
readonly READ_1_PRIMER="GTTCGATTCCCGGCCAATGCA"     # 21 nt
readonly READ_2_PRIMER="GCCTTATTTCAACTTGCTATGCTGTT" # 26 nt
readonly SPACER_LEN=20
readonly IBAR_LEN=12
readonly STEM_LEN=19
# tRNA 4-nt prefix → name map (RC of what appears in R2)
# GCCC→tRNA_P, TGCA→tRNA_G, ACCT→tRNA_Q, TCCA→tRNA_A
readonly TRNA_PREFIXES="GCCC:tRNA_P TGCA:tRNA_G ACCT:tRNA_Q TCCA:tRNA_A"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
R1=""
R2=""
LIBRARY_CSV=""
OUTPUT_PREFIX=""
METHOD="align"
SAMPLE_ID=""
MAX_READS=10000000
USE_TRNA=0
CUSTOM_PRIMERS=0
SINGLE_FASTQ=0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
die() { echo "ERROR: $*" >&2; exit 1; }

usage() {
    sed -n '/^# Usage:/,/^$/p' "$0" | sed 's/^# \?//'
    exit 0
}

# Reverse-complement a DNA string (pure bash, used for small sequences in
# the summary step; the hot path uses awk).
rc_bash() {
    echo "$1" | tr 'ACGTacgt' 'TGCAtgca' | rev
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
[[ $# -eq 0 ]] && usage

while [[ $# -gt 0 ]]; do
    case "$1" in
        -1|--r1)            R1="$2";            shift 2 ;;
        -2|--r2)            R2="$2";            shift 2 ;;
        -l|--library)       LIBRARY_CSV="$2";   shift 2 ;;
        -o|--output)        OUTPUT_PREFIX="$2"; shift 2 ;;
        -m|--method)        METHOD="$2";        shift 2 ;;
        -s|--sample-id)     SAMPLE_ID="$2";     shift 2 ;;
        --max-reads)        MAX_READS="$2";     shift 2 ;;
        --use-tRNA)         USE_TRNA=1;         shift ;;
        --custom-primers)   CUSTOM_PRIMERS=1;   shift ;;
        --single-fastq)     SINGLE_FASTQ=1;     shift ;;
        -h|--help)          usage ;;
        *) die "Unknown argument: $1" ;;
    esac
done

# ---------------------------------------------------------------------------
# Validate inputs
# ---------------------------------------------------------------------------
[[ -z "$R1" ]]             && die "-1/--r1 is required"
[[ -z "$LIBRARY_CSV" ]]    && die "-l/--library is required"
[[ -z "$OUTPUT_PREFIX" ]]  && die "-o/--output is required"
[[ -f "$R1" ]]             || die "R1 file not found: $R1"
[[ -f "$LIBRARY_CSV" ]]    || die "Library CSV not found: $LIBRARY_CSV"
[[ "$METHOD" == "align" || "$METHOD" == "position" ]] \
    || die "--method must be 'align' or 'position'"

if [[ "$SINGLE_FASTQ" -eq 1 ]]; then
    R2="$R1"
fi

[[ -z "$R2" ]] && die "-2/--r2 is required (or use --single-fastq)"
[[ -f "$R2" ]] || die "R2 file not found: $R2"

if [[ -z "$SAMPLE_ID" ]]; then
    SAMPLE_ID="$(basename "$R1" | sed 's/_R1.*//; s/\.fastq.*//; s/\.fq.*//')"
fi

# ---------------------------------------------------------------------------
# Step 1: Determine column indices in library CSV
# ---------------------------------------------------------------------------
# Required columns: spacer_1, iBAR_1, spacer_2, iBAR_2
# Optional: tRNA (only needed when --use-tRNA)
HEADER_LINE="$(head -1 "$LIBRARY_CSV")"

col_index() {
    # col_index "header,line" "colname" → 1-based column number
    local header="$1" colname="$2"
    echo "$header" | tr ',' '\n' | grep -n "^${colname}$" | cut -d: -f1
}

COL_SP1=$(col_index "$HEADER_LINE" "spacer_1")
COL_IB1=$(col_index "$HEADER_LINE" "iBAR_1")
COL_SP2=$(col_index "$HEADER_LINE" "spacer_2")
COL_IB2=$(col_index "$HEADER_LINE" "iBAR_2")
COL_TRNA=$(col_index "$HEADER_LINE" "tRNA")

[[ -z "$COL_SP1" ]] && die "Library CSV missing column: spacer_1"
[[ -z "$COL_IB1" ]] && die "Library CSV missing column: iBAR_1"
[[ -z "$COL_SP2" ]] && die "Library CSV missing column: spacer_2"
[[ -z "$COL_IB2" ]] && die "Library CSV missing column: iBAR_2"
[[ "$USE_TRNA" -eq 1 && -z "$COL_TRNA" ]] && die "Library CSV missing column: tRNA (required with --use-tRNA)"

# ---------------------------------------------------------------------------
# Step 2: Open FASTQ files (handle gzip transparently)
# ---------------------------------------------------------------------------
open_fastq() {
    local f="$1"
    case "$f" in
        *.gz|*.gzip) echo "gzip -dc '$f'" ;;
        *)           echo "cat '$f'" ;;
    esac
}

CMD_R1="$(open_fastq "$R1")"
CMD_R2="$(open_fastq "$R2")"

# ---------------------------------------------------------------------------
# Step 3: Extract barcodes with awk, count unique combos
# The awk script writes one tab-separated line per read:
#   spacer_1  iBAR_1  spacer_2  iBAR_2  tRNA_short  combo_key
# Then we sort | uniq -c to count.
# ---------------------------------------------------------------------------

# Build the tRNA prefix map string for awk
TRNA_AWK_MAP=""
for pair in $TRNA_PREFIXES; do
    prefix="${pair%%:*}"
    name="${pair##*:}"
    TRNA_AWK_MAP="${TRNA_AWK_MAP}trna_map[\"${prefix}\"]=\"${name}\"; "
done

AWK_PROGRAM='
BEGIN {
    FS = "\t"
    OFS = "\t"
    # tRNA prefix → name
    '"$TRNA_AWK_MAP"'
    reads_processed = 0
}

# Reverse-complement a DNA string
function rc(seq,    i, c, result, len) {
    result = ""
    len = length(seq)
    for (i = len; i >= 1; i--) {
        c = substr(seq, i, 1)
        if      (c == "A") result = result "T"
        else if (c == "T") result = result "A"
        else if (c == "C") result = result "G"
        else if (c == "G") result = result "C"
        else               result = result "N"
    }
    return result
}

# Safe substring: returns N*len if coordinates are out of range
function safe_sub(seq, start1, len,    s) {
    if (start1 < 1 || start1 + len - 1 > length(seq)) {
        s = ""
        while (length(s) < len) s = s "N"
        return s
    }
    return substr(seq, start1, len)
}

# Repeat char n times (for N padding)
function nchar(c, n,    s) {
    s = ""
    while (length(s) < n) s = s c
    return s
}

{
    # paste output: every 4 lines = one read pair
    # line 1 (NR%4==1): headers
    # line 2 (NR%4==2): sequences  ← we want this
    # line 3 (NR%4==3): "+"
    # line 4 (NR%4==0): quality

    if (NR % 4 != 2) next

    reads_processed++
    if (reads_processed > MAX_READS) exit

    r1 = $1   # R1 sequence (single fastq: alternating read handling done below)
    r2 = $2   # R2 sequence

    if (METHOD == "align") {
        # Find stem_1 in R1 (0-based index = awk_index - 1)
        p1 = index(r1, STEM_1) - 1   # 0-based; -1 means not found

        # Find RC(stem_2) in R2
        p2 = index(r2, STEM_2_RC) - 1

        if (p1 >= 0) {
            # spacer_1: [p1-20 .. p1)  — no RC
            sp1 = safe_sub(r1, p1 - SPACER_LEN + 1, SPACER_LEN)
            # iBAR_1: [p1+19 .. p1+31)  — RC
            ib1 = rc(safe_sub(r1, p1 + STEM_LEN + 1, IBAR_LEN))
        } else {
            sp1 = nchar("N", SPACER_LEN)
            ib1 = nchar("N", IBAR_LEN)
        }

        if (p2 >= 0) {
            # spacer_2: [p2+19 .. p2+39)  — RC
            sp2 = rc(safe_sub(r2, p2 + STEM_LEN + 1, SPACER_LEN))
            # iBAR_2: [p2-12 .. p2)  — no RC
            ib2 = safe_sub(r2, p2 - IBAR_LEN + 1, IBAR_LEN)
            # tRNA 4-nt prefix: [p2+19+20 .. p2+43)  — RC
            trna_short = rc(safe_sub(r2, p2 + STEM_LEN + SPACER_LEN + 1, 4))
        } else {
            sp2 = nchar("N", SPACER_LEN)
            ib2 = nchar("N", IBAR_LEN)
            trna_short = "NNNN"
        }

    } else {
        # METHOD == "position"
        # With custom primers the read starts at position 0 (dialout primer already trimmed).
        # Without custom primers the read starts with the sequencing primer; skip primer length.
        r1_off = (CUSTOM_PRIMERS == 1) ? 0 : R1_PRIMER_LEN   # 0-based
        r2_off = (CUSTOM_PRIMERS == 1) ? 0 : R2_PRIMER_LEN

        # All coords are 0-based; add 1 for awk substr (1-based)
        sp1 = safe_sub(r1, r1_off + 1,                SPACER_LEN)         # [off .. off+20)
        ib1 = rc(safe_sub(r1, r1_off + 39 + 1,        IBAR_LEN))          # [off+39 .. off+51)  RC
        sp2 = rc(safe_sub(r2, r2_off + 31 + 1,        SPACER_LEN))        # [off+31 .. off+51)  RC
        ib2 = safe_sub(r2, r2_off + 1,                IBAR_LEN)           # [off .. off+12)
        trna_short = rc(safe_sub(r2, r2_off + 51 + 1, 4))                  # [off+51 .. off+55)  RC
    }

    # Map tRNA 4-nt prefix to name
    trna_name = (trna_short in trna_map) ? trna_map[trna_short] : "NA"

    # Build unique combo key (matches generate_unique_construct_identifiers logic)
    if (USE_TRNA == 1)
        combo = sp1 "-" ib1 "-" trna_name "-" sp2 "-" ib2
    else
        combo = sp1 "-" ib1 "-" sp2 "-" ib2

    # Output one record per read
    print sp1, ib1, sp2, ib2, trna_short, trna_name, combo
}
'

# Temp files
TMPDIR_WORK="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_WORK"' EXIT

RAW_BARCODES="$TMPDIR_WORK/raw_barcodes.tsv"
COMBO_COUNTS="$TMPDIR_WORK/combo_counts.tsv"

echo "[$SAMPLE_ID] Extracting barcodes (method=$METHOD, max_reads=$MAX_READS)..." >&2

if [[ "$SINGLE_FASTQ" -eq 1 ]]; then
    # Single-fastq mode: R1 and R2 are interleaved (alternating records).
    # Read two consecutive FASTQ records per iteration; assign to R1/R2 by stem presence.
    # We convert to paired-like format for the awk script by interleaving lines.
    eval "$CMD_R1" | awk -v STEM1="$CSM_STEM_1" -v STEM2RC="$CSM_STEM_2_RC" '
    BEGIN { rec = 0 }
    {
        rec++
        line[rec] = $0
        if (rec == 8) {
            # Records 1-4: first fragment; 5-8: second fragment
            seq1 = line[2]; seq2 = line[6]
            # Identify which is R1/R2 by stem presence
            if (index(seq1, STEM1) > 0 || index(seq2, STEM2RC) == 0) {
                r1seq = seq1; r2seq = seq2
                r1hdr = line[1]; r2hdr = line[5]
                r1plus = line[3]; r2plus = line[7]
                r1qual = line[4]; r2qual = line[8]
            } else {
                r1seq = seq2; r2seq = seq1
                r1hdr = line[5]; r2hdr = line[1]
                r1plus = line[7]; r2plus = line[3]
                r1qual = line[8]; r2qual = line[4]
            }
            print r1hdr "\t" r2hdr
            print r1seq "\t" r2seq
            print r1plus "\t" r2plus
            print r1qual "\t" r2qual
            rec = 0
        }
    }' | awk \
        -v METHOD="$METHOD" \
        -v STEM_1="$CSM_STEM_1" \
        -v STEM_2_RC="$CSM_STEM_2_RC" \
        -v SPACER_LEN="$SPACER_LEN" \
        -v IBAR_LEN="$IBAR_LEN" \
        -v STEM_LEN="$STEM_LEN" \
        -v R1_PRIMER_LEN="${#READ_1_PRIMER}" \
        -v R2_PRIMER_LEN="${#READ_2_PRIMER}" \
        -v CUSTOM_PRIMERS="$CUSTOM_PRIMERS" \
        -v USE_TRNA="$USE_TRNA" \
        -v MAX_READS="$MAX_READS" \
        "$AWK_PROGRAM" > "$RAW_BARCODES"
else
    # Standard paired-end: paste the two streams so each line has R1\tR2
    paste <(eval "$CMD_R1") <(eval "$CMD_R2") | awk \
        -v METHOD="$METHOD" \
        -v STEM_1="$CSM_STEM_1" \
        -v STEM_2_RC="$CSM_STEM_2_RC" \
        -v SPACER_LEN="$SPACER_LEN" \
        -v IBAR_LEN="$IBAR_LEN" \
        -v STEM_LEN="$STEM_LEN" \
        -v R1_PRIMER_LEN="${#READ_1_PRIMER}" \
        -v R2_PRIMER_LEN="${#READ_2_PRIMER}" \
        -v CUSTOM_PRIMERS="$CUSTOM_PRIMERS" \
        -v USE_TRNA="$USE_TRNA" \
        -v MAX_READS="$MAX_READS" \
        "$AWK_PROGRAM" > "$RAW_BARCODES"
fi

echo "[$SAMPLE_ID] Counting unique barcode combinations..." >&2
# Sort and count: output is "count combo sp1 ib1 sp2 ib2 tshort tname"
# We keep all 7 fields from the raw file and count by the combo key (field 7)
sort -t$'\t' -k7,7 "$RAW_BARCODES" | awk -F'\t' '
BEGIN { OFS="\t"; prev=""; cnt=0 }
{
    if ($7 == prev) {
        cnt++
    } else {
        if (prev != "") print prev, cnt, sp1, ib1, sp2, ib2, tshort, tname
        prev=$7; cnt=1; sp1=$1; ib1=$2; sp2=$3; ib2=$4; tshort=$5; tname=$6
    }
}
END { if (prev != "") print prev, cnt, sp1, ib1, sp2, ib2, tshort, tname }
' > "$COMBO_COUNTS"
# COMBO_COUNTS columns (tab-sep):
#   1:combo  2:count  3:sp1  4:ib1  5:sp2  6:ib2  7:tshort  8:tname

TOTAL_READS=$(wc -l < "$RAW_BARCODES" | tr -d ' ')
echo "[$SAMPLE_ID] Total reads processed: $TOTAL_READS" >&2

# ---------------------------------------------------------------------------
# Step 4: Load library design; build valid-element sets and combo→row map
# ---------------------------------------------------------------------------
echo "[$SAMPLE_ID] Mapping to library design..." >&2

LIB_COMBOS="$TMPDIR_WORK/lib_combos.tsv"
LIB_ELEMENTS="$TMPDIR_WORK/lib_elements.tsv"

# Extract combo keys and individual elements from library CSV
# Output: combo_key, row_index (1-based, skipping header)
awk -F',' -v sp1c="$COL_SP1" -v ib1c="$COL_IB1" \
           -v sp2c="$COL_SP2" -v ib2c="$COL_IB2" \
           -v trnac="${COL_TRNA:-0}" \
           -v use_trna="$USE_TRNA" \
'NR==1 { next }
{
    sp1 = $sp1c; ib1 = $ib1c; sp2 = $sp2c; ib2 = $ib2c
    # Remove any surrounding whitespace or quotes
    gsub(/^[ \t"]+|[ \t"]+$/, "", sp1)
    gsub(/^[ \t"]+|[ \t"]+$/, "", ib1)
    gsub(/^[ \t"]+|[ \t"]+$/, "", sp2)
    gsub(/^[ \t"]+|[ \t"]+$/, "", ib2)
    if (use_trna == 1 && trnac > 0) {
        trna = $trnac
        gsub(/^[ \t"]+|[ \t"]+$/, "", trna)
        combo = sp1 "-" ib1 "-" trna "-" sp2 "-" ib2
    } else {
        combo = sp1 "-" ib1 "-" sp2 "-" ib2
    }
    print combo "\t" (NR-1)
    # also print individual elements for element-level mapping stats
    print "SP1\t" sp1 > "/dev/stderr"
    print "IB1\t" ib1 > "/dev/stderr"
    print "SP2\t" sp2 > "/dev/stderr"
    print "IB2\t" ib2 > "/dev/stderr"
}' "$LIBRARY_CSV" > "$LIB_COMBOS" 2>"$LIB_ELEMENTS"

N_CONSTRUCTS=$(wc -l < "$LIB_COMBOS" | tr -d ' ')

# ---------------------------------------------------------------------------
# Step 5: Count per-construct reads, compute per-element mapping flags
# ---------------------------------------------------------------------------
MAPPED_READS_FILE="$TMPDIR_WORK/mapped_reads.tsv"

awk -F'\t' \
    -v lib_combos_file="$LIB_COMBOS" \
    -v lib_elements_file="$LIB_ELEMENTS" \
'BEGIN {
    # Load library combos: combo → row_index
    while ((getline line < lib_combos_file) > 0) {
        n = split(line, a, "\t")
        lib_combo[a[1]] = a[2] + 0
    }
    # Load individual valid elements
    while ((getline line < lib_elements_file) > 0) {
        n = split(line, a, "\t")
        if (a[1] == "SP1") valid_sp1[a[2]] = 1
        else if (a[1] == "IB1") valid_ib1[a[2]] = 1
        else if (a[1] == "SP2") valid_sp2[a[2]] = 1
        else if (a[1] == "IB2") valid_ib2[a[2]] = 1
    }
}
{
    combo=$1; cnt=$2+0; sp1=$3; ib1=$4; sp2=$5; ib2=$6; tname=$8

    m_sp1 = (sp1 in valid_sp1) ? 1 : 0
    m_ib1 = (ib1 in valid_ib1) ? 1 : 0
    m_sp2 = (sp2 in valid_sp2) ? 1 : 0
    m_ib2 = (ib2 in valid_ib2) ? 1 : 0
    m_trna = (tname != "NA") ? 1 : 0
    all_map = (m_sp1 && m_ib1 && m_sp2 && m_ib2) ? 1 : 0

    row_idx = (combo in lib_combo) ? lib_combo[combo] : 0

    print row_idx, cnt, m_sp1, m_ib1, m_sp2, m_ib2, m_trna, all_map
}' OFS='\t' "$COMBO_COUNTS" > "$MAPPED_READS_FILE"
# MAPPED_READS_FILE columns (tab-sep):
#   1:row_idx(0=unmapped)  2:count  3:m_sp1  4:m_ib1  5:m_sp2  6:m_ib2
#   7:m_trna  8:all_map

# ---------------------------------------------------------------------------
# Step 6: Aggregate counts per construct row and compute summary stats
# ---------------------------------------------------------------------------
COUNTS_BY_ROW="$TMPDIR_WORK/counts_by_row.tsv"

# Sum counts by row_idx; also accumulate summary stats
awk -F'\t' -v total_reads="$TOTAL_READS" -v n_constructs="$N_CONSTRUCTS" \
'BEGIN {
    total=0; mapped_constructs=0
    sum_sp1=0; sum_ib1=0; sum_sp2=0; sum_ib2=0; sum_trna=0; sum_all=0
}
{
    row_idx=$1+0; cnt=$2+0
    total += cnt
    sum_sp1  += cnt * $3
    sum_ib1  += cnt * $4
    sum_sp2  += cnt * $5
    sum_ib2  += cnt * $6
    sum_trna += cnt * $7
    sum_all  += cnt * $8

    if (row_idx > 0) {
        row_count[row_idx] += cnt
        mapped_constructs  += cnt
    }
}
END {
    # Print per-row counts (row_idx → count); 0 for missing rows handled later
    for (r in row_count) print r "\t" row_count[r]
    # Print summary stats to stderr for downstream capture
    print "TOTAL\t"       total        > "/dev/stderr"
    print "MAPPED\t"      mapped_constructs > "/dev/stderr"
    print "SUM_SP1\t"     sum_sp1      > "/dev/stderr"
    print "SUM_IB1\t"     sum_ib1      > "/dev/stderr"
    print "SUM_SP2\t"     sum_sp2      > "/dev/stderr"
    print "SUM_IB2\t"     sum_ib2      > "/dev/stderr"
    print "SUM_TRNA\t"    sum_trna     > "/dev/stderr"
    print "SUM_ALL\t"     sum_all      > "/dev/stderr"
}' "$MAPPED_READS_FILE" > "$COUNTS_BY_ROW" 2>"$TMPDIR_WORK/agg_stats.tsv"

# ---------------------------------------------------------------------------
# Step 7: Join counts back to library design CSV; write PREFIX_counts.csv
# ---------------------------------------------------------------------------
echo "[$SAMPLE_ID] Writing counts CSV..." >&2

COUNTS_OUTPUT="${OUTPUT_PREFIX}_counts.csv"

awk -F',' -v counts_file="$COUNTS_BY_ROW" -v sample_id="$SAMPLE_ID" \
'BEGIN {
    while ((getline line < counts_file) > 0) {
        split(line, a, "\t")
        row_count[a[1]+0] = a[2]+0
    }
    OFS=","
}
NR==1 { print $0, sample_id; next }
{
    idx = NR - 1   # 1-based row index (matches row_idx in awk above)
    cnt = (idx in row_count) ? row_count[idx] : 0
    print $0, cnt
}' "$LIBRARY_CSV" > "$COUNTS_OUTPUT"

echo "[$SAMPLE_ID] Counts written to: $COUNTS_OUTPUT" >&2

# ---------------------------------------------------------------------------
# Step 8: Compute summary statistics
# ---------------------------------------------------------------------------
echo "[$SAMPLE_ID] Computing summary statistics..." >&2

# Read aggregated stats (bash 3.2 compatible — no associative arrays)
_lookup_stat() { awk -F'\t' -v k="$1" '$1==k{print $2}' "$TMPDIR_WORK/agg_stats.tsv"; }

TOTAL=$(   _lookup_stat TOTAL);    TOTAL="${TOTAL:-0}"
MAPPED=$(  _lookup_stat MAPPED);   MAPPED="${MAPPED:-0}"
SUM_SP1=$( _lookup_stat SUM_SP1);  SUM_SP1="${SUM_SP1:-0}"
SUM_IB1=$( _lookup_stat SUM_IB1);  SUM_IB1="${SUM_IB1:-0}"
SUM_SP2=$( _lookup_stat SUM_SP2);  SUM_SP2="${SUM_SP2:-0}"
SUM_IB2=$( _lookup_stat SUM_IB2);  SUM_IB2="${SUM_IB2:-0}"
SUM_TRNA=$(  _lookup_stat SUM_TRNA);  SUM_TRNA="${SUM_TRNA:-0}"
SUM_ALL=$( _lookup_stat SUM_ALL);  SUM_ALL="${SUM_ALL:-0}"

# Compute Gini coefficient, 90/10 ratio, and dropout using awk on the per-row counts
UNIFORMITY_STATS="$TMPDIR_WORK/uniformity_stats.tsv"

awk -v n_constructs="$N_CONSTRUCTS" \
'BEGIN {
    for (i = 1; i <= n_constructs; i++) counts[i] = 0
}
{ counts[$1+0] = $2+0 }
END {
    # Build sorted array
    n = n_constructs
    for (i = 1; i <= n; i++) arr[i] = counts[i]

    # Bubble sort (fine for library sizes up to ~10k constructs)
    for (i = 1; i <= n; i++) {
        for (j = 1; j <= n - i; j++) {
            if (arr[j] > arr[j+1]) {
                tmp = arr[j]; arr[j] = arr[j+1]; arr[j+1] = tmp
            }
        }
    }

    # Gini coefficient
    total = 0; weighted = 0
    for (i = 1; i <= n; i++) { total += arr[i]; weighted += i * arr[i] }
    coef = (total > 0) ? (2.0 * weighted / (n * total) - (n + 1.0) / n) : 0

    # 90/10 ratio — mirrors Python: sorted_arr[int(n/10)] / sorted_arr[-int(n/10)]
    # Python uses 0-based indexing; awk array is 1-based, so add 1 for p10
    p10_n = int(n / 10)
    if (p10_n < 1) p10_n = 1
    p10_idx = p10_n + 1              # 0-based p10_n → 1-based p10_n+1
    p90_idx = n - p10_n + 1          # 0-based (n-p10_n) → 1-based (n-p10_n+1)
    if (p10_idx > n) p10_idx = n
    if (p90_idx > n) p90_idx = n
    p10_val = arr[p10_idx]
    p90_val = arr[p90_idx]
    ratio = (p10_val > 0) ? (p90_val / p10_val) : 0

    # Dropout (zero-count constructs)
    dropout = 0
    for (i = 1; i <= n; i++) if (arr[i] == 0) dropout++

    print "GINI\t"    coef
    print "RATIO9010\t" ratio
    print "DROPOUT\t" dropout
}' "$COUNTS_BY_ROW" > "$UNIFORMITY_STATS"

GINI=$(    awk -F'\t' '$1=="GINI"{print $2}'     "$UNIFORMITY_STATS"); GINI="${GINI:-0}"
RATIO9010=$(awk -F'\t' '$1=="RATIO9010"{print $2}' "$UNIFORMITY_STATS"); RATIO9010="${RATIO9010:-0}"
DROPOUT=$( awk -F'\t' '$1=="DROPOUT"{print $2}'  "$UNIFORMITY_STATS"); DROPOUT="${DROPOUT:-0}"

# Derived fractions (use awk for floating point)
read FRAC_MAPPED FRAC_RECOMBINED FRAC_SP1 FRAC_IB1 FRAC_SP2 FRAC_IB2 FRAC_TRNA NGS_COV < <(
    awk -v total="$TOTAL" -v mapped="$MAPPED" \
         -v sum_sp1="$SUM_SP1" -v sum_ib1="$SUM_IB1" \
         -v sum_sp2="$SUM_SP2" -v sum_ib2="$SUM_IB2" \
         -v sum_trna="$SUM_TRNA" -v sum_all="$SUM_ALL" \
         -v n_constructs="$N_CONSTRUCTS" \
    'BEGIN {
        frac_mapped     = (total > 0) ? mapped / total : 0
        frac_recombined = (sum_all > 0) ? 1 - mapped / sum_all : 0
        frac_sp1        = (total > 0) ? sum_sp1 / total : 0
        frac_ib1        = (total > 0) ? sum_ib1 / total : 0
        frac_sp2        = (total > 0) ? sum_sp2 / total : 0
        frac_ib2        = (total > 0) ? sum_ib2 / total : 0
        frac_trna       = (total > 0) ? sum_trna / total : 0
        ngs_cov         = (n_constructs > 0) ? total / n_constructs : 0
        printf "%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
            frac_mapped, frac_recombined,
            frac_sp1, frac_ib1, frac_sp2, frac_ib2, frac_trna, ngs_cov
    }'
)

# ---------------------------------------------------------------------------
# Step 9: Write summary CSV
# ---------------------------------------------------------------------------
SUMMARY_OUTPUT="${OUTPUT_PREFIX}_summary.csv"

{
    echo "sample_ID,total_read_count,mapped_constructs_count,fraction_mapped,\
spacer_1_map,iBAR_1_map,spacer_2_map,iBAR_2_map,tRNA_map,\
all_elements_mapped_count,fraction_recombined,dropout_count,\
gini_coefficient,ratio_90_10,NGS_coverage,n_constructs"

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$SAMPLE_ID" \
        "$TOTAL" \
        "$MAPPED" \
        "$FRAC_MAPPED" \
        "$FRAC_SP1" \
        "$FRAC_IB1" \
        "$FRAC_SP2" \
        "$FRAC_IB2" \
        "$FRAC_TRNA" \
        "$SUM_ALL" \
        "$FRAC_RECOMBINED" \
        "$DROPOUT" \
        "$GINI" \
        "$RATIO9010" \
        "$NGS_COV" \
        "$N_CONSTRUCTS"
} > "$SUMMARY_OUTPUT"

echo "[$SAMPLE_ID] Summary written to: $SUMMARY_OUTPUT" >&2

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo "[$SAMPLE_ID] Done." >&2
echo "  Counts: $COUNTS_OUTPUT" >&2
echo "  Summary: $SUMMARY_OUTPUT" >&2
