

# CROPseq-multi one-step cloning

CSM_BsmBI_left = "CGTCTCAATGCA"
CSM_stem_1 = "GTTTGAGAGCTAAGCAGGA"
CSM_stem_2 = "GTTTCAGAGCTATGCTGGA"
CSM_BsmBI_right = "AACATGAGACG"
CSM_tracr_1 = "GACTGCTTAGCAAGTTCAAATAAGGCTGGTCCGTACACAACTTGGAAAAGTGGCAGCCGAGTCGGCTGC"
CSM_tRNA_leader_default = "AACAAA"

tRNA_A_seq = "GGGGGTATAGCTCAGTGGTAGAGCGCGTGCTTAGCATGCACGAGGTCCTGGGTTCGATCCCCAGTACCTCCA"
tRNA_P_seq = "GGCTCGTTGGTCTAGGGGTATGATTCTCGCTTAGGGTGCGAGAGGTCCCGGGTTCAAATCCCGGACGAGCCC"
tRNA_Q_seq = "GGTTCCATGGTGTAATGGTTAGCACTCTGGACTCTGAATCCAGCGATCCGAGTTCAAATCTCGGTGGAACCT"
tRNA_seqs = {'tRNA_A':tRNA_A_seq, 'tRNA_P':tRNA_P_seq, 'tRNA_Q':tRNA_Q_seq}

template_1_step_oligo = "{dialout_fwd}{CSM_BsmBI_left}{spacer_1}{CSM_stem_1}{iBAR_1}{tracr_1}{tRNA_leader}{tRNA}{spacer_2}{CSM_stem_2}{iBAR_2}{CSM_BsmBI_right}{dialout_rev}"

template_1_step_oligo_single_guide = "{dialout_fwd}{CSM_BsmBI_left}{spacer}{CSM_stem_2}{iBAR}{CSM_BsmBI_right}{dialout_rev}"

read_1_primer_seq = "GTTCGATTCCCGGCCAATGCA"
read_2_primer_seq = "GCCTTATTTCAACTTGCTATGCTGTT"

# CROPseq-multi two-step cloning

BbsI_filler_tRNA_A = "GACTGCNNGTCTTCNNNNNNNNNNGAAGACNNTCCA"
BbsI_filler_tRNA_P = "GACTNNGTCTTCNNNNNNNNNNGAAGACNNGAGCCC"
BbsI_filler_tRNA_Q = "GACTGNNGTCTTCNNNNNNNNNNGAAGACNNAACCT"
BbsI_fillers = {'tRNA_A':BbsI_filler_tRNA_A, 'tRNA_P':BbsI_filler_tRNA_P, 'tRNA_Q':BbsI_filler_tRNA_Q}

template_2_step_oligo = "{dialout_fwd}{CSM_BsmBI_left}{spacer_1}{CSM_stem_1}{iBAR_1}{BbsI_filler}{spacer_2}{CSM_stem_2}{iBAR_2}{CSM_BsmBI_right}{dialout_rev}"
