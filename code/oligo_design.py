import os
import pandas as pd
import numpy as np
import random
from tqdm import tqdm

from constants import *


##################################################################################################
#                                        basic utilities
##################################################################################################

def reverse_complement(seq):
    watson_crick = {'A': 'T',
                'T': 'A',
                'C': 'G',
                'G': 'C',
                'U': 'A',
                'N': 'N'}

    watson_crick.update({k.lower(): v.lower() 
        for k, v in watson_crick.items()})

    return ''.join(watson_crick[x] for x in seq)[::-1]

def calculate_gc(s):
    s = s.upper()
    return (s.count('G') + s.count('C')) / len(s)

def contains_BsmBI(seq):
    return (seq.find('CGTCTC') != -1) | (seq.find('GAGACG') != -1) 

def contains_BbsI(seq):
    return (seq.find('GAAGAC') != -1) | (seq.find('GTCTTC') != -1) 

def count_BsmBI(seq):
    return seq.count('CGTCTC') + seq.count('GAGACG')

def count_BbsI(seq):
    return seq.count('GAAGAC') + seq.count('GTCTTC') 

def contains_RE(seq):
    return contains_BsmBI(seq) | contains_BbsI(seq)

def contains_U6_term(seq):
    return seq.find("TTTT")!=-1

def has_homopolymer(x, n):
    a = 'A'*n in x
    t = 'T'*n in x
    g = 'G'*n in x
    c = 'C'*n in x
    return a | t | g | c

def count_homopolymer(seq,n):
    return seq.count('A'*n) + seq.count('C'*n) + seq.count('G'*n) + seq.count('T'*n)

def fill_degenerate_bases(seq):
    while seq.find("N") != -1:
        n_index = seq.find("N")
        base = np.random.choice(['A','C','T','G'])
        seq = seq[:n_index] + base + seq[n_index+1:]
    return seq

##################################################################################################
#                                   CROPseq-multi one-step cloning
##################################################################################################

# check barcode with iBAR1 flanking sequences for forbidden sequence features
def check_iBAR1(seq):
    seq=seq.upper()
    seq_w_flank = 'GCAGGA'+reverse_complement(seq)+'GACTGCT'
    return not (contains_RE(seq_w_flank) | has_homopolymer(seq_w_flank,4))

# check barcode with iBAR2 flanking sequences for forbidden sequence features
def check_iBAR2(seq):
    seq=seq.upper()
    seq_w_flank = 'GCTGGA'+reverse_complement(seq)+'AACATG'
    return not (contains_RE(seq_w_flank) | has_homopolymer(seq_w_flank,4))

# check candidate spacer 1 with flanking sequences for forbidden sequence features
def check_spacer_1(seq):
    seq=seq.upper()
    seq_w_flank = 'AATGCA'+seq+'GTTTGA'
    return (not contains_RE(seq_w_flank) | contains_U6_term(seq_w_flank))

# check candidatespacer 2 with flanking sequences for forbidden sequence features
def check_spacer_2(seq, tRNA_req = 'any'):
    """
    tRNA_req = 'any' requires that a spacer is compatible with at least one of the tRNAs
    tRNA_req = 'all' requires that a spacer is compatible with all of the tRNAs

    """

    tRNA_flanks = ['CCTCCA','GAGCCC','GAACCT']
    seq=seq.upper()
    seqs_w_flank = [flank+seq+'GTTTCA' for flank in tRNA_flanks]
    checks = [not (contains_RE(seq_w_flank) | contains_U6_term(seq_w_flank)) for seq_w_flank in seqs_w_flank]
    
    if tRNA_req == 'any':
        return max(checks)
    elif tRNA_req == 'all':
        return min(checks)

##################################################################################################
#                                 evaluating decoding performance
##################################################################################################


def determine_minimal_cycle_number(sequence_list, dual=False, directional=False):
    """
    determine minimal cycle number to uniquely identify all sequences in a set
    it is critical that barcodes are provided in the orientation that they are sequenced,
    and not the reverse complement!

    sequence_list: list of strings
    dual: dual barcodes read out simultaneously,
        barcodes of equal length, separated by "_" delimiter
        i.e. for dual iBARs: iBAR1_iBAR2
    directional: whether or not barcodes can be assigned to position 1 and 2
        True if A_B can be distinguished from B_A
        False if A_B is indistinguishable from B_A
        i.e. for dual iBARs, if both iBARs are read out simultaneously without labeling,
        then the mapping of the two barcodes to iBARs 1 and 2 is not necessarily known.
        If one iBAR is labeled (e.g. with a fluorescent oligo), then the mapping is known.
    """
    
    n_seq = len(sequence_list)
    
    if not dual:
        max_cycles = len(sequence_list[0])
        for cycle in range(1,max_cycles+1):
            cycle_seq = [i[:cycle] for i in sequence_list]
            if len(list(set(cycle_seq))) == n_seq:
                return cycle
            
    #dual barcodes, use '_' delimiter to separate barcode pair
    else:
        seq1 = [i.split('_')[0] for i in sequence_list]
        seq2 = [i.split('_')[1] for i in sequence_list]
        max_cycles = len(seq1[0])
        
        if directional:
            for cycle in range(1,max_cycles+1):
                cycle_seq = [i[:cycle]+'_'+j[:cycle] for (i,j) in zip(seq1, seq2)]
                if len(list(set(cycle_seq))) == n_seq:
                    return cycle
        else:
            for cycle in range(1,max_cycles+1):
                cycle_seq = [i[:cycle]+'_'+j[:cycle] for (i,j) in zip(seq1, seq2)]
                cycle_seq = cycle_seq + [j[:cycle]+'_'+i[:cycle] for (i,j) in zip(seq1, seq2)]
                if len(list(set(cycle_seq))) == 2*n_seq:
                    return cycle        
            
    # if barcodes aren't unique, return None
    return None


def test_recombination_detection(
    barcode_pairs,
    directional=True, 
    n_test=5000,
):
    
    n_pairs = len(barcode_pairs)
    barcode_pairs = np.array(barcode_pairs).T
    bc_df = pd.DataFrame(barcode_pairs.T, columns=['bc1','bc2'])

    # record recombination detection rates at each read length
    recomb_detect_rates = {k:0 for k in range(1, len(barcode_pairs[0][0])+1)}

    # randomly select non-paired barcode pairs to simulate recombination
    indices_1 = np.random.choice(range(n_pairs),n_test)
    indices_2 = np.random.choice(range(n_pairs),n_test)
    same_index = np.where(indices_1 == indices_2)
    indices_1 = np.delete(indices_1, same_index)
    indices_2 = np.delete(indices_2, same_index)
    recomb_df = pd.DataFrame(np.array([barcode_pairs[0][indices_1], barcode_pairs[1][indices_2]]).T,
                          columns=['bc1','bc2']
    )

    for seq_len in range(1, len(barcode_pairs[0][0])+1):
    
        if directional:
            bc_df[str(seq_len)] = bc_df['bc1'].str[:seq_len] + "_" + bc_df['bc2'].str[:seq_len]
            
            recomb_df[str(seq_len)] = recomb_df['bc1'].str[:seq_len] + "_" + recomb_df['bc2'].str[:seq_len]
            
            recomb_detect_rates[seq_len] = sum(
                [recomb not in bc_df[str(seq_len)].values for recomb in recomb_df[str(seq_len)].values])/len(recomb_df)
        else:
            bc_df[str(seq_len)+'_1_2'] = bc_df['bc1'].str[:seq_len] + "_" + bc_df['bc2'].str[:seq_len]
            bc_df[str(seq_len)+'_2_1'] = bc_df['bc2'].str[:seq_len] + "_" + bc_df['bc1'].str[:seq_len]
            
            recomb_df[str(seq_len)+'_1_2'] = recomb_df['bc1'].str[:seq_len] + "_" + recomb_df['bc2'].str[:seq_len]
            recomb_df[str(seq_len)+'_2_1'] = recomb_df['bc2'].str[:seq_len] + "_" + recomb_df['bc1'].str[:seq_len]
            
            recomb_detect_rates[seq_len] = sum(
                [recomb not in (bc_df[str(seq_len)+'_1_2'].values+bc_df[str(seq_len)+'_2_1'].values)\
                 for recomb in (recomb_df[str(seq_len)+'_1_2'].values+recomb_df[str(seq_len)+'_2_1'].values)])/len(recomb_df)
      
    return recomb_detect_rates


def check_barcode_pair_cycle_requirements(
    barcode_pair_df,
    barcode_1 = 'iBAR_1', barcode_2 = 'iBAR_2',
):
    
    print('barcode 1 only:\n %s cycles'% 
        determine_minimal_cycle_number(barcode_pair_df[barcode_1].values))
    print('barcode 2 only:\n %s cycles: '% 
        determine_minimal_cycle_number(barcode_pair_df[barcode_2].values))
    print('barcode 1 or 2:\n %s cycles '% 
        determine_minimal_cycle_number(barcode_pair_df[[barcode_1,barcode_2]].values.flatten()))
    
    min_dual_cycle = determine_minimal_cycle_number(barcode_pair_df[barcode_1]+'_'+barcode_pair_df[barcode_2], 
                                         dual=True, directional=False)
    recomb_detection = test_recombination_detection(barcode_pair_df[[barcode_1,barcode_2]],
                                                   directional=False)
    
    print('barcode 1-2 or 2-1:\n %s cycles with %s%% recombination detection'%(
        min_dual_cycle, round(100*recomb_detection[min_dual_cycle])))

    print(' or:\n %s cycles with %s%% recombination detection'%(
        min_dual_cycle+1, round(100*recomb_detection[min_dual_cycle+1])))

    min_dual_cycle = determine_minimal_cycle_number((barcode_pair_df[barcode_1]+'_'+barcode_pair_df[barcode_2]).values, 
                                         dual=True, directional=True)
    recomb_detection = test_recombination_detection((barcode_pair_df[[barcode_1,barcode_2]]).values,
                                                   directional=True)
    
    print('barcode 1-2:\n %s cycles with %s%% recombination detection'%(
        min_dual_cycle, round(100*recomb_detection[min_dual_cycle])))
    print(' or:\n %s cycles with %s%% recombination detection'%(
        min_dual_cycle+1, round(100*recomb_detection[min_dual_cycle+1])))



##################################################################################################
#                               iBAR selection workflows
##################################################################################################


def complete_iBARs_v1(
    df, n_constructs,
    max_it_degenerate_bases = 25,
    max_it_bc_pairing = 1000,
    verbose=1
):
    df_barcodes = df.sample(frac=1).copy() # important to randomize barcode order!
    
    # bring barcodes to length 12 with degenerate bases    
    if len(df_barcodes['barcode'][0])!=12:
        df_barcodes['barcode'] = df_barcodes['barcode'] + 'N'*(12-len(df_barcodes['barcode'][0]))
    
    # maintain homopolymer and GC limits from original design
    homopolymer_max = df_barcodes['homopolymer'][0]
    gc_min = df_barcodes['gc_min'][0]
    gc_max = df_barcodes['gc_max'][0]
    
    iBAR1s_filtered = []
    iBAR2s_filtered = []
    
    if verbose>0:
        print('generating and filtering complete iBARs')

    for bc in tqdm(df_barcodes['barcode'],total=len(df_barcodes)):
        i=0
        while i<max_it_degenerate_bases:
            i+=1
            bc_filled = fill_degenerate_bases(bc)
            if (calculate_gc(bc_filled) > gc_max) or\
                   (calculate_gc(bc_filled) < gc_min) or\
                   has_homopolymer(bc_filled, homopolymer_max):
                continue

            # assign barcodes to position by first base identity 
            # check compatibility with flanking sequences in the appropriate position
            if bc[0] in ['A','G']:
                if (check_iBAR1(bc_filled)):
                    iBAR1s_filtered.append(bc_filled)
                    break
            elif bc[0] in ['C','T']:
                if (check_iBAR2(bc_filled)):
                    iBAR2s_filtered.append(bc_filled)          
                    break
            else:
                continue
    
    min_barcode_count = min([len(iBAR1s_filtered), len(iBAR2s_filtered)])
    if min_barcode_count < n_constructs:
        print('Failed: Only %s barcodes after filtering. At least %s required.'%(min_barcode_count, n_constructs))
        return None
    
    # now optimize the barcode pairing
    min_cycles = []
    position_encoded = True
    df_pairs = pd.DataFrame()
    
    if verbose>0:
        print('optimizing barcode pairing')
    
    for test_seed in tqdm(range(max_it_bc_pairing)):

        rng = np.random.default_rng(seed=test_seed)

        if position_encoded:
            selected_barcodes_1 = rng.choice(iBAR1s_filtered, n_constructs, replace=False)
            selected_barcodes_2 = rng.choice(iBAR2s_filtered, n_constructs, replace=False)
        else:
            barcodes = barcodes_filtered
            selected_barcodes = rng.choice(barcodes, 2*n_constructs, replace=False)
            selected_barcodes_1, selected_barcodes_2 = selected_barcodes[:n_constructs], selected_barcodes[n_constructs:]

        df_pairs[['iBAR_1','iBAR_2']]= np.array([selected_barcodes_1, selected_barcodes_2]).T

        min_cycles.append(np.array([
        determine_minimal_cycle_number(df_pairs['iBAR_1']),
        determine_minimal_cycle_number(df_pairs['iBAR_2']),
        determine_minimal_cycle_number(df_pairs['iBAR_1']+'_'+df_pairs['iBAR_2'], dual=True),
        determine_minimal_cycle_number(df_pairs['iBAR_1']+'_'+df_pairs['iBAR_2'], dual=True, directional=True),
            ]))

    # now go back and select barcodes with best observed combination    
    rng = np.random.default_rng(seed=np.array(min_cycles).sum(axis=1).argmin())

    if position_encoded:
        selected_barcodes_1 = rng.choice(iBAR1s_filtered, n_constructs, replace=False)
        selected_barcodes_2 = rng.choice(iBAR2s_filtered, n_constructs, replace=False)
    else:
        barcodes = barcodes_filtered
        selected_barcodes = rng.choice(barcodes, 2*n_constructs, replace=False)
        selected_barcodes_1, selected_barcodes_2 = selected_barcodes[:n_constructs], selected_barcodes[n_constructs:]

    df_pairs[['iBAR_1','iBAR_2']]= np.array([selected_barcodes_1, selected_barcodes_2]).T

    print('sequencing requirements:')
    check_barcode_pair_cycle_requirements(df_pairs)

    return df_pairs

##################################################################################################
#                             pairing guides from a CRISPick output
##################################################################################################

def pair_guides_single_target_CRISPick(guide_input_df, constructs_per_gene=3):

    guide_pairs_df = pd.DataFrame()

    # check spacer compatibility in each position
    guide_input_df['spacer1_check']= guide_input_df['sgRNA Sequence'].apply(check_spacer_1)
    guide_input_df['spacer2_check']= guide_input_df['sgRNA Sequence'].apply(check_spacer_2)

    for target, target_df in guide_input_df.groupby('Target Gene Symbol'):
        target_df.sort_values('Pick Order')

        spacer_1_canditates = list(target_df[target_df['spacer1_check']]['sgRNA Sequence'].values)
        spacer_2_canditates = list(target_df[target_df['spacer2_check']]['sgRNA Sequence'].values)

        target_guide_pairs = []
        while len(target_guide_pairs) < constructs_per_gene:

            # print a warning if we run out of guides for a target
            if (len(set(spacer_1_canditates+spacer_2_canditates))==1) or\
                (min([len(spacer_1_canditates),len(spacer_1_canditates)])==0):
                print('ran out of guides for target ',target)
                break

            # randomly choose spacer 1 or spacer 2 first
            # to prevent bias in selecting higher ranked guides in either position
            if random.getrandbits(1):
                spacer_1 = spacer_1_canditates[0]
                spacer_1_canditates.remove(spacer_1)
                spacer_2_canditates.remove(spacer_1)
                spacer_2 = spacer_2_canditates[0]
                spacer_1_canditates.remove(spacer_2)
                spacer_2_canditates.remove(spacer_2)
                target_guide_pairs.append((target, spacer_1, spacer_2))
            else:
                spacer_2 = spacer_2_canditates[0]
                spacer_1_canditates.remove(spacer_2)
                spacer_2_canditates.remove(spacer_2)
                spacer_1 = spacer_1_canditates[0]
                spacer_1_canditates.remove(spacer_1)
                spacer_2_canditates.remove(spacer_1)
                target_guide_pairs.append((target, spacer_1, spacer_2))

        guide_pairs_df = pd.concat(
            [guide_pairs_df,
            pd.DataFrame(data=target_guide_pairs, columns=['target','spacer_1','spacer_2'])
            ], ignore_index=True)
        
    return guide_pairs_df

##################################################################################################
#                           build oligos for CROPseq-multi one-step cloning
##################################################################################################

def build_CROPseq_multi_one_step_oligo(
    spacer_1, iBAR_1, spacer_2, iBAR_2,
    template, dialout_fwd, dialout_rev, filler_version=None,
):
    
    def build_oligo(filler_seq):
        return template.format(
            dialout_fwd = dialout_fwd,
            CSM_BsmBI_left = CSM_BsmBI_left,
            spacer_1 = spacer_1,
            CSM_stem_1 = CSM_stem_1,
            iBAR_1 = reverse_complement(iBAR_1), # reverse complement iBARs
            scaffold_tRNA = filler_seq,
            spacer_2 = spacer_2,
            CSM_stem_2 = CSM_stem_2,
            iBAR_2 = reverse_complement(iBAR_2), # reverse complement iBARs
            CSM_BsmBI_right = CSM_BsmBI_right,
            dialout_rev = reverse_complement(dialout_rev),
        ).upper()
    
    # randomly select a tRNA version
    if filler_version is None:
        filler_version = np.random.choice(list(scaffold_tRNA_fillers.keys()))
    filler_seq = scaffold_tRNA_fillers[filler_version]
    
    oligo = build_oligo(filler_seq)

    # check for BsmBI and U6 terminator "TTTT" sites
    if (count_BsmBI(oligo)!=2) | (oligo.find("TTTT")!=-1):
        # try to fix any with alternate tRNA choice
        for filler_v in scaffold_tRNA_fillers.keys():
            filler_version = filler_v
            filler_seq = scaffold_tRNA_fillers[filler_version]
            oligo_candidate = build_oligo(filler_seq)
            if (count_BsmBI(oligo_candidate)==2) &\
                (oligo_candidate.find("TTTT")==-1):
                oligo = oligo_candidate
                break

    return oligo, filler_version

def build_CROPseq_multi_one_step_oligos(
    df_guides_input, 
    spacer_1_col='spacer_1',
    spacer_2_col='spacer_2', 
    ibar1_col='iBAR_1', 
    ibar2_col='iBAR_2', 
    tRNA_col='tRNA', 
    dialout_fwd_col ='dialout_fwd',
    dialout_rev_col ='dialout_rev',
    # column names
    template = template_1_step_oligo
):
    
    df_guides = df_guides_input.copy()

    oligos = []
    filler_versions = []

    for spacer_1, spacer_2, iBAR_1, iBAR_2, tRNA, dialout_fwd, dialout_rev in df_guides[
        [spacer_1_col, spacer_2_col, ibar1_col, ibar2_col, tRNA_col, dialout_fwd_col, dialout_rev_col]].values:

        oligo, filler_version = build_CROPseq_multi_one_step_oligo(
            spacer_1, iBAR_1, spacer_2, iBAR_2, filler_version=tRNA,
            dialout_fwd=dialout_fwd, dialout_rev=dialout_rev,
            template = template_1_step_oligo, 
            )
        
        # final check for BsmBI and U6 terminator sequences
        if (count_BsmBI(oligo)==2) & (oligo.find("TTTT")==-1) & (oligo.find("N")==-1):
            pass
        else:
            errors = []
            if count_BsmBI(oligo)!=2:
                errors.append('%s BsmBI sites'%(count_BsmBI(oligo)!=2))
            if contains_U6_term(oligo):
                errors.append('contains "TTTT"')
            if oligo.find("N")!=-1:
                errors.append('degenerate bases present')
            print('oligo failed with error(s): %s'%(', '.join(errors)), spacer_1, spacer_2, iBAR_1, iBAR_2 )
            oligo='failed'
        if filler_version != tRNA:
            print('changed a tRNA from %s to %s'%(tRNA, filler_version))
        
        oligos.append(oligo)
        filler_versions.append(filler_version)
        
    df_guides['oligo'] = oligos
    df_guides['tRNA'] = filler_versions # because some may be changed

    return df_guides


##################################################################################################
#                           build oligos for CROPseq-multi two-step cloning
##################################################################################################

# # Serial cloning steps with (1) BsmBI and then (2) BbsI assembly
# # Shorter oligos, probably less PCR-mediated recombination, but requires serial cloning

# # We're not currently recommending this alternate library construction approach

# def build_CROPseq_multi_two_step_oligo(
#     spacer_1, iBAR_1, spacer_2, iBAR_2,
#     template, dialout_fwd, dialout_rev, filler_version=None,
# ):
    
#     def build_oligo(filler_seq):
#         return template.format(
#             dialout_fwd = dialout_fwd,
#             CSM_BsmBI_left = CSM_BsmBI_left,
#             spacer_1 = spacer_1,
#             CSM_stem_1 = CSM_stem_1,
#             iBAR_1 = reverse_complement(iBAR_1), # reverse complement iBARs
#             BbsI_filler = filler_seq,
#             spacer_2 = spacer_2,
#             CSM_stem_2 = CSM_stem_2,
#             iBAR_2 = reverse_complement(iBAR_2), # reverse complement iBARs
#             CSM_BsmBI_right = CSM_BsmBI_right,
#             dialout_rev = reverse_complement(dialout_rev),
#         ).upper()
    
#     def fill_degenerate_bases(seq):
#         while seq.find("N") != -1:
#             n_index = seq.find("N")
#             base = np.random.choice(['A','C','T','G'])
#             seq = seq[:n_index] + base + seq[n_index+1:]
#         return seq
    
#     # randomly select a BbsI filler version
#     if filler_version is None:
#         filler_version = np.random.choice(list(BbsI_fillers.keys()))
#     filler_seq = BbsI_fillers[filler_version]
    
#     oligo = build_oligo(filler_seq)
    
#     # check for BsmBI, BbsI, and U6 terminator "TTTT" sites
#     if (count_BsmBI(oligo)!=2) | (count_BbsI(oligo)!=2) | (oligo.find("TTTT")!=-1):
#         oligo_temp = oligo
#         oligo = "failed"
#         # try to fix any with alternate tRNA choice
#         for filler_version in BbsI_fillers.keys():
#             filler_seq = BbsI_fillers[filler_version]
#             oligo_candidate = build_oligo(filler_seq)
#             if (count_BsmBI(oligo_candidate)==2) &\
#                 (count_BbsI(oligo_candidate)==2) &\
#                 (oligo_candidate.find("TTTT")==-1):
#                 print('substituted original tRNA selection')
#                 oligo = oligo_candidate
#                 print(oligo)
#                 break
                
#     # unable to remove BsmBI, BbsI, or U6 terminator "TTTT" motif
#     if oligo == "failed":
#         print('failed to remove BsmBI, BbsI, or U6 terminator')
#         print(oligo_temp)
#         return oligo, filler_version
    
#     # fill any "N" sequences with random choice while avioding homopolymers and RE sites 
#     i=0
#     while oligo.find("N") != -1:
#         if i>50:
#             print('failed to fill degenerate bases')
#             print(oligo)
#             oligo='failed'
#             break
#         i+=1
#         oligo_candidate = fill_degenerate_bases(oligo)
#         if (count_BsmBI(oligo_candidate)==2) & (count_BbsI(oligo_candidate)==2):
#             # require that no new homopolymers are created
#             if count_homopolymer(oligo, 4) == count_homopolymer(oligo_candidate, 4):
#                 oligo = oligo_candidate
#                 break
#             else:
#                     continue
#         else:
#             continue       
    
#     return oligo, filler_version

# def build_CROPseq_multi_two_step_oligos(
#     df_input, 
#     template,
#     spacer_1_column = 'spacer_1',
#     iBar_1_column ='iBAR_1',
#     spacer_2_column = 'spacer_2',
#     iBar_2_column ='iBAR_2',
#     tRNA_column = 'tRNA',
#     dialout_fwd_col ='dialout_fwd',
#     dialout_rev_col ='dialout_rev',
#                          ):
    
#     # what to record:
#     # complete oligo sequence
#     # iBAR 1 and 2 sequences
#     # tRNA version
#     # dialout pair
    
#     df = df_input.copy()

#     oligo_designs = []
#     filler_versions = []
#     for spacer_1, iBAR_1, spacer_2, iBAR_2, tRNA, dialout_fwd, dialout_rev in df[
#         [spacer_1_column, iBar_1_column, 
#          spacer_2_column, iBar_2_column, 
#          tRNA_column, 
#          dialout_fwd_col, dialout_rev_col]].values:
        
#         oligo, filler_version = build_CROPseq_multi_two_step_oligo(
#             spacer_1, iBAR_1, spacer_2, iBAR_2, dialout_fwd, dialout_rev, 
#             filler_version=tRNA,
#             template = template_2_step_oligo)
        
#         oligo_designs.append(oligo)
#         filler_versions.append(filler_version)
#     df['oligo'] = oligo_designs
#     df['tRNA'] = filler_versions
    
#     return df



