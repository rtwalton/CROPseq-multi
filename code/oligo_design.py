import os
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import warnings

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

    tRNA_flanks = [seq[-6:] for seq in tRNA_seqs.values()] #['CCTCCA','GAGCCC','GAACCT']
    seq=seq.upper()
    seqs_w_flank = [flank+seq+'GTTTCA' for flank in tRNA_flanks]
    checks = [not (contains_RE(seq_w_flank) | contains_U6_term(seq_w_flank)) for seq_w_flank in seqs_w_flank]
    
    if tRNA_req == 'any':
        return max(checks)
    elif tRNA_req == 'all':
        return min(checks)


def check_tRNA_leader(seq, tRNA_req = 'all'):
    """
    tRNA_req = 'any' requires compatibility with at least one of the tRNAs
    tRNA_req = 'all' requires compatibility with all of the tRNAs

    """

    tRNA_flanks = [seq[:6] for seq in tRNA_seqs.values()]
    seq=seq.upper()
    seqs_w_flank = ['GGCTGC'+seq+flank for flank in tRNA_flanks]
    checks = [not (contains_BsmBI(seq_w_flank) | contains_U6_term(seq_w_flank)) for seq_w_flank in seqs_w_flank]
    
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
    
    barcode_1_cycles = determine_minimal_cycle_number(barcode_pair_df[barcode_1].values)
    barcode_2_cycles = determine_minimal_cycle_number(barcode_pair_df[barcode_2].values)
    barcode_1_or_2_cycles = determine_minimal_cycle_number(barcode_pair_df[[barcode_1,barcode_2]].values.flatten())

    if barcode_1_cycles is None:
        print('barcode 1 is not unique in set(barcode 1)')
    else:
        print('barcode 1 is unique in set(barcode 1) in:\n %s cycles'% barcode_1_cycles)
    
    if barcode_2_cycles is None:
        print('barcode 2 is not unique in set(barcode 2)')
    else:
        print('barcode 2 is unique in set(barcode 2) in:\n %s cycles'% barcode_2_cycles)

    if barcode_1_or_2_cycles is None:
        print('barcodes in set(barcode 1, barcode 2) are not unique')
    else:
        print('barcodes in set(barcode 1, barcode 2) are unique in:\n %s cycles'% barcode_1_or_2_cycles)
    
    min_dual_cycle = determine_minimal_cycle_number(barcode_pair_df[barcode_1]+'_'+barcode_pair_df[barcode_2], 
                                         dual=True, directional=False)
    recomb_detection = test_recombination_detection(barcode_pair_df[[barcode_1,barcode_2]],
                                                   directional=False)

    if min_dual_cycle is None:
        print('barcode pair is not unique in set(1-2, 2-1)')
    else:
        print('barcode pair is unique in set(1-2, 2-1) in :\n %s cycles with %s%% recombination detection'%(
            min_dual_cycle, round(100*recomb_detection[min_dual_cycle])))
        if round(100*recomb_detection[min_dual_cycle]) < 100:
            print(' or:\n %s cycles with %s%% recombination detection'%(
                min_dual_cycle+1, round(100*recomb_detection[min_dual_cycle+1])))

    min_dual_cycle = determine_minimal_cycle_number((barcode_pair_df[barcode_1]+'_'+barcode_pair_df[barcode_2]).values, 
                                         dual=True, directional=True)
    recomb_detection = test_recombination_detection((barcode_pair_df[[barcode_1,barcode_2]]).values,
                                                   directional=True)
    

    if min_dual_cycle is None:
        print('barcode pair is not unique in set(1-2)')
    else:
        print('barcode pair is unique in set(1-2) in:\n %s cycles with %s%% recombination detection'%(
            min_dual_cycle, round(100*recomb_detection[min_dual_cycle])))
        if round(100*recomb_detection[min_dual_cycle]) < 100:
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

def filter_guides(
    guides_df,
    position_req='any',tRNA_req='any'
    ):

    """
    filter spacers for compatibility in positions of the vector
    expects CRISPick output as input format

    """
    
    # check spacer compatibility in each position
    guides_df['spacer1_check']= guides_df['sgRNA Sequence'].apply(check_spacer_1)
    guides_df['spacer2_check']= guides_df['sgRNA Sequence'].apply(check_spacer_2, tRNA_req=tRNA_req)

    # if spacers must work in all positions, filter out candidates that are one or the other
    if position_req =='all':
        guides_df = guides_df[guides_df['spacer1_check'] & guides_df['spacer2_check']]
    elif position_req =='any':
        guides_df = guides_df[guides_df['spacer1_check'] | guides_df['spacer2_check']]

    return guides_df

def pair_guides_single_target_CRISPick(
    guide_input_df, constructs_per_gene=3, position_req='any',tRNA_req='any'
    ):
    """
    Use CRISPick output to pair two guides targeting the same gene in one construct
    """

    guide_pairs_df = pd.DataFrame()

    # check spacer compatibility in each position
    guide_input_df['spacer1_check']= guide_input_df['sgRNA Sequence'].apply(check_spacer_1)
    guide_input_df['spacer2_check']= guide_input_df['sgRNA Sequence'].apply(check_spacer_2, tRNA_req=tRNA_req)

    # if spacers must work in all positions, filter out candidates that are one or the other
    if position_req =='all':
        guide_input_df = guide_input_df[guide_input_df['spacer1_check'] & guide_input_df['spacer2_check']]

    for target, target_df in guide_input_df.groupby('Target Gene Symbol'):
        # sort guides by pick order, ascending
        target_df.sort_values('Pick Order', inplace=True)

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
            ])

    guide_pairs_df['target_version'] = guide_pairs_df.index.astype(int) + 1
    guide_pairs_df.reset_index(inplace=True, drop=True)

    return guide_pairs_df


def pair_guides_target_and_control_CRISPick(
    guide_input_df, 
    control_guides_df,
    constructs_per_gene=4,
    tRNA_req='all',
    rand_seed=0,
    ):

    """
    pair gene targeting guides with controls (e.g. nontargeting, intergenic, olfactory receptors)
    guide input should be CRISPick output format
    """
        
    rng = np.random.default_rng(seed=rand_seed)
    
    guide_pairs_df = pd.DataFrame()

    # check spacer compatibility in each position
    guide_input_df['spacer1_check'] = guide_input_df['sgRNA Sequence'].apply(check_spacer_1)
    guide_input_df['spacer2_check'] = guide_input_df['sgRNA Sequence'].apply(check_spacer_2, tRNA_req=tRNA_req)
    control_guides_df['spacer1_check']= control_guides_df['sgRNA Sequence'].apply(check_spacer_1)
    control_guides_df['spacer2_check']= control_guides_df['sgRNA Sequence'].apply(check_spacer_2, tRNA_req=tRNA_req)    

    # require that spacers work in all positions since we're not deciding position yet
    guide_input_df = guide_input_df[guide_input_df['spacer1_check'] & guide_input_df['spacer2_check']]
    control_guides_df = control_guides_df[control_guides_df['spacer1_check'] & control_guides_df['spacer2_check']]

    control_candidates = list(control_guides_df['sgRNA Sequence'].values)

    for target, target_gene_df in guide_input_df.groupby('Target Gene Symbol'):
        target_gene_df.sort_values('Pick Order', inplace=True)

        gene_targeting_guides = list(target_gene_df['sgRNA Sequence'].values)[:constructs_per_gene]
        # print a warning if we run out of guides for a target
        if len(gene_targeting_guides)<constructs_per_gene:
            print('not enough compatible guides for target ',target)
            continue
            
        control_guides = rng.choice(control_candidates, constructs_per_gene)
        
        df_target = pd.DataFrame()
        df_target['target'] = [target]*constructs_per_gene
        df_target['targeting_spacer'] = gene_targeting_guides
        df_target['control_spacer'] = control_guides

        # alternate targeting position, randomly starting in either position
        j = rng.choice([1,2])
        df_target['target_spacer_pos'] = [(j+i)%2+1 for i in range(len(df_target))]
        df_target['spacer_1'] = pd.Series(dtype='str')
        df_target['spacer_2'] = pd.Series(dtype='str')
        # assign spacer 1
        df_target.loc[df_target.target_spacer_pos==1,'spacer_1'] = df_target.loc[
        df_target.target_spacer_pos==1, 'targeting_spacer']
        df_target.loc[df_target.target_spacer_pos==2,'spacer_1'] = df_target.loc[
        df_target.target_spacer_pos==2, 'control_spacer']
        # assign spacer 2
        df_target.loc[df_target.target_spacer_pos==1,'spacer_2'] = df_target.loc[
        df_target.target_spacer_pos==1, 'control_spacer']
        df_target.loc[df_target.target_spacer_pos==2,'spacer_2'] = df_target.loc[
        df_target.target_spacer_pos==2, 'targeting_spacer']

        guide_pairs_df = pd.concat([guide_pairs_df, df_target])

    guide_pairs_df['target_version'] = guide_pairs_df.index.astype(int) + 1
    guide_pairs_df.reset_index(inplace=True, drop=True)

    return guide_pairs_df    


def pair_guides_for_GI_CRISPick(
    gene_pairs, # list of gene pairs
    GI_input_df, # combinations to design
    guide_input_df, # CRISPick design file with all genes for 
    control_guides_df, # CRISPick design file with control guides to pull from
    tRNA_req='all',
    rand_seed=0,
    ):
    
    rng = np.random.default_rng(seed=rand_seed)
    
    guide_pairs_df = pd.DataFrame()
    
    # some formatting of GI set dataframe
    # GI_sets_df = GI_input_df.copy()
    GI_input_df['GI_version'] = (GI_input_df['target_pos1']+'_'+GI_input_df['target_pos2']).map({
        'A_B':'AB','B_A':'AB','A_ctrl':'A','ctrl_A':'A','B_ctrl':'B','ctrl_B':'B'})
    # determine number of unique guides needed per target
    n_A_guides = len(set(list(GI_input_df[(GI_input_df['target_pos1']=='A')]['spacer_version_pos1'].unique()) +\
                         list(GI_input_df[(GI_input_df['target_pos2']=='A')]['spacer_version_pos1'].unique())))
    n_B_guides = len(set(list(GI_input_df[(GI_input_df['target_pos1']=='B')]['spacer_version_pos1'].unique()) +\
                         list(GI_input_df[(GI_input_df['target_pos2']=='B')]['spacer_version_pos1'].unique())))
    n_ctrl_guides = len(set(list(GI_input_df[(GI_input_df['target_pos1']=='ctrl')]['spacer_version_pos1'].unique()) +\
                            list(GI_input_df[(GI_input_df['target_pos2']=='ctrl')]['spacer_version_pos1'].unique())))
    
    # check spacer compatibility in each position
    guide_input_df['spacer1_check'] = guide_input_df['sgRNA Sequence'].apply(check_spacer_1)
    guide_input_df['spacer2_check'] = guide_input_df['sgRNA Sequence'].apply(check_spacer_2, tRNA_req=tRNA_req)
    control_guides_df['spacer1_check'] = control_guides_df['sgRNA Sequence'].apply(check_spacer_1)
    control_guides_df['spacer2_check'] = control_guides_df['sgRNA Sequence'].apply(check_spacer_2, tRNA_req=tRNA_req)    

    # require that spacers work in all positions
    guide_input_df = guide_input_df[guide_input_df['spacer1_check'] & guide_input_df['spacer2_check']]
    control_guides_df = control_guides_df[control_guides_df['spacer1_check'] & control_guides_df['spacer2_check']]

    control_candidates = list(control_guides_df['sgRNA Sequence'].values)

    # iterate through gene pairs
    for gene_A, gene_B in gene_pairs:

        # subsetting guide input for target genes
        target_A_df = guide_input_df[guide_input_df['Target Gene Symbol']==gene_A].sort_values('Pick Order')
        target_B_df = guide_input_df[guide_input_df['Target Gene Symbol']==gene_B].sort_values('Pick Order')
        
        # select guides based on target and pick order, + controls
        spacer_dict = {'A_'+str(i+1): list(target_A_df['sgRNA Sequence'].values)[i] for i in range(n_A_guides)}
        spacer_dict.update({'B_'+str(i+1): list(target_B_df['sgRNA Sequence'].values)[i] for i in range(n_B_guides)})
        spacer_dict.update({'ctrl_'+str(i+1):rng.choice(control_candidates, n_ctrl_guides)[i] for i in range(n_ctrl_guides)})

        df_target = GI_input_df.copy()
        # randomly assign tRNA if unspecified
        df_target.loc[df_target['tRNA_version'].isna(),'tRNA_version'] = rng.choice(
            list(range(1,len(tRNA_seqs.keys())+1)), len(df_target.loc[df_target['tRNA_version'].isna()]))
        tRNA_order = rng.choice(list(tRNA_seqs.keys()), len(tRNA_seqs.keys()), replace=False)
        tRNA_assignment = {i+1:tRNA_order[i] for i in range(len(tRNA_order))}
        df_target['tRNA'] = df_target['tRNA_version'].map(tRNA_assignment)
        
        # pair guides as defined in GI set input
        df_target['target'] = df_target['GI_version'].map({'A':gene_A,'B':gene_B, 'AB':str(gene_A)+'_'+str(gene_B)})
        df_target['GI_pair'] = str(gene_A)+'_'+str(gene_B)
        df_target['spacer_1'] = (df_target['target_pos1']+'_'+df_target['spacer_version_pos1'].astype(str)).map(spacer_dict)
        df_target['spacer_1_target'] = df_target['target_pos1'].map({'A':gene_A,'B':gene_B, 'ctrl':'ctrl'})
        df_target['spacer_2'] = (df_target['target_pos2']+'_'+df_target['spacer_version_pos2'].astype(str)).map(spacer_dict)
        df_target['spacer_2_target'] = df_target['target_pos2'].map({'A':gene_A,'B':gene_B, 'ctrl':'ctrl'})

        df_target['GI_version_index'] = df_target.groupby('GI_pair').cumcount()+1
        
        guide_pairs_df = pd.concat([guide_pairs_df, df_target])

    return guide_pairs_df

##################################################################################################
#                           build oligos for CROPseq-multi one-step cloning
##################################################################################################

def build_CROPseq_multi_one_step_oligo(
    spacer_1, iBAR_1, spacer_2, iBAR_2,
    template, dialout_fwd, dialout_rev, 
    tRNA_leader_seq = None,
    tRNA_choice = None,
):
    
    def build_oligo(tRNA_leader_seq, tRNA_seq):
        return template.format(
            dialout_fwd = dialout_fwd,
            CSM_BsmBI_left = CSM_BsmBI_left,
            spacer_1 = spacer_1,
            CSM_stem_1 = CSM_stem_1,
            iBAR_1 = reverse_complement(iBAR_1), # reverse complement iBARs
            tracr_1 = CSM_tracr_1,
            tRNA_leader = tRNA_leader_seq,
            tRNA = tRNA_seq,
            spacer_2 = spacer_2,
            CSM_stem_2 = CSM_stem_2,
            iBAR_2 = reverse_complement(iBAR_2), # reverse complement iBARs
            CSM_BsmBI_right = CSM_BsmBI_right,
            dialout_rev = reverse_complement(dialout_rev),
        ).upper()
    
    # randomly select a tRNA version if none specified
    if tRNA_choice is None:
        tRNA_choice = np.random.choice(list(tRNA_seqs.keys()))
    tRNA_seq = tRNA_seqs[tRNA_choice]

    # set tRNA leader to default if none specified
    if tRNA_leader_seq is None:
        tRNA_leader_seq = CSM_tRNA_leader_default
    
    oligo = build_oligo(tRNA_leader_seq, tRNA_seq)

    # check for BsmBI and U6 terminator "TTTT" sites
    if (count_BsmBI(oligo)!=2) | (oligo.find("TTTT")!=-1):
        # try to fix any with alternate tRNA choice
        for tRNA_choice_2 in tRNA_seqs.keys():
            tRNA_seq = tRNA_seqs[tRNA_choice_2]
            oligo_candidate = build_oligo(tRNA_leader_seq, tRNA_seq)
            if (count_BsmBI(oligo_candidate)==2) &\
                (oligo_candidate.find("TTTT")==-1):
                oligo = oligo_candidate
                tRNA_choice = tRNA_choice_2
                break

    return oligo, tRNA_choice

def build_CROPseq_multi_one_step_oligos(
    df_guides_input, 
    spacer_1_col='spacer_1',
    spacer_2_col='spacer_2', 
    ibar1_col='iBAR_1', 
    ibar2_col='iBAR_2', 
    tRNA_leader_col = 'tRNA_leader',
    tRNA_col='tRNA', 
    dialout_fwd_col ='dialout_fwd',
    dialout_rev_col ='dialout_rev',
    # column names
    template = template_1_step_oligo
):
    
    df_guides = df_guides_input.reset_index(drop=True).copy()

    # start by checking whether some columns were specified

    # if tRNA leader sequence column is not present, create and fill with default
    if tRNA_leader_col not in df_guides.columns:
        df_guides[tRNA_leader_col] = CSM_tRNA_leader_default
    # if provided, fill any missing values with default
    elif df_guides[tRNA_leader_col].isna().values.any():
        print("unspecified leader sequences were set to the default: %s"%CSM_tRNA_leader_default)
        df_guides[tRNA_leader_col]  = df_guides[tRNA_leader_col].fillna(CSM_tRNA_leader_default)

    # if tRNA column is not present, create and fill at random
    if tRNA_leader_col not in df_guides.columns:
        print("tRNAs not specified by 'tRNA_leader_col' = %s - assign tRNAs at random "% tRNA_leader)
        df_guides[tRNA_col] = np.random.choice(tRNA_seqs.keys(), len(df_guides))
    # if provided, fill any missing values with default
    elif df_guides[tRNA_col].isna().values.any():
        print("unspecified tRNAs were assigned at random")
        # df_guides[tRNA_col] = df_guides[tRNA_col].apply(
        #     lambda trna: trna if not np.isnan(trna) else np.random.choice(list(tRNA_seqs.keys())) )
        df_guides.loc[df_guides[tRNA_col].isna(),tRNA_col] = np.random.choice(list(tRNA_seqs.keys()),
            len(df_guides[df_guides[tRNA_col].isna()]))

    oligos = []
    tRNA_choices = []
    failed_designs = pd.DataFrame()


    # iterate through and design oligos
    for index, row in df_guides.iterrows():

        oligo, tRNA_selection = build_CROPseq_multi_one_step_oligo(
            row[spacer_1_col],
            row[ibar1_col],
            row[spacer_2_col],
            row[ibar2_col],
            tRNA_leader_seq=row[tRNA_leader_col],
            tRNA_choice=row[tRNA_col],
            dialout_fwd=row[dialout_fwd_col],
            dialout_rev=row[dialout_rev_col],
            template = template_1_step_oligo,
            )

        # final check for BsmBI and U6 terminator sequences
        if (count_BsmBI(oligo)==2) & (oligo.find("TTTT")==-1) & (oligo.find("N")==-1):
            pass
        else:
            errors = []
            if count_BsmBI(oligo)!=2:
                errors.append('%s BsmBI sites'%(count_BsmBI(oligo)))
            if contains_U6_term(oligo):
                errors.append('contains "TTTT"')
            if oligo.find("N")!=-1:
                errors.append('degenerate bases present')


            failed_designs = pd.concat([failed_designs, df_guides.iloc[index:index+1]])
            failed_designs.loc[index,'failure_cause'] = ', '.join(errors)
            failed_designs.loc[index,'oligo'] = oligo

            oligo='failed'

        if row[tRNA_col] != tRNA_selection:
            print('changed a tRNA from %s to %s'%(row[tRNA_col], tRNA_selection))
        
        oligos.append(oligo)
        tRNA_choices.append(tRNA_selection)
        
    df_guides['oligo'] = oligos
    df_guides['tRNA'] = tRNA_choices # because some may be changed

    return df_guides, failed_designs


##################################################################################################
#                            build oligos for ROPseq-multi two-step cloning
##################################################################################################


def build_CROPseq_multi_two_step_oligo(
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
            BbsI_filler = filler_seq,
            spacer_2 = spacer_2,
            CSM_stem_2 = CSM_stem_2,
            iBAR_2 = reverse_complement(iBAR_2), # reverse complement iBARs
            CSM_BsmBI_right = CSM_BsmBI_right,
            dialout_rev = reverse_complement(dialout_rev),
        ).upper()
    
    def fill_degenerate_bases(seq):
        while seq.find("N") != -1:
            n_index = seq.find("N")
            base = np.random.choice(['A','C','T','G'])
            seq = seq[:n_index] + base + seq[n_index+1:]
        return seq
    
    # randomly select a BbsI filler version
    if filler_version is None:
        filler_version = np.random.choice(list(BbsI_fillers.keys()))
    filler_seq = BbsI_fillers[filler_version]
    
    oligo = build_oligo(filler_seq)
    
    # check for BsmBI, BbsI, and U6 terminator "TTTT" sites
    if (count_BsmBI(oligo)!=2) | (count_BbsI(oligo)!=2) | (oligo.find("TTTT")!=-1):
        oligo_temp = oligo
        oligo = "failed"
        # try to fix any with alternate tRNA choice
        for filler_version in BbsI_fillers.keys():
            filler_seq = BbsI_fillers[filler_version]
            oligo_candidate = build_oligo(filler_seq)
            if (count_BsmBI(oligo_candidate)==2) &\
                (count_BbsI(oligo_candidate)==2) &\
                (oligo_candidate.find("TTTT")==-1):
                print('substituted original tRNA selection')
                oligo = oligo_candidate
                print(oligo)
                break
                
    # unable to remove BsmBI, BbsI, or U6 terminator "TTTT" motif
    if oligo == "failed":
        print('failed to remove BsmBI, BbsI, or U6 terminator')
        print(oligo_temp)
        return oligo, filler_version
    
    # fill any "N" sequences with random choice while avioding homopolymers and RE sites 
    i=0
    while oligo.find("N") != -1:
        if i>50:
            print('failed to fill degenerate bases')
            print(oligo)
            oligo='failed'
            break
        i+=1
        oligo_candidate = fill_degenerate_bases(oligo)
        if (count_BsmBI(oligo_candidate)==2) & (count_BbsI(oligo_candidate)==2):
            # require that no new homopolymers are created
            if count_homopolymer(oligo, 4) == count_homopolymer(oligo_candidate, 4):
                oligo = oligo_candidate
                break
            else:
                    continue
        else:
            continue       
    
    return oligo, filler_version

def build_CROPseq_multi_two_step_oligos(
    df_input, 
    template,
    spacer_1_column = 'spacer_1',
    iBar_1_column ='iBAR_1',
    spacer_2_column = 'spacer_2',
    iBar_2_column ='iBAR_2',
    tRNA_column = 'tRNA',
    dialout_fwd_col ='dialout_fwd',
    dialout_rev_col ='dialout_rev',
                         ):
    
    # what to record:
    # complete oligo sequence
    # iBAR 1 and 2 sequences
    # tRNA version
    # dialout pair
    
    df = df_input.copy()

    oligo_designs = []
    filler_versions = []
    for spacer_1, iBAR_1, spacer_2, iBAR_2, tRNA, dialout_fwd, dialout_rev in df[
        [spacer_1_column, iBar_1_column, 
         spacer_2_column, iBar_2_column, 
         tRNA_column, 
         dialout_fwd_col, dialout_rev_col]].values:
        
        oligo, filler_version = build_CROPseq_multi_two_step_oligo(
            spacer_1, iBAR_1, spacer_2, iBAR_2, dialout_fwd, dialout_rev, 
            filler_version=tRNA,
            template = template_2_step_oligo)
        
        oligo_designs.append(oligo)
        filler_versions.append(filler_version)
    df['oligo'] = oligo_designs
    df['tRNA'] = filler_versions
    
    return df



