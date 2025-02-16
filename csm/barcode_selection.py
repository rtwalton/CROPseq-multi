
import pandas as pd
import numpy as np
import random
import tqdm.notebook as tqdm
import warnings
import Levenshtein
import math
from glob import glob

from constants import *
from utils import *


##################################################################################################
#                               iBAR selection workflows
##################################################################################################

def automated_iBAR_assignment(df, distance=3, method='positional'):
    """
    Automatically assign barcode pairs to a DataFrame by selecting an appropriate pre-designed barcode set.
    Wrapper for select_complete_and_pair_barcodes()

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to assign barcodes to. Each row will get a unique barcode pair.
    distance : int, optional
        Minimum edit distance required between barcodes. Default is 3.
    method : str, optional
        Method for pairing barcodes. Options are:
        - 'positional': Position encoded in first base (default)
        - 'random_unique': Random unique barcodes independent of position
        - 'random_shared': Position-specific but shared between positions
        - 'matched': Identical barcodes in both positions
        See select_complete_and_pair_barcodes() for detailed descriptions.

    Returns
    -------
    pandas.DataFrame
        Input DataFrame with additional columns for barcode pairs:
        - 'iBAR1': First barcode sequence
        - 'iBAR2': Second barcode sequence
    """

    # import all the barcode sets:
    bc_sets = pd.DataFrame()
    search = 'designed_barcode_sets/barcodes_n*_k*_*.noBsmBI.csv'
    for file in glob(search):
        df_barcodes = pd.read_csv(file)
        bc_sets.loc[file, 'length'] = df_barcodes['n'][0]
        bc_sets.loc[file, 'distance'] = df_barcodes['k'][0]
        bc_sets.loc[file, 'metric'] = file.split('_')[-1].split('.')[0]
        bc_sets.loc[file, 'n_barcodes'] = len(df_barcodes)
    
    iBAR_pairs = select_complete_and_pair_barcodes(bc_sets, len(df), distance=distance, method=method)

    return df.merge(iBAR_pairs, left_index=True, right_index=True)



def select_complete_and_pair_barcodes(df_bc_sets, n_pairs, distance, method, **kwargs):
    """
    Automated  (1) selection of barcode sets based on the number of barcodes needed, taking into
    account edit distance and the method of barcode pairing, (2) completion of barcode to required 
    full length and (3) pairing of barcodes.
    
    df_bc_sets (pd.DataFrame): DataFrame summarizing the available pre-defined barcode sets. The barcode set 
    with the shortest barcode length and the required edit distance will be selected.
    n_pairs (int): (minimum) number of barcode pairs needed
    distance (int): required edit distance (Levenshtein or Hamming)
    
    method (str): method for pairing bacodes. options:
    
        'random_unique'
            iBAR assignment is random and all iBARs are unique, independent of position. Any one 
            barcode is sufficient to uniquely identify a construct, even if it is not known whether the barcode 
            observed is iBAR1 or iBAR2. Requires >2*n_pairs barcodes.
        'random_shared'
            iBAR assignment is random and all iBARs are unique to a position (iBAR1 or iBAR2), however barcodes 
            can be reused between positions (iBAR1 and iBAR2). Any one barcode is sufficient to uniquely 
            identify a construct as long as it is known whether the barcode observed is iBAR1 or iBAR2.
            Requires >n_pairs barcodes.
        'positional'
            iBAR position (1 or 2) is encoded in the first base. iBAR1 starts with A/G. iBAR2 starts with C/T.
            All iBARs are unique and any one barcode is sufficient to uniquely identify a construct. Requires 
            >2*n_pairs barcodes. Facilitates identification of iBAR position with multiplexed detection.
        'matched'
            Both iBARs of a construct encode the same barcode. iBAR1 = iBAR2. All iBARs are unique to a construct
            (other than the identical pair in the same construct).
            
    Returns
        pd.DataFrame: dataframe with >=n_pairs of barcodes
        
    
    """

    if method == 'random_shared':
        min_n_barcodes = n_pairs
    else: 
        min_n_barcodes = 2*n_pairs
    
    df_bc_sets_subset = df_bc_sets[(df_bc_sets['distance']==distance) & (df_bc_sets.n_barcodes>=min_n_barcodes)]
    
    iBAR_pairs = None
    while iBAR_pairs is None:
        
        if len(df_bc_sets_subset)==0:
            print('\nRan out of barcode files')
            break
            
        selected_bc_file = df_bc_sets_subset[df_bc_sets_subset.n_barcodes > min_n_barcodes].iloc[
            df_bc_sets_subset[df_bc_sets_subset.n_barcodes > min_n_barcodes]['n_barcodes'].argmin()].name 
        
        df_barcodes = pd.read_csv(selected_bc_file)
        
        print('\nEdit distance %s in %s cycles'%(df_barcodes['k'][0],df_barcodes['n'][0]))

        iBAR_pairs = complete_iBARs(df_barcodes, n_pairs, method, **kwargs)
        
        if iBAR_pairs is None:
            df_bc_sets_subset = df_bc_sets_subset[df_bc_sets_subset.index != selected_bc_file]
            print('\nTrying again with next smallest barcode set.')
        else:
            return iBAR_pairs


def complete_iBARs(
    df, 
    n_constructs,
    method='positional', 
    verbose=1,
    **kwargs
    ):
    """
    Wrapper for methods of generating complete pairs of iBARs.
    """
    df_barcodes = df.sample(frac=1).copy() # important to randomize barcode order!

    if method in ['positional', 'matched', 'random_unique']:
        iBAR1s_filtered, iBAR2s_filtered = complete_iBARs_v1(df_barcodes, n_constructs, method, **kwargs)

    elif method == 'random_shared':
        iBAR1s_filtered, iBAR2s_filtered = complete_iBARs_v2(df_barcodes, n_constructs, **kwargs)

    else:
        raise ValueError(f"method {method} not recognized")
        
    if iBAR1s_filtered is None:
        return None

    df_pairs = pair_iBARs(iBAR1s_filtered, iBAR2s_filtered, n_constructs, method, **kwargs)
    print(f'\ndesigned {len(df_pairs)} barcode pairs')

    print('\ndetermining cycling requirements for decoding...')
    check_barcode_pair_cycle_requirements(df_pairs)
    
    return df_pairs

def pair_iBARs(iBAR1s_filtered, iBAR2s_filtered, n_constructs, method, verbose=1):
    """
        Determine how complete iBAR sequences will be paired.
    """
    # pair barcodes
    min_cycles = []
    df_pairs = pd.DataFrame()

    if method != 'matched':
        if verbose>0:
            print('\noptimizing barcode pairing...')

        # determine optimal pairs with respect to decoding efficiency with multiplexed dection
        barcode_pairs = determine_optimial_barcode_pairs(iBAR1s_filtered, iBAR2s_filtered, n_constructs)
        
        if barcode_pairs is None:
            print('\ndefaulting to random iBAR pairings.')
            rng = np.random.default_rng(seed=0)
            selected_barcodes_1 = rng.choice(iBAR1s_filtered, n_constructs, replace=False)
            selected_barcodes_2 = rng.choice(iBAR2s_filtered, n_constructs, replace=False)
            df_pairs[['iBAR_1','iBAR_2']]= np.array([selected_barcodes_1, selected_barcodes_2]).T
        else:
            df_pairs[['iBAR_1','iBAR_2']] = barcode_pairs
            
    elif method == 'matched':
        rng = np.random.default_rng(seed=0)
        selected_barcodes_1 = rng.choice(iBAR1s_filtered, n_constructs, replace=False)
        selected_barcodes_2 = selected_barcodes_1
        
        df_pairs[['iBAR_1','iBAR_2']]= np.array([selected_barcodes_1, selected_barcodes_2]).T
    
    return df_pairs


def complete_iBARs_v1(
    df, n_constructs,
    method =  'positional',
    max_it_degenerate_bases = 25,
    verbose=1,
    **kwargs
):

    """
    Complete iBAR sequences and return candidates for each iBAR position.
    For pairing 'methods' ['positional', 'matched', 'random_unique']
    """
    
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
        print('\ngenerating and filtering complete iBARs...')

    for bc in tqdm(df_barcodes['barcode'], total=len(df_barcodes)):
        i=0
        while i<max_it_degenerate_bases:
            i+=1
            bc_filled = fill_degenerate_bases(bc)
            if (calculate_gc(bc_filled) > gc_max) or\
                   (calculate_gc(bc_filled) < gc_min) or\
                   has_homopolymer(bc_filled, homopolymer_max):
                continue

            if method == 'positional':
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
                    
            elif method == 'matched':
                # same barcode is used for both iBARs of a construct
                if (check_iBAR1(bc_filled)) & (check_iBAR2(bc_filled)):
                    iBAR1s_filtered.append(bc_filled)
                    iBAR2s_filtered.append(bc_filled)
                    break
                else:
                    continue
                
            elif method == 'random_unique':
                # randomly assign barcode to either position, as compatibility allows
                if (check_iBAR1(bc_filled)) & (check_iBAR2(bc_filled)):
                    if len(iBAR1s_filtered) < len (iBAR2s_filtered):
                        iBAR1s_filtered.append(bc_filled)
                        break
                    else:
                        iBAR2s_filtered.append(bc_filled)
                        break
                elif (check_iBAR1(bc_filled)):
                    iBAR1s_filtered.append(bc_filled)
                    break
                elif (check_iBAR2(bc_filled)):
                    iBAR2s_filtered.append(bc_filled)
                    break
    
    min_barcode_count = min([len(iBAR1s_filtered), len(iBAR2s_filtered)])
    if min_barcode_count < n_constructs:
        print('\nFailed: Only %s barcodes after filtering. At least %s required.'%(min_barcode_count, n_constructs))
        return None, None
        
    else:
        return iBAR1s_filtered, iBAR2s_filtered

def complete_iBARs_v2(
    df, n_constructs,
    max_it_degenerate_bases = 25,
    verbose=1,
    **kwargs
):
    """
    Complete iBAR sequences and return candidates for each iBAR position.
    For pairing method 'random_shared'
    """
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
        print('\ngenerating and filtering complete iBARs...')
    
    for bc in tqdm(df_barcodes['barcode'], total=len(df_barcodes)):
        i=0
        iBAR_1_used = False
        iBAR_2_used = False

        # try to find full barcodes that work
        while i<max_it_degenerate_bases:
            i+=1
            bc_filled = fill_degenerate_bases(bc)
            if (calculate_gc(bc_filled) > gc_max) or\
                   (calculate_gc(bc_filled) < gc_min) or\
                   has_homopolymer(bc_filled, homopolymer_max):
                continue
                
            if (~iBAR_1_used) & (check_iBAR1(bc_filled)):
                iBAR_1_used = True
                iBAR1s_filtered.append(bc_filled)
                
            if (~iBAR_2_used) & (check_iBAR2(bc_filled)):
                iBAR_2_used = True
                iBAR2s_filtered.append(bc_filled)
            
            if iBAR_1_used & iBAR_2_used:
                break

    min_barcode_count = min([len(iBAR1s_filtered), len(iBAR2s_filtered)])
    if min_barcode_count < n_constructs:
        print('\nFailed: Only %s barcodes after filtering. At least %s required.'%(min_barcode_count, n_constructs))
        return None, None
        
    else:
        return iBAR1s_filtered, iBAR2s_filtered


def determine_optimial_barcode_pairs(barcodes_1, barcodes_2, n_constructs):
    """
    Generate barcode pairs optimized for multiplexed decoding with minimal sequencing cycles.

    barcodes_1 (list): barcode candidates for iBAR1
    barcodes_2 (list): barcode candidates for iBAR2
    n_constructs (int): (minimum) number of barcode pairs required

    Returns: pd.DataFrame with barcode pairs 
    
    """

    # initial estimate for minimal number of cycles needed for multiplexed decoding
    min_bases = np.ceil(math.log(n_constructs, 4)).astype(int)
    min_cycles = max(2, int(np.ceil(min_bases/2)))

    # set seed for reproducibility
    random.seed(5)
    
    n_attempt = 1
    max_n_attempt = 3
    while n_attempt <= max_n_attempt:

        print(f'attempt {n_attempt}/{max_n_attempt}')
        # set up the possible combinations
        min_bc1s = list(set([i[:min_cycles] for i in barcodes_1]))
        min_bc2s = list(set([i[:min_cycles] for i in barcodes_2]))
                
        binned_bc1s = {bc_short:[] for bc_short in min_bc1s}
        binned_bc2s = {bc_short:[] for bc_short in min_bc2s}
        
        for barcode in barcodes_1:
            binned_bc1s[barcode[:min_cycles]].append(barcode)
        for barcode in barcodes_2:
            binned_bc2s[barcode[:min_cycles]].append(barcode)

        # shuffle order
        for key in binned_bc1s.keys():
            random.shuffle(binned_bc1s[key])
        for key in binned_bc2s.keys():
            random.shuffle(binned_bc2s[key])

        # some fancy matrix math to get all compatible combinations while requiring 
        # that iBAR1 != iBAR2 (A-A not allowed) and unique pairs (A-B exlucdes B-A)
        all_bc = sorted(set(min_bc1s + min_bc2s))
        bc_pair_mat = np.ones( (len(all_bc), len(all_bc)) ) - np.eye(len(all_bc) )
        
        # barcodes incompatible with position 1
        bc_pair_mat = bc_pair_mat.T * np.array([bc in min_bc1s for bc in all_bc])
        # barcodes incompatible with position 2
        bc_pair_mat = (bc_pair_mat.T * np.array([bc in min_bc2s for bc in all_bc]) ).T

        bc_pair_mask = (np.triu(bc_pair_mat) + np.bitwise_and(
            ~np.triu(bc_pair_mat).astype(bool).T, np.tril(bc_pair_mat).astype(bool))
                       ).astype(bool)
        
        short_bc_pairs = np.array([ 
            np.array([all_bc[bc_pair[0]],
                      all_bc[bc_pair[1]]]
                    ) for bc_pair in np.argwhere(bc_pair_mask.T)])     

        if (set(short_bc_pairs.T[0,:]) > set(min_bc1s)) | (set(short_bc_pairs.T[1,:]) > set(min_bc2s)):
            raise ValueError("error optimizing compatible barcode pairs")

        short_bc_pairs = list(short_bc_pairs)
        random.shuffle(short_bc_pairs)
        
        barcode_pairs = []
        
        for short_bc1, short_bc2 in tqdm(short_bc_pairs):
        
            if short_bc1 != short_bc2:
                min_bc_count = 1
            else:
                min_bc_count = 2
                
            # select a pair of barcode from binned sets
            if (len(binned_bc1s[short_bc1]) >= min_bc_count) & (len(binned_bc2s[short_bc2]) >= min_bc_count):            
                
                bc1 = binned_bc1s[short_bc1][0]
                binned_bc1s[short_bc1].remove(bc1)
                # if bc1 in min_bc2s:
                #     if bc1 in binned_bc2s[short_bc1]: binned_bc2s[short_bc1].remove(bc1)
                
                bc2 = binned_bc2s[short_bc2][0]
                binned_bc2s[short_bc2].remove(bc2)
                # if bc2 in min_bc1s:
                #     if bc2 in binned_bc1s[short_bc2]: binned_bc1s[short_bc2].remove(bc2)
                        
                barcode_pairs.append([bc1, bc2])

            else:
                continue
                
        barcode_pairs = pd.DataFrame(barcode_pairs, columns=['iBAR_1','iBAR_2'])
        
        if len(barcode_pairs) >= n_constructs:
            return barcode_pairs
        elif n_attempt <= max_n_attempt:
            n_attempt +=1
            min_cycles +=1
            
    print('Failed to determine optimal barcode pairing.')
    return None

#TODO: complete iBARs_single_guide
def complete_iBARs_single_guide(
    df, n_constructs,
    max_it_degenerate_bases = 25,
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
    
    iBARs_filtered = []
    df_iBARs = pd.DataFrame()

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

            # check compatibility with flanking sequences
            # single guide iBAR has flanking sequence of iBAR2
            if (check_iBAR2(bc_filled)): 
                iBARs_filtered.append(bc_filled)          
                break
            else:
                continue
    
    if len(iBARs_filtered) < n_constructs:
        print('Failed: Only %s barcodes after filtering. At least %s required.'%(len(iBARs_filtered), n_constructs))
        return None
    
    df_iBARs['iBAR']= iBARs_filtered[:len(n_constructs)]

    print('sequencing requirements:')
    barcode_cycles = determine_minimal_cycle_number(df_iBARs['iBAR'].values)
    if barcode_cycles is None:
        print('barcodes are not unique')
    else:
        print('barcodes are unique in:\n %s cycles'% barcode_cycles)
    
    return df_iBARs


##################################################################################################
#                                 evaluating decoding performance
##################################################################################################


def determine_minimal_cycle_number(sequence_list, dual=False, directional=False):
    """
    Determine the minimum number of sequencing cycles needed to uniquely identify all sequences in a set.
    
    Parameters
    ----------
    sequence_list : list of str
        List of barcode sequences to analyze. Must be provided in the orientation they will be 
        sequenced (not the reverse complement!).
    dual : bool, optional
        Whether sequences are dual barcodes are read out simultaneously. If True, expects
        barcodes of equal length separated by "_" delimiter (e.g. "iBAR1_iBAR2"). Default is False.
    directional : bool, optional 
        Whether barcode order/position matters for dual barcodes. Only used if dual=True.
        If True, "A_B" is considered distinct from "B_A".
        If False, "A_B" and "B_A" are considered equivalent.
        For example, with dual iBARs:
        - If iBARs are labeled (e.g. with fluorescent oligos), set directional=True since 
          iBAR1 and iBAR2 can be distinguished.
        - If iBARs are unlabeled and read simultaneously, set directional=False since the 
          mapping of barcodes to iBAR1/iBAR2 positions is ambiguous.
        Default is False.

    Returns
    -------
    int or None
        Minimum number of cycles needed for unique identification.
        Returns None if sequences cannot be uniquely identified.
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
    """
    Simulate recombination events between barcode pairs and determine how many sequencing cycles 
    are needed to detect the recombined barcodes as invalid pairs.

    Parameters
    ----------
    barcode_pairs : pandas.DataFrame
        DataFrame containing barcode pairs, with columns 'bc1' and 'bc2' containing the barcode sequences
    directional : bool, optional
        Whether the relative position of barcodes can be determined experimentally. If True, barcode pair A-B 
        can be distinguished from B-A. If False, A-B and B-A are treated as equivalent. Default is True.
    n_test : int, optional
        Number of recombination events to simulate. Default is 5000.

    Returns
    -------
    dict
        Dictionary mapping number of sequencing cycles to fraction of recombination events detected.
        Keys are integers representing number of cycles, values are floats between 0 and 1 representing
        the fraction of recombination events that can be detected with that many cycles.
    """
    
    n_pairs = len(barcode_pairs)
    barcode_pairs = np.array(barcode_pairs).T
    bc_df = pd.DataFrame(barcode_pairs.T, columns=['bc1','bc2'])

    # initial estimate for minimum number of cycles needed
    min_bases = np.ceil(math.log(n_pairs, 4)).astype(int)
    min_cycles = int(np.ceil(min_bases/2))
    
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

    print("\nsimulating recombination detection...")
    pbar = tqdm(range(min_cycles, len(barcode_pairs[0][0])+1), leave=False)
    for seq_len in pbar:
    
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
        
        # return as soon as 100% recombination detection is met
        if recomb_detect_rates[seq_len] == 1:
            return recomb_detect_rates
            
    return recomb_detect_rates

def check_barcode_pair_cycle_requirements(
    barcode_pair_df,
    barcode_1 = 'iBAR_1', barcode_2 = 'iBAR_2',
    check_distance = False,
):
    """
    Determine the minimum number of sequencing cycles needed to uniquely identify library members 
    with various decoding strategies. Particularly relevant for in situ sequencing approaches.

    This function analyzes barcode pairs to determine:
    1. Cycles needed for unique identification using only barcode 1
    2. Cycles needed for unique identification using only barcode 2 
    3. Cycles needed for unique identification of any barcode from the set (barcode 1 and barcode 2)
    4. Cycles needed for unique identification of any construct using multiplexed decoding of barcode 1 and barcode 2
    5. Cycles needed for unique identification of any construct using directional multiplexed decoding of barcodes 1 and 2 when 
       the relative positions of the barcodes can NOT be determined experimentally (i.e. A-B can NOT be distinguished from B-A)
       - also quantifies the ability to detect recombination events
    6. Cycles needed for unique identification of any construct using directional multiplexed decoding (i.e. A-B can when 
       the relative positions of the barcodes can be determined experimentally (i.e. A-B can be distinguished from B-A)
        - also quantifies the ability to detect recombination events

    Parameters
    ----------
    barcode_pair_df : pandas.DataFrame
        DataFrame containing pairs of barcodes, with columns for barcode 1 and barcode 2
    barcode_1 : str, optional
        Column name for first barcode in pair, by default 'iBAR_1'
    barcode_2 : str, optional
        Column name for second barcode in pair, by default 'iBAR_2'
    check_distance : bool, optional
        Whether to verify Levenshtein distances between barcodes for error detection/correction.
        By default False since this is typically guaranteed by initial barcode selection.

    Returns
    -------
    None
        Results are printed to stdout
    """

    
    barcode_1_cycles = determine_minimal_cycle_number(barcode_pair_df[barcode_1].values)
    barcode_2_cycles = determine_minimal_cycle_number(barcode_pair_df[barcode_2].values)
    barcode_1_or_2_cycles = determine_minimal_cycle_number(barcode_pair_df[[barcode_1,barcode_2]].values.flatten())

    if barcode_1_cycles is None:
        print('\nbarcode 1 is not unique in set(barcode 1)')
    else:
        print('\nbarcode 1 is unique in set(barcode 1) in:\n %s cycles'% barcode_1_cycles)
        if check_distance:
            check_cycle_requirements_correct_detect(barcode_pair_df[barcode_1].values, barcode_1_cycles)
    
    if barcode_2_cycles is None:
        print('\nbarcode 2 is not unique in set(barcode 2)')
    else:
        print('\nbarcode 2 is unique in set(barcode 2) in:\n %s cycles'% barcode_2_cycles)
        if check_distance:
            check_cycle_requirements_correct_detect(barcode_pair_df[barcode_2].values, barcode_2_cycles)
            
    if barcode_1_or_2_cycles is None:
        print('\nbarcodes in set(barcode 1, barcode 2) are not unique')
    else:
        print('\nbarcodes in set(barcode 1, barcode 2) are unique in:\n %s cycles'% barcode_1_or_2_cycles)
    
    min_dual_cycle = determine_minimal_cycle_number(barcode_pair_df[barcode_1]+'_'+barcode_pair_df[barcode_2], 
                                         dual=True, directional=False)
    if min_dual_cycle is None:
        print('\nbarcode pair is not unique in set(1-2, 2-1)')
    else:
        recomb_detection = test_recombination_detection(
            barcode_pair_df[[barcode_1,barcode_2]], directional=False)
        print('\nbarcode pair is unique in set(1-2, 2-1) in :\n %s cycles with %s%% recombination detection'%(
            min_dual_cycle, round(100*recomb_detection[min_dual_cycle])))
        if round(100*recomb_detection[min_dual_cycle]) < 100:
            print(' or:\n %s cycles with %s%% recombination detection'%(
                min_dual_cycle+1, round(100*recomb_detection[min_dual_cycle+1])))

    min_dual_cycle = determine_minimal_cycle_number((barcode_pair_df[barcode_1]+'_'+barcode_pair_df[barcode_2]).values, 
                                         dual=True, directional=True)
    if min_dual_cycle is None:
        print('\nbarcode pair is not unique in set(1-2)')
    else:
        recomb_detection = test_recombination_detection(
            (barcode_pair_df[[barcode_1,barcode_2]]).values, directional=True)
        
        print('\nbarcode pair is unique in set(1-2) in:\n %s cycles with %s%% recombination detection'%(
            min_dual_cycle, round(100*recomb_detection[min_dual_cycle])))
        if round(100*recomb_detection[min_dual_cycle]) < 100:
            print(' or:\n %s cycles with %s%% recombination detection'%(
                min_dual_cycle+1, round(100*recomb_detection[min_dual_cycle+1])))

def check_cycle_requirements_correct_detect(sequence_list, cycle_start):
    """
    Check minimum cycles needed for edit detection and correction.
    Slow for large barcode sets!
    """
    check_detection=True
    check_correction=True

    max_cycles = len(sequence_list[0])
    
    for cycles in range(cycle_start, max_cycles+1):
        distmat = barcode_distance_matrix([bc[:cycles] for bc in sequence_list])
        mask = np.ones(distmat.shape, dtype=bool)
        np.fill_diagonal(mask, 0)
        min_dist = distmat[mask].min()
        
        if check_detection & (min_dist >= 2):
            print(f' error detection in {cycles} cycles')
            check_detection = False
    
        if check_correction & (min_dist >= 3):
            print(f' error correction in {cycles} cycles')
            check_correction = False
            break
        else:
            cycles +=1
        
    if check_detection:
        print(f'no error detection up to {max_cycles} cycles')
    if check_correction:
        print(f'no error correction up to {max_cycles} cycles')

def barcode_distance_matrix(barcodes_1, barcodes_2=False, distance_metric='levenshtein'):
    """
    Calculate pairwise distances between sequences in two lists of barcodes.

    Parameters
    ----------
    barcodes_1 : list
        First list of barcode sequences
    barcodes_2 : list or bool, optional
        Second list of barcode sequences. If False (default), calculates distances
        between all pairs in barcodes_1.
    distance_metric : str, optional
        Method for calculating sequence distances. Options are:
        - 'hamming': Hamming distance (number of positions that differ)
        - 'levenshtein': Levenshtein distance (minimum number of edits required)
        Default is 'hamming'.

    Returns
    -------
    numpy.ndarray
        2D array of shape (len(barcodes_1), len(barcodes_2)) containing the
        pairwise distances between sequences.
    """

    if distance_metric == 'hamming':
        distance = lambda i, j: Levenshtein.hamming(i, j)
    elif distance_metric == 'levenshtein':
        distance = lambda i, j: Levenshtein.distance(i, j)
    else:
        warnings.warn('distance_metric must be "hamming" or "levenshtein" - defaulting to "hamming"')
        distance = lambda i, j: Levenshtein.hamming(i, j)

    if isinstance(barcodes_2, bool):
        barcodes_2 = barcodes_1

    # create distance matrix for barcodes
    bc_distance_matrix = np.zeros((len(barcodes_1), len(barcodes_2)))
    for a, i in enumerate(barcodes_1):
        for b, j in enumerate(barcodes_2):
            bc_distance_matrix[a, b] = distance(i, j)

    return bc_distance_matrix

