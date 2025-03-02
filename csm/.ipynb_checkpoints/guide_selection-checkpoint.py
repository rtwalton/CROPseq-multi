import os
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import warnings
import Levenshtein
import math
import itertools

from constants import *
from utils import *


##################################################################################################
#                             pairing guides from a CRISPick output
##################################################################################################

def filter_guides(
    guides_df,
    position_req='any',
    tRNA_req='any',
    single_guide=False,
    ):

    """
    filter spacers for compatibility in positions of the vector
    expects CRISPick output as input format

    """
    
    # for single guides, only need to check one spacer
    if single_guide:
        guides_df['spacer1_check']= guides_df['sgRNA Sequence'].apply(check_spacer_1, single_guide=True)
        return guides_df
    
    # check spacer compatibility in each position
    else:
        guides_df['spacer1_check']= guides_df['sgRNA Sequence'].apply(check_spacer_1)
        guides_df['spacer2_check']= guides_df['sgRNA Sequence'].apply(check_spacer_2, tRNA_req=tRNA_req)

    # if spacers must work in all positions, filter out candidates that are one or the other
    if position_req =='all':
        guides_df = guides_df[guides_df['spacer1_check'] & guides_df['spacer2_check']]
    elif position_req =='any':
        guides_df = guides_df[guides_df['spacer1_check'] | guides_df['spacer2_check']]

    return guides_df

def select_single_guides_CRISPick(
    guide_input_df, 
    constructs_per_gene=4,
    ):
    """
    select guides to clone a single guide into CROPseq-multi
    """

    selected_guides = pd.DataFrame()

    # check spacer compatibility
    guide_input_df['spacer_check']= guide_input_df['sgRNA Sequence'].apply(check_spacer_1, single_guide=True)

    for target, target_df in guide_input_df.groupby('Target Gene Symbol'):
        target_df.sort_values('Pick Order', inplace=True)
        
        target_guides = target_df[target_df['spacer_check']]['sgRNA Sequence'].values[:constructs_per_gene]
        # print a warning if we run out of guides for a target
        if (len(target_df)==0):
            warnings.warn('ran out of guides for target ',target)

        selected_guides = pd.concat(
            [selected_guides,
            pd.DataFrame(data=target_guides, columns=['target','spacer'])
            ])

    selected_guides['target_version'] = selected_guides.index.astype(int) + 1
    selected_guides.reset_index(inplace=True, drop=True)


def pair_guides_single_target_CRISPick(
    guide_input_df, constructs_per_gene=3, 
    n_genes = 'all',
    tRNA_req='any', pairing_method = 'pick_sum',
    ):
    """
    Use CRISPick output to pair two guides targeting the same gene in one construct

    guide_input_df (pd.DataFrame): CRISPick output with selected guides.
    constructs_per_gene (int): Number of constructs to design per gene. 2*constructs_per_gene
    unique guides will be chosen, prioritizing the top constructs_per_gene by "Pick Order"

    n_genes (str or int): how many genes to select for designing constructs from 'guide_input_df'
    default 'all' will design guides for every gene listed. Otherwise, an 'int' value will allow
    randomly selecting that many genes. Useful for designing a set of controls (e.g. select 100
    olfactory receptor genes to design constructs)
    
    tRNA_req (str): whether guides (spacer sequences) must be compatible with 'any' tRNA or
        with 'all' tRNAs
    
    pairing_method (str): How to use CRISPick "Pick Order" in pairing guides. 
    
    Default is 'pick_sum'
        will attempt to pair guides such that the sum of the Pick Orders of all guide pairs is equal 
        i.e. for 3 constructs, pairings are 1+6, 2+5, 3+4. The rationale is to maximize the probability 
        that all designed constructs have at least one "good" guide and minimize the difference in 
        activity between constructs.
    Alternative choice is 'descending'
        Guides are paired in descending order. i.e. for 3 constructs, pairings are 1+2, 3+4, 5+6. 
        This is a useful approach to assess performance dropoff with higher pick orders and select 
        a minimum number of constructs/gene but is expected to yield greater performance variablity 
        between constructs relative to 'pick_sum'

    Returns: pd.DataFrame containing guide pairs
    """

    # if requested, subset n_genes at random, usually to design a set of controls
    if n_genes != 'all':
        if isinstance(n_genes, int):
            rng = np.random.default_rng(seed=0)
            if n_genes > len(guide_input_df['Target Gene ID'].unique()):
                print('n_genes is greater than the number of unique genes provided. defaulting to n_genes="all"')
            else:
                selected_genes = rng.choice(guide_input_df['Target Gene ID'].unique(), n_genes, replace=False)
                guide_input_df = guide_input_df[guide_input_df['Target Gene ID'].isin(selected_genes)]
        else:
            raise ValueError('n_genes must be "all" or an integer value')

    # how to use pick order in pairing guides
    if pairing_method == 'descending':
        # i.e. 1-2, 3-4, 5-6
        second_selection_index = 0
    elif pairing_method == 'pick_sum':
        # i.e. 1-6, 2-5, 3-4
        second_selection_index = -1
    else:
        warnings.warn("Valid values for pairing_method are 'descending' and 'pick_sum' - defaulting to method 'pick_sum'")
        second_selection_index = -1
    
    # check spacer compatibility in each position, require that they work in both for simplicity
    guide_input_filt =  guide_input_df[guide_input_df['sgRNA Sequence'].apply(check_spacer_1) &\
        guide_input_df['sgRNA Sequence'].apply(check_spacer_2, tRNA_req=tRNA_req)]
    all_guide_pairs = []
    
    for target, target_df in guide_input_filt.groupby('Target Gene ID'):

        target_symbol = target_df['Target Gene Symbol'].values[0]
        
        # sort guides by pick order and take the top n = 2*constructs_per_gene
        subset_target_df = target_df.sort_values('Pick Order')[:2*constructs_per_gene]

        target_guide_pairs = []
        
        # spacer candidates sorted by pick order
        spacer_candidates = list(subset_target_df['sgRNA Sequence'].values)
        spacer_pickorders = list(subset_target_df['Pick Order'].values.astype(int))
                    
        while len(target_guide_pairs) < constructs_per_gene:
            
            # print a warning if we run out of guides for a target
            if len(spacer_candidates)<2 :
                print('ran out of guides for target ',target)
                break
         
            spacer_a = spacer_candidates[0]
            spacer_candidates.remove(spacer_a)
            spacer_a_po = spacer_pickorders[0]
            spacer_pickorders.remove(spacer_a_po)
            
            spacer_b = spacer_candidates[second_selection_index]
            spacer_candidates.remove(spacer_b)
            spacer_b_po = spacer_pickorders[second_selection_index]
            spacer_pickorders.remove(spacer_b_po)

            target_version = len(target_guide_pairs) + 1
            
            if random.getrandbits(1):
                target_guide_pairs.append([target, target_symbol, spacer_a, spacer_b, spacer_a_po, spacer_b_po, target_version])
            else:
                target_guide_pairs.append([target, target_symbol, spacer_b, spacer_a, spacer_b_po, spacer_a_po, target_version])

        all_guide_pairs += target_guide_pairs
        
    return pd.DataFrame(
        data=all_guide_pairs, 
        columns=['target','target_symbol','spacer_1','spacer_2', 'spacer_1_pick_order', 'spacer_2_pick_order', 'target_version']
    )


def pair_guides_single_target_controls(
    n_OR_genes=0, n_OR_constructs_per_gene=0,
    n_intergenic_constructs=100,
    n_nontargeting_constructs=100,
    modality = 'CRISPRko',
):
    """

    modality (str): 'CRISPRko', 'CRISPRi', and 'CRISPRa' are implemented. Changes guide selection for
    olfactory receptors.
    """

    # read in guides targeting olfactory receptors
    if modality == 'CRISPRko':
        df_OR = pd.read_table('input_files/CRISPick_CRISPRko_OR_controls.txt')
    elif modality == 'CRISPRi':
        df_OR = pd.read_table('input_files/CRISPick_CRISPRi_OR_controls.txt')
    elif modality == 'CRISPRa':
        df_OR = pd.read_table('input_files/CRISPick_CRISPRa_OR_controls.txt')    
    else:
        raise ValueError(
            f'modality {modality} not recognized. Options are "CRISPRko", "CRISPRi", and "CRISPRa"')
    
    # select 50 olfactory receptor genes (at random) and design 2 constructs per gene
    OR_targeting_pairs = pair_guides_single_target_CRISPick(
        df_OR, 
        n_genes = n_OR_genes, # select 50 genes from df_OR at random
        constructs_per_gene=n_OR_constructs_per_gene,
    )  
    OR_targeting_pairs['category']='OLFACTORY_RECEPTOR'

    # intergenic controls
    df_intergenic_guides = pd.read_table('input_files/CRISPick_intergenic_guides.txt')
    df_intergenic_guides['Target Gene ID'] = 'INTERGENIC_CONTROL'
    df_intergenic_guides['Target Gene Symbol'] = 'INTERGENIC_CONTROL'
    intergenic_targeting_pairs = pair_guides_single_target_CRISPick(
        df_intergenic_guides, constructs_per_gene=n_intergenic_constructs)
    intergenic_targeting_pairs['category']='INTERGENIC_CONTROL'
    
    # nontargeting controls
    df_nontargeting_guides = pd.read_table('input_files/CRISPick_nontargeting_guides.txt')
    df_nontargeting_guides['Target Gene ID'] = 'NONTARGETING_CONTROL'
    df_nontargeting_guides['Target Gene Symbol'] = 'NONTARGETING_CONTROL'
    non_targeting_pairs = pair_guides_single_target_CRISPick(
        df_nontargeting_guides, constructs_per_gene=n_nontargeting_constructs)
    non_targeting_pairs['category']='NONTARGETING_CONTROL'

    
    return pd.concat([ OR_targeting_pairs,  non_targeting_pairs, intergenic_targeting_pairs,
                      ], ignore_index=True).reset_index(drop=True)


def pair_guides_target_and_control_CRISPick(
    guide_input_df, 
    control_guides_df,
    constructs_per_gene=4,
    tRNA_req='all',
    rand_seed=0,
    ):

    """
    Pair gene targeting guides with controls (e.g. nontargeting, intergenic, olfactory receptors)
    guide input should be CRISPick output format

    guide_input_df (pd.DataFrame): CRISPick output with selected guides targeting genes.
    control_guides_df (pd.DataFrame): CRISPick output with control guides (e.g. nontargeting, 
        intergenic, olfactory receptors). Control guides will be reused.
    constructs_per_gene (int): Number of constructs to design per gene. This many unique 
        gene-targeting guides will be selected, prioritizing by "Pick Order", and paired 
        with control guides
    tRNA_req (str): whether guides (spacer sequences) must be compatible with 'any' tRNA or
        with 'all' tRNAs

    Returns pd.DataFrame with gene-targeting guides paired to control guides
    
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

    """
    gene_pairs (list): a list of CRISPick gene symbols for which single- and dual-targeting
        constructs are to be designed.
    
    GI_input_df (pd.DataFrame): DataFrame that describes the specific way that all constructs
        for each genetic interaction are to be designed. For example, how many single- and 
        dual-targeting constructs should be designed, how pick order should be used in pairing
        guides, how spacers should be positioned, which tRNAs to use, etc. 
        See example in 'input_files/GI_set_design.csv'
    
    guide_input_df (pd.DataFrame): CRISPick output with selected guides targeting genes.
    
    control_guides_df (pd.DataFrame): CRISPick output with control guides (e.g. nontargeting, 
        intergenic, olfactory receptors). Control guides will be reused. Control guides are 
        paired with gene-targeting guides for sigle-gene targeting constructs.
        
    tRNA_req (str): whether guides (spacer sequences) must be compatible with 'any' tRNA or
        with 'all' tRNAs    
    """
    
    rng = np.random.default_rng(seed=rand_seed)

    if isinstance(gene_pairs, pd.DataFrame):
        gene_pairs = gene_pairs.values
    
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

    guide_pairs_df['target_version'] = guide_pairs_df.groupby('target').cumcount()+1

    return guide_pairs_df

