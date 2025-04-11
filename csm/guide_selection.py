import pandas as pd
import numpy as np
import random
import warnings
import iteround
from natsort import natsorted
import matplotlib.pyplot as plt
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
    
    Parameters
    ----------
    guides_df : pandas.DataFrame
        DataFrame containing CRISPick guide output with columns including 'sgRNA Sequence'
    position_req : str, optional
        Requirement for guide compatibility in vector positions. Options are:
        - 'any': guide must work in at least one position (default)
        - 'all': guide must work in all positions
    tRNA_req : str, optional
        tRNA requirement for second position. Options are:
        - 'any': no specific tRNA requirement (default)
        - 'all': guide must work with all tRNAs, passed to check_spacer_2()
    single_guide : bool, optional
        If True, only checks compatibility for single guide format. Default False.

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame containing only guides meeting the position requirements.
        Adds boolean columns 'spacer1_check' and 'spacer2_check' (if not single_guide)
        indicating compatibility in each position.

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


def pair_guides_single_target_CRISPick(
    guide_input_df, 
    constructs_per_gene=3, 
    n_genes = 'all',
    tRNA_req='any', 
    pairing_method = 'pick_sum',
    return_missing_guides = False,
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
    rng = np.random.default_rng(seed=0)

    # if requested, subset n_genes at random, usually to design a set of controls
    if n_genes != 'all':
        if isinstance(n_genes, int):
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
    insufficient_guides = []
    
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
                print(f'ran out of guides for {target}')
                insufficient_guides.append([target, target_symbol,
                                             2*constructs_per_gene - 2*len(target_guide_pairs) - len(spacer_candidates)])
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
    
    guide_pairs = pd.DataFrame(
        data=all_guide_pairs, 
        columns=['target','target_symbol','spacer_1','spacer_2', 'spacer_1_pick_order', 'spacer_2_pick_order', 'target_version']
    )
    insufficient_guides = pd.DataFrame(
        data=insufficient_guides, 
        columns=['target','target_symbol','n_additional_guides_needed']
    )

    if return_missing_guides:
        return guide_pairs, insufficient_guides
    else:
        return guide_pairs

def pair_guides_single_target_controls(
    n_OR_genes=0, n_OR_constructs_per_gene=0,
    n_intergenic_constructs=100,
    n_nontargeting_constructs=100,
    modality = 'CRISPRko',
    ko_intergenic_pairing_method='adjacent'
):
    """
    Pairs guides for single target controls based on the specified modality.

    Parameters
    ----------
    n_OR_genes : int, optional
        Number of olfactory receptor genes to select. Default is 0.
    n_OR_constructs_per_gene : int, optional
        Number of constructs per olfactory receptor gene. Default is 0.
    n_intergenic_constructs : int, optional
        Number of intergenic constructs to design. Default is 100.
        Intergenic constructs will be selected proportionally across chromosomes.
    n_nontargeting_constructs : int, optional
        Number of non-targeting constructs to design. Default is 100.
    modality : str, optional
        Specifies the modality for guide selection. Currently only impacts olfactory receptor guide selection.
        Options are:
        - 'CRISPRko': Knockout modality
        - 'CRISPRi': Interference modality
        - 'CRISPRa': Activation modality
        Default is 'CRISPRko'.
    ko_intergenic_pairing_method : str, optional
        Method for pairing intergenic guides in CRISPRko. Options are:
        - 'adjacent': Pair guides that are roughly adjacent to each other on the same chromosome
        - 'random': Randomly pair guides on the same chromosome
        Default is 'adjacent'.
    Returns
    -------
    pandas.DataFrame
        DataFrame containing paired guides for olfactory receptor, intergenic, and non-targeting controls.
        Columns include 'target', 'target_symbol', 'spacer_1', 'spacer_2', 'spacer_1_pick_order', 
        'spacer_2_pick_order', 'target_version', and 'category'.
    """


    # read in guides targeting olfactory receptors
    if modality in ['CRISPRko', 'CRISPRi', 'CRISPRa']:
        df_OR = pd.read_table(f'input_files/CRISPick_{modality}_OR_controls.txt')
    else:
        raise ValueError(
            f'modality {modality} not recognized. Options are "CRISPRko", "CRISPRi", and "CRISPRa"')
    
    # select n_OR_genes olfactory receptor genes (at random) and design n_OR_constructs_per_gene
    OR_targeting_pairs = pair_guides_single_target_CRISPick(
        df_OR, 
        n_genes = n_OR_genes, # select n_OR_genes olfactory receptor genes (at random)
        constructs_per_gene=n_OR_constructs_per_gene,
    )  
    OR_targeting_pairs['category']='OLFACTORY_RECEPTOR'

    # select intergenic controls, taking care how we pair guides
    # currently this is implemented primarily considering CRISPR-KO design concerns
    intergenic_targeting_pairs = pair_intergenic_guides(n_intergenic_constructs, ko_intergenic_pairing_method)
    
    # select nontargeting controls
    df_nontargeting_guides = pd.read_table('input_files/CRISPick_nontargeting_guides.txt')
    non_targeting_pairs = pair_guides_single_target_CRISPick(
        df_nontargeting_guides, constructs_per_gene=n_nontargeting_constructs)
    non_targeting_pairs['category']='NONTARGETING_CONTROL'
    
    return pd.concat([ OR_targeting_pairs,  non_targeting_pairs, intergenic_targeting_pairs,
                      ], ignore_index=True).reset_index(drop=True)


def pair_intergenic_guides(
        n_intergenic_constructs,
        ko_intergenic_pairing_method,
):
    """
    Pair intergenic guides, taking care to sample chromosomes and considering relative positioning of guides.
    Sample chromosomes approximately according to their size.
    For ko_intergenic_pairing_method = 'random', randomly pair intergenic guides 
        (pre-filtered to match activity and specificity of CRISPick Jacquere selections)
    For ko_intergenic_pairing_method = 'random', pair guides tiling chromosomes, matching activity and 
    specificity of CRISPick Jacquere selections, with inter-guide distances comparable to gene-targeting pairs.
    """
    
    intergenic_targeting_pairs = []

    if ko_intergenic_pairing_method == 'adjacent':

        # open pre-paired set of intergenic guides
        # in this file, 'Target Gene ID' is unique to a guide pair, defined by chromosome and coordinates
        df_intergenic_guides = pd.read_table('input_files/CRISPick_Jacquere_intergenic_guide_pairs.txt')
        df_intergenic_guides.sort_values(['chr','start'], inplace=True)

        # determine how many guides for each chromosome based by distributing approximately by chromosome size
        # actually just by total number of control guides but it's reasonably close
        n_intergenic_constructs_per_gene = { chr:x for (chr, x) in zip(
            df_intergenic_guides.value_counts('chr').index, 
            [int(i) for i in iteround.saferound(
                df_intergenic_guides.value_counts('chr') / len(df_intergenic_guides) * n_intergenic_constructs, 0)
                ])}
        
        # iterate through chromosomes
        for chr, df in df_intergenic_guides.groupby('chr'):
            
            # select re
            if len(df)< 2*n_intergenic_constructs_per_gene[chr]:
                indices = range(len(df))
            else:
                indices = np.linspace(0, len(df) - 1, n_intergenic_constructs_per_gene[chr], dtype=int)

            regions = df.iloc[indices]['Target Gene ID'].values

            for region in regions:
                df_region = df[df['Target Gene ID']==region]
                df_region_pair = pair_guides_single_target_CRISPick(df_region, constructs_per_gene=1)
                intergenic_targeting_pairs.append(df_region_pair)

            # if we run out of guides, print a warning and break
            if len(df)< 2*n_intergenic_constructs_per_gene[chr]:
                print(f'ran out of guides for {chr}')
                break
    
    elif ko_intergenic_pairing_method == 'random':

        # open unpaired set of intergenic guides
        df_intergenic_guides = pd.read_table('input_files/CRISPick_Jacquere_intergenic_guides.txt')

        # distribute selection proportionally across chromosomes
        n_intergenic_constructs_per_gene = { chr:x for (chr, x) in zip(
            df_intergenic_guides.value_counts('Target Gene Symbol').index, 
            [int(i) for i in iteround.saferound(
                df_intergenic_guides.value_counts('Target Gene Symbol') / len(df_intergenic_guides) * n_intergenic_constructs, 0)
                ])}
        
        for target in df_intergenic_guides['Target Gene Symbol'].unique():
                
                target_df = df_intergenic_guides[df_intergenic_guides['Target Gene Symbol'] == target].copy().reset_index(drop=True)
                # select the guide pair, randomly pairing guides within a chromosome
                intergenic_targeting_pairs.append(
                    pair_guides_single_target_CRISPick(
                        target_df, constructs_per_gene=n_intergenic_constructs_per_gene[target]
                        ))
    
    else:
        raise ValueError(f'ko_intergenic_pairing_method {ko_intergenic_pairing_method} not recognized. Options are "adjacent" and "random"')

    intergenic_targeting_pairs = pd.concat(intergenic_targeting_pairs, ignore_index=True)
    intergenic_targeting_pairs['target_version'] = intergenic_targeting_pairs.groupby('target_symbol').cumcount()+1
    intergenic_targeting_pairs['category']='INTERGENIC_CONTROL'

    return intergenic_targeting_pairs




def pair_guides_target_and_control_CRISPick(
    guide_input_df, 
    control_guides_df,
    constructs_per_gene=4,
    tRNA_req='all',
    rand_seed=0,
    ):

    """
    Pairs gene-targeting guides with control guides (e.g. nontargeting, intergenic, olfactory receptors)
      to create dual-guide constructs where one guide targets the gene of interest and the other guide
      targets a randomly selected control.

    Parameters
    ----------
    guide_input_df : pandas.DataFrame
        CRISPick output containing guides targeting genes of interest. Must include columns:
        'sgRNA Sequence', 'Target Gene Symbol', and 'Pick Order'.
    control_guides_df : pandas.DataFrame 
        CRISPick output containing control guides (e.g. nontargeting, intergenic, or olfactory 
        receptor guides). Must include column 'sgRNA Sequence'. Control guides may be reused 
        across multiple constructs.
    constructs_per_gene : int, default 4
        Number of constructs to design per target gene. This many unique gene-targeting guides 
        will be selected per gene, prioritizing by "Pick Order", and each paired with a 
        randomly selected control guide.
    tRNA_req : {'any', 'all'}, default 'all'
        Whether guide sequences must be compatible with any tRNA ('any') or all tRNAs ('all')
        when used in the second position of the construct.
    rand_seed : int, default 0
        Random seed for reproducible guide pairing.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing paired guides with columns:
        - target: Target gene symbol
        - targeting_spacer: Guide sequence targeting the gene
        - control_spacer: Control guide sequence
        - target_spacer_pos: Position (1 or 2) of targeting guide in construct
        - spacer_1: Guide sequence in first position
        - spacer_2: Guide sequence in second position
    
    """
        
    rng = np.random.default_rng(seed=rand_seed)
    
    guide_pairs_df = pd.DataFrame()
    guides_df = guide_input_df.copy()
    controls_df = control_guides_df.copy()

    # check spacer compatibility in each position
    guides_df['spacer1_check'] = guides_df['sgRNA Sequence'].apply(check_spacer_1)
    guides_df['spacer2_check'] = guides_df['sgRNA Sequence'].apply(check_spacer_2, tRNA_req=tRNA_req)
    controls_df['spacer1_check']= controls_df['sgRNA Sequence'].apply(check_spacer_1)
    controls_df['spacer2_check']= controls_df['sgRNA Sequence'].apply(check_spacer_2, tRNA_req=tRNA_req)    

    # require that spacers work in all positions since we're not deciding position yet
    guides_df = guides_df[guides_df['spacer1_check'] & guides_df['spacer2_check']]
    controls_df = controls_df[controls_df['spacer1_check'] & controls_df['spacer2_check']]

    control_candidates = list(controls_df['sgRNA Sequence'].values)

    for target, target_gene_df in guides_df.groupby('Target Gene ID'):
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

    if set(guide_input_df['Target Gene ID']) != set(guide_pairs_df['target']):
        print('failed to design any constructs for gene(s): %s'% ' '.join(
            set(guide_input_df['Target Gene ID']) - set(guide_pairs_df['target'])
        ))

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
    Pairs guides for genetic interaction (GI) experiments. Uses CRISPick-designed guides.
    For two genes, A and B, this is typically used to create constructs targeting combinations
    of A + control, B + control, and A + B. The exact design of a genetic interaction set
    can be customized.

    Parameters
    ----------
    gene_pairs : list
        List of tuples containing pairs of gene symbols (e.g. [('geneA','geneB'), ...])
        for which single- and dual-targeting constructs will be designed.
    
    GI_input_df : pandas.DataFrame
        DataFrame specifying the design parameters for each genetic interaction construct.
        Required columns include:
        - target_pos1, target_pos2: Which target (A/B/ctrl) goes in each position
        - spacer_version_pos1, spacer_version_pos2: Which guide version to use
        - tRNA_version: Which tRNA to use (optional, will randomly assign if not specified)
        See example in 'input_files/GI_set_design.csv'
    
    guide_input_df : pandas.DataFrame
        CRISPick output containing guides targeting the genes of interest.
        Must include columns: 'Target Gene Symbol', 'sgRNA Sequence', 'Pick Order'
    
    control_guides_df : pandas.DataFrame
        CRISPick output containing control guides (e.g. nontargeting, intergenic).
        Must include column 'sgRNA Sequence'. Control guides will be reused and
        paired with gene-targeting guides for single-gene targeting constructs.
        
    tRNA_req : str, optional
        Whether guides must be compatible with 'any' tRNA or with 'all' tRNAs.
        Default is 'all'.
        
    rand_seed : int, optional
        Random seed for reproducibility. Default is 0.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the paired guides and their positions for each construct.
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


def select_single_guides_CRISPick(
    guide_input_df, 
    constructs_per_gene=4,
    target_col = 'Target Gene Symbol',
    spacer_col = 'sgRNA Sequence',
    ):
    """
    Select guides to clone a single guide into CROPseq-multi. This is a non-standard use of CROPseq-multi, 
    analagous to the usage of a standard CROPseq vector (with some added functionalities for iBARs and 
    in situ detection performance).

    **For most applications, including single-gene targeting and combinatorial libraries, dual-guide 
    constructs are preferred.** However, some applications, such as base editor tiling screens, 
    single-guide designs may be more appropriate.

    Parameters
    ----------
    guide_input_df : pandas.DataFrame
        Guides to select from. Often CRISPick output, but requires only two columns:
        - target_col (identifier for the target of the guide)
        - spacer_col (spacer sequence)
            - defaults are the column names in the CRISPick output
    constructs_per_gene : int or str, default 4
        Number of constructs/guides to select per gene. If 'all', all compatible guides will 
        be selected.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing selected guides with columns:
        - target: Target gene symbol
        - spacer: Guide sequence
        - target_version: Version number of the target gene
    """

    selected_guides = []

    # check spacer compatibility
    guide_input_df['spacer_check']= guide_input_df[spacer_col].apply(check_spacer_1, single_guide=True)

    # check if 'Pick Order' is a column in the input df
    if 'Pick Order' not in guide_input_df.columns:
        warnings.warn('Pick Order column not found in input df. No prioritization will be used.')
        guide_input_df['Pick Order'] = 0

    for target, target_df in guide_input_df.groupby(target_col):
        target_df.sort_values('Pick Order', inplace=True)
        
        target_guides = target_df[target_df['spacer_check']]
        # print a warning if we run out of guides for a target
        if constructs_per_gene == 'all':
            pass
        elif (len(target_guides) < constructs_per_gene):
            warnings.warn(f'ran out of guides for {target}')
            if len(target_guides) == 0:
                continue
        else:
            target_guides = target_guides[:constructs_per_gene]
        selected_guides.append(target_guides[[target_col, spacer_col, 'Pick Order']])

    selected_guides = pd.concat(selected_guides, ignore_index=True).rename(
        columns={target_col:'target', spacer_col:'spacer', 'Pick Order':'pick_order'})
    selected_guides.reset_index(inplace=True, drop=True)
    selected_guides['target_version'] = selected_guides.groupby('target').cumcount() + 1

    return selected_guides


def plot_genome_lengths(df_input):

    # Create figure and axis

    plt.figure(figsize=(15, 5))
    ax = plt.gca()

    current_position = 0
    chromosome_boundaries = []
    chromosome_centers = []

    df = df_input.copy()
    # expecting this format for 'target': INTERGENIC_CONTROL_chr<chr>_<start>-<end>
    # e.g. INTERGENIC_CONTROL_chr12_18874105-18878798
    df['chr'] = df['target'].str.split("_").str.get(2)
    df['start'] =  df['target'].str.split("_").str.get(3).str.split('-').str.get(0).astype(int)
    df['end'] =  df['target'].str.split("_").str.get(3).str.split('-').str.get(1).astype(int)

    # Plot each chromosome's data
    for chrom in natsorted(df['chr'].unique()):

        chrom_data = df[df['chr'] == chrom]
        
        # Calculate center position of each region
        centers = (chrom_data['start'] + chrom_data['end']) / 2
        
        # Calculate lengths
        lengths = (chrom_data['end'] - chrom_data['start'])
        
        # Shift centers by current_position
        shifted_centers = centers - centers.min() + current_position
        
        # Plot region centers vs lengths
        ax.scatter(shifted_centers, lengths, alpha=0.5, s=4, label=chrom)
        
        # Store chromosome boundary and center for labeling
        chromosome_boundaries.append(current_position)
        chromosome_centers.append(current_position + (centers.max() - centers.min())/2)
        
        # Update current_position for next chromosome
        current_position = shifted_centers.max() + 1e6  # Add 1Mb gap between chromosomes

    # Customize plot
    ax.set_yscale('log')
    ax.set_xlabel('Relative Position')
    ax.set_ylabel('Region Length (bp)')
    ax.set_title('Region Lengths Across Chromosomes')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.25)

    # Add chromosome labels at the center of each chromosome's data
    ax.set_xticks(chromosome_centers)
    ax.set_xticklabels(natsorted(df['chr'].unique()), rotation=45)

    # Add vertical lines between chromosomes
    for boundary in chromosome_boundaries[1:]:
        ax.axvline(x=boundary - 5e5, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()

