import pandas as pd
import warnings

import constants
from utils import *

def generate_feature_reference(
        df_design, 
        capture = '5P', 
        read = 'R2',  
        barcode='spacer', 
        control_categories=['NONTARGETING_CONTROL','INTERGENIC_CONTROL','OR_CONTROL'],
        iBAR2_UMI = False,
        ):
    """
    Generate a feature reference file for use with 10X Cell Ranger feature barcoding.

     Parameters
    ----------
    df_design : pd.DataFrame
        Dataframe containing the library design
    capture: str, default = '5P'
        Capture technology. Options are "5P" for 5' and "3P" for 3' capture.
    read: str, default = 'R2'
        Which read contains the barcode. Options are 'R1' and 'R2'.
    map_feature : str, default = 'spacer'
        Which feature to use for mapping reads. Options are 'spacer', 'iBAR', or 'both'
    control_categories: list
        List of values in "category" column to use as controls for built-in CellRanger differential 
        expression analysis.
        Default is ['NONTARGETING_CONTROL','INTERGENIC_CONTROL','OR_CONTROL']
    iBAR2_UMI: bool, default = False
        Whether iBAR2 has been repurposed as a UMI. If True, sgRNA2 will be mapped using the spacer.
        # TODO: not sure how to collect UMI information yet

    Returns
    -------
    pandas.DataFrame
        Dataframe with one row per sgRNA (two per construct), formatted as a feature reference csv for
        10X Cell Ranger feature barcoding.

    """
    df = df_design.copy()
    # for now, just replacing iBAR2 with N bases if it's a UMI
    #TODO: make this better... not sure how Cell Ranger will handle these yet
    if iBAR2_UMI:
        df['iBAR_2'] = 12*'N'

        if barcode != 'both':
            warnings.warn('Using iBAR2 as a UMI requires using barcode="both". Defaulting to barcode="both"')
            barcode='both'
    
    # start by making unique ID column
    df['id'] = df['target'].astype(str) + '_v' + df['target_version'].astype(str)
    if not df['id'].is_unique:
        # add another indexer to force all IDs to be unique
        df['id'] = df['id'] + '_' + (df.groupby('target').cumcount()+1).astype(str)
    
    df['name'] = df['target_symbol'] + '-v' + df['target_version'].astype(str)

    # generate patterns and barcodes
    df[['sgRNA_1_pattern', 'sgRNA_1_BC_seq','sgRNA_2_pattern', 'sgRNA_2_BC_seq']] = feature_reference_pattern(
        df, capture=capture, map_feature=barcode, read=read)

    df['read']=read
    df['feature_type'] = 'CRISPR Guide Capture'
    df['target_gene_id'] = df['target']
    df['target_gene_name'] = df['target_symbol']

    # rename 'target_gene_id' to 'Non-Targeting' for selected controls for differential expression analysis
    # 'Non-Targeting' is the specific term Cell Ranger expects, though it may be inaccurate in the context
    # of other controls (e.g. cutting controls like intergenic and olfactory receptor targeting)
    df.loc[df['category'].isin(control_categories),'target_gene_id'] = 'Non-Targeting'

    # make a separate reference for each sgRNA (two per construct) and add sgRNA position to ID
    df_sgRNA1 = df[['id','name','read','sgRNA_1_pattern','sgRNA_1_BC_seq','feature_type','target_gene_id', 'target_gene_name']].copy()
    df_sgRNA1['id'] = df_sgRNA1['id'] +'-'+ 'sgRNA_1'
    df_sgRNA1.rename(columns={'sgRNA_1_pattern':'pattern','sgRNA_1_BC_seq':'sequence'}, inplace=True)

    df_sgRNA2 = df[['id','name','read','sgRNA_2_pattern','sgRNA_2_BC_seq','feature_type','target_gene_id', 'target_gene_name']].copy()
    df_sgRNA2['id'] = df_sgRNA2['id'] +'-'+ 'sgRNA_2'
    df_sgRNA2.rename(columns={'sgRNA_2_pattern':'pattern','sgRNA_2_BC_seq':'sequence'}, inplace=True)

    df_complete = pd.concat([df_sgRNA1, df_sgRNA2]).sort_values('id').sort_index()

    if not df_complete.apply(lambda x: x.pattern.replace('(BC)' ,x.sequence), axis=1).is_unique:
        warnings.warn('Reads may map to multiple rows in the feature reference. (Barcode sequences in the pattern are not unique)')

    return df_complete



def feature_reference_pattern(df, capture ='5P', map_feature='spacer', read = 'R2'):
    """
    design reference patterns and sequences
    """
    if read == 'R1':

        if capture != '5P':
            raise ValueError(f'read="R1" is only supported for capture="5P". Provided values were capture={capture} and read={read}.')

        if map_feature == 'spacer':

            sgRNA_1_pattern = '(BC)' + constants.CSM_stem_1
            sgRNA_2_pattern = '(BC)' + constants.CSM_stem_2

            df['sgRNA_1_BC_seq'] = df['spacer_1']
            df['sgRNA_2_BC_seq'] = df['spacer_2']


        elif map_feature == 'iBAR':
            
            sgRNA_1_pattern =  constants.CSM_stem_1 + '(BC)'
            sgRNA_2_pattern =  constants.CSM_stem_2 + '(BC)'

            df['sgRNA_1_BC_seq'] = df['iBAR_1'].apply(reverse_complement)
            df['sgRNA_2_BC_seq'] = df['iBAR_2'].apply(reverse_complement)

        elif map_feature == 'both':

            sgRNA_1_pattern =  '(BC)' + constants.CSM_tracr_1[:20]
            sgRNA_2_pattern =  '(BC)' + constants.CSM_tracr_2[:20]

            df['sgRNA_1_BC_seq'] = df['spacer_1'] + constants.CSM_stem_1 + df['iBAR_1'].apply(reverse_complement)
            df['sgRNA_2_BC_seq'] = df['spacer_2'] + constants.CSM_stem_2 + df['iBAR_2'].apply(reverse_complement)
    
    elif read == 'R2':
        # as implemented, this is the same for 3' and 5' capture

        if map_feature == 'spacer':

            sgRNA_1_pattern = reverse_complement(constants.CSM_stem_1) + '(BC)'
            sgRNA_2_pattern = reverse_complement(constants.CSM_stem_2) + '(BC)'

            df['sgRNA_1_BC_seq'] = df['spacer_1'].apply(reverse_complement)
            df['sgRNA_2_BC_seq'] = df['spacer_2'].apply(reverse_complement)


        elif map_feature == 'iBAR':
            
            sgRNA_1_pattern =  '(BC)' + reverse_complement(constants.CSM_stem_1)
            sgRNA_2_pattern =  '(BC)' + reverse_complement(constants.CSM_stem_2)

            df['sgRNA_1_BC_seq'] = df['iBAR_1']
            df['sgRNA_2_BC_seq'] = df['iBAR_2']

        elif map_feature == 'both':

            sgRNA_1_pattern =  reverse_complement(constants.CSM_tracr_1[:20]) + '(BC)'
            sgRNA_2_pattern =  reverse_complement(constants.CSM_tracr_2[:20]) + '(BC)'

            df['sgRNA_1_BC_seq'] = df['iBAR_1'] + reverse_complement(constants.CSM_stem_1) + df['spacer_1'].apply(reverse_complement)
            df['sgRNA_2_BC_seq'] = df['iBAR_2'] + reverse_complement(constants.CSM_stem_2) + df['spacer_2'].apply(reverse_complement)

    df['sgRNA_1_pattern'] = sgRNA_1_pattern
    df['sgRNA_2_pattern'] = sgRNA_2_pattern

    return df[['sgRNA_1_pattern', 'sgRNA_1_BC_seq','sgRNA_2_pattern', 'sgRNA_2_BC_seq']]


def generate_full_length_reference(df, iBAR2_UMI = False, template = constants.template_U6_tRNA_G_dual_guide_ref):
    """
    design reference patterns and sequences for custom alignment workflows
    """

    df_copy = df.copy()

    def build_reference(row):
        return template.format(
            upstream_seq = constants.U6_tRNA_G_seq,
            spacer_1 = row['spacer_1'],
            stem_1 = constants.CSM_stem_1,
            iBAR_1 = reverse_complement(row['iBAR_1']), # reverse complement iBARs
            tracr_1 = constants.CSM_tracr_1,
            tRNA_leader = row['tRNA_leader'],
            tRNA = constants.tRNA_seqs[row['tRNA']],
            spacer_2 = row['spacer_2'],
            stem_2 = constants.CSM_stem_2,
            iBAR_2 = reverse_complement(row['iBAR_2']), # reverse complement iBARs
            tracr_2 = constants.CSM_tracr_2,
        ).upper()

    for index, row in df_copy.iterrows():

        if iBAR2_UMI:
            row['iBAR_2'] = 'N'*len(row['iBAR_2'])
        
        reference_oligo = build_reference(row)
        df_copy.loc[index,'full_reference'] = reference_oligo

    df['full_reference'] = df_copy['full_reference']
    df['full_reference_RC'] = df['full_reference'].apply(reverse_complement)

    return df