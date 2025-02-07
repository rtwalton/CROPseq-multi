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
#                           build oligos for CROPseq-multi one-step cloning
##################################################################################################

def assign_tRNAs(df, method='random', overwrite=False):
    """
    add tRNA assignments to construct designs

    method - not implemented
    overwrite (bool): if True, overwrite any existing tRNA assignments. If False, only assign 
    missing (NaN) tRNA assignments.
    """

    if overwrite:
        tRNA_assignments = []
        for tRNA in ['tRNA_A','tRNA_P','tRNA_Q']:
            tRNA_assignments += [tRNA for i in range(np.ceil(len(df)/3).astype(int))]
        random.shuffle(tRNA_assignments)
        tRNA_assignments = tRNA_assignments[:len(df)]
        df['tRNA'] = tRNA_assignments

    else:
        if 'tRNA' in (df.columns):
            n_tRNAs = int(df['tRNA'].isna().sum())
        else:
            df['tRNA']=pd.NA
            n_tRNAs = len(df)
            
        if n_tRNAs==0:
            return df

        tRNA_assignments = []
        for tRNA in ['tRNA_A','tRNA_P','tRNA_Q']:
            tRNA_assignments += [tRNA for i in range(np.ceil(n_tRNAs/3).astype(int))]
        random.shuffle(tRNA_assignments)
        tRNA_assignments = tRNA_assignments[: n_tRNAs]
        df.loc[df['tRNA'].isna(),'tRNA'] = tRNA_assignments

    return df


def generate_oligos(df, dialout=0):
    """
    wrapper for oligo construction, adding dialout primers
    """
    
    # import dialout primer sequences
    df_dialout = pd.read_csv('input_files/kosuri_dialout_primers.csv')
    # dialout primers are truncated to 14 nt to keep total oligo length within 300 nt
    df_dialout['fwd_short'] = df_dialout['fwd'].str.slice(-14)
    df_dialout['rev_short'] = df_dialout['rev'].str.slice(-14)

    # here we encode all members under the same dialout primer pair
    # sublibraries can be encoded within an order with different dialout primer pairs
    df['dialout']=dialout

    df['dialout_fwd'] = df['dialout'].map(df_dialout['fwd_short'])
    df['dialout_rev'] = df['dialout'].map(df_dialout['rev_short'])

    # build the oligos
    oligo_design_df, failed_designs_df = build_CROPseq_multi_one_step_oligos(df)
    oligo_design_df['oligo_len'] = oligo_design_df['oligo'].str.len()

    if not failed_designs_df.empty:
        print('failed to design the following constructs:')
        display(failed_designs_df)
        sys.exit()

    return oligo_design_df, failed_designs_df

def build_CROPseq_multi_single_guide(
    spacer, iBAR,
    template, dialout_fwd, dialout_rev, 
):
    
    def build_oligo():
        return template.format(
            dialout_fwd = dialout_fwd,
            CSM_BsmBI_left = CSM_BsmBI_left,
            spacer = spacer,
            CSM_stem_2 = CSM_stem_2,
            iBAR = reverse_complement(iBAR), # reverse complement iBAR
            CSM_BsmBI_right = CSM_BsmBI_right,
            dialout_rev = reverse_complement(dialout_rev),
        ).upper()
    
    oligo = build_oligo()

    # check for BsmBI and U6 terminator "TTTT" sites
    if (count_BsmBI(oligo)!=2) | (oligo.find("TTTT")!=-1):
        return None

    return oligo


def build_CROPseq_multi_single_guide_oligos(
    df_guides_input, 
    spacer_col='spacer',
    ibar_col='iBAR', 
    dialout_fwd_col ='dialout_fwd',
    dialout_rev_col ='dialout_rev',
    # column names
    template = template_1_step_oligo_single_guide
):
    
    df_guides = df_guides_input.copy()
    oligos = []
    failed_designs = pd.DataFrame()


    # iterate through and design oligos
    for index, row in df_guides.iterrows():

        oligo, tRNA_selection = build_CROPseq_multi_single_guide(
            row[spacer_col],
            row[ibar_col],
            dialout_fwd=row[dialout_fwd_col],
            dialout_rev=row[dialout_rev_col],
            template = template,
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
        
        oligos.append(oligo)
        
    df_guides['oligo'] = oligos

    return df_guides, failed_designs


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
            template = template,
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
#                            build oligos for CROPseq-multi two-step cloning
##################################################################################################

### NOT FULLY IMPLEMENTED. One-step cloning is recommended.

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

