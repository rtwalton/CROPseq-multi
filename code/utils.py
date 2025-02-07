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
def check_iBAR1(seq, one_step=True):
    seq=seq.upper()
    seq_w_flank = 'GCAGGA'+reverse_complement(seq)+'GACTGCT'
    if one_step:
        check_RE = contains_BsmBI
    else:
        check_RE = contains_RE
    return not (check_RE(seq_w_flank) | has_homopolymer(seq_w_flank,4))

# check barcode with iBAR2 flanking sequences for forbidden sequence features
def check_iBAR2(seq, one_step=True):
    seq=seq.upper()
    seq_w_flank = 'GCTGGA'+reverse_complement(seq)+'AACATG'
    if one_step:
        check_RE = contains_BsmBI
    else:
        check_RE = contains_RE
    return not (check_RE(seq_w_flank) | has_homopolymer(seq_w_flank,4))

# check candidate spacer 1 with flanking sequences for forbidden sequence features
def check_spacer_1(seq, one_step=True, single_guide = False):
    seq=seq.upper()
    if single_guide:
        seq_w_flank = 'AATGCA'+seq+'GTTTCA' # tracr_2 flanks
    else:
        seq_w_flank = 'AATGCA'+seq+'GTTTGA' # tracr_1 flanks
    if one_step:
        check_RE = contains_BsmBI
    else:
        check_RE = contains_RE
    return (not check_RE(seq_w_flank) | contains_U6_term(seq_w_flank))

# check candidatespacer 2 with flanking sequences for forbidden sequence features
def check_spacer_2(seq, tRNA_req = 'any', one_step=True):
    """
    tRNA_req = 'any' requires that a spacer is compatible with at least one of the tRNAs
    tRNA_req = 'all' requires that a spacer is compatible with all of the tRNAs
    """
    tRNA_flanks = [seq[-6:] for seq in tRNA_seqs.values()] #['CCTCCA','GAGCCC','GAACCT']
    seq=seq.upper()
    seqs_w_flank = [flank+seq+'GTTTCA' for flank in tRNA_flanks]
    if one_step:
        check_RE = contains_BsmBI
    else:
        check_RE = contains_RE
    checks = [not (check_RE(seq_w_flank) | contains_U6_term(seq_w_flank)) for seq_w_flank in seqs_w_flank]
    if tRNA_req == 'any':
        return max(checks)
    elif tRNA_req == 'all':
        return min(checks)


def check_tRNA_leader(seq, tRNA_req = 'all', one_step=True):
    """
    tRNA_req = 'any' requires compatibility with at least one of the tRNAs
    tRNA_req = 'all' requires compatibility with all of the tRNAs
    """
    tRNA_flanks = [seq[:6] for seq in tRNA_seqs.values()]
    seq=seq.upper()
    seqs_w_flank = ['GGCTGC'+seq+flank for flank in tRNA_flanks]
    if one_step:
        check_RE = contains_BsmBI
    else:
        check_RE = contains_RE
    checks = [not (check_RE(seq_w_flank) | contains_U6_term(seq_w_flank)) for seq_w_flank in seqs_w_flank]
    if tRNA_req == 'any':
        return max(checks)
    elif tRNA_req == 'all':
        return min(checks)
