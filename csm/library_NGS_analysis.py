import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from Bio.Seq import Seq
import gzip
from tqdm import tqdm
import constants


tRNA_seq_dict = {
	'GGACGAGCCC':'tRNA_P',
	'GGCCAATGCA':'tRNA_G',
	'GGTGGAACCT':'tRNA_Q',
	'AGTACCTCCA':'tRNA_A',
}

def lorenz_curve(X, library_name='library'):
	X_sorted = X.copy()
	X_sorted.sort()
	X_lorenz = X_sorted.cumsum() / X_sorted.sum()
	X_lorenz = np.insert(X_lorenz, 0, 0) 
	X_lorenz[0], X_lorenz[-1]
	fig, ax = plt.subplots(figsize=[3,3])
	## scatter plot of Lorenz curve
	ax.plot(np.arange(X_lorenz.size)/(X_lorenz.size-1), X_lorenz, 
			   color='dodgerblue', label=library_name)
	## line plot of equality
	ax.plot([0,1], [0,1], color='k', label='line of equality')
	plt.xlabel('fraction of reads')
	plt.ylabel('fraction of elements')
	plt.legend()
	
def gini(arr):
	## first sort
	sorted_arr = arr.copy()
	sorted_arr.sort()
	n = arr.size
	coef_ = 2. / n
	const_ = (n + 1.) / n
	weighted_sum = sum([(i+1)*yi for i, yi in enumerate(sorted_arr)])
	return coef_*weighted_sum/(sorted_arr.sum()) - const_

def ratio_9010(arr, percentile=10):
	sorted_arr = arr.copy()
	sorted_arr.sort()
	n_percentile = int(arr.size/percentile)
	return sorted_arr[-n_percentile]/sorted_arr[n_percentile]


def generate_unique_construct_identifiers(df, use_tRNA=False, iBAR2_UMI=False):
	"""
	return df with added unique construct identifier (combination of spacers, iBARs, tRNA)
	"""
	if use_tRNA:
		if iBAR2_UMI:
			df['unique_barcode_combo'] =\
				df['spacer_1'] + '-' + \
				df['iBAR_1'] + '-' + \
				df['tRNA'] + '-' + \
				df['spacer_2']
		else:
			df['unique_barcode_combo'] =\
				df['spacer_1'] + '-' + \
				df['iBAR_1'] + '-' + \
				df['tRNA'] + '-' + \
				df['spacer_2'] + '-' +\
				df['iBAR_2']
	else:
		if iBAR2_UMI:
			df['unique_barcode_combo'] =\
				df['spacer_1'] + '-' + \
				df['iBAR_1'] + '-' + \
				df['spacer_2']
		else:
			df['unique_barcode_combo'] =\
				df['spacer_1'] + '-' + \
				df['iBAR_1'] + '-' + \
				df['spacer_2']+ '-' +\
				df['iBAR_2']

	return df

def extract_barcodes_from_fastq_pair(
	fastq_pair, 
	use_tRNA=False, 
	iBAR2_UMI=False,
	custom_read_primers = True,
	single_fastq = False,
	max_reads=1e7):
	"""
	parse pair of read 1 and read 2 fastq files and extract sequences
	"""

	if custom_read_primers:
		read_1_ref_pos = 0
		read_2_ref_pos = 0
	else:
		read_1_ref_pos = len(constants.read_1_primer_seq)
		read_2_ref_pos = len(constants.read_2_primer_seq)

	# first sgRNA spacer and iBAR in read 1
	spacer1_coords = (read_1_ref_pos,read_1_ref_pos+20)
	iBar1_coords = (read_1_ref_pos+39,read_1_ref_pos+51)

	# second sgRNA spacer and iBAR in read 2
	spacer2_coords = (read_2_ref_pos+31,read_2_ref_pos+51)
	iBar2_coords = (read_2_ref_pos,read_2_ref_pos+12)
	# middle tRNA in read 2
	tRNA_2_coords = (read_2_ref_pos+51,read_2_ref_pos+51+10)

	infileR1 = gzip.open(fastq_pair[0], 'rt')
	if single_fastq:
		infileR2 = gzip.open(fastq_pair[0], 'rt')
	else:
		infileR2 = gzip.open(fastq_pair[1], 'rt')

	fastq_data = []
	total_reads = 0
	# this is slow so probably don't want to run this on tens of million of reads

	while infileR1.readline() and infileR2.readline():

		if not single_fastq:
			# paired reads are corresponding rows in separate files
			read_sequenceR1 = infileR1.readline().strip()
			infileR1.readline()
			infileR1.readline()
			read_sequenceR2 = infileR2.readline().strip()
			infileR2.readline()
			infileR2.readline()
		else:
			# paried reads are alternating entries in single fastq
			read_sequence_1 = infileR1.readline().strip()
			infileR1.readline()
			infileR1.readline()
			infileR1.readline()
			# paired read 1 is subsequent entry
			read_sequence_2 = infileR1.readline().strip()
			infileR1.readline()
			infileR1.readline()
			# use first 8 nt of read to assign to read 1 or read 2
			if (read_sequence_1[:8] == constants.read_1_primer_seq[:8]) or\
				(read_sequence_2[:8] == constants.read_2_primer_seq[:8]):
				read_sequenceR1 = read_sequence_1
				read_sequenceR2 = read_sequence_2
			elif (read_sequence_2[:8] == constants.read_1_primer_seq[:8]) or\
				 (read_sequence_1[:8] == constants.read_2_primer_seq[:8]):
				read_sequenceR1 = read_sequence_2
				read_sequenceR2 = read_sequence_1
			else: # for now, just defaulting to this option so that reads will be counted as 'not mapped'
				read_sequenceR1 = read_sequence_1
				read_sequenceR2 = read_sequence_2

		sg1_seq = read_sequenceR1[spacer1_coords[0]:spacer1_coords[1]]
		# iBARs in SBS orientation (reverse complement of sgRNA orientation)
		iBar1_seq = str(Seq(read_sequenceR1[iBar1_coords[0]:iBar1_coords[1]]).reverse_complement())

		sg2_seq = str(Seq(read_sequenceR2[spacer2_coords[0]:spacer2_coords[1]]).reverse_complement())
		# iBARs in SBS orientation (reverse complement of sgRNA orientation)
		iBar2_seq = read_sequenceR2[iBar2_coords[0]:iBar2_coords[1]]
		
		tRNA_2_seq = str(Seq(read_sequenceR2[tRNA_2_coords[0]:tRNA_2_coords[1]]).reverse_complement())

		fastq_data.append([sg1_seq, iBar1_seq, sg2_seq, iBar2_seq, tRNA_2_seq])
		total_reads += 1
		if total_reads >= max_reads:
			break

	df_barcodes = pd.DataFrame(
		columns =  ['spacer_1', 'iBAR_1', 'spacer_2', 'iBAR_2', 'tRNA_10'], 
		data=fastq_data)

	df_barcodes['tRNA']= df_barcodes['tRNA_10'].map(tRNA_seq_dict)

	df_barcodes = generate_unique_construct_identifiers(df_barcodes, use_tRNA, iBAR2_UMI)

	return df_barcodes

def extract_barcodes_from_fastq_pair_align(
    fastq_pair, 
    use_tRNA=False, 
    iBAR2_UMI=False,
    single_fastq = False,
    spacer_len = 20,
    iBAR_len = 12,
    max_reads=1e7):
    """
    parse pair of read 1 and read 2 fastq files and extract sequences
    """

    infileR1 = gzip.open(fastq_pair[0], 'rt')
    if single_fastq:
        infileR2 = gzip.open(fastq_pair[0], 'rt')
    else:
        infileR2 = gzip.open(fastq_pair[1], 'rt')

    fastq_data = []
    total_reads = 0
    # this is slow so probably don't want to run this on tens of million of reads

    while infileR1.readline() and infileR2.readline():

        if not single_fastq:
            # paired reads are corresponding rows in separate files
            read_sequenceR1 = infileR1.readline().strip()
            infileR1.readline()
            infileR1.readline()
            read_sequenceR2 = infileR2.readline().strip()
            infileR2.readline()
            infileR2.readline()
        else:
            # paried reads are alternating entries in single fastq
            read_sequence_1 = infileR1.readline().strip()
            infileR1.readline()
            infileR1.readline()
            infileR1.readline()
            # paired read 1 is subsequent entry
            read_sequence_2 = infileR1.readline().strip()
            infileR1.readline()
            infileR1.readline()
            # identify which read is which
            if (read_sequence_1.find(constants.CSM_stem_1)!=-1) or\
                (read_sequence_2.find(str(Seq(constants.CSM_stem_2).reverse_complement()))!=-1):
                read_sequenceR1 = read_sequence_1
                read_sequenceR2 = read_sequence_2
            elif (read_sequence_1.find(str(Seq(constants.CSM_stem_2).reverse_complement()))!=-1) or\
                (read_sequence_2.find(constants.CSM_stem_1)!=-1):
                read_sequenceR1 = read_sequence_2
                read_sequenceR2 = read_sequence_1
            else: # for now, just defaulting to this option so that reads will be counted as 'not mapped'
                read_sequenceR1 = read_sequence_1
                read_sequenceR2 = read_sequence_2

        read_1_ref_pos = read_sequenceR1.find(constants.CSM_stem_1)
        read_2_ref_pos = read_sequenceR2.find(str(Seq(constants.CSM_stem_2).reverse_complement()))
    
        # first sgRNA spacer and iBAR in read 1
        spacer1_coords = (read_1_ref_pos-spacer_len, read_1_ref_pos)
        iBar1_coords = (read_1_ref_pos+len(constants.CSM_stem_1), read_1_ref_pos+len(constants.CSM_stem_1)+iBAR_len)
        
        # second sgRNA spacer and iBAR in read 2
        spacer2_coords = (read_2_ref_pos+len(constants.CSM_stem_2), read_2_ref_pos+len(constants.CSM_stem_2)+spacer_len)
        iBar2_coords = (read_2_ref_pos-iBAR_len, read_2_ref_pos)
        # middle tRNA in read 2
        tRNA_2_coords = (spacer2_coords[1], spacer2_coords[1]+10) # use 10 nt for identification

        def extract_seq(read, cooridnates):
            try:
                return read[cooridnates[0]:cooridnates[1]]
            except:
                return 'N'*int(cooridnates[1]-cooridnates[0])
        
        sg1_seq = extract_seq(read_sequenceR1, spacer1_coords)
        iBar1_seq = str(Seq(extract_seq(read_sequenceR1, iBar1_coords)).reverse_complement())
        sg2_seq = str(Seq(extract_seq(read_sequenceR2, spacer2_coords)).reverse_complement())
        iBar2_seq = extract_seq(read_sequenceR2, iBar2_coords)
        tRNA_2_seq = str(Seq(extract_seq(read_sequenceR2, tRNA_2_coords)).reverse_complement())
        
        fastq_data.append([sg1_seq, iBar1_seq, sg2_seq, iBar2_seq, tRNA_2_seq])
        total_reads += 1
        if total_reads >= max_reads:
            break

    df_barcodes = pd.DataFrame(
        columns =  ['spacer_1', 'iBAR_1', 'spacer_2', 'iBAR_2', 'tRNA_10'], 
        data=fastq_data)

    df_barcodes['tRNA']= df_barcodes['tRNA_10'].map(tRNA_seq_dict)

    df_barcodes = generate_unique_construct_identifiers(df_barcodes, use_tRNA, iBAR2_UMI)

    return df_barcodes

def count_constructs(
    lib_info_df, 
    lib_design_input_df,
    use_tRNA=False, 
    iBAR2_UMI=False,
    custom_read_primers = True,
    single_fastq = False,
    return_raw_barcodes=False,
    method = 'position', # or 'align' - either input fixed positions or try to align sequences
    max_reads=1e7):
    
    # generate unique construct identifiers for library design if one design is passed as argument
    if lib_design_input_df is not None:
        lib_design_df = generate_unique_construct_identifiers(lib_design_input_df, use_tRNA, iBAR2_UMI)

    df_total = pd.DataFrame()
    df_summary = pd.DataFrame()

    for index, row in tqdm(lib_info_df.iterrows(), total=lib_info_df.shape[0]):
        
        # generate unique construct identifiers for library design if construct design is not passed as argument
        # and instead is specified in the library info file
        if lib_design_input_df is None:
            lib_design_df = generate_unique_construct_identifiers(pd.read_csv(row['design']), use_tRNA, iBAR2_UMI)

        if 'dialout' in lib_info_df.columns:
            lib_design_index = {
                k:v for (v,k) in lib_design_df[ lib_design_df['dialout'] == row['dialout'] 
                ]['unique_barcode_combo'].to_dict().items()
            }
        else:
            lib_design_index = {
                k:v for (v,k) in lib_design_df['unique_barcode_combo'].to_dict().items()
            }

        fastq_pair = [row['fastq_R1'], row['fastq_R2']]

        if method == 'position':
            df_barcodes = extract_barcodes_from_fastq_pair(fastq_pair, 
                use_tRNA=use_tRNA, iBAR2_UMI=iBAR2_UMI, 
                custom_read_primers = custom_read_primers,
                single_fastq = single_fastq,
                max_reads=max_reads)
        elif method == 'align':
            df_barcodes = extract_barcodes_from_fastq_pair_align(fastq_pair, 
                            use_tRNA=use_tRNA, iBAR2_UMI=iBAR2_UMI, 
                            single_fastq = single_fastq,
                            max_reads=max_reads)
        else:
            warnings.warn("'method' not recognized - will attempt to count constructs by method 'align'")
            df_barcodes = extract_barcodes_from_fastq_pair_align(fastq_pair, 
                use_tRNA=use_tRNA, iBAR2_UMI=iBAR2_UMI, 
                single_fastq = single_fastq,
                max_reads=max_reads)
        
        # map individual elements
        df_barcodes['spacer_1_map'] = df_barcodes['spacer_1'].isin(lib_design_df['spacer_1'])
        df_barcodes['iBAR_1_map'] = df_barcodes['iBAR_1'].isin(lib_design_df['iBAR_1'])
        df_barcodes['spacer_2_map'] = df_barcodes['spacer_2'].isin(lib_design_df['spacer_2'])
        df_barcodes['iBAR_2_map'] = df_barcodes['iBAR_2'].isin(lib_design_df['iBAR_2'])
        df_barcodes['tRNA_map'] = df_barcodes['tRNA'].notna()

        # map full constructs
        df_barcodes['design_index'] = df_barcodes['unique_barcode_combo'].map(lib_design_index)

        df_barcodes['sample']=row['sample_ID']
        df_barcodes['timepoint']=row['timepoint']
        df_barcodes['replicate']=row['replicate']

        if return_raw_barcodes:     
            df_total = pd.concat([df_total,df_barcodes])

        # add oligo counts to library design dataframe
        if index==0:
            lib_design_counts_df = pd.merge(
                lib_design_df, 
                pd.Series(df_barcodes['design_index'].value_counts(),
                    name=row['sample_ID']),
                    left_index=True, right_index=True, how='outer').fillna(0)

        else: 
            lib_design_counts_df = pd.merge(
                lib_design_counts_df, 
                pd.Series(df_barcodes['design_index'].value_counts(),
                    name=row['sample_ID']),
                left_index=True, right_index=True, how='outer').fillna(0)

        if iBAR2_UMI:
            lib_design_counts_df = pd.merge(
                lib_design_counts_df, 
                pd.Series(df_barcodes.groupby('design_index')['iBAR_2'].nunique(),
                    name=row['sample_ID']+'_UMI'),
                    left_index=True, right_index=True, how='outer').fillna(0)

        if 'dialout' in lib_info_df.columns:
            sublib_design_counts_df = lib_design_counts_df[lib_design_counts_df['dialout']==row['dialout']]
        else:
            sublib_design_counts_df = lib_design_counts_df
        # record summary statistics
        df_summary.loc[row['sample_ID'], 'timepoint'] = row['timepoint']
        df_summary.loc[row['sample_ID'], 'replicate'] = row['replicate']
        df_summary.loc[row['sample_ID'], 'n_constructs'] = len(lib_design_df)
        df_summary.loc[row['sample_ID'], 'tot_reads'] = len(df_barcodes)
        df_summary.loc[row['sample_ID'], 'NGS_coverage'] = len(df_barcodes)/len(lib_design_df)
        df_barcodes_mapped = df_barcodes[df_barcodes['design_index'].notna()]

        df_summary.loc[row['sample_ID'], 'spacer_1_map'] = df_barcodes['spacer_1_map'].sum()/len(df_barcodes)
        df_summary.loc[row['sample_ID'], 'iBAR_1_map'] = df_barcodes['iBAR_1_map'].sum()/len(df_barcodes)
        df_summary.loc[row['sample_ID'], 'spacer_2_map'] = df_barcodes['spacer_2_map'].sum()/len(df_barcodes)
        df_summary.loc[row['sample_ID'], 'iBAR_2_map'] = df_barcodes['iBAR_2_map'].sum()/len(df_barcodes)
        df_summary.loc[row['sample_ID'], 'tRNA_map'] = df_barcodes['tRNA_map'].sum()/len(df_barcodes)
        
        if iBAR2_UMI:
            df_summary.loc[row['sample_ID'], 'iBAR_2_UMI'] = sublib_design_counts_df[row['sample_ID']+'_UMI'].mean()

        if use_tRNA:
            if iBAR2_UMI:
                df_summary.loc[row['sample_ID'], 'all_elements_mapped'] =  (
                    df_barcodes['spacer_1_map'] & df_barcodes['iBAR_1_map'] &\
                    df_barcodes['tRNA_map'] &\
                    df_barcodes['spacer_2_map']
                    ).sum()
            else:
                df_summary.loc[row['sample_ID'], 'all_elements_mapped'] =  (
                    df_barcodes['spacer_1_map'] & df_barcodes['iBAR_1_map'] &\
                    df_barcodes['tRNA_map'] &\
                    df_barcodes['spacer_2_map'] & df_barcodes['iBAR_2_map']
                    ).sum()
        else:
            if iBAR2_UMI:
                df_summary.loc[row['sample_ID'], 'all_elements_mapped'] =  (
                    df_barcodes['spacer_1_map'] & df_barcodes['iBAR_1_map'] &\
                    df_barcodes['spacer_2_map']
                    ).sum()
            else:
                df_summary.loc[row['sample_ID'], 'all_elements_mapped'] =  (
                    df_barcodes['spacer_1_map'] & df_barcodes['iBAR_1_map'] &\
                    df_barcodes['spacer_2_map'] & df_barcodes['iBAR_2_map']
                    ).sum()
        
        df_summary.loc[row['sample_ID'], 'mapped_constructs'] = len(df_barcodes_mapped)
        # frequencies of mapping and recombination
        df_summary.loc[row['sample_ID'], 'fraction_mapped'] = df_summary.loc[row['sample_ID'], 'mapped_constructs']/\
            df_summary.loc[row['sample_ID'], 'tot_reads']
        df_summary.loc[row['sample_ID'], 'fraction_recombined'] = 1 - \
            df_summary.loc[row['sample_ID'], 'mapped_constructs'] / df_summary.loc[row['sample_ID'], 'all_elements_mapped']
        # dropout and uniformity
        df_summary.loc[row['sample_ID'], 'dropout_count'] = (sublib_design_counts_df[row['sample_ID']]==0).sum()
        df_summary.loc[row['sample_ID'], 'gini_coefficient'] =  gini(sublib_design_counts_df[row['sample_ID']].values)
        df_summary.loc[row['sample_ID'], 'ratio_90_10'] = ratio_9010(sublib_design_counts_df[row['sample_ID']].values)

    if return_raw_barcodes:
        return lib_design_counts_df, df_summary, df_total
    else:
        return lib_design_counts_df, df_summary


def count_constructs_v2(
    lib_info_df, 
    use_tRNA=False, 
    iBAR2_UMI=False,
    custom_read_primers = True,
    single_fastq = False,
    return_raw_barcodes=False,
    method = 'position', # or 'align' - either input fixed positions or try to align sequences
    max_reads=1e7):
    """
    in this version, a path to the library design csv is passed as a column in the library info file
    so that each sample in the library info file can have a different design

    instead of returning a single design count dataframe, a dictionary of dataframes is returned
    where the keys are the unique design names and the values are dataframes with counts for all samples 
    that correspond to that design
    """

    df_total = pd.DataFrame()
    design_count_dict = {
        design:generate_unique_construct_identifiers(pd.read_csv(design), use_tRNA, iBAR2_UMI
                                                     ) for design in lib_info_df['design'].unique()
        }
    df_summary = pd.DataFrame()

    for index, row in tqdm(lib_info_df.iterrows(), total=lib_info_df.shape[0]):
        
        # generate unique construct identifiers for the library design
        lib_design_df = generate_unique_construct_identifiers(pd.read_csv(row['design']), use_tRNA, iBAR2_UMI)

        if 'dialout' in lib_info_df.columns:
            lib_design_index = {
                k:v for (v,k) in lib_design_df[ lib_design_df['dialout'] == row['dialout'] 
                ]['unique_barcode_combo'].to_dict().items()
            }
        else:
            lib_design_index = {
                k:v for (v,k) in lib_design_df['unique_barcode_combo'].to_dict().items()
            }

        fastq_pair = [row['fastq_R1'], row['fastq_R2']]

        if method == 'position':
            df_barcodes = extract_barcodes_from_fastq_pair(fastq_pair, 
                use_tRNA=use_tRNA, iBAR2_UMI=iBAR2_UMI, 
                custom_read_primers = custom_read_primers,
                single_fastq = single_fastq,
                max_reads=max_reads)
        elif method == 'align':
            df_barcodes = extract_barcodes_from_fastq_pair_align(fastq_pair, 
                            use_tRNA=use_tRNA, iBAR2_UMI=iBAR2_UMI, 
                            single_fastq = single_fastq,
                            max_reads=max_reads)
        else:
            warnings.warn("'method' not recognized - will attempt to count constructs by method 'align'")
            df_barcodes = extract_barcodes_from_fastq_pair_align(fastq_pair, 
                use_tRNA=use_tRNA, iBAR2_UMI=iBAR2_UMI, 
                single_fastq = single_fastq,
                max_reads=max_reads)
        
        # map individual elements
        df_barcodes['spacer_1_map'] = df_barcodes['spacer_1'].isin(lib_design_df['spacer_1'])
        df_barcodes['iBAR_1_map'] = df_barcodes['iBAR_1'].isin(lib_design_df['iBAR_1'])
        df_barcodes['spacer_2_map'] = df_barcodes['spacer_2'].isin(lib_design_df['spacer_2'])
        df_barcodes['iBAR_2_map'] = df_barcodes['iBAR_2'].isin(lib_design_df['iBAR_2'])
        df_barcodes['tRNA_map'] = df_barcodes['tRNA'].notna()

        # map full constructs
        df_barcodes['design_index'] = df_barcodes['unique_barcode_combo'].map(lib_design_index)

        df_barcodes['sample']=row['sample_ID']
        df_barcodes['timepoint']=row['timepoint']
        df_barcodes['replicate']=row['replicate']

        if return_raw_barcodes:     
            df_total = pd.concat([df_total,df_barcodes])

        # add oligo counts to library design dataframe
        design_count_dict[row['design']] = pd.merge(
                design_count_dict[row['design']],
                pd.Series(df_barcodes['design_index'].value_counts(),
                    name=row['sample_ID']),
                    left_index=True, right_index=True, how='outer').fillna(0)
        if iBAR2_UMI:
            design_count_dict[row['design']] = pd.merge(
                design_count_dict[row['design']],
                pd.Series(df_barcodes.groupby('design_index')['iBAR_2'].nunique(),
                    name=row['sample_ID']+'_UMI'),
                    left_index=True, right_index=True, how='outer').fillna(0)
        
        sublib_design_counts_df = design_count_dict[row['design']]
        # record summary statistics
        df_summary.loc[row['sample_ID'], 'timepoint'] = row['timepoint']
        df_summary.loc[row['sample_ID'], 'replicate'] = row['replicate']
        df_summary.loc[row['sample_ID'], 'n_constructs'] = len(lib_design_df)
        df_summary.loc[row['sample_ID'], 'tot_reads'] = len(df_barcodes)
        df_summary.loc[row['sample_ID'], 'NGS_coverage'] = len(df_barcodes)/len(lib_design_df)
        df_barcodes_mapped = df_barcodes[df_barcodes['design_index'].notna()]

        df_summary.loc[row['sample_ID'], 'spacer_1_map'] = df_barcodes['spacer_1_map'].sum()/len(df_barcodes)
        df_summary.loc[row['sample_ID'], 'iBAR_1_map'] = df_barcodes['iBAR_1_map'].sum()/len(df_barcodes)
        df_summary.loc[row['sample_ID'], 'spacer_2_map'] = df_barcodes['spacer_2_map'].sum()/len(df_barcodes)
        df_summary.loc[row['sample_ID'], 'iBAR_2_map'] = df_barcodes['iBAR_2_map'].sum()/len(df_barcodes)
        df_summary.loc[row['sample_ID'], 'tRNA_map'] = df_barcodes['tRNA_map'].sum()/len(df_barcodes)
        
        if iBAR2_UMI:
            df_summary.loc[row['sample_ID'], 'iBAR_2_UMI'] = sublib_design_counts_df[row['sample_ID']+'_UMI'].mean()

        if use_tRNA:
            if iBAR2_UMI:
                df_summary.loc[row['sample_ID'], 'all_elements_mapped'] =  (
                    df_barcodes['spacer_1_map'] & df_barcodes['iBAR_1_map'] &\
                    df_barcodes['tRNA_map'] &\
                    df_barcodes['spacer_2_map']
                    ).sum()
            else:
                df_summary.loc[row['sample_ID'], 'all_elements_mapped'] =  (
                    df_barcodes['spacer_1_map'] & df_barcodes['iBAR_1_map'] &\
                    df_barcodes['tRNA_map'] &\
                    df_barcodes['spacer_2_map'] & df_barcodes['iBAR_2_map']
                    ).sum()
        else:
            if iBAR2_UMI:
                df_summary.loc[row['sample_ID'], 'all_elements_mapped'] =  (
                    df_barcodes['spacer_1_map'] & df_barcodes['iBAR_1_map'] &\
                    df_barcodes['spacer_2_map']
                    ).sum()
            else:
                df_summary.loc[row['sample_ID'], 'all_elements_mapped'] =  (
                    df_barcodes['spacer_1_map'] & df_barcodes['iBAR_1_map'] &\
                    df_barcodes['spacer_2_map'] & df_barcodes['iBAR_2_map']
                    ).sum()
        
        df_summary.loc[row['sample_ID'], 'mapped_constructs'] = len(df_barcodes_mapped)
        # frequencies of mapping and recombination
        df_summary.loc[row['sample_ID'], 'fraction_mapped'] = df_summary.loc[row['sample_ID'], 'mapped_constructs']/\
            df_summary.loc[row['sample_ID'], 'tot_reads']
        df_summary.loc[row['sample_ID'], 'fraction_recombined'] = 1 - \
            df_summary.loc[row['sample_ID'], 'mapped_constructs'] / df_summary.loc[row['sample_ID'], 'all_elements_mapped']
        # dropout and uniformity
        df_summary.loc[row['sample_ID'], 'dropout_count'] = (sublib_design_counts_df[row['sample_ID']]==0).sum()
        df_summary.loc[row['sample_ID'], 'gini_coefficient'] =  gini(sublib_design_counts_df[row['sample_ID']].values)
        df_summary.loc[row['sample_ID'], 'ratio_90_10'] = ratio_9010(sublib_design_counts_df[row['sample_ID']].values)
    
    df_summary.index.rename('sample_ID', inplace=True)
    if return_raw_barcodes:
        return design_count_dict, df_summary, df_total
    else:
        return design_count_dict, df_summary


def plot_summary_metrics(summary_df):
    df_summary_melt = pd.melt(
        summary_df.reset_index(), 
        id_vars='sample_ID', 
        value_vars=['spacer_1_map','iBAR_1_map','spacer_2_map','iBAR_2_map','fraction_mapped']
        )
    plt.figure(figsize=(1.5*len(summary_df),3))
    sns.barplot(df_summary_melt, x='sample_ID', y='value', hue='variable', palette='tab20')
    plt.ylabel('fraction of reads')
    plt.ylim(0,1)
    plt.legend(bbox_to_anchor=(1,1))
    plt.show()

    plt.figure(figsize=(0.5*len(summary_df),3))
    sns.barplot(summary_df, x=summary_df.index, y='fraction_recombined')
    # plt.ylim(0,1)
    plt.ylim(0.0001,1)
    plt.yscale('log')
    plt.xticks(rotation=90)
    plt.show()

    plt.figure(figsize=(0.5*len(summary_df),3))
    sns.barplot(summary_df, x=summary_df.index, y='gini_coefficient')
    plt.xticks(rotation=90)
    plt.show()

    plt.figure(figsize=(0.5*len(summary_df),3))
    sns.barplot(summary_df, x=summary_df.index, y='ratio_90_10')    
    plt.xticks(rotation=90)
    plt.show()

def plot_lorenz_curves(samples_df, counts_dict, share_ax=True):
    """
    plot lorenz curves for all samples in summary_df
    """

    for ref, df in counts_dict.items():
        samples = [s for s in df.columns if s in samples_df['sample_ID'].values]
        for sample in samples:
            lorenz_curve(df[sample].values, library_name=sample)
            plt.title(sample)

def plot_uniformity_histograms(samples_df, counts_dict):
    """
    plot histogram of gini coefficients for all samples in summary_df
    """
    for ref, df in counts_dict.items():
        samples = [s for s in df.columns if s in samples_df['sample_ID'].values]
        for sample in samples:
            plt.figure(figsize=(3,3))
            sns.histplot(
                 df[sample], kde=True, 
                 kde_kws={'bw_adjust': 2, 'cut': 0},
                 color='dodgerblue',
                 )
            plt.title(sample)
            plt.xlabel('reads per construct')
            plt.ylabel('constructs')
            plt.show()