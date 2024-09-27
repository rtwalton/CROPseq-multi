import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Bio.Seq import Seq
import gzip
from tqdm import tqdm



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
		read_1_ref_pos = 21
		read_2_ref_pos = 26

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
			# read 2 is encoded by first line of fastq
			read_sequenceR2 = infileR1.readline().strip()
			infileR1.readline()
			infileR1.readline()
			infileR1.readline()
			# paired read 1 is subsequent entry
			read_sequenceR1 = infileR1.readline().strip()
			infileR1.readline()
			infileR1.readline()

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


def count_constructs(
	lib_info_df, 
	lib_design_input_df,
	use_tRNA=False, 
	iBAR2_UMI=False,
	custom_read_primers = True,
	single_fastq = False,
	max_reads=1e7):
	
	lib_design_df = generate_unique_construct_identifiers(lib_design_input_df, use_tRNA, iBAR2_UMI)

	# df_total = pd.DataFrame()
	df_summary = pd.DataFrame()

	for index, row in tqdm(lib_info_df.iterrows(), total=lib_info_df.shape[0]):
		
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
		
		df_barcodes = extract_barcodes_from_fastq_pair(fastq_pair, 
			use_tRNA=use_tRNA, iBAR2_UMI=iBAR2_UMI, 
			custom_read_primers = custom_read_primers,
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
		
		# df_total = pd.concat([df_total,df_barcodes])

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
		df_summary.loc[row['sample_ID'], 'tot_reads'] = len(df_barcodes)
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
		df_summary.loc[row['sample_ID'], 'pct_mapped'] = df_summary.loc[row['sample_ID'], 'mapped_constructs']/\
			df_summary.loc[row['sample_ID'], 'tot_reads']
		df_summary.loc[row['sample_ID'], 'pct_recombination'] = 1 - \
			df_summary.loc[row['sample_ID'], 'mapped_constructs'] / df_summary.loc[row['sample_ID'], 'all_elements_mapped']
		# dropout and uniformity
		df_summary.loc[row['sample_ID'], 'dropout_count'] = (sublib_design_counts_df[row['sample_ID']]==0).sum()
		df_summary.loc[row['sample_ID'], 'gini_coefficient'] =  gini(sublib_design_counts_df[row['sample_ID']].values)
		df_summary.loc[row['sample_ID'], 'ratio_90_10'] = ratio_9010(sublib_design_counts_df[row['sample_ID']].values)

	return lib_design_counts_df, df_summary

