import numpy as np
import pandas as pd 
from collections.abc import Iterable, Sequence


class CSV2Latex: 
	'''
		This class provides all functionalities for creating latex tables from a CSV file> 
	'''

	def __init__(self): 

		# all the metrics from confusion matrix that are encoded as numbers between 0 and 1 should be encoded as percentage 
		# thus multiply by 100
		self.recode_list = ['Accuracy', 'PrecisionMacro', 'PrecisionMicro', 'RecallMacro', 'RecallMicro', 'fScoreMacro', 'fScoreMicro']
		# TODO: what to do with: 'LogFScoreSum', 'LogFScoreMean' ??? 

		# a dictionary that stores for each column identifier, whether maximal or minimal is desired
		self.opt_criterion_mapping = {
			'Accuracy' : 'max',
			'Running Loss' : 'min',
			'PrecisionMacro' : 'max',
			'PrecisionMicro' : 'max', 
			'RecallMacro' : 'max', 
			'RecallMicro' : 'max', 
			'fScoreMacro' : 'max', 
			'fScoreMicro' : 'max',
			'AIG ANDs' : 'min', 
			'AIG Levels' : 'min'
		}

		# dict that stores information about the experiment names: 
		# k = experiment name (without run), v = list with numbers of the runs
		self.experiment_names_dict = {} 

		self.df = None 

	def preprocess_csv(self, csv_file, remove_experiment_name_prefix=None, time_stamp=True, drop_columns_list=None): 
		'''
			This method takes a csv file and creates a pre-process pandas dataframe from it. 
			Args: 
				csv_file: full path to csv file - NOTE: first line assumed to define column headers 
				remove_experiment_name_prefix: optional prefix in each experiment name that should be removed 
				time_stamp: Boolean to set to True if at first column the time stamp is encoded
				drop_columns_list: any iterable that includes the column names that should not be used for evaluation
			Returns: 
				the pre-processed data frame (that is also assigned to self.df)
		'''

		# TODO: rename remove_experiment_name_prefix to something more sensefull and allow to replace multiple parts (make it a list)

		with open(csv_file, 'r') as f: 
			lines = f.readlines()
			processed_lines = [] # nested list to store the values of each line (containing one list per line)

			# line 0 = experiment column names 
			# column 0 = time_stamp --> remove 
			# column 1 = experiment name 

			# NOTE: it is always assumed that: 
			# - experiment name is the second column if timestamp is first
			# - experiment name is the first column if there is no timestamp
			if time_stamp: 
				exp_name_pos = 1 
			else: 
				exp_name_pos = 0

			for idx, line in enumerate(lines): 
				l = line.split(';') 
				l[-1] = l[-1].replace('\n', '') # replacing last line's enter 

				if idx > 0: # meaning actual experiment results and not column headers
					
					experiment_name = l[exp_name_pos] # recall l[0] = timestamp

					# remove the prefix if wanted and if it is inside
					if remove_experiment_name_prefix is not None: 
						experiment_name = experiment_name.replace(remove_experiment_name_prefix, '')
						l[exp_name_pos] = experiment_name

					# remove number of run, if inside 
					if 'run' in experiment_name: 
						k = experiment_name.split('_run')[0]
						v = experiment_name.split('_run')[-1]
						if '_' in v: # in case of an additional suffix because of the logic creation
							v = v.split('_')[0] 
						v = int(v)

					else: # no run inside
						k = experiment_name
						v = 0

					# add experiment name to dict 
					if k in self.experiment_names_dict.keys(): 
						self.experiment_names_dict[k].append(v)
					else: 
						self.experiment_names_dict[k] = [v]

					# replacing the name in the the actual data with removed run-identifier
					l[exp_name_pos] = k


				# NOTE: if there is a timestamp it is removed here
				l = np.array(l[exp_name_pos:], dtype=np.str)
				processed_lines.append(l) 

			processed_lines = np.array(processed_lines, dtype=np.object)
			
			# creating a new data frame that stores the information
			# NOTE: at this point the numeric data in the columns is still encoded as strings 
			df = pd.DataFrame(processed_lines[1:], columns=processed_lines[0])
			col_names = df.columns 

			# encoding all columns except for the first one as numeric 
			df[col_names[1:]] = df[col_names[1:]].apply(pd.to_numeric)

			for c in self.recode_list: 
				if df[c].max() <= 1.0 and df[c].min() >= 0.0: 
					df[c] = df[c]*100.0

			# drop columns if this is wanted 
			if drop_columns_list is not None: 
				df = self.df_drop_columns(df, drop_columns_list)

			self.df = df
			return df

	def df_drop_columns(self, df, drop_columns_list): 
		'''
			This method drops the columns in a data frame. 
			Args: 
				df: the pandas data frame
				drop_columns_list: any iterable that includes the column names that should not be used for evaluation
			Returns: 
				data frame with dropped columns
		'''

		assert isinstance(drop_columns_list, Iterable), 'The declarations of columns to drop must iterable.'
		# ignore the column names that should be dropped that do not exist anyway
		l = []
		for e in drop_columns_list: 
			if e in df.columns: 
				l.append(e)
		df = df.drop(columns=l)
		return df


	def df_to_latex(self, df, index=False, col_space=None, float_format=None, column_format=None, longtable=None, escape=None, decimal='.', caption=None, label=None): 
		'''
			This method takes a data frame and creates a latex table string.
			NOTE: first call df_to_top1() or df_to_topn() 
			NOTE: the following args are just wrappers around some arguments the pandas method provided.
			NOTE: the wrapper is created to be consistent with the rest and in order to not support every argument provided by pandas
			Args: 
				df: the pandas dataframe 
		        col_space : The minimum width of each column (int).
		        index: Write row names (index) (bool).
		        float_format: Formatter for floating point numbers. For example %%.2f or {:0.2f} (one-parameter function or str)
		        column_format: Columns format as specified in https://en.wikibooks.org/wiki/LaTeX/Tables, e.g. rcl for 3 (str)
		        longtable: Use a longtable environment instead of tabular. Requires adding a usepackage{longtable} to LaTeX preamble. (bool)
		        escape: When set to False prevents from escaping latex special characters in column names. (bool)
		        decimal: Character recognized as decimal separator, e.g. ',' in Europe. (str, default '.')
		        caption: Latex caption (string)
		        label: Latex label (string)
		'''

		s = str(df.to_latex(index=index, col_space=col_space, float_format=float_format, column_format=column_format, longtable=longtable, escape=escape, decimal=decimal)) # caption=caption, label=label
		
		# removing weird stuff that pandas does to textbf string
		s = s.replace("\\textbackslash textbf\\", '\\textbf')
		s = s.replace('\\}', '}')
		print(s)


	def df_mark_columnwise_winner(self, df): 
		'''
			This method takes a dataframe, turns the values of each column into strings.
			It then replaces the columnwise winner with textbf for the latex table.
			Args: 
				df: pandas dataframe to process
			Returns: 
				the changed pandas data frame (each column already encoded as string values)
		'''

		text_bf_elements = []
		
		# if this option is chosen, the columnwise winner will be marked in bolt
		col_names = df.columns.tolist()
		for col in col_names: 
			# use only columns that have an optimality criterion (e.g. not AIG Inputs)
			if col in self.opt_criterion_mapping.keys():
				df[col] = pd.to_numeric(df[col])
				if self.opt_criterion_mapping[col] == 'max': 
					opt_idx = df[col].idxmax()
				else: 
					opt_idx = df[col].idxmin()
				text_bf_elements.append((col, opt_idx))

		# encoding the whole data frame as string values 
		df[col_names] = df[col_names].astype(str)
		
		# replacing the columnwise winner strings by textbf
		for bf in text_bf_elements: 
			col, idx = bf
			val = df[col].iloc[idx]
			df[col].iloc[idx] = val.replace(val, '\textbf{}{}{}'.format('{', val, '}'))

		return df

	def df_replace_experiment_name_with_columns(self, df, identifiers, replace_strings_dict=None): 
		'''
			This method takes a data frame, where the experiment name is assumed to be at the first column.
			It then replaces this column by columns for each setting that is part of the experiment name. 
			Args: 
				df: the pandas data frame (at least pre-processed, can also already be top1 or topn)
				identifiers: list with identifiers of the experiment name that should define the splits
				replace_strings_dict: optional dictionary with key also being in identifiers and value being a list of sequences (string that should be replaced, string that is the replacement)
			Returns: 
				a changed, new data frame 
		'''

		assert isinstance(identifiers, list), 'Identifiers must be a list of strings.'
		df_copy = df.copy()
		exp_column_identifier = df_copy.columns[0] # name of the first column that encodes the experiment name 
		
		for id in identifiers: 
			df_copy[id] = df_copy[exp_column_identifier] # making a copy of the experiment name column
			df_copy[id] = df_copy[id].astype(str)
			# making first split on the id
			df_copy[id] = [f.split(id)[-1] for f in df_copy[id]]
			# removing the rest of the experiment name 
			df_copy[id] = [f.split('*')[0] for f in df_copy[id]]

			# doring the replacements 
			if replace_strings_dict is not None and id in replace_strings_dict.keys(): 
				error_msg = 'Parameter replace_strings_dict must be a dictionary with list of sequences (that contain two strings each) as values.'
				assert isinstance(replace_strings_dict[id], list), error_msg
				for v in replace_strings_dict[id]: 
					assert isinstance(v, Sequence), error_msg
					# v1 is the string key that should be replaced, v2 is the string key that replaces v1
					v1, v2 = v
					assert isinstance(v1, str), error_msg
					assert isinstance(v1, str), error_msg
					df_copy[id] = [f.replace(v1, v2) for f in df_copy[id]]
		
		# drop the experiment name column
		df_copy = df_copy.drop(columns=[exp_column_identifier])

		# rearranging the columns (move the newly created ones to the front)
		col_names = df_copy.columns.tolist()
		new_cols = col_names[-1*len(identifiers):] + col_names[:-1*len(identifiers)]
		df_copy = df_copy[new_cols]
		return df_copy


	def df_sort_rows(self, df, column_identifiers_list, ascending=False, reindex=True): 
		'''
			This method sorts the rows (observations) by the given columns in the list.
			It then replaces this column by columns for each setting that is part of the experiment name. 
			Args: 
				df: the pandas data frame (at least pre-processed, can also already be top1 or topn)
				column_identifiers_list: list of strings that correspond to row names
				ascending: True if ascending sorting, otherwise descending
				reindex: Boolean to set to True, if the rows and columns should get updated ascending indices after sorting
			Returns: 
				a changed, new data frame 
		'''
		assert isinstance(column_identifiers_list, list), 'Parameter column_identifiers_list must be a list of strings, where each string corresponds to a column name.'
		col_names = df.columns.tolist()
		for c in column_identifiers_list: 
			assert isinstance(c, str), 'Parameter column_identifiers_list must be a list of strings, where each string corresponds to a column name.'
			assert c in col_names, 'ID {} is not a column name of the given dataframe.'.format(c)
		
		df = df.sort_values(by=column_identifiers_list, inplace=False, ascending=ascending)
		if reindex: 
			df = df.reset_index(drop=True)
		return df 


	def df_replace_column_names(self, df, replace_dict): 
		'''
			This method replaces the current column names of a data frame by new ones. 
			Args:
				df: the data frame to change
				replace_dict: optional dictionary with keys being current column names and values being the strings of the new column names
			Returns: 
				dataframe with new column names
		'''
		assert isinstance(replace_dict, dict), 'Parameter replace_dict has to be a dictionary with keys and values being strings.'
		
		for k, v in replace_dict.items(): 
			assert isinstance(k, str), 'Parameter replace_dict has to be a dictionary with keys and values being strings.'
			assert isinstance(v, str), 'Parameter replace_dict has to be a dictionary with keys and values being strings.'
			df = df.rename(columns={k: v})

		return df 


	def df_to_top1(self, df, opt_column='Accuracy'):
		'''
			This method takes a preprocessed data frame (from preprocess_csv method call) and creates the top1 dataframe. 
			This method calculates max for each experiment name with multiple runs and reports it in the table. 
			The opt_column parameter decides for the column to which opt_criterion parameter should be applied.
			Args: 
				df: the created pandas dataframe 
				opt_column: the identifier of the column that should be used for extract the top1 model
				opt_criterion: string that can be max or min - chooses whether minimal or maximal value is best for the opt_column 
			Returns: 
				the changed pandas data frame
		'''

		# now create new data frame that is filled in the following with only one row for each experiment
		col_names = df.columns 
		assert opt_column in col_names, 'There is no column named {} in the data frame. The column names are: {}'.format(opt_column, col_names)
		exp_column_identifier = df.columns[0] # name of the first column that encodes the experiment name 
		df_new  = pd.DataFrame(columns=col_names)

		# iterating over all experiments 
		for k in self.experiment_names_dict.keys(): 
			# from df extract all experiments with the same name 
			df_sub = df[(df[exp_column_identifier] == k)]

			assert opt_column in self.opt_criterion_mapping.keys(), 'The opt_column {} does not exist in attribute self.opt_criterion_mapping. First, create it.'.format(opt_column)
			# extract the index of the optimal value from the chosen column
			if self.opt_criterion_mapping[opt_column] == 'max': 
				opt_idx = df_sub[opt_column].idxmax()
			else: 
				opt_idx = df_sub[opt_column].idxmin()
			
			# fill the corresponding row into the new data frame
			df_new.loc[len(df_new)] = df_sub.loc[opt_idx].tolist()

		return df_new


	def df_to_topn(self, df):
		'''
			This method takes a preprocessed data frame (from preprocess_csv method call) and creates the topn dataframe. 
			This method calculates mean and stdv for each experiment name with multiple runs and reports it in the table as separate column. 
			Args: 
				df: the created pandas dataframe 
			Returns: 
				the changed pandas data frame 
		'''
		# now create new data frame that is filled in the following with only one row for each experiment
		col_names = df.columns.tolist()
		exp_column_identifier = df.columns[0] # name of the first column that encodes the experiment name 
		df_new  = None # pd.DataFrame(columns=col_names)

		# iterating over all experiments 
		for k in self.experiment_names_dict.keys(): 
			# from df extract all experiments with the same name 
			df_sub = df[(df[exp_column_identifier] == k)]

			# calculate the column_wise mean - is a pandas series at this point
			mean = df_sub[col_names[1:]].mean()
			# calculate the columnwise stdv - is a pandas series at this point
			stdv = df_sub[col_names[1:]].std()

			# turn the series into a dataframes where with columns
			df_mean = mean.to_frame().T
			df_stdv = stdv.to_frame().T

			# create the combined strings for each column
			# NOTE: inserting the information of stdv as separate columns into df_mean and using df_mean as a carrier
			for col in df_mean.columns.tolist(): # NOTE: the column names at this point are the same for stdv and mean
				
				# renaming the columns
				df_mean = df_mean.rename(columns={col: col+' Mean'})
				df_stdv = df_stdv.rename(columns={col: col+' Stdv'})
				
			# create a new column at the beginning that encodes the experiment name
			df_mean.insert(0, exp_column_identifier, k)
			
			# doing a concatenation where the columns of mean and stdv are always next to each other 
			# starting with experiment name
			df_concat = df_mean[exp_column_identifier] 

			for c in df_stdv.columns.tolist(): 
				df_concat = pd.concat([df_concat, df_mean[c.replace('Stdv', 'Mean')], df_stdv[c]], axis=1)

			# resetting the indices
			df_concat = df_concat.reset_index(drop=True)

			if df_concat is None: # in the first round assign initialize the new dataframe 
				df_new = df_concat
			else: # fill the corresponding new row into the new data frame - row-wise concatenation
				df_new = pd.concat([df_new, df_concat], axis=0)

		# resetting the row indices
		df_new = df_new.reset_index(drop=True)
		return df_new


	def df_topn_fuse(self, df, decimal_digits_mean=2, decimal_digits_stdv=2):
		'''
			This method takes a topn data frame and fuses the mean and stdv columns. 
			NOTE: sorting should be done beforehand because at this point the column values are encoded as strings afterwards.
			Args: 
				df: the created pandas dataframe 
				decimal_digits_mean: the number of digits after the decimal point that should be shown for the mean representations
				decimal_digits_stdv: the number of digits after the decimal point that should be shown for the stdv representations
			Returns: 
				the changed pandas data frame 
		'''
		# now create new data frame that is filled in the following with only one row for each experiment
		col_names = df.columns.tolist()
		exp_column_identifier = df.columns[0] # name of the first column that encodes the experiment name 

		# get the names of the columns that refer to either the mean or the stdv
		mean_col_names = []
		stdv_col_names = []

		for col in col_names: 
			if 'Stdv' in col: 
				stdv_col_names.append(col)
			if 'Mean' in col: 
				mean_col_names.append(col)

		# now creating subsets of the dataframe to which the corresponding formatting can be applied
		# NOTE: after this operation, the values in the columns are encoded as strings
		s = "{0:." + str(decimal_digits_mean) +"f}"
		df_mean = df[mean_col_names]
		df_mean = df_mean.applymap(s.format)

		s = "{0:." + str(decimal_digits_stdv) +"f}"
		df_stdv = df[stdv_col_names]
		df_stdv = df_stdv.applymap(s.format)

		# now combine the corresponding mean and stdv columns into one by writing the stdv information into the mean column (using as carrier)
		# create the combined strings for each column
		for col in mean_col_names: # NOTE: the column names at this point are the same for stdv and mean
			df_mean[col] = df_mean[col].astype(str) + ' (' + df_stdv[col.replace('Mean', 'Stdv')].astype(str) + ')'
			# rename the column 
			df_mean = df_mean.rename(columns={col: col.replace(' Mean', '')})

		# reconcatenate the experiment name column to the beginning of df_mean
		df_mean.insert(0, exp_column_identifier, df[exp_column_identifier])
			
		# resetting the indices
		return df_mean.reset_index(drop=True)


	def df_combine_top1_topn(self, df_top1, df_topn, sorting='top1'):
		'''
			This method takes a topn data frame and fuses the mean and stdv columns. 
			NOTE: sorting should be done beforehand because at this point the column values are encoded as strings afterwards.
			Args: 
				df_top1: the top1 dataframe 
				df_topn: the topN dataframe 
				sorting: which sorting to use, top1 sorting of rows or topn (string that is either top1 or topn)
			Returns: 
				the combined pandas data frame
		'''

		assert sorting in ['top1', 'topn'], 'Sorting parameter for top1 and topn combination needs to be one of the following: {}'.format(['top1', 'topn'])
		
		# NOTE: this assumes the following points (not checked and might crash otherwise - TODO): 
		# - for both df's the, the experiment name was not yet turned into single columns
		# - top1 and topn have the same experiment name identifier 
		# - top1 and topn cover the same experiments (i.e. same number of rows and observations)

		# defining which sorting of which df is used 
		if sorting == 'top1': 
			df_sorted = df_top1
			df_unsorted = df_topn
		else: 
			df_sorted = df_topn
			df_unsorted = df_top1

		# name of the first column that encodes the experiment name 
		exp_column_identifier = df_sorted.columns[0] 

		# this data frame will contain the content of df_unsorted, but sorted in the way as df_sorted
		df_unsorted_sorted = None

		# sorting the unsorted df
		for exp_name in df_sorted[exp_column_identifier]: 
			unsorted_row = df_unsorted[df_unsorted[exp_column_identifier] == exp_name]
			if df_unsorted_sorted is None: 
				df_unsorted_sorted = unsorted_row
			else: 
				df_unsorted_sorted = pd.concat([df_unsorted_sorted, unsorted_row], axis=0)
		
		# resetting the indices
		df_unsorted_sorted = df_unsorted_sorted.reset_index(drop=True)

		# reassigning to top1 and topn - so that these identifiers can be used again and knowing that they are sorted the same now
		if sorting == 'top1': 
			df_topn_new = df_unsorted_sorted
			df_top1_new = df_top1
		else: 
			df_top1_new = df_unsorted_sorted
			df_topn_new = df_topn
	

		# renaming the columns 
		for col in df_top1_new.columns.tolist(): 
			if col != exp_column_identifier:
				df_top1_new = df_top1_new.rename(columns={col: col + ' (Top1)'})

		# renaming the columns 
		for col in df_topn_new.columns.tolist(): 
			if col != exp_column_identifier:
				df_topn_new = df_topn_new.rename(columns={col: col + ' (TopN)'})

		# now combine the two dfs with each measure of top1 and topn being next to each other
		df_new_col_names = []
		for col in df_top1_new.columns.tolist(): 
			if col == exp_column_identifier:
				df_new_col_names.append(col)
			else: 
				df_new_col_names.append(col)
				df_new_col_names.append(col.replace('Top1', 'TopN'))

		df_new = pd.DataFrame(columns=df_new_col_names)

		for col in df_new_col_names: 
			if 'Top1' in col:
				df_new[col] = df_top1_new[col]
			else: 
				df_new[col] = df_topn_new[col]

		df_new.reset_index(drop=True)
		return df_new 


if __name__ == '__main__':

	# This script can code for Latex tables from the csv-files that are created within this repro's pipeline. 
	# NOTE: again the code below is just example code and probably needs to be adapted by yourself

	# general information
	csv_file = None # TODO: fill in the string with absolute path to your CSV-file
	remove_prefix = '01_coimbra_rf_eval*' # TODO: change this according to your experiment name prefix that you used within experiment_designer.py
	columns_to_drop = ['LogFScoreMean', 'LogFScoreSum', 'Running Loss',  'AIG Inputs', 'AIG Outputs']
	columns_to_drop.extend(['PrecisionMacro', 'PrecisionMicro', 'RecallMacro', 'RecallMicro', 'fScoreMacro', 'fScoreMicro'])
	
	replace_strings_dict = {
		'nodes': [('_', ' - ')],
		'optim': [('sgd', 'SGD'), ('radam', 'RAdam'), ('plain', 'Plain'), ('adam', 'Adam'), ('Adamw', 'AdamW')],
		'norm': [('1', '1: Z-Score Normalization'), ('2', '2: Min-Max-Scaling'), ('3', '3: Max-Abs-Scaling'), ('4', '4: Robust-Scaling'), ('5', '5: Power-Transforming'), ('6', '6. Quantile-Transforming'), ('7', '7. Independent Normalization')],
		'fb': [('_nn_final', '')],
		'tb' : [('tb', '')]
	} 
	

	replace_dict = {
		'nodes' : 'Hidden Layer Nodes', 
		'optim' : 'Optimizer',
		'norm' : 'Normalization Mode',
		'tb' : 'Total Bits',
		'fb' : 'Fractional Bits',
		'rf_estimators': 'Estimators',
		'rf_max_depth' : 'Max. Depth',
		'rf_bitwise' : 'Bitwise',
		'lgn_depth' : 'Depth',
		'lgn_width' : 'Width', 
		'lgn_lutsize': 'LUT Size'
		}

	# general preprocessing
	c = CSV2Latex()
	df = c.preprocess_csv(csv_file, remove_experiment_name_prefix=remove_prefix, drop_columns_list=columns_to_drop)

	# top1 procedure
	print('\n--------- Top1 ---------\n\n')
	df_top1 = c.df_to_top1(df)
	# df_top1 = c.df_sort_rows(df_top1, ['AIG ANDs'], ascending=True)
	df_top1 = c.df_sort_rows(df_top1, ['Accuracy'], ascending=False) 
	df_top1 = c.df_mark_columnwise_winner(df_top1)
	df_top1_replaced = c.df_replace_experiment_name_with_columns(df_top1, ['tb', 'fb', 'lgn_depth', 'lgn_width', 'lgn_lutsize'], replace_strings_dict=replace_strings_dict)
	df_top1_replaced = c.df_replace_column_names(df_top1_replaced, replace_dict)
	c.df_to_latex(df_top1_replaced, caption=None, label=None, index=False, longtable=False)
	# quit()

	# topn procedure
	print('\n--------- TopN ---------\n\n')
	df_topn = c.df_to_topn(df)
	# df_topn = c.df_sort_rows(df_topn, ['AIG ANDs Mean'], ascending=True)
	df_topn = c.df_sort_rows(df_topn, ['Accuracy Mean'], ascending=False)
	df_topn = c.df_topn_fuse(df_topn)
	df_topn_replaced = c.df_replace_experiment_name_with_columns(df_topn, ['tb', 'fb', 'lgn_depth', 'lgn_width', 'lgn_lutsize'], replace_strings_dict=replace_strings_dict)
	df_topn_replaced = c.df_replace_column_names(df_topn_replaced, replace_dict)
	c.df_to_latex(df_topn_replaced, caption=None, label=None, index=True, longtable=True)

	# combining top1 and topn procedure
	print('\n--------- Top1 and TopN Combination ---------\n\n')
	df_combined = c.df_combine_top1_topn(df_top1, df_topn, sorting='top1')
	df_combined = c.df_replace_experiment_name_with_columns(df_combined, ['tb', 'fb', 'lgn_depth', 'lgn_width', 'lgn_lutsize'], replace_strings_dict=replace_strings_dict)
	df_combined = c.df_replace_column_names(df_combined, replace_dict)
	df_combined = c.df_drop_columns(df_combined, ['Total Bits', 'Fractional Bits'])
	c.df_to_latex(df_combined, caption=None, label=None, index=False, longtable=False)

	# TODO for Top1: 
	# 	- decimal point formatting

	# TODO for TopN: 
	# 	- textbf option
	# 	- mean and stv for each experiment where there are multiple runs, and for all others not
	
	# TODO general: 
	# 	- exclude columns option
	# 	- allow to make more table splits
	# 	- extract rows according to a column=... use np.unique and create one table each? 
	# 	- all latex options visible already here 
	# 	- setting commas on AIG levels and nodes


