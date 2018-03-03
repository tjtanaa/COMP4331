from collections import defaultdict

# @ param: 	file := filename
# @ return: dict_of_dicts
# @ data:	dict_of_dicts
# 			freq_itemset_dict
# @
# @ description:
# The frequent_itemset_dataset from the 'relim_output.txt' has the output format:
#	frozenset(['803', '209', '722']) 197
# The data has the following format: <frequent itemset> <support>
# When we extract the data, we have to parse the data into usable format
# We chose to strip 'frozenset([\'' and replace '\'])' with ' ' and replace '\', \'' with ' '
# 	output := 803 209 722 197
# Then we convert the string of output into a list:
# 	list := [803 209 722 197]
# To support a faster computation in the algorithmic search, we convert the list into dictionary key/value
# dict_of_dicts has the following format
"""
	Data_Structure of dict_of_dicts:

	dict<key := length ,k, of the frequent itemset, value := dictionary that contains k-length itemset>
	
	For example: 
	dict[3] returns a dictionary that contains 3-itemsets
	The 3-itemsets dictionary contains <key := frequent_itemset, value := support> 
"""
def parseData(file):
	f = open(file,'r')
	freq_itemset_dict = defaultdict()
	dict_of_dicts = defaultdict(lambda: defaultdict(lambda: 0))
	for line in f:
		freq_itemset_dict.clear()
		line = line.lstrip('frozenset([\'').replace('\'])', ' ').replace('\', \'', ' ')
		
		freq_itemset_support = line.split()		
		# <list> freq_itemset_support contains <frequent itemset> <support>
		
		support = freq_itemset_support.pop()
		# <list> freq_itemset_support contains <frequent itemset> <support> so we have to parse the support out
		
		freq_itemset_support.sort()
		# <list> we sort the freq_itemset_support, which now contains the frequent itemset only,
		# so that later on the freq_itemset will look the same during comparison in the algorithm
		
		dict_of_dicts[len(freq_itemset_support)][frozenset(freq_itemset_support)] = support
		# <dict> dict_of_dicts stores the frozenset(frequent_itemset) into the corresponding k-length dictionary
		# the len(freq_itemset_support) determines which k-length dictionary to access
	f.close()
	return dict_of_dicts

# used for debugging the processedData()
def print_processedData(dict_of_dicts):
	for length, _dict in dict_of_dicts.items():
		for itemset, support in _dict.items():
			print itemset
			print support

if __name__ == "__main__":
	dict_of_dicts = parseData('relim_output.txt')
	# print_processedData(dict_of_dicts)
	# print dict_of_dicts[1].values()