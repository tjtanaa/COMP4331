from parseData import parseData

"""
    Maximal Frequent Itemset Mining Algorithm:
    Intuition:
    We have already prune all the infrequent itemset using of the frequent itemset mining Algorithm
    Thus mining the Maximal_Freq_Itemset is easy as we only need to find those
    frequent itemset that has zero super itemset
    The number of comparison has already greatly reduced to just compare among
    the freuent itemsets
    The parseData has return a suitable data structure for this task.
    We only need to start the with longest length of itemset as the itemsets with the
    longest length is definitedly the Maximal Frequent Itemset as they are at the leaf nodes of the
    frequent itemset graph
    The by level wise, we iterate through the dictionaries by comparing whether the k-itemset 
    is the subset of any k+1-itemset.
    Those k-itemset that does not have any superset will be the Maximal Frequent Itemset
"""



def list_of_keys(dict_of_dicts):
    return dict_of_dicts.keys()

def MaxFI(dict_of_dicts):
    keyList = list_of_keys(dict_of_dicts)
    Maximal_Freq_Itemset = []
    # the itemsets with the longest length is definitedly the Maximal Frequent Itemset 
    # as they are at the leaf nodes of the frequent itemset graph
    for itemset_max in dict_of_dicts[keyList[-1]].keys():
        Maximal_Freq_Itemset.append(itemset_max)
        
    for i in reversed(range(len(keyList)-1)):
        key = keyList[i]
        key2 = keyList[i+1]
        dict_i = dict_of_dicts[key]
        dict_i_plus = dict_of_dicts[key2]
        itemset_i = dict_i.keys()
        itemset_i_plus = dict_i_plus.keys()
        for itemset1 in itemset_i:
            count = 0 # reset the count to zero for each iteration
            for itemset2 in itemset_i_plus:
                # print (itemset1.issubset(itemset2))
                if itemset1.issubset(itemset2): 
                # to check whether itemset1 is in itemset2
                # note that the length of itemset1 is i and itemset2 is i+1
                    count += 1 # increment the count by one if the itemset1 is in itemset2
            if count == 0:  # if the count is zero which means that all its superset are infrequent
                            # thus the itemset1 is the Maximal Frequent Itemset
                Maximal_Freq_Itemset.append(itemset1)
    return Maximal_Freq_Itemset

if __name__ == "__main__":
    dict_of_dicts = parseData('relim_output.txt')
    MaximalFrequentItemset = MaxFI(dict_of_dicts)
    f = open('Maximal_Frequent_Itemset.txt' , 'w')
    for MFI in MaximalFrequentItemset:
        f.write(str(MFI))
        f.write('\n')
    f.close()

