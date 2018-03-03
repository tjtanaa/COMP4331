from parseData import parseData

"""
    Closed Frequent Itemset Mining Algorithm:
    Intuition:
    We have already prune all the infrequent itemset using of the frequent itemset mining Algorithm
    Thus mining the Closed_Freq_Itemset is easy as we only need to find those
    frequent itemset that has zero super itemset which has the same frequency as it does.
    The number of comparison has already greatly reduced to just compare among
    the freqent itemsets.
    Those infrequent definitedly has different support than the frequent itemset.

    The parseData has return a suitable data structure for this task.

    We only need to start the with longest length of itemset as the itemsets with the
    longest length is definitedly the Closed Frequent Itemset as they are at the leaf nodes of the
    frequent itemset graph
    Then by level-wise iteration, we iterate through the dictionaries by comparing 
    whether the k-itemset has the same support as the k+1-itemset. 
    (We compare the supports only if k-itemset is the subset of k+1-itemset)
    
    Those k-itemset that does not have the same support as any of its superset 
    will be the Closed Frequent Itemset
"""


def list_of_keys(dict_of_dicts):
    return dict_of_dicts.keys()

def ClosedFI(dict_of_dicts):
    keyList = list_of_keys(dict_of_dicts)
    Closed_Freq_Itemset = []

    for itemset_closed in dict_of_dicts[keyList[-1]].keys():
        Closed_Freq_Itemset.append(itemset_closed)
        

    for i in reversed(range(len(keyList)-1)):
        key = keyList[i]
        key2 = keyList[i+1]
        dict_i = dict_of_dicts[key]
        dict_i_plus = dict_of_dicts[key2]
        itemset_i = dict_i.keys()
        itemset_i_plus = dict_i_plus.keys()
        for itemset1 in itemset_i:
            count = 0
            superset_equal_support_exist = False 
            for itemset2 in itemset_i_plus:
                # print (itemset1.issubset(itemset2))
                if itemset1.issubset(itemset2):
                    count += 1
                    if dict_i[itemset1] == dict_i_plus[itemset2]:
                        superset_equal_support_exist = True
                        break
            if count == 0: # this is the case where all its superset are infrequent
                Closed_Freq_Itemset.append(itemset1)
            elif superset_equal_support_exist == False: 
            # this is the case where it has superset which is frequent however ( its support != support of superset )
                Closed_Freq_Itemset.append(itemset1)
    return Closed_Freq_Itemset

if __name__ == "__main__":
    dict_of_dicts = parseData('relim_output.txt')
    ClosedFrequentItemset = ClosedFI(dict_of_dicts)
    f = open('Closed_Frequent_Itemset.txt' , 'w')
    for CFI in ClosedFrequentItemset:
        f.write(str(CFI))
        f.write('\n')
    f.close()
