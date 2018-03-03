from pymining import itemmining
import timeit
def get_dataset(filename):
	f = open(filename, 'r')
	for line in f:
		line = line.strip().rstrip('\n')
		yield line.split(' ')
	f.close()

def print_Output(freq_itemset):
	f = open('relim_output.txt', 'w')
	for itemset, support in freq_itemset.items():
		f.write(str(itemset) + ' ' + str(support) + '\n')
	f.close()

transactions = get_dataset('freq_items_dataset.txt')

start = timeit.default_timer()
print start
relim_input = itemmining.get_relim_input(transactions)
mid = timeit.default_timer()
print mid
freq_itemset = itemmining.relim(relim_input, min_support=100)
end = timeit.default_timer()

f = open('relim_time.txt', 'w')
f.write("start: %f \n" % start)
f.write("mid: %f \n" % mid)
f.write("end: %f \n"% end)
f.close()
print_Output(freq_itemset)