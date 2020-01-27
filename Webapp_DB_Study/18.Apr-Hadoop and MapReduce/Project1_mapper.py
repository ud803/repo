
import sys


#with open('./purchases.txt.ignore') as f:
#        for line in f :
for line in sys.stdin :

    data = line.strip().split("\t")

    if len(data) == 6 :
        date, time, store, item, cost, payment = data
        print("{0}\t{1}".format(item, cost))
