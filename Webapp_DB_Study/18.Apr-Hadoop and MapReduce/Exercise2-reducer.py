import sys

oldKey = None
TotalSales = 0
count = 0

for line in sys.stdin :
    data = line.strip().split()
    date, sales = data

    thisKey = date

    if oldKey and oldKey != thisKey :
        print("{0}\t{1}".format(oldKey, TotalSales/float(count)))
        TotalSales = 0
        count = 0

    TotalSales += float(sales)
    count += 1
    oldKey = thisKey

if oldKey != None :
    print("{0}\t{1}".format(oldKey, TotalSales/float(count)))
