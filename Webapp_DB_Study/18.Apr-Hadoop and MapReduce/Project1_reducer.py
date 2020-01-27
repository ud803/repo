
import sys
import re

oldKey = None
salesTotal = 0
count = 0
r = re.compile(r'\([\'\"]([\w \']+)[\'\"][ .,]*[\'\"](\d+.?\d*)[\'\"]\)')

for line in sys.stdin :
    m = r.search(line.strip())
    data = m.group(1,2)

    if len(data) != 2 :
        continue
    thisKey, thisSale = data
    thisSale = float(thisSale)
    salesTotal += thisSale
    count += 1

print("{0}\t{1}".format(count, salesTotal))
'''
    if oldKey and oldKey != thisKey :
        print ("{0}\t{1}".format(oldKey, salesTotal))

        salesTotal = 0

    oldKey = thisKey
    salesTotal += float(thisSale)

if oldKey != None :
    print ("{0}\t{1}".format(oldKey, salesTotal))
'''
