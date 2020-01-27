#!usr/bin/python3

import sys

oldKey = None
timeline = [0 for _ in range(24)]

for line in sys.stdin :

    data = line.strip().split()

    author_id, hour = data

    thisKey = author_id

    if oldKey and oldKey != thisKey :
        m = max(timeline)
        argmax = [i+1 for i, j in enumerate(timeline) if j == m ]
        for i in argmax :
            print("{0}\t{1}".format(oldKey, i))
        timeline = [0 for _ in range(24)]

    timeline[int(hour)-1] += 1
    oldKey = thisKey
    #author_id, hour

if oldKey != None :
    m = max(timeline)
    argmax = [i+1 for i, j in enumerate(timeline) if j == m ]
    for i in argmax :
        print("{0}\t{1}".format(oldKey, i))
