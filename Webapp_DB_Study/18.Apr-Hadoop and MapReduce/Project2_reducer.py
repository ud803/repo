import sys
import re

rule = re.compile(r'\[[\'\"](.*[\']?)[\'\"], \'(.*)\'\]')
oldKey = None
hitTotal = 0


for line in sys.stdin :
    match = rule.match(line.strip())

    try :
        thisKey = match.group(1)
    except :
        print("NOT WORKING", line)
        break
    if oldKey and oldKey != thisKey :
        print("{0}\t{1}".format(oldKey, hitTotal))
        hitTotal = 0

    hitTotal += 1
    oldKey = thisKey

if oldKey != None :
    print("{0}\t{1}".format(oldKey, hitTotal))
