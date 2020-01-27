
import re
import csv
import sys
'''
with open('forum_node.tsv.ignore', 'r') as csvfile :
    reader = csv.reader(csvfile, delimiter='\t')
    i = 0
    for item[4] in reader :
'''
count = 0
num = 0
RULE = re.compile(r'\bfantastic\b', flags=re.IGNORECASE)
data = sys.stdin.read()
b = data.split("\r\n")

for (i, item) in zip(range(len(b)), b) :
    c = item.split("\t")

    try :
        if (RULE.search(c[4])) :
            num = len(RULE.findall(c[4]))
            count += num
            print(c[0], RULE.findall(c[4]))
    except :
        pass

print(count)
