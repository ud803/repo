
import sys

temp_list = list()
i = 0

for line in sys.stdin :
    item = line.strip().split("\t")
    temp_list.append(tuple(item))

temp_list = sorted(temp_list, key = lambda temp_list : temp_list[0])

for item in temp_list :
    print(item)
