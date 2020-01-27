import sys

temp_list = list()

for line in sys.stdin :
    path, ip = line.strip().split()
    temp_list.append([path, ip])

temp_list = sorted(temp_list, key = lambda temp_list : temp_list[0])

for item in temp_list :
    print(item)
