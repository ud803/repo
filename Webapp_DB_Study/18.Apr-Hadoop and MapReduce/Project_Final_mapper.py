#!usr/bin/python3

import sys
import csv

a = sys.stdin.read()



b = a.split("\"\r\n\"")


for item in b[1:] :
    # header 없애주기 위해서 1부터

    try :
        data = item.split("\t")
        author_id, time = data[3].strip("\""), data[8].strip("\"")[11:13]
        print("{0}\t{1}".format(author_id, time))
    except :
        print(item, "ERROR HERE")
    # 3, 8
