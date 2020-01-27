
import sys
from datetime import datetime

for line in sys.stdin :
    data = line.strip().split()

    if len(data) == 6 :
        date, time, store, item, cost, card = data

        print("{0}\t{1}".format(date, cost))
        # Mon 0 to Sun 6
