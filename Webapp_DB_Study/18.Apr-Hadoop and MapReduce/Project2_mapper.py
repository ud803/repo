import sys

for line in sys.stdin :
    #ip, identity, username, time, request, status, size =
    data = (line.strip().split())
    ip, path = data[0], data[6]
    print(path, ip)
