def solution(i):
    # Your code here
    skip_bin = {}
    prime_bin = {}

    cur_pos = 2

    prime_list = ''

    while(True):
        triggered = False

        if(cur_pos in skip_bin):
            cur_pos += 1
            continue

        for prime in prime_bin:
            if(cur_pos % prime == 0):
                triggered = True
                break
        if not triggered:
            prime_list += str(cur_pos)
            prime_bin[cur_pos] = 1
            skip_bin[cur_pos * cur_pos] = 1
        cur_pos += 1

        if(len(prime_list) >= i+5):
            break
    print(prime_list[i:i+5])

solution(3)