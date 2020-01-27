

def solution(L):
    '''
    Google foobar challenge.

    Problem 02-2. please-pass-the-coded-messages
    '''

    three_list = [i for i in L if i%3 == 0]
    not_three_list = [i for i in L if i%3 != 0]
    n = len(not_three_list)
    three_candidates = []
    
    for i in range(2**(n)):
        number = 0
        origin_number = []
        digits = str(bin(i))[2:].zfill(n)
        for idx, digit in enumerate(digits):
            if(digit == '1'):
                number += not_three_list[idx]
                origin_number.append(not_three_list[idx])
        origin_number = sorted(origin_number, reverse=True)
        if(number %3 == 0):
            three_candidates.append((digits, origin_number, sum([int(i) for i in digits])))
    three_candidate = sorted(three_candidates, key=lambda x: (x[2], x[1]), reverse=True)[0][1]
    all_list = three_list + three_candidate

    if(len(all_list) == 0):
        return 0
    return ''.join([str(i) for i in sorted(all_list, reverse=True)])


solution([3, 1, 4, 1, 5, 9])
solution([3, 1, 4, 1])
# solution([3, 1, 4, 1, 5, 9])
# 다 풀고 3의배수 조건 봐보자.