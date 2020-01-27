

def solution(x, y):
    '''
    Google foobar challenge.

    Problem 03-1. Bomb, baby
    '''
    x = int(x)
    y = int(y)
    count = 0

    while(True):
        if( (x, y) == (1, 1)):
            break

        # Case 1
        if(x > y):
            if y == 1:
                count += (x-1)
                break
            else:
                if(x%y ==0):
                    return "impossible"
                count += x//y
                x %= y
        # Case 2
        elif(x < y):
            if x == 1:
                count += (y-1)
                break
            else:
                if(y%x ==0):
                    return "impossible"
                count += y//x
                y %= x
    return str(count)

solution('7', '4')