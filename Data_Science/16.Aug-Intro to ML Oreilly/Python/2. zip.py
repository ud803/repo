zip(*iterables)
# 반복 가능한 인자들을 받아서 하나의 tuple로 만든다

x = [1, 2, 3]
y = [4, 5, 6]

zipped = zip(x,y)
# zipped = (1,4), (2,5), (3,6)

zipped = zip(zipped)
# zipped = (1,2,3), (4,5,6)
