import numpy as np
from sklearn.model_selection import train_test_split


#1. numpy.arange(n)
# 해당 n의 개수만큼 0부터 정수를 생성
# numpy.arange(3,7) - 3부터 7미만까지 정수를 생성
aray = np.arange(3)


#2. numpy.bincount(x), x는 array_like, 1dimension, nonnegative ints
# 해당 list에 0부터 최대값까지의 개수를 각각 셈

np.bincount(np.arange(5))

# array([1,1,1,1,1]) # 왜? np.arange는 0부터 4까지 생성, 각각 1개씩 있으므로 1

np.bincount([0,1,3])
# array([1,1,0,1]) # 최대값인 3까지 개수를 세지만, 2는 없기에 2의 자리는 0


#3. numpy.reshape(x,y)
# 해당 list를 x행 y열로 쪼갬
