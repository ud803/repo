'''
                            Naive Bayes Model
'''



'''
나이브 베이즈 분류는 선형 모델과 매우 유사하다. 하지만, 훈련 과정이 훨씬 더 빠르다.
하지만 그 대가로서 generalization 성능이 LogisticRegression이나 SVC보다는 떨어진다.

나이브 베이즈 모델이 빠른 이유는, 각각의 특성에 대해 클래스별 통계를 내어 파라미터를 얻기 때문이다. scikit-learn에는 3가지 베이즈 모델이 제공된다.
    GaussianNB, BernoulliNB, MultinomialNB

가우시안 NB는 연속 데이터에 적용될 수 있다.

베르누이 NB는 이항 데이터를 가정한다.

MultinomialNB는 카운트 데이터 (횟수나 개수를 센 데이터)를 가정한다.
텍스트 데이터 분류에서는 베르누이NB와 MultinomialNB를 많이 사용한다.
'''

import numpy as np

X = np.array([0,1,0,1], [1,0,1,1], [0,0,0,1], [1,0,1,0])

y = np.array([0,1,0,1])
    # 4개의 데이터셋을 형성했다.
    # 클래스는 0,1로 2개이다 (y)
    # 클래스 0을 보면, 특성1은 두 번 0이 나왔고, 0번 nonzero가 나왔다.
    #   특성2는 한 번 0이 나왔고, 1번 nonzero가 나왔다.
    # 이런 식으로 카운트를 하는 것.


#그 카운트를 하는 것은 아래와 같은 모습이다.
counts = {}
for label in np.unique(y):
    # iterate over each class
    # count (sum) entries of 1 per feature
    counts[label] = X[y == label].sum(axis=0)
print("Feature counts:\n{}".format(counts))


'''
나머지 두 베이즈 모델은 어떤 통계를 다루느냐에 따라 약간 다르다.

MultinomialNB는 각 클래스별 각 feature의 평균 값을 계산한다.
GaussianNB는 평균 값뿐 아니라 표준편차도 계산한다.

예측을 위해서, 한 데이터 값을 각 클래스의 통계수치에 비교해서 가장 잘 맞는 것을 예측한다.
흥미로운 사실은, MultinomialNB와 BernoulliNB 모두 이 방식을 취할 시 선형 모델의 공식과 같아진다는 사실이다. 하지만 w가 갖는 의미는 다르므로 주의하자.
'''


'''Bayes 마무리

1) MultinomialNB, BernoulliNB

MultinomialNB와 BernoulliNB는 alpha라는 단일 파라미터를 가지고 있다.
이 alpha는 모델의 복잡도를 결정한다. alpha는 다음과 같이 작동한다.
    알고리즘은 데이터셋에 alpha개 만큼의 가상 데이터를 추가한다.
    이 데이터는 모든 특성이 양의 값인 것들이다.
    이는 통계를 'smooth'하게 만들어준다.
alpha가 크면 smoothing이 더 크고, 더 간단한 모델을 의미한다.
하지만 alpha가 알고리즘의 성능에 큰 영향을 미치지는 않는다.


2) GaussianNB

나머지 둘은 문자열과 같은 분리된 카운트 데이터에 사용되는 반면, GaussianNB는 아주 고차원의 데이터에 주로 사용된다. MultinomialNB는 BernoulliNB보다 nonzero feature가 많을 때 성능이 좋다.


3)

나이브 베이즈 모형은 선형 모형과 그 특징이 비슷하다.
나이브 베이즈 모형은 훌륭한 기초 모형이고, 그렇기에 아주 큰 데이터셋에 많이 사용된다.
(그리고 선형 모형보다도 그 속도가 빠르다!)
