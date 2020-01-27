'''
                        Preprocessing and Transformation
'''

#앞서 우리는 SVM과 뉴럴 네트워크같은 몇며 알고리즘은 데이터의 스케일링에 민감하다는 것을 보았다.
#여기서는 간단한 데이터 가공을 살펴볼 것이다.

import mglearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
cancer = load_breast_cancer()

mglearn.plots.plot_scaling()
plt.show()
    #그림의 x값은 10~15 사이, y값은 1~9 사이에 있다.
    #그림은 데이터를 변형하는 4가지 방법을 보여준다.


#1. Different Kinds of Preprocessing

    #1) StandardScaler
    #scikit-learn의 StandardScaler는 각 feature마다 평균이 0이고 분산이 1이 되도록 한다.
    #그리고 모든 feature를 같은 크기로 만든다.
    #하지만 이는 각 feature마다 특정 최소값이나 최대값을 지정하지 않는다.

    #2) RobustScaler
    #RobustScaler는 Standard와 비슷하지만, 얘는 median(중위값)과 quartiles(분위값)을 사용한다.
    #이는 특이점(outlier)에 있는 수치들을 무시하도록 만들어준다.

    #3) MinMaxScaler
    #MinMaxScaler는 모든 값들이 정확히 0과 1 사이에 있도록 데이터를 움직인다.

    #4) Normalizer
    #Normalizer는 약간 다른 방식의 스케일링이다.
    #이는 feature 벡터가 1의 길이(Euclidean)를 갖도록 각 데이터를 재조정한다.
    #다시 말해서, 한 데이터를 반경 1인 원(혹은 구)에 투사한다.
    #이는 각 데이터가 서로 다른 숫자에 의해 스케일 된다는 의미이다.
    #이 정규화는 feature 벡터의 크기가 아닌, 데이터의 방향만이 중요할 때 쓰인다.



#2. Applying Data Transformations
    #이제 위의 데이터들을 실제 데이터에 적용시켜보자.

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=1)
print(X_train.shape)
print(X_test.shape)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train)

#transform train data
X_train_scaled = scaler.transform(X_train)

#print dataset properties before and after scaling
print("transformed shape: {}".format(X_train_scaled.shape))
print("per-feature minimum before scaling:\n {}".format(X_train.min(axis=0)))
print("per-feature maximum before scaling:\n {}".format(X_train.max(axis=0)))
print("per-feature minimum after scaling:\n {}".format(X_train_scaled.min(axis=0)))
print("per-feature maximum after scaling:\n {}".format(X_train_scaled.max(axis=0)))



#transform test data
X_test_scaled = scaler.transform(X_test)
#print test data properties after scaling
print("per-feature minimum after scaling:\n{}".format(X_test_scaled.min(axis=0)))
print("per-feature maximum after scaling:\n{}".format(X_test_scaled.max(axis=0)))
    #여기서 값이 0과 1을 벗어나는 것을 알 수 있다.
    #이는 MinMaxScaler가 항상 같은 형태의 변형을 트레이닝과 테스트 셋에 시행하기 때문이다.
    #트레이닝 셋의 min 값을 빼서 범위로 나누기 때문에, 테스트 셋에는 차이가 있다.




#3. Scaling Training and Test Data the Same way
    #테스트 셋에 실험하기 위해서는 정확히 같은 방식의 transform을 취하는 게 중요하다.
    #위와 달리 테스트 셋의 방식을 사용했으면 아래와 같은 결과가 나타난다.
    #즉, 트레이닝 셋의 Scaler를 그대로 쓰는게 맞다는 말!!!

from sklearn.datasets import make_blobs
# make synthetic data
X, _ = make_blobs(n_samples=50, centers=5, random_state=4, cluster_std=2)
# split it into training and test sets
X_train, X_test = train_test_split(X, random_state=5, test_size=.1)

#plot the training and test set
fig, axes = plt.subplots(1, 3, figsize=(13,4))

axes[0].scatter(X_train[:,0], X_train[:,1], c=mglearn.cm2(0), label="Training set", s=60)
axes[0].scatter(X_test[:,0], X_test[:,1], marker='^', c=mglearn.cm2(1), label="Test set", s=60)
axes[0].legend(loc='upper left')
axes[0].set_title("Original Data")

#scale the data using MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

#visualize the properly scaled data
axes[1].scatter(X_train_scaled[:,0], X_train_scaled[:,1], c=mglearn.cm2(0), label="Training set", s=60)
axes[1].scatter(X_test_scaled[:,0], X_test_scaled[:,1], marker='^', c=mglearn.cm2(1), label="Test set", s=60)
axes[1].set_title("Scaled Data")

#rescale the test set separately
#so test set min is 0 and test set max is 1
#DO NOT DO THIS! For illustration purposes only.
test_scaler = MinMaxScaler()
test_scaler.fit(X_test)
X_test_scaled_badly = test_scaler.transform(X_test)

# visualize wrongly scaled data
axes[2].scatter(X_train_scaled[:,0], X_train_scaled[:,1], c=mglearn.cm2(0), label="training set", s=60)
axes[2].scatter(X_test_scaled_badly[:,0], X_test_scaled_badly[:,1], marker='^', c=mglearn.cm2(1), label="test set", s=60)
axes[2].set_title("Improperly Scaled Data")

for ax in axes:
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")

plt.show()
    #두 번째와 첫 번째 그림이 완벽히 같은 것을 알 수 있다.(단위를 제외하고)
    #두 번째 그림에서,
    #트레이닝 셋은 0과 1의 최소 최대값을,
    #테스트 셋은 이를 지키지 않는 값을 가지고 있다.
    #세 번째 그림은 테스트 셋의 위치가 잘못된 것을 알 수 있다.
    #그러니까 저렇게 하지마!

'''
ShortCut
scaler = StandardScaler()
1) X_scaled = scaler.fit(X).transform(X)
2) X_scaled_onlyfortraining = scaler.fit_transform(X)
1은 메서드 체이닝 방식이고, 2는 transform 메서드를 가진 모든 모델이 가지고 있는 메서드이다. 트레이닝 셋을 변형할 때는 2가 좋다.
'''




#4. Effect of preprocessing on Supervised learning
    #이제 그 차이를 알았으니, 다시 cancer 데이터로 돌아가자.
    #먼저 가공하지 않은 데이터에 SVM을 적용한다.

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

svm = SVC(C=100)
svm.fit(X_train,y_train)
print("Test set accuracy: {:.2f}".format(svm.score(X_test, y_test)))

#MinMaxScaler를 이용해 가공해보자.
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm.fit(X_train_scaled, y_train)
print("Scaled test set accuracy: {:.2f}".format(svm.score(X_test_scaled, y_test)))
    #데이터 가공이 정말 중요하다는 것을 알 수 있다!
    #모든 스케일러는 같은 인터페이스를 갖고 있기 때문에,
    #그저 scaler = StandardScaler()만 변경해줘도 모두 변경할 수 있다.



'''
여기서 우리는 전처리 작업을 위한 간단한 데이터 변형이 어떻게 이루어지는지 보았다.
다음에는 Transformation의 흥미로운 예제들을 더 살펴보자.
'''
