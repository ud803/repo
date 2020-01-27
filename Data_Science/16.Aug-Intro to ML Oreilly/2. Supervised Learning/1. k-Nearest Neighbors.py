'''
k-NN algo is the simplest machine learning algorithm. Building the model consists only of storing the training dataset. To make a prediction for a new data point, the algo finds the closes data points in the training dataset - its "nearest neighbors."
'''


'''
    Classification
'''
# 1. k-Neighbors Classification
    # 가장 단순한 형태는 가장 가까운 오직 하나의 점만을 고려하는 방법이다.
    # 우리가 예측하고 싶은 데이터와 가장 가까운 트레이닝셋의 라벨이 예측값이 된다.
    # 하나의 데이터가 아닌, k개의 이웃으로부터 가까운 알고리즘을 만들 수도 있다.
    # 하나 이상의 이웃일 때, 우리는 "voting"을 써서 라벨을 결정한다.
    # 즉, 각 클래스 별로 vote를 하여 가장 가까운 값으로 라벨이 결정된다.

# forge 예제 및 필요 라이브러리 불러오기
from sklearn.model_selection import train_test_split
import mglearn
import numpy as np
import matplotlib.pyplot as plt
X, y = mglearn.datasets.make_forge()

# X,y를 Test set과 Training set으로 분리함
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)

# kNN 함수 호출하여 인스턴스화, 여기서 이웃(k의 값)은 3
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)

# kNN 모델에 training set을 대입
clf.fit(X_train, y_train)

# Test set 예측 및 점수 측정
print("\nkNN(n=3) score\n")
print("Test set predictions: {}".format(clf.predict(X_test)))
print("Test set accuracy: {:.2f}".format(clf.score(X_test,y_test)))


# 2. Analyzing KNeighborsClassifier
    # 2차원 x,y plane에서 각 지점이 어느 클래스에 속하는지를 시각화
    # k가 1, 3, 9 일 때의 모습이다.
    # 그림을 참고했을 때,
    # k의 값이 작아질수록 더 복잡한 모델을 의미하고
    # k의 값이 커질수록 더 간단한 모델을 의미한다 (실제로 경계선이 부드러워진다)
    # 만약 k = n of samples라면 최대 sample이 모든 지점에서의 label이 된다
fig, axes = plt.subplots(1,3,figsize=(10,3))

for n_neighbors, ax in zip([1,3,9], axes) :
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X,y)
    mglearn.plots.plot_2d_separator(clf,X,fill=True, eps=0.5,ax=ax, alpha=.4)
    mglearn.discrete_scatter(X[:,0],X[:,1],y,ax=ax)
    ax.set_title("{} neighbors(s)".format(n_neighbors))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
axes[0].legend(loc=3)
#plt.show()

'''
그렇다면 여기서 '모델의 복잡도'와 'generalization' 사이의 관계를 말할 수 있을까?
forge 모델은 힘들고, 데이터가 많은 실제 사례를 가져와야 한다.
아래에서부터는 Breast Cancer 예제를 사용
'''

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=66)

training_accuracy =[]
test_accuracy = []
# k의 자리에 1부터 10까지 대입
neighbors_settings = range(1,11)

for n_neighbors in neighbors_settings:
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train,y_train)
    training_accuracy.append(clf.score(X_train,y_train))
    test_accuracy.append(clf.score(X_test,y_test))

plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
#plt.show()

    # 결과적으로,
    # k = 1 일 때의 훈련 정확도는 100%이다. 하지만 k가 증가할수록 모델은 단순해지고, 훈련의 정확도도 떨이진다.
    # k = 1 일 때의 테스트 정확도는 k가 클 때보다 더 낮다. 즉, 모델이 너무 복잡하다는 말이다.
    # k = 10일 때는 모델이 너무 단순하기 때문에 정확도도 더 떨어진다.
    # 최상의 성과는 중간 지점, 즉 k = 6 근일 때이다.



'''
    Regression
'''
# 3. k-Neighbors Regression
    # kNN 방식으로 Regression을 할 수도 있다.
    # 여기서는 wave dataset을 사용한다.
    # 그래프를 보면, x축 기준으로 가장 가까운 것이 near의 의미이다.

'''
Figure 2-8, 2-9
mglearn.plots.plot_knn_regression(n_neighbors=3)
plt.show()
'''

from sklearn.neighbors import KNeighborsRegressor

X,y = mglearn.datasets.make_wave(n_samples=40)

# X, y를 test와 training으로 분리
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)

# 모델을 인스턴스화 하고 k를 3으로 설정
reg = KNeighborsRegressor(n_neighbors=3)

# 모델에 데이터 입력
reg.fit(X_train,y_train)

# test set 예측 및 정확도 측정
print("Test set predictions:\n{}".format(reg.predict(X_test)))
print("Test set R^2: {:.2f}".format(reg.score(X_test,y_test)))



# 4. Analyzing KNeighborsRegressor
    # Classification과 마찬가지로 각 지점에서의 값을 예측해볼 수 있다.

fig, axes = plt.subplots(1,3,figsize=(15,4))
# create 1000 data points, evenly spaced btw -3 and 3
line = np.linspace(-3,3,1000).reshape(-1,1)
for n_neighbors, ax in zip([1,3,9],axes):
    # make predictions using 1, 3, 9 neighbors
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train,y_train)
    ax.plot(line,reg.predict(line))
    ax.plot(X_train,y_train,'^',c=mglearn.cm2(0),markersize=8)
    ax.plot(X_test,y_test,'v',c=mglearn.cm2(1),markersize=8)

    ax.set_title(
        "{} neighbor(s)\n train score: {:.2f} test score: {:.2f}".format(n_neighbors, reg.score(X_train,y_train), reg.score(X_test,y_test))
    )
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
axes[0].legend(["Model predictions", "Training data/target", "Test data/target"], loc="best")
#plt.show()



''' kNN 마무리
KNeighbors classifier에 중요한 요소 2가지
    1. Number of neighbors
    2. How you measure distance btw data points

Strengths & Weaknesses
    1. 모델이 이해하기 쉬움
    2. 별 수정 없이 성능이 좋음 -> 고도의 알고리즘 사용 전에 테스트 가능
    3. 모델 설정 자체는 빠르지만, 데이터가 늘어나면 속도가 느려진다
        -> preprocess를 거쳐야함 (Ch3 참고)
    4. feature가 많은 dataset에 적절하지 않다 (특히 sparse dataset)

So, while the kNN algo is easy to understand, it is not often used in practice, due to prediction being slow and its inability to handle many features.!
'''
