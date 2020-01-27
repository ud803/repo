from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import mglearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
import numpy as np

def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.show()

where=input("1 for Random Forests, 2 for Gradient Boosting : ")
if(where=="1") :

    '''
    Ensembles(앙상블 러닝)은 다수의 머신 러닝 기법을 조합해 만든 강력한 모델이다. 이 앙상블에 속하는 많은 모델들이 있지만, 큰 데이터셋을 분류하고 회귀하는 데 효과적이라고 알려진 두 개의 앙상블 모델이 있다.
    그 둘 모두 결정 트리를 기본 요소로서 갖는데, 각각 "Random Forests"와 "Gradient Boosted DT"이다.
    '''


    '''
                                        Random Forests
    '''


# Basic Idea
    #결정 트리의 문제는 트레이닝 데이터에 overfit 한다는 것이다. 랜덤 포레스트는 이를 해결할 방법이다.
    #랜덤 포레스트는 기본적으로 결정 트리의 집합인데, 각 트리는 서로 약간씩 다르다.
    #랜덤 포레스트의 아이디어는, 각 트리가 예측은 잘 하지만 데이터의 특정 부분에 overfit한다는 것이다.
    #그렇다면, 많은 트리를 만들게 되면 모두 잘 예측은 하되 서로 다른 방식으로 overfit할 것이고,
    #그 결과를 평균 내어 overfit을 줄일 수 있다.
    #즉, "rigorous mathematics"를 통해 트리의 정확도는 유지하면서 overfit을 줄일 수 있다.


# Implementation
    #먼저 여러 트리를 만드는 것에서 시작한다. 두 가지 방법이 있다.
    #   첫 번째는 트리를 만드는 데 쓸 데이터를 지정하는 것
    #   두 번째는 각 스플릿 테스트에서 feature를 지정하는 것


#1. Building random forests
    #만들 트리의 개수를 정한다. (n_estimators)
    #Bootstrap Sample을 만든다.
    #   *n개의 sample에서 중복 허용하여 1개씩 n번 뽑는 것 (결국 기존 데이터와 크기 같음)
    #원래 결정 트리의 방법처럼 best test를 뽑는 대신, (max_features)
    #   각 노드는 임의로 feature의 집합을 골라서 그 중 하나를 포함하는 best test를 선정
    #따라서 각 노드는 서로 다른 feature의 집합을 가지고 트리를 만듦
    # 1.Bootstrap sampling때문에 각 결정트리는 서로 약간씩 다른 데이터셋을 기준으로 만들어짐
    # 2.각 노드에서도 서로 다른 feature 집합을 기준으로 분리됨
    #이 둘이 합쳐져 모든 트리는 서로 다르게 된다!

        #   *max_feature의 값을
        #   n_feature로 해버리면 모든 feature를 살피는 것이기 때문에 randomness가 사라진다.
        #   1로 하면 각 스플릿은 선택권이 없어진다.
        #   이 값이 크면 랜덤 포레스트의 트리들이 비슷해지는 것이고, 데이터에 잘 맞게 된다.
        #   이 값이 작으면 트리들이 서로 달라지고, 각 트리가 데이터에 잘 맞으려면 deep하게 가야 한다.


#2. Predicting using random forests
    #숲에 있는 모든 트리에 대해서 먼저 예측을 하는데,
    #   회귀에서는, 이 결과들을 평균내어 예측을 한다.
    #   분류에서는 "soft voting" 전략이 사용된다. 이는 각 트리가 가능한 결과값에 대한 확률을 제공하고, 그 확률들을 평균내어 가장 높은 값의 라벨을 받는 것이다.


#3. Analyzing random forests


# Trees on Moon Dataset


    X,y= make_moons(n_samples=100, noise=0.25, random_state=3)
    X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=y, random_state=42)

    forest = RandomForestClassifier(n_estimators=5, random_state=2)
    forest.fit(X_train,y_train)

    fig, axes = plt.subplots(2,3, figsize=(20,10))
    for i, (ax, tree) in enumerate(zip(axes.ravel(),forest.estimators_)):
        ax.set_title("Tree {}".format(i))
        mglearn.plots.plot_tree_partition(X_train,y_train,tree,ax=ax)

    mglearn.plots.plot_2d_separator(forest, X_train, fill=True, ax=axes[-1,-1], alpha=0.4)
    axes[-1,-1].set_title("Random Forest")
    mglearn.discrete_scatter(X_train[:,0], X_train[:,1], y_train)
    plt.show()
        #decision boundary가 확실히 다른 것을 알 수 있다.
        #실제로는 훨씬 많은 트리(수백~수천)를 사용해서 더 부드러운 boundary를 만든다.




# Trees on Breast Cancer Dataset

    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
    forest=RandomForestClassifier(n_estimators=100,random_state=0)
    forest.fit(X_train, y_train)

    print("\nRandom Forest Score\n")
    print("Accuracy on training set: {:.3f}".format(forest.score(X_train,y_train)))
    print("Accuracy on test set: {:.3f}".format(forest.score(X_test,y_test)))




    plot_feature_importances_cancer(forest)
        #랜덤 포레스트도 일반 결정트리처럼 Feature importance를 제공한다
        #결정트리보다 훨씬 큰 그림을 그려준다



#4. Strengths & Weaknesses
    #가장 널리 쓰이고, 아주 강력한 방법이다. (별도로 파라미터의 수정이 필요 없다)
    #결정트리가 선택 과정을 집약적으로 보여주기에 적합하다. (nonexpert에게 보여주기 쉽다)
    #n_jobs=-1을 통해 모든 컴퓨터 코어를 사용 가능
    #랜덤포레스트는 말 그대로 랜덤이기 때문에, random_state을 바꿔주면 값도 바뀐다!
    #반복되는 결과를 원하면 그 값을 고정시켜줘야 한다.





else :


    '''
                    Gradient boosted regression Trees ( Gradient Boosting )
    '''


#Basic Idea
    #그래디언트 부스팅은 그 이름과 달리 '회귀'와 '분류'에 모두 쓰인다.
    #랜덤 포레스트의 접근법과는 반대로, 이전 트리의 실수를 바로잡으며 진행되는 순차적인 방법이다.
    #기본적으로는 확률적(random) 요소는 없고, 강력한 pre-pruning이 사용된다.
    #보통 아주 얕은 트리를 사용하는데 (깊이 1-5), 덕분에 모델이 작고 빠르다.
    #핵심 아이디어는 많은 작은 모델들(약한 학습기)을 결합하여 성능을 향상시킨다는 것이다.

    #이 방법은 머신러닝 대회에서 우승자들이 사용하는 방법이고, 파라미터들을 조금만 잘 맞추면 높은 성능을 낸다.
    #트리의 개수와 pre-pruning을 제외한 파라미터로는 learning_rate이 있다.
    #이는 각 트리가 이전의 트리의 실수로부터 얼마나 피드백을 받느냐의 정도이다.

    #디폴트 값으로 100개의 트리, 깊이 3, 학습률 0.1이 쓰인다!



#1. Building
    from sklearn.ensemble import GradientBoostingClassifier
    cancer = load_breast_cancer()

    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

    gbrt = GradientBoostingClassifier(random_state=0)
    gbrt.fit(X_train,y_train)

    print("\nGradient Boosting Score\n")
    print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train,y_train)))
    print("Accuracy on training set: {:.3f}".format(gbrt.score(X_test, y_test)))
        #테스트 점수가 100이므로 overfit 가능성이 크다. pre-pruning을 더 하거나, learning rate을 낮춰준다.

# Stronger pre-pruning - max_depth=1
    gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
    gbrt.fit(X_train,y_train)

    print("\nGradient Boosting Score (max_depth=1)\n")
    print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train,y_train)))
    print("Accuracy on training set: {:.3f}".format(gbrt.score(X_test, y_test)))

# Lowering learning_rate
    gbrt = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
    gbrt.fit(X_train,y_train)

    print("\nGradient Boosting Score (learning_rate=0.01)\n")
    print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train,y_train)))
    print("Accuracy on training set: {:.3f}".format(gbrt.score(X_test, y_test)))


#2. Feature Importances
    gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
    gbrt.fit(X_train, y_train)

    plot_feature_importances_cancer(gbrt)
        #보통 랜덤포레스트와 feature importance는 비슷하게 나온다. (하지만 그래디언트는 특정 feature 완전 무시)
        #일반적으로 랜덤 포레스트로 먼저 실험하고, 그 시간이 길거나 정확도를 조금더 높여야 할 때
        #그래디언트 부스팅을 쓴다.




#3. Strengths & Weaknesses
    #파라미터를 잘 조율해야하고, 훈련하는데 오래 걸릴 수 있다.
    #다른 트리기반 모델처럼, 스케일링이 필요 없고 feature의 특징을 타지 않는다.
    #다른 트리기반 모델처럼, 높은 차원의 sparse한 데이터에 적합하지 않다.
    #랜덤 포레스트와 달리, 트리의 수를 무작정 늘리는 것은 모델을 복잡하게 만들어 overfit 위험이 있음.
    #따라서 일반적으로는 시간과 메모리 여유에 따라 트리 수를 정하고, 학습률을 바꾸어 나가는게 적절
    #처음에 n의 숫자를 맞추고, learning rate을 조절해라.
    
