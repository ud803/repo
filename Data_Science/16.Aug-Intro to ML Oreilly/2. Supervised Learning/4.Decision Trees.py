'''
                             Decision Trees (결정 트리)
'''

'''
결정 트리는 분류와 회귀에 자주 사용되는 모형이다. 결정 트리는 if/else문의 계층을 통해 '결정'에 다다르게 된다.

if/else 질문은 머신러닝에서 "test"라고 부른다.

트리에서 각 노드는 하나의 질문을 의미하거나, 답을 포함하는 터미널 노드(잎 노드)를 의미한다.

결정 트리 모형은 직접 만들 필요 없이, supervised learning을 이용해 학습할 수 있다.
'''

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import mglearn


where = input("1 for Classification, 2 for Regression : ")
if(where=='1'):



    '''
                            Classification
    '''



    #1. Building decision Trees
        # 여기서는 2D 분류를 예로 들 것이다.
        # 데이터셋은 2개의 반달 모양으로 구성되어 있고, 각각 75개의 데이터를 갖고 있다.
        # 보통, 데이터는 binary형식으로 있지 않다. 오히려 연속된 데이터로 주어지기 마련이다.
        # 연속된 데이터에 쓰이는 테스트는 "특징 i가 값 a보다 큽니까?"의 형태다.

    # 결정 트리가 생성되는 과정은 다음과 같다.
        # 알고리즘은 모든 가능한 테스트를 찾고, 그 중 가장 유용한 것을 뽑는다.
        # 그렇게 선정된 테스트는 첫 번째 테스트가 되고, x[1] <= 0.0596과 같은 모습이다.
        # 이 테스트에 의해 True이면 왼쪽, False이면 오른쪽 노드로 내려간다.
        #   여기서는 True일 경우 [2, 32], False일 경우 [48,18]의 점수가 매겨진다.
        #   클래스 1이 나뉘어진 영역에서 아래에 2개, 위에 48개
        #   클래스 2는 나뉘어진 영역에서 아래에 32개, 아래에 18개 있다는 의미
        # 각 노드로 내려와서 또 다른 테스트를 시행한다.
        # 그렇게 또 점수를 매겨나가면서 재귀적으로 반복한다.
        # 모든 잎이 오직 하나의 클래스만을 포함하게 되면 그 과정이 끝난다. (pure)
    # 이렇게 결정 트리를 생성해 놓고, 예측값을 트리에 넣어 어디로 분류되는지를 보는 것이다.
    # Regression에도 정확히 같은 방법으로 적용할 수 있다.




    #2. Controlling Complexity of decision Trees
        # 보통 위와 같은 방법으로 트리를 만들게 되면 아주 복잡하고 overfit한 모델이 된다.
        # 이를 방지하기 위한 2가지 방법이 있다.
        #   1. 트리의 생성을 막는 것 (pre-pruning)
        #       - tree의 depth 제한, maximum leaves 제한, min points 요구
        #   2. 트리를 생성하고 나서, 불필요한 노드를 지우는 것 (post-pruning, pruning)
        # scikit-learn에서는 pre-pruning만 제공한다.


    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
    tree = DecisionTreeClassifier(random_state=0)
    tree.fit(X_train,y_train)

    print("\n\t<<Tree>>\n")
    print("Accuracy on training sets: {:.3f}".format(tree.score(X_train,y_train)))
    print("Accuracy on test set: {:.3f}".format(tree.score(X_test,y_test)))
        # 예상대로, 트레이닝 셋의 정확도는 100%이다 - 잎이 pure하기 때문에.
        # test 점수는 선형 모형보다 약간 더 낮다.
        # 결정 트리의 깊이를 제한하지 않는다면, 트리는 무제한적으로 복잡해진다.
        # 그렇게 되면 overfitting하고, generalize하기 어렵다.

    # 따라서 여기서는 pre-pruning 중, 트리의 깊이를 제한하는 방식을 사용한다.
    tree = DecisionTreeClassifier(max_depth=4, random_state=0)
    tree.fit(X_train,y_train)

    print("\n\t<<pre-pruning Tree>>\n")
    print("Accuracy on training sets: {:.3f}".format(tree.score(X_train,y_train)))
    print("Accuracy on test set: {:.3f}".format(tree.score(X_test,y_test)))
        # 0.988 0.951로 성능이 더 좋아졌다.




    #3. Analyzing decision Trees

    # 다음의 방법을 통해 결정 트리를 시각화 할 수 있다.
    # 근데 여기서는 안됨..
    from sklearn.tree import export_graphviz
    export_graphviz(tree, out_file="tree.dot", class_names=["malignant", "benign"], feature_names=cancer.feature_names, impurity=False, filled=True)

    import graphviz

    with open("tree.dot") as f:
        dot_graph = f.read()
    graphviz.Source(dot_graph)
        # 트리의 시각화는 심층적인 이해롤 도와준다.
        # 책의 예를 보면, 오른쪽의 트리는 142(8,134)에서 더 나뉘어 내려간다.
        # 결국, 처음의 142개 샘플 중 132개가 오른쪽으로 가는데, 이는 redundant할 수 있음.




    #4. Feature Importance in Trees

    # 트리의 요약 정보를 쉽게 볼 수 있는 방법이 있는데, "feature importance"라고 부른다
    # 1로 갈수록 예측을 잘 한 것이고, 모든 값의 총 합은 1이다.

    print("Feature importances:\n{}".format(tree.feature_importances_))
        # 이 데이터를 가지고 우리가 w를 도식화한 것처럼 그림으로 나타낼 수 있다.


    def plot_feature_importances_cancer(model):
        n_features = cancer.data.shape[1]
        plt.barh(range(n_features), model.feature_importances_, align='center')
        plt.yticks(np.arange(n_features), cancer.feature_names)
        plt.xlabel("Feature importance")
        plt.ylabel("Feature")
        plt.show()

    plot_feature_importances_cancer(tree)
        # 특정 feature가 매우 중요하다는 것을 알 수 있다.
        # 하지만 그 값이 낮다고 해서 정보력이 낮다는 의미는 아니다.
        # 그냥 그 feature가 트리에게 선택받지 않은 것이다. (다른 feature와 공유해서)
        # 그리고 수치가 클수록 "중요하다"는 것을 의미하지만, 샘플을 분류해주지는 않는다.
        # 다음의 예를 살펴보자.


    tree = mglearn.plots.plot_tree_not_monotone()
    #display(tree)
    #### 이거안됨 #########
        # 여기서 하고자 하는 말은,
        # 한 feature의 중요도가 높다고 해서 나오는 결과값을 알 수는 없다는 말!


else:

    '''
                            Regression
    '''

    '''
    Regression도 Classification과 매우 유사하다. 하지만 트리 모델의 중요한 특징이 있다.
    트리를 기반으로한 모든 모형들은 "추론"할 수 없다. 즉, 트레이닝셋의 범위를 벗어난 데이터에 대해 예측할 수 없다.
    다음 예제를 통해 살펴보자.
    '''


    # historical computer memory prices
    ram_prices = pd.read_csv("../Material/introduction_to_ml_with_python-master/data/ram_price.csv")

    plt.semilogy(ram_prices.date, ram_prices.price)
    plt.xlabel("Year")
    plt.ylabel("Price in $/Mbyte")
    plt.show()
        # x축엔 날짜가, y축엔 그 해 램의 가격이 나와있다.
        # y축은 로그단위로 증가하는 것에 주목
        # 로그 함수로 그림을 그리면, 관계가 linear하게 보이기 때문에 예측이 쉬워 보인다.
    # 이 데이터를 가지고 예측을 해 볼 것이다. Linear Regressor와 DecisionTreeRegressor
    # 관계를 선형으로 나타내기 위해 가격에 로그를 취해줄 것이다.
    # DT에는 아무 영향이 없지만, LR에는 큰 영향을 미친다. (Ch4에서 다룰 것)
    # 마지막에는 다시 exp를 취해주어 원래대로 돌릴 것이다.

    from sklearn.tree import DecisionTreeRegressor
    from sklearn.linear_model import LinearRegression


    # use historical data to forecast prices after the year 2000
    data_train = ram_prices[ram_prices.date<2000]
    data_test = ram_prices[ram_prices.date>=2000]

    # predict prices based on date
    X_train = data_train.date[:, np.newaxis]

    y_train = np.log(data_train.price)

    tree = DecisionTreeRegressor().fit(X_train,y_train)
    linear_reg = LinearRegression().fit(X_train, y_train)

    X_all = ram_prices.date[:,np.newaxis]

    pred_tree = tree.predict(X_all)
    pred_lr = linear_reg.predict(X_all)

    price_tree = np.exp(pred_tree)
    price_lr = np.exp(pred_lr)

    plt.semilogy(data_train.date, data_train.price, label="Training data")
    plt.semilogy(data_test.date, data_test.price, label="Test data")
    plt.semilogy(ram_prices.date, price_tree, label="Tree Prediction")
    plt.semilogy(ram_prices.date, price_lr, label="Linear Prediction")
    plt.legend()
    plt.show()
        # 선형 모델은 꽤 잘 예측을 하지만, 트리 모델은 training 이후로 예측을 하지 못한다.
        # 이것이 가장 큰 문제점이다!


'''DT 마무리

1)
DT에서 모델의 복잡도를 결정하는 파라미터는 'pre-pruning'이다.
앞에서 말한 pre-pruning 기법 중 한 가지만을 써도 overfitting을 방지할 수 있다.

2) 장점
시각화가 쉽고 비전문가가 이해하기 좋다.
알고리즘이 데이터의 크기에 영향받지 않는다.
개별 feature를 보기 때문에, 정규화나 표준화 같은 데이터 가공이 필요없다.

3) 단점
pre-pruning을 써도 overfit하는 경향이 있다. 따라서 generalization이 안좋다.
그래서 다음에 배울 ensemble DT가 많이 쓰인다.
'''
