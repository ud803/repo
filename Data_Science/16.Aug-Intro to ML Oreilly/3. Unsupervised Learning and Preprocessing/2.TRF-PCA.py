'''
                    Visualization & Reduction using PCA
'''



'''
비지도 학습을 이용하여 데이터를 변형하는 것은 많은 이유가 있다.
가장 중요한 이유는 1) 시각화, 2) 데이터 압축, 3) 더 나은 데이터 표현이다.

이 모두를 위한 가장 간단하고 널리 사용되는 알고리즘은 Principal Component Analysis (PCA)이다.
그리고 feature extraction에 사용되는 non-negative matrix factorization(NMF)
시각화에 사용되는 t-SNE가 있다.
'''


import mglearn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

where=input("1 for Visualization, 2 for Extraction : ")
if(where=='1'):

    '''
                #1. Principal Component Analysis(PCA)
    '''

    #1. Basic Idea
    #PCA는 회전된 feature들이 통계적으로 연관되어있지 않도록 데이터를 회전시키는 방법
    #이 회전 후에, 새롭게 만들어진 feature들 중 그 중요도에 따라 일부 집합만을 선택한다
    #다음 예를 살펴보자.


    mglearn.plots.plot_pca_illustration()
    plt.show()
    #그림을 보면 다음과 같다.

    #1) 그림1
    #먼저 오리지널 데이터 포인트에서 최대 분산의 방향을 찾는다. (컴포넌트 1)
    #이 컴포넌트1 벡터가 가장 중요한 데이터를 많이 포함한 방향이고, 바꾸어 말하면 feature들이 가장 서로와 연관되어있는 방향이다.

    #2)
    #두 번째로 이 벡터와 수직이면서 가장 많은 정보를 포함한 벡터를 찾는다.
    #2차원에서는 하나의 수직벡터가 존재하지만, 고차원에서는 feature의 개수만큼 존재한다.

    #3)
    #위 1) 2) 절차를 거쳐 찾아낸 벡터를 "Principal Components"라고 부른다.

    #4) 그림2
    #데이터에서 평균을 빼서, 변형된 데이터가 0 주위에 있도록 만든다.
    #그러고 나서, 컴포넌트1 벡터가 x축과 수직이도록 회전한다.
    #여기서, 두 축은 상관관계가 없다. 즉, 데이터의 상관계수 행렬이 대각을 제외하고 0이다.

    #5) 그림3
    #여기서 몇 개의 principal Components만 남겨서 dimensionality reduction을 할 수 있다.
    #이 예제에서는 그림3처럼 컴포넌트1만 남겨두겠다.
    #이제 2차원 데이터셋이 1차원 데이터셋으로 줄여졌다.
    #여기서 주의해야할 점은, 우리가 기존 feature 중 하나를 남긴 것이 아니라, 가장 흥미로운(정보를 많이 주는) 방향을 찾아서 이 방향을 남겼다는 것이다.

    #6) 그림4
    #마지막으로, 회전을 다시 반대로 주고, 평균을 다시 데이터에 더한다.
    #이는 그림4와 같이 된다.
    #이 점들은 기존 feature 공간에 있지만, 컴포넌트1에 포함된 정보만을 가지고 있다.
    #이 변형은 데이터로부터 노이즈를 제거하거나 Principal Components를 이용했을 때 어떤 정보가 남는지 시각화하기 위해 사용된다.



    #2. Applying PCA to the cancer dataset for visualization
    #PCA의 주된 목적 중 하나는 고차원의 데이터를 시각화하는 것이다.
    #챕터1의 Iris 데이터처럼, pair plot을 그릴 수도 있지만, 만약 30개의 feature를 가진 데이터를 그리려면 30*14=420개의 도식이 필요하다!
    #훨씬 간단한 시각화 방법이 있는데,
    #두 클래스 별로 각 feature의 히스토그램을 계산하는 것이다.

    cancer = load_breast_cancer()
    fig, axes = plt.subplots(15, 2, figsize=(10,20))
    malignant = cancer.data[cancer.target ==0]
    benign = cancer.data[cancer.target == 1 ]

    ax = axes.ravel()

    for i in range(30) :
        _, bins = np.histogram(cancer.data[:,i], bins=50)
        ax[i].hist(malignant[:,i], bins=bins, color=mglearn.cm3(0), alpha=.5)
        ax[i].hist(benign[:,i], bins=bins, color=mglearn.cm3(2),alpha=.5)
        ax[i].set_title(cancer.feature_names[i])
        ax[i].set_yticks(())
        ax[0].legend(["malignant","benign"], loc="best")
    fig.tight_layout()
    plt.show()
    #각 feature별로 각 클래스의 해당 feature 값이 어떻게 분포되어 있는지 알 수 있다.
    #smoothness error는 둘 모두 겹쳐있어서 uninformative하다.
    #worst concave points는 둘 사이의 차이를 잘 말해준다.
    #하지만 이 히스토그램은 변수들간의 상호작용을 보여주지 않고, 이 값들이 클래스의 결정에 어떤 영향을 미치는지 보여주지 않는다.
    #따라서 우리는 PCA를 사용할 것이다.


    #먼저 데이터를 스케일 해준다.
    scaler = StandardScaler()
    scaler.fit(cancer.data)
    X_scaled = scaler.transform(cancer.data)

    #PCA 적용
    #keep the first two principal components of the data
    pca = PCA(n_components=2)
    # fit PCA model to breast cancer data
    pca.fit(X_scaled)

    #transform data onto the first two principal Components
    X_pca = pca.transform(X_scaled)
    print("Original shape: {}".format(str(X_scaled.shape)))
    print("Reduced shape: {}".format(str(X_pca.shape)))
    #이제 주요 컴포넌트 2개를 그려준다.
    plt.figure(figsize=(8,8))
    mglearn.discrete_scatter(X_pca[:,0], X_pca[:,1], cancer.target)
    plt.legend(cancer.target_names, loc="best")
    plt.gca().set_aspect("equal")
    plt.xlabel("First principal component")
    plt.ylabel("Second principal component")
    plt.show()
    #그림으로 봤을 때, 간단한 선형 분류로도 둘을 구분할 수 있을 것 같다.
    #하지만 PCA의 단점은, 그림의 두 축이 해석하기 쉽지 않다는 점이다.
    #컴포넌트들은 오리지널 데이터의 방향을 나타내고, 이는 기존 feature의 결합이다.
    #이 결합들은 복잡하기에 해석이 어렵다.
    #principal components들은 components_ 에 들어있다.

    print("PCA component shape: {}".format(pca.components_.shape))
    #행은 각 컴포넌트를 의미하고, 중요도가 높은 것이 먼저 나와있다.
    #열은 기존 feature에 대응된다.
    print("PCA components:\n{}".format(pca.components_))

    plt.matshow(pca.components_,cmap='viridis')
    plt.yticks([0,1], ["First component", "Second component"])
    plt.colorbar()
    plt.xticks(range(len(cancer.feature_names)), cancer.feature_names, rotation=60, ha='left')
    plt.xlabel("Feature")
    plt.ylabel("Principal components")
    plt.show()







elif(where=='2') :
    from sklearn.datasets import fetch_lfw_people
    people = fetch_lfw_people(min_faces_per_person=20, resize=.7)
    image_shape = people.images[0].shape

    fix, axes = plt.subplots(2, 5, figsize=(15,8), subplot_kw={'xticks': (), 'yticks' : ()})
    for target, image, ax in zip(people.target, people.images, axes.ravel()):
        ax.imshow(image)
        ax.set_title(people.target_names[target])
    plt.show()

    print("people.images.shape: {}".format(people.images.shape))
    print("Number of classes: {}".format(len(people.target_names)))
        #3023장의 이미지, 87*65 픽셀의 크기이며, 62명의 사진을 갖고 있다.

#데이터셋은 약간 왜곡되어 있는데, 특정 인물의 사진이 많기 때문이다.
#각 인물당 몇 장의 사진이 있는지 확인해보자.

    #count how often each target appears
    counts = np.bincount(people.target)
    #print counts next to target names
    for i, (count, name) in enumerate(zip(counts, people.target_names)):
        print("{0:25} {1:3}".format(name,count), end='  ')
        if(i+1)%3 ==0:
            print()
    print()
#각 인물당 최대 50장까지만 허용하자.
#그리고 그레이스케일을 0과 1 사이의 값으로 조정한다.

    mask = np.zeros(people.target.shape, dtype=np.bool)
    for target in np.unique(people.target):
        mask[np.where(people.target == target)[0][:50]] =1

    X_people = people.data[mask]
    y_people = people.target[mask]

    #scale the grayscale values to be btw 0 and 1
    #instead of 0 and 255 for better numeric stability
    X_people = X_people / 255.


#얼굴 인식의 원리는, 이 새로운 얼굴이 데이터베이스에 있는 사람과 일치하는지를 매번 물어보는 것이다.
#각 사람마다 클래스를 나누어주면 되는데, 하지만 보통 다양한 사람들의 얼굴 데이터는 있지만,
#그 각각의 사진이 많지는 않다. (= 트레이닝 데이터가 많지 않다.)
#그래서 대부분의 분류기에는 적합하지 않다.
#가장 적합한 것이 kNN에서 k를 1로 두는 것이다.

    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    print("Test set score of 1-nn: {:.2f}".format(knn.score(X_test,y_test)))

#정확도가 23%밖에 되지 않는다.
#여기에 PCA가 사용되는데, 기존 픽셀의 거리를 계산하는 것은 좋지 않은 방법이다.
#우리는 그레이스케일 값을 계산하는데, 사람이 인식하는 방식과는 사뭇 다르다.
#게다가 이 방식에서, 얼굴을 한 픽셀실 오른쪽으로만 움직여도 아주 큰 변화를 가져온다.
#따라서 우리는 PCA의 "whitening" 옵션을 사용한다.
#이 옵션은 주요 컴포넌트를 리스케일하여 같은 크기를 갖도록 만든다.
#이는 변형 후에 StandardScaler를 사용하는 것과 원리가 같다.

    mglearn.plots.plot_pca_whitening()
    plt.show()

    pca=PCA(n_components=100, whiten=True,random_state=0).fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    print("X_train_pca.shape: {}".format(X_train_pca.shape))
        #새로운 데이터는 100개의 주요 컴포넌트를 가지고 있다.
        #이제 이 데이터로 다시 kNN을 써보자.

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train_pca, y_train)
    print("Test set accuracy: {:.2f}".format(knn.score(X_test_pca, y_test)))
        #정확도가 상당히 올라간 것을 볼 수 있다.


    print("pca.components_.shape: {}".format(pca.components_.shape))
    fix, axes = plt.subplots(3,5,figsize=(15,12), subplot_kw={'xticks':(), 'yticks': ()})
    for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel())):
        ax.imshow(component.reshape(image_shape), cmap='viridis')
        ax.set_title("{}.component".format((i+1)))
    plt.show()
        #각 컴포넌트가 얼굴의 어떤 점을 특징하고 있는지 알 수 있다.
        #하지만 이는 사람들의 얼굴 인식 방식과는 다르다.
        #알고리즘은 사람과는 다른 방식으로 데이터를 이해할 수 있다는 사실을 명심하자.


#PCA의 다른 해석들은 다음과 같다.
#1) 각 사진을 principal components들의 가중합으로 나타낼 수도 있다. (선형결합으로)
#2) 컴포넌트 부분 집합만을 이용하여 기존 데이터를 복원할 수 있다.
#   이는 아래 그림에서 자세히 살펴보자.

    mglearn.plots.plot_pca_faces(X_train,X_test,image_shape)
    plt.show()
        #챕터 앞에서 한 것처럼, 우리는 몇가지 컴포넌트를 드랍하여 컴포넌트를 줄이고, 다시 회전시켜서 원래 데이터로 돌아갈 수 있다.
        #이는 "inverse_transform" 메서드를 이용해서 할 수 있다.
        #그림에서, 더 많은 컴포넌트를 사용할 수록 이미지의 디테일이 살아난다.
        #모든 컴포넌트를 사용한다는 것은, 우리가 회전 이후에 어떠한 정보도 버리지 않겠다는 의미이고, 이미지를 완벽하게 복원할 수 있을 것이다.

#또한 모든 얼굴을 두개의 컴포넌트를 이용하여 2차원 평면에 나타낼 수 있다.
    mglearn.discrete_scatter(X_train_pca[:,0], X_train_pca[:,1], y_train)
    plt.xlabel("First principal component")
    plt.ylabel("Second principal component")
    plt.show()
        #이는 당연하게도 말도 안되는 그림을 보여준다.
        #왜냐하면 우리는 오직 '2'개의 컴포넌트만을 사용했기 때문이다.
        #이전 그림에서, 10개의 컴포넌트 조차 얼굴의 윤곽만을 잡는 것을 보았다.
        #2개는 턱도 없다.
