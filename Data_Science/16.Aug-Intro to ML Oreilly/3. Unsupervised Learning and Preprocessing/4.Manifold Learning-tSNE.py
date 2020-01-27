'''
                        Manifold Learning with t-SNE
'''

'''
PCA가 데이터를 변형하여 시각화하기에 좋고, 처음으로 시도할만한 알고리즘이지만, 본질적으로 회전을 하여 방향을 없애는 방법은 그 유용성을 제한한다. (앞의 얼굴 예제에서 봤듯이)

시각화를 위한 알고리즘들이 있는데, "manifold learning" 알고리즘이라고 불린다. 이들은 훨씬 복잡한 매핑과 더 나은 시각화를 제공한다. 특히 그 중 t-SNE를 살펴보겠다.
'''
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#1. Basic idea
    #매니폴드 러닝 알고리즘은 시각화에 초점이 맞춰져있다.
    #두 개 이상의 새로운 feature를 만드는 데 거의 이용되지 않는다.
    #트레이닝셋을 새롭게 표현하기는 하지만, 새로운 데이터의 변형은 허용하지 않는다.
    #즉, 이 알고리즘은 테스트셋에 적용될 수 없다는 말이다. 따라서 학습의 최종 목적이 지도 학습이라면, 이 알고리즘은 거의 쓰이지 않는다.

    #점들 사이의 간격을 유지하는 2차원 표현식을 찾아내는 것이 그 목적이다.
    #   먼저 각 데이터의 임의의 2차원 표현을 찾는다.
    #   원래 feature에서 가까운 점은 더 가까이, 먼 것은 더 멀게 만든다.
    #   서로 이웃인 점들에 대한 정보를 유지하려는 것.


#hand written digits 예제를 살펴보자

digits = load_digits()

fig, axes = plt.subplots(2,5,figsize=(10,5), subplot_kw={'xticks':(), 'yticks':()})
for ax, img in zip(axes.ravel(), digits.images):
 ax.imshow(img)
plt.show()
    #PCA를 사용해 에티러르 2차원으로 낮춰보자.

pca=PCA(n_components=2)
pca.fit(digits.data)
digits_pca = pca.transform(digits.data)
colors = ["#476A2A", '#7851B8', '#BD3430', '#4A2D4E','#875525', '#A83683', '#4E655E', '#853541', '#3A3120', '#535D8E']

plt.figure(figsize=(10,10))
plt.xlim(digits_pca[:,0].min(), digits_pca[:,0].max())
plt.ylim(digits_pca[:,1].min(), digits_pca[:,1].max())
for i in range(len(digits.data)):
    # actually plot the digits as text instead of using scatter
    plt.text(digits_pca[i,0], digits_pca[i,1], str(digits.target[i]), color= colors[digits.target[i]], fontdict={'weight' : 'bold', 'size':9})
plt.xlabel("First principal Components")
plt.ylabel("Second principal Components")
plt.show()
    #0, 4, 6은 비교적 잘 구분되어 있지만, 다른 것들은 상당히 많이 섞여있다.
    #이번에는 t-SNE를 사용해 결과를 비교해보자.
    #t-SNE는 새로운 데이터의 변형을 허용하지 않으므로, transform 메서드가 없다.
    #대신, fit_transform 메서드를 사용한다.

from sklearn.manifold import TSNE
tsne = TSNE(random_state=42)
# use fit_transform instead of fit
digits_tsne = tsne.fit_transform(digits.data)

plt.figure(figsize=(10,10))
plt.xlim(digits_tsne[:,0].min(), digits_tsne[:,0].max())
plt.ylim(digits_tsne[:,1].min(), digits_tsne[:,1].max())
for i in range(len(digits.data)):
    # actually plot the digits as text instead of using scatter
    plt.text(digits_tsne[i,0], digits_tsne[i,1], str(digits.target[i]), color= colors[digits.target[i]], fontdict={'weight' : 'bold', 'size':9})
plt.xlabel("First principal Components")
plt.ylabel("Second principal Components")
plt.show()
    #그 결과는 놀랍다.
    #t-SNE를 이용하자, 모든 클래스가 잘 분류되었다.
    #중요한 사실은, 이 메서드는 클래스 라벨에 대한 정보가 없다는 것이다.
    #우리는 학습 방법을 지정해줬을 뿐이고, 완전히 'unsupervised'된 상태이다.
    #원래 공간에서 각 점들이 얼마나 가까이 있는가에 대한 정보만으로 이렇게 분류 한 것!
