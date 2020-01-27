import matplotlib.pyplot as plt

from sklearn import datasets, svm, metrics

digits = datasets.load_digits()

print(digits.keys())

'''
print(digits.data.shape)

print("data",digits.data.shape,"\n",digits.data)
print("names", digits.target_names.shape,"\n",digits.target_names)
print("target", digits.target.shape, "\n",digits.target)
print("images", digits.images.shape, "\n",digits.images)
'''

images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2,4,index+1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' %label)

n_samples = len(digits.images)
data = digits.images.reshape((n_samples,-1))

print(digits.images)
print("n_samples : {}".format(n_samples))
print("\ndata :{}".format(data))
classifier = svm.SVC(gamma=0.001)

classifier.fit(data[:n_samples // 2], digits.target[:n_samples //2])

expected = digits.target[n_samples//2:]
predicted = classifier.predict(data[n_samples//2:])

print("Classification report for classifier %s:\n%s\n" % (classifier, metrics.classification_report(expected, predicted)))

print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected,predicted))

images_and_predictions = list(zip(digits.images[n_samples//2:], predicted))


for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2,4,index+5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.show()


print(digits.target_names)


#
# http://yann.lecun.com/exdb/mnist/
# 위 사이트 참고!!!
