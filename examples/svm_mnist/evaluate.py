# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
import numpy as np

def evaluate(C, gamma):
    # The digits dataset
    digits = datasets.load_digits()

    # To apply a classifier on this data, we need to flatten the image, to
    # turn the data in a (samples, feature) matrix:
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    # Create a classifier: a support vector classifier
    classifier = svm.SVC(C=C, gamma=gamma)

    # We learn the digits on the first half of the digits
    classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

    # Now predict the value of the digit on the second half:
    expected = digits.target[n_samples // 2:]
    predicted = classifier.predict(data[n_samples // 2:])

    return np.mean(expected == predicted)
