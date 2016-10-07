import numpy as np
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.utils import shuffle
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold


def get_hog(X):
    features = []
    for image in X:
        ft = hog(image.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1),
                 visualise=False)
        features.append(ft)

    return np.array(features)


def main():
    dataset = datasets.fetch_mldata('MNIST Original')
    X = dataset.data
    y = dataset.target

    X, y = shuffle(X, y, random_state=0)
    X = X / 255.0 * 2 - 1

    X = get_hog(X)

    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42)
    classifier = LinearSVC()
    classifier.fit(X_train, y_train)

    print("predicting")
    print "score: ", classifier.score(X_test, y_test)

    joblib.dump(classifier, "../classifier/linear_svm_hog.pkl")

if __name__ == '__main__':
    main()
