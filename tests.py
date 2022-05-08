import os
import sys
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, make_classification, make_moons, make_swiss_roll, make_s_curve, make_circles, \
    make_blobs, load_breast_cancer, load_wine, load_digits
from mvlearn.datasets import load_UCImultifeature
from topomap import TopoMap
import time


def get_data(dataset="classification"):
    if dataset == "iris":
        iris = load_iris()
        data = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])
        data.drop_duplicates(inplace=True)
        X = data.drop('target', axis=1).to_numpy()
        y = data['target'].to_numpy()
    elif dataset == "cancer":
        cancer = load_breast_cancer()
        X, y = cancer['data'], cancer['target']
    elif dataset == "wine":
        wine = load_wine()
        X, y = wine['data'], wine['target']
    elif dataset == "digits":
        digits = load_digits()
        X, y = digits['data'], digits['target']
    elif dataset == "mfeat":
        X, y = load_UCImultifeature()
        X = X[2]
        X, indices = np.unique(X, axis=0, return_index=True)
        y = y[indices]
    elif dataset == "moons":
        X, y = make_moons(n_samples=1000, random_state=1234)
    elif dataset == "swissroll":
        X, y = make_swiss_roll(n_samples=1000, random_state=1234)
    elif dataset == "s_curve":
        X, y = make_s_curve(n_samples=1000, random_state=1233)
    elif dataset == "circles":
        X, y = make_circles(n_samples=1000, random_state=1234)
    elif dataset == "blobs":
        X, y = make_blobs(n_samples=1000, random_state=1206)
    elif dataset == "small":
        X, y = make_classification(n_samples=6, n_features=3, n_informative=3, n_redundant=0, n_classes=2, random_state=1234)
    else:
        X, y = make_classification(n_samples=1000, n_features=10, class_sep=5, n_informative=5, n_classes=3,
                                   n_clusters_per_class=1, flip_y=0.00, random_state=1234)
    return X, y


def test(dataset="classification", method="default"):
    X, y = get_data(dataset)
    start = time.time()
    t = TopoMap(method)
    t.fit(X, y)
    end = time.time()
    print("time elapsed:", end-start)
    print("method used:", t.get_params()['method'])
    t.plot_data()
    t.plot()


if __name__ == '__main__':
    print(os.getcwd())
    arguments = sys.argv[1:]
    if len(arguments) == 0:
        test()
    elif len(arguments) == 1:
        test(dataset=arguments[0])
    else:
        test(dataset=arguments[0], method=arguments[1])
