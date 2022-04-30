import sys

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, make_classification, make_moons, make_swiss_roll, make_s_curve, make_circles, \
    make_blobs
from topomap import TopoMap
import time


def get_data(dataset="classification"):
    if dataset == "iris":
        iris = load_iris()
        data = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])
        data.drop_duplicates(inplace=True)
        X = data.drop('target', axis=1).to_numpy()
        y = data['target'].to_numpy()
    elif dataset == "moons":
        X, y = make_moons(n_samples=1000)
    elif dataset == "swissroll":
        X, y = make_swiss_roll(n_samples=1000)
    elif dataset == "s_curve":
        X, y = make_s_curve(n_samples=1000)
    elif dataset == "circles":
        X, y = make_circles(n_samples=1000)
    elif dataset == "blobs":
        X, y = make_blobs(n_samples=1000)
    else:
        X, y = make_classification(n_samples=1000, n_features=10, class_sep=5, n_informative=5, n_classes=3,
                                   n_clusters_per_class=1, flip_y=0.00)
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


if __name__ == '__main__':
    arguments = sys.argv[1:]
    if len(arguments) == 0:
        test()
    elif len(arguments) == 1:
        test(dataset=arguments[0])
    else:
        test(dataset=arguments[0], method=arguments[1])
