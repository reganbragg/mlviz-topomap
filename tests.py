import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, make_classification
from topomap import TopoMap
import time


def get_data(dataset="default"):
    if dataset == "default":
        X, y = make_classification(n_samples=1000, n_features=10, class_sep=5, n_informative=5, n_classes=3,
                                   n_clusters_per_class=1, flip_y=0.00)
        return X, y
    elif dataset == "iris":
        iris = load_iris()
        data = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])
        data.drop_duplicates(inplace=True)
        X = data.drop('target', axis=1).to_numpy()
        y = data['target'].to_numpy()
    return X, y


def test(dataset="default", method="default"):
    X, y = get_data(dataset)
    start = time.time()
    t = TopoMap(method)
    t.fit(X, y)
    end = time.time()
    print(end-start)
    print(t.get_params())
    t.plot()


test(dataset="iris")
