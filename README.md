## Topomap Implementation in Python
This API is based off the TopoMap method published in [TopoMap: A 0-dimensional Homology Preserving Projection of
High-Dimensional Data](https://arxiv.org/pdf/2009.01512.pdf). This implementation extends upon [existing C++ documentation](https://github.com/harishd10/TopoMap).

#### The following implementation supports several convex hull edge selection techniques:
- *Original:* replicating the methods presented in the published paper. Selects the point in the convex hull vertex set that minimizes the Euclidean distance to the EMST edge vertex, and the consequent point for the rotating edge.
- *First:* selects the first two points in counterclockwise order to form the rotating edge.
- *Random:* selects a random edge from the convex hull as the rotating edge.
 
### Methods
```fit(X, y=None)```

Fits high-dimensional data X into an embedded Euclidean space.

| Parameters  | X: *ndarray of shape (n_samples, n_features)* <br> Y: *ndarray of shape (n_samples, 1)* representing class labels |
| :---        | :---        |
| **Returns** | ***None***  |

```fit_transform(X, y=None)```

Fits high-dimensional data X into an embedded Euclidean space, and returns the transformed points.

| Parameters  | X: *ndarray of shape (n_samples, n_features)* <br> Y: *ndarray of shape (n_samples, 1)* representing class labels |
| :---        | :---        |
| **Returns** | **X_new: *ndarray of shape (n_samples, 2)*** |

```get_params()```

Get parameters for this estimator.

| Parameters  | None |
| :---        | :---        |
| **Returns** | **params: *dict*** <br> Parameter names mapped to their values|

```set_params(**params)```

Set parameters for this estimator.

| Parameters  | method: *method*. Options are 'default', 'first', and 'random' |
| :---        | :---        |
| **Returns** | ***None*** |


```plot()```

Produces a 2D plot of the transformed points.

| Parameters  | None |
| :---        | :---        |
| **Returns** | ***Plot: matplotlib scatterplot of transformed points*** |


```plot_persistence(tranformed=True)```

Produces a persistence plot for the homology of the original or transformed points.

| Parameters  | transformed: *boolean*. Indicate *False* for a persistence plot of the original points, otherwise form a persistence plot for the transformed points.|
| :---        | :---        |
| **Returns** | ***Plot: matplotlib persistence diagram*** |

