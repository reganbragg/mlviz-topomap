import random
from mlpack import emst
import numpy as np
import math
from scipy.spatial import ConvexHull
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ripser
import persim


def get_closest_edge(hull_points, point):
    length = hull_points.shape[0]
    min_dist = float("inf")
    for i in range(hull_points.shape[0]):
        p1 = hull_points[i]
        p2 = hull_points[(i+1) % length]
        p = p2-p1
        norm = np.sum(np.square(p))
        u = np.dot(np.subtract(point, p1), p)/norm
        if u > 1:
            u = 1
        elif u < 0:
            u = 0
        xy = np.add(p1, u*p)
        dxy = np.subtract(xy, point)
        dist = np.sqrt(np.sum(np.square(dxy)))
        if dist < min_dist:
            index = i
            min_dist = dist
    return index


# Get the angle between two points
def get_angle(p1, p2):
    delta_y = p2[1] - p1[1]
    delta_x = p2[0] - p1[0]
    theta = -1 * math.atan2(delta_y, delta_x)
    return theta


# Rotate the hull so that the edge corresponding to the first two endpoints is either aligned at the top or bottom
def rotate_hull(points, vertices, top, method, edge_point):
    idx = 0
    length = vertices.shape[0]

    if method == 'random' and length > 2:
        idx = random.randint(1, vertices.shape[0] - 1)

    if method == "default" and length > 2:
        vertex_points = points[vertices]
        # dists = np.linalg.norm(vertex_points - edge_point, axis=1)
        # idx = np.where(dists == np.amin(dists))[0][0]
        idx = get_closest_edge(vertex_points, edge_point)

    # Get the angle to rotate by
    if top:
        theta = get_angle(points[vertices[idx]], points[vertices[(idx + 1) % length]])
    else:
        theta = get_angle(points[vertices[idx]], points[vertices[(idx - 1) % length]]) + math.pi

    center_index = vertices[idx]
    center = points[center_index]
    points_rotated = np.empty((0, 2))

    # For all the points, rotate them around the selected point
    for i in range(points.shape[0]):
        # Translate the point to the origin and rotate
        if i != vertices[idx]:
            y = points[i][1] - center[1]
            x = points[i][0] - center[0]
            y_prime = x * math.sin(theta) + y * math.cos(theta) + center[1]
            x_prime = x * math.cos(theta) - y * math.sin(theta) + center[0]
            points_rotated = np.vstack((points_rotated, np.array([x_prime, y_prime])))
        else:
            points_rotated = np.vstack((points_rotated, center))

    # Translate the points back
    points_rotated = np.subtract(points_rotated, center)
    return points_rotated


class TopoMap:

    def __init__(self, method='default'):
        self.points = None
        self.target = None
        self.r2_points = None
        self.method = 'default'
        methods = ['random', 'first']
        if method in methods:
            self.method = method

    def get_params(self):
        return {'method': self.method}

    def set_params(self, method='default'):
        methods = ['default', 'random', 'first']
        if method in methods:
            self.method = method

    # Emst: matrix where each row represents an edge in the Emst
    # First dimension is lesser index, second dimension is higher index
    # Third dimension is the edge length
    def compute_EMST(self):
        Emst = emst(input=self.points, naive=False, leaf_size=1)
        out = Emst['output']
        vertices = out[:, 0:2]
        edges = out[:, 2]
        return vertices, edges

    def place_points(self, vertices, edges):
        n = self.points.shape[0]

        # P' = maintain locations in R^2 of mapped points
        # Components = maintain indices of points in each component/cluster
        comps = [np.array([i]) for i in range(n)]
        points_prime = np.zeros((n, 2))

        for i in range(n - 1):
            pair = vertices[i]  # Pair is the pair of endpoints of the i-th smallest edge
            d = edges[i]  # d is the length of the smallest edge

            # Get component A and component B containing point A and point B
            comp_a = 0
            comp_b = 0
            for j in range(len(comps)):
                if pair[0] in comps[j]:
                    comp_a = j
                if pair[1] in comps[j]:
                    comp_b = j

            # Get the points in component a and b
            points_a = np.array([points_prime[x] for x in comps[comp_a]])
            points_b = np.array([points_prime[x] for x in comps[comp_b]])
            # print("A:", points_a.shape[0], "B:", points_b.shape[0])
            # print(points_a, "\n\n", points_b, "\n ------------------")

            # Align component a with left edge point at (0, d)
            if points_a.shape[0] == 1:
                rotated_points = np.array([[0, d]])

            elif points_a.shape[0] == 2:
                rotated_points = rotate_hull(points_a, np.array([0, 1]), top=False, method=self.method,
                                             edge_point=points_prime[int(pair[0])])
                rotated_points = np.add(rotated_points, np.array([0, d]))

            else:
                hull = ConvexHull(points_a)
                rotated_points = rotate_hull(points_a, hull.vertices, top=False, method=self.method,
                                             edge_point=points_prime[int(pair[0])])
                rotated_points = np.add(rotated_points, np.array([0, d]))

            for k in range(comps[comp_a].shape[0]):
                points_prime[comps[comp_a][k]] = rotated_points[k]

            # Align component b
            if points_b.shape[0] == 1:
                rotated_points = np.array([[0, 0]])

            elif points_b.shape[0] == 2:
                rotated_points = rotate_hull(points_b, np.array([0, 1]), top=True, method=self.method,
                                             edge_point=points_prime[int(pair[1])])

            else:
                hull = ConvexHull(points_b)
                rotated_points = rotate_hull(points_b, hull.vertices, top=True, method=self.method,
                                             edge_point=points_prime[int(pair[1])])

            for k in range(comps[comp_b].shape[0]):
                points_prime[comps[comp_b][k]] = rotated_points[k]

            # Merge the components together
            comp_merged = np.append(comps[comp_a], comps[comp_b])
            comps[comp_a] = comp_merged
            comps.pop(comp_b)

        # print("Final component:\n", points_prime)
        self.r2_points = points_prime

    def fit(self, X, y=None):
        self.points = X
        self.target = y

        v, e = self.compute_EMST()
        self.place_points(v, e)

    def fit_transform(self, X, y=None):
        self.points = X
        self.target = y

        v, e = self.compute_EMST()
        self.place_points(v, e)
        return self.r2_points

    def plot(self):
        x, y = self.r2_points[:, 0], self.r2_points[:, 1]
        if self.target is not None:
            plt.scatter(x=x, y=y, c=self.target)
        else:
            plt.scatter(x=x, y=y)
        plt.show()

    def plot_data(self):
        x, y = self.points[:, 0], self.points[:, 1]
        if self.target is not None and self.points.shape[1] == 3:
            z = self.points[:, 2]
            fig = plt.figure(figsize=(4,4))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x, y, z, c=self.target)
        elif self.points.shape[1] == 3:
            z = self.points[:, 2]
            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x, y, z)
        elif self.target is not None:
            plt.scatter(x, y, c=self.target)
        else:
            plt.scatter(x, y)
        plt.show()

    def plot_persistence(self):
        dgm_original = ripser.ripser(self.points)['dgms'][0]
        dgm_after_topomap = ripser.ripser(self.r2_points)['dgms'][0]
        distance_bottleneck, matching = persim.bottleneck(dgm_original, dgm_after_topomap, matching=True)
        print(distance_bottleneck)
        # rips.plot(diagrams, show=True)
