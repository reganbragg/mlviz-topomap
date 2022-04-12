from mlpack import emst
import numpy as np
import math
from scipy.spatial import ConvexHull


# Get the angle between two points
def get_angle(p1, p2):
    delta_y = p2[1] - p1[1]
    delta_x = p2[0] - p1[0]
    theta = -1*math.atan2(delta_y, delta_x)
    return theta


# Rotate the hull so that the edge corresponding to the first two endpoints is either aligned at the top or bottom
def rotate_hull(points, vertices, top):

    # Get the angle to rotate by
    if top:
        theta = get_angle(points[vertices[0]], points[vertices[1]])
    else:
        theta = get_angle(points[vertices[0]], points[vertices[1]]) + math.pi

    center_index = vertices[0]
    center = points[center_index]
    points_rotated = np.empty((0, 2))

    # For all the points, rotate them around the first point
    for i in range(points.shape[0]):
        # Translate the point to the origin and rotate
        if i != vertices[0]:
            y = points[i][1] - center[1]
            x = points[i][0] - center[0]
            y_prime = x*math.sin(theta) + y*math.cos(theta) + center[1]
            x_prime = x*math.cos(theta) - y*math.sin(theta) + center[0]
            points_rotated = np.vstack((points_rotated, np.array([x_prime, y_prime])))
        else:
            points_rotated = np.vstack((points_rotated, center))

    # Translate the points back
    points_rotated = np.subtract(points_rotated, center)
    return points_rotated


class TopoMap:

    def __init__(self, points):
        self.points = points
        self.r2_points = None

    def compute_EMST(self):
        Emst = emst(input=self.points, naive=False)
        # Emst: matrix where each row represents an edge in the Emst
        # First dimension is lesser index, second dimension is higher index
        # Third dimension is the edge length
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
            d = edges[i]  # d is the smallest edge

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

            # Align component a with left edge point at (0, d)
            if points_a.shape[0] == 1:
                rotated_points = np.array([[0, d]])

            elif points_a.shape[0] == 2:
                rotated_points = rotate_hull(points_a, np.array([0, 1]), top=True)
                rotated_points = np.add(rotated_points, np.array([0, d]))

            else:
                hull = ConvexHull(points_a)
                rotated_points = rotate_hull(points_a, hull.vertices, top=True)
                rotated_points = np.add(rotated_points, np.array([0, d]))

            for k in range(comps[comp_a].shape[0]):
                points_prime[comps[comp_a][k]] = rotated_points[k]

            # Align component b
            if points_b.shape[0] == 1:
                rotated_points = np.array([[0, 0]])

            elif points_b.shape[0] == 2:
                rotated_points = rotate_hull(points_b, np.array([0, 1]), top=False)

            else:
                hull = ConvexHull(points_b)
                rotated_points = rotate_hull(points_b, hull.vertices, top=False)

            for k in range(comps[comp_b].shape[0]):
                points_prime[comps[comp_b][k]] = rotated_points[k]

            # Merge the components together
            comp_merged = np.append(comps[comp_a], comps[comp_b])
            comps[comp_a] = comp_merged
            comps.pop(comp_b)

        self.r2_points = points_prime

    def algo(self):
        v, e = self.compute_EMST()
        self.place_points(v, e)


data = np.array([[1, 2, 3], [6, 2, 6], [3, 7, 12], [8, 10, 1], [-4, 5, -2]])
t = TopoMap(data)
t.algo()