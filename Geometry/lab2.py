import matplotlib.pyplot as plt
import random

# Node of a KDTree (k-dimensional tree)
class KDTreeNode:
    def __init__(self, coordinates):
        self.coordinates = coordinates  # Each point is represented by a tuple (x, y)
        self.left_child = None
        self.right_child = None

# Function to build a KDTree from a list of points
def build_KDTree(points, depth):
    if not points:  # If no points, return None
        return None
    if len(points) == 1:  # If only one point, create a node and return
        return KDTreeNode(points[0])

    # Alternately partition the points array
    axis = depth % 2

    # Sort points based on axis
    points.sort(key=lambda x: x[axis])

    # Choose the median point
    median_index = len(points) // 2

    # Create a node with the median point
    node = KDTreeNode(points[median_index])

    # Recursively construct left and right subtrees
    node.left_child = build_KDTree(points[:median_index], depth + 1)
    node.right_child = build_KDTree(points[median_index+1:], depth + 1)

    return node

# Helper function to check if a point is within a given region along a specified axis
def is_point_in_region(point, region, axis):
    return region[axis][0] <= point[axis] <= region[axis][1]

# Function to perform a regional search on the KDTree
def regional_search(node, region, depth, points_in_region=None):
    # If points_in_region is not supplied, we initialize it as an empty list
    if points_in_region is None:
        points_in_region = []
        
    # If the node is None (we've reached a leaf), return the points collected so far
    if not node:
        return points_in_region

    # Select the axis based on depth so that axis cycles between 0 and 1
    axis = depth % 2

    # If the point is within the region's boundary along the considered axis,
    # then we need to check both left and right subtrees
    if is_point_in_region(node.coordinates, region, axis):
        # Check the right subtree
        points_in_region = regional_search(node.right_child, region, depth + 1, points_in_region)
        # Check the left subtree
        points_in_region = regional_search(node.left_child, region, depth + 1, points_in_region)

        # If the point is within the region's boundary along the other axis as well,
        # then it's within the region and we add it to the result list
        if is_point_in_region(node.coordinates, region, (axis+1) % 2):
            points_in_region.append(node.coordinates)
    else:
        # If the node's point is outside the region on the current axis,
        # then only one side of the split can contain points that are in the region
        if node.coordinates[axis] < region[axis][0]:
            # If the node's point is to the left of the region, then only points in the right subtree
            # could potentially be in the region
            points_in_region = regional_search(node.right_child, region, depth + 1, points_in_region)
        if node.coordinates[axis] > region[axis][1]:
            # If the node's point is to the right of the region, then only points in the left subtree
            # could potentially be in the region
            points_in_region = regional_search(node.left_child, region, depth + 1, points_in_region)

    # Return the list of points found within the region
    return points_in_region


# List of random points
random_points = [[random.randint(0, 20), random.randint(0, 20)] for _ in range(12)]

# Define region as two ranges [x1, x2] and [y1, y2]
random_region = [[random.randint(0, 10), random.randint(10, 20)], [random.randint(0, 10), random.randint(10, 20)]]

# Build a KDTree
kd_tree_root = build_KDTree(random_points, 0)

# Perform regional search to get the points within the region
points_within_region = regional_search(kd_tree_root, random_region, 0)

# Plot the points and region
fig, ax = plt.subplots(figsize=(8,8))
for point in random_points:
    if point in points_within_region:
        ax.scatter(*point, color='green')
    else:
        ax.scatter(*point, color='red')

# List the corner points of the region
region_corners = [(random_region[0][0], random_region[1][0]), (random_region[0][1], random_region[1][0]), (random_region[0][1], random_region[1][1]), (random_region[0][0], random_region[1][1])]

# Plot the region
x_coords, y_coords = zip(*(region_corners + [region_corners[0]]))  # Connect the last point to the first to complete the region
plt.plot(x_coords, y_coords)

plt.show()
