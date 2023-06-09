import math
import matplotlib.pyplot as plt
import random

# Class to represent a point in 2D space
class PlanePoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Function to calculate Euclidean distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

# Function to find closest pair of points using brute force
def brute_force_closest_points(point_list):
    min_distance = float("inf")  # Initialize minimum distance as infinite
    closest_pair = []  # Initialize closest pair

    # Compare each pair of points
    for i in range(len(point_list)):
        for j in range(i+1, len(point_list)):
            current_distance = calculate_distance(point_list[i], point_list[j])

            # If current pair is closer, update closest pair and minimum distance
            if current_distance < min_distance:
                closest_pair = [point_list[i], point_list[j]]
                min_distance = current_distance

    return closest_pair

# Function to find closest pair in a strip of points
def closest_points_in_strip(strip):
    min_distance = float("inf")
    closest_pair = []

    # Sort strip by y coordinate
    strip.sort(key=lambda point: point.y)

    # Compare each pair of points in strip
    for i in range(len(strip)):
        for j in range(i+1, len(strip)):

            # Break loop if difference in y coordinate is greater than minimum distance
            if (strip[j].y - strip[i].y) >= min_distance:
                break

            current_distance = calculate_distance(strip[i], strip[j])

            # If current pair is closer, update closest pair and minimum distance
            if current_distance < min_distance:
                closest_pair = [strip[i], strip[j]]
                min_distance = current_distance

    return closest_pair

# Recursive function to find closest pair of points
def recursive_closest_points(point_list, num_points):
    # If 3 or fewer points, use brute force
    if num_points <= 3:
        return brute_force_closest_points(point_list)

    mid = num_points // 2
    mid_point = point_list[mid]

    # Recursively find closest pairs in left and right halves
    left_closest = recursive_closest_points(point_list[:mid], mid)
    right_closest = recursive_closest_points(point_list[mid:], num_points - mid)

    # Find closer pair of the two halves
    if calculate_distance(left_closest[0], left_closest[1]) < calculate_distance(right_closest[0], right_closest[1]):
        closest_pair = left_closest
        min_distance = calculate_distance(left_closest[0], left_closest[1])
    else:
        closest_pair = right_closest
        min_distance = calculate_distance(right_closest[0], right_closest[1])

    # Initialize strip
    strip = [point for point in point_list if abs(point.x - mid_point.x) < min_distance]

    # Find closest pair in strip if it exists
    if len(strip) > 1:
        strip_closest = closest_points_in_strip(strip)
        if calculate_distance(strip_closest[0], strip_closest[1]) < min_distance:
            closest_pair = strip_closest

    return closest_pair

# Function to find closest pair of points in a set
def closest_points_pair(point_list):
    # Sort list by x coordinate
    point_list.sort(key=lambda point: point.x)

    # Find closest pair
    return recursive_closest_points(point_list, len(point_list))

# Create list of random points
random_points = [PlanePoint(random.uniform(0, 20), random.uniform(0, 20)) for _ in range(20)]

# Find closest pair of points
closest_points = closest_points_pair(random_points)

# Calculate distance between closest pair of points
min_distance = calculate_distance(closest_points[0], closest_points[1])

# Plot points and closest pair
fig, ax = plt.subplots(figsize=(8,8))
for point in random_points:
    if point in closest_points:
        ax.scatter(point.x, point.y, color='red')
    else:
        ax.scatter(point.x, point.y, color='blue')

# Plot line between closest pair of points
plt.plot([closest_points[0].x, closest_points[1].x], [closest_points[0].y, closest_points[1].y], color='red')

# Label plot with distance
plt.text((closest_points[0].x + closest_points[1].x) / 2, (closest_points[0].y + closest_points[1].y) / 2 + 1, f'Distance: {min_distance:.2f}')

# Show plot
plt.show()
