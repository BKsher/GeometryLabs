import copy
import math

import matplotlib.pyplot as plt
import numpy


# Define a class for Vertex
class Vertex:
    # The constructor for the Vertex class
    def __init__(self, x_coordinate, y_coordinate, index) -> None:
        self.incomingVertices = None
        self.outgoingVertices = None

        self.x = x_coordinate
        self.y = y_coordinate
    
        self.index = index  # The index of this vertex
        self.weight = 1  # Initial weight of the vertex
        self.neighbours = []  # The list of neighbouring vertices

    # Overload the "<" operator to compare two vertices based on their y and x coordinates
    def __lt__(self, other):
        return self.y < other.y or (self.y == other.y and self.x < other.x)

    # Overload the "str()" function to print the vertex index and coordinates
    def __str__(self):
        return str(f"{self.index}: ({self.x}; {self.y})")

    # Method to create incoming and outgoing vertices lists
    def createInOutLists(self):
        self.incomingVertices = []  # The list of incoming vertices
        self.outgoingVertices = []  # The list of outgoing vertices

        # For each neighbouring vertex, categorize it as incoming or outgoing based on its index
        for neighbour in self.neighbours:
            if neighbour.index < self.index:
                self.incomingVertices.append(neighbour)
            else:
                self.outgoingVertices.append(neighbour)

        # Sort the incoming vertices in clockwise order and then reverse the list
        self.incomingVertices = sortVerticesClockwise(self.incomingVertices, self)
        self.incomingVertices.reverse()

        # Sort the outgoing vertices in clockwise order
        self.outgoingVertices = sortVerticesClockwise(self.outgoingVertices, self)

    # Method to draw a plot of the vertex on the given Axes object
    def buildPlot(self, ax):
        # Plot the vertex as a red point
        ax.scatter(self.x, self.y, color="red")
        return ax



# Define a class for Graph
class Graph:
    # The constructor for the Graph class
    def __init__(self):
        self.vertices = []  # The list of vertices in the graph
        self.adjacencyMatrix = None  # The adjacency matrix representing the graph

    # Method to add a vertex to the graph
    def addVertex(self, vertex):
        self.vertices.append(vertex)

    # Method to add an edge between two vertices
    def addEdge(self, firstVertexIndex, secondVertexIndex):
        self.vertices[firstVertexIndex].neighbours.append(self.vertices[secondVertexIndex])
        self.vertices[secondVertexIndex].neighbours.append(self.vertices[firstVertexIndex])

    # Method to initialize the adjacency matrix of the graph
    def initializeAdjacencyMatrix(self):
        self.adjacencyMatrix = []
        for i, _ in enumerate(self.vertices):
            self.adjacencyMatrix.append([-1] * len(self.vertices))

        # Assigning weight to each edge in the adjacency matrix
        for vertex in self.vertices:
            for neighbour in vertex.neighbours:
                self.adjacencyMatrix[vertex.index][neighbour.index] = 1

    # Method to calculate the chains in the graph
    def getChains(self):
        self.sortVertices()  # Sort the vertices in ascending order based on their coordinates
        self.balanceVertices()  # Balance the vertices to ensure each outgoing edge is utilized as much as possible
        adjacencyMatrixCopy = copy.deepcopy(self.adjacencyMatrix)  # Create a copy of the adjacency matrix

        chainsCount = 0
        # Count the number of chains by summing the weights in the adjacency matrix of the first vertex's outgoing edges
        for vertex in self.vertices[0].outgoingVertices:
            chainsCount += adjacencyMatrixCopy[self.vertices[0].index][vertex.index]

        chains = []
        # Construct the chains based on the calculated chain count
        for _ in range(chainsCount):
            chain = []
            currentVertex = self.vertices[0]
            # Traverse the graph to build the chain until reaching the last vertex
            while currentVertex != self.vertices[-1]:
                chain.append(currentVertex)
                j = 0
                # Find the next outgoing vertex with a positive weight in the adjacency matrix
                while adjacencyMatrixCopy[currentVertex.index][currentVertex.outgoingVertices[j].index] < 1:
                    j += 1

                # Decrease the weights in the adjacency matrix for the selected edges
                adjacencyMatrixCopy[currentVertex.index][currentVertex.outgoingVertices[j].index] -= 1
                adjacencyMatrixCopy[currentVertex.outgoingVertices[j].index][currentVertex.index] -= 1
                currentVertex = currentVertex.outgoingVertices[j]

            chain.append(self.vertices[-1])  # Append the last vertex to complete the chain
            chains.append(chain)  # Add the constructed chain to the list of chains

        return chains


    # Method to sort the vertices
    def sortVertices(self):
        self.vertices.sort()
        for i, vertex in enumerate(self.vertices):
            vertex.index = i

    # Method to plot the graph
    def buildPlot(self):
        xCoordinates = []
        yCoordinates = []
        labels = []

        for vertex in self.vertices:
            xCoordinates.append(vertex.x)
            yCoordinates.append(vertex.y)
            labels.append(str(vertex.index))

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

        ax.scatter(xCoordinates, yCoordinates, color="green")

        ax.set_xlim([min(xCoordinates) - 1, max(xCoordinates) + 1])
        ax.set_ylim([min(yCoordinates) - 1, max(yCoordinates) + 1])

        for vertex in self.vertices:
            for neighbour in vertex.neighbours:
                ax.plot([vertex.x, neighbour.x], [vertex.y, neighbour.y], "green")

        for i in range(len(xCoordinates)):
            ax.annotate(labels[i], (xCoordinates[i], yCoordinates[i]), xytext=(xCoordinates[i] - 0.025, yCoordinates[i] + 0.1))

        ax.set_ylabel("y", rotation=0, labelpad=20)
        ax.set_xlabel("x", rotation=0, labelpad=20)

        return fig, ax

    # Method to balance the vertices
    def balanceVertices(self):
        self.initializeAdjacencyMatrix()

        self.createInOutVerticesLists()

        # Balance the outgoing vertices
        for index in range(1, len(self.vertices) - 1):
            currentVertex = self.vertices[index]
            leftMostVertex = currentVertex.outgoingVertices[0]
            weightIncoming = sum(self.adjacencyMatrix[vertex.index][currentVertex.index] for vertex in currentVertex.incomingVertices)
            weightOutgoing = sum(self.adjacencyMatrix[currentVertex.index][vertex.index] for vertex in currentVertex.outgoingVertices)

            if weightIncoming > weightOutgoing:
                self.adjacencyMatrix[currentVertex.index][leftMostVertex.index] += weightIncoming - weightOutgoing
                self.adjacencyMatrix[leftMostVertex.index][currentVertex.index] = self.adjacencyMatrix[currentVertex.index][leftMostVertex.index]

        # Balance the incoming vertices
        for index in range(len(self.vertices) - 1, 0, -1):
            currentVertex = self.vertices[index]
            leftMostVertex = currentVertex.incomingVertices[0]
            weightIncoming = sum(self.adjacencyMatrix[vertex.index][currentVertex.index] for vertex in currentVertex.incomingVertices)
            weightOutgoing = sum(self.adjacencyMatrix[currentVertex.index][vertex.index] for vertex in currentVertex.outgoingVertices)

            if weightOutgoing > weightIncoming:
                self.adjacencyMatrix[leftMostVertex.index][currentVertex.index] += weightOutgoing - weightIncoming
                self.adjacencyMatrix[currentVertex.index][leftMostVertex.index] = self.adjacencyMatrix[leftMostVertex.index][currentVertex.index]

    # Method to create incoming and outgoing vertices lists
    def createInOutVerticesLists(self):
        for vertex in self.vertices:
            vertex.createInOutLists()

    
# Determine vertices of subgraphs that contain the given point
def getPointLocationVertices(chain1, chain2, point):
    chainEnd = max(len(chain1), len(chain2))
    chain1Index = 0
    chain2Index = 0
    subgraphs = [[]]

    # Merge the chains into subgraphs based on matching vertex indices
    while chain1Index < chainEnd and chain2Index < chainEnd:
        if chain1[chain1Index].index == chain2[chain2Index].index:
            if len(subgraphs) > 0:
                subgraphs[-1].append(chain1[chain1Index])

            subgraphs.append([])
            subgraphs[-1].append(chain1[chain1Index])
            chain1Index += 1
            chain2Index += 1
        else:
            while chain1[chain1Index].index < chain2[chain2Index].index:
                subgraphs[-1].append(chain1[chain1Index])
                chain1Index += 1

            while chain2[chain2Index].index < chain1[chain1Index].index:
                subgraphs[-1].append(chain2[chain2Index])
                chain2Index += 1

    vertexIndices = []
    # Check each subgraph to see if it contains the given point
    for subgraph in subgraphs:
        if isPointInSubgraph(subgraph, point):
            for vertex in subgraph:
                vertexIndices.append(vertex.index)

    return vertexIndices



# Determine if a point is inside a subgraph
def isPointInSubgraph(subgraph, point):
    vertexCount = len(subgraph)
    isInside = False

    firstVertexX, firstVertexY = subgraph[0].x, subgraph[0].y

    # Check if the point is inside the subgraph using the winding number algorithm
    for i in range(vertexCount + 1):
        secondVertexX, secondVertexY = subgraph[i % vertexCount].x, subgraph[i % vertexCount].y

        if min(firstVertexY, secondVertexY) < point.y <= max(firstVertexY, secondVertexY):
            if point.x <= max(firstVertexX, secondVertexX):
                if firstVertexY != secondVertexY:
                    # Calculate the x-coordinate of the intersection between the horizontal line passing through the point
                    # and the line segment formed by the current and next vertices
                    xIntersect = (point.y - firstVertexY) * (secondVertexX - firstVertexX) / (secondVertexY - firstVertexY) + firstVertexX

                # Check if the point is on the left side of the line segment
                if firstVertexX == secondVertexX or point.x <= xIntersect:
                    isInside = not isInside

        firstVertexX, firstVertexY = secondVertexX, secondVertexY

    return isInside



# Localize the point in the chains
def localizePoint(currentChains, point):
    isOnLeft = False
    centerIndex = int(len(currentChains) / 2 - 1)

    # If there are more than one chains
    if len(currentChains) > 1:
        isOnLeft = discriminatePoint(currentChains[centerIndex], point)

        # Check if the point is between the current center chain and the next chain
        if not isOnLeft and discriminatePoint(currentChains[centerIndex + 1], point):
            return [currentChains[centerIndex], currentChains[centerIndex + 1]]

    # If there is only one chain
    if len(currentChains) == 1:
        if discriminatePoint(currentChains[0], point):
            return [None, currentChains[0]]  # The point is on the left side of the chain
        else:
            return [currentChains[0], None]  # The point is on the right side of the chain

    # Recursively localize the point in the appropriate half of the remaining chains
    if isOnLeft:
        return localizePoint(currentChains[:centerIndex + 1], point)  # Localize in the left half
    else:
        return localizePoint(currentChains[centerIndex + 1:], point)  # Localize in the right half



# Determine on which side of the chain the point is
def discriminatePoint(chain, point):
    if len(chain) == 2:
        return isPointOnLeft(chain[0], chain[1], point)

    chainLength = len(chain)
    centerIndex = int(chainLength / 2)
    if point.y < chain[centerIndex].y:
        return discriminatePoint(chain[:centerIndex + 1], point)
    else:
        return discriminatePoint(chain[centerIndex:], point)


# Determine if the point is on the left side of the line between chainPoint1 and chainPoint2
def isPointOnLeft(chainPoint1, chainPoint2, point):
    return ((chainPoint2.x - chainPoint1.x) * (point.y - chainPoint1.y) - (chainPoint2.y - chainPoint1.y) * (point.x - chainPoint1.x)) >= 0


# Sort vertices in clockwise order around a given center_vertex
def sortVerticesClockwise(vertices, centerVertex):
    return sorted(vertices, key=lambda vertex: math.atan2(vertex.y - centerVertex.y, vertex.x - centerVertex.x), reverse=True)


# Get graph data from a file
def getData(path):
    file = open(path, "r")
    
    nVertices = int(file.readline())
    nEdges = int(file.readline())

    graph = Graph()

    for i in range(nVertices):
        line = str(file.readline())
        coordinates = line.split(sep=" ")
        graph.addVertex(Vertex(float(coordinates[0]), float(coordinates[1]), i))

    for i in range(nEdges):
        line = str(file.readline())
        vertices = line.split(sep=" ")
        graph.addEdge(int(vertices[0]), int(vertices[1]))

    return graph


# Build plot for a chain of vertices
def buildPlotForChain(chain, ax, color, offset):
    for i in range(1, len(chain)):
        ax.plot([chain[i - 1].x + offset, chain[i].x + offset], [chain[i - 1].y + offset, chain[i].y + offset], color=color)

    return ax


# Get graph data from input file
graph = getData("input.txt")

# Create a point vertex
point = Vertex(6, 4, -1)

# Get chains in the graph
chains = graph.getChains()

# Localize the point within the chains
pointLocationChains = localizePoint(chains, point)
pointLocationChains = list(filter(lambda item: item is not None, pointLocationChains))

# Build the plot of the graph
figure, axes = graph.buildPlot()

# If the point is located between multiple chains
if len(pointLocationChains) > 1:
    print("\nThe point is located between chains: ")
    for chain in pointLocationChains:
        if chain is not None:
            # Uncomment the following line to plot the chain
            # axes = buildPlotForChain(chain, axes, "red", 0)
            
            print("Chain: ", end="")
            for i in range(0, len(chain)):
                print(chain[i].index, end=" ")
            print("")
    
    # Get the vertex indices of subgraphs containing the point
    res = getPointLocationVertices(pointLocationChains[0], pointLocationChains[1], point)
    
    print("\nAnswer: ")
    for i in res:
        print(i, end=" ")

# If the point is outside of the graph
else:
    print("\nThe point is outside of the graph")

# Plot the point on the graph
axes = point.buildPlot(axes)
plt.show()
