import click
import math
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def euclidean_distance(p1, p2):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(p1, p2)))

# Compute centroid of a cluster 
def mean(points):
    n_points = len(points)
    n_dims = len(points[0])
    means = [0] * n_dims
    for point in points:
        for i in range(n_dims):
            means[i] += point[i]
    return [x / n_points for x in means]


def plot_kmeans(data, centroids, labels):
    data = np.array(data)
    centroids = np.array(centroids)
    labels = np.array(labels)

    plt.figure(figsize=(8, 6))

    # Scatter plot of data points colored by label
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.6, marker='o', label='Data Points')

    # Plot centroids with larger red X markers
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, c='red', label='Centroids')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('K-means Clustering Result')
    plt.legend()
    plt.grid(True)
    plt.show()


@click.command()
@click.argument("filepath")
@click.argument("centroids_count")
def kmeans(filepath, centroids_count, max_iters=100):

    centroids_count = int(centroids_count)
    data = pd.read_csv(filepath)
    data = data.values.tolist()
    
    # Randomly initialize centroids from data points
    centroids = random.sample(data, centroids_count)
    
    for _ in range(max_iters):
        clusters = [[] for _ in range(centroids_count)]
        
        # Assign points to nearest centroid
        for point in data:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            min_index = distances.index(min(distances))
            clusters[min_index].append(point)
        
        new_centroids = []
        for cluster in clusters:
            if cluster:  
                new_centroids.append(mean(cluster))
            else:
                new_centroids.append(random.choice(data))
        
        # Check convergence
        if all(euclidean_distance(c1, c2) < 1e-4 for c1, c2 in zip(centroids, new_centroids)):
            break
        
        centroids = new_centroids
    
    # Assign final labels
    labels = []
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        labels.append(distances.index(min(distances)))
    
    print("Labels:", labels)
    print("Centroids:", centroids)
    plot_kmeans(data, centroids, labels)
    
    #return labels, centroids

