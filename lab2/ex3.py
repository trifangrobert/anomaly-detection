import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from pyod.models.knn import KNN
from pyod.models.lof import LOF

def solve(n_neighbors):
    # Generate clusters with different densities
    centers = [(-10, -10), (10, 10)]
    std_devs = [2, 6]
    X, _ = make_blobs(n_samples=[200, 100], centers=centers, cluster_std=std_devs, random_state=42)

    # Specify the contamination rate
    contamination = 0.07

    # Initialize KNN and LOF models
    knn = KNN(contamination=contamination, n_neighbors=n_neighbors)
    lof = LOF(contamination=contamination, n_neighbors=n_neighbors)

    # Fit the models
    knn.fit(X)
    lof.fit(X)
    
    y_pred_knn = knn.predict(X)
    y_pred_lof = lof.predict(X)
    
    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred_knn)
    plt.title(f"KNN (n_neighbors={n_neighbors})")
    
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred_lof)
    plt.title(f"LOF (n_neighbors={n_neighbors})")
    
    plt.tight_layout()
    plt.savefig(f"./ex3_{n_neighbors}.png")
    

def main():
    for n_neighbors in [1, 3, 5]:
        solve(n_neighbors)
    

if __name__ == "__main__":
    main()