from pyod.utils.data import generate_data_clusters
from pyod.models.knn import KNN
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score
import numpy as np

def main():
    X_train, X_test, y_train, y_test = generate_data_clusters(n_train=400, n_test=200, n_clusters=2, n_features=2, contamination=0.1)
    y_train = np.array(y_train, dtype=int)
    y_test = np.array(y_test, dtype=int)
    colors = ['red', 'blue']
    
    
    for n_neighbors in [1, 3, 5]:
        clf = KNN(n_neighbors=n_neighbors)
        clf.fit(X_train)
        
        y_train_pred = clf.labels_
        y_test_pred = clf.predict(X_test)
        
        train_balanced_accuracy = balanced_accuracy_score(y_train, y_train_pred)
        test_balanced_accuracy = balanced_accuracy_score(y_test, y_test_pred)
        
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        
        axs[0, 0].scatter(X_train[:, 0], X_train[:, 1], c=[colors[y] for y in y_train])
        axs[0, 0].set_title(f'Ground truth labels for training data (n_neighbors={n_neighbors})')
        
        axs[0, 1].scatter(X_train[:, 0], X_train[:, 1], c=[colors[y] for y in y_train_pred])
        axs[0, 1].set_title(f'Predicted labels for training data (n_neighbors={n_neighbors}) - BACC: {train_balanced_accuracy:.3f}')
        
        axs[1, 0].scatter(X_test[:, 0], X_test[:, 1], c=[colors[y] for y in y_test])
        axs[1, 0].set_title(f'Ground truth labels for test data (n_neighbors={n_neighbors})')
        
        axs[1, 1].scatter(X_test[:, 0], X_test[:, 1], c=[colors[y] for y in y_test_pred])
        axs[1, 1].set_title(f'Predicted labels for test data (n_neighbors={n_neighbors}) - BACC: {test_balanced_accuracy:.3f}')
        
        plt.tight_layout()
        plt.savefig(f'./ex2_knn_{n_neighbors}.png')
    

if __name__ == "__main__":
    main()