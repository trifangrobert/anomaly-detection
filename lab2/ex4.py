import numpy as np
import matplotlib.pyplot as plt
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from pyod.utils.utility import standardizer
from pyod.models.combination import average, maximization

def preprocess_data(filename='cardio.mat'):
    dataset = loadmat(filename)
    
    print(f"dataset.keys(): {dataset.keys()}")
    print(f"dataset['X'].shape: {dataset['X'].shape}")
    print(f"dataset['y'].shape: {dataset['y'].shape}")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(dataset['X'], dataset['y'], test_size=0.2, random_state=42)
    
    # Normalize the data
    X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
    # X_test = (X_test - X_train.mean(axis=0)) / X_train.std(axis=0) 
    X_test = (X_test - X_test.mean(axis=0)) / X_test.std(axis=0)
    contamination = (y_train == 1).sum() / len(y_train)
    
    return X_train, X_test, y_train, y_test, contamination
    

def solve(X_train, X_test, y_train, y_test, model_type, contamination, n_neighbors):
    model = model_type(n_neighbors=n_neighbors, contamination=contamination)
    
    model.fit(X_train)
    
    y_pred_train = model.labels_
    y_pred_test = model.predict(X_test)
    
    train_balanced_accuracy = balanced_accuracy_score(y_train, y_pred_train)
    test_balanced_accuracy = balanced_accuracy_score(y_test, y_pred_test)
    print(f"model_type: {model_type.__name__}, n_neighbors: {n_neighbors}, train_balanced_accuracy: {train_balanced_accuracy}, test_balanced_accuracy: {test_balanced_accuracy}")
    
    train_score = model.decision_scores_
    test_score = model.decision_function(X_test)
    
    return train_score, test_score
    

def main():
    X_train, X_test, y_train, y_test, contamination = preprocess_data()
    
    knn_train_scores = []
    knn_test_scores = []
    
    lof_train_scores = []
    lof_test_scores = []
    
    for n_neighbors in range(30, 121, 10):
        train_score, test_score = solve(X_train, X_test, y_train, y_test, KNN, contamination, n_neighbors)
        knn_train_scores.append(train_score)
        knn_test_scores.append(test_score)
        
    for n_neighbors in range(30, 121, 10):
        train_score, test_score = solve(X_train, X_test, y_train, y_test, LOF, contamination, n_neighbors)
        lof_train_scores.append(train_score)
        lof_test_scores.append(test_score)
        
    # import pdb; pdb.set_trace()
    knn_train_scores = np.array(knn_train_scores).T
    knn_test_scores = np.array(knn_test_scores).T
    
    lof_train_scores = np.array(lof_train_scores).T
    lof_test_scores = np.array(lof_test_scores).T
    
    # Normalize the scores
    knn_train_scores = standardizer(knn_train_scores)
    knn_test_scores = standardizer(knn_test_scores)
    
    lof_train_scores = standardizer(lof_train_scores)
    lof_test_scores = standardizer(lof_test_scores)
    
    # average
    knn_train_avg = average(knn_train_scores)
    knn_test_avg = average(knn_test_scores)
    
    lof_train_avg = average(lof_train_scores)
    lof_test_avg = average(lof_test_scores)
    
    # maximization
    knn_train_max = maximization(knn_train_scores)
    knn_test_max = maximization(knn_test_scores)
    
    lof_train_max = maximization(lof_train_scores)
    lof_test_max = maximization(lof_test_scores)

    # Compute the threshold
    knn_train_threshold_avg = np.quantile(knn_train_avg, 1 - contamination)
    knn_test_threshold_avg = np.quantile(knn_test_avg, 1 - contamination)
    knn_train_threshold_max = np.quantile(knn_train_max, 1 - contamination)
    knn_test_threshold_max = np.quantile(knn_test_max, 1 - contamination)
    
    lof_train_threshold_avg = np.quantile(lof_train_avg, 1 - contamination)
    lof_test_threshold_avg = np.quantile(lof_test_avg, 1 - contamination)
    lof_train_threshold_max = np.quantile(lof_train_max, 1 - contamination)
    lof_test_threshold_max = np.quantile(lof_test_max, 1 - contamination)
    
    
    # Compute the predictions
    knn_train_pred_avg = knn_train_avg > knn_train_threshold_avg
    knn_test_pred_avg = knn_test_avg > knn_test_threshold_avg
    knn_train_pred_max = knn_train_max > knn_train_threshold_max
    knn_test_pred_max = knn_test_max > knn_test_threshold_max
    
    lof_train_pred_avg = lof_train_avg > lof_train_threshold_avg
    lof_test_pred_avg = lof_test_avg > lof_test_threshold_avg
    lof_train_pred_max = lof_train_max > lof_train_threshold_max
    lof_test_pred_max = lof_test_max > lof_test_threshold_max
    
    # Compute the balanced accuracy
    knn_train_bacc_avg = balanced_accuracy_score(y_train, knn_train_pred_avg)
    knn_test_bacc_avg = balanced_accuracy_score(y_test, knn_test_pred_avg)
    knn_train_bacc_max = balanced_accuracy_score(y_train, knn_train_pred_max)
    knn_test_bacc_max = balanced_accuracy_score(y_test, knn_test_pred_max)
    
    lof_train_bacc_avg = balanced_accuracy_score(y_train, lof_train_pred_avg)
    lof_test_bacc_avg = balanced_accuracy_score(y_test, lof_test_pred_avg)
    lof_train_bacc_max = balanced_accuracy_score(y_train, lof_train_pred_max)
    lof_test_bacc_max = balanced_accuracy_score(y_test, lof_test_pred_max)
    
    print(f"KNN - average - train_bacc: {knn_train_bacc_avg}, test_bacc: {knn_test_bacc_avg}")
    print(f"KNN - maximization - train_bacc: {knn_train_bacc_max}, test_bacc: {knn_test_bacc_max}")
    
    print(f"LOF - average - train_bacc: {lof_train_bacc_avg}, test_bacc: {lof_test_bacc_avg}")
    print(f"LOF - maximization - train_bacc: {lof_train_bacc_max}, test_bacc: {lof_test_bacc_max}")
    

if __name__ == "__main__":
    main()