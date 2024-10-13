from pyod.utils.data import generate_data
import matplotlib.pyplot as plt
from pyod.models.knn import KNN
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import numpy as np


X_train, X_test, y_train, y_test = generate_data(n_train=400, n_test=100, contamination=0.1, n_features=2)

def ex1():
    normal_points = X_train[y_train == 0]
    anomalous_points = X_train[y_train == 1]
    plt.scatter(normal_points[:,0], normal_points[:,1], color="green")
    plt.scatter(anomalous_points[:,0], anomalous_points[:,1], color="orange")
    plt.savefig("ex1.png")
    
def balanced_accuracy(conf_matrix):
    TP = conf_matrix[1,1]
    TN = conf_matrix[0,0]
    FP = conf_matrix[0,1]
    FN = conf_matrix[1,0]
    
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    
    return (TPR + TNR) / 2
    
def classifier(contamination=0.1):
    print(f"Classifying with contamination {contamination}")
    
    clf = KNN(contamination=contamination)
    clf.fit(X_train)
    
    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)
    
    train_pred_probs = clf.decision_function(X_train)
    test_pred_probs = clf.decision_function(X_test)
    
    return train_pred, test_pred, train_pred_probs, test_pred_probs

def plot_roc_curve(ground_truth, pred_probs, c, split_name):
    fpr, tpr, thresholds = roc_curve(ground_truth, pred_probs)
    plt.plot(fpr, tpr, color="blue")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve for contamination {c} and split {split_name}")
    plt.savefig(f"ex2_roc_{split_name}_{c}.png")
    plt.clf()

def ex2():
    for c in [0.1, 0.2, 0.3, 0.4, 0.5]:
        train_pred, test_pred, train_pred_probs, test_pred_probs = classifier(c)
        
        train_cm = confusion_matrix(y_train, train_pred)
        test_cm = confusion_matrix(y_test, test_pred)
        
        train_ba = balanced_accuracy(train_cm)
        test_ba = balanced_accuracy(test_cm)
        
        print(f"Train BA: {train_ba}")
        print(f"Test BA: {test_ba}")    
        
        plot_roc_curve(y_train, train_pred_probs, c, "train")
        plot_roc_curve(y_test, test_pred_probs, c, "test")
    

def ex3():
    contamination_rate = 0.1
    X_train, X_test, y_train, y_test = generate_data(n_train=1000, n_test=0, contamination=contamination_rate, n_features=1)
    
    # detect outliers in the training data using z-score
    mean_value = X_train.mean()
    std_value = X_train.std()
    z_scores = (X_train - mean_value) / std_value
    z_scores = np.abs(z_scores)
    
    threshold = np.quantile(z_scores, 1 - contamination_rate)
    print(f"Threshold: {threshold}")
    
    pred = z_scores > threshold
    y_train = y_train.astype(bool)
    
    cm = confusion_matrix(y_train, pred)
    ba = balanced_accuracy(cm)
    
    print(f"Balanced accuracy: {ba}")
    
# https://stats.stackexchange.com/questions/147210/efficient-fast-mahalanobis-distance-computation
def z_score(x, L):
    # compute z score for one sample
    
    # let y = L^-1 * x
    # then x = L * y
    
    y = np.linalg.solve(L, x)
    return np.linalg.norm(y) ** 2


def ex4():
    n_features = 4
    n_samples = 1000
    contamination_rate = 0.1
    n_outliers = int(n_samples * contamination_rate)
    n_inliers = n_samples - n_outliers
    
    # play around with the mean and covariance matrix to generate different datasets
    mean = np.array([0] * n_features)
    cov_matrix = np.eye(n_features)
    
    inliers = np.random.multivariate_normal(mean, cov_matrix, n_inliers)
    
    # add noise to generate outliers
    noise = np.random.normal(0, 1, (n_outliers, n_features))
    outliers = np.random.multivariate_normal(mean, cov_matrix, n_outliers) + noise
    
    X = np.concatenate((inliers, outliers))
    y = np.concatenate((np.zeros(n_inliers), np.ones(n_outliers)))
    
    L = np.linalg.cholesky(cov_matrix) # L * L^T = cov_matrix        
    
    z_scores = np.array([z_score(x, L) for x in X])
    
    threshold = np.quantile(z_scores, 1 - contamination_rate)
    pred = z_scores > threshold
    
    cm = confusion_matrix(y, pred)
    ba = balanced_accuracy(cm)
    
    print(f"Balanced accuracy: {ba}")
    

if __name__ == "__main__":
    ex1()
    ex2()
    ex3()
    ex4()