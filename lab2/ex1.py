import numpy as np
import random
from matplotlib import pyplot as plt

def compute_leverage_scores(x):
    # compute leverage scores using singular value decomposition
    # H = X * (X^T * X)^-1 * X^T
    # leverage scores = diag(H)
    
    # add a column of ones to X
    X = np.column_stack((x, np.ones(len(x))))
    
    n, d = X.shape
    U, S, V = np.linalg.svd(X)
    
    # print(f"U.shape: {U.shape}")
    # print(f"U values: {U}")
    
    # print(f"S.shape: {S.shape}")
    # print(f"S values: {S}")
    
    # print(f"V.shape: {V.shape}")
    # print(f"V values: {V}")
    
    # X = U * S * V^T
    # H = X * (X^T * X)^-1 * X^T -> H = U * S * V^T * (V * S^2 * V^T)^-1 * V * S * U^T
    # H = U * Id * U^T
    # leverage scores = diag(H)
    
    Id = np.zeros((n, n))
    for i in range(d):
        Id[i, i] = 1
        
    H = U @ Id @ U.T
    print(f"H.shape: {H.shape}")
    return np.diag(H)
    

def generate_data_task1(mean_x, variance_x, mean, variance, a, b, n):
    x = np.random.normal(mean_x, variance_x, n)
    y = a * x + b + np.random.normal(mean, variance, n)
    
    return x, y

def solve_task1(mean_x, variance_x, mean, variance, a, b, n):
    x, y = generate_data_task1(mean_x, variance_x, mean, variance, a, b, n)
    leverage_scores = compute_leverage_scores(x)
    
    return x, y, leverage_scores

def generate_data_task2(mean_x, variance_x, mean, variance, a, b, c, n):
    x1 = np.random.normal(mean_x, variance_x, n)
    x2 = np.random.normal(mean_x, variance_x, n)
    y = a * x1 + b * x2 + c + np.random.normal(mean, variance, n)
    x = np.column_stack((x1, x2))
    
    return x, y

def solve_task2(mean_x, variance_x, mean, variance, a, b, c, n):
    x, y = generate_data_task2(mean_x, variance_x, mean, variance, a, b, c, n)
    leverage_scores = compute_leverage_scores(x)
    
    return x, y, leverage_scores

def task1():
    a = 2
    b = 3
    n = 1000
    
    x_0, y_0, lev_0 = solve_task1(0, 1, 0, 1, a, b, n)
    x_1, y_1, lev_1 = solve_task1(0, 10, 0, 1, a, b, n)
    x_2, y_2, lev_2 = solve_task1(0, 1, 0, 10, a, b, n)
    x_3, y_3, lev_3 = solve_task1(0, 10, 0, 10, a, b, n)
    
    x = [x_0, x_1, x_2, x_3]
    y = [y_0, y_1, y_2, y_3]
    lev = [lev_0, lev_1, lev_2, lev_3]
    
    # print(f"leverage scores: {lev}")
    titles = ["regular", "high var on x", "high var on y", "high var on both"]
    
    # make 4 subplots with the scatter plots of the data
    plt.figure(figsize=(12, 12))
    
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.scatter(x[i], y[i], color="blue")
        pos = np.argsort(lev[i])[-30:]
        plt.scatter(x[i][pos], y[i][pos], color="red")
        plt.title(titles[i])
    
    plt.savefig("task1.png")
    
def task2():
    a = 2
    b = 3
    c = 4
    n = 1000
    
    x_0, y_0, lev_0 = solve_task2(0, 1, 0, 1, a, b, c, n)
    x_1, y_1, lev_1 = solve_task2(0, 10, 0, 1, a, b, c, n)
    x_2, y_2, lev_2 = solve_task2(0, 1, 0, 10, a, b, c, n)
    x_3, y_3, lev_3 = solve_task2(0, 10, 0, 10, a, b, c, n)
    
    x = [x_0, x_1, x_2, x_3]
    y = [y_0, y_1, y_2, y_3]
    lev = [lev_0, lev_1, lev_2, lev_3]
    
    # print(f"leverage scores: {lev}")
    titles = ["regular", "high var on x", "high var on y", "high var on both"]
    
    # make 4 subplots with the scatter plots of the data
    plt.figure(figsize=(12, 12))
    
    for i in range(4):
        x1 = x[i][:, 0]
        x2 = x[i][:, 1]
        plt.subplot(2, 2, i+1, projection="3d")
        plt.scatter(x1, x2, y[i], color="blue")
        # get the 30 points with the highest leverage scores
        pos = np.argsort(lev[i])[-30:]
        plt.scatter(x1[pos], x2[pos], y[i][pos], color="red")
        plt.title(titles[i])
    
    plt.savefig("task2.png")
    
def main():
    task1()
    task2()
    

if __name__ == "__main__":
    main()