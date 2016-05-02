#!/usr/bin/python
import sys, os
import numpy as np
import scipy as sci
from sklearn.preprocessing import scale
from sklearn import linear_model, neighbors, preprocessing, cross_validation, svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, classification_report
from matplotlib import pyplot as plt


def proc(a, b):
    files = os.listdir(a)
    num_var = len(b) - 1

    ##### check file #####
    if ('exam.dat.txt' not in files):
        sys.exit("Error: exam.dat.txt must exist as train set.")

    tempD = np.loadtxt("exam.dat.txt",dtype=np.str_,delimiter=" ")
    row, col = tempD.shape
    for i in range(row):
        for j in range(1,col):
            tempD[i][j] = tempD[i][j].split(":")[1]

    y = np.array(tempD[0:,0],dtype=float)
    X = np.array((tempD[0:,1:]),dtype=float)


    ##### check input #####
    if num_var == 1:
        filename = b[1]
        message1 = "Warning: This is a regular reminder. For possible errors followed,"
        message2 = " please check test file should be in the same format as exam.dat.txt"
        print message1+message2
        if (filename not in files):
            sys.exit("Error: file used for test set not found!")   
        tempD = np.loadtxt("exam.dat.txt",dtype=np.str_,delimiter=" ")
        row_test, col_test = tempD.shape
        for i in range(row_test):
            for j in range(1,col_test):
                tempD[i][j] = tempD[i][j].split(":")[1]

        y_test = np.array(tempD[0:,0],dtype=float)
        X_test = np.array((tempD[0:,1:]),dtype=float)
        X = np.vstack([X_test, X]) 
        X = scale(X, axis=0, with_mean=True, with_std=True, copy=True)
        X_test = X[0:row_test,]
        X = X[row_test:,]
    elif num_var == 4:
        features = b[1:5]
        features = [float(i) for i in features]
        X = np.vstack([features,X])
        X = scale(X, axis=0, with_mean=True, with_std=True, copy=True)
        X_test = X[0,]
        X = X[1:,]
        print X[0,]
        print X_test
    else: sys.exit("Error: num of arguments must be 4 for vector input.")
    return X, y, X_test

##### radial basis function SVM fitting and prediction #####
############    C = 4.7    and    Gamma = 1.1   ############
def pred(a,b):
    Xtr, ytr, Xte = proc(a,b)
    rsvm = svm.SVC(kernel='rbf', C = 4.7, gamma = 1.1)
    rsvm.fit(Xtr,ytr)
    yhat = rsvm.predict(Xte)
    return yhat

ypred = pred('./',sys.argv)
print ypred