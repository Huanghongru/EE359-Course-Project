# classical approach PCA, SVM and Logistic regression

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
import time
import argparse
from utils import *

def load_data(dir=DATA_PATH):
    data = np.load(dir+'data.npy')
    label = np.load(dir+'target.npy')
    print("Successfully load data and labels.")
    return data, label

def doPCA(data,n_components=0):
    """
    Perform PCA on the input data
    :param data: the original data
    :param n_components: param used in sklearn's PCA
    :return: processed data
    """
    start = time.time()
    print("Starting PCA with n_components={}.".format("default" if n_components==0 else n_components))
    if n_components == 0:
        pca_model = PCA()
    else:
        pca_model = PCA(n_components=n_components)
    pca_model.fit(data)
    print("PCA time cost: {}".format(time.time()-start))
    joblib.dump(pca_model, PCA_MODEL_PATH + "n_components={}.pkl".format("default" if n_components==0 else n_components))
    processed_data = pca_model.transform(data)
    np.save(PCA_DATA_PATH + "n_components={}.npy".format("default" if n_components==0 else n_components), processed_data)
    print("Variance ratio:",pca_model.explained_variance_ratio_)
    return processed_data

def LR(data, labels,
       regularization_coef = 1,
       penalty = 'l1',
       verbose = 1):
    """
    Perform logistic regression on data
    :param data: the input data
    :param labels: the corresponding labels
    :return:
    """
    start = time.time()
    print("Starting Logistic Regression.")
    clf = LogisticRegression(penalty=penalty, C=regularization_coef, verbose=verbose)
    scores = cross_val_score(clf, data, labels, cv=5)
    print("LR time cost:",time.time()-start)
    print("CV scores:",scores)
    return np.average(scores)

def LinearSVM(data, labels,
              penalty_coef = 1,
              penalty = 'l1',
              max_iter = 10000,
              verbose = 1):
    start = time.time()
    print("Starting Linear SVM.")
    if data.shape[0] > data.shape[1]:
        clf = LinearSVC(penalty=penalty, C=penalty_coef, max_iter=max_iter, dual=False, verbose=verbose)
    else:
        clf = LinearSVC(penalty='l2', C=penalty_coef, max_iter=max_iter, dual=True, verbose=verbose)
    scores = cross_val_score(clf, data, labels, cv=5)
    print("Linear SVM time cost:",time.time()-start)
    print("CV scores:",scores)
    return np.average(scores)

def SVM(data, lebels,
        penalty_coef = 1,
        kernel = 'rbf',
        verbose = 1):
    start = time.time()
    print("Starting SVM with kernel={}.".format(kernel))
    clf = SVC(kernel=kernel, C=penalty_coef, verbose=verbose)
    scores = cross_val_score(clf, data, labels, cv=5)
    print("Kernel SVM time cost:", time.time() - start)
    print("CV scores:", scores)
    return np.average(scores)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode")
    args = parser.parse_args()

    data, labels = load_data()

    ''' PCA '''
    if args.mode == 'PCA':
        for para in N_COMPONENTS:
            doPCA(data, para)

    ''' LR '''
    if args.mode == 'LR':
        with open(LR_RES_PATH+"LR_result.txt", 'w') as f:
            for c in C:
                acc = LR(data, labels, regularization_coef=c)
                print("original data, c={}: {}".format(c, acc))
                f.write("original data, c={}: {}\n".format(c, acc))
            for c in C:
                for para in N_COMPONENTS:
                    name = "n_components=" + str("default" if para == 0 else para)
                    data = np.load(PCA_DATA_PATH + name + '.npy')
                    acc = LR(data, labels, regularization_coef=c)
                    print("{}, c={}: {}".format(name, c, acc))
                    f.write("{}, c={}: {}\n".format(name, c, acc))

    ''' LSVM '''
    if args.mode == 'LSVM':
        with open(LSVM_RES_PATH+"LSVM_result.txt", 'w') as f:
            for c in C:
                acc = LinearSVM(data, labels, penalty_coef=c)
                print("original data, c={}: {}".format(c, acc))
                f.write("original data, c={}: {}\n".format(c, acc))
            for c in C:
                for para in N_COMPONENTS:
                    name = "n_components=" + str("default" if para == 0 else para)
                    data = np.load(PCA_DATA_PATH + name + '.npy')
                    acc = LinearSVM(data, labels, penalty_coef=c)
                    print("{}, c={}: {}".format(name, c, acc))
                    f.write("{}, c={}: {}\n".format(name, c, acc))

    ''' SVM '''
    if args.mode == 'SVM':
        with open(SVM_RES_PATH+"SVM_result.txt", 'w') as f:
            for kernel in KERNEL:
                for c in C:
                    acc = SVM(data, labels, kernel=kernel, penalty_coef=c)
                    print("original data, kernel={}, c={}: {}".format(kernel, c, acc))
                    f.write("original data, kernel={}, c={}: {}\n".format(kernel, c, acc))
            for kernel in KERNEL:
                for c in C:
                    for para in N_COMPONENTS:
                        name = "n_components=" + str("default" if para == 0 else para)
                        data = np.load(PCA_DATA_PATH + name + '.npy')
                        acc = SVM(data, labels, kernel=kernel, penalty_coef=c)
                        print("{}, kernel={}, c={}: {}".format(name, kernel, c, acc))
                        f.write("{}, kernel={}, c={}: {}\n".format(name, kernel, c, acc))
