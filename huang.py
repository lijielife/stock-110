#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn import svm
from sklearn..grid_search import GridSearchCV

scores = ['accuracy', 'precision', 'recall']

def rbf(data, direction):
    parameters = [{'kernel':['rbf'], 'gamma':[1e-3,1e-4], 'C':[1,10,100,1000]}]
    clf = 
    

def lda(data, direction):
    parameters = [{'kernel':['linear'], 'C':[1,10,100,1000]}]
    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, scoring=score, n_jobs=-1)
    clf.fit(data, direction)
    
    # the best parameter
    print clf.best_estimator_
    
    for params, mean_score, all_score in clf.grid_scores_:
        print "{:.3f} (+/- {:.3f}) for {}".format(mean_score, all_scores.std() / 2, params) 
    
    return clf
    
    """
    for sigma in np.logspace(-4, -2, 8):
        for C in np.logspace(4, 8, 8):
            clf = svm.SVC(C=C, gamma=sigma, 
            scores = cross_validation.
    """

def main():
    sp500 = []
    jpy = []
    nikkei225 = np.array([])
    
    data = np.array(sp500[:-1], jpy[:-1]).T
    data_q = map(lambda x:[x[0],y[1],x[0]**2,x[0]*x[1],x[1]**2], zip(sp500,jpy))
    direction = nikkei225[:-1] - nikkei225[0:]
    
    split = 100
    train, train_q, train_label = data[:split], data_q[:split], direction[:split]
    test, test_q, test_label = data[split:], data_q[split:], direction[spilt:]
    
    # lda
    clf = lda(train, train_label)
    lda_test_predict = clf.predict(test)
    #print classification_report(test_label, lda_test_predict)
    lda_train_predict = clf.predict(train)
    
    # qda
    clf = lda(train_q, train_label)
    qda_test_predict = clf.predict(test)
    qda_train_predict = clf.predict(train_q)
    
    # RNN
    
    # rbf
    clf = rbf(train, train_label)
    rbf_test_predict = clf.predict(test)
    rbf_train_predict = clf.predict(train)
    
    
if __name__ == '__main__':
    main()