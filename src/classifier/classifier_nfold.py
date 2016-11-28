#!/usr/bin/env python
# -*- coding:utf-8 -*-

from sklearn.cross_validation import StratifiedKFold
from util import feature_util
from eval_module import eval_func

class classifier_nfold():

    c1nub = 1
    c2nub = 1

    def __init__(self, nfold):
        self.nfold = nfold
        pass

    def classify(self, X ,y):
        preall = []
        yall = []
        kf = StratifiedKFold(y, n_folds=self.nfold)
        for train, test in kf:
            X_train, y_train = feature_util.getData(X, y, train)
            X_train, y_train = self.multi_feature(X_train, y_train)
            X_test, y_test = feature_util.getData(X, y, test)
            pre = self.use_classify(X_train, y_train, X_test)
            # pre = clf.predict(X_test)
            preall.extend(pre)
            yall.extend(y_test)
            # print eval_func.eval_matrix(yall, preall)
        return preall, yall

    # def classify_proba(self, X, y):
    #     preall = []
    #     yall = []
    #     kf = StratifiedKFold(y, n_folds=5)
    #     for train, test in kf:
    #         X_train, y_train = feature_util.getData(X, y, train)
    #         X_train, y_train = self.multi_feature(X_train, y_train)
    #         X_test, y_test = feature_util.getData(X, y, test)
    #         clf = self.use_classify(X_train, y_train)
    #         pre = clf.predict_proba(X_test)
    #         preall.extend(pre)
    #         yall.extend(y_test)
    #     return preall, yall

    def multi_feature(self, X_train, y_train):
        X_train_re = []
        y_train_re = []
        for index in range(len(y_train)):
            if (y_train[index] == '综合与拓展'):
                for iti in range(self.c2nub):
                    y_train_re.append(y_train[index])
                    X_train_re.append(X_train[index])
            elif (y_train[index] == '分析与应用'):
                for iti in range(self.c1nub):
                    y_train_re.append(y_train[index])
                    X_train_re.append(X_train[index])
            else:
                y_train_re.append(y_train[index])
                X_train_re.append(X_train[index])
        return X_train_re, y_train_re

    def use_classify(self, X_train, y_train, X_test):
        return ''
