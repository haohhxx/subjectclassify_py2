#!/usr/bin/env python
# -*- coding:utf-8 -*-

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier


from classifier.classifier_nfold import classifier_nfold
from classifier.classifier_test import classifier_test
from feature_module.feature_getter import feature_tfidf_parser
from token_stream.token_stream import *
from eval_module import eval_func
import numpy as np
import config

import re

def classify_test_fea(subject):
    # tok = jieba_token_stream()
    # tok = nlpir_token_stream()
    # tok = ngram_token_stream(2)

    tok = {
        '2gram': ngram_token_stream(2),
        'nlpir': nlpir_token_stream(),
        'jieba': jieba_token_stream(),
        'jieba_key': jieba_ext_token_stream(),
        'jieba_pos_n': jieba_pos_token_stream()
    }[config.stream]


    tfidf_parser = feature_tfidf_parser(tok)
    # X, labels5, labels6, q_nub_train, Xt, q_nub_test = tfidf_parser.compute_corpus(subject)
    X, labels5, labels6, q_nub_train = tfidf_parser.compute_train_corpus(subject)
    # X = np.array(X, dtype=np.float64)

    train_dict = {}
    test_dict = {}

    #
    # 选择分类器
    #

    cla1 = classifier_nfold_GradientBoosting(fold)
    cla2 = classifier_nfold_GradientBoosting(fold)


    for index in range(len(labels5)):
        train_dict[q_nub_train[index]] = labels5[index]

    pre, y_n = cla1.classify(X, labels5)
    l5_eval_5fold = eval_func.eval_matrix(y_n, pre)


    for index in range(len(labels6)):
        train_dict[q_nub_train[index]] = labels6[index]

    pre, y_n = cla2.classify(X, labels6)
    l6_eval_5fold = eval_func.eval_matrix(y_n, pre)

    # test pre
    # cla_test = classifier_test_LR_LDA()
    # pre_test = cla_test.classify(X, y, Xt)
    #
    # for index in range(len(q_nub_test)):
    #     test_dict[q_nub_test[index]] = pre_test[index]

    # eval_test = eval_func.eval_test(train_dict, test_dict)
    # save_test_pre(eval_test, subject)
    # save_pre_test(pre_test, subject)
    eval_test = ''
    l5_eval_5fold, number = re.subn('\s{2,15}', '\t', l5_eval_5fold)
    l6_eval_5fold, number = re.subn('\s{2,15}', '\t', l6_eval_5fold)
    save_fold_pre(l5_eval_5fold, l6_eval_5fold, subject)
    # l5_eval_5fold = l5_eval_5fold.replace('\s{2,8}', '\t')
    # l6_eval_5fold = l6_eval_5fold.replace('\s{2,8}', '\t')
    return l5_eval_5fold, l6_eval_5fold

def save_test_pre(evalre_l5, evalre_l6, subject):
    fw_eval_test_l5.write(subject)
    fw_eval_test_l6.write(subject)
    fw_eval_test_l5.write(evalre_l5)
    fw_eval_test_l6.write(evalre_l6)
    fw_eval_test_l5.flush()
    fw_eval_test_l6.flush()
def save_fold_pre(evalre_l5, evalre_l6, subject):
    fw_eval_fold_l5.write(subject)
    fw_eval_fold_l6.write(subject)
    fw_eval_fold_l5.write(evalre_l5)
    fw_eval_fold_l6.write(evalre_l6)
    fw_eval_fold_l5.flush()
    fw_eval_fold_l6.flush()
def save_pre_test(pres, subject):
    fw_pre = open(config.pre_path +subject+".txt", 'w')
    for pre in pres:
        fw_pre.write(pre+'\n')
    fw_pre.close()



class classifier_nfold_KNN(classifier_nfold):
    # c1nub = 1
    # c2nub = 1
    def use_classify(self, X_train, y_train, X_test):
        clf = KNeighborsClassifier()
        clf.fit(X_train, y_train)
        pre = clf.predict(X_test)
        return pre


class classifier_nfold_GradientBoosting(classifier_nfold):
    # c1nub = 1
    # c2nub = 1
    def use_classify(self, X_train, y_train, X_test):
        clf = GradientBoostingClassifier()
        clf.fit(X_train, y_train)
        pre = clf.predict(X_test)
        return pre

class classifier_nfold_LDA(classifier_nfold):
    # c1nub = 4
    # c2nub = 10
    def use_classify(self, X_train, y_train, X_test):
        clf = LDA()
        clf.fit(X_train, y_train)
        pre = clf.predict(X_test)
        return pre

class classifier_nfold_LR_LDA(classifier_nfold):
    # c1nub = 4
    # c2nub = 10
    def use_classify(self, X_train, y_train, X_test):
        clf1 = LogisticRegression()
        clf2 = LDA()
        clf1.fit(X_train, y_train)
        X_train = clf1.predict_proba(X_train)
        clf2.fit(X_train, y_train)

        pre = clf2.predict(X_train)
        return pre


class classifier_test_LR_LDA(classifier_test):
    def use_classify(self, X_train, y_train, X_test):
        clf1 = LogisticRegression()
        clf2 = LDA()
        clf1.fit(X_train, y_train)
        X_train = clf1.predict_proba(X_train)
        clf2.fit(X_train, y_train)

        X_test = clf1.predict_proba(X_test)
        pre = clf2.predict(X_test)
        return pre

class classifier_test_LDA(classifier_test):
    def use_classify(self, X_train, y_train, X_test):
        clf = LDA()
        clf.fit(X_train, y_train)
        return clf.predict(X_test)

fold = 5
fw_eval_fold_l5 = open(config.eval_path+'5fold/l5/GradientBoosting.1.1.txt', 'w')
fw_eval_fold_l6 = open(config.eval_path+'5fold/l6/GradientBoosting.1.1.txt', 'w')
fw_eval_test_l5 = open(config.eval_path+'result_test/l5/GradientBoosting.1.1.txt', 'w')
fw_eval_test_l6 = open(config.eval_path+'result_test/l6/GradientBoosting.1.1.txt', 'w')
# path_config = u"D:/haozhenyuan/学科分类2/data/8.8_nlpir/path.config"11.15_2gram
# path_config = u"D:/haozhenyuan/学科分类2/data/8.8_nlpir/path.config"

if __name__ == "__main__":
    flist = ['biol', 'chem', 'geog', 'hist', 'math', 'phys']
    for fname in flist:
        print fname
        l5_eval_5fold, l6_eval_5fold = classify_test_fea(fname)
        print "5fold_l5" + l5_eval_5fold
        print "5fold_l6" + l6_eval_5fold

    fw_eval_fold_l5.close()
    fw_eval_fold_l6.close()
    fw_eval_test_l5.close()
    fw_eval_test_l6.close()

