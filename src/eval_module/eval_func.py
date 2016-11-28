#!/usr/bin/env python
# -*- coding:utf-8 -*-
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import sklearn.cross_validation as scv

def eval_test(dict_train, dict_test):

    y = []
    predicted = []
    for train_id in dict_train:
        try:
            train_label = dict_train[train_id]
            test_label = dict_test[train_id]
            predicted.append(test_label)
            y.append(train_label)
        except KeyError:
            # print train_id
            pass
    # return y, predicted
    return eval_matrix(y, predicted)

def eval_matrix(labels, pre):
    # npre = confusion_matrix(labels, pre)
    # for x in npre:
    #     for y in x:
    #         print str(y) + '\t',
    #     print '\n'
    return classification_report(labels, pre, digits=5).replace('\n\n', '\n')

# if __name__ == "__main__":
#     # biol chem geog hist math phys
#     subject = 'phys'
#     # test_pre_path = 'D:/haozhenyuan/workspace/subjectClassify_py/TFIDF/classification_subject/re/testalabel/'\
#     #                 + subject + '.txt'
#     test_pre_path = 'D:/haozhenyuan/workspace/subjectClassify_py/TFIDF/classification_subject/re/testalabel/' \
#                     + subject + '.txt'
#     train_dict = io_util.load_csv_train_dict(
#         path_config.train_csv_path + subject + '.csv', 5)
#
#     test_dict = io_util.load_csv_test_dict(
#         path_config.test_csv_path + subject + '.csv', test_pre_path)
#
#     print len(test_dict), len(train_dict)
#     y, predicted = eval(train_dict, test_dict)
#     print eval_matrix(y, predicted)
