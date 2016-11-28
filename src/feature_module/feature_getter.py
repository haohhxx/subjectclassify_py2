#!/usr/bin/env python
# -*- coding:utf-8 -*-

import config
import csv
import codecs

from token_stream.token_stream import nlpir_token_stream
from token_stream.token_stream import ngram_token_stream

from util import feature_util

class feature_tfidf_parser(object):
    def __init__(self, token_stream):
        self.token_stream = token_stream
        pass

    def compute_corpus(self, subject):
        corpus_train = []  # 切词后用空格分开的文本
        labels5 = []  # 标签5
        labels6 = []  # 标签6
        q_nub_train = []  # 题的编号

        corpus_test = []  # 切词后用空格分开的文本
        q_nub = []  # 题的编号

        csv_file = open(config.train_csv_path % subject, 'rb')
        csv_file.readline()
        train_reader = csv.reader(csv_file)
        for line in train_reader:
            q_nub_train.append(line[1])
            labels5.append(line[5])
            labels6.append(line[6])
            corpus_train.append(line[2] + ' ' + line[3] + ' ' + line[4] + ' ' + line[7])

        csv_file = open(config.test_csv_path % subject, 'rb')
        test_reader = csv.reader(csv_file)
        for line in test_reader:
            q_nub.append(line[1])
            try:
                corpus_test.append(line[2] + ' ' + line[3] + ' ' + line[4] + ' ' + line[7])
            except IndexError:
                corpus_test.append(" ")
        corpus_train_test = []
        corpus_train_test.extend(corpus_train)
        corpus_train_test.extend(corpus_test)
        X_Xt = feature_util.tfidf_feature(corpus_train_test)
        X_Xt = feature_util.SVD_Vec(X_Xt, 1000)
        X = X_Xt[0:len(labels5)]
        Xt = X_Xt[(len(labels5)):]
        return X, labels5, labels6, q_nub_train, Xt, q_nub

    def compute_train_corpus(self, subject):
        corpus_train = []  # 切词后用空格分开的文本
        labels5 = []  # 标签5
        labels6 = []  # 标签6
        q_nub_train = []  # 题的编号

        csv_file = open(config.train_csv_path % subject, 'rb')
        csv_file.readline()
        train_reader = csv.reader(csv_file)
        for line in train_reader:
            q_nub_train.append(line[1])
            labels5.append(line[5])
            labels6.append(line[6])
            l = line[2] + ' ' + line[3] + ' ' + line[4] + ' ' + line[7]
            l = self.token_stream.parse(l)
            l = ' '.join(l)
            corpus_train.append(l)

        # print corpus_train
        # fw = open(u'D:\\haozhenyuan\\学科分类2\\data\\8.8_nlpir\\nlpir_stream_train\\' + subject, 'w')
        # for train_line in corpus_train:
        #     # print train_line
        #     fw.write(codecs.encode(train_line, 'utf-8') + '\n')
        #     fw.flush()
        # fw.close()

        X = feature_util.tfidf_feature(corpus_train)
        X = feature_util.SVD_Vec(X, 1000)

        return X, labels5, labels6, q_nub_train


if __name__ == "__main__":
    p = u"函数是顺序执行,遇到return语句或者最后一行函数语句"
    # print p
    # ts = nlpir_token_stream()
    # print ts.parse(p)

    # ts = jieba_token_stream()
    # print ts.parse(p)
    # print ts.jieba_pos_parse(p)
    # print ts.jieba_ext_parse(p)
    #
    ts = ngram_token_stream(3)
    print ts.parse(p)



