#!/usr/bin/env python
# -*- coding:utf-8 -*-
import ConfigParser
import codecs

train_csv_path = u'D:/haozhenyuan/学科分类/原始train7.19 - 副本/%s.csv'
test_csv_path = u'D:/haozhenyuan/学科分类/all-need-tag-data - 副本/%s.csv'
train_csv_path_small = u'D:/haozhenyuan/学科分类/原始train7.19 - 副本/%s.csv'

vec_path = u'D:/haozhenyuan/学科分类/traintest/vec/%s.nolabel.vec'
vec_lab_path = u'D:/haozhenyuan/学科分类/train/NLPIR/%s.txt'

jieba_dict = u"D:/haozhenyuan/subjectclassify2/src/token_stream/dict.txt"

# path_config = u"D:/haozhenyuan/subjectclassify2/data/8.8_nlpir/path.config"
# path_config = u"D:/haozhenyuan/subjectclassify2/data/11.11_jieba/path.config"
# path_config = u"D:/haozhenyuan/subjectclassify2/data/11.13关键词提取/path.config"
path_config = u"D:/haozhenyuan/subjectclassify2/data/11.15_2gram/path.config"
envir = 'WindowsServer2012'
cf = ConfigParser.ConfigParser()

cf.readfp(codecs.open(path_config, "r", "utf-8-sig"))

root_path = cf.get(envir, 'root_path')
eval_path = cf.get(envir, 'eval_path')
cache_path = cf.get(envir, 'cache_path')
pre_path = cf.get(envir, 'pre_path')
nlpir_root = cf.get(envir, 'nlpir_root')
stream = cf.get(envir, 'stream')

import aaaa
if __name__ == "__main__":
    flist = ['biol', 'chem', 'geog', 'hist', 'math', 'phys']
    for fname in flist:
        print fname, stream
        l5_eval_5fold, l6_eval_5fold = aaaa.classify_test_fea(fname)
        print "5fold_l5" + l5_eval_5fold
        print "5fold_l6" + l6_eval_5fold

    aaaa.fw_eval_fold_l5.close()
    aaaa.fw_eval_fold_l6.close()
    aaaa.fw_eval_test_l5.close()
    aaaa.fw_eval_test_l6.close()

