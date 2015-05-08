import time
import numpy as np
#import numexpr as np
import pickle
import operator
from operator import add 
from pyspark import SparkContext, SparkConf
from scipy.sparse import csr_matrix
import sys 
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine

time0 = time.time()
sc = SparkContext(appName = "logmf")

test = sc.textFile('/Users/nali/data/test.txt')\
        .map(lambda x: x.split(','))\
        .map(lambda x: (x[0], (x[1:])))


predict = sc.textFile('predict.txt')\
        .map(lambda x: x.split(','))\
        .map(lambda x: (x[0], x[1]))

inter = test.cogroup(predict).map(lambda x: (len(x[1][0].data[0]),  len(set(x[1][0].data[0]).intersection(set(x[1][1])))))
acc = inter.map(lambda x: x[1]/float(1)).reduce(add)/float(101223)
recall = inter.map(lambda x: x[1]/float(x[0])).reduce(add)/float(101223)
print (1, acc, recall)


for i in range(10):
    n = (i+1) * 5 + 1
    predict = sc.textFile('predict.txt')\
            .map(lambda x: x.split(','))\
            .map(lambda x: (x[0], (x[1:n])))
    inter = test.cogroup(predict).map(lambda x: (len(x[1][0].data[0]),  len(set(x[1][0].data[0]).intersection(set(x[1][1].data[0])))))
    acc = inter.map(lambda x: x[1]/float(n-1)).reduce(add)/float(101223)
    recall = inter.map(lambda x: x[1]/float(x[0])).reduce(add)/float(101223)
    print (n-1, acc, recall)
