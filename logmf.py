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

time_b = time.time()
sc = SparkContext(appName = "logmf")

class LogMF():
    def __init__(self, filename, K, L, num_factors, reg_param=0.6, gamma=1.0, iterations=30):
        self.filename = filename
        self.K = K
        self.L = L
        self.num_factors = num_factors
        self.reg_param = reg_param
        self.gamma = gamma
        self.iterations = iterations
   
    def mapping(self):
        data = sc.textFile(self.filename)\
                .map(lambda x: x.split(','))\
                .map(lambda x: (int(x[0]), int(x[1]), float(x[2])))

        users = data.map(lambda x: x[0]).distinct().collect()
        items = data.map(lambda x: x[1]).distinct().collect()
       
        output1 = open('users.txt','w')
        output2 = open('items.txt','w')

        for i in range(len(users)):
            output1.write(str(users[i])+'\n')
        for i in range(len(items)):
            output2.write(str(items[i])+'\n')

        self.l_u = len(users)
        self.l_i = len(items)
       
        global dict_u
        dict_u = dict(zip(users, range(self.l_u)))
        global dict_i
        dict_i = dict(zip(items, range(self.l_i)))

        self.dict_U = dict(zip(range(self.l_u), users))
        self.dict_I = dict(zip(range(self.l_i), items))
        
        total = data.map(lambda x: x[2]).reduce(add)
        num_zeros = self.l_u * self.l_i - data.count()
        alpha = num_zeros / total
        print 'alpha %.2f' % alpha
        
        self.data_map = data.map(lambda x: (dict_u[x[0]], dict_i[x[1]], x[2]*alpha))
        
        self.l_K = self.l_u/self.K
        self.l_L = self.l_i/self.L

        Row = []
        Col = []
        R = []
        n = -1
        for i in range(self.K):
            a = i * self.l_K
            b = (i+1) * self.l_K
            for j in range(self.L):
                n = n+1
                c = j * self.l_L
                d = (j+1)* self.l_L
                block = self.data_map.filter(lambda x: x[0] >= a and x[0] < b and x[1] >=c and x[1] < d)
                print 'block '+str(n)
                row = block.map(lambda x: x[0]).collect()
                row = map(lambda x: x - i*self.l_K, row)
                col = block.map(lambda x: x[1]).collect()
                col = map(lambda x: x - j*self.l_L, col)
                r = block.map(lambda x: x[2]).collect()
                Row.append(row)
                Col.append(col)
                R.append(r)
        self.Row = Row
        self.Col = Col
        self.R = R

    def train_model(self):
        
        self.user_vectors = np.random.normal(size=(self.l_u,
                                                   self.num_factors))
        self.item_vectors = np.random.normal(size=(self.l_i,
                                                   self.num_factors))
        self.user_biases = np.random.normal(size=(self.l_u, 1))
        self.item_biases = np.random.normal(size=(self.l_i, 1))
        
        user_vec_deriv_sum = np.zeros((self.l_u, self.num_factors))
        item_vec_deriv_sum = np.zeros((self.l_i, self.num_factors))
        user_bias_deriv_sum = np.zeros((self.l_u, 1))
        item_bias_deriv_sum = np.zeros((self.l_i, 1))
        
        def deriv_scale(user):
            if user:
                l = self.l_u
            else:
                l = self.l_i
            vec_deriv = np.zeros((l, self.num_factors))
            bias_deriv = np.zeros((l, 1))
            ones = np.ones((self.l_K, self.l_L)) #Return a new array of given shape and type, filled with ones.
            n = -1
            for i in range(self.K):
                a = i * self.l_K
                b = (i+1) * self.l_K
                u_v = self.user_vectors[a:b]
                u_b = self.user_biases[a:b]
                for j in range(self.L):
                    n += 1
                    c = j * self.l_L
                    d = (j+1)* self.l_L
                    i_v = self.item_vectors[c:d]
                    i_b = self.item_biases[c:d]
                    
                    row = self.Row[n]
                    col = self.Col[n]
                    r = self.R[n]
                    counts = csr_matrix((r,(row, col)),shape=(self.l_K, self.l_L)).toarray()                
                    A = np.dot(u_v, i_v.T)
                    A += u_b
                    A += i_b.T
                    A = np.exp(A)
                    A /= (A + ones)
                    A = (counts + ones) * A
                    if user:
                        v_d = np.dot(counts, i_v) - np.dot(A, i_v) - self.reg_param * u_v
                        b_d = np.expand_dims(np.sum(counts, axis=1), 1)\
                                - np.expand_dims(np.sum(A, axis=1), 1)
                        vec_deriv[a:b] = vec_deriv[a:b] + v_d
                        bias_deriv[a:b] = bias_deriv[a:b] + b_d

                    else:
                        v_d = np.dot(counts.T, u_v) - np.dot(A.T, u_v) - self.reg_param * i_v
                        b_d = np.expand_dims(np.sum(counts, axis=0), 1)\
                                - np.expand_dims(np.sum(A, axis=0), 1)
                        vec_deriv[c:d] = vec_deriv[c:d] + v_d
                        bias_deriv[c:d] = bias_deriv[c:d] + b_d
                    
            return (vec_deriv, bias_deriv)

        for k in range(self.iterations):
            t0 = time.time()
            
            user_vec_deriv, user_bias_deriv = deriv_scale(True)
            user_vec_deriv_sum += np.square(user_vec_deriv)
            user_bias_deriv_sum += np.square(user_bias_deriv)
            vec_step_size = self.gamma / np.sqrt(user_vec_deriv_sum)
            bias_step_size = self.gamma / np.sqrt(user_bias_deriv_sum)
            self.user_vectors += vec_step_size * user_vec_deriv
            self.user_biases += bias_step_size * user_bias_deriv
                    
            item_vec_deriv, item_bias_deriv = deriv_scale(False)
            item_vec_deriv_sum += np.square(item_vec_deriv)
            item_bias_deriv_sum += np.square(item_bias_deriv)
            vec_step_size = self.gamma / np.sqrt(item_vec_deriv_sum)
            bias_step_size = self.gamma / np.sqrt(item_bias_deriv_sum)
            self.item_vectors += vec_step_size * item_vec_deriv
            self.item_biases += bias_step_size * item_bias_deriv
            t1 = time.time()
            print 'iteration %i finished in %f seconds' % (k + 1, t1 - t0)

    def Prob_Sort_MPR(self):
        test = sc.textFile("/Users/nali/Beifei/ximalaya2015/code_ximalaya/data/testMarch17.txt")\
                .map(lambda x: x.split(','))\
                .map(lambda x: (int(x[0]), int(x[1]), float(x[2])))\
                .filter(lambda x: dict_u.has_key(x[0]) and dict_i.has_key(x[1]))\
                .map(lambda x: (dict_u[x[0]], dict_i[x[1]], x[2]))

        sum1 = 0
        sum2 = 0
        ones = np.ones((self.l_K, self.l_i)) #Return a new array of given shape and type, filled with ones.
        
        output_top50 = open('top50.txt','w')
        for i in range(self.K):
            a = i*self.l_K
            b = (i+1)*self.l_K
            u_v = self.user_vectors[a:b]
            u_b = self.user_biases[a:b]
            A = np.dot(u_v, self.item_vectors.T)
            A += u_b
            A += self.item_biases.T
            A = np.exp(A)
            A /= (A + ones)
            Rank = (-1*A).argsort().argsort()
            Sort = A.T[::-1].T
            
            top50 = np.nonzero(Rank < 50)[1].reshape(self.l_K, 50)
            def map_back(x):
                items = map(lambda xx: str(self.dict_I[xx]), x)
                return ','.join(items)
            top50 = '\n'.join(np.apply_along_axis(map_back, 1, top50))

            output_top50.write(top50+'\n')

            test_block = test.filter(lambda x: x[0] in range(a,b))
            row = test_block.map(lambda x: x[0]).collect()
            row = map(lambda x: x - i*self.l_u/self.K, row)
            col = test_block.map(lambda x: x[1]).collect()
            r = test_block.map(lambda x: x[2]).collect()
            counts = csr_matrix((r,(row, col)),shape=(self.l_K, self.l_i)).toarray() 
            
            sum1 += np.sum(Rank * counts / float(self.l_i-1))
            sum2 += np.sum(counts)
        MPR = sum1/sum2
        return MPR

    def correlation(self):
        output_corr = open('corr.txt','w')
        cos = pairwise_distances(self.item_vectors, metric="cosine")
        Rank = cos.argsort().argsort()
        top50 = np.nonzero(Rank < 50)[1].reshape(self.l_i, 50)
        def map_back(x):
            items = map(lambda xx: str(self.dict_I[xx]), x)
            return ','.join(items)
        top50 = '\n'.join(np.apply_along_axis(map_back, 1, top50))
        output_corr.write(top50+'\n')


    def WriteVectors(self):
        sc.parallelize(self.user_vectors.tolist())\
                .map(lambda x: ','.join(map(str, x)))\
                .saveAsTextFile('/Users/nali/Beifei/ximalaya2015/code_ximalaya/data/user_vec')
        sc.parallelize(self.item_vectors.tolist())\
                .map(lambda x: ','.join(map(str, x)))\
                .saveAsTextFile('/Users/nali/Beifei/ximalaya2015/code_ximalaya/data/item_vec')
        sc.parallelize(self.user_biases.tolist())\
                .map(lambda x: ','.join(map(str, x)))\
                .saveAsTextFile('/Users/nali/Beifei/ximalaya2015/code_ximalaya/data/user_bias')
        sc.parallelize(self.item_biases.tolist())\
                .map(lambda x: ','.join(map(str, x)))\
                .saveAsTextFile('/Users/nali/Beifei/ximalaya2015/code_ximalaya/data/item_bias')
    
logmf = LogMF('/Users/nali/Beifei/ximalaya2015/code_ximalaya/data/trainMarch17.txt',9,2, 20)
logmf.mapping()
logmf.train_model()
logmf.WriteVectors()
a = logmf.Prob_Sort_MPR()
open('MPR.txt','w').write(str(a))
time_e = time.time()
print "running time is %.2f seconds" % (time_e - time_b)
logmf.correlation()
result = (logmf.user_vectors, logmf.item_vectors, logmf.user_biases, logmf.item_biases)
pickle.dump(result, open('result.pkl','wb'))
result = pickle.load(open('result.pkl','rb'))
print "running time is %.2f seconds" % (time_e - time_b)
