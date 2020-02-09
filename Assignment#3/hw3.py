import numpy as np
import os
import pickle
import random
import time
from collections import Counter

def folder_list(path , label):
    '''
    读取所有评论文本,并且将文本与对应的label打包存储
    Args:
        path - 评论文本所在的文件夹路径
        label - 评论的情感 ,  {-1 , +1} , 其中-1表示negative , +1 positive
    Returns:
        review - 带label的评论文本 , 二维list(num_reviews , 2) , review[i][0]为第i个评论的文本(一维list(num_words)) , review[i][1]为第i个评论的label
    '''
    file_list = os.listdir(path)
    review = []
    for infile in file_list :
        file = os.path.join(path , infile)
        r = [read_data(file)]
        r.append(label)
        review.append(r)
    return review

def read_data(file):
    '''
    读取对应文件中的评论文本,去掉特殊字符后返回按照空格切割的单词list
    Args:
        file - 文本路径
    Returns:
        words - 文本单词列表 , 一维list(num_words)
    '''
    f = open(file)
    symbols = '${}()[].,:;+-*/&|<>=~" '
    lines = f.readlines()
    words = []
    for line in lines :
        line = line.strip()
        sentences = line.split(' ')
        words = words +  [w for w in sentences if w not in symbols]
    return words

def feature_extraction(X) :
    '''
    特征提取,用log函数降低词频,一方面起到类归一化的作用,另一方面相对减少了常用词的频数.
    Args:
        X - 输入特征的稀疏表示 , 一维list(num_reviews) , X[i]为第i个评论的输入特征对应的字典
    Returns:
        X - 经过处理后的特征的稀疏表示 , 一维list(num_reviews) , X[i]为第i个评论的输入特征对应的字典
    '''
    for X_sample in X :
        for k , v in X_sample.items() :
            X_sample[k] = np.log(v + 1)
    return X 

def feature_TFIDF(X , num_words) :
    '''
    用TFIDF(term frequency - inverse document frequency 词频-逆文本频率 指数)作为特征.
    Args:
        X - 输入特征的稀疏表示 , 一维list(num_reviews) , X[i]为第i个评论的输入特征对应的字典
        num_words - 评论文本的单词数 , 一维list(num_reviews)
    Returns:
        X - 经过处理后的特征的稀疏表示 , 一维list(num_reviews) , X[i]为第i个评论的输入特征对应的字典
    '''
    num_instances = len(X)
    for i in range(num_instances) :
        for k , v in X[i].items() :
            X[i][k] = v / np.log(num_words[i])
    return X

def feature_ngram(review , ngram = 1) :
    '''
    N-Gram模型作为特征.所谓N-gram是将原文本中连续的N个单词看成整体.
    Args:
        review - 评论文本 , 二维list(num_reviews , num_words)
        ngram - N-Gram模型的参数 , 标量 , 默认为1(unigram)
    Returns :
        X - 特征的稀疏表示 , 一维list(num_reviews) , X[i]为第i个评论的输入特征对应的字典(已经过上述特征抽取和频数的优化
    '''
    X = []
    for r in review :
        word_count = {}
        for i in range(len(r[0])) :
            if (i + ngram <= len(r[0])) :
                j = i + ngram - 1
                word_gram = ""
                k = i
                while (k < j) :
                    word_gram += r[0][k] + "_"
                    k += 1
                word_gram += r[0][k]
                word_count[word_gram] = word_count.get(word_gram , 0) + 1
            else :
                break
        X.append(word_count)
    return X


def split_review(review , train_size = 1500 , valid_size = 500 , ngram = 1) :
    '''
    划分训练集和验证集,并进行特征提取以及稀疏表示.
    Args:
        review - 评论文本 , 二维list(num_reviews , num_words)
        train_size - 测试集大小 , 标量
        valid_size - 验证集大小 , 标量
        ngram - N-gram模型参数 , 标量
    Returns:
        X_train - 测试集输入 , 一维lsit(train_size) , 每一位都是特征对应的字典
        y_train - 测试集标签 , 一维lsit(train_size) 
        X_valid - 测试集输入 , 一维lsit(valid_size) , 每一位都是特征对应的字典
        y_valid - 测试集标签 , 一维lsit(valid_size) 
    '''
    y_train = [r[1] for r in review[  : train_size]]
    y_valid = [r[1] for r in review[train_size : train_size + valid_size]]
    num_words_train = [len(r[0]) for r in review[ : train_size]]
    num_words_valid = [len(r[0]) for r in review[train_size : train_size + valid_size]]
    if (ngram == 1) :   
        X_train = [Counter(r[0]) for r in review[ : train_size]]
        #X_train = feature_extraction(X_train)
        #X_train = feature_TFIDF(X_train , num_words_train)
        X_valid = [Counter(r[0]) for r in review[train_size : train_size + valid_size]]
        # X_valid = feature_extraction(X_valid)
        #X_valid = feature_TFIDF(X_valid , num_words_valid)
    else :
        X_train = feature_ngram(review[ : train_size] , ngram)
        X_valid = feature_ngram(review[train_size : train_size + valid_size] , ngram)
    X_train = feature_extraction(X_train)
    X_valid = feature_extraction(X_valid)
        
    return X_train , y_train , X_valid , y_valid

def sparse_dotProduct(X , w) :
    '''
    稀疏特征的内积计算函数.
    Args:
        X - 样本的稀疏特征表示 , 字典
        w - 模型参数的稀疏表示 , 字典
    Returns:
        product - 内积结果 , 标量
    '''
    product = sum(w.get(k , 0) * v for k , v in X.items())
    return product

def update(w , X , scale) :
    '''
    参数的稀疏表示的更新函数. w[k] += X[k] * scale
    Args:
        w - 模型参数的稀疏表示 , 字典
        X - 用来更新的样本的稀疏特征表示 , 字典
        scale - 放缩比例(见上面公式), 标量
    Returns:
        None
    '''
    for k , v in X.items() :
        w[k] = w.get(k , 0) + scale * v

def compute_Gram(X) :
    '''
    计算并存储关于X的Gram矩阵. G[i][j] = sparse_dotProduct(X_i,X_j)
    Args:
        X - 样本的稀疏特征表示输入 , 一维list(num_instances) , 每一位都是特征对应的字典
    Returns:
        G - Gram矩阵 , 二维numpy数组(num_instances , num_instances)
    '''
    num_instances = len(X)
    G = np.zeros((num_instances , num_instances))
    for i in range(num_instances) :
        for j in range(num_instances) :
            G[i][j] = dotProduct(X[i] , X[j])
    return G

def pegasos(X_train , y_train , X_valid , y_valid , w , num_iter = 50 , lambda_reg = 0.01) :
    '''
    peagsos算法计算SVM的最优参数.实际上peagsos算法是应用在SVM损失函数上的次梯度随机下降法.
                次梯度选取 lambda_reg * w - y * X       若 y * <w,X> <  1
                         lambda_reg * w                  y * <w,X> >= 1
                学习率 eta = 1 / (lambda * t) , 递减的学习率
    最原始的peagsos算法效率很低,由于有正则化项,每次更新都需要遍历w,破坏了X稀疏性带来的好处.
    Args:
        X_train - 测试集输入 , 一维lsit(train_size) , 每一位都是特征对应的字典
        y_train - 测试集标签 , 一维lsit(train_size) 
        X_valid - 测试集输入 , 一维lsit(valid_size) , 每一位都是特征对应的字典
        y_valid - 测试集标签 , 一维lsit(valid_size) 
        w - 初始模型参数 , 字典
        num_iter - 最大迭代轮数 , 标量
        lambda_reg - 正则化系数 , 标量
    Returns:
        w - 最优参数
    '''
    num_instances = len(X_train)
    t = 0
    for d in range(num_iter) :
        for i in range(num_instances) :
            t += 1
            eta = 1 / (lambda_reg * t)
            if (y_train[i] * sparse_dotProduct(X_train[i] , w) < 1) :
                update(w , w , - lambda_reg * eta)
                update(w , X_train[i] , eta * y_train[i])
            else :
                update(w , w , -lambda_reg * eta)
    loss = sum(max(1 - y_train[i] * sparse_dotProduct(X_train[i] , w) , 0) for i in range(num_instances)) / num_instances
    return w

def pegasos_tricks(X_train , y_train , X_valid , y_valid , w , num_iter = 10 , lambda_reg = 0.01) :
    '''
    优化的peagsos算法,使用了Leon Bottou’s Stochastic Gradient Tricks.具体地,令w[t] = S[t] * W[t].
                        更新时 , S[t + 1] = S[t] * (1 - lambda_reg * eta) 
                                W[t + 1] = W[t] + (eta * y / S[t + 1]) * X   若 y * <w,X> <  1
                        最后结果为   S*W
    利用这一技巧,我们可以发现只有在模型分类错误的时候,才会对词典进行更新,并且更新的参数个数等于X的特征个数.
    Args:
        X_train - 测试集输入 , 一维lsit(train_size) , 每一位都是特征对应的字典
        y_train - 测试集标签 , 一维lsit(train_size) 
        X_valid - 测试集输入 , 一维lsit(valid_size) , 每一位都是特征对应的字典
        y_valid - 测试集标签 , 一维lsit(valid_size) 
        w - 初始模型参数 , 字典
        num_iter - 最大迭代轮数 , 标量
        lambda_reg - 正则化系数 , 标量
    Returns:
        w - 最优参数
    '''
    num_instances = len(X_train)
    t = 2
    scale = 1
    for d in range(num_iter) :
        for i in range(num_instances) :
            t += 1
            eta = 1 / (lambda_reg * t)
            scale = (1 - eta * lambda_reg) * scale
            if (y_train[i] * scale * sparse_dotProduct(X_train[i] , w) < 1) :
                update(w , X_train[i] , eta * y_train[i] / scale)
#     loss = sum(max(1 - y_train[i] * sparse_dotProduct(X_train[i] , w) * scale , 0) for i in range(num_instances)) / num_instances
    for k , v in w.items() :
        w[k] = v * scale
    return w

def SMO(X_train , y_train ,X_valid , y_valid , G , num_iter = 50 , lambda_reg = 0.007) :
    '''
    利用SMO(sequential minimal optimization)求解SVM的对偶问题.SVM对偶问题是n个变量带约束的二次规划问题,基本思路是每次选取出两个变量alpha_i,alpha_j,固定其他变量,将问题转换
    为二元二次函数的最优化问题,利用KKT条件消掉其中一个变量(如alpha[j]),将问题化为关于alpha[i]的一元二次最优化问题.注意最后的结果要满足KKT条件的限制,因此要做截断.
    Args:
        X_train - 测试集输入 , 一维lsit(train_size) , 每一位都是特征对应的字典
        y_train - 测试集标签 , 一维lsit(train_size) 
        X_valid - 测试集输入 , 一维lsit(valid_size) , 每一位都是特征对应的字典
        y_valid - 测试集标签 , 一维lsit(valid_size) 
        G - Gram矩阵 , 二维numpy数组(num_instances , num_instances)
        num_iter - 最大迭代轮数 , 标量
        lambda_reg - 正则化系数 , 标量
    Returns:
        w - 模型参数 , 字典
        alpha - 关于margin约束的Lagrange系数 , 一维numpy数组(num_instances)
    '''
    C = 1 / lambda_reg
    num_instances = len(X_train)
    alpha = np.zeros(num_instances)
    for d in range(num_iter) :
        for i in range(num_instances) :
            j = 0
            while True :
                j = np.random.randint(num_instances) 
                if (i != j) :
                    break
            B = 1 - y_train[i] * y_train[j]
            D = 0
            A = 2 * G[i][j] - G[i][i] - G[j][j]
            if (A == 0) :
                continue
            for k in range(num_instances) :
                if (k == i or k == j) :
                    continue
                D += alpha[k] * y_train[k]
                B += y_train[i] * y_train[k] * alpha[k] * (G[j][k] + G[i][j] - G[j][j] - G[i][k])
            alpha[i] = -1 * B / A
            if (y_train[i] * y_train[j] > 0) :
                if (alpha[i] < max(0 , -1 * C / num_instances - D * y_train[j])) :
                    alpha[i] = max(0 , -1 * C / num_instances - D * y_train[j])
                elif (alpha[i] > min(-D * y_train[j] , C / num_instances)) :
                    alpha[i] = min(-D * y_train[j] , C / num_instances)
            if (y_train[i] * y_train[j] < 0) :
                if (alpha[i] < max(0 , D * y_train[j])) :
                    alpha[i] = max(0 , D * y_train[j])
                elif (alpha[i] > min(C / num_instances , C / num_instances + D * y_train[j])) :
                    alpha[i] = min(C / num_instances , C / num_instances + D * y_train[j])
            alpha[j] = -1 * D * y_train[j] - alpha[i] * y_train[i] * y_train[j]
                
    w = {}
    for i in range(num_instances) :
        update(w , X_train[i] , alpha[i] * y_train[i])
    return w , alpha

def calc_accuracy(X , y , w) :
    '''
    计算模型的准确率 
    Args:
        X - 待计算集合特征输入 , 一维lsit(num_instances) , 每一位都是特征对应的字典
        y - 待计算集合标签输出 , 一维lsit(num_instances)
        w - 模型参数 , 字典
    Returns:
        acc - 模型准确率
    '''
    num_correct = 0
    for i in range(len(y)) :
        if (y[i] * sparse_dotProduct(X[i] , w) > 0) :
            num_correct += 1
    return num_correct / len(X)

def Experiment1(X_train , y_train , X_valid , y_valid) :
    '''
    实验一,最优正则化系数lambda_reg的parameter tuning. 先选取大区间大步长,绘制lambda - acc曲线,缩小范围减小步长,得到best_lambda.
    Args:
        X_train - 测试集输入 , 一维lsit(train_size) , 每一位都是特征对应的字典
        y_train - 测试集标签 , 一维lsit(train_size) 
        X_valid - 测试集输入 , 一维lsit(valid_size) , 每一位都是特征对应的字典
        y_valid - 测试集标签 , 一维lsit(valid_size) 
    Returns:
        best_lambda - 模型最优参数 , 字典
    '''
    lambda_start = 0.006
    lambda_end = 0.008
    lambda_delta = 0.0001
    lambda_list = []
    accuracy_list = []
    for l in np.arange(0.006 , 0.008 , 0.0001) :
        lambda_list.append(l)
    for lambda_reg in lambda_list :
        w = {}
        w = pegasos_tricks(X_train , y_train , X_valid , y_valid , w , num_iter = 50 , lambda_reg = lambda_reg)
        acc = calc_accuracy(X_valid , y_valid , w)
        accuracy_list.append(acc)
    best_lambda = 0.
    max_acc = 0
    for i in range(len(lambda_list)) :
        if (accuracy_list[i] > max_acc) :
            max_acc = accuracy_list[i]
            best_lambda = lambda_list[i]
    return best_lambda

def Experiment2(X_train , y_train , X_valid , y_valid , review , lambda_reg = 0.007) :
    '''
    实验二,针对分类错误的评论文本,分析可能的出错原因.
    有以下原因:1.中性词和描述剧情语句过多.
             2.评论时较为委婉,如:希望下一部作品能够更棒. better / rewarding这类褒义词语,对模型进行了误导.
    解决方法:词袋模型无法解决具有环境语义的特征抽取建模,需要用RNN/transformer等模型.
    Args:
        X_train - 测试集输入 , 一维lsit(train_size) , 每一位都是特征对应的字典
        y_train - 测试集标签 , 一维lsit(train_size) 
        X_valid - 测试集输入 , 一维lsit(valid_size) , 每一位都是特征对应的字典
        y_valid - 测试集标签 , 一维lsit(valid_size) 
        review - 评论文本 , 二维list(num_reviews , num_words)
        lambda_reg - 正则化系数 , 标量
    Returns:
        None
    '''
    w = {}
    w = pegasos_tricks(X_train , y_train , X_valid , y_valid , w , num_iter = 50 , lambda_reg = lambda_reg)
    num_error = 0
    pos = 0
    max_score = 0 
    for i in range(len(X_valid)) :
        if (y_valid[i] * sparse_dotProduct(X_valid[i] , w) < 0) :
        #print (y_valid[i] , sparse_dotProduct(X_valid[i] , w))
            num_error += 1
            if (y_valid[i] == -1 and abs(sparse_dotProduct(X_valid[i] , w)) > max_score) :
                max_score = sparse_dotProduct(X_valid[i] , w)
                pos = i
#     print (max_score)
#     print (review[pos + 1500])
#     for k , v in X_valid[pos].items() :
#         print (k , abs(v * w.get(k , 0)))

def Experiment3(X_train , y_train , X_valid , y_valid , lambda_reg = 0.007) :
    '''
    实验三,讨论落在margin上的样本点个数.  y * <w,X> = 1
    实验结果,当放大误差允许范围时,向量较多,但tolerance = 1e-3时, 点数=0 .一方面,说明参数未达到最优,最大margin还有进一步优化的可能.
    另一方面,随机次梯度下降法相比求解SVM对偶问题的SMO算法,是直接做凸优化,有效信息损失较多,比如丢掉了w应该是支持向量x的线性组合这些特性.
    Args:
        X_train - 测试集输入 , 一维lsit(train_size) , 每一位都是特征对应的字典
        y_train - 测试集标签 , 一维lsit(train_size) 
        X_valid - 测试集输入 , 一维lsit(valid_size) , 每一位都是特征对应的字典
        y_valid - 测试集标签 , 一维lsit(valid_size) 
        lambda_reg - 正则化系数 , 标量
    Returns:
        num_sv
    '''
    w = {}
    w = pegasos_tricks(X_train , y_train , X_valid , y_valid , w , num_iter = 50 , lambda_reg = lambda_reg)
    tolerance = 1e-2
    num_sv = 0
    for i in range(len(X_train)) :
        if (abs(y_train[i] * sparse_dotProduct(X_train[i] , w) - 1) < tolerance) :
            num_sv += 1
    print (num_sv)

def Experiment4(X_train , y_train , X_valid , y_valid , lambda_reg = 0.007) :
    '''
    实验四,讨论了褒义词和贬义词对应的模型参数的符号以及大小.结果表明SVM模型参数的确学到了影评评论的一些偏好行词语的词性.
    Args:
        X_train - 测试集输入 , 一维lsit(train_size) , 每一位都是特征对应的字典
        y_train - 测试集标签 , 一维lsit(train_size) 
        X_valid - 测试集输入 , 一维lsit(valid_size) , 每一位都是特征对应的字典
        y_valid - 测试集标签 , 一维lsit(valid_size) 
        lambda_reg - 正则化系数 , 标量
    Returns:
        None
    '''
    w = {}
    w = pegasos_tricks(X_train , y_train , X_valid , y_valid , w , num_iter = 50 , lambda_reg = lambda_reg)
    print (w['boring'])
    print (w['great'])

def Experiment5(X_train , y_train , X_valid , y_valid , lambda_reg = 0.007) :
    '''
    实验五,测试SMO算法的效率和准确率.
    Args:
        X_train - 测试集输入 , 一维lsit(train_size) , 每一位都是特征对应的字典
        y_train - 测试集标签 , 一维lsit(train_size) 
        X_valid - 测试集输入 , 一维lsit(valid_size) , 每一位都是特征对应的字典
        y_valid - 测试集标签 , 一维lsit(valid_size) 
        lambda_reg - 正则化系数 , 标量
    Returns:
        None
    '''
    G = compute_Gram(X_train)
    start_time = time.time()
    w , alpha = SMO(X_train , y_train , X_valid , y_valid , G)
    end_time = time.time()
    print (end_time - start_time)
    calc_accuracy(X_valid , y_valid , w)

def main() :
    '''
    主函数,对电影评论文本数据进行了读取,特征抽取,并训练模型分别进行了实验一到实验四,加强了对SVM的理解和认识.
    '''
    print ('loading dataset...')
    pos_path = 'data/pos'
    neg_path = 'data/neg'
    pos_review = folder_list(pos_path , 1)
    neg_review = folder_list(neg_path , -1)
    review = pos_review + neg_review
    random.seed(2020)
    random.shuffle(review)
    print ('feature extraction...')
    X_train , y_train , X_valid , y_valid  = split_review(review , ngram = 2)
    print ('build SVM model...')
    start_time = time.time()
    w = {}
    w = pegasos_tricks(X_train , y_train , X_valid , y_valid , w , num_iter = 50 , lambda_reg = 0.007)
    end_time = time.time()
    print ("times=" , end_time - start_time)

    print("%.9f" % calc_accuracy(X_valid , y_valid , w))
    print("%.9f" % calc_accuracy(X_train , y_train , w))

    best_lambda = Experiment1(X_train , y_train , X_valid , y_valid)
    Experiment2(X_train , y_train , X_valid , y_valid , review , best_lambda)
    Experiment3(X_train , y_train , X_valid , y_valid , best_lambda)
    Experiment4(X_train , y_train , X_valid , y_valid , best_lambda)
    Experiment5(X_train , y_train , X_valid , y_valid)

if __name__ == "__main__" :
    main()
