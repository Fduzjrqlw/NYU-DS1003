import pandas as pd
import numpy as np
from sklearn import linear_model

def data_construction(num_instances = 150 , num_features = 75) :
    '''
    生成数据 , X随机生成.theta为真实参数,前十位均从{-10,10}中随机选取,其余位全为0. y=X*theta+epsilon,其中epsilon符合~N(0,0.1)
    Args:
        num_instances - 样本个数 , 标量
        num_features - 特征个数 , 标量
    Returns:
        X - 输入数据 , 二维numpy矩阵(num_instances , num_features)
        y - label , 一维numpy矩阵(num_instances)
        theta - 用来生成label的模型真实参数 , 一维numpy矩阵(num_features)
    '''
    X = np.random.rand(num_instances , num_features)
    theta = np.zeros(num_features)
    for i in range(10) :
        if (np.random.randint(0,2) % 2 == 0) :
            theta[i] = 10.0 
        else :
            theta[i] = -10.0
    mean = 0
    std = 0.1
    epsilon = np.random.normal(mean , std , num_instances)
    y = np.matmul(X , theta) + epsilon
    
    return X , y , theta

def data_split(X , y , num_instances = 150 , num_train = 80 , num_valid = 20 , num_test = 50) :
    '''
    分别按照前80,20,50依次划分训练集,验证集,测试集.
    Args:
        X - 输入数据 , 二维numpy矩阵(num_instances , num_features)
        y - 标签 , 一维numpy矩阵(num_instances)
        num_instances - 总样本个数 , 标量
        num_train - 训练样本个数 , 标量
        num_valid - 验证集样本个数 , 标量
        num_test - 训练集样本个数 , 标量
    Returns:
        X_train - 训练集输入 , 二维numpy矩阵(num_train , num_features)
        y_train - 训练集标签 , 一维numpy矩阵(num_train)
        X_valid - 验证集输入 , 二维numpy矩阵(num_valid , num_features)
        y_valid - 验证集标签 , 一维numpy矩阵(num_valid)
        X_test - 测试集输入 , 二维numpy矩阵(num_test , num_features)
        y_test - 测试集标签 , 一维numpy矩阵(num_test)
    '''
    X_train = X[ : num_train]
    y_train = y[ : num_train]
    X_valid = X[num_train : num_train + num_valid]
    y_valid = y[num_train : num_train + num_valid]
    X_test = X[-num_test : ]
    y_test = y[-num_test : ]
    return X_train , y_train , X_valid , y_valid , X_test , y_test

def feature_normlization(X_train , X_valid , X_test) :
    '''
    将每一维特征的数据利用线性变换[0,1]对训练数据和测试数据归一化,其中统计量来自于训练数据.
    Args:
        X_train - 训练集输入数据 , 二维numpy数组(num_train , num_features)
        X_valid - 验证集输入数据 , 二维numpy数组(num_valid , num_features)
        X_test - 测试集输入数据 , 二维numpy数组(num_test , num_features)
    
    Returns:
        X_train - 经过归一化的训练集 , 二维numpy数组(num_train , num_features)
        X_valid - 经过归一化的验证集 , 二维numpy数组(num_valid , num_features)
        X_test - 经过归一化的测试集 , 二维numpy数组(num_test , num_features)
    '''
    maximal = np.max(X_train , axis = 0)
    minimal = np.min(X_train , axis = 0)
    X_train = (X_train - minimal + 1e-8) / (maximal - minimal + 1e-8)
    X_valid = (X_valid - minimal + 1e-8) / (maximal - minimal + 1e-8)
    X_test = (X_test - minimal + 1e-8) / (maximal - minimal + 1e-8)
    return X_train , X_valid , X_test

def compute_regularized_square_loss(X , y , theta , lambda_reg = 1) :
    '''
    带正则的线性模型的平方损失函数  loss = |X * theta - y|^2 / m + lambda_reg * |theta|^2
    Args:
        X - 输入特征数据 , 二维numpy数组(num_instances , num_features)
        y - 标签label数据 , 一维numpy数组(num_instances)
        theta - 模型参数数据 , 一维numpy数组(num_features)
        lambda_reg - 正则化项系数 , 标量
    Returns:
        loss - 平方损失 , 标量
    '''
    y_predict = np.matmul(X , theta)
    square_loss = np.mean((y_predict - y) ** 2) / 2
    regularized_loss = lambda_reg * np.sum((theta)**2)
    loss = square_loss + regularized_loss
    return loss

def compute_regularized_square_loss_gradient(X ,  y , theta , lambda_reg = 1) :
    ''' 
    计算平方损失函数关于参数theta的梯度   grad = X^T * (X * theta - y) / m + 2 * lambda_reg *  theta
    Args:
        X - 输入特征数据 , 二维numpy数组(num_instances , num_features)
        y - 标签label数据 , 一维numpy数组(num_instances)
        theta - 模型参数数据 , 一维numpy数组(num_features)
        lambda_reg - 正则化系数 , 标量
    Returns:
        grad - 梯度向量 , 一维numpy数组(num_features)
    '''
    num_instances = X.shape[0]
    y_predict = np.matmul(X , theta)
    grad = np.matmul(X.T , y_predict - y) / num_instances + 2 * lambda_reg * theta
    return grad

def batch_gradient_descent(X , y , alpha = 0.01 , num_iter = 20000 , lambda_reg = 1) :
    '''
    利用梯度下降法求解平方损失函数的线性模型的参数
    Args:
        X - 输入特征数据 , 二维numpy数组(num_instances , num_features)
        y - 标签label数据 , 一维numpy数组(num_instances)
        alpha - 学习率/梯度步长 , 标量
        num_iter - 最大迭代次数 , 标量
    Returns:
        theta_hist - 迭代过程中储存的参数列表 , 二维numpy数组(num_iter , num_features)
        loss_hist - 迭代过程中储存的loss列表,一维numpy数组(num_iter)
    '''
    [num_instances , num_features] = X.shape
    theta = np.ones(num_features)
    
    theta_hist = np.zeros((num_iter , num_features))
    loss_hist = np.zeros(num_iter)
    
    for i in range(num_iter) :
        grad = compute_regularized_square_loss_gradient(X , y , theta , lambda_reg)
        theta -= alpha * grad
        loss = compute_regularized_square_loss(X , y , theta , lambda_reg)
        theta_hist[i] = theta
        loss_hist[i] = loss
    return theta_hist , loss_hist

def compute_L1_square_loss(X , y , theta , lambda_reg = 1e-4) :
    '''
    带正则的线性模型的平方损失函数  loss = |X * theta - y|^2 / m + lambda_reg * |theta|
    Args:
        X - 输入特征数据 , 二维numpy数组(num_instances , num_features)
        y - 标签label数据 , 一维numpy数组(num_instances)
        theta - 模型参数数据 , 一维numpy数组(num_features)
        lambda_reg - 正则化项系数 , 标量
    Returns:
        loss - 平方损失 , 标量
    '''
    y_predict = np.matmul(X , theta)
    square_loss = np.mean((y_predict - y)**2) / 2 
    l1_loss = lambda_reg * np.sum(np.abs(theta))
    loss = square_loss + l1_loss
    return loss

def sign(x) :
    '''
    符号函数
    Args:
        x - 待求变量 , 标量
    Returns :
        sgn - 符号 , {-1,0,1}
    '''
    if (x > 0) :
        return 1
    if (x < 0) :
        return -1 
    return 0

def piecewise_function(a , c , lambda_reg) :
    '''
    用于梯度坐标下降中的参数更新的分段函数  f(a,c,lambda_reg) = sgn(c / a) * max(0 , |c / a| - lambda_reg / a)
    Args:
        a , b , c - 输入变量 , 标量
    Returns:
        f - 分段函数值 , 标量
    
    '''
    if (c > lambda_reg) :
        return (c - lambda_reg) / a
    if (c < -lambda_reg) :
        return (c + lambda_reg) / a
    return 0

def Coordinate_Descent(X , y , theta , num_iter = 2000 ,  lambda_reg = 0.015) :
    '''
    梯度坐标下降法求解Lasso回归问题. 每一轮迭代中,依次选取坐标进行更新,即考虑argmin_{w_j} Loss(w1,w2,...wj,..wd),注意到有闭式解,
    写出来是一个经典的分段二次函数(L1正则的绝对值导致的现象),因此闭式解是一个三段的分段函数.这个分段函数也解释了为什么L1正则使得参数更容易
    变为0.
    Args:
        X - 输入特征数据 , 二维numpy数组(num_instances , num_features)
        y - 标签label数据 , 一维numpy数组(num_instances)
        theta - 模型参数数据 , 一维numpy数组(num_features)
        num_iter - 最大迭代次数 , 标量
        lambda_reg - 正则化项系数 , 标量
    Returns:
        theta_hist - 迭代过程中储存的参数列表 , 二维numpy数组(num_iter , num_features)
        loss_hist - 迭代过程中储存的loss列表,一维numpy数组(num_iter)
    '''
    [num_instances , num_features] = X.shape
    theta_hist = np.zeros((num_iter , num_features))
    loss_hist = np.zeros(num_iter)
    A = np.zeros(num_features)
    C = np.zeros(num_features)
    for step in range(num_iter) :
        for j in range(num_features) :
            A[j] = np.mean((X.T[j]) ** 2)
            C[j] = np.mean((y * X.T[j] + theta[j] * X.T[j] * X.T[j])) - np.mean((np.matmul(X , theta) * X.T[j]))
            theta[j] = piecewise_function(A[j]  , C[j] , lambda_reg)
        theta_hist[step] = theta
        loss_hist[step] = compute_L1_square_loss(X , y , theta , lambda_reg)
    return theta_hist , loss_hist

def projected_gradient_descent(X , y , theta1 , theta2 , alpha = 0.015 ,  num_iter = 10000 , lambda_reg = 0.015) :
    '''
    用投影梯度下降法解决Lasso回归问题.将参数theta进行分解为正部theta1和负部theta2,将原来无约束不可微的凸优化问题转换成了
    带约束的可微的凸优化问题.
    Args:
    X - 输入特征数据 , 二维numpy数组(num_instances , num_features)
    y - 标签label数据 , 一维numpy数组(num_instances)
    theta1 - 模型参数正部 , 一维numpy数组(num_features)
    theta2 - 模型参数负部 , 一维numpy数组(num_features)    theta = theta1 - theta2
    alpha - 学习率 , 标量
    num_iter - 最大迭代次数 , 标量
    lambda_reg - 正则化项系数 , 标量
    Returns:
    theta_hist - 迭代过程中储存的参数列表 , 二维numpy数组(num_iter , num_features)
    loss_hist - 迭代过程中储存的loss列表,一维numpy数组(num_iter)
    '''
    [num_instances , num_features] = X.shape
    
    theta_hist = np.zeros((num_iter , num_features))
    loss_hist = np.zeros(num_iter)
    e = np.ones(num_features)
    
    for i in range(num_iter) :
        grad_theta1 = np.matmul(X.T , np.matmul(X , theta1 - theta2) - y) / num_instances + lambda_reg * e
        grad_theta2 = np.matmul(X.T , np.matmul(X , theta2 - theta1) + y) / num_instances + lambda_reg * e
        
        theta1 -= alpha * grad_theta1
        theta2 -= alpha * grad_theta2
        
        theta1 = np.maximum(0 , theta1)
        theta2 = np.maximum(0 , theta2)
        
        theta_hist[i] = theta1 - theta2
        loss_hist[i] = compute_L1_square_loss(X , y , theta1 - theta2 , lambda_reg)
        
    return theta_hist , loss_hist


def Experiment1(X_train , y_train , X_valid , y_valid):
    '''
    实验1 Ridge regression,注意没有bias term.比较了手写的梯度下降法和sklearn包中的SMO求解的效率和精度的差距.
    sklearn包的Ridge的效果更好,速度更快.
    '''
    lambda_reg = 1e-4
    theta_hist , loss_hist = batch_gradient_descent(X_train , y_train , lambda_reg = lambda_reg)
    model = linear_model.Ridge(alpha = lambda_reg , fit_intercept = False)
    model.fit(X_train , y_train)
    #print (model.coef_)
    #print (compute_regularized_square_loss(X_valid , y_valid , model.coef_ , lambda_reg))
    y_predict = model.predict(X_valid)
    #print (np.mean((y_predict - y_valid)**2)/2 + lambda_reg * np.sum((model.coef_)**2))
    y_predict = model.predict(X_train)
    #print (np.mean((y_predict - y_train)**2)/2 + lambda_reg * np.sum((model.coef_)**2))
    return model

def Experiment2(X_train , y_train , X_valid , y_valid) :
    '''
    实验2验证了坐标梯度下降法来求解Lasso回归,选取不同大小的正则化系数lambda,观察系数的变化
    Lasso回归调lambda超参数的两个技巧
        1.当lambda充分大的时候,所有的w都会被压成0,因此lambda的范围[0,lambda_max]可以快速定下来
        2.一个观察就是类似lipschitz连续,lambda_1和lambda_2很接近,则对应的最优解w_1和w_2也会很接近.所以当调节lambda的时候,
          起点不需要从w=0开始计算,而是可以从上一个lambda对应的最优解出发.
    '''
    #theta = model.coef_
    theta = np.ones(X_train.shape[1])
    theta_hist , loss_hist = Coordinate_Descent(X_train , y_train , theta)
    print (theta_hist[-1])
    print (loss_hist[-1])

def Experiment3(X_train , y_train , X_valid , y_valid) :
    '''
    实验3验证了投影梯度下降法来求解Lasso回归,选取不同大小的正则化系数lambda,观察系数的变化
    '''
    theta1 = np.ones(X_train.shape[1])
    theta2 = np.zeros(X_train.shape[1])
    theta_hist , loss_hist = projected_gradient_descent(X_train , y_train , theta1 , theta2)
    print (theta_hist[-1])
    print (loss_hist[-1])

def main() :
    X , y , true_theta = data_construction()
    X_train , y_train , X_valid ,  y_valid , X_test , y_test = data_split(X , y)
    X_train , X_valid , X_test = feature_normlization(X_train , X_valid , X_test)
    model = Experiment1(X_train , y_train , X_valid , y_valid)
    Experiment2(X_train , y_train , X_valid , y_valid)
    Experiment3(X_train , y_train , X_valid , y_valid)
    



if __name__ == "__main__":
    main()