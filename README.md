# NYU-DS1003
纽约大学数据科学1003课程作业及思考


## 作业1
### 

## 作业2 Lasso Regression
对于人工生成的随机数据,使用Ridge回归,Lasso回归对数据进行拟合.观察L1正则和L2正则的具体效果并分析原因.

(1)随机生成数据,满足y=X\*theta+epsilon,其中epsilon~N(0,0.1)为噪声.

(2)实现坐标梯度下降法(Coordinate_Gradient_Descent)求解Lasso回归的参数.坐标梯度下降更新参数w的分段函数表明,L1正则更容易使得参数为0.

(3)实现投影梯度下降法(Projected_Gradient_Descent)求解Lasso回归的参数.

(4)讨论了Lasso回归超参数L1正则系数lambda的调参技巧.如快速确定lambda_max,在\[0,lambda_max\]里进行选择lambda.注意到目标函数J(w)关于lambda是lipschitz连续的,因此当两个lambda接近的时候,对应的最优解w也充分接近.所以每次更换lambda的时候,w不需要从0更新,而可以选择前一个lambda对应的最优解作为初值.

(5)依次进行实验一～实验三,完成了(2)~(4)中的内容,加深了对优化算法以及Ridge回归和Lasso回归的认识.

## 作业3 SVM
利用SVM对2000条电影评论进行训练及预测,最优模型在样本数为500的验证集上取得了82.2%的准确率效果.

(1)字典存储稀疏特征以及模型参数

(2)特征抽取以及特征工程:TFIDF特征,N-gram模型特征,褒义词/贬义词词性分析

(3)Peagsos算法训练SVM模型参数

(4)利用稀疏性分解w=S\*W,加速模型训练

(5)实现稀疏字典化内积方法

(6)SMO算法训练SVM模型参数,得到支持向量更少的解.

(7)依次进行实验一～实验六,分析了预测错误的样本的原因以及解决方案,并对支持向量的概念进行了回顾和总结.
