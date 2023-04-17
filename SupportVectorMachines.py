import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from imblearn import over_sampling as os


# 从CSV文件读取数据
data = pd.read_csv('D:\我的文件\learning\sem2\COMP7103 Data mining\\assignment 1\A1\mushroom\mushrooms.csv')

#data['Amount']=StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))#对amount进行标准化

# 将字符串类型的列转换为数字编码
labelencoder=LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':  # 判断是否为字符串类型
        # 使用Pandas的factorize()方法将字符串编码为数字
        data[col] = labelencoder.fit_transform(data[col])

# 分离特征和标签
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 标准化，-1到1之间
scaler = MinMaxScaler()
X=scaler.fit_transform(X)
print(X)

# 尝试不同PCA组件数量
up_pac=2#PAC上限
low_pac=2
#up_tree=20#树上限
#low_tree=10


# 记录精度结果
#rf_accuracies = np.zeros((3,3))#手动改！！
#rf_accuracies_b = np.zeros((3,3))
svm_accuracies = []
#svm_accuracies_b = []

#i=0
for n_components in range(low_pac, up_pac+1,5):
    # 使用PCA降维
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    #X_test_b=pca.fit_transform(X_test_bb)
    #j=0

    # 划分训练集和测试集
    X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.3)
        
    svm = SVC(kernel='poly', C=1.0)
    '''
    kernel ：核函数，默认是rbf，可以是‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    　　0 – 线性：u’v
    　　1 – 多项式：(gamma*u’*v + coef0)^degree
    　　2 – RBF函数：exp(-gamma|u-v|^2)
    　　3 –sigmoid：tanh(gamma*u’*v + coef0)
    degree ：多项式poly函数的维度，默认是3，选择其他核函数时会被忽略。
    gamma ： ‘rbf’,‘poly’ 和‘sigmoid’的核函数参数。默认是’auto’，则会选择1/n_features
    coef0 ：核函数的常数项。对于‘poly’和 ‘sigmoid’有用。
    probability ：是否采用概率估计？.默认为False
    shrinking ：是否采用shrinking heuristic方法，默认为true
    tol ：停止训练的误差值大小，默认为1e-3
    cache_size ：核函数cache缓存大小，默认为200
    class_weight ：类别的权重，字典形式传递。设置第几类的参数C为weight*C(C-SVC中的C)
    verbose ：允许冗余输出？
    max_iter ：最大迭代次数。-1为无限制。
    decision_function_shape ：‘ovo’, ‘ovr’ or None, default=None3
    random_state ：数据洗牌时的种子值，int值
    ————————————————
    '''

    '''    
    # 训练随机森林分类器并计算精度
    rf.fit(X_train_pca, y_train)
    rf_y_pred = rf.predict(X_test_pca)
    rf_y_pred_b = rf.predict(X_test_b)
    rf_acc = accuracy_score(y_test, rf_y_pred)
    rf_acc_b = accuracy_score(y_test_b, rf_y_pred_b)
    rf_accuracies[i][j]=rf_acc
    rf_accuracies_b[i][j]=rf_acc_b
    '''

    # 训练支持向量机分类器并计算精度
    svm.fit(X_train_pca, y_train)
    svm_y_pred = svm.predict(X_test_pca)
    #svm_y_pred_b = svm.predict(X_test_b)
    svm_acc = accuracy_score(y_test, svm_y_pred)
    #svm_acc_b = accuracy_score(y_test_b, svm_y_pred_b)
    svm_accuracies.append(svm_acc)
    #svm_accuracies_b.append(svm_acc_b)
    
    #j+=1
    #i+=1
    
# 输出结果
fp = open("D:\我的文件\learning\sem2\COMP7103 Data mining\\assignment 1\A1\mushroom\svm_output_.txt", "a+")  # a+ 如果文件不存在就创建。存在就在文件内容的后面继续追加
print(f"SVM Accuracy:\n{svm_accuracies}",file=fp)
fp.close()

# 将结果保存为图像文件

# 生成网格数据用于绘制分类边界
x_min, x_max = min(X_pca[:, 0]) - 1, max(X_pca[:, 0]) + 1
y_min, y_max = min(X_pca[:, 1]) - 1, max(X_pca[:, 1]) + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])

# 将分类结果用颜色填充网格
Z = Z.reshape(xx.shape)
#cmap = plt.cm.get_cmap('RdYlBu')
plt.contourf(xx, yy, Z, cmap='RdYlBu', alpha=0.8)

# 将训练集和测试集用散点图表示出来
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='RdYlBu', edgecolors='k')
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap='RdYlBu', edgecolors='k', alpha=0.6)
#cmap=plt.cm.RdYlBu  ???

# 设置图形的标题和坐标轴标签
plt.title('SVM Classification')
plt.xlabel('Component 1')
plt.ylabel('Component 2')

# 保存图形为一张图片
plt.savefig('D:\我的文件\learning\sem2\COMP7103 Data mining\\assignment 1\A1\mushroom\svm_classification.png')


'''
# 画热图
sns.heatmap(rf_accuracies, cmap='coolwarm')

# 添加标题和轴标签
plt.title('Accuracy for test')
plt.xlabel('Number of PCA Components')
plt.ylabel('Number of RandomForest Estimators')

plt.savefig('D:\我的文件\learning\sem2\COMP7103 Data mining\\assignment 1\data\CreditCard\\Result.png')

# 画热图
sns.heatmap(rf_accuracies_b, cmap='coolwarm')

# 添加标题和轴标签
plt.clf()#重置
plt.title('Accuracy for test_b')
plt.xlabel('Number of PCA Components')
plt.ylabel('Number of RandomForest Estimators')

plt.savefig('D:\我的文件\learning\sem2\COMP7103 Data mining\\assignment 1\data\CreditCard\\Result_b.png')
'''