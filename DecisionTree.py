import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import entropy
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
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



'''
for col in data.columns:
    # 计算每个类别的出现频率
    class_freq = data[col].value_counts(normalize=True)

    # 计算每个类别的信息熵
    class_entropy = entropy(class_freq, base=2)

    # 输出每个类别的信息熵
    print(f"{col}变量的信息熵：{class_entropy}")
'''

# 分离特征和标签
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 标准化，-1到1之间
scaler = MinMaxScaler()
X=scaler.fit_transform(X)
print(X)

# 使用SMOTE算法进行过采样
#smote = os.SMOTE(random_state=42)
#X_resampled, y_resampled = smote.fit_resample(X, y)


# 划分训练和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y)#随机划分，容量约5.7K random_state=42


# 构建决策树
dtc = DecisionTreeClassifier(criterion="entropy", max_depth=4, min_samples_split=200,min_samples_leaf=100,random_state=2)
dtc.fit(X_train, y_train)

"""
DecisionTreeClassifier参数解释：
criterion：表示选择划分特征的标准，可选的值为"gini"或"entropy"，默认为"gini"。这个参数的意思是使用基尼不纯度或信息熵作为划分标准，这两种标准都是用来度量数据的纯度或不确定性。

splitter：表示决策树在每个节点的划分策略，可选的值为"best"或"random"，默认为"best"。这个参数的意思是使用最佳划分或随机划分。

max_depth：表示决策树的最大深度，即决策树的层数。默认为None，表示不限制决策树的深度。如果设置了这个参数，决策树将在达到最大深度之后停止分裂。

min_samples_split：表示节点分裂的最小样本数，如果某个节点的样本数小于这个值，则不会进行分裂。默认为2。

min_samples_leaf：表示叶子节点的最小样本数，如果某个叶子节点的样本数小于这个值，则该叶子节点将被剪枝。默认为1。

max_features：表示每个节点在分裂时最多考虑的特征数，可选的值为"auto"、"sqrt"、"log2"或None。默认为None，表示考虑所有特征。

random_state：表示随机数生成器的种子，用于控制随机性。设置这个参数可以让每次运行得到相同的结果。
"""

# 可视化决策树

plt.figure(figsize=(18,6),dpi=150) 

plot_tree(dtc,fontsize=12, filled = True)

plt.savefig("D:\我的文件\learning\sem2\COMP7103 Data mining\\assignment 1\A1\mushroom\decision_tree_depth=5.png")

# 预测并计算准确率
'''
y_pred_train = dtc.predict(X_train)
y_pred_test = dtc.predict(X_test)

acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)#自动划分测试集
'''
acc_train =dtc.score(X_train,y_train)
acc_test=dtc.score(X_test,y_test)

print("准确率（训练集）：", acc_train)
print("准确率（随机测试集）：", acc_test)


'''
data_test=pd.read_csv('D:\我的文件\learning\sem2\COMP7103 Data mining\\assignment 1\data\CreditCard\TestData.csv')#手动划分的测试集
data_test['Amount']=StandardScaler().fit_transform(data_test['Amount'].values.reshape(-1,1))

X_test_b=data_test.iloc[:, :-1]#手动划分的测试集，包含400余条盗刷+400余正常数据
y_test_b = data_test.iloc[:, -1]

y_pred_test_b = dtc.predict(X_test_b)
acc_test_b = accuracy_score(y_test_b, y_pred_test_b)#手动划分测试集
print("准确率（指定测试集）：", acc_test_b)
'''