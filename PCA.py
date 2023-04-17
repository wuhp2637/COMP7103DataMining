import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

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

# 主成分分析
pca = PCA()
pca.fit_transform(X)
covariance=pca.get_covariance()
explained_variance=pca.explained_variance_
ratio=pca.explained_variance_ratio_

# 可视化直方图
plt.figure(figsize=(6,4),dpi=150) 
plt.bar(range(22), explained_variance, alpha=0.5, align='center')
plt.ylabel('Explained variance')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig("D:\我的文件\learning\sem2\COMP7103 Data mining\\assignment 1\A1\mushroom\PCA Explained variance ratio.png")

# 可视化 解释方差比例之和
plt.clf()
plt.plot([i for i in range(X.shape[1])],[sum(ratio[:i+1]) for i in range(X.shape[1])])
plt.ylabel('Accumulated explained variance ratio')
plt.xlabel('Principal components')
plt.savefig("D:\我的文件\learning\sem2\COMP7103 Data mining\\assignment 1\A1\mushroom\PCA Accumulated explained variance ratio.png")

# 选择需要绘制直方图的变量
for i in range(data.shape[1]-1):
    # 绘制直方图
    plt.clf()
    plt.hist(X[:,i])
    plt.title('Distribution of '+ data.columns[i])
    plt.xlabel('values')
    plt.ylabel('Frequency')
    plt.savefig("D:\我的文件\learning\sem2\COMP7103 Data mining\\assignment 1\A1\mushroom\\" + str(i) + "_hist.png")
    plt.show()