
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import recall_score
import numpy as np
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd




from sklearn.preprocessing import StandardScaler

# load CSV file using pandas
data = pd.read_csv('C:\\PycharmProjects\\cs2\\Data.csv', sep=',', header=0)

data = data[data.iloc[:, -1] <= 2]

# separate inputs and labels
inputs = data.iloc[:, 1:-1]
labels = data.iloc[:, -1]

# perform PCA to determine optimal number of components
pca = PCA().fit(inputs)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.grid('True')
plt.show()
# Perform row bias removal
reduced_inputs = pca.fit_transform(inputs)
mean_values = reduced_inputs.mean(axis=0)
bias_removed_inputs = reduced_inputs - mean_values

# Plot the first feature after row bias removal
plt.plot(bias_removed_inputs[:, 0])
plt.xlabel('Samples')
plt.ylabel('Feature 1 (after bias removal)')
plt.title('Row Bias Removal')
plt.show()

# visualize explained variance ratio of each feature
plt.bar(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_)
plt.xlabel('PCA feature')
plt.ylabel('Explained Variance Ratio')
plt.show()

num_components = 8
pca = PCA(n_components=num_components)
reduced_inputs = pca.fit_transform(inputs)
input_feature = reduced_inputs[:, 0]


# X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.1, random_state=999999)
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(bias_removed_inputs, labels, test_size=0.1, random_state=999999)

# 训练分类器
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# 统计每个类别的样本个数
class_counts = pd.Series(y_train).value_counts()

# 绘制柱状图
plt.bar(class_counts.index, class_counts.values)
plt.xlabel('Label')
plt.ylabel('Count')
plt.title('Label Counts')
plt.show()

# 预测测试集的类别
y_pred = classifier.predict(X_test)

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)

# 绘制混淆矩阵图
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(class_counts))
plt.xticks(tick_marks, class_counts.index)
plt.yticks(tick_marks, class_counts.index)
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.show()

#  Logistic Regression逻辑回归分类器 求解器是拟牛顿法 最大迭代次数
classifier1 = LogisticRegression(solver='lbfgs', max_iter=2000)
classifier1.fit(X_train, y_train)
# 针对训练集数据进行预测和性能评估
y_train_pred = classifier1.predict(X_train)
train_accuracy = classifier1.score(X_train, y_train)
train_recall = recall_score(y_train, y_train_pred, average='weighted')
train_f1 = f1_score(y_train, y_train_pred,average='weighted')

print('Training Set - Accuracy:', train_accuracy)
print('Training Set - Recall:', train_recall)
print('Training Set - F1-score:', train_f1)
#测试集
y_pred = classifier1.predict(X_test)
# 测试正确率
accuracy = classifier1.score(X_test, y_test)
print('The accuracy of classifier1:', accuracy)
# 测试召回率
recall = recall_score(y_test, y_pred, average='binary')
print('The recall of classifier1:', recall)
# F-Score
f1 = f1_score(y_test, y_pred)

print("The F-score of classifier1:", f1)

# Random Forest随机森林分类器
classifier2 = RandomForestClassifier()
classifier2.fit(X_train, y_train)
# 针对训练集数据进行预测和性能评估
y_train_pred = classifier1.predict(X_train)
train_accuracy = classifier1.score(X_train, y_train)
train_recall = recall_score(y_train, y_train_pred, average='weighted')
train_f1 = f1_score(y_train, y_train_pred,average='weighted')

print('Training Set - Accuracy:', train_accuracy)
print('Training Set - Recall:', train_recall)
print('Training Set - F1-score:', train_f1)

y_pred = classifier2.predict(X_test)

accuracy = classifier2.score(X_test, y_test)
print('The accuracy of classifier2:', accuracy)
recall = recall_score(y_test, y_pred, average='binary')
print('The recall of classifier2:', recall)

f1 = f1_score(y_test, y_pred)
print("The F-score of classifier2:", f1)

# SVM支持向量机分类器
classifier3 = SVC()
classifier3.fit(X_train, y_train)
# 针对训练集数据进行预测和性能评估
y_train_pred = classifier1.predict(X_train)
train_accuracy = classifier1.score(X_train, y_train)
train_recall = recall_score(y_train, y_train_pred, average='weighted')
train_f1 = f1_score(y_train, y_train_pred,average='weighted')

print('Training Set - Accuracy:', train_accuracy)
print('Training Set - Recall:', train_recall)
print('Training Set - F1-score:', train_f1)

y_pred = classifier3.predict(X_test)

accuracy = classifier3.score(X_test, y_test)
print('The accuracy of classifier3:', accuracy)
recall = recall_score(y_test, y_pred, average='binary')
print('The recall of classifier3:', recall)

f1 = f1_score(y_test, y_pred)

print("The F-score of classifier3:", f1)


# #KMeans
# kmeans = KMeans(n_clusters=2, n_init=50, random_state=666)
# kmeans.fit(bias_removed_inputs)
# # evaluate performance of the clustering
# labels_pred = kmeans.predict(bias_removed_inputs)
# silhouette_avg = silhouette_score(bias_removed_inputs, labels_pred)
# print('Silhouette Score:', silhouette_avg)
#
# # 提取K-means的聚类结果和噪点
# labels_kmeans = kmeans.labels_
# noise_kmeans = bias_removed_inputs[labels_kmeans == -1]
# # 可视化K-means的聚类结果和噪点
# plt.scatter(bias_removed_inputs[:, 0], bias_removed_inputs[:, 1], c=labels_kmeans, cmap='viridis')
# plt.scatter(noise_kmeans[:, 0], noise_kmeans[:, 1], c='red', marker='x', label='Noise')
# plt.legend()
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.title('K-means Clustering with Noise')
# plt.show()
# #图像k和silhouette的关系
#
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
#
# k_values = range(2, 7)
# silhouette_scores = []
#
# for k in k_values:
#     kmeans = KMeans(n_clusters=k, random_state=666)
#     labels = kmeans.fit_predict(bias_removed_inputs)
#     score = silhouette_score(bias_removed_inputs, labels)
#     silhouette_scores.append(score)
#
# plt.plot(k_values, silhouette_scores, marker='o')
# plt.xlabel('Number of Clusters (k)')
# plt.ylabel('Silhouette Score')
# plt.title('Silhouette Score vs. Number of Clusters')
# plt.show()

# # 提取K-means的聚类结果和噪点
# labels_kmeans = kmeans.labels_
# noise_kmeans = bias_removed_inputs[labels_kmeans == -1]
# # 可视化K-means的聚类结果和噪点
# plt.scatter(bias_removed_inputs[:, 0], bias_removed_inputs[:, 1], c=labels_kmeans, cmap='viridis')
# plt.scatter(noise_kmeans[:, 0], noise_kmeans[:, 1], c='red', marker='x', label='Noise')
# plt.legend()
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.title('K-means Clustering with Noise')
# plt.show()


# Create a DBSCAN object
dbscan = DBSCAN(eps=1, min_samples=5)
# Fit the DBSCAN model to the reduced inputs
dbscan.fit(bias_removed_inputs)
# Evaluate performance of the clustering
labels_pred = dbscan.labels_
silhouette_avg = silhouette_score(bias_removed_inputs, labels_pred)
print('Silhouette Score:', silhouette_avg)
# 提取DBSCAN的聚类结果和噪点
labels_dbscan = dbscan.labels_
noise_dbscan = bias_removed_inputs[labels_dbscan == -1]
# 可视化DBSCAN的聚类结果和噪点
plt.scatter(bias_removed_inputs[:, 0], bias_removed_inputs[:, 1], c=labels_dbscan, cmap='viridis')
plt.scatter(noise_dbscan[:, 0], noise_dbscan[:, 1], c='red', marker='x', label='Noise')
plt.legend()
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('DBSCAN Clustering with Noise')
plt.show()
# 计算噪点数量
num_noise_points = len(noise_dbscan)
print('噪点数量:', num_noise_points)

