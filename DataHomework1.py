import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# 生成正常数据点
np.random.seed(42)
normal_data = np.random.randn(200, 2) * 1

# 生成一些异常点
outliers = np.random.uniform(low=-10, high=10, size=(10, 2))

# 合并数据集
data = np.vstack([normal_data, outliers])

# 可视化数据
plt.scatter(data[:, 0], data[:, 1], color='blue', label='Normal data')
plt.scatter(outliers[:, 0], outliers[:, 1], color='red', label='Outliers')
plt.legend()
plt.savefig('result1.png')
plt.show()

def local_outlier_factor(data, k=5):
    # 计算最近邻
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(data)
    distances, indices = nbrs.kneighbors(data)

    # 计算局部可达密度
    lrd = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        lrd[i] = 1 / (np.mean(distances[i, 1:]) + 1e-10)

    # 计算LOF
    lof = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        lof[i] = np.sum(lrd[indices[i, 1:]] / lrd[i]) / k

    return lof


# 计算LOF得分
lof_scores = local_outlier_factor(data, k=5)

# 设置阈值，确定异常点
threshold = 2
outliers = np.where(lof_scores > threshold)[0]

# 可视化结果
plt.figure(figsize=(10, 8))
plt.scatter(data[:, 0], data[:, 1], color='blue', label='Normal data')

# 绘制异常点和它们的LOF得分
for index in outliers:
    plt.scatter(data[index, 0], data[index, 1], color='red')
    plt.text(data[index, 0], data[index, 1], f'{lof_scores[index]:.2f}', color='red', fontsize=12, ha='right')
    circle = plt.Circle((data[index, 0], data[index, 1]), lof_scores[index], color='red', fill=False)
    plt.gca().add_patch(circle)

plt.gca().set_aspect('equal', adjustable='box')
plt.legend(['Normal data', 'Outliers'])
plt.savefig('result2.png')
plt.show()
