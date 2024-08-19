import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 加载鸢尾花数据集
iris = datasets.load_iris()
X, y = iris.data, iris.target

# 随机划分次数
n_splits = 10
# 正则化超参数C候选值
C_range = np.logspace(-3, 3, 10)

results = []

for i in range(n_splits):
    # 随机划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)

    # 交叉验证调整超参数C
    param_grid = {'C': C_range}
    grid_search = GridSearchCV(SVC(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # 最佳参数
    best_C = grid_search.best_params_['C']
    # 训练模型
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append((i, best_C, acc))

# 输出
for result in results:
    print(f"No.{result[0] + 1} random split: Best C range = {result[1]}, Accuracy = {result[2]:.4f}")
