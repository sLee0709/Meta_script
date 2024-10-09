import lightgbm as lgb
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

#0. 模型优化从以下几个方面考虑：
#....(1) 特征选择： feature_importances = model.feature_importances_
#....(2) 数据不平衡：如下所示
#....(3) 调整早停（early_stopping）和交叉验证策略: model = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=500, early_stopping_rounds=10)


# 1. 加载数据集
data = load_iris()
X = data.data
y = (data.target != 0).astype(int)  # 将鸢尾花数据集转换为二分类问题

#加载自己的数据集，其中A是基因表达数据，B是分组信息
'''

# 假设数据A是一个CSV文件，其中行为基因，列为样本
data_a = pd.read_csv('data_A.csv', index_col=0)  # 加载基因表达矩阵数据 A
data_b = pd.read_csv('data_B.csv')  # 加载分组信息数据 B
# 确保样本ID的顺序一致
# 假设 data_b 的第一列是 sample ID，第二列是分组标签
data_b_sorted = data_b.set_index('sampleID')  # 将sampleID设为索引
data_a_sorted = data_a[data_b_sorted.index]  # 根据 data_b 的样本ID顺序对 data_a 排序

# 转换成LightGBM可以使用的格式
X = data_a_sorted.T.values  # 基因表达矩阵作为特征，转置后行为样本，列为基因
y = data_b_sorted['label'].values  # 分组标签作为目标变量

'''

# 2. 创建 StratifiedKFold 对象（这里使用 10-fold 交叉验证）
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)

# 3. LightGBM 参数网格设置
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],  # 学习率
    'num_leaves': [31, 50, 100],  # 叶子节点数
    'max_depth': [-1, 10, 20],  # 树的最大深度
    'min_child_samples': [20, 30, 50],  # 一个叶子节点中最小的样本数
    'subsample': [0.8, 0.9, 1.0],  # 数据采样率
    'colsample_bytree': [0.8, 0.9, 1.0],  # 特征采样率
    'reg_alpha': [0, 0.01, 0.1],  # L1 正则化
    'reg_lambda': [0, 0.01, 0.1],  # L2 正则化
    #'scale_pos_weight': [pos_weight]  方法1：手动设置两类数据的权重，在这之前先计算权重：pos_weight = sum(y == 0) / sum(y == 1)
    #'is_unbalance': [True]  方法2：通过is_balance参数自动调整数据平衡
}

# 4. 定义 LightGBM 分类器
lgb_estimator = lgb.LGBMClassifier(objective='binary', metric='auc')

# 5. 使用 GridSearchCV 进行超参数调优和交叉验证
grid = GridSearchCV(estimator=lgb_estimator, param_grid=param_grid, cv=kf, scoring='roc_auc', verbose=1, n_jobs=-1)
grid.fit(X, y)

# 6. 输出最佳参数和最佳交叉验证 AUC
print(f"最佳参数: {grid.best_params_}")
print(f"10-fold 交叉验证的最佳 AUC: {grid.best_score_}")

# 7. 使用最佳参数重新训练模型
best_params = grid.best_params_
best_model = lgb.LGBMClassifier(**best_params, objective='binary', metric='auc')

# 8. 重新训练模型，并用 StratifiedKFold 验证最终结果
auc_scores = []
for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # 使用最佳参数训练模型
    best_model.fit(X_train, y_train)

    # 预测概率，用于计算 AUC
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # 计算 AUC
    auc = roc_auc_score(y_test, y_pred_proba)
    auc_scores.append(auc)

# 9. 输出最终的平均 AUC
mean_auc = np.mean(auc_scores)
print(f"使用最佳参数进行 10-fold 交叉验证的平均 AUC: {mean_auc:.2f}")

##S1. 特征重要性可视化
import matplotlib.pyplot as plt

# 生成特征名称（如果有特征名称）
feature_importances = best_model.feature_importances_
feature_names = data.feature_names

# 绘制特征重要性
plt.figure(figsize=(10, 6))
plt.barh(np.arange(len(feature_importances)), feature_importances, align='center')
plt.yticks(np.arange(len(feature_importances)), feature_names)
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("LightGBM Feature Importances")
plt.show()