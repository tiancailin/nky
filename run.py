import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# 读取csv文件
df = pd.read_csv('test.csv', comment='#', na_values='无此项', header=0)

# 删除空白信息列
df.dropna(subset=['Cost_National'], inplace=True)

# 选择数值型数据
df_numeric = df.select_dtypes(include=np.number).copy()

# 删除无用的列
df_numeric.drop(columns=['Order'], inplace=True)

# 统计每列缺失值数量
missing_count = df_numeric.isnull().sum()

# 根据年份选择数据
df_year = df_numeric[df_numeric['Year'] == 2018].copy()
df_year.drop(columns=['Year', 'ID'], inplace=True)

# 去除包含缺失值的行
df_year.dropna(inplace=True)

# 拆分训练集和测试集
train_indices = np.random.rand(len(df_year)) < 0.7
train = df_year[train_indices]
test = df_year[~train_indices]

# 构建随机森林模型
rf = RandomForestRegressor(n_estimators=500, random_state=123)

print(df_year)

# 拟合训练集
rf.fit(train.drop(columns=['Score']), train['Score'])

# 预测测试集
predictions = rf.predict(test.drop(columns=['Score']))

# 绘制预测结果与实际结果的散点图
plt.scatter(test['Score'], predictions)
plt.xlabel('Actual Score')
plt.ylabel('Predicted Score')
plt.title('Random Forest Regression')
plt.savefig('predictions.png')

# 将模型结果保存至文件
import joblib
joblib.dump(rf, 'model.pkl')
