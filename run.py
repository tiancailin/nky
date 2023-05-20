import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 读取csv文件
# df = pd.read_csv('test.csv', comment='#', na_values='无此项', header=0)
df = pd.read_csv('ALL.Data.csv', comment='#', na_values='无此项', header=0)

# 删除空白信息行
df.dropna(subset=['Cost_National'], inplace=True)

#删除对应的列，根据询问lee和对比r语言代码，应删除的列有Organization、Department、Province、City、Distract、Region、Result、Order、Year、ID、
columns_to_drop = ['Organization', 'Department', 'Province', 'City', 'Distract', 'Region', 'Region', 'Order', 'Year', 'ID']
df.drop(columns=columns_to_drop, inplace=True)

#删除带"_Score"关键字的列
columns_to_drop = [col for col in df.columns if '_Score' in col]
df.drop(columns=columns_to_drop,inplace=True)

# df数据框目前删除了不必要的数据列和Cost_National为空的数据行！！！！！！！

# 选择数值型数据
df_numeric = df.select_dtypes(include=np.number).copy()
print('Number of rows:', len(df_numeric))

# print(df_numeric)
# 只选取2018年的数据
df_numeric = df_numeric[df_numeric["Year"] == 2018]
print('Number of 训练数据:', len(df_numeric))
missing_count = df_numeric.isnull().sum()
print("缺失条数：",len(missing_count))
print(df_numeric.isna())
# 删除无用的列
df_numeric.drop(columns=['Order'], inplace=True)

# 统计每列缺失值数量
# missing_count = df_numeric.isnull().sum()


# 去除包含缺失值的行
# df_numeric.dropna(inplace=True)
print('最终可训练数据:', len(df_numeric))
# print("this is missing")
# print(missing_count)
# print(na_count)





# 拆分训练集和测试集
# train_indices = np.random.rand(len(df_numeric)) < 0.7
# train = df_numeric[train_indices]
# test = df_numeric[~train_indices]
train, test = train_test_split(df_numeric, train_size=0.7, test_size=0.3, random_state=123)

# 构建随机森林模型
rf = RandomForestRegressor(n_estimators=500, random_state=123)

print(train)
print(len(train))
# input("132312332:")


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
