import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import joblib

# 文件保存实际路径
save_root = "xls分析202311/"
split_csv = save_root + "processed_allData.csv"
reslut_PDF01 = save_root + "2023.sample.RF.Validation.pdf"
reslut_PDF02 = save_root + "2023.sample.RF.Importance.pdf"

# 读取csv文件
# df = pd.read_csv('test.csv', comment='#', na_values='无此项', header=0)
df = pd.read_csv('2018-2020年考核单位的得分情况(2023.11）-分析.csv', comment='#', na_values='无此项', header=0, low_memory=False)
print(df)
input()

# 删除空白信息行
df.dropna(subset=['Cost_National'], inplace=True)

# 删除对应的列，根据询问lee和对比r语言代码，应删除的列有Organization、Department、Province、City、Distract、Region、Result、Order、Year、ID
# 2023.11.24新增delete系列和unknow系列
columns_to_drop = ['Organization', 'Department', 'Province', 'City', 'Distract', 'Region', 'Result', 'Order', 'ID', 'delete01', 'delete02', 'delete03', 'delete04', 'delete05', 'delete06', 'delete07', 'delete08', 'delete09', 'delete10', 'delete11', 'delete12', 'delete13', 'delete14', 'delete15', 'delete16', 'unknow01', 'unknow02' ] 
# Year先保留后面需要筛选2018年的数据
df.drop(columns=columns_to_drop, inplace=True)
print(df)
input()

#删除带"_Score"关键字的列
columns_to_drop = [col for col in df.columns if '_Score' in col]
# print("这里是被删除的带_Score的列：",columns_to_drop)
df.drop(columns=columns_to_drop,inplace=True)
print(df)
input()

# df数据框目前删除了不必要的数据列和Cost_National为空的数据行！！！！！！！

# 选择数值型数据
df_numeric = df.select_dtypes(include=np.number).copy()
print('Number of rows:', len(df_numeric))

# print(df_numeric)
# 只选取特定年的数据
df_numeric = df_numeric[df_numeric["Year"] == 2023]
# 删除Year列 
df_numeric.drop(columns=['Year'], inplace=True)

print('Number of 训练数据:', len(df_numeric))
missing_count = df_numeric.isnull().sum()
print("缺失条数：",len(missing_count))
print(df_numeric.isna())

#以0回填NaN空位
df_numeric.fillna(0, inplace=True)


# 去除包含缺失值的行
# df_numeric.dropna(inplace=True)
print('最终可训练数据:', len(df_numeric))
# print("this is missing")
# print(missing_count)
# print(na_count)

# 2023.11.27开始添加的重要内容，完全实现李博R语言部分的功能
# save to csv，从这里开始
# df_numeric.to_csv("output_20231127.csv", index=False)  # For xlsx csv

# 写入CSV文件
df_numeric.to_csv(split_csv, index=False)

# 设置随机数种子
np.random.seed(123)

# 划分训练集
samples = np.random.choice(df_numeric.index, size=int(np.ceil(len(df_numeric) * 0.7)), replace=False)
train_data = df_numeric.loc[samples]

# 构建随机森林模型
rf_model = RandomForestRegressor(n_estimators=500, random_state=123)
rf_model.fit(train_data.drop(columns='Score'), train_data['Score'])

# 输出随机森林模型，并保存
print(rf_model)
joblib.dump(rf_model, save_root + 'model_20231127.pkl')

# 在训练集上进行预测
train_predictions = rf_model.predict(train_data.drop(columns='Score'))

# 绘制训练集图表
plt.figure(figsize=(9, 6))
plt.subplot(1, 2, 1)
plt.scatter(train_data['Score'], train_predictions)
plt.title('Train')
plt.xlabel('Score')
plt.ylabel('Predict')
plt.plot([0, 100], [0, 100], color='red')
plt.text(15, 85, f"R={np.corrcoef(train_data['Score'], train_predictions)[0, 1]:.2f}", color='blue')

# 在测试集上进行预测
# test_predictions = rf_model.predict(df_numeric.drop(samples)['Score']) # 之前报错的
test_predictions = rf_model.predict(df_numeric.drop(samples).drop(columns='Score'))

# 绘制测试集图表
plt.subplot(1, 2, 2)
plt.scatter(df_numeric.drop(samples)['Score'], test_predictions)
plt.title('Test')
plt.xlabel('Score')
plt.ylabel('Predict')
plt.plot([0, 100], [0, 100], color='red')
plt.text(15, 85, f"R={np.corrcoef(df_numeric.drop(samples)['Score'], test_predictions)[0, 1]:.2f}", color='blue')

# 保存PDF图表
plt.savefig(reslut_PDF01)
plt.show()

# 计算 Gini Importance
gini_importance = rf_model.feature_importances_

# 计算均方误差的减少
mse_reduction = []
for feature in train_data.drop(columns='Score').columns:
    X_permuted = train_data.drop(columns='Score').copy()
    X_permuted[feature] = np.random.permutation(X_permuted[feature])
    mse_permuted = ((rf_model.predict(X_permuted) - train_data['Score']) ** 2).mean()
    mse_reduction.append(mse_permuted)

# 计算排列重要性
perm_importance = permutation_importance(rf_model, train_data.drop(columns='Score'), train_data['Score'], n_repeats=30, random_state=123)
perm_importance_mean = perm_importance.importances_mean

# 提取变量重要性
importance_otu = pd.DataFrame({
    'Feature': train_data.drop(columns='Score').columns,
    'Gini_Importance': gini_importance,
    'MSE_Reduction': mse_reduction,
    'Permutation_Importance': perm_importance_mean
})

# 按照 Gini Importance 从大到小排序
# importance_otu = importance_otu.sort_values(by='Gini_Importance', ascending=False)

# 分别按照三种重要性从大到小排序
importance_otu_gini = importance_otu.sort_values(by='Gini_Importance', ascending=False)
importance_otu_mse = importance_otu.sort_values(by='MSE_Reduction', ascending=False)
importance_otu_permutation = importance_otu.sort_values(by='Permutation_Importance', ascending=False)
print(importance_otu_gini)
print(importance_otu_mse)
print(importance_otu_permutation)
# 输出前几行变量重要性
print(importance_otu.head())

# 保存变量重要性图表为PDF
plt.figure(figsize=(12, 24)) # 调整图表大小
# plt.title('Variable Importance')

# Gini Importance
plt.subplot(1, 3, 1)
plt.title('Gini Importance')
plt.barh(importance_otu_gini['Feature'][:30], importance_otu_gini['Gini_Importance'][:30][::-1])

# MSE Reduction
plt.subplot(1, 3, 2)
plt.title('MSE Reduction')
plt.barh(importance_otu_mse['Feature'][:30], importance_otu_mse['MSE_Reduction'][:30][::-1], color='orange')

# Permutation Importance
plt.subplot(1, 3, 3)
plt.title('Permutation Importance')
plt.barh(importance_otu_permutation['Feature'][:30], importance_otu_permutation['Permutation_Importance'][:30][::-1], color='green')

plt.tight_layout()  # 调整布局以避免重叠
plt.savefig(reslut_PDF02)
plt.show()


"""
# 之前跑通的，到结束
# 提取变量重要性
importance_otu = pd.DataFrame({'Feature': df_numeric.drop(columns='Score').columns, 'Importance': rf_model.feature_importances_})

# 按照重要性排序
importance_otu = importance_otu.sort_values(by='Importance', ascending=False)

# 输出前几行变量重要性
print(importance_otu.head())

# 保存变量重要性图表为PDF
plt.figure(figsize=(9, 6))
plt.title('Variable Importance')

# 交换 x 和 y 轴，并按重要性从大到小排列
# plt.barh(importance_otu['Feature'][:30], importance_otu['Importance'][:30])
plt.barh(importance_otu['Feature'][:30][::-1], importance_otu['Importance'][:30][::-1])
plt.savefig(reslut_PDF02)
plt.show()
"""






"""
以下是2023.5月份前的旧代码不要的部分，暂时保存，并移动至文件尾部
紧跟print('最终可训练数据:', len(df_numeric))

df_numeric.to_csv("output_20231127.csv", index=False)  # For xlsx csv

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
plt.show()
#plt.savefig('prediction.png')

# 将模型结果保存至文件
import joblib
joblib.dump(rf, 'model_20231124.pkl')
"""