import os
import pandas as pd

# 指定文件夹路径
folder_path = "your_folder_path" # 此为存放政策分词、分词后词频.txt的目录，应与本read.py处于目录中平级的位置

# 存储数据的列表
data = []

# 遍历文件夹及子文件夹中的txt文件
for root, dirs, files in os.walk(folder_path):
    for file_name in files:
        if file_name.endswith(".txt") and "_分词后_词频.txt" in file_name:
            print("正在处理      " + file_name + ".........")
            file_path = os.path.join(root, file_name)
            
            # 提取文件名中的"*******"部分
            file_id = file_name.split("_")[0]
            
            # 读取文件的前5行数据
            with open(file_path, 'r') as file:
                lines = [line.strip().split() for line in file.readlines()[:5]]
            
            # 提取数据并添加到列表中
            for line in lines:
                if len(line) == 2:
                    data.append([line[0], int(line[1]), file_id])

# 创建DataFrame
df = pd.DataFrame(data, columns=["高频词", "次数", "出处"])

# 打印DataFrame
print(df)

# 以下暂时不用
# # 创建一个字典，用于存储第三列的压缩结果
# compressed_data = {}

# # 遍历第三列，将相同出处的高频词和次数压缩成一格
# for idx, row in df.iterrows():
#     key = row["出处"]
#     if key in compressed_data:
#         compressed_data[key]["高频词"].append(row["高频词"])
#         compressed_data[key]["次数"].append(row["次数"])
#     else:
#         compressed_data[key] = {"高频词": [row["高频词"]], "次数": [row["次数"]]}

# # 重新构建DataFrame，将压缩后的结果写入
# compressed_df = pd.DataFrame(columns=["高频词", "次数", "出处"])
# for key, value in compressed_data.items():
#     compressed_df = pd.concat([compressed_df, pd.DataFrame({"高频词": [", ".join(value["高频词"])], "次数": [sum(value["次数"])], "出处": [key]})], ignore_index=True)

# # 打印DataFrame
# print(compressed_df)
# output_file_path2 = "output2.xlsx"
# compressed_df.to_excel(output_file_path2)

# 保存为xlsx文件
output_file_path = "output.xlsx"
df.to_excel(output_file_path)


