
import torch
import pandas as pd
from torch.utils.data import Dataset


    
def data_preprocessing(path_file):
    # read file
    df = pd.read_excel(path_file, header=1)
    # 修改TOAST分型的数据
    df['TOAST分型'] = df['TOAST分型'].replace(['大'], 0)
    df['TOAST分型'] = df['TOAST分型'].replace(['小'], 1)
    df['TOAST分型'] = df['TOAST分型'].replace(['心源性', '心源', '心'], 2) 
    
   
    # 清洗数据，删除特定列包含非数字的行
    columns_to_check = df.columns.tolist()
    columns_to_check.pop(0)  # 移除第一列
    df[columns_to_check] = df[columns_to_check].apply(pd.to_numeric, errors='coerce')
    # 删除有缺失数据的行
    df = df.dropna() 
    
    # 有两行不正确的数据，判断是错误数据,所以在这里删掉了
    df = df[df['TG'] < 20]
    df['FEIS病灶'] = df['FEIS病灶'] - 1
    print(len(df))
    return df
   
    
class MyDataset(Dataset):
    def __init__(self, path_file):
        df = data_preprocessing(path_file)
        continuous_features = df[['年龄', '溶栓前NIHSS评分']]
        class_features = df[['责任病灶部位', 'TOAST分型', 'SBI']]
        binary_features = df[['性别', '房颤', '高血压', '糖尿病', '吸烟', '饮酒', '循环一致性', '部位一致性','FEIS病灶']]
        label = (df['预后'] >= 4).astype(int)

        # 连续特征归一化处理
        continuous_data = torch.tensor(continuous_features.values, dtype=torch.float32)
        # 均值、标准差归一化
        continuous_data_normalized = (continuous_data - continuous_data.mean(dim=0)) / continuous_data.std(dim=0)
      
        # one_hot编码
        encoded_features = []
        for feature in class_features:
            # 转换为张量并应用独热编码
            encoded_tensor = torch.tensor(df[feature].values, dtype=torch.int64)
            encoded_one_hot = torch.nn.functional.one_hot(encoded_tensor).float()
            # 将编码后的特征添加到列表中
            encoded_features.append(encoded_one_hot)
        # 拼接one_hot编码 
        all_encoded_features = torch.cat(encoded_features, dim=1)
        # 二进制数据
        binary_data = torch.tensor(binary_features.values, dtype=torch.float32)
        # 合并连续和二元特征
        self.input_features_tensor = torch.cat([continuous_data_normalized, all_encoded_features, binary_data], dim=1)
        self.label_tensor = torch.tensor(label.values, dtype=torch.float32)

    def __len__(self):
        return self.label_tensor.size(0)
    
    def __getitem__(self, idx):
        input_data = self.input_features_tensor[idx]
        label = self.label_tensor[idx]
        return input_data, label



# 测试用
# test_dataset = MyDataset(path_file="./visual/RAWdata/raw_data.xlsx")
