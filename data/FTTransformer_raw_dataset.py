
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

    # 分类特征的值统一修改为从0开始
    df['FEIS病灶'] = df['FEIS病灶'] - 1
    print(len(df))
    return df
   
    
class FT_rawDataset(Dataset):
    def __init__(self, path_file):
        df = data_preprocessing(path_file)

        # 数值特征
        continuous_features = df[['年龄', '溶栓前NIHSS评分']]
        # 分类特征
        class_features = df[['责任病灶部位', 'TOAST分型', 'SBI', '性别', '房颤', '高血压', '糖尿病', 'FEIS病灶', '吸烟', '饮酒', '循环一致性', '部位一致性']]
        # 标签
        label = (df['预后'] >= 4).astype(int)

        self.num_features = torch.tensor(continuous_features.values, dtype=torch.float32)
        self.cat_features = torch.tensor(class_features.values, dtype=torch.int32)
        self.label_features = torch.tensor(label.values, dtype=torch.float32)

    def __len__(self):
        return self.label_features.size(0)
    
    def __getitem__(self, idx):
        return self.num_features[idx], self.cat_features[idx], self.label_features[idx]



# 测试用
# test_dataset = MyDataset(path_file="./visual/RAWdata/raw_data.xlsx")
