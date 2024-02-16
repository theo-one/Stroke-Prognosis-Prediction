

import torch
import pandas as pd
from torch.utils.data import Dataset

def data_preprocessing(path_file):
    df = pd.read_csv(path_file)

    # 去除所有对照组的数据和所有不合规的数据
    filter_df = df[(df['treatment'] == 'rt-PA') & (df['randvioltype'].isna())]

    # 剔除对应列有空值的行
    # '
    filter_df = filter_df.dropna(subset=['age', 'gender', 'treatdelay', 'nihss', 'stroketype', 'brainsite7', 
                                'atrialfib_rand', 'hypertension_pre', 'diabetes_pre', 'ohs6', 'sbprand', 'dbprand', 'glucose', 'gcs_score_rand', 'stroke_pre'])
    
    # 删除房颤、高血压、糖尿病以及性别列中不合法的数值
    columns_to_check = ['atrialfib_rand', 'hypertension_pre', 'diabetes_pre', 'gender', 'stroke_pre']
    for col in columns_to_check:
        filter_df = filter_df[filter_df[col].isin([1, 2])]

    # 将二分类中的1和2改为 -> 1表示患有该疾病，0表示不患病
    for col in columns_to_check:
        filter_df[col] = filter_df[col].replace(2, 0)
    
    # 使得该列满足从0开始计数的分类特征
    filter_df['stroketype'] = filter_df['stroketype'].replace(5, 0)

    print(len(filter_df))
    return filter_df

class ISTDataset(Dataset):
    def __init__(self, path_file):
        filter_df = data_preprocessing(path_file)
        
        # 筛选出输入模型的特征和标签
        continuous_features = filter_df[['age', 'nihss', 'treatdelay', 'gcs_score_rand', 'sbprand', 'dbprand', 'glucose']]
        class_features = filter_df[[ 'brainsite7', 'stroketype']]
        binary_features = filter_df[['gender', 'atrialfib_rand',  'hypertension_pre', 'diabetes_pre', 'stroke_pre']]
        label_features = filter_df['ohs6'].apply(lambda x: 1 if x <= 2 else 0)

        # 连续数据进行归一化处理
        continuous_features = torch.tensor(continuous_features.values, dtype=torch.float32)
        continuous_features = (continuous_features - continuous_features.mean(dim=0)) / continuous_features.std(dim=0)

        # one_hot编码
        encoded_features = []
        for feature_name in class_features:
            # 转换为张量并应用独热编码
            encoded_tensor = torch.tensor(class_features[feature_name].values, dtype=torch.int64)
            encoded_one_hot = torch.nn.functional.one_hot(encoded_tensor).float()
            # 将编码后的特征添加到列表中
            encoded_features.append(encoded_one_hot)

        # 拼接one_hot编码 
        all_encoded_features = torch.cat(encoded_features, dim=1)
        # 二进制数据
        binary_features = torch.tensor(binary_features.values, dtype=torch.float32)

        # 拼接所有特征
        self.all_input_features = torch.cat((continuous_features, all_encoded_features, binary_features), dim=1)
        self.label_features = torch.tensor(label_features, dtype=torch.float32)


    def __len__(self):
        return self.all_input_features.size(0)
    
    def __getitem__(self, idx):
        return self.all_input_features[idx], self.label_features[idx]


# 测试用
# test = ISTDataset("./data/visual/ISTdata/datashare_aug2015.csv")