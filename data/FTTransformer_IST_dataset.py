

import torch
import pandas as pd
from torch.utils.data import Dataset

def data_preprocessing(path_file):
    df = pd.read_csv(path_file)

    # 去除所有对照组的数据和所有不合规的数据
    filter_df = df[(df['treatment'] == 'rt-PA') & (df['randvioltype'].isna())]

    # 剔除对应列有空值的行
    # 'treatdelay',
    filter_df = filter_df.dropna(subset=['age', 'gender', 'treatdelay', 'nihss', 'stroketype', 'brainsite7', 
                                'atrialfib_rand', 'hypertension_pre', 'diabetes_pre', 'ohs6', 'sbprand', 'dbprand', 'glucose', 'stroke_pre', 'gcs_score_rand'])
    
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

class FT_ISTDataset(Dataset):
    def __init__(self, path_file):
        filter_df = data_preprocessing(path_file)
        
        # 数值特征
        continuous_features = filter_df[['age', 'nihss', 'treatdelay', 'sbprand', 'dbprand', 'glucose', 'gcs_score_rand']]
        # 分类特征
        class_features = filter_df[['brainsite7', 'stroketype', 'gender', 'atrialfib_rand',  'hypertension_pre', 'diabetes_pre', 'stroke_pre']]
        # 标签
        label_features = filter_df['ohs6'].apply(lambda x: 1 if x <= 2 else 0)


        self.num_features = torch.tensor(continuous_features.values, dtype=torch.float32)
        self.cat_features = torch.tensor(class_features.values, dtype=torch.int32)
        self.label_features = torch.tensor(label_features, dtype=torch.float32)


    def __len__(self):
        return self.label_features.size(0)
    
    def __getitem__(self, idx):
        return self.num_features[idx], self.cat_features[idx], self.label_features[idx]


# 测试用
# test = ISTDataset("./data/visual/ISTdata/datashare_aug2015.csv")