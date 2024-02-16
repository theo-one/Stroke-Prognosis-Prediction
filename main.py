
import torch
import argparse
import numpy as np
import random

from data.raw_dataset import MyDataset
from data.IST_dataset import ISTDataset
from data.FTTransformer_IST_dataset import FT_ISTDataset
from data.FTTransformer_raw_dataset import FT_rawDataset

from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset, random_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from model.MDNM import DNMModel
from model.MLP import MLPModel
from model.ResNet import ResNet
from model.FTTransformer import FTTransformer

from train.train_dnm import dnm_trainer
from train.train_mlp import mlp_trainer
from train.train_resnet import resnet_trainer
from train.train_FTTransformer import FTTransformer_trainer

from eval.eval_common import evalCommon






def extract_data_from_loader(loader):
    x, y = [], []
    for data in loader:
        inputs, labels = data
        x.extend(inputs.numpy())
        y.extend(labels.numpy())
    return np.array(x), np.array(y)

param_space = {
    'dendriteNum': [5, 6, 7, 8, 9, 10],
    'k': [2, 3, 4, 5, 6, 7, 8],
    'ks': [2, 3, 4, 5, 6, 7, 8],
    'thetas': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    'lr': [2 * 0.1, 2 * 0.001, 2 * 0.001]
}

# 从参数空间中随机选择一组参数
def select_random_params(param_space):
    return {k: random.choice(v) for k, v in param_space.items()}

def main(model_type, data_type): 
    # 设置随机种子
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 数据集
    if data_type == 'IST3':
        path_file = "./data/visual/ISTdata/datashare_aug2015.csv"
        if model_type == "FT-Transformer":
            full_dataset = FT_ISTDataset(path_file)
            num_features = 7
            cat_card = [3, 5, 2, 2, 2, 2, 2]
        else:
            full_dataset = ISTDataset(path_file=path_file)
            feature_dims = 20
    elif data_type == "RAW":
        path_file = "./data/visual/RAWdata/raw_data.xlsx"
        if model_type == "FT-Transformer":
            full_dataset = FT_rawDataset(path_file)
            num_features = 2
            cat_card = [4, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        else:
            full_dataset = MyDataset(path_file=path_file)
            feature_dims = 21
   
    # 10折交叉验证
    n_splits = 10
    # 重复3次 
    repeats = 3
    # K折交叉验证数据集划分
    kf = KFold(n_splits=10, shuffle=True, random_state=seed)
    train_loaders = []
    val_loaders = []
    test_loaders = []
    train_val_loaders = []
    for fold, (train_val_idx, test_idx) in enumerate(kf.split(full_dataset)):
        train_val_subset = Subset(full_dataset, train_val_idx)
        test_subset = Subset(full_dataset, test_idx)

        train_subset, val_subset = random_split(train_val_subset, [len(train_val_subset) - int(len(train_val_subset) * 0.2), int(len(train_val_subset) * 0.2)])
        # 创建DataLoader
        train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=128, shuffle=False)
        test_loader = DataLoader(test_subset, batch_size=128, shuffle=False)
        train_val_loader = DataLoader(train_val_subset, batch_size=128, shuffle=True)
        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
        test_loaders.append(test_loader)
        train_val_loaders.append(train_val_loader)
    # 3次10折交叉验证的平均评估指标
    total_metrics = {
        'accuracy': 0,
        'recall' : 0,
        'precision':0,
        'f1':0,
        'roc_auc': 0,
    }

    # 统计所有重复次数中预测的样本和标签，画出平均的ROC-auc曲线图
    all_repeat_labels = []
    all_repeat_preds = []

    for repeat in range(repeats):
        # 汇总10折交叉验证模型的结果和标签
        all_labels = []
        all_preds = []
        params = select_random_params(param_space)
        for idx in range(n_splits):
            print(f"Repeat: {repeat} Model {idx+1}------------------------------")
            if model_type == "MDNM":
                # 一个dnm神经元
                model = DNMModel(input_dim=feature_dims, dendriteNum=params['dendriteNum'], k=params['k'], ks=params['ks'], thetas=params['thetas'])
                # 训练模型
                dnm_trainer(model, train_loaders[idx], val_loaders[idx], num_epochs=2000, learning_rate=params['lr'], num=idx, repeat=repeat).train()
                # 模型评估
                model.load_state_dict(torch.load(f'./result/modelBest/dnm_{idx}_modelBestParameters'))
            elif model_type == "dnm_res":
                # 一个dnm神经元
                model = DNMModel(input_dim=feature_dims, dendriteNum=params['dendriteNum'], k=params['k'], ks=params['ks'], thetas=params['thetas'])
                # 训练模型
                dnm_trainer(model, train_loaders[idx], val_loaders[idx], num_epochs=2000, learning_rate=params['lr'], num=idx, repeat=repeat).train()
                # 模型评估
                model.load_state_dict(torch.load(f'./result/modelBest/dnm_{idx}_modelBestParameters'))
            elif model_type == "LR":
                # 从DataLoader中提取数据
                X_train, y_train = extract_data_from_loader(train_val_loaders[idx])
                X_test, y_test = extract_data_from_loader(test_loaders[idx])
                # 创建并训练逻辑回归模型
                model = LogisticRegression()
                model.fit(X_train, y_train)
            elif model_type == "SVM":
                # 从DataLoader中提取数据
                X_train, y_train = extract_data_from_loader(train_val_loaders[idx])
                X_test, y_test = extract_data_from_loader(test_loaders[idx])
                model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
                model.fit(X_train, y_train)
            elif model_type == "RF":
                # 从DataLoader中提取数据
                X_train, y_train = extract_data_from_loader(train_val_loaders[idx])
                X_test, y_test = extract_data_from_loader(test_loaders[idx])
                # 创建并训练随机森林模型
                model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=seed)
                model.fit(X_train, y_train)
            elif model_type == "XGBoost":
                # 从DataLoader中提取数据
                X_train, y_train = extract_data_from_loader(train_val_loaders[idx])
                X_val, y_val = extract_data_from_loader(val_loaders[idx])
                X_test, y_test = extract_data_from_loader(test_loaders[idx])
                # 创建并训练XGBoost模型
                model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, use_label_encoder=False, eval_metric="logloss")
                model.fit(X_train, y_train, early_stopping_rounds=50, eval_set=[(X_val, y_val)])
            elif model_type == "MLP":
                model = MLPModel(input_dim=feature_dims, output_dim=1)
                mlp_trainer(model, train_loaders[idx], val_loaders[idx], num_epochs=2000, learning_rate=2 * 1e-3, num=idx, repeat=repeat).train()
                model.load_state_dict(torch.load(f'./result/modelBest/mlp_{idx}_modelBestParameters'))
            elif model_type == "ResNet":
                model = ResNet(d_in=feature_dims, n_blocks=2, d_main=32, d_hidden=64, dropout_first=0.1, dropout_second=0.1, d_out=1)
                resnet_trainer(model, train_loaders[idx], val_loaders[idx], num_epochs=2000, learning_rate=2 * 1e-3, num=idx, repeat=repeat).train()
                model.load_state_dict(torch.load(f'./result/modelBest/resnet_{idx}_modelBestParameters'))
            elif model_type == "FT-Transformer":
                model = FTTransformer.make_default(
                        n_num_features=num_features,
                        cat_cardinalities=cat_card,
                        d_out=1,
                    )
                FTTransformer_trainer(model, train_loaders[idx], val_loaders[idx], num_epochs=2000, learning_rate=1e-4, num=idx, repeat=repeat).train()
            else:
                raise ValueError("Invalid model type provided") 

            # 模型在测试集中的表现
            if model_type in ["MDNM", "MLP", "ResNet"]:
                preds = []
                labels = []
                for input, label in test_loaders[idx]:
                    outputs = model(input)
                    preds.append(outputs.detach().numpy())
                    labels.append(label.detach().numpy())
                preds = np.concatenate(preds)
                labels = np.concatenate(labels)
            elif model_type == "FT-Transformer":
                preds = []
                labels = []
                for num, cat, label in test_loaders[idx]:
                    outputs = model(num, cat)
                    preds.append(outputs.detach().numpy())
                    labels.append(label.detach().numpy())
                preds = np.concatenate(preds)
                labels = np.concatenate(labels)
            elif model_type in ["LR", "SVM", "RF", "XGBoost"]:
                preds = model.predict_proba(X_test)[:, 1]
                labels = y_test

            all_preds.append(preds)
            all_labels.append(labels)
        
        # 将统计的预测结果和标签拼接在一起
        all_labels_np = np.concatenate(all_labels)
        all_preds_np = np.concatenate(all_preds)

        # 对于10折的评估指标
        new_metrics = evalCommon(all_labels_np, all_preds_np).evaluation()
        for key, value in total_metrics.items():
            total_metrics[key] += new_metrics[key]
    
        all_repeat_labels.append(all_labels_np)
        all_repeat_preds.append(all_preds_np)
    
    # 平均的准确率和loss
    total_average_metrics = {k : v / repeats for k, v in total_metrics.items()}
    for metric, value in total_average_metrics.items():
        print(f"{metric.capitalize()}: {value:.2f}")

    # 合并3次重复的标签和对应的结果
    all_repeat_labels = np.concatenate(all_repeat_labels)
    all_repeat_preds = np.concatenate(all_repeat_preds)

    # 保存文件可视化所有模型的ROC曲线图
    # np.save(f'./data/visual/ROCData/{model_type}_{data_type}_labels.npy', all_repeat_labels)
    # np.save(f'./data/visual/ROCData/{model_type}_{data_type}_preds.npy', all_repeat_preds)
  
    # 特征重要性解释
    if model_type in ['RF', 'XGBoost']:
        # 特征重要性排名
        importance_coefficients = model.feature_importances_  
        np.save(f'./data/visual/FIData/{model_type}_{data_type}_importance_coe.npy', importance_coefficients)
    elif model_type == "LR":
        importance_coefficients = model.coef_[0]
        np.save(f'./data/visual/FIData/{model_type}_{data_type}_importance_coe.npy', importance_coefficients)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='FT-Transformer', help= 'Type of model to train')
    parser.add_argument('--data_type', type=str, default='IST3', help='Type of data')
    args = parser.parse_args()
    main(args.model_type, args.data_type)
