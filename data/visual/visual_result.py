import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
import seaborn as sns
import argparse

def process_mutiple(model_type, data_type, importance_coefficients):
    if model_type in ['RF', 'XGBoost']:
        if data_type == 'IST3':
            # 求和
            indices_1 = [7, 8, 9, 10, 11]
            indices_2 = [12, 13, 14]

            sum_1 = importance_coefficients[indices_1].sum()
            sum_2 = importance_coefficients[indices_2].sum()

            # 替换并调整数组大小
            importance_coefficients = np.delete(importance_coefficients, indices_1 + indices_2)
            importance_coefficients = np.insert(importance_coefficients, indices_1[0], sum_1)
            importance_coefficients = np.insert(importance_coefficients, indices_2[0] - len(indices_1) + 1, sum_2)
        else:
            indices_1 = [2, 3, 4, 5]
            indices_2 = [6, 7, 8]
            indices_3 = [9, 10, 11]

            sum_1 = importance_coefficients[indices_1].sum()
            sum_2 = importance_coefficients[indices_2].sum()
            sum_3 = importance_coefficients[indices_3].sum()

            # 替换并调整数组大小
            importance_coefficients = np.delete(importance_coefficients, indices_1 + indices_2 + indices_3)
            importance_coefficients = np.insert(importance_coefficients, indices_1[0], sum_1)
            importance_coefficients = np.insert(importance_coefficients, indices_2[0] - len(indices_1) + 1, sum_2)
            importance_coefficients = np.insert(importance_coefficients,
                                                indices_3[0] - len(indices_2) - len(indices_1) + 1, sum_3)
    else:
        if data_type == 'IST3':
            # 求和
            indices_1 = [7, 8, 9, 10, 11]
            indices_2 = [12, 13, 14]
        
            mean_absolute_coefficient_1 = np.mean(np.abs(importance_coefficients[indices_1]))
            mean_absolute_coefficient_2 = np.mean(np.abs(importance_coefficients[indices_2]))

            # 替换并调整数组大小
            importance_coefficients = np.delete(importance_coefficients, indices_1 + indices_2)
            importance_coefficients = np.insert(importance_coefficients, indices_1[0], mean_absolute_coefficient_1)
            importance_coefficients = np.insert(importance_coefficients, indices_1[0] + 1, mean_absolute_coefficient_2)
        else:
            indices_1 = [2, 3, 4, 5]
            indices_2 = [6, 7, 8]
            indices_3 = [9, 10, 11]

            mean_absolute_coefficient_1 = np.mean(np.abs(importance_coefficients[indices_1]))
            mean_absolute_coefficient_2 = np.mean(np.abs(importance_coefficients[indices_2]))
            mean_absolute_coefficient_3 = np.mean(np.abs(importance_coefficients[indices_3]))

            # 替换并调整数组大小
            importance_coefficients = np.delete(importance_coefficients, indices_1 + indices_2 + indices_3)
            importance_coefficients = np.insert(importance_coefficients, indices_1[0], mean_absolute_coefficient_1)
            importance_coefficients = np.insert(importance_coefficients, indices_1[0] + 1, mean_absolute_coefficient_2)
            importance_coefficients = np.insert(importance_coefficients, indices_1[0] + 2, mean_absolute_coefficient_3)

    return importance_coefficients


def main(visual_type, data_type):
    if visual_type == 'ROC':
        # 读入所有的数据
        # 'dnm_res'
        machine_model_dicts = {'LR', 'RF', 'SVM', 'XGBoost'}
        deep_model_dicts = {'MDNM', 'MLP', 'ResNet', 'FT-Transformer'}
        machine_models = {}
        deep_models = {}
        for model_type in machine_model_dicts:
            label_file_path = f'./ROCData/{model_type}_{data_type}_labels.npy'
            preds_file_path = f'./ROCData/{model_type}_{data_type}_preds.npy'
            machine_models[model_type] = (np.load(label_file_path), np.load(preds_file_path))

        for model_type in deep_model_dicts:
            label_file_path = f'./ROCData/{model_type}_{data_type}_labels.npy'
            preds_file_path = f'./ROCData/{model_type}_{data_type}_preds.npy'
            deep_models[model_type] = (np.load(label_file_path), np.load(preds_file_path))
        
        # 机器学习方法的AUC图
        plt.figure(figsize=(10, 8))
        palette = sns.color_palette("bright", len(machine_models))  # Use a bright color palette

        for (model_name, (labels, preds)), color in zip(machine_models.items(), palette):
            fpr, tpr, thresholds = roc_curve(labels, preds)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})', lw=2, color=color)

        plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
        plt.xlabel('False Positive Rate', fontdict={'family': 'Times New Roman', 'color': 'black', 'size': 26})
        plt.ylabel('True Positive Rate', fontdict={'family': 'Times New Roman', 'color': 'black', 'size': 26})
        # plt.title('Receiver Operating Characteristic', fontsize=18)
        plt.xticks(fontsize=26, fontname='Times New Roman', color='black')
        plt.yticks(fontsize=26, fontname='Times New Roman', color='black')
        plt.grid(True)

        # Move legend to the lower right corner
        plt.legend(loc='lower right', prop={'family': 'Times New Roman', 'size': 22}, frameon=True)

        plt.tight_layout()
        # plt.savefig(f'{data_type}_Multiple_Machine_ROC_auc.pdf', dpi=1000, format='pdf', bbox_inches='tight')
        plt.savefig(f'{data_type}_Machine_ROC.eps', dpi=1000, format='eps', bbox_inches='tight')

        plt.show()

        # 深度学习方法的AUC图
        # plt.figure(figsize=(10, 8))
        # palette = sns.color_palette("bright", len(deep_models))  # Use a bright color palette

        # for (model_name, (labels, preds)), color in zip(deep_models.items(), palette):
        #     fpr, tpr, thresholds = roc_curve(labels, preds)
        #     roc_auc = auc(fpr, tpr)
        #     plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})', lw=2, color=color)

        # plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
        # plt.xlabel('False Positive Rate', fontdict={'family': 'Times New Roman', 'color': 'black', 'size': 26})
        # plt.ylabel('True Positive Rate', fontdict={'family': 'Times New Roman', 'color': 'black', 'size': 26})
        # # plt.title('Receiver Operating Characteristic', fontsize=18)
        # plt.xticks(fontsize=26, fontname='Times New Roman', color='black')
        # plt.yticks(fontsize=26, fontname='Times New Roman', color='black')
        # plt.grid(True)

        # # Move legend to the lower right corner
        # plt.legend(loc='lower right', prop={'family': 'Times New Roman', 'size': 22}, frameon=True)

        # plt.tight_layout()
      
        # plt.savefig(f'{data_type}_Deep_ROC.eps', dpi=1000, format='eps', bbox_inches='tight')

        # plt.show()

    elif visual_type == 'FI':
        # 读入所有的数据
        model_dicts = {'LR', 'RF', 'XGBoost'}
        models = {}

        if data_type == 'IST3':
            feature_names = ['age', 'NIHSS', 'TD', 'GCS', 'SBP', 'DBP', 'Glucose', 'BS', 'ST', 'gender', 'AF', 'HIN',
                             'DM', 'Stkp']
            
        else:
            feature_names = ['age', 'NIHSS', 'FS', 'TOAST', 'SBI', 'gender', 'AF', 'HIN',
                             'DM', 'SMK', 'DRK', 'cyclicon', 'sitecon', 'FEIS']

        for model_type in model_dicts:
            label_file_path = f'./FIData/{model_type}_{data_type}_importance_coe.npy'
            importance_coefficients = np.load(label_file_path)
            importance_coefficients = process_mutiple(model_type, data_type, importance_coefficients)

            models[model_type] = {'Feature': feature_names, 'Importance_coefficient': importance_coefficients}

        # 设置Seaborn的样式
        sns.set(style="whitegrid")
        # 设置图表大小
        plt.figure(figsize=(15, 12))
        # 设置颜色，使用高对比度的颜色调色板
        palette = sns.color_palette("deep", len(models))
        # 初始化条形图的位置和宽度
        bar_width = 0.25  # 条形图的宽度略微减小以增加清晰度
        additional_space = 0.5  # 增加额外空间以增加清晰度
        number_of_models = len(models)  # 模型的数量
        # 计算每组特征柱子组之间的总空间
        total_space_per_feature = (number_of_models * bar_width) + additional_space
        # 生成间隔更大的indices数组
        indices = np.arange(len(feature_names)) * total_space_per_feature

        plt.rc('font', family='Times New Roman', size=28)
        # 为每个模型生成并排条形图
        for i, (model_name, data) in enumerate(models.items()):
            if data_type == "IST3":
                new_data = [(i if i >= 0 else i / 2) for i in data['Importance_coefficient']]
            else:
                new_data = [(i / 2 ) for i in data['Importance_coefficient']]
            # 计算每个模型条形图的位置
            model_indices = indices + (i - (len(models) - 1) / 2) * bar_width
            plt.bar(model_indices, new_data, width=bar_width, label=model_name, color=palette[i], linewidth=1)

            # 在每个条形图上添加数值标签
            for index, loc, importance in zip(model_indices, new_data, data['Importance_coefficient']):
                if abs(importance) > 0.3 or abs(importance) < 0.02:  # 非零值才添加标签，以减少杂乱
                    plt.text(index, loc + (1 if loc >= 0 else -3) * 0.01, round(importance, 2), ha='center', va='bottom')

        # 设置坐标轴标签和标题
        # plt.xlabel('Features', fontsize=16, )  # 增大字体大小
        # plt.ylabel('Importance Coefficient', fontsize=16, )  # 增大字体大小
        plt.xticks(indices + bar_width / 2 - additional_space / 2, feature_names, rotation=45, ha="center", fontsize=30, fontname='Times New Roman', color='black')  # 增大字体大小
        plt.yticks(fontsize=30, fontname='Times New Roman', color='black')  # 增大字体大小

        ticks = plt.gca().get_yticks()
        if data_type == "IST3":
            new_ticks = [round(tick, 2) if tick >= 0 else round(tick * 2, 2) for tick in ticks]
        else:
            new_ticks = [round(tick * 2, 2)  for tick in ticks]
        plt.gca().set_yticklabels(new_ticks)
        # 添加图例
        plt.legend(loc='upper center', prop={'family': 'Times New Roman', 'size': 26}, frameon=True)  # 增大字体大小
        # 添加网格线
        plt.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey')
        # 调整布局和保存图表为矢量图形格式
        plt.tight_layout()
        # plt.savefig(f'{data_type}_feature_importance_comparison.pdf', dpi=1000, format='pdf',
        #             bbox_inches='tight')  # 保存为PDF
        
        plt.savefig(f'{data_type}_feature_importance_comparison.eps', dpi=1000, format='eps', bbox_inches='tight')

        # 显示图表
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--visual_type', type=str, default='FI', help='Type of visual')
    parser.add_argument('--data_type', type=str, default='RAW', help='Type of data')
    args = parser.parse_args()
    main(args.visual_type, args.data_type)


