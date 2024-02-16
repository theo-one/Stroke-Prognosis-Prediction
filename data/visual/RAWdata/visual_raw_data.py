
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import argparse
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import scipy.stats as stats
import numpy as np

dict_column = {'年龄': 'age', '溶栓前NIHSS评分': 'NIHSS', '尿酸': 'uric', '同型半胱氨酸': 'hcy', 'TG': 'TG', 'LDL': 'LDL', 
               '责任病灶部位': 'focalsite' ,'TOAST分型': 'TOAST', 'SBI': 'SBI', '性别' : 'gender', '房颤': 'atrialfib', '高血压': 'hypertension',
                '糖尿病': 'diabetes', '吸烟': 'smoking', '饮酒': 'drinking', '循环一致性': 'cycliconsist', '部位一致性': 'siteconsist', 'FEIS病灶': 'FEIS',
                '预后': 'label'}

def data_preprocessing():
    path_file = "./raw_data.xlsx"
    df = pd.read_excel(path_file, header=1)

    # 修改TOAST分型的数据
    df['TOAST分型'] = df['TOAST分型'].replace(['大'], 0)
    df['TOAST分型'] = df['TOAST分型'].replace(['小'], 1)
    df['TOAST分型'] = df['TOAST分型'].replace(['心源性', '心源', '心'], 2)

    print(len(df))
    # 清洗数据，删除特定列包含非数字的行
    columns_to_check = df.columns.tolist()
    columns_to_check.pop(0)  # 移除第一列
    df[columns_to_check] = df[columns_to_check].apply(pd.to_numeric, errors='coerce')
    # 删除有缺失数据的行
    df = df.dropna()

    df['预后'] = df['预后'].apply(lambda x: 1 if x >= 4 else 0)

    df.rename(columns=dict_column, inplace=True)
    df = df[df['hcy'] < 100]
    df = df[df['TG'] < 20]
    print(len(df))
    return df
   
def vis_continous_property(df, real_label):
    # 直方图
    plt.figure(figsize=(10, 6))
    sns.histplot(df[f'{real_label}'], kde=True, color='teal', alpha=0.5, linewidth=0.3)
    plt.title(f'{real_label} Distribution', fontsize=16, fontweight='bold', color='midnightblue')
    plt.xlabel(f'{real_label}', fontsize=13, fontweight='bold', color='darkslategray')
    plt.ylabel('Frequency', fontsize=13, fontweight='bold', color='darkslategray')
    plt.xticks(fontsize=11, fontweight='bold', color='gray')
    plt.yticks(fontsize=11, fontweight='bold', color='gray')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # 箱形图
    plt.figure(figsize=(10, 6))
    # 箱型图
    sns.boxplot(x=df[f'{real_label}'], color='lightgreen', fliersize=5, linewidth=2.5, whis=1.5)
    # 设置标题和标签样式
    plt.title(f'Box Plot of {real_label}', fontsize=15, fontweight='bold', color='navy', pad=20)
    plt.xlabel(f'{real_label}', fontsize=12, fontweight='bold', color='darkgreen', labelpad=15)
    plt.xticks(fontsize=10, fontweight='bold', color='darkslategray')
    # 优化背景和网格线样式
    plt.grid(True, linestyle='--', alpha=0.7, color='lightgrey')
    sns.set_style("whitegrid")
    # 显示图表
    plt.show()

def vis_ohs6(df, real_label):
    positive_count = df[df[f'{real_label}'] == 1].shape[0]
    negative_count = df[df[f'{real_label}'] == 0].shape[0]

    counts = [positive_count, negative_count]
    labels = ['Positive', 'Negative']

    # 设置图形的风格
    sns.set(style="whitegrid")

    # 创建柱状图
    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts, width=0.2)

    # 添加数值标签
    for i, count in enumerate(counts):
        plt.text(i, count + 0.05 * max(counts), str(count), ha='center', va='bottom', fontsize=12)

    # 设置标题和标签
    plt.ylabel('Count', fontsize=13, fontweight='bold')
    plt.title('Positive and Negative Sample Counts', fontsize=16)

    # 设置刻度样式
    plt.xticks(fontsize=11, fontweight='bold')
    plt.yticks(fontsize=11, fontweight='bold')

    # 设置y轴的上限
    plt.ylim(0, max(counts) * 1.2)

    # 设置网格
    plt.grid(True, linestyle='--', alpha=0.7, color='grey')

    # 调整图形布局
    plt.tight_layout()

    # 保存为矢量图形格式和高分辨率PNG格式
    plt.savefig('sample_counts_improved.svg', format='svg', bbox_inches='tight')
    plt.savefig('sample_counts_improved.png', dpi=300, bbox_inches='tight', format='png')

    # 如果要在屏幕上显示图形，取消下面的注释
    # plt.show()

    # 返回生成的图像文件路径
    ('sample_counts_improved.svg', 'sample_counts_improved.png')

def vis_binary_property(df, real_label):
    is_label_pre = df[df[f'{real_label}'] == 1].shape[0]
    not_label_pre = df[df[f'{real_label}'] == 0].shape[0]

    counts = [is_label_pre, not_label_pre]
    if real_label == 'gender':
        labels = [f'Female', f'Male']
    else:
        labels = [f'is_{real_label}', f'not_{real_label}']

    # 设置图形的大小
    plt.figure(figsize=(8, 6))
    # 创建柱状图，可以调整柱体的宽度和颜色
    plt.bar(labels, counts, width=0.2)
    # 在柱体上方添加数据标签
    for i, count in enumerate(counts):
        plt.text(i, count + 0.05 * max(counts), str(count), ha='center', va='bottom')
    # 设置标题和标签
    plt.xlabel('Sample Type', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'{real_label} count', fontsize=14)
    plt.ylim(0, max(counts) * 1.1)
    # 显示图形
    plt.tight_layout()
    plt.show()

def vis_class_property(df, real_label):
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x=f'{real_label}', data=df, color='teal', alpha=0.5)

    # 设置柱子的宽度
    ax.bar_label(ax.containers[0])

    plt.title(f'{real_label} Distribution', fontsize=16, fontweight='bold', color='midnightblue')
    plt.xlabel(f'{real_label}', fontsize=13, fontweight='bold', color='darkslategray')
    plt.ylabel('Frequency', fontsize=13, fontweight='bold', color='darkslategray')
    plt.xticks(fontsize=11, fontweight='bold', color='gray')
    plt.yticks(fontsize=11, fontweight='bold', color='gray')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def calculate_vif(df):
    X = add_constant(df)
    vifs = pd.DataFrame()
    vifs["Variable"] = X.columns
    vifs["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vifs

def vis_multicollinearity(df):

    # 连续变量计算相关系数矩阵
    # 'gender', 'atrialfib', 'hypertension', 'diabetes', 'smoking', 'drinking', 'cycliconsist', 'siteconsist'
    # features_df = df[['age', 'NIHSS', 'uric', 'hcy', 'TG', 'LDL']]
    # corr_matrix = features_df.corr()

    # # 创建热力图
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    # plt.title("Heatmap of Correlation Matrix")
    # plt.show()



    features_df = df[['gender', 'atrialfib', 'hypertension', 'diabetes', 'smoking', 'drinking', 'cycliconsist', 'siteconsist', 'FEIS']]

    # 初始化一个空的相关性矩阵
    corr_matrix = pd.DataFrame(data=0, columns=features_df.columns, index=features_df.columns)

    # 计算Phi系数
    for col1 in features_df.columns:
        for col2 in features_df.columns:
            if col1 != col2:
                table = pd.crosstab(features_df[col1], features_df[col2])
                chi2, p, dof, expected = stats.chi2_contingency(table)
                phi = np.sqrt(chi2 / features_df.shape[0])
                corr_matrix.loc[col1, col2] = phi
            else:
                corr_matrix.loc[col1, col2] = 1

    # 打印相关性矩阵
    print(corr_matrix)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

     # # 设置标题和字体样式
    # plt.title("Heatmap of Correlation Matrix", fontsize=16, fontweight='bold')
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # 调整图形布局
    plt.tight_layout()

    # 保存为矢量图形格式和高分辨率PNG格式
    # plt.savefig('heatmap_correlation_matrix.svg', format='svg', bbox_inches='tight')
    plt.savefig('heatmap_correlation_matrix.png', dpi=800, bbox_inches='tight', format='png')

    plt.show()

def vis_continous_label_property(df, real_label):
    # 箱型图
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=f'{real_label[-5:]}', y=f'{real_label[0:-6]}', data=df)
    plt.title(f'Boxplot of {real_label[0:-6]} for each {real_label[-5:]} Category')
    plt.show()

def vis_class_label_property(df, real_label):
    # 设置图表风格
    sns.set_style("whitegrid")

    # 创建图表
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x=f'{real_label[0:-6]}', hue=f'{real_label[-5:]}', data=df, palette='Set2')
    plt.title(f'Count of {real_label[-5:]} by {real_label[0:-6]}', fontsize=14, fontweight='bold')
    plt.xlabel(f'{real_label[0:-6]}', fontsize=12)
    plt.ylabel('Count', fontsize=12)

    # 在每个条形上添加计数
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 9), 
                    textcoords='offset points', 
                    fontsize=10)

    # 显示图表
    plt.show()

def vis_binary_label_property(df, real_label):
    gender_col = f'{real_label[0:-6]}'
    if real_label[0:-6] == "gender":
        series = ['Female', 'Male']
    else:
        series = [f'is_{real_label[0:-6]}', f'not_{real_label[0:-6]}']
    # 将数字转换为文本标签
    df[gender_col] = df[gender_col].replace({1.0: series[0], 0.0: series[1]})

    # 绘制条形图
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x=gender_col, hue=f'{real_label[-5:]}', data=df, palette='Set2')
    plt.title(f"Count Plot of {gender_col} by {real_label[-5:]}")

    for p in ax.patches:
        height = p.get_height()  # 获取条形的高度
        ax.text(p.get_x() + p.get_width() / 2., height + 3, int(height), ha="center")

    plt.show()
        
def main(vis_label):
    df = data_preprocessing()

    if vis_label in ['age', 'NIHSS', 'uric', 'hcy', 'TG', 'LDL']:
        vis_continous_property(df, vis_label)
    elif vis_label in ['focalsite', 'TOAST']:
        vis_class_property(df, vis_label)
    elif vis_label in ['gender', 'atrialfib', 'hypertension', 'diabetes', 'smoking', 'drinking', 'cycliconsist', 'siteconsist']:
        vis_binary_property(df, vis_label)
    elif vis_label == 'label':
        vis_ohs6(df, vis_label)
    elif vis_label == "multicollinearity":
        vis_multicollinearity(df)
    elif vis_label in ['age_label', 'NIHSS_label', 'uric_label', 'hcy_label', 'TG_label', 'LDL_label']:
        vis_continous_label_property(df, vis_label)
    elif vis_label in ['focalsite_label', 'TOAST_label']:
        vis_class_label_property(df, vis_label)
    elif vis_label in ['gender_label', 'atrialfib_label', 'hypertension_label', 'diabetes_label', 'smoking_label', 'drinking_label', 'cycliconsist_label', 'siteconsist_label']:
        vis_binary_label_property(df, vis_label)
    else:
        print('error property')
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vis_property', type=str, default='focalsite', help= 'Type of model to train')
    args = parser.parse_args()
    main(args.vis_property)

 
   

