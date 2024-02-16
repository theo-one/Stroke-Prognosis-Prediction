
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from scipy.stats import chi2_contingency
import scipy.stats as stats
import numpy as np


dict_column = {'atrialfib_rand': 'AF', 'hypertension_pre': 'HIN', 'diabetes_pre': 'DM'}

def data_preprocessing():
    path_file = "./datashare_aug2015.csv"
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

    filter_df['ohs6'] = (filter_df['ohs6']<=2).astype(int)

    filter_df.rename(columns=dict_column, inplace=True)
    print(len(filter_df))
    return filter_df

def vis_continous_property(df, real_label):
    # 计算关键统计量
    mean_value = df[real_label].mean()
    median_value = df[real_label].median()
    std_value = df[real_label].std()

    # 创建图表
    plt.figure(figsize=(10, 6))
    sns.histplot(df[real_label], kde=True, color='teal', alpha=0.5, linewidth=0, bins=12)

    # 优化线条和边缘
    sns.set(style="whitegrid")

    plt.rc('font', family='Times New Roman', size=20)
    plt.axvline(mean_value, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_value:.2f}')
    plt.axvline(median_value, color='gold', linestyle='-', linewidth=2, label=f'Median: {median_value:.2f}')

    # 设置标题和标签
    plt.xlabel(f'{real_label}',  fontdict={'family': 'Times New Roman', 'color': 'black', 'size': 20})
    plt.ylabel('Frequency',  fontdict={'family': 'Times New Roman', 'color': 'black', 'size': 20})

    # 设置刻度样式
    plt.xticks(fontsize=18, fontname='Times New Roman', color='black')
    plt.yticks(fontsize=18, fontname='Times New Roman', color='black')

    # 显示图例
    plt.legend(prop={'family': 'Times New Roman', 'size': 16})

    # 设置网格
    plt.grid(True, linestyle='--', alpha=0.7, color='grey')

    # 调整描述性统计信息的位置，以避免与图例重叠
    plt.text(x=0.95, y=0.75, s=f'Std Dev: {std_value:.2f}', transform=plt.gca().transAxes, 
            horizontalalignment='right', verticalalignment='top', color='black')
    # 保存为高分辨率图像
    # plt.savefig(f'{real_label}_distribution_improved.pdf', dpi=1000, format='pdf', bbox_inches='tight')
    plt.savefig(f'{real_label}_distribution_improved.eps', dpi=1000, format='eps', bbox_inches='tight')

    # 如果要在屏幕上显示图形，取消下面的注释
    plt.show()

def vis_ohs6(df):
    # fontdict={'family': 'Times New Roman', 'color': 'black', 'size': 20}
    positive_count = df[df['ohs6'] == 1].shape[0]
    negative_count = df[df['ohs6'] == 0].shape[0]

    counts = [positive_count, negative_count]
    labels = ['Positive', 'Negative']

    # 设置图形的风格
    sns.set(style="whitegrid")

    # 全局设置 Times New Roman 字体
    plt.rc('font', family='Times New Roman', size=20)

    # 创建柱状图
    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts, width=0.2)

    # 添加数值标签
    for i, count in enumerate(counts):
        plt.text(i, count + 0.05 * max(counts), str(count), ha='center', va='bottom', color='black') 
    # 设置标题和标签
    plt.ylabel('Count', fontdict={'family': 'Times New Roman', 'color': 'black', 'size': 20})
    # plt.title('Positive and Negative Sample Counts', fontsize=20)

    # 设置刻度样式
    plt.xticks(fontsize=20, fontname='Times New Roman', color='black') 
    plt.yticks(fontsize=20, fontname='Times New Roman', color='black') 
    # 设置y轴的上限
    plt.ylim(0, max(counts) * 1.2)

    # 设置网格
    plt.grid(True, linestyle='--', alpha=0.7, color='grey')

    # 调整图形布局
    plt.tight_layout()

    # 保存为矢量图形格式和高分辨率PNG格式
    # plt.savefig('sample_counts_improved.pdf', dpi=1000, format='pdf', bbox_inches='tight')

    # 保存为eps格式的文件
    plt.savefig('sample_counts_improved.eps', dpi=1000, format='eps', bbox_inches='tight')

    # 如果要在屏幕上显示图形，取消下面的注释
    plt.show()
   
def vis_binary_property(df, real_label):
    is_label_pre = df[df[f'{real_label}'] == 1].shape[0]
    not_label_pre = df[df[f'{real_label}'] == 2].shape[0]

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
    # 计算相关系数矩阵
    # 'brainsite7', 'stroketype'
    features_df = df[['gender', 'AF',  'HIN', 'DM', 'stroke_pre']]

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
    ax = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', annot_kws={"size": 16})
    plt.rc('font', family='Times New Roman', size=20)
    # # 设置标题和字体样式
    # plt.title("Heatmap of Correlation Matrix", fontsize=16, fontweight='bold')

    plt.xticks(rotation=45, ha='center', fontsize=20, fontname='Times New Roman', color='black')
    plt.yticks(rotation=45, ha='right', fontsize=20, fontname='Times New Roman', color='black')

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)

    # 调整图形布局
    plt.tight_layout()

    # 保存为矢量图形格式和高分辨率PNG格式
    # plt.savefig('heatmap_correlation_matrix.pdf', dpi=1000, format='pdf', bbox_inches='tight')
    plt.savefig('heatmap_correlation_matrix.eps', dpi=1000, format='eps', bbox_inches='tight')

    plt.show()

def vis_continous_label_property(df, real_label):
    # 箱型图
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=f'{real_label[-4:]}', y=f'{real_label[0:-5]}', data=df)
    # plt.title(f'Boxplot of {real_label[0:-5]} for each {real_label[-4:]} Category')

    # 设置标题和标签（假设 real_label 变量是一个组合了两个字段的名字）
    plt.xlabel(real_label[-4:], fontdict={'family': 'Times New Roman', 'color': 'black', 'size': 20})
    plt.ylabel(real_label[:-5], fontdict={'family': 'Times New Roman', 'color': 'black', 'size': 20})

    # 设置刻度样式
    plt.xticks(fontsize=20, fontname='Times New Roman', color='black')
    plt.yticks(fontsize=20, fontname='Times New Roman', color='black')

    # 设置网格
    plt.grid(True, linestyle='--', alpha=0.7, color='grey')

    # 调整图形布局
    plt.tight_layout()

    # 保存为矢量图形格式和高分辨率PNG格式
    # plt.savefig('boxplot_improved.pdf', dpi=1000, format='pdf', bbox_inches='tight')
    plt.savefig('boxplot_improved.eps', dpi=1000, format='eps', bbox_inches='tight')

    # 如果要在屏幕上显示图形，取消下面的注释
    plt.show()

def vis_class_label_property(df, real_label):
    # 设置图表风格
    sns.set_style("whitegrid")

    # 创建图表
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x=f'{real_label[0:-5]}', hue=f'{real_label[-4:]}', data=df, palette='Set2')
    plt.title(f'Count of {real_label[-4:]} by {real_label[0:-5]}', fontsize=14, fontweight='bold')
    plt.xlabel(f'{real_label[0:-5]}', fontsize=12)
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
    # 创建交叉表
    cross_tab = pd.crosstab(df[f'{real_label[0: -5]}'], df[f'{real_label[-4:]}'])
    print(cross_tab)
    # 进行卡方检验
    chi2, p, dof, excepted = chi2_contingency(cross_tab)

    print("Chi-squared: ", chi2)
    print("p-value", p)
    print("freedom", dof)

   
    # 将数字转换为文本标签 
    # df["gender"] = df["gender"].replace({1.0: "Female", 2.0: "Male"})
    # df["AF"] = df["AF"].replace({1.0: f"is", 2.0: f"not"})
    # df["HIN"] = df["HIN"].replace({1.0: "is", 2.0: "not"})
    # df["DM"] = df["DM"].replace({1.0: "is", 2.0: "not"})
    # class_df = df[["gender", "AF", "HIN", "DM"]]
   

    # fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # 绘制每个变量的堆叠条形图
    # for i, (variable, order) in enumerate(class_df.items()):
    #     ax = axs[i // 2, i % 2]  # 选择适当的子图
    #     counts = pd.crosstab(df[variable], df['ohs6'])
    #     counts.plot(kind='bar', stacked=True, ax=ax, color=['#80CBC4', 'coral'])
    #     ax.set_ylabel('Count')
    #     ax.set_title(f'{variable.replace("_", " ").title()} and ohs6')
    #     ax.legend(title='ohs6')


    # # 调整布局
    # plt.tight_layout()
    # plt.show()
        
def main(vis_label):
    df = data_preprocessing()

    if vis_label in ['age', 'nihss']:
        vis_continous_property(df, vis_label)
    elif vis_label in ['brainsite7', 'stroketype']:
        vis_class_property(df, vis_label)
    elif vis_label in ['gender', 'AF', 'HIN', 'DM']:
        vis_binary_property(df, vis_label)
    elif vis_label == "ohs6":
        vis_ohs6(df)
    elif vis_label == "multicollinearity":
        vis_multicollinearity(df)
    elif vis_label in ['age_ohs6', 'nihss_ohs6']:
        vis_continous_label_property(df, vis_label)
    elif vis_label in []:
        vis_class_label_property(df, vis_label)
    elif vis_label in ['gender_ohs6', 'AF_ohs6', 'HIN_ohs6', 'DM_ohs6', 'stroke_pre_ohs6', 'brainsite7_ohs6', 'stroketype_ohs6']:
        vis_binary_label_property(df, vis_label)
    else:
        print('error property')
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vis_property', type=str, default='multicollinearity', help= 'Type of model to train')
    args = parser.parse_args()
    main(args.vis_property)

 
   