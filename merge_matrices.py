#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
合并每个细胞类型的expected和observed矩阵，
按照不同的样本分组进行置换检验，计算P值并绘制热图
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from coexistence_utils import merge_matrices_with_pvalue, element_wise_permutation_test

def get_sample_id(filename):
    """
    从文件名中提取样本ID (SX-X)
    
    参数:
    filename: 文件名
    
    返回:
    样本ID，例如'S1-1'
    """
    match = re.match(r'(S\d+-\d+)_', os.path.basename(filename))
    if match:
        return match.group(1)
    return None

def merge_observed_expected_with_pvalue(cell_type, analysis_type="coexistence"):
    """
    为特定细胞类型的不同样本分组间进行置换检验，计算P值，并将结果绘制为热图
    
    参数:
    cell_type: 细胞类型名称
    analysis_type: 分析类型，例如"coexistence"
    """
    print(f"开始处理细胞类型: {cell_type}, 分析类型: {analysis_type}")
    
    # 获取所有相关文件
    expected_files = glob.glob(f"./results/*_{analysis_type}_expected_{cell_type}.csv")
    observed_files = glob.glob(f"./results/*_{analysis_type}_observed_{cell_type}.csv")
    
    if not expected_files or not observed_files:
        print(f"警告: 找不到细胞类型 {cell_type} 的 {analysis_type} 矩阵文件")
        return None
    
    # 根据样本ID对文件进行分组
    expected_by_sample = {}
    observed_by_sample = {}
    
    for file_path in expected_files:
        sample_id = get_sample_id(file_path)
        if sample_id:
            expected_by_sample.setdefault(sample_id, []).append(file_path)
    
    for file_path in observed_files:
        sample_id = get_sample_id(file_path)
        if sample_id:
            observed_by_sample.setdefault(sample_id, []).append(file_path)
    
    # 检查样本ID是否匹配
    all_sample_ids = set(expected_by_sample.keys()) | set(observed_by_sample.keys())
    print(f"找到的样本分组: {all_sample_ids}")
    
    # 读取每个样本的矩阵数据
    sample_data = {}
    gene_names = None
    
    for sample_id in all_sample_ids:
        expected_sample_files = expected_by_sample.get(sample_id, [])
        observed_sample_files = observed_by_sample.get(sample_id, [])
        
        if not expected_sample_files or not observed_sample_files:
            print(f"警告: 样本 {sample_id} 的期望或观察文件缺失，跳过")
            continue
        
        # 确保文件数量匹配
        expected_sample_files.sort()
        observed_sample_files.sort()
        
        # 读取矩阵
        expected_matrices = []
        observed_matrices = []
        
        for expected_file in expected_sample_files:
            expected_df = pd.read_csv(expected_file, index_col=0)
            if gene_names is None:
                gene_names = list(expected_df.index)
            expected_matrices.append(expected_df.values)
        
        for observed_file in observed_sample_files:
            observed_df = pd.read_csv(observed_file, index_col=0)
            observed_matrices.append(observed_df.values)
        
        # 计算该样本的平均矩阵
        if expected_matrices and observed_matrices:
            # 确保所有矩阵形状相同
            matrix_shape = expected_matrices[0].shape
            for mat in expected_matrices + observed_matrices:
                if mat.shape != matrix_shape:
                    print(f"警告: 样本 {sample_id} 中存在形状不一致的矩阵，跳过")
                    continue
            
            # 计算平均矩阵
            expected_avg = np.mean(expected_matrices, axis=0)
            observed_avg = np.mean(observed_matrices, axis=0)
            
            sample_data[sample_id] = {
                'expected': expected_avg,
                'observed': observed_avg,
                'gene_names': gene_names
            }
    
    # 进行样本间的置换检验
    if len(sample_data) < 2:
        print(f"警告: 细胞类型 {cell_type} 只有 {len(sample_data)} 个有效样本组，至少需要2个样本组才能进行比较")
        return None
    
    # 创建输出目录
    os.makedirs("./permutation_test", exist_ok=True)
    
    # 对所有样本对进行置换检验
    comparison_results = {}
    
    sample_ids = list(sample_data.keys())
    for i in range(len(sample_ids)):
        for j in range(i+1, len(sample_ids)):
            sample1 = sample_ids[i]
            sample2 = sample_ids[j]
            
            print(f"比较样本 {sample1} 和 {sample2}")
            
            # 获取两个样本的数据
            observed1 = sample_data[sample1]['observed']
            expected1 = sample_data[sample1]['expected']
            observed2 = sample_data[sample2]['observed']
            expected2 = sample_data[sample2]['expected']
            gene_names = sample_data[sample1]['gene_names']
            
            # 对观察值进行置换检验
            observed_diff = observed1 - observed2
            
            # 对期望值进行置换检验
            expected_diff = expected1 - expected2
            
            # 计算观察值与期望值之差的差异
            o_e_diff1 = observed1 - expected1
            o_e_diff2 = observed2 - expected2
            diff_of_diff = o_e_diff1 - o_e_diff2
            
            # 进行元素级别的置换检验
            p_value_matrix, _ = element_wise_permutation_test(
                o_e_diff1, o_e_diff2, n_permutations=1000
            )
            
            # 保存比较结果
            comparison_name = f"{sample1}_vs_{sample2}"
            comparison_results[comparison_name] = {
                'observed_diff': observed_diff,
                'expected_diff': expected_diff,
                'o_e_diff1': o_e_diff1,
                'o_e_diff2': o_e_diff2,
                'diff_of_diff': diff_of_diff,
                'p_value': p_value_matrix,
                'gene_names': gene_names
            }
            
            # 绘制和保存结果
            plot_comparison_results(
                comparison_results[comparison_name], 
                cell_type, 
                analysis_type, 
                comparison_name
            )
    
    return comparison_results

def plot_comparison_results(results, cell_type, analysis_type, comparison_name):
    """
    绘制样本比较结果的热图
    
    参数:
    results: 包含比较结果的字典
    cell_type: 细胞类型名称
    analysis_type: 分析类型
    comparison_name: 比较名称，如"S1-1_vs_S2-3"
    """
    # 提取矩阵和基因名称
    observed_diff = results['observed_diff']
    expected_diff = results['expected_diff']
    diff_of_diff = results['diff_of_diff']
    p_value_matrix = results['p_value']
    gene_names = results['gene_names']
    
    # 转换为DataFrame
    observed_diff_df = pd.DataFrame(observed_diff, index=gene_names, columns=gene_names)
    expected_diff_df = pd.DataFrame(expected_diff, index=gene_names, columns=gene_names)
    diff_of_diff_df = pd.DataFrame(diff_of_diff, index=gene_names, columns=gene_names)
    p_value_df = pd.DataFrame(p_value_matrix, index=gene_names, columns=gene_names)
    
    # 创建图形
    plt.figure(figsize=(20, 15))
    
    # 绘制四个子图
    plt.subplot(2, 2, 1)
    sns.heatmap(observed_diff_df, cmap='coolwarm', center=0, xticklabels=True, yticklabels=True)
    plt.title(f'Observed Difference - {comparison_name}')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    
    plt.subplot(2, 2, 2)
    sns.heatmap(expected_diff_df, cmap='coolwarm', center=0, xticklabels=True, yticklabels=True)
    plt.title(f'Expected Difference - {comparison_name}')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    
    plt.subplot(2, 2, 3)
    sns.heatmap(diff_of_diff_df, cmap='coolwarm', center=0, xticklabels=True, yticklabels=True)
    plt.title(f'Diff of (Observed-Expected) - {comparison_name}')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    
    plt.subplot(2, 2, 4)
    # 使用对数比例显示p值
    log_p_value_df = -np.log10(p_value_df.clip(lower=1e-10))
    sns.heatmap(log_p_value_df, cmap='viridis', xticklabels=True, yticklabels=True)
    plt.title(f'Significance (-log10 P-value) - {comparison_name}')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(f'./plots/permutation_test/{analysis_type}_{cell_type}_{comparison_name}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存结果为CSV
    observed_diff_df.to_csv(f'./plots/permutation_test/{analysis_type}_{cell_type}_{comparison_name}_observed_diff.csv')
    expected_diff_df.to_csv(f'./plots/permutation_test/{analysis_type}_{cell_type}_{comparison_name}_expected_diff.csv')
    diff_of_diff_df.to_csv(f'./plots/permutation_test/{analysis_type}_{cell_type}_{comparison_name}_diff_of_diff.csv')
    p_value_df.to_csv(f'./plots/permutation_test/{analysis_type}_{cell_type}_{comparison_name}_pvalue.csv')
    
    # 创建带有显著性标记的热图
    plt.figure(figsize=(14, 12))
    
    # 标记显著的位置 (p < 0.05)
    significant_mask = p_value_df < 0.05
    
    # 绘制热图
    ax = sns.heatmap(diff_of_diff_df, cmap='coolwarm', center=0, 
                     xticklabels=True, yticklabels=True, 
                     annot=False, fmt='.2f')
    
    # 在显著位置添加星号标记
    for i in range(len(gene_names)):
        for j in range(len(gene_names)):
            if significant_mask.iloc[i, j]:
                # p < 0.05时添加*
                if p_value_df.iloc[i, j] < 0.05 and p_value_df.iloc[i, j] >= 0.01:
                    ax.text(j + 0.5, i + 0.5, '*', 
                            horizontalalignment='center', verticalalignment='center',
                            color='black', fontsize=12)
                # p < 0.01时添加**
                elif p_value_df.iloc[i, j] < 0.01 and p_value_df.iloc[i, j] >= 0.001:
                    ax.text(j + 0.5, i + 0.5, '**', 
                            horizontalalignment='center', verticalalignment='center',
                            color='black', fontsize=12)
                # p < 0.001时添加***
                elif p_value_df.iloc[i, j] < 0.001:
                    ax.text(j + 0.5, i + 0.5, '***', 
                            horizontalalignment='center', verticalalignment='center',
                            color='black', fontsize=12)
    
    plt.title(f'Diff of (Observed-Expected) with Significance - {comparison_name}')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'./plots/permutation_test/{analysis_type}_{cell_type}_{comparison_name}_with_significance.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def process_all_cell_types(analysis_type="coexistence"):
    """
    处理所有可用的细胞类型
    
    参数:
    analysis_type: 分析类型，例如"coexistence"
    """
    # 从文件名中提取所有可用的细胞类型
    all_files = glob.glob(f"./plots/*_{analysis_type}_expected_*.csv")
    
    if not all_files:
        print(f"警告: 找不到任何 {analysis_type} 矩阵文件")
        return
    
    # 从文件名中提取细胞类型
    cell_types = set()
    for file_path in all_files:
        file_name = os.path.basename(file_path)
        # 提取细胞类型（在"_expected_"和".csv"之间的部分）
        parts = file_name.split(f"_{analysis_type}_expected_")
        if len(parts) == 2:
            cell_type = parts[1].replace(".csv", "")
            cell_types.add(cell_type)
    
    print(f"找到的细胞类型: {cell_types}")
    
    # 处理每个细胞类型
    for cell_type in cell_types:
        print(f"处理细胞类型: {cell_type}")
        merge_observed_expected_with_pvalue(cell_type, analysis_type)

if __name__ == "__main__":
    # 默认处理所有细胞类型的"coexistence"分析
    process_all_cell_types("coexistence")
    
    # 如果有需要，也可以处理特定细胞类型
    # merge_observed_expected_with_pvalue("Guard_cell", "coexistence") 