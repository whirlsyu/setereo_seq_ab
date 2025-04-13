#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
合并每个细胞类型的所有expected和observed矩阵，
计算permutation test的P值并绘制热图
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_sample_id(filename):
    """
    从文件名中提取样本ID (SX-X)
    
    参数:
    filename: 文件名
    
    返回:
    样本ID，例如'S1-1'
    """
    match = re.search(r'(S\d+-\d+)', os.path.basename(filename))
    if match:
        return match.group(1)
    return None

def permutation_test(observed, expected, n_permutations=10000):
    """
    执行排列测试，比较观察值和期望值
    
    参数:
    observed: 观察到的矩阵
    expected: 期望的矩阵
    n_permutations: 排列次数
    
    返回:
    p_values: 每个元素的p值矩阵
    """
    # 确保矩阵形状一致
    assert observed.shape == expected.shape, "观察矩阵和期望矩阵必须具有相同的形状"
    
    # 计算实际差异（观察-期望）
    actual_diff = observed - expected
    
    # 初始化p值矩阵
    p_values = np.zeros_like(observed)
    
    # 获取非NaN位置的索引
    valid_indices = ~np.isnan(actual_diff)
    
    # 对非NaN位置执行排列测试
    for i in range(n_permutations):
        if i % 100 == 0:
            logger.info(f"正在执行排列测试: {i}/{n_permutations}")
        
        # 创建随机排列
        perm_matrix = np.random.permutation(observed[valid_indices])
        perm_diff = perm_matrix - expected[valid_indices]
        
        # 更新p值计数
        p_values[valid_indices] += (np.abs(perm_diff) >= np.abs(actual_diff[valid_indices])).astype(float)
    
    # 计算p值
    p_values[valid_indices] /= n_permutations
    
    return p_values
# 使用向量化方法优化对角线归零
def zero_diagonal(df):
    """将 DataFrame 的对角线元素设为 0（高效向量化版本）"""
    df_plot = df.copy()
    np.fill_diagonal(df_plot.values, 0)
    return df_plot
def plot_results(expected_avg, observed_avg, diff_matrix, p_values, output_dir, cell_type):
    """
    绘制热图结果
    
    参数:
    expected_avg: 期望值平均矩阵
    observed_avg: 观察到的平均矩阵
    diff_matrix: 差异矩阵（观察-期望）
    p_values: p值矩阵
    output_dir: 输出目录
    cell_type: 细胞类型
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存原始矩阵为CSV
    expected_avg.to_csv(os.path.join(output_dir, f"{cell_type}_expected_avg.csv"))
    observed_avg.to_csv(os.path.join(output_dir, f"{cell_type}_observed_avg.csv"))
    diff_matrix.to_csv(os.path.join(output_dir, f"{cell_type}_diff.csv"))
    pd.DataFrame(p_values, index=expected_avg.index, columns=expected_avg.columns).to_csv(
        os.path.join(output_dir, f"{cell_type}_pvalues.csv")
    )
    
    # 绘制热图
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # 处理并绘制所有矩阵
    expected_avg_plot = zero_diagonal(expected_avg)
    sns.heatmap(expected_avg_plot, ax=axes[0, 0], cmap="viridis", annot=False)
    axes[0, 0].set_title(f"Expected Co-occurrence Matrix - {cell_type}")

    observed_avg_plot = zero_diagonal(observed_avg)
    sns.heatmap(observed_avg_plot, ax=axes[0, 1], cmap="viridis", annot=False)
    axes[0, 1].set_title(f"Observed Co-occurrence Matrix - {cell_type}")

    diff_matrix_plot = zero_diagonal(diff_matrix)
    sns.heatmap(diff_matrix_plot, ax=axes[1, 0], cmap="coolwarm", center=0, annot=False)
    axes[1, 0].set_title(f"Difference (Observed - Expected) - {cell_type}")
        
    # 显著性热图 (p < 0.05标记为显著）
    # 确保significance矩阵具有与diff_matrix相同的行列名
    significance = pd.DataFrame((p_values < 0.05).astype(float), 
                                index=diff_matrix.index, 
                                columns=diff_matrix.columns)

    # 确保significance矩阵是对称的
    significance = (significance + significance.T) / 2
    
    # 将对角线元素设置为不显著（False）
    for i in range(len(significance.index)):
        if significance.index[i] in significance.columns:
            significance.iloc[i, i] = 0.0
    
    sns.heatmap(significance, ax=axes[1, 1], cmap="Reds", annot=False)
    axes[1, 1].set_title(f"Significance (p < 0.05) - {cell_type}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{cell_type}_heatmap.png"), dpi=300)
    plt.close()
    
    logger.info(f"successfully saved {os.path.join(output_dir, f'{cell_type}_heatmap.png')}")

def merge_and_analyze_matrices(cell_type, analysis_type="stereoseq_coexistence"):
    """
    合并并分析特定细胞类型的观察和期望矩阵
    
    参数:
    cell_type: 细胞类型
    analysis_type: 分析类型
    
    返回:
    成功时返回True，失败时返回False
    """
    # 设置输出目录
    output_dir = os.path.join("./results/permutation_test", cell_type)
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有期望和观察文件
    expected_pattern = f"*_{analysis_type}_expected_{cell_type}.csv"
    observed_pattern = f"*_{analysis_type}_observed_{cell_type}.csv"
    
    expected_files = glob.glob(os.path.join("./results", expected_pattern))
    observed_files = glob.glob(os.path.join("./results", observed_pattern))
    
    if not expected_files:
        logger.warning(f"can not found expected files: {expected_pattern}")
        return False
    
    if not observed_files:
        logger.warning(f"can not found observed files: {observed_pattern}")
        return False
    
    logger.info(f"find {len(expected_files)} expected files and {len(observed_files)} observed files")
    
    # 读取所有矩阵
    expected_matrices = []
    observed_matrices = []
    samples = []
    
    for expected_file in expected_files:
        sample_id = get_sample_id(expected_file)
        if not sample_id:
            logger.warning(f"can not extract sample ID from {expected_file} ")
            continue
        
        # 寻找匹配的观察文件
        observed_file = None
        for obs_file in observed_files:
            if sample_id in obs_file:
                observed_file = obs_file
                break
        
        if not observed_file:
            logger.warning(f"can not found {sample_id} in observed files")
            continue
        
        try:
            # 读取文件
            expected_df = pd.read_csv(expected_file, index_col=0)
            observed_df = pd.read_csv(observed_file, index_col=0)
            
            # 确保行列标签匹配
            common_rows = expected_df.index.intersection(observed_df.index)
            common_cols = expected_df.columns.intersection(observed_df.columns)
            
            if len(common_rows) == 0 or len(common_cols) == 0:
                logger.warning(f"sample {sample_id} has different row or column labels")
                continue
            
            expected_df = expected_df.loc[common_rows, common_cols]
            observed_df = observed_df.loc[common_rows, common_cols]
            
            expected_matrices.append(expected_df)
            observed_matrices.append(observed_df)
            samples.append(sample_id)
            
            logger.info(f"successfully loaded {sample_id} matrix (shape: {expected_df.shape})")
            
        except Exception as e:
            logger.error(f"precessing {sample_id} matrix failed: {str(e)}")
            continue
    
    if not expected_matrices or not observed_matrices:
        logger.error("no matrix was loaded")
        return False
    
    # 找到所有矩阵共享的基因
    common_genes = set(expected_matrices[0].index)
    for matrix in expected_matrices + observed_matrices:
        common_genes = common_genes.intersection(matrix.index)
    
    common_genes = sorted(list(common_genes))
    logger.info(f"found {len(common_genes)} common genes")
    
    if len(common_genes) == 0:
        logger.error("can not find any common genes")
        return False
    
    # 过滤每个矩阵，只包含共同基因
    for i in range(len(expected_matrices)):
        expected_matrices[i] = expected_matrices[i].loc[common_genes, common_genes]
        observed_matrices[i] = observed_matrices[i].loc[common_genes, common_genes]
    
    # 计算平均矩阵
    expected_avg = pd.concat(expected_matrices).groupby(level=0).mean()
    observed_avg = pd.concat(observed_matrices).groupby(level=0).mean()
    
    # 计算差异矩阵（观察-期望）
    diff_matrix = observed_avg - expected_avg
    
    logger.info("testing ...")
    # 执行排列测试
    p_values = permutation_test(observed_avg.values, expected_avg.values)
    
    # 绘制结果
    logger.info("heatmap ...")
    plot_results(expected_avg, observed_avg, diff_matrix, p_values, output_dir, cell_type)
    
    logger.info(f"completed processing {cell_type} ")
    return True

def process_all_cell_types(analysis_type="stereoseq_coexistence"):
    """
    处理所有可用的细胞类型
    
    参数:
    analysis_type: 分析类型
    
    返回:
    成功处理的细胞类型列表
    """
    # 查找所有可用的细胞类型
    pattern = f"*_{analysis_type}_expected_*.csv"
    expected_files = glob.glob(os.path.join("./results", pattern))
    
    cell_types = set()
    for file in expected_files:
        match = re.search(fr"{analysis_type}_expected_(.+?)\.csv", file)
        if match:
            cell_types.add(match.group(1))
    
    logger.info(f"found {len(cell_types)} cell types: {', '.join(cell_types)}")
    
    successful_types = []
    for cell_type in cell_types:
        logger.info(f"processing cell type: {cell_type}")
        success = merge_and_analyze_matrices(cell_type, analysis_type)
        if success:
            successful_types.append(cell_type)
    
    return successful_types

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="combine matrices and perform permutation test")
    parser.add_argument("--cell_type", help="要分析的细胞类型 (不指定则分析所有)")
    parser.add_argument("--analysis_type", default="stereoseq_coexistence", help="分析类型")
    
    args = parser.parse_args()
    
    if args.cell_type:
        logger.info(f"analyzing cell type: {args.cell_type}")
        merge_and_analyze_matrices(args.cell_type, args.analysis_type)
    else:
        logger.info(f"analyzing all cell types: {args.analysis_type}")
        process_all_cell_types(args.analysis_type)