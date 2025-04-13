import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 设置后端为 Agg
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300  # 设置全局 dpi

from coexistence_utils import calculate_coexistence_ratio, calculate_coexistence_ratio_probability
import seaborn as sns
import scanpy as sc
import anndata
import scipy.sparse as sp
from scipy import sparse

def calculate_cpm(adata):
    if not sparse.issparse(adata.X):
        adata.X = sparse.csr_matrix(adata.X)
    
    total_counts_per_cell = np.asarray(adata.X.sum(axis=1)).flatten()
    cpm = adata.X.copy()
    cpm = cpm.multiply(1e6 / total_counts_per_cell[:, np.newaxis])
    mean_cpm = np.asarray(cpm.mean(axis=0)).flatten()
    
    return pd.DataFrame({
        'gene': adata.var_names,
        'mean_cpm': mean_cpm
    }).sort_values('mean_cpm', ascending=False)

def calculate_counts(adata):
    if sparse.issparse(adata.X):
        counts = adata.X.toarray()
    else:
        counts = adata.X
    
    mean_counts = np.mean(counts, axis=0)
    
    return pd.DataFrame({
        'gene': adata.var_names,
        'mean_counts': mean_counts
    }).sort_values('mean_counts', ascending=False)


def filter_and_rename_genes(adata, filter_file):
    # 读取过滤文件
    filter_df = pd.read_csv(filter_file, sep='\t')
    
    # 创建 Locus 到 EXO70_info 的映射
    locus_to_exo70 = dict(zip(filter_df['Locus'], filter_df['EXO70_info']))
    
    # 获取在过滤文件中的基因
    genes_to_keep = [gene for gene in adata.var_names if gene in filter_df['Locus'].values]
    
    # 过滤 adata 对象
    adata_filtered = adata[:, genes_to_keep]
    
    # 重命名基因
    new_var_names = [locus_to_exo70.get(gene, gene) for gene in adata_filtered.var_names]
    adata_filtered.var_names = new_var_names
    adata_filtered.var_names_make_unique()
    
    return adata_filtered

def plot_spatial_distribution(adata, output_path, title):
    # 获取空间坐标的范围
    x_range = adata.obsm['spatial'][:, 0].max() - adata.obsm['spatial'][:, 0].min()
    y_range = adata.obsm['spatial'][:, 1].max() - adata.obsm['spatial'][:, 1].min()
    
    # 根据坐标范围决定图像方向
    if x_range > y_range:
        figsize = (8, 10)  # 竖直方向
        img_key = "hires"
    else:
        figsize = (10, 8)  # 水平方向
        img_key = None  # 不使用背景图像
    
    fig, ax = plt.subplots(figsize=figsize)
    sc.pl.spatial(adata, color="cell_type", img_key=img_key, spot_size=40, alpha=0.8, show=False, ax=ax)
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_spatial_distribution_filted_umap(adata, output_path, title):
    # 获取空间坐标的范围
    x_range = adata.obsm['spatial'][:, 0].max() - adata.obsm['spatial'][:, 0].min()
    y_range = adata.obsm['spatial'][:, 1].max() - adata.obsm['spatial'][:, 1].min()
    
    # 根据坐标范围决定图像方向
    if x_range > y_range:
        figsize = (8, 10)  # 竖直方向
        img_key = "hires"
    else:
        figsize = (10, 8)  # 水平方向
        img_key = None  # 不使用背景图像
    
    fig, ax = plt.subplots(figsize=figsize)
    # 绘制细胞位置图,使用Leiden聚类结果着色
    sc.pl.spatial(adata, 
                color="leiden",  # 使用Leiden聚类结果着色
                img_key=img_key,  # 使用高分辨率图像
                spot_size=40,    # 设置点的大小
                alpha=0.8,       # 设置点的透明度
                ax=ax,
                show=False,
                title=title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()




def analyze_spatial_data(input_file, output_folder, filter_file):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 读取数据
    adata = anndata.read_h5ad(input_file)
    
    # 基本信息
    print(f"细胞数量: {adata.n_obs}")
    print(f"基因数量: {adata.n_vars}")
    
    # 计算非零元素比例
    if sp.issparse(adata.X):
        non_zero_count = adata.X.nnz
        total_elements = adata.X.shape[0] * adata.X.shape[1]
    else:
        non_zero_count = np.count_nonzero(adata.X)
        total_elements = adata.X.size
    
    print(f"表达矩阵中的非零元素比例: {non_zero_count / total_elements:.2%}")
    
    # 空间分布图
    output_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_file))[0]}_spatial_distribution.png")
    title = f"spatial_distribution - {os.path.splitext(os.path.basename(input_file))[0]}"
    plot_spatial_distribution(adata, output_path, title)

    
    # UMAP图
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.tl.louvain(adata)
    sc.pl.umap(adata, color='louvain', title='UMAP', show=False)
    plt.savefig(os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_file))[0]}_umap.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # QC指标
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts'], jitter=0.4, multi_panel=True, show=False)
    plt.savefig(os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_file))[0]}_qc_violin.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts', show=False)
    plt.savefig(os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_file))[0]}_qc_scatter.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # CPM分析
    cpm_df = calculate_cpm(adata)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(cpm_df['mean_cpm'], bins=100, kde=True, log_scale=True)
    plt.title('Distribution of Mean CPM')
    plt.xlabel('Mean CPM (log scale)')
    plt.ylabel('Number of Genes')
    plt.savefig(os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_file))[0]}_cpm_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.ecdfplot(data=cpm_df, x='mean_cpm')
    plt.xscale('log')
    plt.title('Cumulative Distribution Function of Mean CPM')
    plt.xlabel('Mean CPM (log scale)')
    plt.ylabel('Cumulative Proportion')
    plt.savefig(os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_file))[0]}_cpm_ecdf.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 计数分析
    counts_df = calculate_counts(adata)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(counts_df['mean_counts'], bins=100, kde=True, log_scale=True)
    plt.title('Distribution of Mean Counts')
    plt.xlabel('Mean Counts (log scale)')
    plt.ylabel('Number of Genes')
    plt.savefig(os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_file))[0]}_counts_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.ecdfplot(data=counts_df, x='mean_counts')
    plt.xscale('log')
    plt.title('Cumulative Distribution Function of Mean Counts')
    plt.xlabel('Mean Counts (log scale)')
    plt.ylabel('Cumulative Proportion')
    plt.savefig(os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_file))[0]}_counts_ecdf.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 过滤基因
    adata_filtered = filter_and_rename_genes(adata, filter_file)

    # UMAP图
    sc.pp.neighbors(adata_filtered)
    sc.tl.umap(adata_filtered)
    sc.tl.leiden(adata_filtered)
    sc.pl.umap(adata_filtered, color='leiden', title='UMAP', show=False)
    plt.savefig(os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_file))[0]}_filtered_umap.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    output_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_file))[0]}_filtered_umap_spatial_distribution.png")
    title = f"filtered_umap_spatial_distribution - {os.path.splitext(os.path.basename(input_file))[0]}"
    plot_spatial_distribution_filted_umap(adata_filtered, output_path, title)    
    # 基因相关性分析
    cell_types = adata_filtered.obs['cell_type'].unique()
    for cell_type in cell_types:
        cell_type_data = adata_filtered[adata_filtered.obs['cell_type'] == cell_type].X
        corr = pd.DataFrame(data=cell_type_data.toarray() if sp.issparse(cell_type_data) else cell_type_data, 
                            columns=adata_filtered.var_names).corr(method='spearman')
        
        # 按字母顺序排序基因名
        sorted_genes = sorted(corr.columns)
        corr = corr.loc[sorted_genes, sorted_genes]
        # 保存相关性数据
        corr.to_csv(os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_file))[0]}_correlation_{cell_type}.csv"))
         
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, cmap='coolwarm', vmin=-1, vmax=1, xticklabels=True, yticklabels=True)
        plt.title(f'Gene Correlation Heatmap - {cell_type}')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_file))[0]}_correlation_{cell_type}.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # 基因共存分析
    for cell_type in cell_types:
        cell_type_data = adata_filtered[adata_filtered.obs['cell_type'] == cell_type].X
        ratio = calculate_coexistence_ratio(cell_type_data)
        
        if ratio.size == 0:
            print(f"警告: {cell_type} 的数据为空，跳过此细胞类型")
            continue
        
        # 将比率转换为DataFrame并按字母顺序排序基因名
        ratio_df = pd.DataFrame(ratio, index=adata_filtered.var_names, columns=adata_filtered.var_names)
        sorted_genes = sorted(ratio_df.columns)
        ratio_df = ratio_df.loc[sorted_genes, sorted_genes]
        # 保存共存比率数据
        ratio_df.to_csv(os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_file))[0]}_coexistence_{cell_type}.csv"))
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(ratio_df, cmap='YlOrRd', vmin=0, vmax=1, 
                    xticklabels=True, yticklabels=True)
        plt.title(f'Gene Coexistence Ratio Heatmap - {cell_type}')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_file))[0]}_coexistence_{cell_type}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
    # 基因共存概率分析
    for cell_type in cell_types:
        cell_type_data = adata_filtered[adata_filtered.obs['cell_type'] == cell_type].X
        ratio, observed, expected  = calculate_coexistence_ratio_probability(cell_type_data)
        
        if ratio.size == 0:
            print(f"警告: {cell_type} 的数据为空，跳过此细胞类型")
            continue
        
        # 将比率转换为DataFrame并按字母顺序排序基因名
        ratio_df = pd.DataFrame(ratio, index=adata_filtered.var_names, columns=adata_filtered.var_names)
        
        observed_df = pd.DataFrame(observed, index=adata_filtered.var_names, columns=adata_filtered.var_names)
        expected_df = pd.DataFrame(expected, index=adata_filtered.var_names, columns=adata_filtered.var_names)
        
        sorted_genes = sorted(ratio_df.columns)
        ratio_df = ratio_df.loc[sorted_genes, sorted_genes]
        # 保存共存比率数据
        ratio_df.to_csv(os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_file))[0]}_coexistence_probability_{cell_type}.csv"))
        observed_df.to_csv(os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_file))[0]}_coexistence_observed_{cell_type}.csv"))
        expected_df.to_csv(os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_file))[0]}_coexistence_expected_{cell_type}.csv"))
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(ratio_df, cmap='YlOrRd', vmin=ratio_df.min().min()-0.1, vmax=ratio_df.max().max()+0.1, 
                    xticklabels=True, yticklabels=True)
        plt.title(f'Gene Coexistence Probability Heatmap - {cell_type}')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_file))[0]}_coexistence_probability_{cell_type}.png"), dpi=300, bbox_inches='tight')
        plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze spatial transcriptomics data")
    parser.add_argument("input_file", help="Path to the input h5ad file")
    parser.add_argument("output_folder", help="Path to the output folder")
    parser.add_argument("--filter_file", help="Path to the filter file", default='filtered_gene_results.tsv')
    args = parser.parse_args()
    
    analyze_spatial_data(args.input_file, args.output_folder, args.filter_file)