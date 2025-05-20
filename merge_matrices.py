#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
merge all matrices and perform permutation test
calculate p-value and plot heatmap
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
    obtain sample ID (SX-X)
    
    params:
    filename: file name
    
    return:
    sample id, for instance:'S1-1'
    """
    match = re.match(r'(S\d+-\d+)_', os.path.basename(filename))
    if match:
        return match.group(1)
    return None

def merge_observed_expected_with_pvalue(cell_type, analysis_type="coexistence"):
    """
    analyze observed and expected matrices and calculate p-value for each sample
    
    params:
    cell_type: cell type, for instance: 'Bcell'
    analysis_type: analysis type, for instance:"coexistence"
    """
    print(f"starting to analyze cell type: {cell_type}, analysis type: {analysis_type}")
    
    # get all the expected and observed matrices
    expected_files = glob.glob(f"./results/*_{analysis_type}_expected_{cell_type}.csv")
    observed_files = glob.glob(f"./results/*_{analysis_type}_observed_{cell_type}.csv")
    
    if not expected_files or not observed_files:
        print(f"warning: can not find any {cell_type} or {analysis_type} files")
        return None
    
    # group files by sample ID
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
    
    # check sample IDs
    all_sample_ids = set(expected_by_sample.keys()) | set(observed_by_sample.keys())
    print(f"finished checking sample IDs, all sample IDs: {all_sample_ids}")
    
    # reading matrices
    sample_data = {}
    gene_names = None
    
    for sample_id in all_sample_ids:
        expected_sample_files = expected_by_sample.get(sample_id, [])
        observed_sample_files = observed_by_sample.get(sample_id, [])
        
        if not expected_sample_files or not observed_sample_files:
            print(f"warning: sample {sample_id} has no expected or observed files, skip")
            continue
        
        # make sure the files are sorted as same as expected
        expected_sample_files.sort()
        observed_sample_files.sort()
        
        # read matrices
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
        
        # calculate the matrices
        if expected_matrices and observed_matrices:
            # 确保所有矩阵形状相同
            matrix_shape = expected_matrices[0].shape
            for mat in expected_matrices + observed_matrices:
                if mat.shape != matrix_shape:
                    print(f"warning: sample {sample_id} has inconsistent matrix shape, skip")
                    continue
            
            # get the average matrices
            expected_avg = np.mean(expected_matrices, axis=0)
            observed_avg = np.mean(observed_matrices, axis=0)
            
            sample_data[sample_id] = {
                'expected': expected_avg,
                'observed': observed_avg,
                'gene_names': gene_names
            }
    
    # perform permutation test
    if len(sample_data) < 2:
        print(f"warning: cell type:{cell_type} only have {len(sample_data)} samples, at least 2 samples are required, skip")
        return None
    
    # create output folder
    os.makedirs("./permutation_test", exist_ok=True)
    
    # do the permutation test
    comparison_results = {}
    
    sample_ids = list(sample_data.keys())
    for i in range(len(sample_ids)):
        for j in range(i+1, len(sample_ids)):
            sample1 = sample_ids[i]
            sample2 = sample_ids[j]
            
            print(f"comparing {sample1} and {sample2}")
            
            # get the observation and expected matrices
            observed1 = sample_data[sample1]['observed']
            expected1 = sample_data[sample1]['expected']
            observed2 = sample_data[sample2]['observed']
            expected2 = sample_data[sample2]['expected']
            gene_names = sample_data[sample1]['gene_names']
            
            # calculate the difference between observed and expected
            observed_diff = observed1 - observed2
            expected_diff = expected1 - expected2
            
            # calculate the difference between observed and expected
            o_e_diff1 = observed1 - expected1
            o_e_diff2 = observed2 - expected2
            diff_of_diff = o_e_diff1 - o_e_diff2
            
            # calculate the p-value matrix
            p_value_matrix, _ = element_wise_permutation_test(
                o_e_diff1, o_e_diff2, n_permutations=1000
            )
            
            # save the results
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
            
            # plot and save the results
            plot_comparison_results(
                comparison_results[comparison_name], 
                cell_type, 
                analysis_type, 
                comparison_name
            )
    
    return comparison_results

def plot_comparison_results(results, cell_type, analysis_type, comparison_name):
    """
    plot the comparison results
    
    params:
    results: the dict of comparison results
    cell_type: cell type, for instance: 'Guard cell'
    analysis_type: analysis type, for instance:"coexistence"
    comparison_name: comparison name, for instance:"S1-1_vs_S2-3"
    """
    
    observed_diff = results['observed_diff']
    expected_diff = results['expected_diff']
    diff_of_diff = results['diff_of_diff']
    p_value_matrix = results['p_value']
    gene_names = results['gene_names']
    
    # convert to data frame
    observed_diff_df = pd.DataFrame(observed_diff, index=gene_names, columns=gene_names)
    expected_diff_df = pd.DataFrame(expected_diff, index=gene_names, columns=gene_names)
    diff_of_diff_df = pd.DataFrame(diff_of_diff, index=gene_names, columns=gene_names)
    p_value_df = pd.DataFrame(p_value_matrix, index=gene_names, columns=gene_names)
    
    # create a figure
    plt.figure(figsize=(20, 15))
    
    # four subplots
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
    # using log for p-value
    log_p_value_df = -np.log10(p_value_df.clip(lower=1e-10))
    sns.heatmap(log_p_value_df, cmap='viridis', xticklabels=True, yticklabels=True)
    plt.title(f'Significance (-log10 P-value) - {comparison_name}')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(f'./plots/permutation_test/{analysis_type}_{cell_type}_{comparison_name}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # save the results
    observed_diff_df.to_csv(f'./plots/permutation_test/{analysis_type}_{cell_type}_{comparison_name}_observed_diff.csv')
    expected_diff_df.to_csv(f'./plots/permutation_test/{analysis_type}_{cell_type}_{comparison_name}_expected_diff.csv')
    diff_of_diff_df.to_csv(f'./plots/permutation_test/{analysis_type}_{cell_type}_{comparison_name}_diff_of_diff.csv')
    p_value_df.to_csv(f'./plots/permutation_test/{analysis_type}_{cell_type}_{comparison_name}_pvalue.csv')
    
    # create a heatmap
    plt.figure(figsize=(14, 12))
    
    # mark the significant genes
    significant_mask = p_value_df < 0.05
    
    # plot the heatmap
    ax = sns.heatmap(diff_of_diff_df, cmap='coolwarm', center=0, 
                     xticklabels=True, yticklabels=True, 
                     annot=False, fmt='.2f')
    
    # mark the significant genes
    for i in range(len(gene_names)):
        for j in range(len(gene_names)):
            if significant_mask.iloc[i, j]:
                # add * for p < 0.05
                if p_value_df.iloc[i, j] < 0.05 and p_value_df.iloc[i, j] >= 0.01:
                    ax.text(j + 0.5, i + 0.5, '*', 
                            horizontalalignment='center', verticalalignment='center',
                            color='black', fontsize=12)
                # add ** for p < 0.01
                elif p_value_df.iloc[i, j] < 0.01 and p_value_df.iloc[i, j] >= 0.001:
                    ax.text(j + 0.5, i + 0.5, '**', 
                            horizontalalignment='center', verticalalignment='center',
                            color='black', fontsize=12)
                # add *** for p < 0.001
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
    process all cell types
    
    param:
    analysis_type: analysis type, for instance:"coexistence"
    """
    # get all the csv files
    all_files = glob.glob(f"./plots/*_{analysis_type}_expected_*.csv")
    
    if not all_files:
        print(f"warning: can not find any {analysis_type} files")
        return
    
    # subset the files by cell type
    cell_types = set()
    for file_path in all_files:
        file_name = os.path.basename(file_path)
        # get cell type by file name
        parts = file_name.split(f"_{analysis_type}_expected_")
        if len(parts) == 2:
            cell_type = parts[1].replace(".csv", "")
            cell_types.add(cell_type)
    
    print(f"finding cell types: {cell_types}")
    
    # process each cell type
    for cell_type in cell_types:
        print(f"processing cell type: {cell_type}")
        merge_observed_expected_with_pvalue(cell_type, analysis_type)

if __name__ == "__main__":
    # by default, process all cell types
    process_all_cell_types("coexistence")
    
    # if you want to process a specific cell type
    # merge_observed_expected_with_pvalue("Guard_cell", "coexistence") 