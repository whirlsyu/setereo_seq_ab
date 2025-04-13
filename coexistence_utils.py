# coexistence_utils.py

import numpy as np
import scipy.sparse as sp

def calculate_coexistence_ratio(data, epsilon=1e-10):
    print(f"Input data type: {type(data)}")
    print(f"Input data shape: {data.shape}")
    
    if sp.issparse(data):
        binary_data = data.astype(bool).astype(int)
    else:
        binary_data = (data > 0).astype(int)
    
    print(f"Binary data shape: {binary_data.shape}")
    
    if binary_data.shape[1] == 1:
        # 如果只有一个基因,返回1x1的矩阵
        return np.array([[1.0]])
    
    if sp.issparse(binary_data):
        coexistence_matrix = binary_data.T @ binary_data
        gene_sums = binary_data.sum(axis=0).A1
    else:
        coexistence_matrix = np.dot(binary_data.T, binary_data)
        gene_sums = binary_data.sum(axis=0)
    
    print(f"Coexistence matrix shape: {coexistence_matrix.shape}")
    
    # 计算分母矩阵
    denominator_matrix = gene_sums[:, np.newaxis] + gene_sums[np.newaxis, :] - coexistence_matrix
    
    # 避免除以零
    denominator_matrix[denominator_matrix == 0] = epsilon
    
    result = coexistence_matrix / denominator_matrix
    
    print(f"Result shape: {result.shape}")
    print(f"Result type: {type(result)}")
    
    if sp.issparse(result):
        result = result.toarray()
    
    return result

# 修改计算共存比率的函数
def calculate_coexistence_ratio_all_as_base(data, epsilon=1e-10):
    print(f"Input data type: {type(data)}")
    print(f"Input data shape: {data.shape}")
    
    if sp.issparse(data):
        coexistence = (data > 0).astype(int)
        print(f"Coexistence shape: {coexistence.shape}")
        coexistence_matrix = coexistence.T @ coexistence
    else:
        coexistence = (data > 0).astype(int)
        print(f"Coexistence shape: {coexistence.shape}")
        coexistence_matrix = np.dot(coexistence.T, coexistence)
    
    print(f"Coexistence matrix shape: {coexistence_matrix.shape}")
    
    total_cells = data.shape[0]
    print(f"Total cells: {total_cells}")
    
    result = coexistence_matrix / (total_cells + epsilon)
    print(f"Result shape: {result.shape}")
    print(f"Result type: {type(result)}")
    
    if result.ndim != 2:
        print(f"Warning: Resulting matrix is not 2D. Shape: {result.shape}")
        if result.size == 1:
            result = np.array([[result]])
        else:
            result = np.atleast_2d(result)
    
    return result


def calculate_coexistence_ratio_probability(data, epsilon=1e-10):
    print(f"Input data type: {type(data)}")
    print(f"Input data shape: {data.shape}")
    
    if sp.issparse(data):
        binary_data = data.astype(bool).astype(int)
    else:
        binary_data = (data > 0).astype(int)
    
    print(f"Binary data shape: {binary_data.shape}")
    
    if binary_data.shape[1] == 1:
        return np.array([[1.0]])
    
    total_cells = binary_data.shape[0]  # 计算总细胞数
    coexistence_matrix = binary_data.T @ binary_data
    
    # 计算基因表达比率（基因1和基因2有表达的细胞占总细胞的百分比）
    gene_expression_ratios = binary_data.sum(axis=0) / total_cells  # 每个基因的表达比率
    
    # 确保 gene_expression_ratios 是一维数组
    gene_expression_ratios = np.asarray(gene_expression_ratios).flatten()

    # 计算共存比率
    
    expected_data = (gene_expression_ratios[:, np.newaxis] * gene_expression_ratios[np.newaxis, :])
    observed_data = (coexistence_matrix / total_cells)
    result = observed_data - expected_data
    
    print(f"Result shape: {result.shape}")
    print(f"Result type: {type(result)}")
    
    if sp.issparse(result):
        result = result.toarray()
    if sp.issparse(expected_data):
        expected_data = expected_data.toarray()
    if sp.issparse(observed_data):
        observed_data = observed_data.toarray()
    
    return (result,observed_data,expected_data)

def permutation_test(observed_data, expected_data, n_permutations=10000):
    # 计算观察值的差异
    observed_diff = np.mean(observed_data) - np.mean(expected_data)
    
    # 合并数据
    combined_data = np.concatenate([observed_data, expected_data])
    
    # 随机重排并计算差异
    perm_diffs = []
    for _ in range(n_permutations):
        np.random.shuffle(combined_data)  # 随机重排
        perm_observed = combined_data[:len(observed_data)]
        perm_expected = combined_data[len(observed_data):]
        perm_diff = np.mean(perm_observed) - np.mean(perm_expected)
        perm_diffs.append(perm_diff)
    
    # 计算 P 值
    p_value = (np.abs(np.array(perm_diffs)) >= np.abs(observed_diff)).mean()
    
    return observed_diff, p_value

def element_wise_permutation_test(observed_matrix, expected_matrix, n_permutations=1000):
    """
    对观察矩阵和期望矩阵的每个元素进行位置对应的置换检验
    
    参数:
    observed_matrix: 观察到的共存矩阵，通常是from calculate_coexistence_ratio_probability返回的observed_data
    expected_matrix: 期望的共存矩阵，通常是from calculate_coexistence_ratio_probability返回的expected_data
    n_permutations: 置换次数，默认为1000次
    
    返回:
    p_value_matrix: 每个位置的p值矩阵
    diff_matrix: 观察值与期望值之间的差异矩阵
    """
    # 确保矩阵形状相同
    assert observed_matrix.shape == expected_matrix.shape, "观察矩阵和期望矩阵形状必须相同"
    
    rows, cols = observed_matrix.shape
    p_value_matrix = np.zeros((rows, cols))
    diff_matrix = observed_matrix - expected_matrix
    
    # 对每个元素进行置换检验
    for i in range(rows):
        for j in range(cols):
            observed_val = observed_matrix[i, j]
            expected_val = expected_matrix[i, j]
            
            # 如果观察值和期望值相同，则p值为1
            if observed_val == expected_val:
                p_value_matrix[i, j] = 1.0
                continue
            
            # 执行置换检验
            observed_diff = observed_val - expected_val
            
            # 将两个值放入数组中进行置换
            values = np.array([observed_val, expected_val])
            
            # 随机重排并计算差异
            perm_diffs = []
            for _ in range(n_permutations):
                np.random.shuffle(values)  # 随机重排
                perm_diff = values[0] - values[1]
                perm_diffs.append(perm_diff)
            
            # 计算 P 值
            p_value = np.mean(np.abs(np.array(perm_diffs)) >= np.abs(observed_diff))
            p_value_matrix[i, j] = p_value
    
    return p_value_matrix, diff_matrix

def merge_matrices_with_pvalue(observed_matrices_list, expected_matrices_list, gene_names, cell_type="Unknown", n_permutations=1000):
    """
    合并多个观察矩阵和期望矩阵，并计算每个位置的p值
    
    参数:
    observed_matrices_list: 观察矩阵列表，每个元素是一个n×n的numpy数组
    expected_matrices_list: 期望矩阵列表，每个元素是一个n×n的numpy数组
    gene_names: 基因名称列表，长度为n
    cell_type: 细胞类型名称，用于输出信息
    n_permutations: 置换检验的次数，默认为1000
    
    返回:
    merged_result: 包含合并后的观察矩阵、期望矩阵、差异矩阵和p值矩阵的字典
    """
    print(f"处理细胞类型: {cell_type}")
    
    # 检查输入矩阵列表的长度是否相同
    if len(observed_matrices_list) != len(expected_matrices_list):
        raise ValueError("观察矩阵列表和期望矩阵列表长度必须相同")
    
    if len(observed_matrices_list) == 0:
        raise ValueError("矩阵列表不能为空")
    
    # 检查所有矩阵的形状是否相同
    matrix_shape = observed_matrices_list[0].shape
    for obs_mat, exp_mat in zip(observed_matrices_list, expected_matrices_list):
        if obs_mat.shape != matrix_shape or exp_mat.shape != matrix_shape:
            raise ValueError("所有矩阵的形状必须相同")
    
    # 初始化合并后的矩阵
    n_matrices = len(observed_matrices_list)
    merged_observed = np.zeros(matrix_shape)
    merged_expected = np.zeros(matrix_shape)
    
    # 合并矩阵
    for obs_mat, exp_mat in zip(observed_matrices_list, expected_matrices_list):
        merged_observed += obs_mat
        merged_expected += exp_mat
    
    # 计算平均值
    merged_observed /= n_matrices
    merged_expected /= n_matrices
    
    print(f"已合并 {n_matrices} 个矩阵")
    print(f"合并后的矩阵形状: {merged_observed.shape}")
    
    # 对每个矩阵位置执行置换检验
    p_value_matrix, diff_matrix = element_wise_permutation_test(
        merged_observed, merged_expected, n_permutations
    )
    
    # 创建结果字典
    merged_result = {
        'observed': merged_observed,
        'expected': merged_expected,
        'diff': diff_matrix,
        'p_value': p_value_matrix,
        'gene_names': gene_names,
        'cell_type': cell_type
    }
    
    return merged_result

