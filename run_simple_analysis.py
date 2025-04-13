#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
运行细胞类型的observed和expected矩阵比较分析
将所有样本(S1-S4)合并成一个整体进行分析
"""

import os
import sys
import argparse
import logging
from merge_matrices_simple import merge_and_analyze_matrices, process_all_cell_types

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    主函数，处理命令行参数并运行分析
    """
    parser = argparse.ArgumentParser(description="分析基因共存数据")
    parser.add_argument("--cell-type", dest="cell_type", help="要分析的细胞类型，不指定则分析所有类型")
    parser.add_argument("--analysis-type", dest="analysis_type", default="stereoseq_coexistence", 
                        help="分析类型，默认为stereoseq_coexistence")
    
    args = parser.parse_args()
    
    # 确保结果目录存在
    os.makedirs("./results/permutation_test", exist_ok=True)
    
    logger.info("starting ...")
    
    if args.cell_type:
        logger.info(f"analyzing cell type: {args.cell_type}")
        success = merge_and_analyze_matrices(args.cell_type, args.analysis_type)
        if success:
            logger.info(f"successfully analyzed cell type: {args.cell_type} ")
        else:
            logger.error(f"failed to analyze cell type: {args.cell_type} ")
    else:
        logger.info(f"analyzing all cell types: {args.analysis_type}")
        processed_types = process_all_cell_types(args.analysis_type)
        
        if processed_types:
            logger.info(f"successfully analyzed cell types: {', '.join(processed_types)}")
        else:
            logger.warning("no cell types were analyzed")
    
    logger.info("completed.")
    logger.info(f"results are saved in ./results/permutation_test/ ")

if __name__ == "__main__":
    main() 