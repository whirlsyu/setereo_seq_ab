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
    
    logger.info("开始基因共存分析...")
    
    if args.cell_type:
        logger.info(f"分析特定细胞类型: {args.cell_type}")
        success = merge_and_analyze_matrices(args.cell_type, args.analysis_type)
        if success:
            logger.info(f"成功完成细胞类型 {args.cell_type} 的分析！")
        else:
            logger.error(f"分析细胞类型 {args.cell_type} 时出错")
    else:
        logger.info(f"分析所有可用的细胞类型，分析类型: {args.analysis_type}")
        processed_types = process_all_cell_types(args.analysis_type)
        
        if processed_types:
            logger.info(f"成功分析的细胞类型: {', '.join(processed_types)}")
        else:
            logger.warning("没有成功分析任何细胞类型")
    
    logger.info("分析完成！")
    logger.info(f"结果保存在 ./results/permutation_test/ 目录下")

if __name__ == "__main__":
    main() 