#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
运行细胞类型间共存矩阵的置换检验分析
"""

import sys
import argparse
from merge_matrices import merge_observed_expected_with_pvalue, process_all_cell_types

def main():
    parser = argparse.ArgumentParser(description="对细胞类型的共存矩阵进行置换检验分析")
    parser.add_argument("--cell-type", type=str, help="要分析的细胞类型，不指定则分析所有细胞类型")
    parser.add_argument("--analysis-type", type=str, default="coexistence", 
                        help="分析类型，默认为'coexistence'")
    
    args = parser.parse_args()
    
    if args.cell_type:
        print(f"开始分析细胞类型: {args.cell_type}")
        merge_observed_expected_with_pvalue(args.cell_type, args.analysis_type)
    else:
        print("开始分析所有细胞类型")
        process_all_cell_types(args.x)
    
    print("分析完成！")

if __name__ == "__main__":
    main() 