#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
analyze observed and expected co-existence matrices for cell types and perform permutation test
"""

import sys
import argparse
from merge_matrices import merge_observed_expected_with_pvalue, process_all_cell_types

def main():
    parser = argparse.ArgumentParser(description="analyze observed and expected matrices for cell types and perform permutation test")
    parser.add_argument("--cell-type", type=str, help="cell type to analyze, default is all")
    parser.add_argument("--analysis-type", type=str, default="coexistence", 
                        help="cell type to analyze, default is'coexistence'")
    
    args = parser.parse_args()
    
    if args.cell_type:
        print(f"start analyzing cell type: {args.cell_type}")
        merge_observed_expected_with_pvalue(args.cell_type, args.analysis_type)
    else:
        print("start analyzing all cell types")
        process_all_cell_types(args.x)
    
    print("completed.")

if __name__ == "__main__":
    main() 