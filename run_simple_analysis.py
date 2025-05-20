#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run analysis of observed and expected matrices
all the samples are merged into one
"""

import os
import sys
import argparse
import logging
from merge_matrices_simple import merge_and_analyze_matrices, process_all_cell_types

# set log level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    main function command line interface
    """
    parser = argparse.ArgumentParser(description="analysis of observed and expected matrices")
    parser.add_argument("--cell-type", dest="cell_type", help="cell type to analyze")
    parser.add_argument("--analysis-type", dest="analysis_type", default="stereoseq_coexistence", 
                        help="analysis type: stereoseq_coexistence")
    
    args = parser.parse_args()
    
    # make output folder
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