#!/bin/bash

# 检查filtered_gene_results.tsv文件是否存在
if [ ! -f "filtered_gene_results.tsv" ]; then
    echo "Error: filtered_gene_results.tsv not found"
    exit 1
fi

# 遍历当前目录下的所有h5ad文件
for file in *.h5ad
do
    # 检查文件是否存在（以防止*.h5ad不匹配任何文件）
    if [ -f "$file" ]; then
        # 提取文件名（不包括扩展名）
        filename=$(basename "$file" .h5ad)
        
        # 创建结果文件夹
        #mkdir -p "./results_$filename"
        
        # 运行Python脚本
        echo "preprocessing file: $file"
        python analyze_spatial_data.py "$file" "./results"
        
        echo "finished processing file: $file"
        echo "results are stored in folder: ./results"
        echo "----------------------------"
    fi
done

echo "所有文件处理完毕"
