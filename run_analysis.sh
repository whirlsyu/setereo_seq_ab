#!/bin/bash

# check filtered_gene_results.tsv 
if [ ! -f "filtered_gene_results.tsv" ]; then
    echo "Error: filtered_gene_results.tsv not found"
    exit 1
fi

# iterates through all the h5ad files in the current directory
for file in *.h5ad
do
    # checks if the file exists
    if [ -f "$file" ]; then
        # get the filename not including the extension
        filename=$(basename "$file" .h5ad)
        
        # make a new directory for the results optional
        #mkdir -p "./results_$filename"
        
        # run the analysis
        echo "preprocessing file: $file"
        python analyze_spatial_data.py "$file" "./results"
        
        echo "finished processing file: $file"
        echo "results are stored in folder: ./results"
        echo "----------------------------"
    fi
done

echo "all files processed"
