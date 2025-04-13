# Gene Co-occurrence Analysis

This project provides tools for analyzing gene co-occurrence patterns in different cell types using single-cell RNA sequencing data. It includes scripts for processing expected and observed gene co-occurrence matrices, performing statistical tests, and generating visualizations.

## Project Structure

```
.
├── merge_matrices_simple.py    # Core analysis functions
├── run_simple_analysis.py      # Command-line interface
├── coexistence_utils.py      # Core tool functions
└── analyze_spatial_data.py  # Analysis results for each cell type
```

## Features

- Process expected and observed gene co-occurrence matrices
- Calculate average matrices across multiple samples
- Perform permutation tests to assess statistical significance
- Generate comprehensive heatmap visualizations
- Support for multiple cell types and analysis methods

## Requirements

- Python 3.6+
- Required Python packages:
  - numpy
  - pandas
  - matplotlib
  - seaborn

## Usage

### Basic Analysis

To analyze all available cell types:
```bash
python run_simple_analysis.py
```

To analyze a specific cell type:
```bash
python run_simple_analysis.py --cell-type Guard_cell
```

### Analysis Types

The script supports different analysis types:
- `stereoseq_correlation` (default)
- `stereoseq_pearson`
- `stereoseq_spearman`

To specify an analysis type:
```bash
python run_simple_analysis.py --analysis-type stereoseq_pearson
```

## Output Files

For each cell type, the following files are generated in `./results/permutation_test/{cell_type}/`:

- `*_expected_avg.csv`: Average matrix of expected values
- `*_observed_avg.csv`: Average matrix of observed values
- `*_diff.csv`: Difference matrix (observed - expected)
- `*_pvalues.csv`: Matrix of p-values
- `*_heatmap.png`: Heatmap visualization containing four subplots:
  - Expected values
  - Observed values
  - Difference values
  - Statistical significance

## Notes

- The significance matrix is symmetric and diagonal elements are set to 0 (non-significant)
- The difference heatmap has diagonal elements set to 0 for better visualization
- Results are saved in CSV format for further analysis 
