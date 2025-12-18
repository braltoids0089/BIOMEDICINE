# Single-Cell Transcriptomics Notebook Improvements

## Overview
This document outlines improvements to the single-cell transcriptomics notebook, organized by implementation phase.

---

## Phase 1: Data Loading and Preprocessing

### Objective
Enhance data loading efficiency and improve preprocessing reproducibility.

### Code Examples

#### 1.1 Optimized Data Loading
```python
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path

# Define data paths
data_dir = Path("data/single_cell")
sample_files = list(data_dir.glob("*.h5ad"))

# Efficient batch loading with lazy evaluation
def load_single_cell_data(file_path, cache=True):
    """
    Load single-cell data with caching support.
    
    Parameters:
    -----------
    file_path : str or Path
        Path to the h5ad file
    cache : bool
        Whether to cache the loaded data
        
    Returns:
    --------
    adata : AnnData object
    """
    adata = sc.read_h5ad(file_path, backed='r' if not cache else None)
    return adata

# Load multiple samples
samples = {f.stem: load_single_cell_data(f) for f in sample_files[:3]}
```

#### 1.2 Quality Control Pipeline
```python
def quality_control(adata, min_genes=200, min_cells=3, max_mt=20):
    """
    Perform comprehensive quality control.
    
    Parameters:
    -----------
    adata : AnnData object
    min_genes : int
        Minimum number of genes expressed per cell
    min_cells : int
        Minimum number of cells expressing each gene
    max_mt : float
        Maximum percentage of mitochondrial genes
        
    Returns:
    --------
    adata : filtered AnnData object
    """
    # Calculate metrics
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)
    
    # Filter cells
    adata = adata[
        (adata.obs['n_genes_by_counts'] > min_genes) & 
        (adata.obs['pct_counts_mt'] < max_mt)
    ]
    
    # Filter genes
    adata = adata[:, (adata.var['n_cells_by_counts'] >= min_cells)]
    
    return adata

# Apply QC
adata_qc = quality_control(samples[list(samples.keys())[0]])
print(f"Cells after QC: {adata_qc.n_obs}")
print(f"Genes after QC: {adata_qc.n_vars}")
```

---

## Phase 2: Normalization and Feature Selection

### Objective
Implement robust normalization and improved feature selection methods.

### Code Examples

#### 2.1 Advanced Normalization
```python
def normalize_and_log_transform(adata, target_sum=1e4, layer='X'):
    """
    Perform library size normalization and log transformation.
    
    Parameters:
    -----------
    adata : AnnData object
    target_sum : float
        Target library size for normalization
    layer : str
        Layer to normalize
        
    Returns:
    --------
    adata : normalized AnnData object
    """
    # Library size normalization
    sc.pp.normalize_total(adata, target_sum=target_sum)
    
    # Log transformation
    sc.pp.log1p(adata)
    
    # Store raw counts in separate layer
    adata.layers['raw_counts'] = adata.X.copy() if not isinstance(adata.X, np.ndarray) else adata.X
    
    return adata

# Apply normalization
adata_norm = normalize_and_log_transform(adata_qc)
```

#### 2.2 Enhanced Feature Selection
```python
def feature_selection(adata, method='highly_variable', n_top_genes=2000):
    """
    Select informative genes using multiple methods.
    
    Parameters:
    -----------
    adata : AnnData object
    method : str
        'highly_variable' or 'hvg_batch'
    n_top_genes : int
        Number of features to select
        
    Returns:
    --------
    adata : AnnData object with selected features
    """
    if method == 'highly_variable':
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_top_genes,
            flavor='seurat_v3'
        )
    elif method == 'hvg_batch':
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_top_genes,
            flavor='seurat_v3',
            batch_key='batch'  # Adjust batch key as needed
        )
    
    # Log HVG statistics
    hvg_count = adata.var['highly_variable'].sum()
    print(f"Selected {hvg_count} highly variable genes")
    
    return adata

# Apply feature selection
adata_hvg = feature_selection(adata_norm, method='highly_variable', n_top_genes=2000)
adata_hvg = adata_hvg[:, adata_hvg.var['highly_variable']]
```

---

## Phase 3: Dimensionality Reduction

### Objective
Implement efficient dimensionality reduction with validation metrics.

### Code Examples

#### 3.1 PCA with Variance Explained
```python
def compute_pca(adata, n_comps=50, random_state=42):
    """
    Compute PCA with detailed variance analysis.
    
    Parameters:
    -----------
    adata : AnnData object
    n_comps : int
        Number of PCs to compute
    random_state : int
        Random seed
        
    Returns:
    --------
    adata : AnnData object with PCA results
    """
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=n_comps, random_state=random_state)
    
    # Calculate cumulative variance explained
    cum_var = np.cumsum(adata.uns['pca']['variance_ratio'])
    n_comps_95 = np.argmax(cum_var >= 0.95) + 1
    
    print(f"PCs needed for 95% variance: {n_comps_95}")
    print(f"Total variance explained by top 50 PCs: {cum_var[49]:.2%}")
    
    return adata

# Apply PCA
adata_pca = compute_pca(adata_hvg, n_comps=50)
```

#### 3.2 UMAP Projection
```python
def compute_umap(adata, n_neighbors=15, min_dist=0.1, random_state=42):
    """
    Compute UMAP with optimized parameters.
    
    Parameters:
    -----------
    adata : AnnData object with PCA
    n_neighbors : int
        Number of neighbors for UMAP
    min_dist : float
        Minimum distance for UMAP
    random_state : int
        Random seed
        
    Returns:
    --------
    adata : AnnData object with UMAP coordinates
    """
    # Compute neighborhood graph
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X_pca')
    
    # Compute UMAP
    sc.tl.umap(adata, min_dist=min_dist, random_state=random_state)
    
    return adata

# Apply UMAP
adata_umap = compute_umap(adata_pca, n_neighbors=15, min_dist=0.1)
```

---

## Phase 4: Clustering and Annotation

### Objective
Implement robust clustering with multiple resolution validation.

### Code Examples

#### 4.1 Multi-Resolution Clustering
```python
def cluster_leiden(adata, resolutions=[0.5, 1.0, 1.5], random_state=42):
    """
    Perform Leiden clustering at multiple resolutions.
    
    Parameters:
    -----------
    adata : AnnData object
    resolutions : list
        Resolution parameters to test
    random_state : int
        Random seed
        
    Returns:
    --------
    adata : AnnData object with clustering results
    """
    for res in resolutions:
        sc.tl.leiden(
            adata,
            resolution=res,
            random_state=random_state,
            key_added=f'leiden_{res}'
        )
        
        n_clusters = len(adata.obs[f'leiden_{res}'].unique())
        print(f"Resolution {res}: {n_clusters} clusters")
    
    return adata

# Apply clustering
adata_clustered = cluster_leiden(adata_umap, resolutions=[0.5, 1.0, 1.5])
```

#### 4.2 Cell Type Annotation
```python
def annotate_cell_types(adata, marker_genes_dict, method='score'):
    """
    Annotate cell types using marker genes.
    
    Parameters:
    -----------
    adata : AnnData object
    marker_genes_dict : dict
        Dictionary of cell types and their marker genes
    method : str
        'score' or 'expression_threshold'
        
    Returns:
    --------
    adata : AnnData object with annotations
    """
    if method == 'score':
        for cell_type, genes in marker_genes_dict.items():
            # Filter genes present in the dataset
            genes_present = [g for g in genes if g in adata.var_names]
            
            if genes_present:
                sc.tl.score_genes(adata, genes_present, score_name=f'{cell_type}_score')
    
    return adata

# Define marker genes
marker_genes = {
    'T_cells': ['CD3D', 'CD3E', 'CD3G'],
    'B_cells': ['CD19', 'MS4A1', 'CD79A'],
    'NK_cells': ['NKG7', 'GNLY', 'FGFBP2'],
    'Macrophages': ['CD14', 'LYZ', 'CST3']
}

# Annotate
adata_annotated = annotate_cell_types(adata_clustered, marker_genes)
```

---

## Phase 5: Visualization and Quality Metrics

### Objective
Generate comprehensive visualizations and quality control metrics.

### Code Examples

#### 5.1 Quality Metrics Report
```python
def generate_qc_report(adata):
    """
    Generate comprehensive QC metrics report.
    
    Parameters:
    -----------
    adata : AnnData object
    
    Returns:
    --------
    report : dict with QC metrics
    """
    report = {
        'n_cells': adata.n_obs,
        'n_genes': adata.n_vars,
        'mean_reads_per_cell': adata.obs['n_counts'].mean(),
        'median_reads_per_cell': adata.obs['n_counts'].median(),
        'mean_genes_per_cell': adata.obs['n_genes_by_counts'].mean(),
        'median_genes_per_cell': adata.obs['n_genes_by_counts'].median(),
        'n_clusters': len(adata.obs['leiden_1.0'].unique()) if 'leiden_1.0' in adata.obs else 0
    }
    
    print("=" * 50)
    print("QUALITY CONTROL REPORT")
    print("=" * 50)
    for key, value in report.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    print("=" * 50)
    
    return report

# Generate report
qc_report = generate_qc_report(adata_annotated)
```

#### 5.2 Comprehensive Visualization
```python
import matplotlib.pyplot as plt

def create_visualization_panel(adata, save_path=None):
    """
    Create comprehensive visualization panel.
    
    Parameters:
    -----------
    adata : AnnData object
    save_path : str
        Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # UMAP colored by cluster
    sc.pl.umap(adata, color='leiden_1.0', ax=axes[0, 0], show=False)
    axes[0, 0].set_title('UMAP: Leiden Clustering (res=1.0)')
    
    # UMAP colored by n_genes
    sc.pl.umap(adata, color='n_genes_by_counts', ax=axes[0, 1], show=False)
    axes[0, 1].set_title('UMAP: Gene Count per Cell')
    
    # Violin plot of gene counts
    sc.pl.violin(adata, ['n_genes_by_counts'], ax=axes[1, 0], show=False)
    axes[1, 0].set_title('Distribution of Gene Counts')
    
    # PCA variance explained
    variance_ratio = adata.uns['pca']['variance_ratio']
    cum_var = np.cumsum(variance_ratio)
    axes[1, 1].plot(cum_var[:50], marker='o', linestyle='-')
    axes[1, 1].set_xlabel('PC')
    axes[1, 1].set_ylabel('Cumulative Variance Explained')
    axes[1, 1].set_title('PCA: Cumulative Variance Explained')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()

# Create visualization
create_visualization_panel(adata_annotated, save_path='figures/scRNA_analysis_panel.png')
```

---

## Complete Pipeline Summary

```python
# Full pipeline execution
def run_complete_pipeline(data_path, output_dir):
    """
    Execute complete single-cell transcriptomics pipeline.
    """
    # Phase 1: Load and QC
    adata = load_single_cell_data(data_path)
    adata = quality_control(adata)
    
    # Phase 2: Normalize and select features
    adata = normalize_and_log_transform(adata)
    adata = feature_selection(adata)
    adata = adata[:, adata.var['highly_variable']]
    
    # Phase 3: Dimensionality reduction
    adata = compute_pca(adata)
    adata = compute_umap(adata)
    
    # Phase 4: Clustering and annotation
    adata = cluster_leiden(adata)
    adata = annotate_cell_types(adata, marker_genes)
    
    # Phase 5: QC and visualization
    qc_report = generate_qc_report(adata)
    create_visualization_panel(adata, save_path=f'{output_dir}/analysis_panel.png')
    
    # Save results
    adata.write(f'{output_dir}/adata_processed.h5ad')
    
    return adata, qc_report

# Execute pipeline
# adata_final, report = run_complete_pipeline('data/sample.h5ad', 'results/')
```

---

## Installation Requirements

```bash
# Core dependencies
pip install scanpy==1.10.0
pip install anndata==0.10.0
pip install pandas==2.0.0
pip install numpy==1.24.0
pip install scipy==1.10.0
pip install scikit-learn==1.2.0
pip install matplotlib==3.7.0
pip install seaborn==0.12.0
pip install umap-learn==0.5.3
pip install leidenalg==0.10.0
```

---

## Version History

- **v1.0** (2025-12-18): Initial implementation with 5-phase pipeline structure
  - Phase 1: Data loading and QC
  - Phase 2: Normalization and feature selection
  - Phase 3: Dimensionality reduction
  - Phase 4: Clustering and annotation
  - Phase 5: Visualization and reporting

---

## Notes

- All phases are designed to be modular and can be executed independently
- Parameters should be adjusted based on your specific dataset characteristics
- Consider batch effects when working with multiple samples
- Validate clustering results using multiple resolution parameters
- Keep intermediate results for troubleshooting and validation
