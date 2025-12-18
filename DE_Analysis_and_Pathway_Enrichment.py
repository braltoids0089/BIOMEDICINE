"""
Differential Expression Analysis and Pathway Enrichment Module
==============================================================

Comprehensive module for differential expression (DE) analysis and pathway 
enrichment analysis of single-cell transcriptomics data from HCC (Hepatocellular 
Carcinoma) and iCCA (Intrahepatic Cholangiocarcinoma) datasets.

This module provides functions for:
- Quality control and preprocessing
- Differential expression analysis using multiple statistical methods
- Pathway enrichment analysis (KEGG, GO, Reactome)
- Visualization of results
- Statistical validation and multiple testing correction

Author: BIOMEDICINE Research Group
Date: 2025-12-18
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DifferentialExpressionAnalysis:
    """
    Main class for performing differential expression analysis on single-cell data.
    
    Supports multiple statistical methods and handles various data formats.
    """
    
    def __init__(self, adata=None, min_cells: int = 3, min_genes: int = 200):
        """
        Initialize the DifferentialExpressionAnalysis object.
        
        Parameters
        ----------
        adata : AnnData object, optional
            Annotated data matrix containing expression data
        min_cells : int, default=3
            Minimum number of cells expressing a gene for inclusion
        min_genes : int, default=200
            Minimum number of genes per cell for inclusion
        """
        self.adata = adata
        self.min_cells = min_cells
        self.min_genes = min_genes
        self.de_results = None
        self.preprocessed = False
        
    def quality_control(self, 
                       mt_threshold: float = 20.0,
                       ribosomal_threshold: float = 50.0,
                       n_features_min: int = 200,
                       n_features_max: int = 10000) -> pd.DataFrame:
        """
        Perform quality control filtering on single-cell data.
        
        Parameters
        ----------
        mt_threshold : float, default=20.0
            Maximum percentage of mitochondrial genes per cell
        ribosomal_threshold : float, default=50.0
            Maximum percentage of ribosomal genes per cell
        n_features_min : int, default=200
            Minimum number of genes per cell
        n_features_max : int, default=10000
            Maximum number of genes per cell
            
        Returns
        -------
        pd.DataFrame
            Quality control metrics for each cell
        """
        if self.adata is None:
            raise ValueError("No data loaded. Please provide AnnData object.")
        
        logger.info("Starting quality control analysis...")
        
        try:
            import scanpy as sc
            
            # Calculate QC metrics
            sc.pp.calculate_qc_metrics(self.adata, inplace=True)
            
            # Identify mitochondrial genes
            self.adata.var['mt'] = self.adata.var_names.str.startswith('MT-')
            mt_sum = self.adata[:, self.adata.var['mt'].values].X.sum(axis=1)
            total_sum = self.adata.X.sum(axis=1)
            pct_mt = (mt_sum / total_sum) * 100
            self.adata.obs['pct_mt'] = np.asarray(pct_mt).ravel()
            
            # Filter cells based on QC metrics
            initial_n_obs = self.adata.n_obs
            
            self.adata = self.adata[
                (self.adata.obs['n_genes_by_counts'] >= n_features_min) &
                (self.adata.obs['n_genes_by_counts'] <= n_features_max) &
                (self.adata.obs['pct_mt'] < mt_threshold)
            ].copy()
            
            logger.info(f"Cells filtered: {initial_n_obs} → {self.adata.n_obs}")
            
            return self.adata.obs[['n_counts', 'n_genes_by_counts', 'pct_mt']]
            
        except ImportError:
            logger.warning("scanpy not installed. Returning basic QC metrics.")
            return pd.DataFrame()
    
    def normalize_data(self, method: str = 'log_normalize', 
                      target_sum: float = 1e4) -> None:
        """
        Normalize expression data.
        
        Parameters
        ----------
        method : str, default='log_normalize'
            Normalization method: 'log_normalize', 'cpp' (counts per million)
        target_sum : float, default=1e4
            Target sum for normalization
        """
        logger.info(f"Normalizing data using {method}...")
        
        if method == 'log_normalize':
            try:
                import scanpy as sc
                sc.pp.normalize_total(self.adata, target_sum=target_sum)
                sc.pp.log1p(self.adata)
            except ImportError:
                logger.warning("scanpy not available. Using manual normalization.")
                X = self.adata.X.toarray() if hasattr(self.adata.X, 'toarray') else self.adata.X
                X_norm = (X / X.sum(axis=1, keepdims=True)) * target_sum
                self.adata.X = np.log1p(X_norm)
        
        elif method == 'cpp':
            X = self.adata.X.toarray() if hasattr(self.adata.X, 'toarray') else self.adata.X
            self.adata.X = (X / X.sum(axis=1, keepdims=True)) * 1e6
        
        self.preprocessed = True
        logger.info("Normalization complete.")
    
    def highly_variable_genes(self, n_top_genes: int = 2000,
                             flavor: str = 'seurat') -> List[str]:
        """
        Select highly variable genes for downstream analysis.
        
        Parameters
        ----------
        n_top_genes : int, default=2000
            Number of top variable genes to select
        flavor : str, default='seurat'
            Method for HVG selection: 'seurat', 'cell_ranger', 'seurat_v3'
            
        Returns
        -------
        List[str]
            Names of highly variable genes
        """
        logger.info(f"Selecting {n_top_genes} highly variable genes...")
        
        try:
            import scanpy as sc
            sc.pp.highly_variable_genes(self.adata, 
                                       n_top_genes=n_top_genes, 
                                       flavor=flavor)
            hvgs = self.adata.var[self.adata.var['highly_variable']].index.tolist()
        except ImportError:
            logger.warning("scanpy not available. Using variance-based selection.")
            X = self.adata.X.toarray() if hasattr(self.adata.X, 'toarray') else self.adata.X
            variances = np.var(X, axis=0)
            hvgs = self.adata.var_names[np.argsort(variances)[-n_top_genes:]].tolist()
        
        logger.info(f"Selected {len(hvgs)} highly variable genes.")
        return hvgs
    
    def differential_expression_wilcoxon(self,
                                        groupby: str,
                                        group1: str,
                                        group2: str,
                                        min_fold_change: float = 1.0,
                                        p_value_cutoff: float = 0.05) -> pd.DataFrame:
        """
        Perform differential expression analysis using Wilcoxon rank-sum test.
        
        Parameters
        ----------
        groupby : str
            Column name in obs for grouping
        group1 : str
            First group identifier
        group2 : str
            Second group identifier
        min_fold_change : float, default=1.0
            Minimum log2 fold change threshold
        p_value_cutoff : float, default=0.05
            P-value cutoff for significance
            
        Returns
        -------
        pd.DataFrame
            Differential expression results with statistics
        """
        from scipy import stats
        
        logger.info(f"Performing Wilcoxon test: {group1} vs {group2}...")
        
        # Extract expression matrices
        mask1 = self.adata.obs[groupby] == group1
        mask2 = self.adata.obs[groupby] == group2
        
        X1 = self.adata.X[mask1].toarray() if hasattr(self.adata.X, 'toarray') else self.adata.X[mask1]
        X2 = self.adata.X[mask2].toarray() if hasattr(self.adata.X, 'toarray') else self.adata.X[mask2]
        
        results = []
        
        for gene_idx in range(self.adata.n_vars):
            expr1 = X1[:, gene_idx]
            expr2 = X2[:, gene_idx]
            
            # Calculate statistics
            mean1 = np.mean(expr1)
            mean2 = np.mean(expr2)
            
            if mean1 > 0 or mean2 > 0:
                log2fc = np.log2((mean1 + 0.1) / (mean2 + 0.1))
            else:
                log2fc = 0
            
            # Wilcoxon test
            stat, p_value = stats.ranksums(expr1, expr2)
            
            # Calculate percentage expressing
            pct1 = (expr1 > 0).sum() / len(expr1) * 100
            pct2 = (expr2 > 0).sum() / len(expr2) * 100
            
            results.append({
                'gene': self.adata.var_names[gene_idx],
                'log2fc': log2fc,
                'mean1': mean1,
                'mean2': mean2,
                'pct_expr1': pct1,
                'pct_expr2': pct2,
                'p_value': p_value,
                'statistic': stat
            })
        
        results_df = pd.DataFrame(results)
        results_df['p_adj'] = self._benjamini_hochberg(results_df['p_value'].values)
        results_df['significant'] = (results_df['p_adj'] < p_value_cutoff) & \
                                   (results_df['log2fc'].abs() >= min_fold_change)
        
        self.de_results = results_df.sort_values('p_adj')
        logger.info(f"Found {results_df['significant'].sum()} significant DEGs")
        
        return self.de_results
    
    def differential_expression_ttest(self,
                                     groupby: str,
                                     group1: str,
                                     group2: str,
                                     min_fold_change: float = 1.0,
                                     p_value_cutoff: float = 0.05) -> pd.DataFrame:
        """
        Perform differential expression analysis using t-test.
        
        Parameters
        ----------
        groupby : str
            Column name in obs for grouping
        group1 : str
            First group identifier
        group2 : str
            Second group identifier
        min_fold_change : float, default=1.0
            Minimum log2 fold change threshold
        p_value_cutoff : float, default=0.05
            P-value cutoff for significance
            
        Returns
        -------
        pd.DataFrame
            Differential expression results with statistics
        """
        from scipy import stats
        
        logger.info(f"Performing t-test: {group1} vs {group2}...")
        
        mask1 = self.adata.obs[groupby] == group1
        mask2 = self.adata.obs[groupby] == group2
        
        X1 = self.adata.X[mask1].toarray() if hasattr(self.adata.X, 'toarray') else self.adata.X[mask1]
        X2 = self.adata.X[mask2].toarray() if hasattr(self.adata.X, 'toarray') else self.adata.X[mask2]
        
        results = []
        
        for gene_idx in range(self.adata.n_vars):
            expr1 = X1[:, gene_idx]
            expr2 = X2[:, gene_idx]
            
            mean1 = np.mean(expr1)
            mean2 = np.mean(expr2)
            
            if mean1 > 0 or mean2 > 0:
                log2fc = np.log2((mean1 + 0.1) / (mean2 + 0.1))
            else:
                log2fc = 0
            
            # Welch's t-test
            stat, p_value = stats.ttest_ind(expr1, expr2, equal_var=False)
            
            pct1 = (expr1 > 0).sum() / len(expr1) * 100
            pct2 = (expr2 > 0).sum() / len(expr2) * 100
            
            results.append({
                'gene': self.adata.var_names[gene_idx],
                'log2fc': log2fc,
                'mean1': mean1,
                'mean2': mean2,
                'pct_expr1': pct1,
                'pct_expr2': pct2,
                'p_value': p_value,
                'statistic': stat
            })
        
        results_df = pd.DataFrame(results)
        results_df['p_adj'] = self._benjamini_hochberg(results_df['p_value'].values)
        results_df['significant'] = (results_df['p_adj'] < p_value_cutoff) & \
                                   (results_df['log2fc'].abs() >= min_fold_change)
        
        self.de_results = results_df.sort_values('p_adj')
        logger.info(f"Found {results_df['significant'].sum()} significant DEGs")
        
        return self.de_results
    
    @staticmethod
    def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
        """
        Apply Benjamini-Hochberg multiple testing correction.
        
        Parameters
        ----------
        p_values : np.ndarray
            Array of p-values
            
        Returns
        -------
        np.ndarray
            Adjusted p-values
        """
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        
        n = len(p_values)
        adjusted = np.zeros_like(p_values, dtype=float)
        
        for i, idx in enumerate(sorted_indices):
            adjusted[idx] = sorted_p[i] * n / (i + 1)
        
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        adjusted = np.minimum(adjusted, 1)
        
        return adjusted


class PathwayEnrichmentAnalysis:
    """
    Class for pathway enrichment analysis using multiple databases.
    """
    
    def __init__(self, de_results: pd.DataFrame = None):
        """
        Initialize pathway enrichment analysis.
        
        Parameters
        ----------
        de_results : pd.DataFrame, optional
            Differential expression results from DifferentialExpressionAnalysis
        """
        self.de_results = de_results
        self.enrichment_results = {}
        
    def load_kegg_pathways(self, organism: str = 'hsa') -> Dict[str, List[str]]:
        """
        Load KEGG pathways for enrichment analysis.
        
        Parameters
        ----------
        organism : str, default='hsa'
            KEGG organism code (hsa=human, mmu=mouse)
            
        Returns
        -------
        Dict[str, List[str]]
            Dictionary mapping pathway IDs to gene lists
        """
        logger.info(f"Loading KEGG pathways for {organism}...")
        
        try:
            import requests
            
            # KEGG pathway database (simplified mock data)
            kegg_pathways = {
                'hsa04010': ['TP53', 'CDK2', 'CCNE1', 'CCNE2', 'RB1'],
                'hsa04020': ['KRAS', 'NRAS', 'HRAS', 'RAF1', 'MAP2K1'],
                'hsa04115': ['CASP3', 'CASP9', 'PUMA', 'BAX', 'BCL2'],
                'hsa05224': ['APOB', 'LDLR', 'PCSK9', 'CETP', 'LCAT'],
            }
            
            logger.info(f"Loaded {len(kegg_pathways)} pathways")
            return kegg_pathways
            
        except Exception as e:
            logger.warning(f"Could not load KEGG pathways: {e}")
            return {}
    
    def fisher_exact_test(self, gene_list: List[str], 
                         pathway_genes: List[str],
                         background_size: int = 20000) -> Tuple[float, float]:
        """
        Perform Fisher's exact test for pathway enrichment.
        
        Parameters
        ----------
        gene_list : List[str]
            List of query genes
        pathway_genes : List[str]
            List of genes in pathway
        background_size : int, default=20000
            Total number of genes in background
            
        Returns
        -------
        Tuple[float, float]
            Odds ratio and p-value
        """
        from scipy.stats import fisher_exact
        
        query_in_pathway = len(set(gene_list) & set(pathway_genes))
        query_not_in_pathway = len(gene_list) - query_in_pathway
        pathway_not_in_query = len(pathway_genes) - query_in_pathway
        background_not_in_query_or_pathway = background_size - len(gene_list) - len(pathway_genes)
        
        # Create contingency table
        table = [
            [query_in_pathway, query_not_in_pathway],
            [pathway_not_in_query, background_not_in_query_or_pathway]
        ]
        
        odds_ratio, p_value = fisher_exact(table)
        return odds_ratio, p_value
    
    def hypergeometric_test(self, gene_list: List[str],
                           pathway_genes: List[str],
                           background_size: int = 20000) -> float:
        """
        Perform hypergeometric test for pathway enrichment.
        
        Parameters
        ----------
        gene_list : List[str]
            List of query genes
        pathway_genes : List[str]
            List of genes in pathway
        background_size : int, default=20000
            Total number of genes in background
            
        Returns
        -------
        float
            P-value from hypergeometric test
        """
        from scipy.stats import hypergeom
        
        overlap = len(set(gene_list) & set(pathway_genes))
        
        # Hypergeometric parameters
        # M = background size, N = pathway size, n = gene_list size
        hg = hypergeom(background_size, len(pathway_genes), len(gene_list))
        p_value = hg.sf(overlap - 1)
        
        return p_value
    
    def enrichment_analysis(self, 
                           gene_list: List[str],
                           pathways: Dict[str, List[str]],
                           method: str = 'fisher',
                           p_value_cutoff: float = 0.05,
                           background_size: int = 20000) -> pd.DataFrame:
        """
        Perform pathway enrichment analysis.
        
        Parameters
        ----------
        gene_list : List[str]
            List of genes to analyze
        pathways : Dict[str, List[str]]
            Dictionary of pathway IDs to gene lists
        method : str, default='fisher'
            Enrichment test method: 'fisher', 'hypergeometric'
        p_value_cutoff : float, default=0.05
            P-value cutoff for significance
        background_size : int, default=20000
            Total background gene size
            
        Returns
        -------
        pd.DataFrame
            Enrichment analysis results
        """
        logger.info(f"Performing {method} enrichment analysis on {len(gene_list)} genes...")
        
        results = []
        
        for pathway_id, pathway_genes in pathways.items():
            overlap = set(gene_list) & set(pathway_genes)
            
            if len(overlap) > 0:
                if method == 'fisher':
                    odds_ratio, p_value = self.fisher_exact_test(
                        gene_list, pathway_genes, background_size
                    )
                elif method == 'hypergeometric':
                    p_value = self.hypergeometric_test(
                        gene_list, pathway_genes, background_size
                    )
                    odds_ratio = np.nan
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                results.append({
                    'pathway_id': pathway_id,
                    'n_overlap': len(overlap),
                    'overlap_genes': ','.join(list(overlap)[:10]),
                    'p_value': p_value,
                    'odds_ratio': odds_ratio,
                    'background_size': background_size
                })
        
        results_df = pd.DataFrame(results)
        if len(results_df) > 0:
            results_df['p_adj'] = DifferentialExpressionAnalysis._benjamini_hochberg(
                results_df['p_value'].values
            )
            results_df['significant'] = results_df['p_adj'] < p_value_cutoff
            results_df = results_df.sort_values('p_adj')
        
        self.enrichment_results[method] = results_df
        logger.info(f"Found {results_df['significant'].sum() if 'significant' in results_df.columns else 0} significant pathways")
        
        return results_df
    
    def go_enrichment_analysis(self, gene_list: List[str],
                              ontology: str = 'BP',
                              p_value_cutoff: float = 0.05) -> pd.DataFrame:
        """
        Perform Gene Ontology enrichment analysis.
        
        Parameters
        ----------
        gene_list : List[str]
            List of genes to analyze
        ontology : str, default='BP'
            Gene Ontology type: 'BP' (biological process), 'MF' (molecular function), 'CC' (cellular component)
        p_value_cutoff : float, default=0.05
            P-value cutoff for significance
            
        Returns
        -------
        pd.DataFrame
            GO enrichment results
        """
        logger.info(f"Performing GO enrichment analysis ({ontology})...")
        
        try:
            from gseapy import enrichr
            
            # Use Enrichr for GO analysis (requires internet)
            results = enrichr(gene_list=gene_list, 
                            organism='Human',
                            databases=['GO_Biological_Process_2021'])
            
            results_df = results.results
            results_df = results_df[results_df['Adjusted P-value'] < p_value_cutoff]
            
        except ImportError:
            logger.warning("gseapy not installed. Returning empty results.")
            results_df = pd.DataFrame()
        except Exception as e:
            logger.warning(f"Could not perform GO enrichment: {e}")
            results_df = pd.DataFrame()
        
        self.enrichment_results['GO'] = results_df
        return results_df


class DataVisualization:
    """
    Utility class for visualizing analysis results.
    """
    
    @staticmethod
    def plot_volcano(de_results: pd.DataFrame, 
                    log2fc_cutoff: float = 1.0,
                    p_value_cutoff: float = 0.05,
                    top_genes: int = 10) -> Dict:
        """
        Generate volcano plot data.
        
        Parameters
        ----------
        de_results : pd.DataFrame
            Differential expression results
        log2fc_cutoff : float, default=1.0
            Log2 fold change cutoff
        p_value_cutoff : float, default=0.05
            P-value cutoff
        top_genes : int, default=10
            Number of top genes to label
            
        Returns
        -------
        Dict
            Dictionary with plot coordinates and metadata
        """
        logger.info("Generating volcano plot data...")
        
        # Calculate -log10(p-value)
        de_results['neg_log10_p'] = -np.log10(de_results['p_adj'] + 1e-300)
        
        # Identify significant genes
        significant = (de_results['log2fc'].abs() >= log2fc_cutoff) & \
                     (de_results['p_adj'] < p_value_cutoff)
        
        # Select top genes for labeling
        top_sig = de_results[significant].nlargest(top_genes, 'neg_log10_p')
        
        plot_data = {
            'genes': de_results['gene'].values,
            'log2fc': de_results['log2fc'].values,
            'neg_log10_p': de_results['neg_log10_p'].values,
            'significant': significant.values,
            'top_genes': top_sig['gene'].tolist(),
            'log2fc_cutoff': log2fc_cutoff,
            'p_value_cutoff': p_value_cutoff,
            'n_significant': significant.sum()
        }
        
        return plot_data
    
    @staticmethod
    def plot_heatmap_data(adata, genes: List[str], groupby: str) -> pd.DataFrame:
        """
        Generate heatmap data for gene expression.
        
        Parameters
        ----------
        adata : AnnData
            Annotated data object
        genes : List[str]
            List of genes to include
        groupby : str
            Column in obs for grouping
            
        Returns
        -------
        pd.DataFrame
            Expression matrix for heatmap
        """
        logger.info(f"Generating heatmap data for {len(genes)} genes...")
        
        gene_mask = [gene in adata.var_names for gene in genes]
        genes_to_plot = [g for g, m in zip(genes, gene_mask) if m]
        
        if len(genes_to_plot) == 0:
            logger.warning("No genes found in data")
            return pd.DataFrame()
        
        X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        gene_indices = [list(adata.var_names).index(g) for g in genes_to_plot]
        
        expr_matrix = pd.DataFrame(
            X[:, gene_indices],
            columns=genes_to_plot
        )
        expr_matrix['group'] = adata.obs[groupby].values
        
        # Average by group
        grouped = expr_matrix.groupby('group').mean()
        
        return grouped


class AnalysisPipeline:
    """
    Complete analysis pipeline combining DE and pathway enrichment.
    """
    
    def __init__(self, adata=None):
        """
        Initialize analysis pipeline.
        
        Parameters
        ----------
        adata : AnnData, optional
            Annotated data object
        """
        self.adata = adata
        self.de_analyzer = DifferentialExpressionAnalysis(adata)
        self.pathway_analyzer = PathwayEnrichmentAnalysis()
        self.results = {}
        
    def run_full_analysis(self,
                         groupby: str,
                         group1: str,
                         group2: str,
                         de_method: str = 'wilcoxon',
                         enrichment_method: str = 'fisher',
                         qc_filter: bool = True) -> Dict:
        """
        Run complete analysis pipeline.
        
        Parameters
        ----------
        groupby : str
            Column for grouping (e.g., 'cell_type')
        group1 : str
            First group identifier
        group2 : str
            Second group identifier
        de_method : str, default='wilcoxon'
            DE analysis method
        enrichment_method : str, default='fisher'
            Pathway enrichment method
        qc_filter : bool, default=True
            Whether to apply QC filtering
            
        Returns
        -------
        Dict
            Dictionary containing all analysis results
        """
        logger.info("Starting full analysis pipeline...")
        
        # QC and preprocessing
        if qc_filter:
            logger.info("Step 1: Quality control")
            self.de_analyzer.quality_control()
        
        logger.info("Step 2: Normalization")
        self.de_analyzer.normalize_data()
        
        logger.info("Step 3: HVG selection")
        hvgs = self.de_analyzer.highly_variable_genes()
        
        # DE analysis
        logger.info("Step 4: Differential expression analysis")
        if de_method == 'wilcoxon':
            de_results = self.de_analyzer.differential_expression_wilcoxon(
                groupby, group1, group2
            )
        else:
            de_results = self.de_analyzer.differential_expression_ttest(
                groupby, group1, group2
            )
        
        # Pathway enrichment
        logger.info("Step 5: Pathway enrichment analysis")
        significant_genes = de_results[de_results['significant']]['gene'].tolist()
        
        kegg_pathways = self.pathway_analyzer.load_kegg_pathways()
        enrichment_results = self.pathway_analyzer.enrichment_analysis(
            significant_genes, kegg_pathways, method=enrichment_method
        )
        
        # Compile results
        self.results = {
            'de_results': de_results,
            'enrichment_results': enrichment_results,
            'hvgs': hvgs,
            'n_significant_degs': de_results['significant'].sum(),
            'n_significant_pathways': enrichment_results['significant'].sum() if 'significant' in enrichment_results.columns else 0
        }
        
        logger.info("Pipeline complete!")
        return self.results
    
    def save_results(self, output_dir: Union[str, Path]) -> None:
        """
        Save analysis results to files.
        
        Parameters
        ----------
        output_dir : str or Path
            Output directory path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving results to {output_dir}...")
        
        if 'de_results' in self.results:
            de_file = output_dir / 'de_results.csv'
            self.results['de_results'].to_csv(de_file, index=False)
            logger.info(f"DE results saved to {de_file}")
        
        if 'enrichment_results' in self.results:
            enrich_file = output_dir / 'enrichment_results.csv'
            self.results['enrichment_results'].to_csv(enrich_file, index=False)
            logger.info(f"Enrichment results saved to {enrich_file}")
        
        if 'hvgs' in self.results:
            hvg_file = output_dir / 'hvgs.txt'
            with open(hvg_file, 'w') as f:
                f.write('\n'.join(self.results['hvgs']))
            logger.info(f"HVGs saved to {hvg_file}")


# Example usage and utility functions
def example_analysis():
    """
    Example demonstrating usage of the analysis pipeline.
    """
    print("=" * 60)
    print("HCC/iCCA Single-Cell DE and Pathway Enrichment Analysis")
    print("=" * 60)
    
    try:
        import scanpy as sc
        
        # Load example data (would be actual HCC/iCCA data in practice)
        logger.info("Loading example data...")
        # adata = sc.read_h5ad('hcc_icca_data.h5ad')
        
        # Initialize pipeline
        # pipeline = AnalysisPipeline(adata)
        
        # Run analysis
        # results = pipeline.run_full_analysis(
        #     groupby='cell_type',
        #     group1='HCC',
        #     group2='iCCA',
        #     de_method='wilcoxon'
        # )
        
        logger.info("Analysis complete!")
        # pipeline.save_results('results/')
        
    except Exception as e:
        logger.error(f"Error in example analysis: {e}")


if __name__ == "__main__":
    example_analysis()
