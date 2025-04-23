#!/usr/bin/env python
# coding: utf-8

# # Getting started
# Importing python packages. Currently running with scanpy_version2.sif. 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import anndata
import scanpy as sc
import harmonypy
import scrublet as scr


# Commented out for installing python packages to drive. Also calling current package information. 

# In[2]:


sc.settings.verbosity = 3
sc.logging.print_header()
sc.settings.set_figure_params(frameon=False, dpi_save=1000)


# # Parameters
# Setting values for pre-processing of fetal samples

# In[3]:


genes_by_counts_max = 2500 # maximum number of counts per cell
genes_by_counts_min = 500 # minimum number of counts per cell
mtThresh_fetal = 25 # percent mitochondrial gene maximum
mtThresh_adult = 40 # percent mitochondrial gene maximum
nPCs_fetal = 40
nPCs_adult = 40
minCells = 3 # use only genes expressed in at last 3 cells
neighbors = 15

#varsToRegress = ['total_counts', 'pct_counts_mt']
varsToRegress = ['total_counts', 'pct_counts_mt', 'S_score', 'G2M_score']

#cell_cycle_genes = [x.strip() for x in open("/content/drive/MyDrive/SusztakLabFiles/cellCycleGenes.txt")]
cell_cycle_genes = [x.strip() for x in open("/home/levinsj/Fetal_dir/Analysis/referenceFiles/cellCycleGenes.txt")]
s_genes = cell_cycle_genes[:43]
g2m_genes = cell_cycle_genes[43:]


# # Importing and Doublet Calling
# We will import the cellBender corrected matricies and remove doublets via scrublet.

# In[4]:


fetalSamples = "/home/levinsj/Fetal_dir/CellBenderCorrected/01_rawCellBender/HK2716_cellBender.h5ad", "/home/levinsj/Fetal_dir/CellBenderCorrected/01_rawCellBender/HK2718_cellBender.h5ad", "/home/levinsj/Fetal_dir/CellBenderCorrected/01_rawCellBender/HK2722_cellBender.h5ad",  "/home/levinsj/Fetal_dir/CellBenderCorrected/01_rawCellBender/HK2723_cellBender.h5ad", "/home/levinsj/Fetal_dir/CellBenderCorrected/01_rawCellBender/HK2725_cellBender.h5ad"

#sampleCellBenderProcessed = ["/content/drive/MyDrive/SusztakLabFiles/scRNAseq/HK2716preprocessed.h5ad", "/content/drive/MyDrive/SusztakLabFiles/scRNAseq/HK2718preprocessed.h5ad", "/content/drive/MyDrive/SusztakLabFiles/scRNAseq/HK2722preprocessed.h5ad", "/content/drive/MyDrive/SusztakLabFiles/scRNAseq/HK2723preprocessed.h5ad", "/content/drive/MyDrive/SusztakLabFiles/scRNAseq/HK2725preprocessed.h5ad"]
fetalNames = ["HK2716", "HK2718", "HK2722", "HK2723", "HK2725"]
fetal_list = []
for index, item in enumerate(fetalSamples):
    print(fetalNames[index])
    adata = sc.read_h5ad(fetalSamples[index])
    adata.var_names_make_unique()
    adata.obs["sample"] = fetalNames[index]
    adata.obs["type"] = "fetal"
    adata.layers["counts"] = adata.X.copy()
    scrub = scr.Scrublet(adata.X, expected_doublet_rate=0.06)
    print(scrub)
    doublet_scores, predicted_doublets = scrub.scrub_doublets(min_counts=2,min_cells=3,min_gene_variability_pctl=85, n_prin_comps=40)
    adata.obs['doublet_scores'] = doublet_scores
    adata.obs['predicted_doublets'] = predicted_doublets
    if predicted_doublets is None:
      predicted_doublets = scrub.call_doublets(threshold=0.25)
      print(predicted_doublets)
      adata.obs['predicted_doublets'] = predicted_doublets
    scrub.plot_histogram()
    adata = adata[adata.obs['predicted_doublets'] == False,:]        ### remove doublets-- we use all data that are not prdicted doublets
    fetal_list.append(adata)
    print(adata.obs.index[0])
    adata.write("/home/levinsj/Fetal_dir/CellBenderCorrected/02_postScrublet/"+fetalNames[index]+'_cellBender_scrublet_final.h5ad')    


# # Merging and pre-processing
# Let's merge the samples together and perform some pre-processing and QC.

# In[5]:


adata_fetal = anndata.concat(fetal_list, index_unique = "-", keys = ["0","1","2","3","4"])

sc.pp.filter_cells(adata_fetal, min_genes=200)
sc.pp.filter_genes(adata_fetal, min_cells=minCells)
adata_fetal.var['mt'] = adata_fetal.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(adata_fetal, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
adata_fetal = adata_fetal[adata_fetal.obs.n_genes_by_counts < genes_by_counts_max, :]
adata_fetal = adata_fetal[adata_fetal.obs.n_genes_by_counts > genes_by_counts_min, :]
adata_fetal = adata_fetal[adata_fetal.obs.pct_counts_mt < mtThresh_fetal, :]

sc.pp.normalize_total(adata_fetal, target_sum=1e4)
sc.pp.log1p(adata_fetal)
adata_fetal.layers["log1p"] = adata_fetal.X.copy()
sc.pp.highly_variable_genes(adata_fetal, min_mean=0.0125, max_mean=3, min_disp=0.5, batch_key = 'sample')
sc.tl.score_genes_cell_cycle(adata_fetal, s_genes=s_genes, g2m_genes=g2m_genes)
sc.pp.regress_out(adata_fetal, keys = varsToRegress)
sc.tl.pca(adata_fetal, svd_solver='arpack')
sc.pp.neighbors(adata_fetal, n_neighbors=neighbors, n_pcs=nPCs_fetal)
sc.tl.leiden(adata_fetal)
adata_fetal.rename_categories('sample', fetalNames)

sc.pl.violin(adata_fetal, ['n_genes_by_counts'], jitter=0.4, size = 0, groupby = 'sample', rotation= 45, save='genes_by_counts_fetal.svg')
sc.pl.violin(adata_fetal, ['total_counts'], jitter=0.4, size = 0, groupby = 'sample', rotation= 45, save='t_counts_fetal.svg')

sc.pl.violin(adata_fetal, ['pct_counts_mt'], jitter=0.4, size = 0, groupby = 'sample', rotation= 45, save='pctMT_fetal.svg')


# # Harmony Integration

# In[6]:


#pre-Harmony plotting
sc.tl.paga(adata_fetal)
sc.pl.paga(adata_fetal, plot=False)  # remove `plot=False` if you want to see the coarse-grained graph
sc.tl.umap(adata_fetal, min_dist=0.3)
sc.pl.umap(adata_fetal, color="sample", save='preHarmony_UMAP_CB_sample_fetal_only.svg')

# Harmony integration
import scanpy.external as sce
sce.pp.harmony_integrate(adata_fetal, 'sample')
'X_pca_harmony' in adata_fetal.obsm

# Plot Harmony itegrated samples
sc.pp.neighbors(adata_fetal, n_neighbors=neighbors, n_pcs=nPCs_fetal, use_rep= "X_pca_harmony")
sc.tl.leiden(adata_fetal)
sc.tl.paga(adata_fetal)
sc.tl.umap(adata_fetal, min_dist=0.3)
sc.pl.umap(adata_fetal, color="sample", save='postHarmony_UMAP_CB_sample_fetal_only.svg')
sc.pl.umap(adata_fetal, color="total_counts", save ='postHarmony_UMAP_CB_totalCounts_fetal_only.svg')
sc.pl.umap(adata_fetal, color="pct_counts_mt", save ='postHarmony_UMAP_CB_precentMT_fetal_only.svg')
sc.pl.umap(adata_fetal, color="phase", save ='postHarmony_UMAP_CB_phase_fetal_only.svg')


# # Plotting known markers

# In[7]:


sc.pl.umap(adata_fetal, color = ["COL3A1","COL1A1"], color_map = 'viridis_r') # stroma
sc.pl.umap(adata_fetal, color = ["PODXL","MAFB","NPHS2"], color_map = 'viridis_r') # podocyte
sc.pl.umap(adata_fetal, color = ["SLC5A2","SLC3A1"], color_map = 'viridis_r') # proximal tubule cell
sc.pl.umap(adata_fetal, color = ["UMOD","SLC12A1"], color_map = 'viridis_r') # LOH
sc.pl.umap(adata_fetal, color = ["SLC12A3","TFAP2A"], color_map = 'viridis_r') # distal tubule
sc.pl.umap(adata_fetal, color = "ATP6V1G3", color_map = 'viridis_r') # interecolated
sc.pl.umap(adata_fetal, color = "EGFL7", color_map = 'viridis_r') # endothelial
sc.pl.umap(adata_fetal, color = "AQP2", color_map = 'viridis_r') # principle cell
sc.pl.umap(adata_fetal, color = ["SIX2","CITED1","UNCX"], color_map = 'viridis_r') # stem cells
sc.tl.rank_genes_groups(adata_fetal, 'leiden', method='wilcoxon')
sc.pl.rank_genes_groups(adata_fetal, n_genes=25, sharey=False)
sc.pl.umap(adata_fetal, color='leiden', legend_loc='on data', frameon = False)

adata_fetal.write_h5ad(filename = "/home/levinsj/Fetal_dir/CellBenderCorrected/03_merged/fetal_merged_noAnno_final.h5ad")

