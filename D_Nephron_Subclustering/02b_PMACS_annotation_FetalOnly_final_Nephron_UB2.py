#!/usr/bin/env python
# coding: utf-8

# Currently running with scanpy_version2.sif

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import anndata
import scanpy as sc
import matplotlib as mpl


# In[2]:


sc.settings.verbosity = 3
sc.logging.print_header()
mpl.rcParams['figure.dpi'] = 450


# In[3]:


genes_by_counts_max = 2500 # maximum number of counts per cell
genes_by_counts_min = 500 # minimum number of counts per cell
mtThresh_fetal = 25 # percent mitochondrial gene maximum
mtThresh_adult = 40 # percent mitochondrial gene maximum
nPCs_fetal = 40
nPCs_adult = 40
minGenes = 200 # use only cells with at least 200 genes
minCells = 3 # use only genes expressed in at last 3 cells
neighbors = 15

#varsToRegress = ['total_counts', 'pct_counts_mt']
varsToRegress = ['total_counts', 'pct_counts_mt', 'S_score', 'G2M_score']

#cell_cycle_genes = [x.strip() for x in open("/content/drive/MyDrive/SusztakLabFiles/cellCycleGenes.txt")]
cell_cycle_genes = [x.strip() for x in open("/home/levinsj/Fetal_dir/Analysis/referenceFiles/cellCycleGenes.txt")]
s_genes = cell_cycle_genes[:43]
g2m_genes = cell_cycle_genes[43:]


# In[4]:


adata_merge = sc.read_h5ad("/home/levinsj/Fetal_dir/CellBenderCorrected/04_annotated/cellSubsets/FetalNephronAndUB_annotated_final.h5ad")



# In[5]:


# need to re-run after calculating velocities
absorption_prob = pd.read_csv('/home/levinsj/Fetal_dir/Velocyto/03_CellRank/Absorption_probabilities_nephron_UB_a1.csv')
absorption_prob.index = adata_merge.obs.index

adata_merge.obs["NPC_prob"]= absorption_prob[["NPC"]]

adata_merge.X = adata_merge.layers["counts"] 

sc.pl.violin(adata_merge, keys = ['UMOD'],  size = 1, groupby = 'sample', rotation= 45, layer = "counts", use_raw = False)

sc.pl.umap(adata_merge, color = ["NPC_prob"])
sc.pl.umap(adata_merge, color = "SLC12A3", color_map = 'viridis_r')
sc.pl.umap(adata_merge, color = "CLDN16", color_map = 'viridis_r')
sc.pl.umap(adata_merge, color = "GATA3", color_map = 'viridis_r')
sc.pl.umap(adata_merge, color = "PTH1R", color_map = 'viridis_r')
sc.pl.umap(adata_merge, color = "UMOD", color_map = 'viridis_r')
sc.pl.umap(adata_merge, color = "leiden", color_map = 'viridis_r')


sc.pl.umap(adata_merge, color = "SLC12A3", color_map = 'viridis_r')
sc.pl.umap(adata_merge, color = "RET", color_map = 'viridis_r')
sc.pl.umap(adata_merge, color = "ADAMTS18", color_map = 'viridis_r')
sc.pl.umap(adata_merge, color = "TNFRSF19", color_map = 'viridis_r')
sc.pl.umap(adata_merge, color = "ATP6V1G3", color_map = 'viridis_r')


adata_NPC = adata_merge[adata_merge.obs["NPC_prob"] >= 0.5]
adata_UB = adata_merge[adata_merge.obs["NPC_prob"] < 0.5]

sc.pl.umap(adata_UB, color = "SLC12A3", color_map = 'viridis_r')
sc.pl.umap(adata_UB, color = "RET", color_map = 'viridis_r')
sc.pl.umap(adata_UB, color = "ADAMTS18", color_map = 'viridis_r')
sc.pl.umap(adata_UB, color = "TNFRSF19", color_map = 'viridis_r')
sc.pl.umap(adata_UB, color = "ATP6V1G3", color_map = 'viridis_r')


# # Subcluster Just NPC derived cells

# In[6]:


adata_UB.X = adata_UB.layers["counts"] 
sc.pp.normalize_total(adata_UB, target_sum=1e4)
sc.pp.log1p(adata_UB)
sc.pp.highly_variable_genes(adata_UB, min_mean=0.0125, max_mean=3, min_disp=0.5, batch_key = 'sample')
sc.pp.regress_out(adata_UB, keys = varsToRegress)
sc.pp.neighbors(adata_UB, n_neighbors=neighbors, n_pcs=nPCs_fetal, use_rep= "X_pca_harmony")
sc.tl.leiden(adata_UB)
sc.tl.paga(adata_UB)
sc.pl.paga(adata_UB, plot=False)  # remove `plot=False` if you want to see the coarse-grained graph
sc.tl.umap(adata_UB, min_dist=0.3)


sc.pl.umap(adata_UB, color = "SLC12A3", color_map = 'viridis_r')
sc.pl.umap(adata_UB, color = "RET", color_map = 'viridis_r')
sc.pl.umap(adata_UB, color = "ADAMTS18", color_map = 'viridis_r')
sc.pl.umap(adata_UB, color = "TNFRSF19", color_map = 'viridis_r')
sc.pl.umap(adata_UB, color = "ATP6V1G3", color_map = 'viridis_r')
sc.pl.umap(adata_UB, color = "leiden", color_map = 'viridis_r')


# In[7]:


cell_identities = {'0': 'UB', '1': 'UB', '2': 'Urethelium', '3': 'CNT', '4': 'UB', '5': 'UB', '6': 'UB', '7': 'IC', '8': 'other'}
adata_UB.obs["cellType"] = adata_UB.obs['leiden'].map(cell_identities).astype('category')


# In[8]:


adata_UB = adata_UB[adata_UB.obs["leiden"] != "8"]


# In[9]:


sc.pl.umap(adata_UB, color = "cellType")


# In[10]:


adata_UB.write_h5ad(filename = "/home/levinsj/Fetal_dir/CellBenderCorrected/04_annotated/cellSubsets/FetalUBOnly_annotated_final.h5ad")

