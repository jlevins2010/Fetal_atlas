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
absorption_prob = pd.read_csv('/home/levinsj/Fetal_dir/Velocyto/03_CellRank/Absorption_probabilities_nephron_UB_a2.csv')
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

adata_NPC = adata_merge[adata_merge.obs["NPC_prob"] >= 0.5]
adata_UB = adata_merge[adata_merge.obs["NPC_prob"] < 0.5]

sc.pl.umap(adata_NPC, color = ["NPC_prob"])


# # Subcluster Just NPC derived cells

# In[6]:


adata_NPC.X = adata_NPC.layers["counts"] 
sc.pp.normalize_total(adata_NPC, target_sum=1e4)
sc.pp.log1p(adata_NPC)
sc.pp.highly_variable_genes(adata_NPC, min_mean=0.0125, max_mean=3, min_disp=0.5, batch_key = 'sample')
sc.pp.regress_out(adata_NPC, keys = varsToRegress)
sc.pp.neighbors(adata_NPC, n_neighbors=neighbors, n_pcs=nPCs_fetal, use_rep= "X_pca_harmony")
sc.tl.leiden(adata_NPC)
sc.tl.paga(adata_NPC)
sc.pl.paga(adata_NPC, plot=False)  # remove `plot=False` if you want to see the coarse-grained graph
sc.tl.umap(adata_NPC, min_dist=0.3)

sc.pl.violin(adata_NPC, keys = ['UMOD'],  size = 1, groupby = 'sample', rotation= 45, layer = "counts", use_raw = False)


# In[7]:


cell_identities = {'0': 'Podocyte', '1': 'NPC', '2': 'Int', '3': 'Int', '4': 'Int', '5': 'PT', '6': 'Int', '7': 'Podocyte', '8': 'LOH', '9': 'PEC', '10': 'DCT', '11': 'Int'}
adata_NPC.obs["cellType"] = adata_NPC.obs['leiden'].map(cell_identities).astype('category')


# In[8]:


sc.tl.umap(adata_NPC, min_dist=0.3, n_components=2)

sc.pl.umap(adata_NPC, color = "leiden", legend_loc='on data', legend_fontsize=10, legend_fontoutline=2, palette='Set3')
sc.pl.umap(adata_NPC, color = "phase")
sc.pl.umap(adata_NPC, color = "sample")

cell_identities = {'0': 'Podocyte', '1': 'NPC', '2': 'Int', '3': 'Int', '4': 'PT', '5': 'Int', '6': 'DCT', '7': 'Podocyte', '8': 'Int', '9': 'PEC', '10': 'LOH', '11': 'NPC'}
adata_NPC.obs["cellType"] = adata_NPC.obs['leiden'].map(cell_identities).astype('category')

cell_identities = {'0': 'Podocyte_1', '1': 'NPC_1', '2': 'Int_1', '3': 'Int_2', '4': 'PT', '5': 'Int_3', '6': 'DCT', '7': 'Podocyte_2', '8': 'Int_4', '9': 'PEC', '10': 'LOH', '11': 'NPC_2'}
adata_NPC.obs["cellType2"] = adata_NPC.obs['leiden'].map(cell_identities).astype('category')

sc.pl.umap(adata_NPC, color = "cellType", legend_loc='on data', legend_fontsize=10, legend_fontoutline=2)
sc.pl.umap(adata_NPC, color = "cellType", legend_loc='on data', legend_fontsize=8, legend_fontoutline=2)

sc.pl.umap(adata_NPC, color = "UMOD", color_map = 'viridis_r')

sc.tl.rank_genes_groups(adata_NPC, 'leiden', method='wilcoxon')
sc.pl.rank_genes_groups(adata_NPC, n_genes=25, sharey=False)


# In[9]:


markers = ["CITED1", "CUBN","PTPRO","CFH", "UMOD","TMEM52B"]
order = ["Int","NPC","PT","Podocyte", "PEC", "LOH","DCT"]
sc.pl.dotplot(adata_NPC, markers, groupby='cellType',categories_order = order, cmap='Blues', log = True, vmax = 1)

adata_NPC.obs["leiden_orig"] = adata_NPC.obs["leiden"]

sc.tl.leiden(adata_NPC, resolution = 2)
adata_NPC.obs["leiden_2.0"] = adata_NPC.obs["leiden"]
sc.pl.umap(adata_NPC, color = "leiden_2.0", legend_loc='on data', legend_fontsize=10, legend_fontoutline=2, palette='Set3')

#adata_NPC.write_h5ad(filename = "/home/levinsj/Fetal_dir/CellBenderCorrected/04_annotated/cellSubsets/FetalNephrogenicOnly_annotated_final.h5ad")
#adata_NPC.write_h5ad(filename = "/home/levinsj/Fetal_dir/CellBenderCorrected/04_annotated/cellSubsets/FetalNephrogenicOnly_annotated_final_superClean.h5ad")


# In[10]:


adata_NPC.X = adata_NPC.layers["counts"]

sc.pl.umap(adata_NPC, color = "GATA3", color_map = 'viridis_r')
sc.pl.umap(adata_NPC, color = "PTH1R", color_map = 'viridis_r')
sc.pl.umap(adata_NPC, color = "CLDN16", color_map = 'viridis_r')
sc.pl.umap(adata_NPC, color = "IRX1", color_map = 'viridis_r')
sc.pl.umap(adata_NPC, color = "IRX2", color_map = 'viridis_r')
sc.pl.umap(adata_NPC, color = "JAG1", color_map = 'viridis_r')
sc.pl.umap(adata_NPC, color = "CDH1", color_map = 'viridis_r')
sc.pl.umap(adata_NPC, color = "SLC12A3", color_map = 'viridis_r')
sc.pl.umap(adata_NPC, color = "CITED1", color_map = 'viridis_r')
sc.pl.umap(adata_NPC, color = "SIX2", color_map = 'viridis_r')
sc.pl.umap(adata_NPC, color = "WT1", color_map = 'viridis_r')
sc.pl.umap(adata_NPC, color = "ACE2", color_map = 'viridis_r')


sc.pl.umap(adata_NPC, color = "MECOM", color_map = 'viridis_r')
sc.pl.umap(adata_NPC, color = "MEIS2", color_map = 'viridis_r')
sc.pl.umap(adata_NPC, color = "NR2F1", color_map = 'viridis_r')
sc.pl.umap(adata_NPC, color = "MEOX1", color_map = 'viridis_r')
sc.pl.umap(adata_NPC, color = "UNCX", color_map = 'viridis_r')
sc.pl.umap(adata_NPC, color = "PAX2", color_map = 'viridis_r')

sc.pl.umap(adata_NPC, color = "GADD45A", color_map = 'viridis_r')
sc.pl.umap(adata_NPC, color = "EZR", color_map = 'viridis_r')
sc.pl.umap(adata_NPC, color = "LHX1", color_map = 'viridis_r')
sc.pl.umap(adata_NPC, color = "DACH1", color_map = 'viridis_r')

sc.pl.umap(adata_NPC, color = "DAB2", color_map = 'viridis_r')
sc.pl.umap(adata_NPC, color = "TCF21", color_map = 'viridis_r')

sc.pl.umap(adata_NPC, color = "phase")


# In[11]:


sc.pl.umap(adata_NPC, color = "SLC34A1", color_map = 'viridis_r')


# In[12]:


umap = adata_NPC.obsm["X_umap"]
dataFrame = pd.DataFrame(umap, columns=list(range(umap.shape[1])))
dataFrame.index = adata_NPC.obs["cellType"].index
dataFrame["cellType"] = adata_NPC.obs["cellType"]

sc.pl.violin(adata_NPC, keys = ['UMOD'],  size = 1, groupby = 'sample', rotation= 45, layer = "counts", use_raw = False)

#dataFrame.to_csv('/home/levinsj/Fetal_dir/3D_UMAP_Nephron_clean.csv')  
adata_NPC.write_h5ad(filename = "/home/levinsj/Fetal_dir/CellBenderCorrected/04_annotated/cellSubsets/FetalNephrogenicOnly_annotated_final.h5ad")


# In[13]:


sc.tl.embedding_density(adata_NPC, basis='umap', groupby='sample')
sc.pl.embedding_density(adata_NPC, basis='umap', key='umap_density_sample')

