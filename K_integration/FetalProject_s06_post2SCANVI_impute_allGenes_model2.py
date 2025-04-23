#!/usr/bin/env python
# coding: utf-8

# In[1]:


from anndata import AnnData
import scanpy as sc
import squidpy as sq
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy

from sklearn.neighbors import NearestNeighbors

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 450


# Set parameters

# In[2]:


neighbors = 15

cell_cycle_genes = [x.strip() for x in open("/home/levinsj/Fetal_dir/Analysis/referenceFiles/cellCycleGenes.txt")]
s_genes = cell_cycle_genes[:43]
g2m_genes = cell_cycle_genes[43:]


# load model 2

# In[3]:


adata = sc.read_h5ad("/home/levinsj/spatial/adata/project_Files/Fetal/model2/fetal_PostSCANVI_Abs.h5ad")
adata.X = adata.layers["counts"]


# In[4]:


sc.pl.umap(adata, color = "tech", frameon = False)


# In[5]:


sc.pl.umap(adata, color = ["cellType","cellType_CosMx_1"], legend_loc='on data', legend_fontsize=10, legend_fontoutline=2, frameon = False)


# In[6]:


adata_imp = ad.AnnData(adata.raw.X)

adata_imp.obs_names = adata.raw.obs_names
adata_imp.var_names = adata.raw.var_names
adata_imp.obs = adata.obs
adata_imp.obsm = adata.obsm
adata_imp.obsp = adata.obsp
adata_imp.ubs = adata.uns

print(adata_imp)

adata_imp.layers["counts"] = adata_imp.X.copy()


# get cell IDs for each technology

# In[7]:


from sklearn.neighbors import NearestNeighbors

# Specify the number of neighbors (e.g., 15)
n_neighbors = 15

# Filter cells where ["tech"] is 'Cosmx'
Cosmx_cells_mask = (adata_imp.obs['tech'] == 'CosMx')
scRNA_cells_mask = (adata_imp.obs['tech'] != 'CosMx')

CosMx_index = np.where(Cosmx_cells_mask)[0]
scRNA_index = np.where(scRNA_cells_mask)[0]

nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
nn.fit(adata_imp.obsm["X_scANVI"][scRNA_cells_mask])

distances_all_to_non_Cosmx, indices_all_to_non_Cosmx = nn.kneighbors(adata_imp.obsm["X_scANVI"][Cosmx_cells_mask])

print(len(indices_all_to_non_Cosmx))

expression_df = adata_imp.to_df()
scRNA_df = expression_df.iloc[scRNA_index].T
CosMx_df = expression_df.iloc[CosMx_index]
CosMx_df.loc[:] = np.nan

adj = (distances_all_to_non_Cosmx ** -2.0).sum(axis=1)
affinity_array = (distances_all_to_non_Cosmx ** -2)

pd.options.mode.chained_assignment = None

for i in range(len(indices_all_to_non_Cosmx)):
      CosMx_df.iloc[i,:] = (scRNA_df.iloc[:,indices_all_to_non_Cosmx[i]] * affinity_array[i]).sum(axis=1)/adj[i]
        
CosMx_df[CosMx_df < 0.1] = 0 # remove lowest values to make matrix sparse and save space

adata_CosMx = adata_imp[adata_imp.obs["tech"] == "CosMx"]
adata_scRNA = adata_imp[adata_imp.obs["tech"] == "scRNA"]
adata_CosMx.layers["SCVI_imputed"] = scipy.sparse.csr_matrix(CosMx_df)
adata_CosMx.X = adata_CosMx.layers["SCVI_imputed"]

sc.pl.umap(adata_CosMx,color=['SIX2','CITED1','WT1',"JAG1"], wspace=0.1 , frameon = False, layer = "SCVI_imputed", cmap = "viridis_r")
sc.pl.umap(adata_CosMx,color=['SIX2','CITED1','WT1',"JAG1"], wspace=0.1 , frameon = False, layer = "counts", cmap = "viridis_r")

sc.pl.umap(adata_CosMx,color=['DAB2','IRX1','SIX1','UNCX'], wspace=0.1 , frameon = False, layer = "SCVI_imputed", cmap = "viridis_r")
sc.pl.umap(adata_CosMx,color=['DAB2','IRX1','SIX1','UNCX'], wspace=0.1 , frameon = False, layer = "counts", cmap = "viridis_r")


# In[8]:


sc.tl.score_genes_cell_cycle(adata_CosMx, s_genes=s_genes, g2m_genes=g2m_genes)
sc.pl.umap(adata_CosMx, color=['phase'], frameon = False)


# In[9]:


print(adata_CosMx)


# In[10]:


adata_CosMx.write_h5ad(filename = "/home/levinsj/spatial/adata/project_Files/Fetal/model2/fetal_PostSCANVI_imputedExpression.h5ad")


# In[11]:


sc.pl.umap(adata_CosMx, color='NPC_SCVI', cmap = "plasma", frameon = False)
sc.pl.umap(adata_CosMx, color='Podo_absorbtion_SCVI', cmap = "plasma", frameon = False)
sc.pl.umap(adata_CosMx, color='PseudoTime_SCVI', cmap = "plasma", frameon = False)
sc.pl.umap(adata_CosMx, color='PT_absorbtion_SCVI', cmap = "plasma", frameon = False)
sc.pl.umap(adata_CosMx, color='DCT_absorbtion_SCVI', cmap = "plasma", frameon = False)
sc.pl.umap(adata_CosMx, color='LOH_absorbtion_SCVI', cmap = "plasma", frameon = False)
sc.pl.umap(adata_CosMx, color='Glomerular_SCVI', cmap = "coolwarm", frameon = False)

